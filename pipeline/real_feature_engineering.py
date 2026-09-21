"""
Core NLP feature engineering for Yelp reviews.

Real, fully-computed feature values, plus built-in validation checks that
make it impossible to silently ship constant/fake columns.

Runs single-machine (pandas + spaCy + VADER + NRCLex) - no Spark/EMR needed
for this scale of data (tens of thousands to low hundreds of thousands of rows).

Usage:
    python real_feature_engineering.py                          # both default samples
    python real_feature_engineering.py --files my.csv           # one file
    python real_feature_engineering.py --sample 500             # quick test run
    python real_feature_engineering.py --vad-lexicon path.txt   # custom lexicon path
"""

import hashlib
import re
from io import BytesIO
import pandas as pd
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VAD_LEXICON_PATH = "../data/lexicons/NRC-VAD-Lexicon-v2_1.txt"

NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "cannot", "cant", "can't", "won't", "wont", "isn't", "isnt", "wasn't",
    "wasnt", "aren't", "arent", "weren't", "werent", "doesn't", "doesnt",
    "didn't", "didnt", "don't", "dont", "hasn't", "hasnt", "haven't",
    "havent", "hadn't", "hadnt", "shouldn't", "shouldnt", "wouldn't",
    "wouldnt", "couldn't", "couldnt", "n't",
}

# Also used for deterministic tie-breaking in dominant_emotion (documented there).
EMOTION_ORDER = ["anger", "fear", "joy", "sadness", "anticipation", "disgust", "surprise", "trust"]

WORD_RE = re.compile(r"[a-zA-Z']+")

# Feature columns this pipeline OWNS. Any of these present in an input file are
# dropped and fully regenerated from raw text - never trusted from old data.
# wcst_count / worry_core_count / anxiety_score_avg / yelp_sentiment_avg have
# no recoverable definition under these legacy names: dropped, not regenerated
# (see iteration2_features.py for the real, differently-named replacements).
GENERATED_COLUMNS = [
    "stripped_review", "word_count", "char_count", "avg_word_len", "avg_sentence_len",
    "num_excl", "num_ques", "num_caps", "num_at", "num_hash", "sentence_count",
    "anger_int_avg", "fear_int_avg", "joy_int_avg", "sadness_int_avg",
    "anticipation_int_avg", "disgust_int_avg", "surprise_int_avg", "trust_int_avg",
    "Valence_avg", "Arousal_avg", "Dominance_avg",
    "wcst_count", "worry_core_count", "anxiety_score_avg", "yelp_sentiment_avg",
    "vader_sentiment_score", "vader_pos", "vader_neu", "vader_neg",
    "anger_count", "fear_count", "joy_count", "sadness_count", "anticipation_count",
    "disgust_count", "surprise_count", "trust_count", "positive_count", "negative_count",
    "negation_count", "person_count", "location_count", "product_count",
    "dominant_emotion", "vad_matched_words",
    "hf_sentiment_label", "hf_sentiment_confidence", "hf_computed_sentiment",
    "hf_emotion_label", "hf_emotion_confidence",
]

# Columns added only when the optional --transformers path is enabled.
TRANSFORMER_COLUMNS = [
    "hf_sentiment_label", "hf_sentiment_confidence", "hf_computed_sentiment",
    "hf_emotion_label", "hf_emotion_confidence",
]

# Module-level model handles - populated by init_models(), not at import time,
# so `from real_feature_engineering import ...` in a notebook has no side effects.
nlp = None
vader = None
VAD_LOOKUP = None
_nrc = None
_hf_sentiment = None
_hf_emotion = None
_loaded_vad_path = None


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_vad_lexicon(vad_lexicon_path: str | bytes) -> dict:
    """Load single-word NRC-VAD terms without treating words as missing data."""
    # `null` is a valid English term in NRC-VAD. Pandas otherwise interprets
    # it as an NA marker, which both drops the score and breaks string filters.
    source = (
        BytesIO(bytes(vad_lexicon_path))
        if isinstance(vad_lexicon_path, (bytes, bytearray))
        else vad_lexicon_path
    )
    vad_df = pd.read_csv(
        source,
        sep="\t",
        keep_default_na=False,
        na_values=[],
    )
    vad_df = vad_df[~vad_df["term"].str.contains(" ")]
    return vad_df.set_index("term")[["valence", "arousal", "dominance"]].to_dict("index")


def init_models(
    vad_lexicon_path: str = DEFAULT_VAD_LEXICON_PATH,
    vad_lexicon_payload: bytes | None = None,
) -> None:
    """Load spaCy, VADER, NRC emotion lexicon, and the NRC-VAD lexicon.

    Must be called once before process_dataframe(). Kept out of module scope so
    importing this file (e.g. from the analysis notebook) is cheap and safe.
    """
    global nlp, vader, VAD_LOOKUP, _nrc, _loaded_vad_path
    resolved_vad_path = (
        "payload:" + hashlib.sha256(vad_lexicon_payload).hexdigest()
        if vad_lexicon_payload is not None
        else str(vad_lexicon_path)
    )
    # Spark commonly reuses a Python worker for several partitions. Avoid
    # reloading spaCy and reparsing 44K VAD terms every time that happens.
    if nlp is not None and _loaded_vad_path == resolved_vad_path:
        return
    import spacy

    print("Loading models...")
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "attribute_ruler", "parser"])
    nlp.add_pipe("sentencizer")  # lightweight sentence boundaries without full parser
    vader = SentimentIntensityAnalyzer()
    _nrc = NRCLex()  # one shared instance; per-review state is set via load_token_list()

    # NRC-VAD Lexicon v2.1 (Mohammad, 2018) - real valence/arousal/dominance in [-1, 1].
    # Single-word entries only (44,728 of 54,801 rows); multi-word phrase entries are
    # skipped for now (marginal gain, added lookup complexity at this scale).
    VAD_LOOKUP = load_vad_lexicon(
        vad_lexicon_payload if vad_lexicon_payload is not None else vad_lexicon_path
    )
    _loaded_vad_path = resolved_vad_path
    source_label = "serverless payload" if vad_lexicon_payload is not None else vad_lexicon_path
    print(f"Loaded NRC-VAD lexicon: {len(VAD_LOOKUP):,} single-word terms from {source_label}")


def init_transformers() -> None:
    """OPTIONAL: load HuggingFace transformer pipelines (iteration 1's approach).

    Transformer models are contextual and noticeably more accurate than the
    lexicon methods, especially for emotion (a lexicon can't tell "sad to see
    them close" in a 5-star review from genuine sadness). The tradeoff is
    compute: ~25 min for 15k reviews on CPU in iteration 1; a GPU makes this
    fast. Enable via --transformers if your hardware is adequate.

    Requires:  pip install transformers torch
    Models:    distilbert-base-uncased-finetuned-sst-2-english (sentiment)
               j-hartmann/emotion-english-distilroberta-base   (7-way emotion)
    """
    global _hf_sentiment, _hf_emotion
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise SystemExit(
            "--transformers requires the 'transformers' and 'torch' packages.\n"
            "Install with:  pip install transformers torch\n"
            "(Or omit --transformers to use the default lexicon-based features.)"
        )
    print("Loading HuggingFace pipelines (downloads models on first run)...")
    _hf_sentiment = hf_pipeline("sentiment-analysis", truncation=True)
    _hf_emotion = hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        truncation=True,
    )
    print("Transformer pipelines ready.")


def transformer_features(text: str) -> dict:
    """HuggingFace sentiment + emotion for one review (mirrors iteration 1).

    hf_computed_sentiment reuses iteration 1's formula:
    (2 * label - 1) * confidence, giving a signed score in [-1, 1].
    """
    if not text:
        return {c: None for c in TRANSFORMER_COLUMNS}
    s = _hf_sentiment(text)[0]
    label = 1 if s["label"].lower() == "positive" else 0
    e = _hf_emotion(text)[0]
    return {
        "hf_sentiment_label": label,
        "hf_sentiment_confidence": round(s["score"], 4),
        "hf_computed_sentiment": round((2 * label - 1) * s["score"], 4),
        "hf_emotion_label": e["label"],
        "hf_emotion_confidence": round(e["score"], 4),
    }


def _require_models():
    if nlp is None:
        raise RuntimeError("Models not loaded - call init_models() first.")


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def simple_tokenize(text: str):
    return WORD_RE.findall(text.lower())


def clean_text(raw_text: str) -> str:
    """Cleans the full review text - no truncation.

    Order matters: strip URLs first, THEN collapse whitespace, so removed URLs
    don't leave double spaces behind.
    """
    if not isinstance(raw_text, str):
        return ""
    t = re.sub(r"http\S+|www\.\S+", " ", raw_text)  # strip URLs
    t = re.sub(r"\s+", " ", t)                       # collapse whitespace/newlines
    return t.strip()


def linguistic_features(text: str, doc=None) -> dict:
    words = text.split()
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0

    if doc is None:
        doc = nlp(text)
    sentence_count = max(len(list(doc.sents)), 1)

    return {
        "word_count": word_count,
        "char_count": len(text),
        "avg_word_len": round(float(avg_word_len), 3),
        "avg_sentence_len": round(word_count / sentence_count, 3),
        "num_excl": text.count("!"),
        "num_ques": text.count("?"),
        "num_caps": sum(1 for w in words if w.isupper() and len(w) > 1),
        "num_at": text.count("@"),
        "num_hash": text.count("#"),
        "sentence_count": sentence_count,
        "negation_count": sum(1 for w in simple_tokenize(text) if w in NEGATION_WORDS),
    }, doc


def sentiment_features(text: str) -> dict:
    scores = vader.polarity_scores(text)
    return {
        "vader_sentiment_score": round(scores["compound"], 4),
        "vader_pos": round(scores["pos"], 4),
        "vader_neu": round(scores["neu"], 4),
        "vader_neg": round(scores["neg"], 4),
    }


def emotion_features(text: str) -> dict:
    """NRC emotion lexicon word counts and proportions.

    NOTE on naming: *_int_avg are PROPORTIONS (this emotion's share of all
    emotion-bearing words in the review), not word-level intensity averages -
    named for schema consistency with the `_int_avg` intensity columns
    elsewhere in this schema, not because they are one.
    """
    _nrc.load_token_list(simple_tokenize(text))
    raw = _nrc.raw_emotion_scores      # counts per emotion actually found in lexicon
    freq = _nrc.affect_frequencies     # normalized proportions

    out = {}
    for emo in EMOTION_ORDER:
        out[f"{emo}_count"] = raw.get(emo, 0)
        out[f"{emo}_int_avg"] = round(freq.get(emo, 0.0), 4)
    out["positive_count"] = raw.get("positive", 0)
    out["negative_count"] = raw.get("negative", 0)
    return out


def vad_features(text: str) -> dict:
    """Average Valence/Arousal/Dominance across matched words (real NRC-VAD lookup).

    Reviews with zero lexicon matches get 0.0 (scale midpoint = neutral);
    vad_matched_words lets downstream users filter or down-weight those rows.
    """
    matched = [VAD_LOOKUP[w] for w in simple_tokenize(text) if w in VAD_LOOKUP]
    if not matched:
        return {"Valence_avg": 0.0, "Arousal_avg": 0.0, "Dominance_avg": 0.0, "vad_matched_words": 0}
    return {
        "Valence_avg": round(float(np.mean([m["valence"] for m in matched])), 4),
        "Arousal_avg": round(float(np.mean([m["arousal"] for m in matched])), 4),
        "Dominance_avg": round(float(np.mean([m["dominance"] for m in matched])), 4),
        "vad_matched_words": len(matched),
    }


def entity_features(doc) -> dict:
    person = sum(1 for e in doc.ents if e.label_ == "PERSON")
    location = sum(1 for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC"))
    product = sum(1 for e in doc.ents if e.label_ in ("PRODUCT", "ORG"))
    return {"person_count": person, "location_count": location, "product_count": product}


def dominant_emotion(row) -> str:
    """Highest-proportion emotion; 'neutral' if no emotion words matched.

    Ties are broken deterministically by EMOTION_ORDER position (first listed
    wins). Ties are rare (<1% of reviews) but the rule is explicit, not accidental.
    """
    scores = {emo: row[f"{emo}_int_avg"] for emo in EMOTION_ORDER}
    if max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_dataframe(df: pd.DataFrame, text_col: str = "raw_review",
                      use_transformers: bool = False) -> pd.DataFrame:
    _require_models()
    if use_transformers and _hf_sentiment is None:
        raise RuntimeError("use_transformers=True but init_transformers() was not called.")
    print(f"Processing {len(df):,} rows...")
    records = []
    texts = [clean_text(raw) for raw in df[text_col].fillna("")]
    for i, (text, doc) in enumerate(zip(texts, nlp.pipe(texts, batch_size=128))):
        if i % 2000 == 0:
            print(f"  {i:,}/{len(df):,}")
        ling, doc = linguistic_features(text, doc=doc)
        sent = sentiment_features(text)
        emo = emotion_features(text)
        vad = vad_features(text)
        ent = entity_features(doc)
        # stripped_review keeps the FULL cleaned text - no silent truncation.
        row = {"stripped_review": text, **ling, **sent, **emo, **vad, **ent}
        row["dominant_emotion"] = dominant_emotion(row)
        if use_transformers:
            row.update(transformer_features(text))
        records.append(row)
    feat_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Validation - catches placeholder/constant/mistracked columns before they ship
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, text_col: str = "raw_review") -> dict:
    print("\n=== VALIDATION ===")
    failures = []
    metrics = {}

    real_word_count = df[text_col].fillna("").str.split().str.len()
    corr = df["word_count"].corr(real_word_count)
    metrics["word_count_corr"] = round(float(corr), 3)
    print(f"word_count vs actual text length correlation: {corr:.3f}")
    if corr < 0.9:
        failures.append(f"word_count barely tracks real text length (r={corr:.3f})")

    # num_at/num_hash are legitimately near-zero in restaurant/salon reviews (rare real
    # symbols), so a small sample can look "constant" without being a bug. Exclude
    # known-sparse-by-nature columns from this check.
    expected_sparse = {"num_at", "num_hash"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    const_cols = [c for c in numeric_cols if df[c].nunique(dropna=False) == 1 and c not in expected_sparse]
    metrics["constant_columns"] = const_cols
    print(f"Constant columns: {const_cols if const_cols else 'none'}")
    if const_cols:
        failures.append(f"Found constant columns (a real feature should vary across reviews): {const_cols}")

    if "stars" in df.columns:
        sent_corr = df["vader_sentiment_score"].corr(df["stars"])
        metrics["vader_vs_stars_corr"] = round(float(sent_corr), 3)
        print(f"vader_sentiment_score vs stars correlation: {sent_corr:.3f}")
        if sent_corr < 0.3:
            failures.append(f"Sentiment doesn't track star rating (r={sent_corr:.3f}) - check pipeline")

        if "Valence_avg" in df.columns:
            val_corr = df["Valence_avg"].corr(df["stars"])
            metrics["valence_vs_stars_corr"] = round(float(val_corr), 3)
            print(f"Valence_avg vs stars correlation: {val_corr:.3f}")
            if val_corr < 0.2:
                failures.append(f"Valence_avg doesn't track star rating (r={val_corr:.3f}) - check VAD lookup")

    metrics["passed"] = len(failures) == 0
    metrics["failures"] = failures

    if failures:
        print("\n FAILED VALIDATION:")
        for f in failures:
            print("  -", f)
        raise AssertionError("Feature engineering output failed validation - see above. Do not ship this data.")
    print("\nAll checks passed.")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Iteration 3 real feature engineering pipeline.")
    parser.add_argument("--files", nargs="+", default=["chipotle_sample_15k.csv", "hair_sample_15k.csv"],
                        help="Input CSV(s) to process. Default: both industry samples.")
    parser.add_argument("--text-col", default="raw_review", help="Column containing the raw review text.")
    parser.add_argument("--vad-lexicon", default=DEFAULT_VAD_LEXICON_PATH, help="Path to NRC-VAD lexicon TSV.")
    parser.add_argument("--outdir", default=".", help="Where to write *_REAL.csv outputs + run log.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional row cap for quick testing (must be >= 1).")
    parser.add_argument("--transformers", action="store_true",
                        help="ALSO run HuggingFace transformer sentiment/emotion (iteration 1's "
                             "models). More accurate, much slower on CPU - GPU recommended. "
                             "Requires: pip install transformers torch")
    args = parser.parse_args()

    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be >= 1")

    init_models(vad_lexicon_path=args.vad_lexicon)  # CLI arg actually used now
    if args.transformers:
        init_transformers()

    run_entries = []
    for infile in args.files:
        name = Path(infile).stem
        outfile = str(Path(args.outdir) / f"{name}_REAL.csv")

        print(f"\n{'='*60}\nPROCESSING: {infile}\n{'='*60}")
        df = pd.read_csv(infile)
        if args.sample is not None:
            df = df.head(args.sample)

        # Regenerate every pipeline-owned column from raw text; never trust old values.
        df = df[[c for c in df.columns if c not in GENERATED_COLUMNS]]

        result = process_dataframe(df, text_col=args.text_col,
                                   use_transformers=args.transformers)
        metrics = validate(result, text_col=args.text_col)
        result.to_csv(outfile, index=False)
        print(f"Saved: {outfile}")
        run_entries.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input": infile,
            "rows": len(result),
            "output": outfile,
            "vad_lexicon": args.vad_lexicon,
            "transformers": args.transformers,
            "validation": metrics,
        })

    # Append-only run history: past entries are preserved, never overwritten.
    log_path = Path(args.outdir) / "pipeline_run_log.json"
    history = json.loads(log_path.read_text()) if log_path.exists() else []
    if isinstance(history, dict):  # migrate old dict-format log
        history = list(history.values())
    history.extend(run_entries)
    log_path.write_text(json.dumps(history, indent=2))
    print(f"\nRun log appended: {log_path} ({len(history)} total entries)")
