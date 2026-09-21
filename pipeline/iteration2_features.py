"""
Additional analytical features: grammar/POS composition, a time-decayed
"weighted star" rating, three specialized lexicons (WorryWords, NRC-WCST,
Yelp AFFLEX-NEGLEX), and a lexicon-only emotion-detection system to compare
against NRCLex/HuggingFace.

Spark-portable by design: lexicons are loaded once per worker from a
local/DBFS path, same pattern as real_feature_engineering.py's VAD lexicon -
no S3/boto3 dependency baked into the feature logic itself.
"""

from datetime import datetime, timezone
import re

import numpy as np
from textblob import TextBlob

EMOTION_ORDER = ["anger", "fear", "joy", "sadness", "anticipation", "disgust", "surprise", "trust"]
WORD_RE = re.compile(r"[a-zA-Z']+")

# Every column this module owns. Same "never trust old values, always
# regenerate" rule as real_feature_engineering.py's GENERATED_COLUMNS.
ITERATION2_COLUMNS = [
    "noun_pct", "verb_pct", "adj_pct", "adv_pct", "subjectivity_score", "type_token_ratio",
    "worry_word_count", "wcst_warmth_avg", "wcst_competence_avg", "wcst_sociability_avg", "wcst_trust_avg",
    "yelp_sentiment_avg", "yelp_matched_words",
    "nrc_anger_lex", "nrc_fear_lex", "nrc_joy_lex", "nrc_sadness_lex",
    "nrc_anticipation_lex", "nrc_disgust_lex", "nrc_surprise_lex", "nrc_trust_lex",
    "primary_emotion_lex", "primary_emotion_lex_conf", "secondary_emotion_lex", "secondary_emotion_lex_conf",
    "weighted_star",
]


def grammar_features(text: str, doc=None) -> dict:
    """POS composition, subjectivity, and type-token ratio.

    Spark workers reuse the spaCy document already created for every review;
    this avoids a second tagger pass and removes the need for separately
    downloaded NLTK corpora on serverless compute. TextBlob remains the
    subjectivity analyzer.
    """
    blob = TextBlob(text)
    tags = (
        [token.tag_ for token in doc if not token.is_space]
        if doc is not None
        else [tag for _, tag in blob.tags]
    )
    total = len(tags) if tags else 1
    counts = {"NN": 0, "VB": 0, "JJ": 0, "RB": 0}
    for tag in tags:
        key = tag[:2]
        if key in counts:
            counts[key] += 1
    words = text.split()
    word_total = len(words) if words else 1
    return {
        "noun_pct": round(counts["NN"] / total, 4),
        "verb_pct": round(counts["VB"] / total, 4),
        "adj_pct": round(counts["JJ"] / total, 4),
        "adv_pct": round(counts["RB"] / total, 4),
        "subjectivity_score": round(float(blob.sentiment.subjectivity), 4),
        "type_token_ratio": round(len(set(words)) / word_total, 4),
    }


def specialized_lexicon_features(text: str, lexicons: dict) -> dict:
    """WorryWords, WCST (real 4-dim averages, not just a match count), Yelp AFFLEX-NEGLEX,
    and NRC Emotion Intensity - see iteration2_lexicons.py for the loaders."""
    # Match punctuation-adjacent words ("great!", "worried,") instead of
    # silently losing them through whitespace-only tokenization.
    words = WORD_RE.findall(text.lower())
    worry_count = 0
    wcst_sums = {"warmth": 0.0, "competence": 0.0, "sociability": 0.0, "trust": 0.0}
    wcst_n = 0
    yelp_sum = 0.0
    yelp_n = 0
    emo_sums = {e: 0.0 for e in EMOTION_ORDER}

    worry_lex = lexicons["worry"]
    wcst_lex = lexicons["wcst"]
    yelp_lex = lexicons["yelp"]
    nrc_int_lex = lexicons["nrc_intensity"]

    for w in words:
        if w in worry_lex:
            worry_count += 1
        if w in wcst_lex:
            d = wcst_lex[w]
            for k in wcst_sums:
                wcst_sums[k] += d[k]
            wcst_n += 1
        if w in yelp_lex:
            yelp_sum += yelp_lex[w]
            yelp_n += 1
        if w in nrc_int_lex:
            for emo, score in nrc_int_lex[w].items():
                if emo in emo_sums:
                    emo_sums[emo] += score

    total_words = len(words) if words else 1
    out = {
        "worry_word_count": worry_count,
        "wcst_warmth_avg": round(wcst_sums["warmth"] / wcst_n, 4) if wcst_n else 0.0,
        "wcst_competence_avg": round(wcst_sums["competence"] / wcst_n, 4) if wcst_n else 0.0,
        "wcst_sociability_avg": round(wcst_sums["sociability"] / wcst_n, 4) if wcst_n else 0.0,
        "wcst_trust_avg": round(wcst_sums["trust"] / wcst_n, 4) if wcst_n else 0.0,
        "yelp_sentiment_avg": round(yelp_sum / yelp_n, 4) if yelp_n else 0.0,
        "yelp_matched_words": yelp_n,
    }
    for e in EMOTION_ORDER:
        out[f"nrc_{e}_lex"] = round(emo_sums[e] / total_words, 4)
    return out


def lexicon_emotion_labels(nrc_lex_scores: dict) -> dict:
    """Primary/secondary emotion + confidence purely from NRC Emotion Intensity
    scores - a second, lexicon-only emotion system to compare against NRCLex
    proportions and (optionally) HuggingFace's ML labels. Ties broken
    deterministically by EMOTION_ORDER, same convention as
    real_feature_engineering.py's dominant_emotion()."""
    scores = {e: nrc_lex_scores.get(f"nrc_{e}_lex", 0.0) for e in EMOTION_ORDER}
    total = sum(scores.values())
    if total <= 0:
        return {
            "primary_emotion_lex": "neutral", "primary_emotion_lex_conf": 1.0,
            "secondary_emotion_lex": "neutral", "secondary_emotion_lex_conf": 0.0,
        }
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], EMOTION_ORDER.index(kv[0])))
    (p_emo, p_score), (s_emo, s_score) = ranked[0], ranked[1]
    return {
        "primary_emotion_lex": p_emo, "primary_emotion_lex_conf": round(p_score / total, 4),
        "secondary_emotion_lex": s_emo, "secondary_emotion_lex_conf": round(s_score / total, 4),
    }


def weighted_star(stars: float, review_date, reference_date) -> float:
    """Time-decayed star rating: 0.998^days_old * stars (~50% weight after ~1 year).

    Decay is anchored to `reference_date`, not wall-clock "now" - the pipeline
    caller passes the dataset's own max review date once per run (see
    corrected_feature_engineering.py), so decay is relative to "most recent
    review in this dataset", which is what "weighted by recency" actually
    means for a fixed historical dataset. Anchoring to actual wall-clock time
    instead would decay every row toward ~0 for a dataset evaluated years
    after its reviews were written - not a useful feature. This also makes
    the feature reproducible across runs, unlike a live datetime.now() call."""
    try:
        review_ts = _parse_date(review_date)
        if review_ts is None:
            return float(stars)
        days_diff = max(0, (reference_date - review_ts).days)
        decay = 0.998 ** days_diff
        return round(float(stars) * decay, 4)
    except Exception:
        return float(stars)


def _parse_date(value):
    import pandas as pd
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()
