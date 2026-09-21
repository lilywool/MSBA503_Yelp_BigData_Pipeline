"""
The full feature set for this pipeline, distributable via Spark mapInPandas.

= What this is =
Every feature column is a real, fully-computed value - no hardcoded
placeholders, no columns that silently fall back to a default. Lexicons are
parsed against their actual file formats. `validate()` is a mandatory gate on
every run, and every run is logged, pass or fail.

= Column provenance =
  * Core linguistic/sentiment/emotion/VAD/NER columns: real_feature_engineering.py,
    imported UNCHANGED - proven correct at 15K-400K row scale. This also
    supplies negation_count, person_count, location_count, product_count, and
    vader_sentiment_score.
  * Grammar/POS, weighted_star, WorryWords/WCST/Yelp/NRC-intensity lexicons,
    lexicon-based primary/secondary emotion: iteration2_features.py.
  * Optional HuggingFace transformer sentiment/emotion: real_feature_engineering.py's
    --transformers path - a second, ML-based emotion system alongside the
    lexicon-based one, both genuinely computed.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import real_feature_engineering as rfe
import iteration2_features as it2
from iteration2_lexicons import (
    load_all as load_iteration2_lexicons,
    load_all_from_payloads,
)

# rfe.GENERATED_COLUMNS is real_feature_engineering's "strip these from input,
# never trust old values" list - it deliberately INCLUDES 4 legacy column names
# (wcst_count, worry_core_count, anxiety_score_avg, yelp_sentiment_avg) that
# it never regenerates itself ("no recoverable definition"). This repo DOES
# recover real definitions for wcst/worry/yelp (under new names - see
# iteration2_features.ITERATION2_COLUMNS), so the legacy raw names are only
# ever used for scrubbing stale input columns, never for schema/output.
_LEGACY_NO_DEFINITION = {"wcst_count", "worry_core_count", "anxiety_score_avg", "yelp_sentiment_avg"}

# Columns this module actually PRODUCES per row - use this for output schemas
# and column selection (e.g. in Spark's mapInPandas).
OUTPUT_COLUMNS = [c for c in rfe.GENERATED_COLUMNS if c not in _LEGACY_NO_DEFINITION] + it2.ITERATION2_COLUMNS

FEATURE_GROUP_COLUMNS = {
    "linguistic": [
        "stripped_review", "word_count", "char_count", "avg_word_len",
        "avg_sentence_len", "num_excl", "num_ques", "num_caps", "num_at",
        "num_hash", "sentence_count", "negation_count",
    ],
    "vader": ["vader_sentiment_score", "vader_pos", "vader_neu", "vader_neg"],
    "nrc-emotion": [
        *[f"{emotion}_count" for emotion in rfe.EMOTION_ORDER],
        *[f"{emotion}_int_avg" for emotion in rfe.EMOTION_ORDER],
        "positive_count", "negative_count", "dominant_emotion",
    ],
    "vad": ["Valence_avg", "Arousal_avg", "Dominance_avg", "vad_matched_words"],
    "entities": ["person_count", "location_count", "product_count"],
    "grammar": [
        "noun_pct", "verb_pct", "adj_pct", "adv_pct", "subjectivity_score",
        "type_token_ratio",
    ],
    "domain-lexicons": [
        "worry_word_count", "wcst_warmth_avg", "wcst_competence_avg",
        "wcst_sociability_avg", "wcst_trust_avg", "yelp_sentiment_avg",
        "yelp_matched_words",
        *[f"nrc_{emotion}_lex" for emotion in it2.EMOTION_ORDER],
        "primary_emotion_lex", "primary_emotion_lex_conf",
        "secondary_emotion_lex", "secondary_emotion_lex_conf",
    ],
    "time-weighting": ["weighted_star"],
    "transformers": list(rfe.TRANSFORMER_COLUMNS),
}
FEATURE_GROUPS = tuple(FEATURE_GROUP_COLUMNS)
DEFAULT_FEATURE_GROUPS = tuple(group for group in FEATURE_GROUPS if group != "transformers")

# Columns to strip from any INPUT data before regenerating - deliberately broader
# than OUTPUT_COLUMNS so stale copies of the legacy column names get scrubbed too.
ALL_GENERATED_COLUMNS = list(dict.fromkeys(rfe.GENERATED_COLUMNS + it2.ITERATION2_COLUMNS))

_lexicons = None  # the four specialized lexicons, loaded once per worker via init_models()
_lexicon_payload_signature = None


def normalize_feature_groups(feature_groups=None, use_transformers: bool = False) -> tuple[str, ...]:
    """Resolve an ordered, validated set of independently selectable feature groups."""
    requested = set(DEFAULT_FEATURE_GROUPS if feature_groups is None else feature_groups)
    if use_transformers:
        requested.add("transformers")
    unknown = requested - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one feature group must be selected")
    return tuple(group for group in FEATURE_GROUPS if group in requested)


def output_columns_for(feature_groups=None, use_transformers: bool = False) -> list[str]:
    groups = normalize_feature_groups(feature_groups, use_transformers)
    selected = {
        column for group in groups for column in FEATURE_GROUP_COLUMNS[group]
    }
    return [column for column in OUTPUT_COLUMNS if column in selected]


def init_models(vad_lexicon_path: str = None, worry_path: str = None, wcst_path: str = None,
                 yelp_path: str = None, nrc_intensity_path: str = None,
                 preloaded_lexicons: dict = None,
                 lexicon_payloads: dict[str, bytes] = None) -> None:
    """Loads BOTH the core models (spaCy/VADER/NRCLex/VAD) and the four
    specialized lexicons. Call once per worker/process, exactly like
    real_feature_engineering.init_models().

    `preloaded_lexicons`: pass the four specialized lexicons already parsed
    (e.g. a Spark broadcast variable's `.value`, loaded once on the driver)
    instead of re-parsing them from `worry_path`/`wcst_path`/`yelp_path`/
    `nrc_intensity_path` on every call. spaCy/VADER/NRCLex/VAD (via
    rfe.init_models) still load per-call regardless - those are stateful
    model objects, not broadcastable plain data, so every worker loads its
    own. See spark_feature_engineering.py's `run()` for the broadcast path.
    """
    global _lexicons, _lexicon_payload_signature
    if lexicon_payloads is not None:
        signature = tuple(
            (key, hashlib.sha256(value).hexdigest())
            for key, value in sorted(lexicon_payloads.items())
        )
        rfe.init_models(
            vad_lexicon_path=vad_lexicon_path,
            vad_lexicon_payload=lexicon_payloads["vad_lexicon_path"],
        )
        if _lexicons is None or _lexicon_payload_signature != signature:
            _lexicons = load_all_from_payloads(lexicon_payloads)
            _lexicon_payload_signature = signature
    else:
        rfe.init_models(vad_lexicon_path=vad_lexicon_path)
    if preloaded_lexicons is not None:
        _lexicons = preloaded_lexicons
        _lexicon_payload_signature = None
    elif lexicon_payloads is None:
        _lexicons = load_iteration2_lexicons(worry_path, wcst_path, yelp_path, nrc_intensity_path)


def init_transformers() -> None:
    rfe.init_transformers()


def process_dataframe(df: pd.DataFrame, text_col: str = "raw_review",
                       stars_col: str = "stars", date_col: str = "review_date",
                       use_transformers: bool = False, reference_date=None,
                       feature_groups=None) -> pd.DataFrame:
    """Core + exclusive columns, in one pass over the text (each review is only
    cleaned/tokenized once, reused across both column families).

    `reference_date`: the "now" that weighted_star decays against (see
    iteration2_features.weighted_star). IMPORTANT for distributed correctness:
    this MUST be computed once globally (e.g. max review_date across the whole
    dataset) and passed in explicitly when calling this per-partition inside
    Spark's mapInPandas - each partition only sees a slice of the data, so
    letting every partition compute its own local max would silently give
    different reviews different decay anchors depending on which partition
    they landed in. spark_feature_engineering.py computes this once at the
    driver via a Spark aggregate and passes it into every partition's call.
    Only left as None (auto-computed from `df`) for the single-machine CLI
    path, where there's exactly one "partition" by definition.
    """
    rfe._require_models()
    if _lexicons is None:
        raise RuntimeError("specialized lexicons not loaded - call init_models() first.")

    if reference_date is None and stars_col in df.columns and date_col in df.columns:
        reference_date = it2._parse_date(pd.to_datetime(df[date_col], errors="coerce").max())

    groups = set(normalize_feature_groups(feature_groups, use_transformers))
    records = []
    texts = [rfe.clean_text(raw) for raw in df[text_col].fillna("")]
    for i, (text, doc) in enumerate(zip(texts, rfe.nlp.pipe(texts, batch_size=128))):
        if i % 2000 == 0:
            print(f"  {i:,}/{len(df):,}")

        row = {}
        if "linguistic" in groups:
            ling, doc = rfe.linguistic_features(text, doc=doc)
            row.update({"stripped_review": text, **ling})
        if "vader" in groups:
            row.update(rfe.sentiment_features(text))
        if "nrc-emotion" in groups:
            row.update(rfe.emotion_features(text))
            row["dominant_emotion"] = rfe.dominant_emotion(row)
        if "vad" in groups:
            row.update(rfe.vad_features(text))
        if "entities" in groups:
            row.update(rfe.entity_features(doc))
        if "transformers" in groups:
            row.update(rfe.transformer_features(text))

        if "grammar" in groups:
            row.update(it2.grammar_features(text, doc=doc))
        if "domain-lexicons" in groups:
            specialized = it2.specialized_lexicon_features(text, _lexicons)
            row.update(specialized)
            row.update(it2.lexicon_emotion_labels(specialized))

        if "time-weighting" in groups:
            if reference_date is not None:
                row["weighted_star"] = it2.weighted_star(df[stars_col].iloc[i], df[date_col].iloc[i], reference_date)
            else:
                row["weighted_star"] = df[stars_col].iloc[i] if stars_col in df.columns else None

        records.append(row)

    feat_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def validate(df: pd.DataFrame, text_col: str = "raw_review") -> dict:
    """The base validate() checks, PLUS checks specific to a hardcoded-default
    or silently-failed-lexicon bug: a column stuck at a single value is
    exactly what either of those failure modes produces."""
    metrics = rfe.validate(df, text_col=text_col)  # raises AssertionError on failure, same contract

    failures = list(metrics.get("failures", []))
    extra_checks = {}

    # yelp_sentiment_avg / worry_word_count stuck at 0 for every row is exactly
    # the "except: print(error), keep going with empty lexicon" failure shape.
    for col in ("yelp_sentiment_avg", "worry_word_count", "wcst_warmth_avg"):
        if col in df.columns:
            nonzero_share = float((df[col] != 0).mean())
            extra_checks[f"{col}_nonzero_share"] = round(nonzero_share, 4)
            if nonzero_share < 0.01:
                failures.append(
                    f"{col} is ~0 for {100*(1-nonzero_share):.1f}% of rows - lexicon likely failed to load."
                )

    if "stars" in df.columns and "weighted_star" in df.columns:
        # weighted_star should differ from stars for at least some older reviews;
        # if it's identical everywhere, the decay was never applied (another
        # plausible hardcode-to-default failure mode).
        identical_share = float((df["weighted_star"] == df["stars"]).mean())
        extra_checks["weighted_star_identical_to_stars_share"] = round(identical_share, 4)
        if identical_share > 0.999:
            failures.append("weighted_star is identical to stars for every row - time decay was never applied.")

    metrics["extra_checks"] = extra_checks
    metrics["failures"] = failures
    metrics["passed"] = len(failures) == 0
    if failures and not metrics.get("_raised"):
        print("\nFAILED VALIDATION (additional checks):")
        for f in failures:
            print("  -", f)
        raise AssertionError("Output failed validation - see above. Do not ship this data.")
    return metrics
