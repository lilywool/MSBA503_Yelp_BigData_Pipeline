"""
prepare_dashboard_data.py (archived coursework utility)

Supplementary, standalone script preserved outside the live pipeline.
Not part of the distributed feature-engineering pipeline and does not change
anything in the archived coursework notebook. Its only job is producing the three
input files that notebook already expects, in the format and naming it
already expects:

    chipotle_sample_15k.parquet
    hair_sample_15k.parquet
    business_aggregated_sample.csv

Inputs: this repo's own gold-layer output (one CSV per dataset - the result
of pipeline/corrected_feature_engineering.py or
pipeline/spark_feature_engineering.py, already validated). This script does
not compute any NLP features itself - it only converts format, aggregates
to business level, and draws the sample.

Sampling methodology matches how these samples were originally produced:
15,000-row stratified sample by star rating, random_state=222, per dataset.

Usage:
    python prepare_dashboard_data.py \
        --chipotle-full ../data/processed/chipotle_full_REAL_distributed.csv \
        --hair-full ../data/processed/hair_full_REAL_distributed.csv \
        --outdir .
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 222
SAMPLE_SIZE = 15_000

# Maps this repo's real column names to the names the dashboard's sample
# schema originally used. Only renamed where the two names genuinely refer
# to the same computed value - nothing here is invented.
COLUMN_RENAMES = {
    "Valence_avg": "valence_avg",
    "Arousal_avg": "arousal_avg",
    "Dominance_avg": "dominance_avg",
}

# Historical sample columns that were HuggingFace-transformer outputs
# (sentiment_score, primary_emotion, secondary_emotion, and their
# confidences). This repo's default (non --transformers) run doesn't
# produce a top-2 HuggingFace emotion pair, only the single-best
# dominant_emotion (lexicon-based) and, optionally, hf_emotion_label /
# hf_computed_sentiment (top-1 only) when --transformers was used.
# Rather than mislabel a lexicon value as a HuggingFace one, these are
# filled from the closest real equivalent this repo actually computes,
# with the mapping made explicit below - never silently faked.
FALLBACK_FROM_LEXICON = {
    "sentiment_score": "vader_sentiment_score",
    "primary_emotion": "dominant_emotion",
}
# No lexicon equivalent exists for these - left absent from the sample
# rather than filled with a placeholder value.
NO_EQUIVALENT = [
    "sentiment_confidence", "primary_emotion_confidence",
    "secondary_emotion", "secondary_emotion_confidence",
]

ESSENTIAL_COLUMNS = [
    "review_id", "user_id", "business_id",
    "name", "review_date", "year_month",
    "stars", "weighted_star", "user_stars_avg", "user_review_count",
    "sentiment_score", "sentiment_confidence",
    "primary_emotion", "primary_emotion_confidence",
    "secondary_emotion", "secondary_emotion_confidence",
    "primary_emotion_lex", "primary_emotion_lex_conf",
    "secondary_emotion_lex", "secondary_emotion_lex_conf",
    "word_count", "char_count", "avg_word_len", "type_token_ratio",
    "noun_pct", "verb_pct", "adj_pct", "adv_pct",
    "subjectivity_score", "negation_count",
    "raw_review", "stripped_review",
    "person_count", "location_count", "product_count",
    "valence_avg", "arousal_avg", "dominance_avg",
    "worry_word_count", "wcst_warmth_avg", "wcst_competence_avg",
    "wcst_sociability_avg", "wcst_trust_avg", "yelp_sentiment_avg",
    "vader_sentiment_score",
]

BUSINESS_AGG_SPEC = {
    "review_id": "count",
    "stars": ["mean", "std", "min", "max", "median"],
    "weighted_star": "mean",
    "vader_sentiment_score": ["mean", "std", "median"],
    "dominant_emotion": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
    "word_count": "mean",
    "subjectivity_score": "mean",
    "negation_count": "mean",
    "type_token_ratio": "mean",
    "noun_pct": "mean",
    "verb_pct": "mean",
    "adj_pct": "mean",
    "adv_pct": "mean",
    "person_count": "mean",
    "location_count": "mean",
    "product_count": "mean",
}
BUSINESS_AGG_COLUMNS = [
    "business_id", "name", "city", "state", "is_open", "dataset",
    "review_count", "business_star_avg", "business_star_std", "business_star_min",
    "business_star_max", "business_star_median", "business_weighted_star_avg",
    "sentiment_score_mean", "sentiment_score_std", "sentiment_score_median",
    "primary_emotion_mode", "avg_review_length", "avg_subjectivity",
    "negation_rate", "avg_type_token_ratio", "avg_noun_pct", "avg_verb_pct",
    "avg_adj_pct", "avg_adv_pct", "avg_person_mentions", "avg_location_mentions",
    "avg_product_mentions",
]
PRIMARY_INDUSTRY = {"chipotle": "Mexican Restaurant", "hair": "Hair Salons"}

# Columns that are LEGITIMATELY constant in a given output and shouldn't trip
# the constant-column check below. Each *_sample_15k.parquet is single-industry
# by design, so `dataset`/`primary_industry` are supposed to be one value
# throughout that file - that's not a sign anything failed to load.
EXPECTED_CONSTANT = {
    "sample": {"dataset", "primary_industry"},
    "business_aggregate": set(),  # this one combines both industries - dataset/primary_industry SHOULD vary
}


def validate_outputs(df: pd.DataFrame, name: str, kind: str) -> list:
    """Same failure shape this repo's core validate() checks for: a column
    that's fully null, or stuck at one value, is exactly what a silent
    join/rename/lexicon failure looks like - not a legitimate feature.
    Returns a list of failure strings (empty means clean)."""
    failures = []
    if len(df) == 0:
        return [f"{name}: 0 rows - nothing to validate."]

    expected_constant = EXPECTED_CONSTANT.get(kind, set())
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            failures.append(f"{name}.{col}: fully null ({len(df):,} rows, 0 non-null).")
            continue
        if col in expected_constant:
            continue
        if non_null.nunique() <= 1:
            failures.append(
                f"{name}.{col}: constant ({non_null.iloc[0]!r} for every non-null row, "
                f"{len(non_null):,} checked) - expected to vary across reviews/businesses."
            )
    return failures


def _prepare(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAMES).copy()
    for sample_col, lexicon_col in FALLBACK_FROM_LEXICON.items():
        if sample_col not in df.columns and lexicon_col in df.columns:
            df[sample_col] = df[lexicon_col]
    df["dataset"] = dataset_name
    return df


def make_sample(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = _prepare(df, dataset_name)
    cols = [c for c in ESSENTIAL_COLUMNS if c in df.columns]
    missing = [c for c in ESSENTIAL_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [{dataset_name}] not present in gold-layer output, skipped: {missing}")

    if len(df) <= SAMPLE_SIZE:
        print(f"  [{dataset_name}] {len(df):,} rows <= {SAMPLE_SIZE:,}, using all rows.")
        return df[cols].copy()

    sampled, _ = train_test_split(
        df[cols], train_size=SAMPLE_SIZE, stratify=df["stars"], random_state=RANDOM_STATE
    )
    return sampled.reset_index(drop=True)


def make_business_aggregate(full_dfs: dict) -> pd.DataFrame:
    combined = []
    for name, df in full_dfs.items():
        combined.append(_prepare(df, name))
    combined_df = pd.concat(combined, ignore_index=True)

    group_keys = [c for c in ["business_id", "name", "city", "state", "is_open", "dataset"] if c in combined_df.columns]
    agg_spec = {k: v for k, v in BUSINESS_AGG_SPEC.items() if k in combined_df.columns}
    business_df = combined_df.groupby(group_keys).agg(agg_spec).reset_index()

    flat_cols = list(group_keys)
    for col, agg in agg_spec.items():
        if isinstance(agg, list):
            flat_cols.extend(f"{col}_{a}" for a in agg)
        else:
            flat_cols.append(col)
    business_df.columns = flat_cols

    legacy_rename = {
        "review_id_count": "review_count",
        "stars_mean": "business_star_avg", "stars_std": "business_star_std",
        "stars_min": "business_star_min", "stars_max": "business_star_max",
        "stars_median": "business_star_median",
        "weighted_star_mean": "business_weighted_star_avg",
        "vader_sentiment_score_mean": "sentiment_score_mean",
        "vader_sentiment_score_std": "sentiment_score_std",
        "vader_sentiment_score_median": "sentiment_score_median",
        "dominant_emotion": "primary_emotion_mode",
        "word_count_mean": "avg_review_length",
        "subjectivity_score_mean": "avg_subjectivity",
        "negation_count_mean": "negation_rate",
        "type_token_ratio_mean": "avg_type_token_ratio",
        "noun_pct_mean": "avg_noun_pct", "verb_pct_mean": "avg_verb_pct",
        "adj_pct_mean": "avg_adj_pct", "adv_pct_mean": "avg_adv_pct",
        "person_count_mean": "avg_person_mentions",
        "location_count_mean": "avg_location_mentions",
        "product_count_mean": "avg_product_mentions",
    }
    business_df = business_df.rename(columns=legacy_rename)
    if "dataset" in business_df.columns:
        business_df["primary_industry"] = business_df["dataset"].map(PRIMARY_INDUSTRY)

    ordered = [c for c in BUSINESS_AGG_COLUMNS + ["primary_industry"] if c in business_df.columns]
    return business_df[ordered]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chipotle-full", required=True, help="Gold-layer CSV for the Chipotle dataset")
    parser.add_argument("--hair-full", required=True, help="Gold-layer CSV for the Hair dataset")
    parser.add_argument("--outdir", default=".", help="Where to write the three dashboard input files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full_dfs = {
        "chipotle": pd.read_csv(args.chipotle_full),
        "hair": pd.read_csv(args.hair_full),
    }

    # Build and validate ALL THREE outputs before writing ANY of them. All-or-
    # nothing on purpose - the dashboard reads these three files as a set, so
    # writing the two that passed while silently skipping the one that failed
    # would leave a stale/inconsistent trio on disk, which is arguably worse
    # than refusing to write anything.
    print("Building 15k stratified samples (seed=222, stratified by stars)...")
    samples = {}
    all_failures = {}
    for name, df in full_dfs.items():
        samples[name] = make_sample(df, name)
        failures = validate_outputs(samples[name], f"{name}_sample_15k", kind="sample")
        if failures:
            all_failures[f"{name}_sample_15k"] = failures

    print("\nBuilding business-level aggregate...")
    business_df = make_business_aggregate(full_dfs)

    sample_business_ids = set()
    for df in samples.values():
        if "business_id" in df.columns:
            sample_business_ids |= set(df["business_id"].unique())
    if sample_business_ids and "business_id" in business_df.columns:
        before = len(business_df)
        business_df = business_df[business_df["business_id"].isin(sample_business_ids)].copy()
        print(f"  Restricted to businesses present in the samples: {before:,} -> {len(business_df):,}")

    failures = validate_outputs(business_df, "business_aggregated_sample", kind="business_aggregate")
    if failures:
        all_failures["business_aggregated_sample"] = failures

    if all_failures:
        print("\nFAILED VALIDATION - NOTHING was written (all-or-nothing):")
        for output_name, failure_list in all_failures.items():
            print(f"\n  {output_name}:")
            for f in failure_list:
                print(f"    - {f}")
        raise AssertionError(
            "One or more dashboard input files failed validation (fully-null or "
            "constant column) - none of the three were written. Fix the underlying "
            "gold-layer data or column mapping before re-running."
        )

    print("\nAll outputs passed validation (no fully-null or unexpectedly constant columns). Writing...")
    for name, df in samples.items():
        out_path = outdir / f"{name}_sample_15k.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        print(f"  Saved {out_path} ({len(df):,} rows x {len(df.columns)} cols)")

    out_path = outdir / "business_aggregated_sample.csv"
    business_df.to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(business_df):,} rows x {len(business_df.columns)} cols)")

    print("\nDone. The local dashboard input files are ready.")


if __name__ == "__main__":
    main()
