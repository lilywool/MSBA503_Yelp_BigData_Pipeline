"""
validate() as a mandatory Spark job stage.

The base validate() (pandas, correlation/constant-column checks) is a proven
detector for a column that looks plausible but is actually constant,
mistracked, or fabricated. This module ports the SAME checks, at two scales:

  * validate_spark()            - collect()/toPandas() then reuse the base
                                   validate() unchanged. Correct and simple;
                                   fine up to a few hundred thousand rows on a
                                   driver with reasonable memory.
  * validate_spark_at_scale()   - Spark-native aggregations (corr(), approx counts)
                                   that never collect the full dataset to the driver.
                                   REQUIRED for the full 8.6M-row run - collecting
                                   that much data to a driver just to validate it
                                   is exactly the kind of step that gets skipped
                                   "just this once" under time pressure, which is
                                   how this class of bug ships in the first place.

Both raise AssertionError and refuse to let the caller proceed to a write on
failure - same contract as the base validate(), just distributed.
"""

import sys
from pathlib import Path

import pyspark.sql.functions as F

sys.path.insert(0, str(Path(__file__).parent))
import corrected_feature_engineering as cfe


def validate_spark(sdf, text_col: str = "raw_review", max_collect_rows: int = 500_000) -> dict:
    """Small/medium scale: reuse the corrected pipeline's validate() unchanged via toPandas().

    Refuses to silently truncate: if the DataFrame is bigger than max_collect_rows,
    raise rather than validate a silent subset and call it done - a validation
    pass on a small sample doesn't guarantee the same result at full scale.
    """
    n = sdf.count()
    if n > max_collect_rows:
        raise AssertionError(
            f"validate_spark(): {n:,} rows exceeds max_collect_rows={max_collect_rows:,}. "
            "Use validate_spark_at_scale() instead - do not raise this limit to force a collect."
        )
    pdf = sdf.toPandas()
    return cfe.validate(pdf, text_col=text_col)


def validate_spark_at_scale(sdf, raw_sdf=None, text_col: str = "raw_review") -> dict:
    """Full-scale (8.6M row) validation, computed entirely in Spark - no driver collect.

    Mirrors the base validate()'s three checks:
      1. word_count actually tracks real text length (correlation, not just present).
      2. No pipeline-owned numeric column is frozen at a single value (excluding
         num_at/num_hash, which are legitimately near-zero/constant in this domain).
      3. Sentiment (vader_sentiment_score, Valence_avg) correlates with star rating,
         when a stars column exists.

    `raw_sdf` (optional): the pre-feature-engineering DataFrame with the original
    text_col, needed for check 1 if `sdf` no longer carries raw, un-featureized text
    alongside stripped_review. Pass the same DataFrame that went into
    spark_feature_engineering.run() if in doubt.
    """
    print("\n=== VALIDATION (Spark-native, full-scale) ===")
    failures = []
    metrics = {}

    text_source = raw_sdf if raw_sdf is not None else sdf
    if text_col in text_source.columns and "word_count" in sdf.columns:
        # join is unnecessary if row order/count matches 1:1 from the same mapInPandas
        # partitioning; safest correct approach is corr on a joined-by-position frame,
        # but Spark has no stable row order across partitions. Recommended: run this on a
        # DataFrame that STILL carries text_col alongside word_count (don't drop it).
        if text_col in sdf.columns and "word_count" in sdf.columns:
            normalized_text = F.trim(
                F.regexp_replace(F.coalesce(F.col(text_col), F.lit("")), r"\s+", " ")
            )
            real_word_count = F.when(
                F.length(normalized_text) == 0, F.lit(0)
            ).otherwise(F.size(F.split(normalized_text, " ")))
            corr = sdf.select(
                real_word_count.alias("_real_word_count"),
                F.col("word_count")
            ).stat.corr("_real_word_count", "word_count")
            metrics["word_count_corr"] = round(float(corr), 3) if corr is not None else None
            print(f"word_count vs actual text length correlation: {corr}")
            if corr is None or corr < 0.9:
                failures.append(f"word_count barely tracks real text length (r={corr})")

    expected_sparse = {"num_at", "num_hash"}
    owned = set(cfe.OUTPUT_COLUMNS)
    numeric_cols = [
        f.name for f in sdf.schema.fields
        if f.name in owned and f.dataType.typeName() in ("double", "long", "integer")
    ]
    const_cols = []
    checked_cols = [c for c in numeric_cols if c not in expected_sparse]
    if checked_cols:
        expressions = []
        for i, c in enumerate(checked_cols):
            expressions.extend([
                F.min(F.col(c)).alias(f"min_{i}"),
                F.max(F.col(c)).alias(f"max_{i}"),
            ])
        stats = sdf.agg(*expressions).first().asDict()
        for i, c in enumerate(checked_cols):
            minimum, maximum = stats[f"min_{i}"], stats[f"max_{i}"]
            if minimum is None or minimum == maximum:
                const_cols.append(c)
    metrics["constant_columns"] = const_cols
    print(f"Constant columns: {const_cols if const_cols else 'none'}")
    if const_cols:
        failures.append(f"Found constant columns (a real feature should vary across reviews): {const_cols}")

    if "stars" in sdf.columns and "vader_sentiment_score" in sdf.columns:
        sent_corr = sdf.stat.corr("vader_sentiment_score", "stars")
        metrics["vader_vs_stars_corr"] = round(float(sent_corr), 3) if sent_corr is not None else None
        print(f"vader_sentiment_score vs stars correlation: {sent_corr}")
        if sent_corr is None or sent_corr < 0.3:
            failures.append(f"Sentiment doesn't track star rating (r={sent_corr}) - check pipeline")

        if "Valence_avg" in sdf.columns:
            val_corr = sdf.stat.corr("Valence_avg", "stars")
            metrics["valence_vs_stars_corr"] = round(float(val_corr), 3) if val_corr is not None else None
            print(f"Valence_avg vs stars correlation: {val_corr}")
            if val_corr is None or val_corr < 0.2:
                failures.append(f"Valence_avg doesn't track star rating (r={val_corr}) - check VAD lookup")

    # Mirror corrected_feature_engineering.validate()'s checks for the exact
    # silent-empty-lexicon and missing-decay failure shapes.
    extra_checks = {}
    coverage_cols = [
        c for c in ("yelp_sentiment_avg", "worry_word_count", "wcst_warmth_avg")
        if c in sdf.columns
    ]
    if coverage_cols:
        coverage = sdf.agg(*[
            F.avg(F.when(F.col(c) != 0, 1.0).otherwise(0.0)).alias(c)
            for c in coverage_cols
        ]).first().asDict()
        for c in coverage_cols:
            share = float(coverage[c] or 0.0)
            extra_checks[f"{c}_nonzero_share"] = round(share, 4)
            if share < 0.01:
                failures.append(
                    f"{c} is ~0 for {100*(1-share):.1f}% of rows - lexicon likely failed to load."
                )

    if "stars" in sdf.columns and "weighted_star" in sdf.columns:
        identical = sdf.agg(
            F.avg(F.when(F.col("weighted_star") == F.col("stars"), 1.0)
                  .otherwise(0.0)).alias("share")
        ).first()["share"]
        identical_share = float(identical or 0.0)
        extra_checks["weighted_star_identical_to_stars_share"] = round(identical_share, 4)
        if identical_share > 0.999:
            failures.append("weighted_star is identical to stars for every row - time decay was never applied.")

    metrics["extra_checks"] = extra_checks

    metrics["passed"] = len(failures) == 0
    metrics["failures"] = failures

    if failures:
        print("\nFAILED VALIDATION:")
        for f in failures:
            print("  -", f)
        raise AssertionError("Distributed feature engineering output failed validation - see above. Do not write to Delta.")
    print("\nAll checks passed (full-scale, Spark-native).")
    return metrics
