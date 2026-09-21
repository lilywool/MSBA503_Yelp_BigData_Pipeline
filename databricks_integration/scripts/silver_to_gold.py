"""Build one validated, parameterized Gold variant from featured Silver reviews."""

from __future__ import annotations

import argparse
import sys
from functools import reduce
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


_ENTRYPOINT = globals().get("__file__") or globals().get("filename")
if not _ENTRYPOINT:
    raise RuntimeError(
        "Unable to locate the Python task file. Expected __file__ or the "
        "Databricks Workspace wrapper variable 'filename'."
    )
SCRIPT_PATH = Path(_ENTRYPOINT).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(SCRIPT_PATH.parent))

from common import (  # noqa: E402
    append_audit,
    ensure_schema,
    require_columns,
    table_name,
    write_delta_table,
)


GOLD_LEVELS = (
    "brand-sample",
    "sample",
    "business",
    "user",
    "date",
    "industry",
    "sentiment",
    "emotion",
)
DATE_GRANULARITIES = ("day", "week", "month", "quarter", "year")
EMOTION_COLUMNS = ("dominant_emotion", "primary_emotion_lex", "hf_emotion_label")
SENTIMENT_COLUMNS = (
    "vader_sentiment_score",
    "hf_computed_sentiment",
    "yelp_sentiment_avg",
    "Valence_avg",
)
AVERAGE_METRICS = (
    "stars",
    "weighted_star",
    "vader_sentiment_score",
    "Valence_avg",
    "Arousal_avg",
    "Dominance_avg",
    "word_count",
    "subjectivity_score",
    "anger_int_avg",
    "fear_int_avg",
    "joy_int_avg",
    "sadness_int_avg",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="default")
    parser.add_argument("--table-prefix", default="yelp")
    parser.add_argument("--input-table", default=None)
    parser.add_argument("--output-table", default=None)
    parser.add_argument("--gold-level", choices=GOLD_LEVELS, required=True)
    parser.add_argument(
        "--gold-sample-size",
        "--sample-size",
        dest="gold_sample_size",
        type=int,
        help=(
            "Maximum Silver reviews retained in a row-level Gold sample. For "
            "brand-sample this limit is applied independently to each brand. "
            "The --sample-size spelling is a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--gold-sample-seed",
        "--sample-seed",
        dest="gold_sample_seed",
        type=int,
        default=222,
    )
    parser.add_argument("--brand", action="append", dest="brands")
    parser.add_argument("--business-id", action="append", dest="business_ids")
    parser.add_argument("--business-name", action="append", dest="business_names")
    parser.add_argument("--user-id", action="append", dest="user_ids")
    parser.add_argument("--industry", action="append", dest="industries")
    parser.add_argument("--city", action="append", dest="cities")
    parser.add_argument("--state", action="append", dest="states")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--sentiment-label",
        action="append",
        choices=("negative", "neutral", "positive"),
        dest="sentiment_labels",
    )
    parser.add_argument("--emotion-label", action="append", dest="emotion_labels")
    parser.add_argument("--emotion-column", choices=EMOTION_COLUMNS, default="dominant_emotion")
    parser.add_argument(
        "--sentiment-column",
        choices=SENTIMENT_COLUMNS,
        default="vader_sentiment_score",
    )
    parser.add_argument("--min-sentiment", type=float)
    parser.add_argument("--max-sentiment", type=float)
    parser.add_argument("--min-stars", type=float)
    parser.add_argument("--max-stars", type=float)
    parser.add_argument("--date-granularity", choices=DATE_GRANULARITIES, default="month")
    return parser.parse_args(argv)


def _contains_any(column, values: list[str]):
    normalized = F.lower(F.regexp_replace(F.coalesce(column, F.lit("")), r"\s+", " "))
    predicates = [normalized.contains(" ".join(value.lower().split())) for value in values]
    return reduce(lambda left, right: left | right, predicates)


def _sentiment_label(column_name: str):
    """Map the selected score to three bands using its documented scale."""
    score = F.col(column_name)
    if column_name == "Valence_avg":
        return F.when(score >= 0.55, "positive").when(
            score <= 0.45, "negative"
        ).otherwise("neutral")
    return F.when(score >= 0.05, "positive").when(
        score <= -0.05, "negative"
    ).otherwise("neutral")


def apply_gold_filters(
    silver,
    *,
    business_ids=None,
    business_names=None,
    user_ids=None,
    industries=None,
    cities=None,
    states=None,
    start_date=None,
    end_date=None,
    sentiment_labels=None,
    emotion_labels=None,
    emotion_column="dominant_emotion",
    sentiment_column="vader_sentiment_score",
    min_sentiment=None,
    max_sentiment=None,
    min_stars=None,
    max_stars=None,
):
    """Apply optional row-level dependencies before the Gold variant is built."""
    frame = silver
    if business_ids:
        require_columns(frame, {"business_id"}, "Business ID filtering")
        frame = frame.where(F.col("business_id").isin(business_ids))
    if business_names:
        require_columns(frame, {"name"}, "Business name filtering")
        frame = frame.where(_contains_any(F.col("name"), business_names))
    if user_ids:
        require_columns(frame, {"user_id"}, "User filtering")
        frame = frame.where(F.col("user_id").isin(user_ids))
    if industries:
        industry_predicates = []
        for column in ("primary_industry", "secondary_industry", "tertiary_industry"):
            if column in frame.columns:
                industry_predicates.append(_contains_any(F.col(column), industries))
        if not industry_predicates:
            raise ValueError("Industry filtering requires at least one industry column")
        frame = frame.where(reduce(lambda left, right: left | right, industry_predicates))
    if cities:
        require_columns(frame, {"city"}, "City filtering")
        frame = frame.where(F.lower(F.col("city")).isin([value.lower() for value in cities]))
    if states:
        require_columns(frame, {"state"}, "State filtering")
        frame = frame.where(F.upper(F.col("state")).isin([value.upper() for value in states]))
    if start_date:
        require_columns(frame, {"review_date"}, "Start-date filtering")
        frame = frame.where(F.to_date("review_date") >= F.to_date(F.lit(start_date)))
    if end_date:
        require_columns(frame, {"review_date"}, "End-date filtering")
        frame = frame.where(F.to_date("review_date") <= F.to_date(F.lit(end_date)))
    if sentiment_labels:
        require_columns(frame, {sentiment_column}, "Sentiment filtering")
        sentiment = _sentiment_label(sentiment_column)
        frame = frame.where(sentiment.isin(sentiment_labels))
    if emotion_labels:
        require_columns(frame, {emotion_column}, "Emotion filtering")
        frame = frame.where(
            F.lower(F.col(emotion_column)).isin(
                [value.lower() for value in emotion_labels]
            )
        )
    if min_sentiment is not None:
        require_columns(frame, {sentiment_column}, "Sentiment score filtering")
        frame = frame.where(F.col(sentiment_column) >= min_sentiment)
    if max_sentiment is not None:
        require_columns(frame, {sentiment_column}, "Sentiment score filtering")
        frame = frame.where(F.col(sentiment_column) <= max_sentiment)
    if min_stars is not None:
        require_columns(frame, {"stars"}, "Star rating filtering")
        frame = frame.where(F.col("stars") >= min_stars)
    if max_stars is not None:
        require_columns(frame, {"stars"}, "Star rating filtering")
        frame = frame.where(F.col("stars") <= max_stars)
    return frame


def deterministic_sample(frame, sample_size: int, seed: int):
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    require_columns(frame, {"review_id"}, "Deterministic Gold sample")
    return (
        frame.withColumn(
            "_sample_hash",
            F.xxhash64(F.lit(int(seed)), F.col("review_id").cast("string")),
        )
        .orderBy("_sample_hash", "review_id")
        .limit(sample_size)
        .drop("_sample_hash")
    )


def effective_sample_size(requested_size: int | None, eligible_rows: int) -> int | None:
    """Cap a requested Gold sample at the filtered Silver population."""
    if requested_size is None:
        return None
    if requested_size < 1:
        raise ValueError("--gold-sample-size must be at least 1")
    if eligible_rows < 0:
        raise ValueError("eligible_rows cannot be negative")
    return min(requested_size, eligible_rows)


def build_brand_sample(frame, brands: list[str], sample_size: int, seed: int):
    require_columns(frame, {"review_id", "name"}, "Brand Gold sample")
    pieces = []
    for brand in brands:
        matched = frame.where(_contains_any(F.col("name"), [brand]))
        pieces.append(
            deterministic_sample(matched, sample_size, seed)
            .withColumn("gold_variant_level", F.lit("brand-sample"))
            .withColumn("gold_variant_value", F.lit(brand))
            .withColumn("gold_sample_seed", F.lit(seed))
        )
    return reduce(lambda left, right: left.unionByName(right), pieces)


def _aggregate_metrics(frame, group_columns: list[str]):
    expressions = [
        F.count("review_id").alias("review_count"),
        F.avg(F.when(F.col("stars") <= 2, 1.0).otherwise(0.0)).alias("low_star_rate"),
        F.countDistinct("business_id").alias("unique_business_count"),
        F.countDistinct("user_id").alias("unique_user_count"),
    ]
    for column in AVERAGE_METRICS:
        if column in frame.columns:
            expressions.append(F.avg(F.col(column)).alias(f"avg_{column}"))
    return frame.groupBy(*group_columns).agg(*expressions)


def build_gold_variant(
    frame,
    *,
    gold_level: str,
    sample_size: int | None,
    sample_seed: int,
    brands: list[str] | None,
    date_granularity: str,
    emotion_column: str,
    sentiment_column: str,
):
    if gold_level == "brand-sample":
        if not brands:
            raise ValueError("brand-sample requires at least one --brand")
        if sample_size is None:
            raise ValueError("brand-sample requires --sample-size")
        return build_brand_sample(frame, brands, sample_size, sample_seed)
    if gold_level == "sample":
        if sample_size is None:
            raise ValueError("sample requires --sample-size")
        return (
            deterministic_sample(frame, sample_size, sample_seed)
            .withColumn("gold_variant_level", F.lit("sample"))
            .withColumn("gold_variant_value", F.lit("filtered_reviews"))
            .withColumn("gold_sample_seed", F.lit(sample_seed))
        )
    if gold_level == "business":
        require_columns(frame, {"business_id"}, "Business Gold")
        dimensions = frame.groupBy("business_id").agg(
            F.first("name", ignorenulls=True).alias("business_name"),
            F.first("city", ignorenulls=True).alias("city"),
            F.first("state", ignorenulls=True).alias("state"),
            F.first("primary_industry", ignorenulls=True).alias("primary_industry"),
        )
        result = _aggregate_metrics(frame, ["business_id"]).join(
            dimensions, "business_id", "left"
        )
        return result.withColumn("gold_variant_level", F.lit("business")).withColumn(
            "gold_variant_value", F.col("business_id")
        )
    if gold_level == "user":
        require_columns(frame, {"user_id"}, "User Gold")
        dimensions = frame.groupBy("user_id").agg(
            F.first("user_name", ignorenulls=True).alias("user_name")
        )
        result = _aggregate_metrics(frame, ["user_id"]).join(dimensions, "user_id", "left")
        return result.withColumn("gold_variant_level", F.lit("user")).withColumn(
            "gold_variant_value", F.col("user_id")
        )
    if gold_level == "date":
        require_columns(frame, {"review_date"}, "Date Gold")
        dated = frame.withColumn(
            "date_period", F.date_trunc(date_granularity, F.to_timestamp("review_date"))
        ).where(F.col("date_period").isNotNull())
        return _aggregate_metrics(dated, ["date_period"]).withColumn(
            "gold_variant_level", F.lit("date")
        ).withColumn("gold_variant_value", F.col("date_period").cast("string"))
    if gold_level == "industry":
        require_columns(frame, {"primary_industry"}, "Industry Gold")
        return _aggregate_metrics(
            frame.where(F.col("primary_industry").isNotNull()), ["primary_industry"]
        ).withColumn("gold_variant_level", F.lit("industry")).withColumn(
            "gold_variant_value", F.col("primary_industry")
        )
    if gold_level == "sentiment":
        require_columns(frame, {sentiment_column}, "Sentiment Gold")
        labeled = frame.withColumn(
            "sentiment_label",
            _sentiment_label(sentiment_column),
        )
        return _aggregate_metrics(labeled, ["sentiment_label"]).withColumn(
            "gold_variant_level", F.lit("sentiment")
        ).withColumn("gold_variant_value", F.col("sentiment_label"))
    require_columns(frame, {emotion_column}, "Emotion Gold")
    labeled = frame.withColumn("emotion_label", F.lower(F.col(emotion_column))).where(
        F.col("emotion_label").isNotNull()
    )
    return _aggregate_metrics(labeled, ["emotion_label"]).withColumn(
        "gold_variant_level", F.lit("emotion")
    ).withColumn("gold_variant_value", F.col("emotion_label"))


def validate_gold(
    frame,
    gold_level: str,
    sample_size: int | None,
    expected_variants: list[str] | None = None,
) -> dict:
    rows = frame.count()
    failures = []
    if rows == 0:
        failures.append("Gold variant is empty")
    key_map = {
        "sample": ["review_id"],
        "brand-sample": ["gold_variant_value", "review_id"],
        "business": ["business_id"],
        "user": ["user_id"],
        "date": ["date_period"],
        "industry": ["primary_industry"],
        "sentiment": ["sentiment_label"],
        "emotion": ["emotion_label"],
    }
    keys = key_map[gold_level]
    if frame.groupBy(*keys).count().where(F.col("count") > 1).limit(1).count():
        failures.append(f"Gold keys are not unique: {keys}")
    if gold_level == "sample" and sample_size is not None and rows > sample_size:
        failures.append("Sample Gold contains more rows than requested")
    if gold_level == "brand-sample" and sample_size is not None:
        require_columns(frame, {"gold_variant_value"}, "Brand Gold validation")
        oversized = (
            frame.groupBy("gold_variant_value").count()
            .where(F.col("count") > sample_size)
            .limit(1)
            .count()
        )
        if oversized:
            failures.append("A brand sample contains more rows than requested")
        if expected_variants:
            actual = {row[0] for row in frame.select("gold_variant_value").distinct().collect()}
            missing = set(expected_variants) - actual
            if missing:
                failures.append(f"No eligible reviews for requested brands: {sorted(missing)}")
    if failures:
        raise AssertionError("Gold validation failed: " + "; ".join(failures))
    return {"passed": True, "gold_level": gold_level, "gold_rows": rows, "keys": keys}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date must not be after --end-date")
    if args.min_sentiment is not None and args.max_sentiment is not None and args.min_sentiment > args.max_sentiment:
        raise ValueError("--min-sentiment must not exceed --max-sentiment")
    if args.min_stars is not None and args.max_stars is not None and args.min_stars > args.max_stars:
        raise ValueError("--min-stars must not exceed --max-stars")
    spark = SparkSession.builder.appName("yelp-silver-to-gold").getOrCreate()
    ensure_schema(spark, args.catalog, args.schema)
    input_table = args.input_table or table_name(
        args.catalog, args.schema, f"{args.table_prefix}_silver_reviews"
    )
    output_table = args.output_table or table_name(
        args.catalog,
        args.schema,
        f"{args.table_prefix}_gold_{args.gold_level.replace('-', '_')}",
    )
    audit_table = table_name(args.catalog, args.schema, f"{args.table_prefix}_pipeline_audit")
    metrics = {
        "input_table": input_table,
        "output_table": output_table,
        "gold_level": args.gold_level,
        "gold_sample_size": args.gold_sample_size,
        "gold_sample_seed": args.gold_sample_seed,
        "filters": {
            "business_ids": args.business_ids,
            "business_names": args.business_names,
            "user_ids": args.user_ids,
            "industries": args.industries,
            "cities": args.cities,
            "states": args.states,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "sentiment_labels": args.sentiment_labels,
            "emotion_labels": args.emotion_labels,
            "emotion_column": args.emotion_column,
            "sentiment_column": args.sentiment_column,
            "min_sentiment": args.min_sentiment,
            "max_sentiment": args.max_sentiment,
            "min_stars": args.min_stars,
            "max_stars": args.max_stars,
            "brands": args.brands,
        },
    }
    gold = None
    try:
        silver = spark.table(input_table)
        filtered = apply_gold_filters(
            silver,
            business_ids=args.business_ids,
            business_names=args.business_names,
            user_ids=args.user_ids,
            industries=args.industries,
            cities=args.cities,
            states=args.states,
            start_date=args.start_date,
            end_date=args.end_date,
            sentiment_labels=args.sentiment_labels,
            emotion_labels=args.emotion_labels,
            emotion_column=args.emotion_column,
            sentiment_column=args.sentiment_column,
            min_sentiment=args.min_sentiment,
            max_sentiment=args.max_sentiment,
            min_stars=args.min_stars,
            max_stars=args.max_stars,
        )
        metrics["eligible_silver_rows"] = filtered.count()
        effective_gold_sample_size = effective_sample_size(
            args.gold_sample_size,
            metrics["eligible_silver_rows"],
        )
        metrics["requested_gold_sample_size"] = args.gold_sample_size
        metrics["effective_gold_sample_size"] = effective_gold_sample_size
        if (
            args.gold_sample_size is not None
            and effective_gold_sample_size < args.gold_sample_size
        ):
            print(
                "WARNING: requested Gold sample size "
                f"{args.gold_sample_size:,} exceeds the eligible Silver population "
                f"of {metrics['eligible_silver_rows']:,}; using "
                f"{effective_gold_sample_size:,}."
            )
        gold = build_gold_variant(
            filtered,
            gold_level=args.gold_level,
            sample_size=effective_gold_sample_size,
            sample_seed=args.gold_sample_seed,
            brands=args.brands,
            date_granularity=args.date_granularity,
            emotion_column=args.emotion_column,
            sentiment_column=args.sentiment_column,
        ).withColumn("gold_created_at", F.current_timestamp()).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        metrics["validation"] = validate_gold(
            gold,
            args.gold_level,
            effective_gold_sample_size,
            expected_variants=args.brands if args.gold_level == "brand-sample" else None,
        )
        write_delta_table(gold, output_table, "overwrite")
        append_audit(spark, audit_table, "silver_to_gold", "succeeded", metrics)
        print(f"Validated Gold variant is ready: {output_table}")
        return 0
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        try:
            append_audit(spark, audit_table, "silver_to_gold", "failed", metrics)
        except Exception as audit_exc:
            print(f"WARNING: failed to append the failure audit record: {audit_exc}")
        raise
    finally:
        if gold is not None:
            gold.unpersist()


if __name__ == "__main__":
    raise SystemExit(main())
