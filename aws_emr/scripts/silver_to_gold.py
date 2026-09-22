"""Build one validated Glue/S3 Gold variant from EMR Silver reviews."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from medallion_layers import (
    DATE_GRANULARITIES,
    EMOTION_COLUMNS,
    GOLD_LEVELS,
    SENTIMENT_COLUMNS,
    apply_gold_filters,
    build_gold_variant,
    effective_sample_size,
    validate_gold,
)
from emr_common import audit, ensure_database, release_frames, table_name, write_parquet_table


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="yelp")
    parser.add_argument("--table-prefix", default="yelp")
    parser.add_argument("--input-table", default=None)
    parser.add_argument("--output-table", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--audit-log", required=True)
    parser.add_argument("--gold-level", choices=GOLD_LEVELS, required=True)
    parser.add_argument("--gold-sample-size", type=int)
    parser.add_argument("--gold-sample-seed", type=int, default=222)
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
    parser.add_argument("--sentiment-column", choices=SENTIMENT_COLUMNS, default="vader_sentiment_score")
    parser.add_argument("--min-sentiment", type=float)
    parser.add_argument("--max-sentiment", type=float)
    parser.add_argument("--min-stars", type=float)
    parser.add_argument("--max-stars", type=float)
    parser.add_argument("--date-granularity", choices=DATE_GRANULARITIES, default="month")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date must not be after --end-date")
    if args.min_sentiment is not None and args.max_sentiment is not None and args.min_sentiment > args.max_sentiment:
        raise ValueError("--min-sentiment must not exceed --max-sentiment")
    if args.min_stars is not None and args.max_stars is not None and args.min_stars > args.max_stars:
        raise ValueError("--min-stars must not exceed --max-stars")

    spark = SparkSession.builder.appName("yelp-emr-silver-to-gold").getOrCreate()
    ensure_database(spark, args.database)
    input_table = args.input_table or table_name(args.database, f"{args.table_prefix}_silver_reviews")
    output_table = args.output_table or table_name(
        args.database, f"{args.table_prefix}_gold_{args.gold_level.replace('-', '_')}"
    )
    metrics = {
        "input_table": input_table,
        "output_table": output_table,
        "output_path": args.output_path,
        "gold_level": args.gold_level,
        "requested_gold_sample_size": args.gold_sample_size,
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
    persisted = []
    try:
        filtered = apply_gold_filters(
            spark.table(input_table),
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
        ).persist()
        persisted.append(filtered)
        metrics["eligible_silver_rows"] = filtered.count()
        sample_size = effective_sample_size(args.gold_sample_size, metrics["eligible_silver_rows"])
        metrics["effective_gold_sample_size"] = sample_size
        if args.gold_sample_size is not None and sample_size < args.gold_sample_size:
            print(
                f"WARNING: requested {args.gold_sample_size:,} Gold rows but only "
                f"{metrics['eligible_silver_rows']:,} Silver rows are eligible; using {sample_size:,}."
            )
        gold = build_gold_variant(
            filtered,
            gold_level=args.gold_level,
            sample_size=sample_size,
            sample_seed=args.gold_sample_seed,
            brands=args.brands,
            date_granularity=args.date_granularity,
            emotion_column=args.emotion_column,
            sentiment_column=args.sentiment_column,
        ).withColumn("gold_created_at", F.current_timestamp()).persist()
        persisted.append(gold)
        metrics["validation"] = validate_gold(
            gold,
            args.gold_level,
            sample_size,
            expected_variants=args.brands if args.gold_level == "brand-sample" else None,
        )
        write_parquet_table(gold, args.output_path, output_table, "overwrite")
        audit(spark, args.audit_log, "silver_to_gold", "succeeded", metrics)
        print(f"EMR Gold is ready: {output_table} at {args.output_path}")
        return 0
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        try:
            audit(spark, args.audit_log, "silver_to_gold", "failed", metrics)
        except Exception as audit_exc:
            print(f"WARNING: failed to record EMR audit: {audit_exc}")
        raise
    finally:
        release_frames(persisted)


if __name__ == "__main__":
    raise SystemExit(main())
