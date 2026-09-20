"""Publish dashboard tables that depend directly on one row-level Gold variant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    append_audit,
    ensure_schema,
    require_columns,
    require_nonempty,
    table_name,
    write_delta_table,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="default")
    parser.add_argument("--table-prefix", default="yelp")
    parser.add_argument("--gold-table", default=None)
    return parser.parse_args(argv)


def _avg_if_present(columns: set[str], name: str, alias: str):
    if name in columns:
        return F.avg(F.col(name)).alias(alias)
    return F.first(F.lit(None).cast("double")).alias(alias)


def build_business_summary(gold):
    columns = set(gold.columns)
    expressions = [
        F.first("name", ignorenulls=True).alias("business_name"),
        F.first("city", ignorenulls=True).alias("city"),
        F.first("state", ignorenulls=True).alias("state"),
        F.first("primary_industry", ignorenulls=True).alias("primary_industry"),
        F.count("review_id").alias("review_count"),
        F.avg("stars").alias("avg_stars"),
        F.avg(F.when(F.col("stars") <= 2, 1.0).otherwise(0.0)).alias("low_star_rate"),
        _avg_if_present(columns, "weighted_star", "avg_weighted_star"),
        _avg_if_present(columns, "vader_sentiment_score", "avg_sentiment"),
        _avg_if_present(columns, "anger_int_avg", "avg_anger"),
        _avg_if_present(columns, "fear_int_avg", "avg_fear"),
        _avg_if_present(columns, "sadness_int_avg", "avg_sadness"),
        _avg_if_present(columns, "joy_int_avg", "avg_joy"),
    ]
    summary = gold.groupBy("gold_variant_value", "business_id").agg(*expressions)
    risk_score = F.least(
        F.lit(1.0),
        F.greatest(F.lit(0.0), (F.lit(5.0) - F.col("avg_stars")) / F.lit(4.0)) * 0.45
        + F.col("low_star_rate") * 0.35
        + F.greatest(F.lit(0.0), -F.coalesce(F.col("avg_sentiment"), F.lit(0.0))) * 0.20,
    )
    return (
        summary.withColumn("risk_score", F.round(risk_score, 4))
        .withColumn(
            "attention_tier",
            F.when(F.col("risk_score") >= 0.65, "high")
            .when(F.col("risk_score") >= 0.40, "moderate")
            .otherwise("monitor"),
        )
        .withColumn(
            "recommended_focus",
            F.when(F.col("low_star_rate") >= 0.35, "Review recurring low-star pain points")
            .when(F.col("avg_sentiment") < -0.15, "Investigate negative sentiment drivers")
            .when(F.col("avg_joy") < F.col("avg_sadness"), "Strengthen positive service moments")
            .otherwise("Maintain performance and monitor trends"),
        )
    )


def build_monthly_summary(gold):
    columns = set(gold.columns)
    month = F.coalesce(
        F.col("year_month") if "year_month" in columns else F.lit(None),
        F.date_format(F.col("review_date"), "yyyy-MM"),
    )
    return (
        gold.withColumn("year_month", month)
        .groupBy("gold_variant_value", "year_month", "primary_industry")
        .agg(
            F.count("review_id").alias("review_count"),
            F.avg("stars").alias("avg_stars"),
            F.avg(F.when(F.col("stars") <= 2, 1.0).otherwise(0.0)).alias("low_star_rate"),
            _avg_if_present(columns, "vader_sentiment_score", "avg_sentiment"),
        )
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    spark = SparkSession.builder.appName("yelp-data-science-dashboard").getOrCreate()
    ensure_schema(spark, args.catalog, args.schema)

    gold_table = args.gold_table or table_name(
        args.catalog, args.schema, f"{args.table_prefix}_gold_dashboard_variant"
    )
    sample_table = table_name(args.catalog, args.schema, f"{args.table_prefix}_dashboard_review_sample")
    business_table = table_name(args.catalog, args.schema, f"{args.table_prefix}_dashboard_business_summary")
    monthly_table = table_name(args.catalog, args.schema, f"{args.table_prefix}_dashboard_monthly_summary")
    audit_table = table_name(args.catalog, args.schema, f"{args.table_prefix}_pipeline_audit")
    metrics = {"gold_table": gold_table}

    try:
        gold = spark.table(gold_table)
        require_columns(
            gold,
            {
                "review_id", "business_id", "stars", "raw_review", "review_date",
                "primary_industry", "gold_variant_level", "gold_variant_value",
            },
            "Dashboard Gold variant",
        )
        levels = [row[0] for row in gold.select("gold_variant_level").distinct().collect()]
        unsupported = sorted(set(levels) - {"sample", "brand-sample"})
        if unsupported:
            raise ValueError(
                "The dashboard task requires a row-level Gold sample; "
                f"received levels {unsupported}"
            )

        business = build_business_summary(gold)
        monthly = build_monthly_summary(gold)
        metrics["gold_rows"] = require_nonempty(gold, "Dashboard Gold variant")
        metrics["variants"] = [
            row[0] for row in gold.select("gold_variant_value").distinct().collect()
        ]
        metrics["business_rows"] = require_nonempty(business, "Dashboard business summary")
        metrics["monthly_rows"] = require_nonempty(monthly, "Dashboard monthly summary")
        if (
            business.groupBy("gold_variant_value", "business_id")
            .count().where(F.col("count") > 1).limit(1).count()
        ):
            raise ValueError("Dashboard business summary has duplicate variant/business keys")

        write_delta_table(gold, sample_table)
        write_delta_table(business, business_table)
        write_delta_table(monthly, monthly_table)
        metrics["output_tables"] = [sample_table, business_table, monthly_table]
        append_audit(spark, audit_table, "data_science_dashboard", "succeeded", metrics)
        print("Dashboard-serving tables are ready:")
        for destination in metrics["output_tables"]:
            print(f"  - {destination}")
        return 0
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        try:
            append_audit(spark, audit_table, "data_science_dashboard", "failed", metrics)
        except Exception as audit_exc:
            print(f"WARNING: failed to append the failure audit record: {audit_exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
