"""Publish S3/Glue dashboard-serving tables from one row-level EMR Gold variant."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from medallion_layers import (
    build_business_summary,
    build_monthly_summary,
    require_columns,
    require_nonempty,
)
from emr_common import audit, ensure_database, join_uri, table_name, write_parquet_table


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="yelp")
    parser.add_argument("--table-prefix", default="yelp")
    parser.add_argument("--gold-table", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit-log", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    spark = SparkSession.builder.appName("yelp-emr-data-science-dashboard").getOrCreate()
    ensure_database(spark, args.database)
    gold_table = args.gold_table or table_name(
        args.database, f"{args.table_prefix}_gold_dashboard_variant"
    )
    outputs = {
        "review_sample": (
            table_name(args.database, f"{args.table_prefix}_dashboard_review_sample"),
            join_uri(args.output_root, "review_sample"),
        ),
        "business_summary": (
            table_name(args.database, f"{args.table_prefix}_dashboard_business_summary"),
            join_uri(args.output_root, "business_summary"),
        ),
        "monthly_summary": (
            table_name(args.database, f"{args.table_prefix}_dashboard_monthly_summary"),
            join_uri(args.output_root, "monthly_summary"),
        ),
    }
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
        levels = {row[0] for row in gold.select("gold_variant_level").distinct().collect()}
        unsupported = sorted(levels - {"sample", "brand-sample"})
        if unsupported:
            raise ValueError(
                "Dashboard outputs require a row-level Gold sample; "
                f"received levels {unsupported}"
            )
        business = build_business_summary(gold)
        monthly = build_monthly_summary(gold)
        metrics["gold_rows"] = require_nonempty(gold, "Dashboard Gold variant")
        metrics["business_rows"] = require_nonempty(business, "Dashboard business summary")
        metrics["monthly_rows"] = require_nonempty(monthly, "Dashboard monthly summary")
        metrics["variants"] = [
            row[0] for row in gold.select("gold_variant_value").distinct().collect()
        ]
        if business.groupBy("gold_variant_value", "business_id").count().where(
            F.col("count") > 1
        ).limit(1).count():
            raise ValueError("Dashboard business summary has duplicate variant/business keys")

        frames = {
            "review_sample": gold,
            "business_summary": business,
            "monthly_summary": monthly,
        }
        for key, frame in frames.items():
            destination, path = outputs[key]
            write_parquet_table(frame, path, destination, "overwrite")
        metrics["outputs"] = {
            key: {"table": table, "path": path}
            for key, (table, path) in outputs.items()
        }
        audit(spark, args.audit_log, "data_science_dashboard", "succeeded", metrics)
        print("EMR dashboard-serving tables are ready:")
        for value in metrics["outputs"].values():
            print(f"  - {value['table']} at {value['path']}")
        return 0
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        try:
            audit(spark, args.audit_log, "data_science_dashboard", "failed", metrics)
        except Exception as audit_exc:
            print(f"WARNING: failed to record EMR audit: {audit_exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
