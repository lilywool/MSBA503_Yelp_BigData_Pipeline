"""Publish dashboard tables that depend directly on one row-level Gold variant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
sys.path.insert(0, str(SCRIPT_PATH.parent))

from medallion_layers import (  # noqa: E402
    build_business_summary,
    build_monthly_summary,
    require_columns,
    require_nonempty,
)
from common import append_audit, ensure_schema, table_name, write_delta_table  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="default")
    parser.add_argument("--table-prefix", default="yelp")
    parser.add_argument("--gold-table", default=None)
    return parser.parse_args(argv)


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
        levels = {row[0] for row in gold.select("gold_variant_level").distinct().collect()}
        unsupported = sorted(levels - {"sample", "brand-sample"})
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
        if business.groupBy("gold_variant_value", "business_id").count().where(
            F.col("count") > 1
        ).limit(1).count():
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
