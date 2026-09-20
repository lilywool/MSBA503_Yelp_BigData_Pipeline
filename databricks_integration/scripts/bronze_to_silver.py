"""Load Yelp Bronze data and publish the fully featured Silver Delta table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bronze_json_to_parquet import DATASETS, join_path  # noqa: E402
from lexicon_cli import add_lexicon_args, resolve_lexicon_paths  # noqa: E402
from spark_feature_engineering import apply_features, build_silver_frame, cfe  # noqa: E402
from spark_validate import validate_spark_at_scale  # noqa: E402
from common import (  # noqa: E402
    append_audit,
    ensure_schema,
    require_columns,
    require_nonempty,
    table_name,
    write_delta_table,
)


DEFAULT_INPUT_VOLUME = "/Volumes/workspace/default/yelp_academic_raw"
DEFAULT_LEXICON_DIR = "/Volumes/workspace/default/yelp_academic_raw/_pipeline/lexicons"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-volume", default=DEFAULT_INPUT_VOLUME)
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="default")
    parser.add_argument("--table-prefix", default="yelp")
    parser.add_argument("--mode", choices=("overwrite", "errorifexists"), default="overwrite")
    parser.add_argument("--partitions", type=int, default=None)
    parser.add_argument("--arrow-batch-size", type=int, default=5000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Build Silver from a deterministic review subset for a canary run.",
    )
    parser.add_argument("--sample-seed", type=int, default=222)
    parser.add_argument(
        "--nlp-component",
        action="append",
        choices=cfe.FEATURE_GROUPS,
        dest="nlp_components",
        help=(
            "Repeat to choose Silver feature families. Omit for every standard "
            "family except transformers."
        ),
    )
    add_lexicon_args(parser)
    parser.set_defaults(lexicons_dir=DEFAULT_LEXICON_DIR)
    return parser.parse_args(argv)


def _read_json(spark, spec, input_volume: str):
    source = join_path(input_volume, spec.filename)
    return (
        spark.read.schema(spec.schema)
        .option("mode", "FAILFAST")
        .option("multiLine", False)
        .json(source)
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_ingested_at", F.current_timestamp())
    )


def _validate_bronze(frame, spec) -> int:
    require_columns(frame, {spec.key_column}, f"Bronze {spec.name}")
    count = require_nonempty(frame, f"Bronze {spec.name}")
    null_keys = frame.where(F.col(spec.key_column).isNull()).limit(1).count()
    if null_keys:
        raise ValueError(f"Bronze {spec.name} contains a null {spec.key_column}")
    if spec.name in {"review", "business", "user"}:
        duplicates = (
            frame.groupBy(spec.key_column).count().where(F.col("count") > 1).limit(1).count()
        )
        if duplicates:
            raise ValueError(
                f"Bronze {spec.name} contains duplicate {spec.key_column} values"
            )
    return count


def sample_reviews(frame, sample_size: int | None, seed: int):
    """Return a partition-independent review subset, or the full frame."""
    if sample_size is None:
        return frame
    if sample_size < 1:
        raise ValueError("--sample-size must be at least 1")
    require_columns(frame, {"review_id"}, "Deterministic Silver sample")
    return (
        frame.withColumn(
            "_sample_hash",
            F.xxhash64(F.lit(int(seed)), F.col("review_id").cast("string")),
        )
        .orderBy("_sample_hash", "review_id")
        .limit(sample_size)
        .drop("_sample_hash")
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.arrow_batch_size < 1:
        raise ValueError("--arrow-batch-size must be at least 1")
    spark = SparkSession.builder.appName("yelp-bronze-to-silver").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", args.arrow_batch_size)
    ensure_schema(spark, args.catalog, args.schema)
    audit_table = table_name(args.catalog, args.schema, f"{args.table_prefix}_pipeline_audit")
    metrics = {
        "input_volume": args.input_volume,
        "bronze_rows": {},
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "nlp_components": list(
            cfe.normalize_feature_groups(
                args.nlp_components,
                use_transformers=bool(
                    args.nlp_components and "transformers" in args.nlp_components
                ),
            )
        ),
    }

    try:
        destinations = {}
        for dataset_name, spec in DATASETS.items():
            frame = _read_json(spark, spec, args.input_volume).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            destinations[dataset_name] = table_name(
                args.catalog, args.schema, f"{args.table_prefix}_bronze_{dataset_name}"
            )
            try:
                metrics["bronze_rows"][dataset_name] = _validate_bronze(frame, spec)
                write_delta_table(frame, destinations[dataset_name], args.mode)
            finally:
                frame.unpersist()

        joined = build_silver_frame(
            spark.table(destinations["review"]),
            spark.table(destinations["business"]),
            spark.table(destinations["user"]),
        )
        joined = sample_reviews(joined, args.sample_size, args.sample_seed)
        use_transformers = bool(
            args.nlp_components and "transformers" in args.nlp_components
        )
        silver = apply_features(
            spark=spark,
            sdf=joined,
            text_col="raw_review",
            lexicon_paths=resolve_lexicon_paths(args),
            use_transformers=use_transformers,
            num_partitions=args.partitions,
            feature_groups=args.nlp_components,
        ).persist(StorageLevel.MEMORY_AND_DISK)
        try:
            require_columns(
                silver,
                {"review_id", "business_id", "user_id", "raw_review", "review_date"},
                "Silver reviews",
            )
            silver_rows = require_nonempty(silver, "Silver reviews")
            expected_silver_rows = (
                metrics["bronze_rows"]["review"]
                if args.sample_size is None
                else min(args.sample_size, metrics["bronze_rows"]["review"])
            )
            if silver_rows != expected_silver_rows:
                raise ValueError(
                    "Silver row count does not match the selected Bronze review count: "
                    f"{silver_rows} != {expected_silver_rows}"
                )
            metrics["silver_rows"] = silver_rows
            metrics["validation"] = validate_spark_at_scale(
                silver, text_col="raw_review"
            )
            metrics["silver_table"] = table_name(
                args.catalog, args.schema, f"{args.table_prefix}_silver_reviews"
            )

            write_delta_table(silver, metrics["silver_table"], args.mode)
        finally:
            silver.unpersist()

        append_audit(spark, audit_table, "bronze_to_silver", "succeeded", metrics)
        print(f"Bronze and Silver tables are ready. Silver: {metrics['silver_table']}")
        return 0
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        try:
            append_audit(spark, audit_table, "bronze_to_silver", "failed", metrics)
        except Exception as audit_exc:
            print(f"WARNING: failed to append the failure audit record: {audit_exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
