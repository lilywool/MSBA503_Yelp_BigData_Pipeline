"""Build scoped S3/Glue Bronze tables and a fully featured Silver table on EMR."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from bronze_json_to_parquet import DATASETS
from lexicon_cli import add_lexicon_args, resolve_lexicon_paths
from medallion_layers import require_columns, require_nonempty, sample_reviews, scope_related_dataset
from spark_feature_engineering import apply_features, build_silver_frame, cfe
from spark_validate import validate_spark_at_scale
from emr_common import (
    audit,
    ensure_database,
    join_uri,
    release_frames,
    table_name,
    write_parquet_table,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="S3 root containing five Parquet directories.")
    parser.add_argument("--output-root", required=True, help="S3 root for registered medallion outputs.")
    parser.add_argument("--audit-log", required=True, help="Append-only S3 JSON audit prefix.")
    parser.add_argument("--database", default="yelp")
    parser.add_argument("--table-prefix", default="yelp")
    parser.add_argument("--mode", choices=("overwrite", "errorifexists"), default="overwrite")
    parser.add_argument("--partitions", type=int, default=None)
    parser.add_argument("--arrow-batch-size", type=int, default=5000)
    parser.add_argument("--silver-sample-size", type=int, default=None)
    parser.add_argument("--silver-sample-seed", type=int, default=222)
    parser.add_argument(
        "--nlp-component",
        action="append",
        choices=cfe.FEATURE_GROUPS,
        dest="nlp_components",
        help="Repeat to select Silver feature families; omit for all standard non-transformer families.",
    )
    add_lexicon_args(parser)
    return parser.parse_args(argv)


def read_bronze_dataset(spark, spec, input_root: str):
    path = join_uri(input_root, spec.output_dirname)
    return (
        spark.read.schema(spec.schema).parquet(path)
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )


def validate_bronze(frame, spec, *, allow_empty=False) -> int:
    require_columns(frame, {spec.key_column}, f"Bronze {spec.name}")
    count = frame.count()
    if count == 0:
        if allow_empty:
            return 0
        raise ValueError(f"Bronze {spec.name} is empty")
    if frame.where(F.col(spec.key_column).isNull()).limit(1).count():
        raise ValueError(f"Bronze {spec.name} contains a null {spec.key_column}")
    if spec.name in {"review", "business", "user"}:
        if frame.groupBy(spec.key_column).count().where(F.col("count") > 1).limit(1).count():
            raise ValueError(f"Bronze {spec.name} contains duplicate {spec.key_column} values")
    return count


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.arrow_batch_size < 1:
        raise ValueError("--arrow-batch-size must be at least 1")
    spark = SparkSession.builder.appName("yelp-emr-bronze-to-silver").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(args.arrow_batch_size))
    ensure_database(spark, args.database)
    lexicon_paths = resolve_lexicon_paths(args)
    components = list(
        cfe.normalize_feature_groups(
            args.nlp_components,
            use_transformers=bool(args.nlp_components and "transformers" in args.nlp_components),
        )
    )
    metrics = {
        "input_root": args.input_root,
        "output_root": args.output_root,
        "silver_sample_size": args.silver_sample_size,
        "silver_sample_seed": args.silver_sample_seed,
        "nlp_components": components,
        "bronze_rows": {},
    }
    persisted = []
    try:
        selected_reviews = sample_reviews(
            read_bronze_dataset(spark, DATASETS["review"], args.input_root),
            args.silver_sample_size,
            args.silver_sample_seed,
        ).persist()
        persisted.append(selected_reviews)
        selected_reviews.count()

        bronze_tables = {}
        for dataset_name, spec in DATASETS.items():
            if dataset_name == "review":
                frame = selected_reviews
            else:
                source = read_bronze_dataset(spark, spec, args.input_root)
                frame = (
                    scope_related_dataset(source, dataset_name, selected_reviews).persist()
                    if args.silver_sample_size is not None
                    else source.persist()
                )
                persisted.append(frame)
                frame.count()
            metrics["bronze_rows"][dataset_name] = validate_bronze(
                frame,
                spec,
                allow_empty=(args.silver_sample_size is not None and dataset_name in {"checkin", "tip"}),
            )
            destination = table_name(args.database, f"{args.table_prefix}_bronze_{dataset_name}")
            bronze_tables[dataset_name] = destination
            write_parquet_table(
                frame,
                join_uri(args.output_root, "bronze", spec.output_dirname),
                destination,
                args.mode,
            )

        joined = build_silver_frame(
            spark.table(bronze_tables["review"]),
            spark.table(bronze_tables["business"]),
            spark.table(bronze_tables["user"]),
            broadcast_dimensions=True,
        ).persist()
        persisted.append(joined)
        joined.count()
        use_transformers = "transformers" in components
        silver = apply_features(
            spark,
            joined,
            "raw_review",
            lexicon_paths,
            use_transformers,
            args.partitions,
            feature_groups=components,
            use_broadcasts=True,
        ).persist()
        persisted.append(silver)
        require_columns(
            silver,
            {"review_id", "business_id", "user_id", "raw_review", "review_date"},
            "Silver reviews",
        )
        metrics["silver_rows"] = require_nonempty(silver, "Silver reviews")
        if metrics["silver_rows"] != metrics["bronze_rows"]["review"]:
            raise ValueError("Silver row count does not match selected Bronze reviews")
        metrics["validation"] = validate_spark_at_scale(silver, text_col="raw_review")
        silver_table = table_name(args.database, f"{args.table_prefix}_silver_reviews")
        silver_path = join_uri(args.output_root, "silver", "reviews")
        write_parquet_table(silver, silver_path, silver_table, args.mode)
        metrics.update({"silver_table": silver_table, "silver_path": silver_path})
        audit(spark, args.audit_log, "bronze_to_silver", "succeeded", metrics)
        print(f"EMR Silver is ready: {silver_table} at {silver_path}")
        return 0
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        try:
            audit(spark, args.audit_log, "bronze_to_silver", "failed", metrics)
        except Exception as audit_exc:
            print(f"WARNING: failed to record EMR audit: {audit_exc}")
        raise
    finally:
        release_frames(persisted)


if __name__ == "__main__":
    raise SystemExit(main())
