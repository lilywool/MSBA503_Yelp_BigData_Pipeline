"""Convert the five documented Yelp JSON-lines datasets to Bronze Parquet.

The converter is intentionally independent of the feature pipeline. It uses
explicit schemas, creates multiple Parquet part files, and is safe to rerun:
an output containing Spark's ``_SUCCESS`` marker is skipped by default. Use
``--overwrite`` only when deliberately rebuilding an existing dataset.

Example:
    spark-submit bronze_json_to_parquet.py \
        --input-root s3://MY_BUCKET/raw \
        --output-root s3://MY_BUCKET/parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)


def _field(name, data_type):
    return StructField(name, data_type, True)


REVIEW_SCHEMA = StructType([
    _field("review_id", StringType()),
    _field("user_id", StringType()),
    _field("business_id", StringType()),
    _field("stars", DoubleType()),
    _field("useful", LongType()),
    _field("funny", LongType()),
    _field("cool", LongType()),
    _field("text", StringType()),
    _field("date", StringType()),
])

BUSINESS_SCHEMA = StructType([
    _field("business_id", StringType()),
    _field("name", StringType()),
    _field("address", StringType()),
    _field("city", StringType()),
    _field("state", StringType()),
    _field("postal_code", StringType()),
    _field("latitude", DoubleType()),
    _field("longitude", DoubleType()),
    _field("stars", DoubleType()),
    _field("review_count", LongType()),
    _field("is_open", LongType()),
    _field("attributes", MapType(StringType(), StringType(), True)),
    _field("categories", StringType()),
    _field("hours", MapType(StringType(), StringType(), True)),
])

USER_SCHEMA = StructType([
    _field("user_id", StringType()),
    _field("name", StringType()),
    _field("review_count", LongType()),
    _field("yelping_since", StringType()),
    _field("useful", LongType()),
    _field("funny", LongType()),
    _field("cool", LongType()),
    _field("elite", StringType()),
    _field("friends", StringType()),
    _field("fans", LongType()),
    _field("average_stars", DoubleType()),
    _field("compliment_hot", LongType()),
    _field("compliment_more", LongType()),
    _field("compliment_profile", LongType()),
    _field("compliment_cute", LongType()),
    _field("compliment_list", LongType()),
    _field("compliment_note", LongType()),
    _field("compliment_plain", LongType()),
    _field("compliment_cool", LongType()),
    _field("compliment_funny", LongType()),
    _field("compliment_writer", LongType()),
    _field("compliment_photos", LongType()),
])

CHECKIN_SCHEMA = StructType([
    _field("business_id", StringType()),
    _field("date", StringType()),
])

TIP_SCHEMA = StructType([
    _field("user_id", StringType()),
    _field("business_id", StringType()),
    _field("text", StringType()),
    _field("date", StringType()),
    _field("compliment_count", LongType()),
])


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    output_dirname: str
    key_column: str
    schema: StructType
    default_partitions: int


DATASETS = {
    "review": DatasetSpec(
        "review",
        "yelp_academic_dataset_review.json",
        "yelp_academic_dataset_review",
        "review_id",
        REVIEW_SCHEMA,
        192,
    ),
    "business": DatasetSpec(
        "business",
        "yelp_academic_dataset_business.json",
        "yelp_academic_dataset_business",
        "business_id",
        BUSINESS_SCHEMA,
        8,
    ),
    "checkin": DatasetSpec(
        "checkin",
        "yelp_academic_dataset_checkin.json",
        "yelp_academic_dataset_checkin",
        "business_id",
        CHECKIN_SCHEMA,
        8,
    ),
    "tip": DatasetSpec(
        "tip",
        "yelp_academic_dataset_tip.json",
        "yelp_academic_dataset_tip",
        "business_id",
        TIP_SCHEMA,
        16,
    ),
    "user": DatasetSpec(
        "user",
        "yelp_academic_dataset_user.json",
        "yelp_academic_dataset_user",
        "user_id",
        USER_SCHEMA,
        32,
    ),
}


def join_path(root: str, child: str) -> str:
    """Join local, DBFS, and object-storage paths without changing the URI."""
    clean_root = root.rstrip("/\\")
    return f"{clean_root}/{child}"


def _filesystem(spark, path: str):
    jpath = spark._jvm.org.apache.hadoop.fs.Path(path)
    fs = jpath.getFileSystem(spark._jsc.hadoopConfiguration())
    return fs, jpath


def path_exists(spark, path: str) -> bool:
    fs, jpath = _filesystem(spark, path)
    return bool(fs.exists(jpath))


def convert_dataset(
    spark,
    spec: DatasetSpec,
    input_path: str,
    output_path: str,
    partitions: int,
    overwrite: bool = False,
) -> str:
    """Convert one dataset and return ``converted`` or ``skipped``."""
    if partitions < 1:
        raise ValueError(f"{spec.name} partitions must be at least 1")

    output_exists = path_exists(spark, output_path)
    success_path = join_path(output_path, "_SUCCESS")
    if output_exists and not overwrite:
        if path_exists(spark, success_path):
            print(f"[bronze] SKIP {spec.name}: complete output already exists at {output_path}")
            return "skipped"
        raise RuntimeError(
            f"Refusing incomplete existing output at {output_path}. "
            "Inspect it, then rerun with --overwrite only if rebuilding is intended."
        )

    print(f"[bronze] READ {spec.name}: {input_path}")
    frame = (
        spark.read
        .schema(spec.schema)
        .option("mode", "FAILFAST")
        .option("multiLine", False)
        .json(input_path)
    )

    if spec.key_column not in frame.columns:
        raise RuntimeError(f"Explicit schema lost required key column {spec.key_column}")

    mode = "overwrite" if overwrite else "errorifexists"
    print(f"[bronze] WRITE {spec.name}: {output_path} ({partitions} Spark partitions)")
    (
        frame.repartition(partitions)
        .write
        .mode(mode)
        .option("compression", "snappy")
        .parquet(output_path)
    )
    print(f"[bronze] COMPLETE {spec.name}: {output_path}")
    return "converted"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Explicit-schema Yelp JSON-lines to Bronze Parquet converter."
    )
    parser.add_argument("--input-root", required=True,
                        help="Directory/URI containing the three Yelp JSON files")
    parser.add_argument("--output-root", required=True,
                        help="Directory/URI that will receive the three Parquet datasets")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS),
        help="Datasets to convert (default: all five)",
    )
    parser.add_argument("--review-partitions", type=int,
                        default=DATASETS["review"].default_partitions)
    parser.add_argument("--business-partitions", type=int,
                        default=DATASETS["business"].default_partitions)
    parser.add_argument("--user-partitions", type=int,
                        default=DATASETS["user"].default_partitions)
    parser.add_argument("--checkin-partitions", type=int,
                        default=DATASETS["checkin"].default_partitions)
    parser.add_argument("--tip-partitions", type=int,
                        default=DATASETS["tip"].default_partitions)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Deliberately rebuild outputs; without this flag complete outputs are skipped",
    )
    parser.add_argument("--master", default=None,
                        help="Optional local Spark master, for example local[4]")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    builder = SparkSession.builder.appName("yelp-bronze-json-to-parquet")
    if args.master:
        builder = builder.master(args.master)
    spark = builder.getOrCreate()

    partition_counts = {
        "review": args.review_partitions,
        "business": args.business_partitions,
        "user": args.user_partitions,
        "checkin": args.checkin_partitions,
        "tip": args.tip_partitions,
    }
    try:
        for dataset_name in args.datasets:
            spec = DATASETS[dataset_name]
            convert_dataset(
                spark=spark,
                spec=spec,
                input_path=join_path(args.input_root, spec.filename),
                output_path=join_path(args.output_root, spec.output_dirname),
                partitions=partition_counts[dataset_name],
                overwrite=args.overwrite,
            )
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
