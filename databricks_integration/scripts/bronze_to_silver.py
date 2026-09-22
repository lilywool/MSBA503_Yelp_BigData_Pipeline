"""Load Yelp Bronze data and publish the fully featured Silver Delta table."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

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
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(SCRIPT_PATH.parent))

from bronze_json_to_parquet import DATASETS, join_path  # noqa: E402
from lexicon_cli import (  # noqa: E402
    LEXICON_FILENAMES,
    add_lexicon_args,
    resolve_lexicon_paths,
)
from spark_feature_engineering import apply_features, build_silver_frame, cfe  # noqa: E402
from spark_validate import validate_spark_at_scale  # noqa: E402
from common import (  # noqa: E402
    add_materialization_args,
    append_audit,
    ensure_schema,
    materialize_frame,
    parse_materialization_schema,
    release_resources,
    require_columns,
    require_nonempty,
    resolve_materialization_mode,
    table_name,
    write_delta_table,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-volume",
        default=None,
        help=(
            "Volume/local/object-storage directory containing the five Yelp "
            "Parquet dataset directories produced by bronze_json_to_parquet.py."
        ),
    )
    source.add_argument(
        "--google-drive-folder-url",
        default=None,
        help="Google Drive folder containing the five Yelp JSON files.",
    )
    parser.add_argument(
        "--google-drive-connection",
        default=None,
        help="Unity Catalog Google Drive connection used with --google-drive-folder-url.",
    )
    parser.add_argument(
        "--google-drive-lexicons-folder-url",
        default=None,
        help=(
            "Optional Google Drive folder containing the five licensed lexicons. "
            "Defaults to --google-drive-folder-url, so one Drive folder may contain "
            "both the Yelp JSON files and lexicons."
        ),
    )
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="default")
    parser.add_argument("--table-prefix", default="yelp")
    parser.add_argument("--mode", choices=("overwrite", "errorifexists"), default="overwrite")
    parser.add_argument("--partitions", type=int, default=None)
    parser.add_argument("--arrow-batch-size", type=int, default=5000)
    add_materialization_args(parser)
    parser.add_argument(
        "--silver-sample-size",
        "--sample-size",
        dest="silver_sample_size",
        type=int,
        default=None,
        help=(
            "Maximum deterministic review sample that is retained in Bronze and "
            "receives Silver feature engineering. The --sample-size spelling is a "
            "backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--silver-sample-seed",
        "--sample-seed",
        dest="silver_sample_seed",
        type=int,
        default=222,
    )
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
    args = parser.parse_args(argv)
    if args.google_drive_folder_url and not args.google_drive_connection:
        parser.error(
            "--google-drive-connection is required with --google-drive-folder-url"
        )
    if args.google_drive_connection and not args.google_drive_folder_url:
        parser.error(
            "--google-drive-folder-url is required with --google-drive-connection"
        )
    if args.google_drive_lexicons_folder_url and not args.google_drive_folder_url:
        parser.error(
            "--google-drive-lexicons-folder-url requires --google-drive-folder-url"
        )
    return args


def _has_explicit_lexicon_paths(args) -> bool:
    return any(
        getattr(args, name, None)
        for name in (
            "lexicons_dir",
            "vad_lexicon",
            "worry_lexicon",
            "wcst_lexicon",
            "yelp_lexicon",
            "nrc_intensity_lexicon",
        )
    )


@contextmanager
def _google_drive_lexicon_paths(
    spark,
    *,
    folder_url: str,
    connection: str,
):
    """Materialize five small Drive-hosted lexicons for feature workers.

    The Google Drive connector is a Spark data source, while the feature loaders
    need ordinary files. Reading each exact filename as ``binaryFile`` preserves
    its bytes; the temporary driver copies remain alive until all Silver actions
    have completed. ``apply_features`` then distributes them to executors through
    the serverless byte-payload path (or SparkFiles on classic Spark).
    """
    with TemporaryDirectory(prefix="yelp_drive_lexicons_") as temp_dir:
        paths = {}
        for key, filename in LEXICON_FILENAMES.items():
            matches = (
                spark.read.format("binaryFile")
                .option("databricks.connection", connection)
                .option("pathGlobFilter", filename)
                .option("recursiveFileLookup", True)
                .load(folder_url)
                .select("path", "content")
                .collect()
            )
            if len(matches) != 1:
                found = [row["path"] for row in matches]
                raise ValueError(
                    f"Expected exactly one Google Drive lexicon named {filename!r}; "
                    f"found {len(matches)}: {found}"
                )
            target = Path(temp_dir) / filename
            target.write_bytes(bytes(matches[0]["content"]))
            paths[key] = str(target)
        yield paths


@contextmanager
def _feature_lexicon_paths(spark, args):
    """Resolve explicit files, or stage all five from the configured Drive folder."""
    if _has_explicit_lexicon_paths(args):
        yield resolve_lexicon_paths(args)
        return
    if args.google_drive_folder_url:
        with _google_drive_lexicon_paths(
            spark,
            folder_url=(
                args.google_drive_lexicons_folder_url
                or args.google_drive_folder_url
            ),
            connection=args.google_drive_connection,
        ) as paths:
            yield paths
        return
    # Non-Drive deployments must provide --lexicons-dir or all five overrides.
    yield resolve_lexicon_paths(args)


def _google_drive_file_url(
    spark,
    *,
    folder_url: str,
    connection: str,
    filename: str,
) -> str:
    """Resolve one exact Drive filename to the connector's readable file URL.

    Databricks can enumerate a Drive folder through ``binaryFile`` even when a
    structured reader cannot open that folder URL directly. Selecting only
    ``path`` lets column pruning avoid materializing the file content.
    """
    matches = (
        spark.read.format("binaryFile")
        .option("databricks.connection", connection)
        .option("pathGlobFilter", filename)
        .option("recursiveFileLookup", True)
        .load(folder_url)
        .select("path")
        .collect()
    )
    if len(matches) != 1:
        found = [row["path"] for row in matches]
        raise ValueError(
            f"Expected exactly one Google Drive file named {filename!r}; "
            f"found {len(matches)}: {found}"
        )
    return matches[0]["path"]


def _read_dataset(
    spark,
    spec,
    *,
    input_volume: str | None,
    google_drive_folder_url: str | None,
    google_drive_connection: str | None,
):
    if google_drive_folder_url:
        file_url = _google_drive_file_url(
            spark,
            folder_url=google_drive_folder_url,
            connection=google_drive_connection,
            filename=spec.filename,
        )
        frame = (
            spark.read
            .schema(spec.schema)
            .option("mode", "FAILFAST")
            .option("multiLine", False)
            .option("databricks.connection", google_drive_connection)
            .format("json")
            .load(file_url)
        )
    else:
        parquet_path = join_path(input_volume, spec.output_dirname)
        frame = spark.read.schema(spec.schema).parquet(parquet_path)
    return (
        frame
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )


def _validate_bronze(frame, spec, *, allow_empty: bool = False) -> int:
    require_columns(frame, {spec.key_column}, f"Bronze {spec.name}")
    count = frame.count()
    if count == 0:
        if allow_empty:
            return 0
        raise ValueError(f"Bronze {spec.name} is empty")
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
        raise ValueError("--silver-sample-size must be at least 1")
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


def scope_related_dataset(
    frame,
    dataset_name: str,
    selected_reviews,
    *,
    use_broadcasts: bool = True,
):
    """Keep only dimension/fact rows related to the selected review sample."""
    if dataset_name == "review":
        return selected_reviews
    if dataset_name == "user":
        keys = selected_reviews.select("user_id").where(
            F.col("user_id").isNotNull()
        ).distinct()
        join_keys = F.broadcast(keys) if use_broadcasts else keys
        return frame.join(join_keys, "user_id", "left_semi")
    if dataset_name in {"business", "checkin", "tip"}:
        keys = selected_reviews.select("business_id").where(
            F.col("business_id").isNotNull()
        ).distinct()
        join_keys = F.broadcast(keys) if use_broadcasts else keys
        return frame.join(join_keys, "business_id", "left_semi")
    raise ValueError(f"Unsupported Yelp dataset: {dataset_name}")


def configure_arrow_runtime(spark, batch_size: int) -> dict[str, bool]:
    """Apply optional classic-Spark Arrow settings when the runtime permits it.

    Databricks serverless uses Spark Connect and manages these settings itself.
    Its configuration API rejects both keys with CONFIG_NOT_AVAILABLE; that is
    an expected managed-runtime boundary, not a pipeline failure.
    """
    settings = {
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.maxRecordsPerBatch": str(batch_size),
    }
    applied = {}
    for key, value in settings.items():
        try:
            spark.conf.set(key, value)
            applied[key] = True
        except Exception as exc:
            if "CONFIG_NOT_AVAILABLE" not in str(exc):
                raise
            applied[key] = False
            print(
                "INFO: runtime manages unavailable Spark configuration "
                f"{key}; continuing with the serverless default."
            )
    return applied


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.arrow_batch_size < 1:
        raise ValueError("--arrow-batch-size must be at least 1")
    spark = SparkSession.builder.appName("yelp-bronze-to-silver").getOrCreate()
    arrow_runtime_config = configure_arrow_runtime(spark, args.arrow_batch_size)
    materialization_mode = resolve_materialization_mode(args.materialization_mode)
    materialization_catalog, materialization_schema = parse_materialization_schema(
        args.materialization_schema
    )
    ensure_schema(spark, args.catalog, args.schema)
    ensure_schema(spark, materialization_catalog, materialization_schema)
    audit_table = table_name(args.catalog, args.schema, f"{args.table_prefix}_pipeline_audit")
    run_id = uuid4().hex
    metrics = {
        "input_source": (
            {
                "kind": "google_drive",
                "folder_url": args.google_drive_folder_url,
                "connection": args.google_drive_connection,
            }
            if args.google_drive_folder_url
            else {"kind": "volume", "path": args.input_volume}
        ),
        "lexicon_source": (
            {
                "kind": "explicit_paths",
            }
            if _has_explicit_lexicon_paths(args)
            else {
                "kind": "google_drive",
                "folder_url": (
                    args.google_drive_lexicons_folder_url
                    or args.google_drive_folder_url
                ),
                "connection": args.google_drive_connection,
            }
        ),
        "bronze_rows": {},
        "silver_sample_size": args.silver_sample_size,
        "silver_sample_seed": args.silver_sample_seed,
        "arrow_batch_size_requested": args.arrow_batch_size,
        "arrow_runtime_config_applied": arrow_runtime_config,
        "materialization": {
            "requested_mode": args.materialization_mode,
            "mode": materialization_mode,
            "schema": args.materialization_schema,
            "run_id": run_id,
            "temporary_tables": [],
        },
        "nlp_components": list(
            cfe.normalize_feature_groups(
                args.nlp_components,
                use_transformers=bool(
                    args.nlp_components and "transformers" in args.nlp_components
                ),
            )
        ),
    }
    persisted_frames = []

    try:
        destinations = {}
        review_spec = DATASETS["review"]
        selected_reviews_plan = sample_reviews(
            _read_dataset(
                spark,
                review_spec,
                input_volume=args.input_volume,
                google_drive_folder_url=args.google_drive_folder_url,
                google_drive_connection=args.google_drive_connection,
            ),
            args.silver_sample_size,
            args.silver_sample_seed,
        )
        selected_reviews = materialize_frame(
            spark,
            selected_reviews_plan,
            mode=materialization_mode,
            materialization_schema=args.materialization_schema,
            stage="selected_reviews",
            run_id=run_id,
            metadata=metrics,
            persisted_frames=persisted_frames,
        )

        for dataset_name, spec in DATASETS.items():
            if dataset_name == "review":
                frame = selected_reviews
            else:
                source_frame = _read_dataset(
                    spark,
                    spec,
                    input_volume=args.input_volume,
                    google_drive_folder_url=args.google_drive_folder_url,
                    google_drive_connection=args.google_drive_connection,
                )
                if args.silver_sample_size is not None:
                    frame_plan = scope_related_dataset(
                        source_frame,
                        dataset_name,
                        selected_reviews,
                        use_broadcasts=(materialization_mode == "persist"),
                    )
                else:
                    frame_plan = source_frame
                frame = materialize_frame(
                    spark,
                    frame_plan,
                    mode=materialization_mode,
                    materialization_schema=args.materialization_schema,
                    stage=f"bronze_{dataset_name}",
                    run_id=run_id,
                    metadata=metrics,
                    persisted_frames=persisted_frames,
                )
            destinations[dataset_name] = table_name(
                args.catalog, args.schema, f"{args.table_prefix}_bronze_{dataset_name}"
            )
            metrics["bronze_rows"][dataset_name] = _validate_bronze(
                frame,
                spec,
                allow_empty=(
                    args.silver_sample_size is not None
                    and dataset_name in {"checkin", "tip"}
                ),
            )
            write_delta_table(frame, destinations[dataset_name], args.mode)

        joined_plan = build_silver_frame(
            spark.table(destinations["review"]),
            spark.table(destinations["business"]),
            spark.table(destinations["user"]),
            broadcast_dimensions=(materialization_mode == "persist"),
        )
        joined = materialize_frame(
            spark,
            joined_plan,
            mode=materialization_mode,
            materialization_schema=args.materialization_schema,
            stage="core_cleaning",
            run_id=run_id,
            metadata=metrics,
            persisted_frames=persisted_frames,
        )
        use_transformers = bool(
            args.nlp_components and "transformers" in args.nlp_components
        )
        with _feature_lexicon_paths(spark, args) as lexicon_paths:
            silver_plan = apply_features(
                spark=spark,
                sdf=joined,
                text_col="raw_review",
                lexicon_paths=lexicon_paths,
                use_transformers=use_transformers,
                num_partitions=args.partitions,
                feature_groups=args.nlp_components,
                use_broadcasts=(materialization_mode == "persist"),
            )
            silver = materialize_frame(
                spark,
                silver_plan,
                mode=materialization_mode,
                materialization_schema=args.materialization_schema,
                stage="silver",
                run_id=run_id,
                metadata=metrics,
                persisted_frames=persisted_frames,
            )
            require_columns(
                silver,
                {"review_id", "business_id", "user_id", "raw_review", "review_date"},
                "Silver reviews",
            )
            silver_rows = require_nonempty(silver, "Silver reviews")
            expected_silver_rows = metrics["bronze_rows"]["review"]
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
    finally:
        release_resources(spark, persisted_frames, metrics)


if __name__ == "__main__":
    raise SystemExit(main())
