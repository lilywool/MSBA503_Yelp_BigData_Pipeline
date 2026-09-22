"""Small shared helpers for the three Databricks job tasks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MATERIALIZATION_MODES = ("auto", "persist", "delta", "none")


def table_name(catalog: str, schema: str, name: str) -> str:
    parts = (catalog, schema, name)
    invalid = [part for part in parts if not IDENTIFIER.fullmatch(part)]
    if invalid:
        raise ValueError(
            f"Catalog identifiers may contain only letters, numbers, and underscores: {invalid}"
        )
    return ".".join(parts)


def ensure_schema(spark, catalog: str, schema: str) -> None:
    table_name(catalog, schema, "validation_only")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")


def add_materialization_args(parser) -> None:
    parser.add_argument(
        "--materialization-mode",
        choices=MATERIALIZATION_MODES,
        default="auto",
        help=(
            "Reusable-frame strategy: auto selects Delta on Spark Connect/serverless "
            "and persist on classic Spark."
        ),
    )
    parser.add_argument(
        "--materialization-schema",
        default="workspace.default",
        help="Two-part CATALOG.SCHEMA used for temporary managed Delta tables.",
    )


def resolve_materialization_mode(mode: str) -> str:
    if mode not in MATERIALIZATION_MODES:
        raise ValueError(
            f"invalid materialization mode {mode!r}; expected one of "
            f"{MATERIALIZATION_MODES}"
        )
    if mode != "auto":
        return mode
    try:
        from pyspark.sql.utils import is_remote

        return "delta" if is_remote() else "persist"
    except (ImportError, TypeError):
        return "persist"


def parse_materialization_schema(value: str) -> tuple[str, str]:
    parts = value.split(".")
    if len(parts) != 2:
        raise ValueError(
            "--materialization-schema must be a two-part CATALOG.SCHEMA name"
        )
    catalog, schema = parts
    table_name(catalog, schema, "validation_only")
    return catalog, schema


def quote_table_name(value: str) -> str:
    parts = value.split(".")
    if len(parts) != 3:
        raise ValueError(f"Expected a three-part table name, got {value!r}")
    table_name(*parts)
    return ".".join(f"`{part}`" for part in parts)


def materialize_frame(
    spark,
    result_plan,
    *,
    mode: str,
    materialization_schema: str,
    stage: str,
    run_id: str,
    metadata: dict,
    persisted_frames: list,
):
    """Materialize a reusable plan without using cache/persist on serverless."""
    if mode not in {"persist", "delta", "none"}:
        raise ValueError(f"Materialization mode must be resolved first, got {mode!r}")
    if not IDENTIFIER.fullmatch(stage):
        raise ValueError(f"Invalid materialization stage identifier: {stage!r}")
    if not IDENTIFIER.fullmatch(run_id):
        raise ValueError(f"Invalid materialization run identifier: {run_id!r}")
    materialization_metadata = metadata.get("materialization")
    if not isinstance(materialization_metadata, dict) or not isinstance(
        materialization_metadata.get("temporary_tables"), list
    ):
        raise ValueError(
            "metadata.materialization.temporary_tables must be initialized as a list"
        )

    if mode == "persist":
        result = result_plan.persist()
        persisted_frames.append(result)
        result.count()
        return result

    if mode == "delta":
        catalog, schema = parse_materialization_schema(materialization_schema)
        scratch_table = table_name(
            catalog,
            schema,
            f"_yelp_{stage}_{run_id}",
        )
        try:
            (
                result_plan.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(scratch_table)
            )
            result = spark.table(scratch_table)
        except Exception:
            try:
                spark.sql(f"DROP TABLE IF EXISTS {quote_table_name(scratch_table)}")
            except Exception as cleanup_exc:
                print(
                    "WARNING: failed to remove incomplete materialization table "
                    f"{scratch_table}: {cleanup_exc}"
                )
            raise
        materialization_metadata["temporary_tables"].append(scratch_table)
        return result

    result_plan.count()
    return result_plan


def release_resources(spark, dataframes, metadata: dict) -> None:
    """Release persisted frames and drop every tracked scratch Delta table."""
    materialization = metadata.get("materialization", {})
    if materialization.get("mode") == "persist":
        for dataframe in reversed(list(dataframes)):
            try:
                dataframe.unpersist()
            except Exception as exc:
                print(f"WARNING: failed to unpersist a materialized frame: {exc}")

    for scratch_table in reversed(materialization.get("temporary_tables", [])):
        try:
            spark.sql(f"DROP TABLE IF EXISTS {quote_table_name(scratch_table)}")
        except Exception as exc:
            print(
                f"WARNING: failed to drop materialization table {scratch_table}: {exc}"
            )


def require_columns(frame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def require_nonempty(frame, label: str) -> int:
    count = frame.count()
    if count == 0:
        raise ValueError(f"{label} is empty")
    return count


def write_delta_table(frame, destination: str, mode: str = "overwrite") -> None:
    writer = frame.write.format("delta").mode(mode)
    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(destination)


def append_audit(spark, audit_table: str, task: str, outcome: str, metrics: dict) -> None:
    record = {
        "recorded_at_utc": datetime.now(timezone.utc),
        "task": task,
        "outcome": outcome,
        "metrics_json": json.dumps(metrics, sort_keys=True, default=str),
    }
    spark.createDataFrame([record]).write.format("delta").mode("append").saveAsTable(
        audit_table
    )
