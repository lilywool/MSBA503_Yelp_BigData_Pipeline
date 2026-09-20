"""Small shared helpers for the three Databricks job tasks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def table_name(catalog: str, schema: str, name: str) -> str:
    parts = (catalog, schema, name)
    invalid = [part for part in parts if not IDENTIFIER.fullmatch(part)]
    if invalid:
        raise ValueError(
            f"Unity Catalog identifiers may contain only letters, numbers, and underscores: {invalid}"
        )
    return ".".join(parts)


def ensure_schema(spark, catalog: str, schema: str) -> None:
    table_name(catalog, schema, "validation_only")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")


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
