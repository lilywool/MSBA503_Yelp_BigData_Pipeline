"""Classic-Spark, S3, and Glue helpers for the EMR task entry points."""

from __future__ import annotations

import re

from run_logger import log_run


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {label} identifier: {value!r}")
    return value


def table_name(database: str, name: str) -> str:
    return f"{validate_identifier(database, 'database')}.{validate_identifier(name, 'table')}"


def ensure_database(spark, database: str) -> None:
    validate_identifier(database, "database")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS `{database}`")


def join_uri(root: str, *parts: str) -> str:
    value = root.rstrip("/\\")
    for part in parts:
        clean_part = part.strip("/\\")
        value = f"{value}/{clean_part}"
    return value


def quote_table(value: str) -> str:
    parts = value.split(".")
    if len(parts) != 2:
        raise ValueError(f"Expected DATABASE.TABLE, got {value!r}")
    return ".".join(f"`{validate_identifier(part, 'table')}`" for part in parts)


def write_parquet_table(frame, path: str, destination: str, mode: str = "overwrite") -> None:
    """Write S3 Parquet and register its stable location in the Glue metastore."""
    if mode not in {"overwrite", "errorifexists"}:
        raise ValueError(f"Unsupported write mode: {mode!r}")
    quoted = quote_table(destination)
    if mode == "errorifexists" and frame.sparkSession.catalog.tableExists(destination):
        raise ValueError(f"Destination table already exists: {destination}")
    if mode == "overwrite":
        frame.sparkSession.sql(f"DROP TABLE IF EXISTS {quoted}")
    (
        frame.write
        .format("parquet")
        .mode(mode)
        .option("path", path)
        .saveAsTable(destination)
    )


def audit(spark, log_path: str, task: str, outcome: str, metrics: dict) -> None:
    log_run(
        {"task": task, "outcome": outcome, **metrics},
        log_path=log_path,
        spark=spark,
    )


def release_frames(frames) -> None:
    for frame in reversed(list(frames)):
        try:
            frame.unpersist()
        except Exception as exc:
            print(f"WARNING: failed to unpersist a frame: {exc}")
