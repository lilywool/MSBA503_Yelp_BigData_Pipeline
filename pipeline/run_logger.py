"""
Append-only audit trail for every pipeline run, pass or fail.

On Databricks with a Delta-capable workspace, this writes to a Delta table so
the log is queryable, versioned, and survives independently of any one cluster.
Locally (or before a Databricks workspace exists), it falls back to a plain
append-only JSON format (pipeline_run_log.json), so the exact same call works
in both places without an if/else at the call site.

Nothing about this file is allowed to make a failed run disappear: log_run()
is meant to be called with the run's outcome AFTER validate()/validate_spark()
either returns metrics or raises - callers should catch the AssertionError,
still log the failed attempt (with validation.passed = False), then re-raise.
"""

import json
from pathlib import Path


def log_run(record: dict, log_path: str = "pipeline_run_log.json",
            spark=None, delta_table_path: str | None = None) -> None:
    if spark is not None and delta_table_path:
        try:
            sdf = spark.createDataFrame([_flatten(record)])
            sdf.write.format("delta").mode("append").save(delta_table_path)
            print(f"Run logged to Delta table: {delta_table_path}")
            return
        except Exception as e:  # noqa: BLE001 - deliberately broad: fall back, don't crash the run over logging
            print(f"WARNING: Delta logging failed ({e}); falling back to local JSON log.")

    # On EMR (or any plain Spark cluster), append one JSON record through Spark
    # when the audit destination is remote. pathlib/open cannot write s3:// URIs.
    if spark is not None and "://" in log_path:
        sdf = spark.createDataFrame([_flatten(record)])
        sdf.write.mode("append").json(log_path)
        print(f"Run logged to Spark JSON path: {log_path}")
        return

    path = Path(log_path)
    history = json.loads(path.read_text()) if path.exists() else []
    if isinstance(history, dict):  # migrate old dict-format log if one exists
        history = list(history.values())
    history.append(record)
    path.write_text(json.dumps(history, indent=2, default=str))
    print(f"Run logged: {log_path} ({len(history)} total entries)")


def _flatten(record: dict) -> dict:
    """Delta/Spark needs a flat, typed schema - stringify nested validation dicts
    rather than silently dropping fields a nested struct schema might choke on."""
    out = {}
    for k, v in record.items():
        out[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    return out
