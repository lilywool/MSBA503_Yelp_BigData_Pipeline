# 2026 implementation notes

## Scope

The 2026 revision turns the coursework-era feature work into a reproducible
distributed pipeline without reading or publishing the repository's local data
directory. It keeps one canonical pandas feature implementation and distributes
bounded batches through Spark `mapInPandas`.

## Implemented

- Explicit schemas for all five Yelp Academic Dataset JSON files.
- A canonical review/business/user join plus selectable full-corpus feature
  families for the Silver review table.
- Spark-native Parquet, CSV, and Delta writes instead of driver-side collection.
- Mandatory full-scale validation before featured Silver output is published.
- Failure-path audit logging and queryable Delta audit records.
- Executor-local lexicon staging for local, Unity Catalog Volume, and S3 inputs.
- Strict local-versus-Spark parity comparison for schema, keys, null positions,
  row counts, strings, and numeric values.
- A no-skips local release gate with pinned Python, Spark, Arrow, and NLP versions.
- Deterministic EMR module packaging.
- A three-task Databricks workflow:
  `bronze_to_silver -> silver_to_gold -> data_science_dashboard`.
- Yelp-native Gold variants over businesses, users, dates, industries,
  sentiment, emotion, and deterministic samples.
- A deployable Streamlit presentation layer for Databricks Apps.

## Verification record

The expanded strict suite passed in the pinned WSL2 environment on 2026-09-20:
15 tests executed, 0 failures, 0 errors, and 0 skips. The corrected end-to-end
parity comparison had previously passed on two disjoint 1,500-row slices across
all 68 expected feature columns.

Those results verify the shared local/Spark feature path. They do not substitute
for a Databricks or EMR canary. The public cloud configurations remain templates
until their audit table records a successful live run.

## Before a full cloud run

1. Re-run `scripts/bootstrap_local.sh` from a clean Linux or WSL2 checkout.
2. Run a small cluster canary with the same runtime, lexicons, and task code.
3. Inspect row counts, validation metrics, executor memory, and throughput.
4. Confirm dashboard-serving tables and the app's table permissions.
5. Increase workers only after the canary passes.
