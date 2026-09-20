# AWS EMR deployment

AWS was the original distributed platform for this MSBA 503 project. The
coursework workflow used S3 for storage, Glue and Athena for catalog/query work,
and EMR/PySpark for distributed processing before separate transformer
experiments in SageMaker. The edited historical summary is in
`archival/coursework_aws/README.md`.

This folder keeps that pathway runnable against the same canonical feature code
used by Databricks. It is an alternative deployment, not a second feature
implementation.

## Deployment assets

- `emr_cluster_config.json`: transient EMR job-cluster template with Bronze and
  validated Gold steps.
- `bootstrap.sh`: installs the pinned Python NLP stack on every node.
- `build_pipeline_package.py`: creates a deterministic `pipeline_modules.zip`
  for Spark executors.

## Prepare the S3 assets

Build the module archive from the repository root:

```bash
python aws_emr/build_pipeline_package.py
```

Upload the following to the bucket paths referenced by the cluster template:

- `pipeline/bronze_json_to_parquet.py`
- `pipeline/spark_feature_engineering.py`
- `aws_emr/dist/pipeline_modules.zip`
- `aws_emr/bootstrap.sh`
- the five licensed lexicon files listed in `lexicons/README.md`
- Yelp review, business, and user JSON files under the configured raw prefix

The current EMR route materializes the three entities required by the feature
join. The Databricks Bronze task additionally preserves check-in and tip data.

## Run safely

1. Replace every `REPLACE_WITH_...` value in `emr_cluster_config.json`.
2. Confirm the selected EMR release and instance types are available in the
   target Region.
3. Run a small canary first and inspect the Spark-native validation and audit
   output.
4. Keep `KeepJobFlowAliveWhenNoSteps` set to `false` so the cluster terminates
   after its steps.
5. Scale partitions or worker count only after measuring the canary.

The template deliberately omits Spark NLP, transformer dependencies, and
long-lived interactive infrastructure.
