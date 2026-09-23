# AWS EMR deployment

AWS was the original distributed platform for this MSBA 503 project. The
coursework used S3, Glue, Athena, and EMR/PySpark before separate transformer
experiments in SageMaker. That history is preserved in
`archival/coursework_aws/README.md`.

This directory is the maintained 2026 AWS implementation. It mirrors the
logical Databricks pipeline without pretending the platforms have identical
runtimes. Amazon EMR can run Spark through EMR Serverless or through
EC2-backed clusters. This implementation deliberately uses a transient EMR on
EC2 job cluster: it preserves the distributed-cluster methodology of the
original MSBA 503 project while extending that work into a complete
Bronze-to-Silver, Silver-to-Gold, and dashboard-serving medallion pipeline. The
cluster terminates after the ordered steps succeed or on the first failure.

```text
S3 raw JSON
  -> S3 Bronze-source Parquet
  -> EMR Bronze-to-Silver
  -> EMR Silver-to-Gold
  -> EMR dashboard-serving tables
  -> Glue Data Catalog / Athena
```

Databricks uses Unity Catalog, Delta, Spark Connect, and temporary Delta
materialization. EMR uses classic Spark on YARN, S3 Parquet, the Glue metastore,
executor broadcasts, and `persist()`/`unpersist()`. Both routes use the same
canonical feature code and the same Gold/dashboard transformation contracts.

## Maintained tasks

The transient cluster runs four ordered steps:

1. `stage-yelp-bronze-parquet` converts all five Yelp JSON-lines files with
   explicit schemas. Completed outputs are idempotently skipped.
2. `bronze-to-silver-yelp` scopes all five Bronze datasets, joins reviews with
   business and user dimensions, applies the selected NLP families to every
   selected review, validates the result, and publishes S3/Glue Silver.
3. `silver-to-gold-yelp` applies the requested analytical dependencies and
   publishes one validated Gold variant.
4. `data-science-dashboard-yelp` publishes the row-level sample, business
   summary, and monthly summary used by the dashboard layer.

Check-in and tip are retained in Bronze and scoped by business when a Silver
sample is requested. They are not joined into every Silver review because they
do not share a review-level key.

## Deployment assets

- `emr_cluster_config.json`: transient EMR 7.13 cluster and ordered steps.
- `scripts/bronze_to_silver.py`: EMR-native S3/Glue Bronze-to-Silver task.
- `scripts/silver_to_gold.py`: EMR-native parameterized Gold task.
- `scripts/data_science_dashboard.py`: EMR-native serving-table task.
- `scripts/emr_common.py`: validated S3, Glue-table, audit, and cleanup helpers.
- `bootstrap.sh`: pinned Python NLP environment for the EMR workers.
- `build_pipeline_package.py`: deterministic executor dependency archive.
- `launch_cluster.py`: validates placeholders and calls the EMR `RunJobFlow`
  API with the checked-in template.

## Prepare S3

The template expects this layout after replacing the bucket placeholder:

```text
s3://BUCKET/yelp/
|-- raw/
|   |-- yelp_academic_dataset_business.json
|   |-- yelp_academic_dataset_checkin.json
|   |-- yelp_academic_dataset_review.json
|   |-- yelp_academic_dataset_tip.json
|   `-- yelp_academic_dataset_user.json
|-- lexicons/
|   `-- [the five licensed .txt files]
|-- bootstrap/bootstrap.sh
|-- pipeline/
|   |-- bronze_json_to_parquet.py
|   `-- pipeline_modules.zip
`-- tasks/
    |-- bronze_to_silver.py
    |-- silver_to_gold.py
    `-- data_science_dashboard.py
```

Build the deterministic dependency archive from the repository root:

```bash
python aws_emr/build_pipeline_package.py
```

Upload:

- `pipeline/bronze_json_to_parquet.py`
- `aws_emr/dist/pipeline_modules.zip`
- `aws_emr/bootstrap.sh`
- the three files under `aws_emr/scripts/` listed as tasks above
- the five licensed lexicons
- the five Yelp JSON-lines files

Do not upload raw data, generated Parquet, lexicons, credentials, or the built
ZIP into Git.

## Glue and IAM

The cluster configuration points Spark at AWS Glue as its Hive metastore. The
EMR EC2 instance profile therefore needs permission to:

- read the raw, lexicon, bootstrap, package, and task prefixes;
- read/write/delete the Bronze-source and lakehouse prefixes;
- write EMR logs and append audit JSON;
- create, update, read, and delete the project tables in Glue.

Use project-scoped roles in production. The placeholders are intentional; the
repository does not contain account IDs, subnet IDs, bucket names, or role ARNs.

## Parameters

The EMR step arguments deliberately resemble the Databricks job parameters,
but storage and runtime controls are AWS-native:

- Bronze-to-Silver uses `--input-root`, `--output-root`, `--audit-log`,
  `--database`, and S3 `--lexicons-dir` instead of a Unity Catalog Volume.
- `--silver-sample-size` independently bounds the population that reaches
  Silver. Omit it for the full corpus.
- Repeat `--nlp-component` to select Silver feature families.
- Silver-to-Gold supports the same business, user, industry, state,
  city-of-state, date, sentiment, emotion, star, brand, and sample dependencies
  as Databricks.
- If the requested Gold sample exceeds eligible Silver rows, it converges to
  the available population and records both counts.

The checked-in template demonstrates a Chipotle and Great Clips row-level Gold
variant. Replace those arguments with another supported Gold configuration as
needed.

## Transformers

The default EMR template intentionally excludes transformers, matching the
standard Databricks job. Adding `--nlp-component transformers` requires a
separately validated GPU runtime with CUDA-enabled PyTorch and Transformers on
every worker. Do not add the flag to the CPU template and assume that selecting
GPU EC2 instances alone makes the existing environment GPU-ready.

The historical coursework transformer work remains correctly described as a
separate SageMaker stage. A production GPU overlay should be benchmarked on a
representative canary before it is accepted for a one-hour full-corpus target.

## Launch safely

1. Replace every `REPLACE_WITH_...` value in `emr_cluster_config.json`.
2. Build and upload `pipeline_modules.zip` after every code change.
3. Upload the current task scripts and bootstrap action.
4. Confirm the subnet, roles, instance types, EMR 7.13 availability, and quotas
   in the target Region.
5. Add `--silver-sample-size 250` and reduce `--partitions` for the first
   canary.
6. Inspect the S3 audit records, Glue tables, row counts, and validation output.
7. Remove the Silver sample limit only after the canary passes.

Launch from AWS CloudShell, where the AWS SDK and temporary credentials are
already scoped to the signed-in account:

```bash
python aws_emr/launch_cluster.py --region YOUR_REGION
```

For a local launch, install `requirements-deploy.txt` only in a dedicated
project-local deployment environment; do not add boto3 to the NLP runtime or a
global Python installation.

`KeepJobFlowAliveWhenNoSteps` remains `false`, so the cluster terminates after
success or on the first failed step instead of becoming long-lived interactive
infrastructure.
