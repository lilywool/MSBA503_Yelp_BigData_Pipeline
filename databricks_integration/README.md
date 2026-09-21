# Databricks integration

This folder contains the 2026 Databricks implementation of the Yelp pipeline.
It follows one Lakeflow Job with three ordered Python tasks:

```text
bronze_to_silver_yelp -> silver_to_gold_yelp -> data_science_dashboard_yelp
```

The job reads the Yelp Academic Dataset directly from an authenticated Google
Drive connection, writes bounded governed Delta tables in `workspace.default`,
validates the feature layer before publication, and finishes with three
dashboard-serving tables.

## Inputs already selected for this deployment

Use this Google Drive layout:

```text
Yelp RAW Databricks/
|-- yelp_academic_dataset_business.json
|-- yelp_academic_dataset_checkin.json
|-- yelp_academic_dataset_review.json
|-- yelp_academic_dataset_tip.json
|-- yelp_academic_dataset_user.json
`-- Lexicons/
    |-- NRC-Emotion-Intensity-Lexicon-v1.txt
    |-- NRC-VAD-Lexicon-v2.1.txt
    |-- NRC-WCST-Lexicon-v1.0.txt
    |-- worrywords-v1.txt
    `-- Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt
```

The five JSON-lines files at the root are:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_checkin.json
yelp_academic_dataset_review.json
yelp_academic_dataset_tip.json
yelp_academic_dataset_user.json
```

Create a Unity Catalog Google Drive connection and supply the folder URL and
connection name through the job parameters `google_drive_folder_url` and
`google_drive_connection`. The task uses a filename filter and an explicit
schema for each source. It also recursively discovers the five exact lexicon
filenames in the `Lexicons` subfolder, stages their bytes only for the duration
of the run, and distributes those small payloads to serverless Spark workers.
Neither the raw dataset nor
the licensed lexicons are copied into Git or an input Volume.

If the lexicons are ever moved to a different Drive folder, pass that folder's
URL with `--google-drive-lexicons-folder-url`. The supplied Job template needs
only the parent `Yelp RAW Databricks` URL.

The maintained Job uses Databricks Free Edition serverless compute. Its shared
Python environment is declared in `job_config.json` with environment version 5
and these two dependencies from the connected Git folder:

```text
-r /Workspace/Users/lwool@sandiego.edu/MSBA503_Yelp_BigData_Pipeline/databricks_integration/requirements-serverless.txt
/Workspace/Users/lwool@sandiego.edu/MSBA503_Yelp_BigData_Pipeline
```

The first installs the NLP libraries and spaCy model. The second installs this
repository as the shared pipeline package. Databricks supplies Spark, pandas,
NumPy, and PyArrow; the dependency file deliberately does not replace those
runtime-coupled packages. Grammar tags reuse the loaded spaCy document, so the
serverless tasks require neither an init script nor downloaded NLTK corpora.
`init_script.sh` remains only for classic/full-workspace deployments and is not
referenced by the maintained Free Edition Job.

## Task contract

### 1. `bronze_to_silver_yelp`

Reads the review JSON with its explicit schema and selects a deterministic,
bounded review set before any join or NLP work. When a Silver sample is set,
the business, user, check-in, and tip inputs are reduced to rows related to that
review set before five Bronze Delta tables are published. It then joins the
selected reviews, businesses, and users and applies the selected feature
families across every selected review before publishing
`yelp_silver_reviews`. Validation is mandatory before that write.

This task owns all feature-engine and scaling choices:

- `--silver-sample-size` and `--silver-sample-seed` control how many raw
  reviews are retained in Bronze, receive NLP, and reach Silver. The default
  job value is 25,000 reviews; it can be overridden for each run. Omitting the
  size processes the full review corpus. The source must still be scanned to
  select a partition-independent hash sample, but only the selected relational
  subset is persisted in Databricks.
- Repeat `--nlp-component` to choose any combination of `linguistic`, `vader`,
  `nrc-emotion`, `vad`, `entities`, `grammar`, `domain-lexicons`,
  `time-weighting`, and `transformers`. Omitting the option selects every
  standard family except `transformers`.
- `--partitions` and `--arrow-batch-size` control Spark distribution and the
  bounded pandas batch size.
- With a Google Drive source, the five licensed lexicons are discovered in the
  same folder tree automatically. `--google-drive-lexicons-folder-url` can
  point to a different Drive folder. `--lexicons-dir` and the five individual
  overrides remain available only for alternate non-Drive deployments.

The serverless environment installs the standard spaCy, VADER, NRC, TextBlob,
and NLTK feature libraries while retaining the runtime's own pandas, NumPy,
Arrow, and PySpark packages. Transformer selection is explicit and requires a
separately tested compatible serverless environment; it is not enabled in the
supplied Free Edition Job template.

### 2. `silver_to_gold_yelp`

Reads the already featured Silver table. It never recomputes NLP. Instead it
creates one validated Gold variant at the requested grain: `business`, `user`,
`date`, `industry`, `sentiment`, `emotion`, a deterministic `sample`, or a
multi-brand sample.

Dependencies can be combined before the Gold output is built. Repeat
`--business-id`, `--business-name`, `--user-id`, `--industry`, `--city`,
`--state`, `--sentiment-label`, or `--emotion-label` as many times as needed;
values within each repeated option are OR selections and different option
families are combined. Date bounds, star ranges, VADER-score ranges, emotion
source, sentiment source, date granularity, `--gold-sample-size`, and
`--gold-sample-seed` are also configurable. The Gold size is independent of
the Silver size and is applied only after every Gold dependency/filter. For
`brand-sample`, it is a per-brand ceiling. If the requested Gold size exceeds
the eligible Silver population, the effective size automatically converges to
that smaller population; both requested and effective values are recorded in
the audit metrics. Sentiment can be sourced from VADER, the optional
transformer score, Yelp AFFLEX/NEGLEX, or NRC valence when that family exists in
Silver.
For example, any number of industries can be selected with repeated flags:

```text
--industry Restaurants --industry "Hair Salons" --industry Bakeries
```

### 3. `data_science_dashboard_yelp`

Reads exactly one validated row-level Gold `sample` or `brand-sample` table and
publishes:

- `workspace.default.yelp_dashboard_review_sample`
- `workspace.default.yelp_dashboard_business_summary`
- `workspace.default.yelp_dashboard_monthly_summary`

This task does not sample or choose businesses. It inherits the selected Gold
variant and its provenance columns. The business table carries review KPIs,
sentiment/emotion aggregates, a bounded risk score, an attention tier, and a
suggested focus. The monthly table supports trend and industry views. These are
serving tables: the batch task prepares dashboard data but does not keep a web
application running.

Every task appends its outcome and metrics to
`workspace.default.yelp_pipeline_audit`. Failures are recorded and re-raised.

The supplied job template uses a 5,000-row-per-brand Chipotle/Great Clips Gold
variant drawn from a 25,000-review Silver canary as a reproducible example of
the original coursework comparison. Those
physical coursework sample files are archival; changing the Gold parameters is
all that is required to drive the same dashboard from different businesses,
industries, users, dates, sentiment classes, or emotions.

## Create the job from the connected repository

### Parameter formats

Databricks configuration and shell commands express the same controls in
different formats. `job_config.json` is authoritative for the Databricks Job.
Job-level defaults are JSON objects, and Python-script task arguments are a
JSON array of strings with dynamic value references:

```json
{
  "parameters": [
    {"name": "silver_sample_size", "default": "25000"},
    {"name": "gold_sample_size", "default": "5000"}
  ],
  "tasks": [
    {
      "task_key": "bronze_to_silver_yelp",
      "spark_python_task": {
        "parameters": [
          "--silver-sample-size",
          "{{job.parameters.silver_sample_size}}"
        ]
      }
    }
  ]
}
```

The equivalent CLI syntax is only for direct module/script execution:

```text
python bronze_to_silver.py --silver-sample-size 25000
python silver_to_gold.py --gold-level sample --gold-sample-size 5000
```

Do not paste the CLI command into a Databricks JSON parameter field. In the
Jobs UI, edit the job parameter key/value pairs or use **Run now with different
parameters**; the task JSON resolves them into script arguments.

1. Push the public repository and connect it to the Databricks workspace.
2. Create the Google Drive connection. Confirm the five JSON files are at the
   `Yelp RAW Databricks` root and the five lexicons are in its `Lexicons`
   subfolder under the exact filenames shown above.
3. In **Workflows > Jobs**, create a job that uses this GitHub repository and
   branch `main` as its Git source.
4. Add the three Python script tasks in the order shown above. Their relative
   paths and parameters are in `job_config.json`; Git paths do not begin with
   `/` or `./`.
5. Choose **Serverless** compute. Assign all three tasks the shared environment
   key `yelp_pipeline`, Standard environment version `5`, and the two dependencies
   shown above. Do not add a classic cluster, node type, or init script.
6. Set the job parameters before each run. `silver_sample_size` controls the
   bounded Bronze/Silver population; `gold_sample_size` independently controls
   the row-level Gold/dashboard population. Start with 25,000 and 5,000,
   respectively, and inspect table sizes plus the audit before increasing them.

The JSON file is a Jobs API-style template. Replace the Google Drive folder URL
placeholder before importing or submitting it. The maintained
Job uses Google Drive for both raw inputs and lexicons. The script retains
`--input-volume` only as an explicit alternative for other full-workspace
deployments; it never falls back to a Volume implicitly.

Databricks documentation used for this layout:

- [Use Git with Lakeflow Jobs](https://docs.databricks.com/aws/en/jobs/git)
- [Python script task for jobs](https://docs.databricks.com/aws/en/jobs/tasks/python-script)
- [Ingest files from Google Drive](https://docs.databricks.com/gcp/en/ingestion/google-drive)
- [Serverless compute limitations](https://docs.databricks.com/aws/en/compute/serverless/limitations)
- [Serverless environment dependencies](https://docs.databricks.com/aws/en/compute/serverless/dependencies)
