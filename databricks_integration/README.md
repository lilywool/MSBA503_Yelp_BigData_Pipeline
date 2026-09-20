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

Place these five JSON-lines files in one Google Drive folder:

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
schema for each source; the raw files are not copied into the input Volume.

Upload the five licensed lexicon files documented in `lexicons/README.md` to:

```text
/Volumes/workspace/default/yelp_academic_raw/_pipeline/lexicons
```

Upload `databricks_integration/init_script.sh` to:

```text
/Volumes/workspace/default/yelp_academic_raw/_pipeline/init_script.sh
```

The init script installs the same pinned Python feature stack on every cluster
node and downloads the two NLTK resources used by TextBlob. It does not install
Spark NLP, PyTorch, or Hugging Face Transformers.

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
- `--lexicons-dir` or the individual lexicon overrides identify the licensed
  feature resources.

The default init script installs the standard spaCy, VADER, NRC, TextBlob,
NLTK, pandas, NumPy, and Arrow stack. Transformer selection is explicit and
requires attaching `requirements-transformers.txt` to this task's cluster (or
using an equivalent compatible ML runtime). No external AI/API integration is
part of this pipeline.

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
2. Create the Google Drive connection and upload only the small lexicon and
   initialization resources to the documented Volume paths.
3. In **Workflows > Jobs**, create a job that uses this GitHub repository and
   branch `main` as its Git source.
4. Add the three Python script tasks in the order shown above. Their relative
   paths and parameters are in `job_config.json`; Git paths do not begin with
   `/` or `./`.
5. Use a classic multi-node job cluster, select an available memory-oriented
   node type, and attach the Volume-backed init script. The current feature
   distributor uses the classic Spark context for Python files, broadcasts, and
   SparkFiles, so this configuration is intentionally not serverless.
6. Set the job parameters before each run. `silver_sample_size` controls the
   bounded Bronze/Silver population; `gold_sample_size` independently controls
   the row-level Gold/dashboard population. Start with 25,000 and 5,000,
   respectively, and inspect table sizes plus the audit before increasing them.

The JSON file is a Jobs API-style template. Replace the Google Drive folder URL
and node type placeholders before importing or submitting it. The scripts also
retain `--input-volume` as an alternative source for classic/full-workspace
deployments.

Databricks documentation used for this layout:

- [Use Git with Lakeflow Jobs](https://docs.databricks.com/aws/en/jobs/git)
- [Python script task for jobs](https://docs.databricks.com/aws/en/jobs/tasks/python-script)
- [Unity Catalog Volume paths](https://docs.databricks.com/aws/en/volumes/volume-files)
