# Databricks integration

This folder contains the 2026 Databricks implementation of the Yelp pipeline.
It follows one Lakeflow Job with three ordered Python tasks:

```text
bronze_to_silver -> silver_to_gold -> data_science_dashboard
```

The job reads the Yelp Academic Dataset from a Unity Catalog Volume, writes
governed Delta tables in `workspace.default`, validates the feature layer before
publication, and finishes with three dashboard-serving tables.

## Inputs already selected for this deployment

The default input is:

```text
/Volumes/workspace/default/yelp_academic_raw
```

The first task expects these five files at the top of that Volume:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_checkin.json
yelp_academic_dataset_review.json
yelp_academic_dataset_tip.json
yelp_academic_dataset_user.json
```

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

### 1. `bronze_to_silver`

Reads all five JSON-lines files with explicit schemas. It rejects empty inputs,
null entity keys, or duplicate review/business/user keys before publishing five
Bronze Delta tables. It then joins reviews, businesses, and users and applies
the selected feature families across every selected review before publishing
`yelp_silver_reviews`. Validation is mandatory before that write.

This task owns all feature-engine and scaling choices:

- `--sample-size` and `--sample-seed` select a stable review subset for a cheap
  canary. Omitting `--sample-size` processes the full review corpus.
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

### 2. `silver_to_gold`

Reads the already featured Silver table. It never recomputes NLP. Instead it
creates one validated Gold variant at the requested grain: `business`, `user`,
`date`, `industry`, `sentiment`, `emotion`, a deterministic `sample`, or a
multi-brand sample.

Dependencies can be combined before the Gold output is built. Repeat
`--business-id`, `--business-name`, `--user-id`, `--industry`, `--city`,
`--state`, `--sentiment-label`, or `--emotion-label` as many times as needed;
values within each repeated option are OR selections and different option
families are combined. Date bounds, star ranges, VADER-score ranges, emotion
source, sentiment source, date granularity, sample size, and sample seed are
also configurable. Sentiment can be sourced from VADER, the optional
transformer score, Yelp AFFLEX/NEGLEX, or NRC valence when that family exists in
Silver.
For example, any number of industries can be selected with repeated flags:

```text
--industry Restaurants --industry "Hair Salons" --industry Bakeries
```

### 3. `data_science_dashboard`

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

The supplied job template uses a 15,000-row-per-brand Chipotle/Great Clips Gold
variant as a reproducible example of the original coursework comparison. Those
physical coursework sample files are archival; changing the Gold parameters is
all that is required to drive the same dashboard from different businesses,
industries, users, dates, sentiment classes, or emotions.

## Create the job from the connected repository

1. Push the public repository and connect it to the Databricks workspace.
2. Upload the lexicons and init script to the paths above.
3. In **Workflows > Jobs**, create a job that uses this GitHub repository and
   branch `main` as its Git source.
4. Add the three Python script tasks in the order shown above. Their relative
   paths and parameters are in `job_config.json`; Git paths do not begin with
   `/` or `./`.
5. Use a classic multi-node job cluster, select an available memory-oriented
   node type, and attach the Volume-backed init script. The current feature
   distributor uses the classic Spark context for Python files, broadcasts, and
   SparkFiles, so this configuration is intentionally not serverless.
6. Start with the 2-4 worker canary in the template. Inspect the audit and Delta
   tables before increasing the worker range for the full run.

The JSON file is a Jobs API-style template. Replace the node type placeholder
before importing or submitting it. The task code itself defaults to the Volume
and Unity Catalog names above, so the same scripts can also be configured in the
Databricks UI without the JSON template.

Databricks documentation used for this layout:

- [Use Git with Lakeflow Jobs](https://docs.databricks.com/aws/en/jobs/git)
- [Python script task for jobs](https://docs.databricks.com/aws/en/jobs/tasks/python-script)
- [Unity Catalog Volume paths](https://docs.databricks.com/aws/en/volumes/volume-files)
