# Databricks integration

This folder contains the 2026 Databricks implementation of the Yelp pipeline.
It follows one Lakeflow Job with three ordered Python tasks:

```text
bronze_to_silver_yelp -> silver_to_gold_yelp -> data_science_dashboard_yelp
```

The job reads locally converted Bronze Parquet and licensed lexicons from a
Unity Catalog Volume, writes bounded governed Delta tables in
`workspace.default`, validates the feature layer before publication, and
finishes with three dashboard-serving tables.

## Prepare the governed input Volume

The large JSON-lines download is converted locally before it enters
Databricks. This avoids committing data to Git, avoids fragile multi-gigabyte
remote-file streams, and gives Spark partitioned, compressed input that it can
read concurrently.

### 1. Download and extract the Yelp Academic Dataset

Keep the five source files outside this repository:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_checkin.json
yelp_academic_dataset_review.json
yelp_academic_dataset_tip.json
yelp_academic_dataset_user.json
```

### 2. Convert JSON to Parquet locally

Use Linux or WSL2 with the repository's pinned Python 3.11 and Java 17
environment. Keep the repository checkout and generated Parquet in the Linux
filesystem; `RAW_ROOT` may point to an extracted Windows folder through
`/mnt/c/...`.

```bash
cd ~/yelp_pipeline
./scripts/bootstrap_local.sh

RAW_ROOT="/mnt/c/path/to/extracted/yelp_json"
PARQUET_ROOT="$HOME/yelp_parquet"
mkdir -p "$PARQUET_ROOT" "$HOME/yelp_spark_tmp"

export PYSPARK_PYTHON="$PWD/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PWD/.venv/bin/python"

"$PWD/.venv/bin/spark-submit" \
  --master "local[4]" \
  --driver-memory 4g \
  --conf "spark.sql.shuffle.partitions=64" \
  --conf "spark.local.dir=$HOME/yelp_spark_tmp" \
  pipeline/bronze_json_to_parquet.py \
  --input-root "$RAW_ROOT" \
  --output-root "$PARQUET_ROOT" \
  --datasets business checkin tip user review \
  --review-partitions 64 \
  --business-partitions 4 \
  --user-partitions 32 \
  --checkin-partitions 4 \
  --tip-partitions 8
```

The converter uses explicit schemas, writes Snappy-compressed Parquet, and
places an `_SUCCESS` marker in each completed dataset directory. It is safe to
rerun: completed directories are skipped unless `--overwrite` is deliberately
supplied. A successful conversion produces:

```text
~/yelp_parquet/
|-- yelp_academic_dataset_business/
|-- yelp_academic_dataset_checkin/
|-- yelp_academic_dataset_review/
|-- yelp_academic_dataset_tip/
`-- yelp_academic_dataset_user/
```

### 3. Upload Parquet and lexicons to Unity Catalog

In **Catalog Explorer**, create the Volume `workspace.default.yelp_raw` and open
its **Files** tab. Select **Create directory** twice to create `bronze` and
`lexicons`. Open `bronze`, then use **Create directory** five more times to
create the exact dataset directories shown below. Do not upload the five
datasets as loose files at the Volume root and do not combine their schemas.

```text
/Volumes/workspace/default/yelp_raw/
|-- bronze/
|   |-- yelp_academic_dataset_business/   [part-*.snappy.parquet]
|   |-- yelp_academic_dataset_checkin/    [part-*.snappy.parquet]
|   |-- yelp_academic_dataset_review/     [part-*.snappy.parquet]
|   |-- yelp_academic_dataset_tip/        [part-*.snappy.parquet]
|   `-- yelp_academic_dataset_user/       [part-*.snappy.parquet]
`-- lexicons/
    |-- NRC-Emotion-Intensity-Lexicon-v1.txt
    |-- NRC-VAD-Lexicon-v2.1.txt
    |-- NRC-WCST-Lexicon-v1.0.txt
    |-- worrywords-v1.txt
    `-- Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt
```

In Windows File Explorer, open the completed WSL output at
`\\wsl.localhost\Ubuntu\home\<WSL_USER>\yelp_parquet`. Open one generated
dataset folder at a time, select its `part-*.snappy.parquet` files, and
drag/drop them into the matching directory in the Databricks browser. Keep the
upload page open until that dataset finishes, then continue with the next one.
The expected Parquet counts are:

| Directory | Parquet files |
|---|---:|
| `yelp_academic_dataset_business` | 4 |
| `yelp_academic_dataset_checkin` | 4 |
| `yelp_academic_dataset_review` | 64 |
| `yelp_academic_dataset_tip` | 8 |
| `yelp_academic_dataset_user` | 32 |

Open `lexicons` in Databricks, select the five licensed `.txt` files from their
local folder, and drag/drop them in the same manner. `_SUCCESS` is optional. Do
not upload `.crc` files because they are local Hadoop checksum sidecars. The
completed Volume contains 112 Parquet parts and five lexicons. Neither raw
JSON, generated Parquet, nor lexicon contents belongs in Git.

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
- `--input-volume` with `/Volumes/workspace/default/yelp_raw/bronze` selects the
  five locally staged Parquet directories. `--lexicons-dir` with
  `/Volumes/workspace/default/yelp_raw/lexicons` selects the licensed resources.
  The five individual lexicon overrides remain available when a deployment
  stores those files separately.

The serverless environment installs the standard spaCy, VADER, NRC, TextBlob,
and NLTK feature libraries while retaining the runtime's own pandas, NumPy,
Arrow, and PySpark packages. Transformer selection is explicit and requires a
separately tested compatible serverless environment; it is not enabled in the
supplied Free Edition Job template.

### 2. `silver_to_gold_yelp`

Reads the already featured Silver table. It never recomputes NLP. Instead it
creates one validated Gold variant at the requested grain: `business`, `user`,
`date`, `industry`, `state`, `city-of-state`, `sentiment`, `emotion`, a
deterministic `sample`, or a multi-brand sample. `state` produces one aggregate
row per state. `city-of-state` keeps state and city as a compound geographic
key and includes coordinate centroids when the Silver input has business
coordinates.

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
variant and its provenance columns. The business table carries coordinates,
review KPIs, sentiment/emotion aggregates, a bounded risk score, an attention
tier, and a suggested focus. The monthly table supports trend and industry
views. These are serving tables: the batch task prepares dashboard data but
does not keep a web application running.

Every task appends its outcome and metrics to
`workspace.default.yelp_pipeline_audit`. Failures are recorded and re-raised.

The supplied job template uses a 5,000-row-per-brand Chipotle/Great Clips Gold
variant drawn from a 25,000-review Silver canary as a reproducible example of
the original coursework comparison. Those
physical coursework sample files are archival; changing the Gold parameters is
all that is required to drive the same dashboard from different businesses,
industries, users, states, cities within states, dates, sentiment classes, or
emotions.

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
2. Convert the five downloaded JSON-lines files locally with
   `pipeline/bronze_json_to_parquet.py` and confirm all five output directories
   contain `_SUCCESS`.
3. Create `workspace.default.yelp_raw`, create `bronze/` and `lexicons/`, then
   create the five exact dataset directories inside `bronze/`. Drag/drop each
   WSL Parquet folder's part files into its matching directory and the five
   lexicons into `lexicons/`.
4. In **Workflows > Jobs**, create a job that uses this GitHub repository and
   branch `main` as its Git source.
5. Add the three Python script tasks in the order shown above. Their relative
   paths and parameters are in `job_config.json`; Git paths do not begin with
   `/` or `./`.
6. Choose **Serverless** compute. Assign all three tasks the shared environment
   key `yelp_pipeline`, Standard environment version `5`, and the two dependencies
   shown above. Do not add a classic cluster, node type, or init script.
7. Set the job parameters before each run. `silver_sample_size` controls the
   bounded Bronze/Silver population; `gold_sample_size` independently controls
   the row-level Gold/dashboard population. Start with 25,000 and 5,000,
   respectively, and inspect table sizes plus the audit before increasing them.

The JSON file is a Jobs API-style template. The maintained Job passes
`--input-volume /Volumes/workspace/default/yelp_raw/bronze` and
`--lexicons-dir /Volumes/workspace/default/yelp_raw/lexicons` explicitly; it
does not fall back to either location implicitly.

Databricks documentation used for this layout:

- [Use Git with Lakeflow Jobs](https://docs.databricks.com/aws/en/jobs/git)
- [Python script task for jobs](https://docs.databricks.com/aws/en/jobs/tasks/python-script)
- [Work with files in Unity Catalog Volumes](https://docs.databricks.com/aws/en/volumes/volume-files)
- [Serverless compute limitations](https://docs.databricks.com/aws/en/compute/serverless/limitations)
- [Serverless environment dependencies](https://docs.databricks.com/aws/en/compute/serverless/dependencies)
