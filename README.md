# Yelp Review Intelligence: a distributed Big Data pipeline

A Spark pipeline that turns the Yelp Academic Dataset into validated
review-level NLP features and dashboard-ready business intelligence. The project
combines an AWS coursework implementation with a 2026 Databricks upgrade built
around Unity Catalog, Delta tables, Lakeflow Jobs, and a deployable dashboard app.

## Project history

### MSBA 503 coursework: AWS

The original course project used Amazon S3 as the data lake, AWS Glue and Athena
for cataloging and querying, and Amazon EMR with PySpark for distributed data
preparation and feature processing. Amazon SageMaker was then used as the
separate environment for transformer-oriented sentiment and emotion work.

Lily built the data-engineering and NLP pipeline. The team dashboard built on
that work: Alex developed the descriptive and predictive experience, and Eddie
developed the diagnostic and prescriptive modules. The original notebook is
preserved unchanged under `archival/coursework_dashboard/`, with detailed module
ownership documented there. The presentation for the MSBA 503 class is available
in [docs/MSBA 503 Yelp Analytics Dashboard.pdf](<docs/MSBA 503 Yelp Analytics Dashboard.pdf>).

### 2026 upgrade: Databricks

The primary 2026 implementation moves the production-shaped pathway to
Databricks. The supplied Lakeflow Job uses Databricks serverless compute and
reads locally converted Bronze Parquet plus licensed lexicons from the governed
Unity Catalog Volume `workspace.default.yelp_raw`. Serverless was selected for
the Databricks route because it removes cluster administration from the
three-task workflow while retaining Spark distribution, governed Delta tables,
and task-level dependency controls.

```text
Downloaded Yelp JSON files (outside Git)
        |
        v
local explicit-schema JSON -> Parquet conversion
        |
        v
/Volumes/workspace/default/yelp_raw/
  bronze/ + lexicons/
        |
        v
bronze_to_silver_yelp
  deterministic bounded review selection
  + relationally scoped explicit-schema Bronze Delta tables
  + canonical review/business/user join
  + selected NLP, lexicon, grammar, and time features on every selected review
        |
        v
silver_to_gold_yelp
  parameterized business, user, date, industry, NLP, or sampled variants
        |
        v
data_science_dashboard_yelp
  serving views derived directly from the selected row-level Gold variant
        |
        v
Databricks Streamlit App
  deployable interface over the dashboard-serving tables
```

### 2026 maintained alternative: AWS EMR

The AWS route is also a current implementation, not merely an archive of the
coursework environment. Amazon EMR supports both serverless applications and
transient EC2-backed clusters. This repository intentionally uses a transient
EMR on EC2 job cluster because that preserves the methodology of the original
MSBA 503 deployment while modernizing it into the same complete medallion-shaped
pipeline as the Databricks route. EMR provisions a driver and worker nodes for
the ordered workload, uses classic Spark/YARN materialization, and terminates
the cluster automatically after success or on the first failed step.

```text
Yelp JSON-lines files in S3
        |
        v
explicit-schema JSON -> Bronze-source Parquet
        |
        v
bronze_to_silver_yelp
  scoped Bronze tables + canonical review/business/user join
  + selected NLP, lexicon, grammar, and time features
        |
        v
silver_to_gold_yelp
  parameterized analytical variant
        |
        v
data_science_dashboard_yelp
  S3 Parquet serving tables registered in AWS Glue
        |
        v
Athena and downstream dashboard consumers
```

Databricks and EMR therefore share the feature and analytical contracts without
pretending their runtimes are identical: Databricks uses serverless Spark
Connect, Unity Catalog, Delta, and temporary managed Delta materialization; EMR
uses a transient EC2 cluster, YARN, S3 Parquet, Glue/Athena, executor broadcasts,
and `persist()`/`unpersist()`.

### Configure a run by purpose

Each run makes a small set of deliberate choices:

- **Source:** convert the five downloaded Yelp JSON-lines files to partitioned
  Parquet locally, upload the five dataset directories and licensed lexicons to
  `workspace.default.yelp_raw`, and keep both source data and generated files
  outside Git.
- **Workload:** choose how many reviews reach Silver and receive the selected
  feature engineering.
- **Features:** run the complete standard NLP stack or select only the
  linguistic, sentiment, emotion, affect, entity, grammar, domain-lexicon,
  time-weighting, or optional transformer families needed for the run.
- **Analytical scope:** shape Gold around brands, businesses, users, industries,
  time periods, geography, ratings, sentiment, emotions, or a deterministic
  review sample.
- **Presentation:** build the dashboard-serving layer directly from the chosen
  row-level Gold result.

### Stage the Databricks inputs

After converting the five JSON-lines files locally, open **Catalog Explorer >
workspace > default > yelp_raw > Files**. Create `bronze/` and `lexicons/` in
the Volume. Inside `bronze/`, create these five directories exactly:

```text
yelp_academic_dataset_business/
yelp_academic_dataset_checkin/
yelp_academic_dataset_review/
yelp_academic_dataset_tip/
yelp_academic_dataset_user/
```

Open each generated folder under the WSL output `~/yelp_parquet`, select its
`part-*.snappy.parquet` files, and drag/drop them into the matching Databricks
directory. Drag/drop the five licensed `.txt` files into `lexicons/`. Do not
flatten the five Parquet schemas into the Volume root and do not upload local
Hadoop `.crc` files. `_SUCCESS` is optional. The detailed conversion command,
expected file counts, and task paths are in the illustrated guide below.

Silver and Gold have independent limits. If a requested Gold sample is larger
than the eligible Silver population, the pipeline safely uses the available
rows and records both values. The illustrated
[Databricks pipeline parameter guide](docs/databricks_pipeline_parameter_guide.pdf)
explains the three tasks, recommended workflow, JSON formatting, tailored
examples, complete parameter reference, and verification boundary.

## Silver feature layer

The fully featured Silver table adds 68 engineered features while preserving
the joined review, business, and user columns.
One canonical pandas implementation is used locally and inside Spark executor
batches, preventing separate local and cloud feature definitions.

| Family | Examples | Implementation |
|---|---|---|
| Linguistic | word and sentence counts, average lengths, punctuation, capitalization, negation | direct counts + spaCy |
| Sentiment | VADER compound/positive/neutral/negative | VADER |
| Emotion | counts, intensity averages, dominant emotion | NRC resources |
| Affect | valence, arousal, dominance | NRC VAD |
| Entities | person, location, product mentions | spaCy NER |
| Grammar | noun/verb/adjective/adverb shares, subjectivity, type-token ratio | TextBlob/NLTK |
| Domain lexicons | WorryWords, WCST, Yelp AFFLEX/NEGLEX, emotion intensity | word-level lookups |
| Time weighting | `weighted_star` | dataset-anchored review-date decay |

Spark provides distribution rather than a second NLP implementation. Each
executor receives bounded Arrow/pandas batches; pandas never receives the full
6,990,280-review corpus at once. The standard Databricks feature set excludes transformers; transformer
inference is an explicit opt-in that requires its separate model runtime and
suitable compute. The coursework-era SageMaker work remains part of the
historical AWS story.

For a canary, Bronze-to-Silver can select a deterministic review subset before
the expensive feature pass. It can also opt into any combination of feature
families. The default remains the complete non-transformer feature set.

## Repository layout

```text
pipeline/                    canonical feature, ingestion, validation, and audit modules
databricks_integration/      three serverless job tasks, environment, outputs, and dashboard app
aws_emr/                     maintained AWS EMR deployment path
dashboard/                   local dashboard-data compatibility utility
archival/                    curated coursework context and unchanged team notebook
tests/                       synthetic regression and Spark integration tests
scripts/                     isolated local environment and strict no-skips test gate
docs/                        designed Databricks pipeline and verification guide
lexicons/                    download instructions only; licensed files are not tracked
```

Raw Yelp data, licensed lexicon contents, cloud credentials, generated Delta or
Parquet data, and virtual environments are intentionally excluded from Git.

## Databricks outputs

With the default configuration, the job creates these Unity Catalog tables in
`workspace.default`:

- Bronze: `yelp_bronze_business`, `yelp_bronze_checkin`,
  `yelp_bronze_review`, `yelp_bronze_tip`, `yelp_bronze_user`
- Silver: `yelp_silver_reviews`
- Gold example: `yelp_gold_dashboard_variant`; other Gold tables can be created
  by business, user, date, industry, sentiment, emotion, or deterministic sample
- Dashboard: `yelp_dashboard_review_sample`,
  `yelp_dashboard_business_summary`, `yelp_dashboard_monthly_summary`
- Audit: `yelp_pipeline_audit`

The job template uses Chipotle and Great Clips only as an example of a
reproducible brand-comparison Gold variant. The original coursework sample is
archival and is not a live pipeline input. The dashboard app is configured to
query only the selected Gold variant's bounded serving tables through a
Databricks SQL warehouse. When deployed, its service principal should receive
least-privilege access; no token or password is stored in this repository.

## AWS EMR pathway

`aws_emr/` contains the EMR-native scripts, deterministic executor package,
bootstrap action, validated `RunJobFlow` template, and guarded launcher for the
current AWS implementation described above. It processes all five Yelp source
datasets and publishes Bronze, Silver, Gold, and dashboard-serving outputs to
S3 with Glue registrations. See `aws_emr/README.md` for the S3 layout, IAM
boundary, configuration placeholders, canary sequence, and launch procedure.

## Local verification

The recommended local gate is Linux or WSL2 with Python 3.11 and Java 17:

```bash
git clone https://github.com/lilywool/MSBA503_Yelp_BigData_Pipeline.git ~/yelp_pipeline
cd ~/yelp_pipeline
./scripts/bootstrap_local.sh
```

The strict runner rejects missing pins, missing NLP resources, zero discovered
tests, and any skipped test. A genuine pass ends with:

```text
GENUINE PASS: N tests executed; 0 failures, 0 errors, 0 skips.
```

The current pinned WSL2 verification completed 27 tests with no failures, errors,
or skips on 2026-09-22. Corrected local-versus-Spark parity also passed on two
disjoint 1,500-row slices across all 68 expected feature columns. The exact
environment, commands, evidence boundary, implementation architecture, and
Databricks task parameters are consolidated in the
[Databricks pipeline parameter guide](docs/databricks_pipeline_parameter_guide.pdf).

## Current status

- [x] Canonical 68-feature implementation
- [x] Explicit schemas for all five Yelp JSON files
- [x] Local/Spark parity and strict synthetic release gate
- [x] AWS EMR deployment template and deterministic module package
- [x] Three-task Databricks job implementation
- [x] Delta audit and dashboard-serving contracts
- [x] Deployable Databricks Streamlit app
- [ ] Databricks serverless canary recorded in `yelp_pipeline_audit`
- [ ] Full 6,990,280-review Databricks run
- [ ] Dashboard app deployed and share URL recorded

The unchecked items require the configured Databricks workspace and incur cloud
compute. The repository does not present templates or local verification as a
completed cloud run.

## License

Project code and maintained documentation are released under the Apache License
2.0. The archived team notebook/presentation, Yelp dataset, and third-party
lexicons remain subject to their respective contributor or source terms and are
not relicensed or redistributed as pipeline dependencies here.
