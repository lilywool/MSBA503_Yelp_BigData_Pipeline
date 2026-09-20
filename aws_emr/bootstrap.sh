#!/bin/bash
# EMR bootstrap action for the distributed pipeline.
# Runs ONCE per node when the cluster launches (EMR's equivalent of a
# Databricks cluster-scoped init script) - not per job, not per step.
#
# Upload to S3 (e.g. s3://YOUR_BUCKET/bootstrap/bootstrap.sh) and reference
# it in emr_cluster_config.json's BootstrapActions.
#
# mapInPandas is plain Apache Spark (3.0+) - it is NOT a Databricks-only API.
# This script installs the exact same pinned Python libraries as the
# Databricks path (databricks_integration/requirements.txt) so the SAME
# pipeline/spark_feature_engineering.py runs unchanged on EMR.
#
# What this deliberately does NOT do: install spark-nlp, download any JAR,
# add any Maven/Scala coordinate, or touch Spark's classpath at all.

set -euo pipefail

echo "[bootstrap] Installing pinned Python libraries..."
sudo /usr/bin/python3.11 -m pip install --quiet \
    "pandas==2.2.3" \
    "numpy==1.26.4" \
    "pyarrow==12.0.1" \
    "vaderSentiment==3.3.2" \
    "nrclex==4.1.0" \
    "spacy==3.8.16" \
    "textblob==0.20.1" \
    "nltk==3.10.3"

echo "[bootstrap] Installing pinned spaCy English model..."
sudo /usr/bin/python3.11 -m pip install --quiet \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

echo "[bootstrap] Downloading NLTK corpora TextBlob's POS tagger needs..."
sudo /usr/bin/python3.11 -c "
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
"

echo "[bootstrap] Done. No Spark NLP, no JARs, no JVM-side NLP installed on this cluster."
