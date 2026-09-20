#!/bin/bash
# Cluster-scoped setup for the Yelp Databricks job. This script runs on every
# node so Python packages, the spaCy model, and NLTK data are executor-local.

set -euo pipefail

PYTHON_BIN=/databricks/python/bin/python
NLTK_DIR=/local_disk0/nltk_data

mkdir -p "${NLTK_DIR}"
chmod 755 "${NLTK_DIR}"

"${PYTHON_BIN}" -m pip install --quiet \
  "pandas==2.2.3" \
  "numpy==1.26.4" \
  "pyarrow==12.0.1" \
  "vaderSentiment==3.3.2" \
  "nrclex==4.1.0" \
  "spacy==3.8.16" \
  "textblob==0.20.1" \
  "nltk==3.10.3"

"${PYTHON_BIN}" -m pip install --quiet \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

NLTK_DATA="${NLTK_DIR}" "${PYTHON_BIN}" -c "import nltk; nltk.download('punkt_tab', download_dir='${NLTK_DIR}', quiet=True); nltk.download('averaged_perceptron_tagger_eng', download_dir='${NLTK_DIR}', quiet=True)"

echo "Yelp pipeline dependencies installed. No Spark NLP or transformer runtime was added."
