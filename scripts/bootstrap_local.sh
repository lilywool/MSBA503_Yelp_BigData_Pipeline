#!/usr/bin/env bash

set -euo pipefail

skip_tests=false
case "${1:-}" in
    "") ;;
    --skip-tests) skip_tests=true ;;
    *)
        echo "Usage: $0 [--skip-tests]" >&2
        exit 64
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
venv_path="$repo_root/.venv"
venv_python="$venv_path/bin/python"
requirements_path="$repo_root/requirements.txt"
strict_runner="$script_dir/run_test_suite.py"
spacy_model_url="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    case "$repo_root" in
        /mnt/*)
            cat >&2 <<EOF
The repository is on a Windows-mounted filesystem:
$repo_root

For WSL2 verification, clone it into the Linux filesystem (for example,
~/yelp_iter2) and run this bootstrap there. Do not run the Spark gate from
/mnt/c or another /mnt drive.
EOF
            exit 2
            ;;
    esac
fi

if [[ -e "$venv_path" ]]; then
    echo "A .venv already exists at $venv_path." >&2
    echo "Move or remove it deliberately, then rerun so this bootstrap starts clean." >&2
    exit 2
fi

if ! command -v java >/dev/null 2>&1; then
    echo "Java is not on PATH. Install OpenJDK 17 and set JAVA_HOME first." >&2
    exit 2
fi

java_version_text="$(java -version 2>&1)"
java_major="$(printf '%s\n' "$java_version_text" | sed -nE 's/.*version "?([0-9]+).*/\1/p' | head -n 1)"
if [[ -z "$java_major" ]]; then
    echo "Could not parse the Java version from:" >&2
    printf '%s\n' "$java_version_text" >&2
    exit 2
fi
if (( java_major != 17 )); then
    echo "Java 17 is required for the pinned Spark 3.5 gate; detected Java $java_major." >&2
    exit 2
fi

python311="${PYTHON311:-}"
if [[ -z "$python311" ]]; then
    python311="$(command -v python3.11 || true)"
fi
if [[ -z "$python311" ]] && command -v uv >/dev/null 2>&1; then
    echo "Python 3.11 is not on PATH; provisioning it with the existing uv installation."
    uv python install 3.11
    python311="$(uv python find 3.11)"
fi
if [[ -z "$python311" ]]; then
    cat >&2 <<'EOF'
Python 3.11 is required, but python3.11 was not found on PATH.

Ubuntu 24.04 provides Python 3.12 by default, so its stock `python3` does not
satisfy this pinned gate. Install Python 3.11 from a source you trust. If uv is
already installed, this script can use it to provision 3.11; a managed pyenv
installation or approved package repository also works. You may instead set
PYTHON311 to the full Python 3.11 executable path.
EOF
    exit 2
fi

if ! "$python311" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'; then
    echo "PYTHON311 must identify Python 3.11; received: $python311" >&2
    exit 2
fi

if ! "$python311" -m venv "$venv_path"; then
    cat >&2 <<EOF
Python 3.11 could not create a virtual environment. Install the matching venv
support for the Python 3.11 distribution at $python311, then rerun.
EOF
    exit 2
fi

"$venv_python" -m pip install --requirement "$requirements_path"
"$venv_python" -m pip install "$spacy_model_url"
"$venv_python" -c "import nltk, sys; names=('punkt_tab','averaged_perceptron_tagger_eng'); ok=all(nltk.download(name) for name in names); sys.exit(0 if ok else 1)"
"$venv_python" -m pip check

if [[ "$skip_tests" == false ]]; then
    export PYSPARK_PYTHON="$venv_python"
    export PYSPARK_DRIVER_PYTHON="$venv_python"
    "$venv_python" "$strict_runner"
fi

echo "Local environment ready at $venv_path"
echo "Strict suite command: PYSPARK_PYTHON=\"$venv_python\" PYSPARK_DRIVER_PYTHON=\"$venv_python\" \"$venv_python\" \"$strict_runner\""
