"""Run the local verification suite and reject zero-test or skipped-test greens."""

from __future__ import annotations

import importlib.metadata
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSIONS = {
    "pyspark": "3.5.6",
    "pyarrow": "12.0.1",
    "pandas": "2.2.3",
    "numpy": "1.26.4",
    "vaderSentiment": "3.3.2",
    "nrclex": "4.1.0",
    "spacy": "3.8.16",
    "textblob": "0.20.1",
    "nltk": "3.10.3",
    "en-core-web-sm": "3.8.0",
}


def verify_runtime() -> list[str]:
    problems = []
    if sys.version_info[:2] != (3, 11):
        problems.append(
            f"Python 3.11 is required; running {sys.version_info.major}.{sys.version_info.minor}"
        )

    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{package} is not installed (expected {expected})")
            continue
        if actual != expected:
            problems.append(f"{package}=={actual}; expected {package}=={expected}")

    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except Exception as exc:
        problems.append(f"required NLTK corpus is unavailable: {exc}")
    return problems


def main() -> int:
    os.chdir(REPO_ROOT)
    problems = verify_runtime()
    if problems:
        print("RUNTIME CHECK FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "Run scripts/bootstrap_local.sh on Linux/WSL2 or "
            "scripts/bootstrap_local.ps1 on Windows from a fresh checkout.",
            file=sys.stderr,
        )
        return 2

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"), pattern="test_*.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    skip_count = len(result.skipped)

    if result.testsRun == 0:
        print("INVALID GREEN: zero tests were discovered.", file=sys.stderr)
        return 3
    if skip_count:
        print(
            f"INVALID GREEN: {skip_count} of {result.testsRun} tests were skipped.",
            file=sys.stderr,
        )
        return 4
    if not result.wasSuccessful():
        print("SUITE FAILED: see failures/errors above.", file=sys.stderr)
        return 1

    print(
        f"GENUINE PASS: {result.testsRun} tests executed; "
        "0 failures, 0 errors, 0 skips."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
