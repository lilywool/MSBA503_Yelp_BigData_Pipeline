# Local verification before cloud spend

The recommended route on this machine is WSL2/Linux. Clone the repository into
the Linux filesystem, not a Windows drive mounted below `/mnt`:

```bash
git clone <this repo> ~/yelp_iter2
cd ~/yelp_iter2
./scripts/bootstrap_local.sh
```

The Bash bootstrap refuses `/mnt/c/...` and other `/mnt/...` checkouts under
WSL2. Keeping the checkout under `~/` avoids cross-filesystem performance and
permission differences while also matching the Linux hosts used by EMR and
Databricks.

The gate requires Python 3.11 and Java 17 exactly. Ubuntu 24.04 defaults to
Python 3.12 and OpenJDK 21, so its default `python3` and Java do not match the
pinned Spark 3.5 environment. Install OpenJDK 17 and provide Python 3.11 through
one of these deliberate routes:

- make a trusted `python3.11` available on `PATH`;
- set `PYTHON311` to its full executable path; or
- install `uv` through an approved method and let the bootstrap run
  `uv python install 3.11`.

The bootstrap validates both versions before creating `.venv`. It does not add
a third-party package repository, install `uv`, or replace Ubuntu's system
Python.

The native Windows route remains available only when its prerequisites are
already satisfied. Run this from PowerShell at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local.ps1
```

On Windows, run from a short checkout path without parentheses, for example
`C:\dev\yelp_iter2`. PySpark 3.5 starts its JVM through `cmd.exe` batch files;
parentheses inside the repository/`.venv` path can terminate their expressions
and surface as the misleading `JAVA_GATEWAY_EXITED` error. The bootstrap now
checks this before creating or installing the environment. Paths containing
spaces receive a warning as an additional portability precaution. WSL2 and
Linux do not have this Windows batch-parser constraint.

Native Windows also requires a trusted Hadoop build with
`%HADOOP_HOME%\bin\winutils.exe` because the two Spark integration tests perform
real local Parquet writes. The bootstrap checks that prerequisite before
installing. It deliberately does not fetch an unofficial executable. WSL2/Linux
does not require `winutils.exe` and is the recommended route when no verified
Windows Hadoop build is already available.

The bootstrap requires Python 3.11 and Java 17. It creates a fresh
`.venv`, installs every version in `requirements.txt`, installs the compatible
`en_core_web_sm` 3.8.0 model, downloads `punkt_tab` and
`averaged_perceptron_tagger_eng`, checks dependency consistency, and runs the
strict suite. The local gate pins PySpark 3.5.6 to the Spark 3.5 line used by
EMR and the selected Databricks LTS runtime. The project deliberately narrows
the interpreter/JVM choices to Python 3.11 and Java 17 for repeatability.

To run the suite again without rebuilding the environment on Linux/WSL2:

```bash
PYSPARK_PYTHON="$PWD/.venv/bin/python" \
PYSPARK_DRIVER_PYTHON="$PWD/.venv/bin/python" \
./.venv/bin/python ./scripts/run_test_suite.py
```

On Windows:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_test_suite.py
```

A genuine pass ends with this exact shape:

```text
GENUINE PASS: N tests executed; 0 failures, 0 errors, 0 skips.
```

Any `skipped=N`, `Ran 0 tests`, missing dependency, wrong pinned version, or
missing NLP resource makes the strict runner exit unsuccessfully. Ordinary
`python -m unittest` treats skipped tests as success, so it is not the release
gate for this repository.

Last verified on 2026-09-20 under WSL2: `15 tests executed; 0 failures, 0
errors, 0 skips.` The deterministic EMR package built during that run had
SHA-256 `a6c83d49a024469d6814b9fe60a3b215abedcb004444bd3ec1be0059ddc5355d`.
The hash is expected to change whenever a packaged pipeline module changes.

The corrected end-to-end parity check also passed on 2026-09-03 using the
coursework-era Chipotle sample and the five real lexicons. Each of the two disjoint
1,500-row slices matched 1,500 local rows to 1,500 distributed rows across all
68 expected feature columns. Schema, review-key sets, null positions, values,
and row counts all matched.

The local suite uses only synthetic fixtures created in temporary directories.
It does not access the project corpus or any cloud account. Passing it permits
a small EMR or Databricks canary; it does not prove the 8.6-million-row run.
