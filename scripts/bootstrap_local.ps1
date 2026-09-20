param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $RepoRoot "requirements.txt"
$StrictRunner = Join-Path $PSScriptRoot "run_test_suite.py"
$SpacyModelUrl = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

function Confirm-NativeSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

# PySpark 3.5 launches its local JVM through Windows batch files. Parentheses
# in SPARK_HOME (which lives under this repo's .venv) are parsed by cmd.exe as
# control syntax before Spark can start. Detect that known failure before a
# several-hundred-megabyte environment install produces JAVA_GATEWAY_EXITED.
if ($RepoRoot -match '[()]') {
    throw @"
Windows PySpark cannot launch reliably from a path containing '(' or ')':
$RepoRoot

Clone this standalone repository to a simple path such as C:\dev\yelp_iter2,
then run the bootstrap from that checkout. The original checkout can remain
where it is. WSL2/Linux is also unaffected by this cmd.exe limitation.
"@
}
if ($RepoRoot -match '\s') {
    Write-Warning "The repository path contains spaces. They may work, but a short path such as C:\dev\yelp_iter2 is safer for Windows Spark tooling."
}

# Hadoop's local filesystem emulates POSIX permissions through winutils.exe on
# native Windows. The converter and ingestion tests deliberately perform real
# Parquet writes, so a JVM-only Spark install is insufficient for this gate.
if ($env:OS -eq "Windows_NT") {
    $WinutilsPath = if ($env:HADOOP_HOME) {
        Join-Path $env:HADOOP_HOME "bin\winutils.exe"
    } else {
        $null
    }
    if (-not $WinutilsPath -or -not (Test-Path -LiteralPath $WinutilsPath)) {
        throw @"
Native Windows Parquet verification requires HADOOP_HOME to point to a trusted
Windows Hadoop build containing bin\winutils.exe. No usable binary was found.

This bootstrap will not download an unofficial executable automatically. Use a
verified Windows Hadoop distribution, or run the gate under WSL2/Linux instead.
"@
    }
}

if (Test-Path -LiteralPath $VenvPath) {
    throw "A .venv already exists at $VenvPath. Move or remove it deliberately, then rerun so this bootstrap starts clean."
}

$Java = Get-Command java -ErrorAction SilentlyContinue
if (-not $Java) {
    throw "Java is not on PATH. This project pins Java 17 for its PySpark 3.5 environment. Install it and set JAVA_HOME first."
}
$JavaVersionText = & cmd /c "java -version 2>&1" | Out-String
Confirm-NativeSuccess "Java version check"
if ($JavaVersionText -notmatch 'version\s+"?(\d+)') {
    throw "Could not parse the Java version from: $JavaVersionText"
}
if ([int]$Matches[1] -ne 17) {
    throw "Java 17 is required for the pinned Spark 3.5 gate; detected Java $($Matches[1])."
}

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PyLauncher) {
    & py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)"
    Confirm-NativeSuccess "Python 3.11 check"
    & py -3.11 -m venv $VenvPath
    Confirm-NativeSuccess "Virtual environment creation"
} else {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Python) {
        throw "Python 3.11 is required, but neither py nor python is on PATH."
    }
    & python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
    Confirm-NativeSuccess "Python 3.11 check"
    & python -m venv $VenvPath
    Confirm-NativeSuccess "Virtual environment creation"
}

& $VenvPython -m pip install --requirement $RequirementsPath
Confirm-NativeSuccess "Pinned requirement installation"
& $VenvPython -m pip install $SpacyModelUrl
Confirm-NativeSuccess "Pinned spaCy model installation"
& $VenvPython -c "import nltk, sys; names=('punkt_tab','averaged_perceptron_tagger_eng'); ok=all(nltk.download(name) for name in names); sys.exit(0 if ok else 1)"
Confirm-NativeSuccess "NLTK corpus installation"
& $VenvPython -m pip check
Confirm-NativeSuccess "Dependency consistency check"

if (-not $SkipTests) {
    # PySpark's Windows launcher searches for `python3` even when the venv
    # executable is named python.exe. Pin both processes to the verified venv.
    $env:PYSPARK_PYTHON = $VenvPython
    $env:PYSPARK_DRIVER_PYTHON = $VenvPython
    & $VenvPython $StrictRunner
    Confirm-NativeSuccess "Strict local test suite"
}

Write-Host "Local environment ready at $VenvPath"
Write-Host "Strict suite command: .\.venv\Scripts\python.exe .\scripts\run_test_suite.py"
