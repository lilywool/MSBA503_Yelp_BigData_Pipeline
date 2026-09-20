"""
The single most important script in this repo.

Runs the SAME input rows through both:
  (a) the pipeline running locally in plain pandas (corrected_feature_engineering.py), and
  (b) this repo's distributed Spark mapInPandas wrapper (spark_feature_engineering.py)
      wrapping the SAME module, unchanged.

then asserts the two outputs match row-for-row, column-for-column. If they don't,
something about the distribution (partitioning, worker-side model/lexicon init,
schema mismatch) silently changed the numbers.

Uses MORE THAN ONE sample slice on purpose: a benchmark that looks fine on one
small sample can behave differently on a full/sequential run. Two disjoint
slices here by default.

Usage:
    python parity_check.py --file ../data/chipotle_sample_15000.csv --lexicons-dir ../lexicons
"""

import sys
import argparse
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
import corrected_feature_engineering as cfe
from spark_feature_engineering import run as spark_run
from lexicon_cli import add_lexicon_args, resolve_lexicon_paths
from parity_compare import compare_frames

FLOAT_COLS_TOLERANCE = 1e-6


def run_local(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df[[c for c in df.columns if c not in cfe.ALL_GENERATED_COLUMNS]]
    return cfe.process_dataframe(df, text_col=text_col)


def compare(local_df: pd.DataFrame, dist_df: pd.DataFrame, key_col: str) -> dict:
    expected_cols = [
        c for c in cfe.OUTPUT_COLUMNS
        if c not in cfe.rfe.TRANSFORMER_COLUMNS
    ]
    return compare_frames(
        local_df,
        dist_df,
        key_col=key_col,
        expected_cols=expected_cols,
        float_tolerance=FLOAT_COLS_TOLERANCE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--text-col", default="raw_review")
    parser.add_argument("--key-col", default="review_id")
    add_lexicon_args(parser)
    parser.add_argument("--slice1", default="0:1500")
    parser.add_argument("--slice2", default="6000:7500")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--out", default="parity_check_report.json")
    args = parser.parse_args()

    lexicon_paths = resolve_lexicon_paths(args)

    full = pd.read_csv(args.file)
    slices = {}
    for name, spec in [("slice1", args.slice1), ("slice2", args.slice2)]:
        a, b = (int(x) for x in spec.split(":"))
        slices[name] = full.iloc[a:b].copy()

    spark = SparkSession.builder.appName("parity-check").master(f"local[{args.partitions}]").getOrCreate()

    results = {}
    for name, df in slices.items():
        print(f"\n{'='*70}\nPARITY CHECK: {name} ({len(df)} rows)\n{'='*70}")

        with tempfile.TemporaryDirectory(prefix=f"yelp_parity_{name}_") as tmpdir:
            tmp_csv = str(Path(tmpdir) / f"parity_{name}.csv")
            df.to_csv(tmp_csv, index=False)

            cfe.init_models(**lexicon_paths)
            local_out = run_local(df.copy(), args.text_col)
            dist_out = spark_run(spark, tmp_csv, args.text_col, lexicon_paths,
                                  use_transformers=False, num_partitions=args.partitions).toPandas()

        if args.key_col not in dist_out.columns and args.key_col in df.columns:
            dist_out[args.key_col] = df[args.key_col].reset_index(drop=True).values
        if args.key_col not in local_out.columns and args.key_col in df.columns:
            local_out[args.key_col] = df[args.key_col].reset_index(drop=True).values

        cmp = compare(local_out, dist_out, key_col=args.key_col)
        results[name] = cmp
        print(json.dumps(cmp, indent=2)[:3000])

    overall_pass = all(
        r.get("all_columns_match")
        and r.get("schema_match")
        and r.get("key_sets_match")
        and r.get("n_columns_compared") == r.get("n_columns_expected")
        and not r.get("row_count_mismatch")
        for r in results.values()
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file": args.file,
        "overall_pass": overall_pass,
        "slices": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n{'='*70}\nOVERALL PARITY: {'PASS' if overall_pass else 'FAIL'}\nReport: {args.out}\n{'='*70}")
    spark.stop()
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
