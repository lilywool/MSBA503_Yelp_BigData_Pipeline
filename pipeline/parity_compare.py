"""Spark-free comparison logic for local-vs-distributed parity tests."""

import pandas as pd


def compare_frames(local_df: pd.DataFrame, dist_df: pd.DataFrame,
                   key_col: str, expected_cols: list[str],
                   float_tolerance: float = 1e-6) -> dict:
    """Compare complete schemas, unique keys, null positions, and values.

    This module deliberately has no PySpark or feature-pipeline imports so its
    adversarial regression tests run even in a lightweight pandas environment.
    """
    report = {
        "n_local": len(local_df),
        "n_distributed": len(dist_df),
        "n_columns_expected": len(expected_cols),
        "columns": {},
    }

    if len(local_df) != len(dist_df):
        report["row_count_mismatch"] = True
        report["all_columns_match"] = False
        return report
    report["row_count_mismatch"] = False

    missing_local = [c for c in expected_cols if c not in local_df.columns]
    missing_distributed = [c for c in expected_cols if c not in dist_df.columns]
    report["missing_local_columns"] = missing_local
    report["missing_distributed_columns"] = missing_distributed
    report["schema_match"] = not missing_local and not missing_distributed

    if key_col not in local_df.columns or key_col not in dist_df.columns:
        report["key_error"] = f"{key_col!r} must be present in both outputs"
        report["all_columns_match"] = False
        return report

    duplicate_local = int(local_df[key_col].duplicated().sum())
    duplicate_distributed = int(dist_df[key_col].duplicated().sum())
    report["duplicate_local_keys"] = duplicate_local
    report["duplicate_distributed_keys"] = duplicate_distributed
    if duplicate_local or duplicate_distributed:
        report["key_error"] = "comparison key must be unique in both outputs"
        report["all_columns_match"] = False
        return report

    local_df = local_df.set_index(key_col).sort_index()
    dist_df = dist_df.set_index(key_col).sort_index()
    report["key_sets_match"] = bool(local_df.index.equals(dist_df.index))
    if not report["key_sets_match"]:
        report["all_columns_match"] = False
        return report

    all_match = report["schema_match"]
    compared = 0
    for column in expected_cols:
        if column in missing_local or column in missing_distributed:
            report["columns"][column] = {"match": False, "reason": "missing column"}
            continue

        compared += 1
        local_col, dist_col = local_df[column], dist_df[column]
        null_match = local_col.isna().eq(dist_col.isna())
        present = ~(local_col.isna() | dist_col.isna())

        if pd.api.types.is_numeric_dtype(local_col):
            try:
                diff = (local_col[present].astype(float)
                        - dist_col[present].astype(float)).abs()
                values_match = bool((diff <= float_tolerance).all())
                max_diff = float(diff.max()) if len(diff) else 0.0
            except (TypeError, ValueError):
                values_match = False
                max_diff = None
            match = bool(null_match.all() and values_match)
            report["columns"][column] = {
                "match": match,
                "max_abs_diff": max_diff,
                "null_mismatches": int((~null_match).sum()),
            }
        else:
            values_match = local_col[present].astype(str).eq(
                dist_col[present].astype(str)
            )
            match = bool(null_match.all() and values_match.all())
            report["columns"][column] = {
                "match": match,
                "null_mismatches": int((~null_match).sum()),
            }
        all_match = all_match and match

    report["all_columns_match"] = all_match
    report["n_columns_compared"] = compared
    return report
