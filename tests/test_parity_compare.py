"""Regression tests for strict local-vs-distributed parity comparison."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from parity_compare import compare_frames


EXPECTED_COLUMNS = ["feature_a", "feature_b"]


def frame() -> pd.DataFrame:
    data = {"review_id": ["r1", "r2"]}
    for index, column in enumerate(EXPECTED_COLUMNS):
        data[column] = [float(index), float(index + 1)]
    return pd.DataFrame(data)


class CompareTests(unittest.TestCase):
    def test_identical_frames_pass(self):
        report = compare_frames(frame(), frame().copy(), "review_id", EXPECTED_COLUMNS)
        self.assertTrue(report["all_columns_match"])
        self.assertEqual(report["n_columns_compared"], len(EXPECTED_COLUMNS))

    def test_missing_column_fails(self):
        distributed = frame().drop(columns=[EXPECTED_COLUMNS[0]])
        report = compare_frames(frame(), distributed, "review_id", EXPECTED_COLUMNS)
        self.assertFalse(report["all_columns_match"])
        self.assertIn(EXPECTED_COLUMNS[0], report["missing_distributed_columns"])

    def test_null_against_number_fails(self):
        local = frame()
        distributed = frame()
        local.loc[0, EXPECTED_COLUMNS[0]] = np.nan
        report = compare_frames(local, distributed, "review_id", EXPECTED_COLUMNS)
        self.assertFalse(report["all_columns_match"])
        self.assertEqual(report["columns"][EXPECTED_COLUMNS[0]]["null_mismatches"], 1)

    def test_different_key_sets_fail(self):
        distributed = frame()
        distributed.loc[1, "review_id"] = "different"
        report = compare_frames(frame(), distributed, "review_id", EXPECTED_COLUMNS)
        self.assertFalse(report["all_columns_match"])
        self.assertFalse(report["key_sets_match"])


if __name__ == "__main__":
    unittest.main()
