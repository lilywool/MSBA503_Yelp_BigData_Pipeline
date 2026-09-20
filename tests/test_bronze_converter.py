"""Synthetic tests for the explicit-schema Bronze converter."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


@unittest.skipIf(SparkSession is None, "PySpark is not installed")
class BronzeConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("yelp-bronze-converter-test")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_review_conversion_is_explicit_and_idempotent(self):
        from bronze_json_to_parquet import DATASETS, convert_dataset

        self.assertEqual(
            set(DATASETS), {"review", "business", "checkin", "tip", "user"}
        )

        rows = [
            {
                "review_id": "r1", "user_id": "u1", "business_id": "b1",
                "stars": 5.0, "useful": 1, "funny": 0, "cool": 1,
                "text": "Excellent tacos.", "date": "2021-01-02 03:04:05",
                "unknown_future_field": "ignored by the documented schema",
            },
            {
                "review_id": "r2", "user_id": "u2", "business_id": "b2",
                "stars": 2.0, "useful": 0, "funny": 0, "cool": 0,
                "text": "Slow service.", "date": "2021-02-03 04:05:06",
            },
        ]

        with tempfile.TemporaryDirectory(prefix="yelp_bronze_test_") as tmpdir:
            root = Path(tmpdir)
            source = root / "reviews.json"
            source.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            output = root / "reviews_parquet"
            spec = DATASETS["review"]

            first = convert_dataset(
                self.spark, spec, str(source), str(output), partitions=2
            )
            second = convert_dataset(
                self.spark, spec, str(source), str(output), partitions=2
            )

            converted = self.spark.read.parquet(str(output))
            self.assertEqual(first, "converted")
            self.assertEqual(second, "skipped")
            self.assertEqual(converted.count(), 2)
            self.assertEqual(converted.schema["stars"].dataType.simpleString(), "double")
            self.assertNotIn("unknown_future_field", converted.columns)


if __name__ == "__main__":
    unittest.main()
