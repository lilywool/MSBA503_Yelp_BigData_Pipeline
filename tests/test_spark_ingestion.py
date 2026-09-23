"""Synthetic tests for the full-data Spark ingestion/join route."""

import sys
import tempfile
import unittest
from pathlib import Path

try:
    from pyspark.sql import SparkSession
except ImportError:  # allows discovery in lightweight environments
    SparkSession = None

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


@unittest.skipIf(SparkSession is None, "PySpark is not installed")
class JoinedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder
                     .master("local[1]")
                     .appName("yelp-synthetic-ingestion-test")
                     .config("spark.ui.enabled", "false")
                     .config("spark.sql.ansi.enabled", "true")
                     .getOrCreate())

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_join_deduplication_aliases_and_industries(self):
        from spark_feature_engineering import read_joined_input

        reviews = self.spark.createDataFrame([
            ("r1", "u1", "b1", 5.0, "2021-01-02", "Excellent tacos."),
            ("r1", "u1", "b1", 5.0, "2021-01-02", "Excellent tacos."),
            ("r2", "u2", "b2", 2.0, "2021-02-03", "Slow haircut."),
        ], ["review_id", "user_id", "business_id", "stars", "date", "text"])
        businesses = self.spark.createDataFrame([
            ("b1", "Taco Shop", "A", "CA", 4.2, 50, 1, "Restaurants, Mexican, Tacos"),
            ("b2", "Hair Shop", "B", "CA", 3.5, 20, 1, "Hair Salons, Beauty"),
        ], ["business_id", "name", "city", "state", "stars", "review_count", "is_open", "categories"])
        users = self.spark.createDataFrame([
            ("u1", "One", 8, 4.5),
            ("u2", "Two", 3, 2.5),
        ], ["user_id", "name", "review_count", "average_stars"])

        with tempfile.TemporaryDirectory(prefix="yelp_join_test_") as tmpdir:
            root = Path(tmpdir)
            review_path = str(root / "reviews")
            business_path = str(root / "businesses")
            user_path = str(root / "users")
            reviews.write.mode("overwrite").parquet(review_path)
            businesses.write.mode("overwrite").parquet(business_path)
            users.write.mode("overwrite").parquet(user_path)

            result = read_joined_input(
                self.spark, review_path, business_path, user_path
            )
            self.assertEqual(result.count(), 2)
            self.assertTrue({
                "raw_review", "review_date", "year_month", "name",
                "user_name", "user_stars_avg", "primary_industry",
                "secondary_industry", "tertiary_industry",
            }.issubset(result.columns))

            taco = result.where("review_id = 'r1'").first().asDict()
            self.assertEqual(taco["primary_industry"], "Restaurants")
            self.assertEqual(taco["secondary_industry"], "Mexican")
            self.assertEqual(taco["tertiary_industry"], "Tacos")
            self.assertEqual(taco["year_month"], "2021-01")

            hair = result.where("review_id = 'r2'").first().asDict()
            self.assertEqual(hair["primary_industry"], "Hair Salons")
            self.assertEqual(hair["secondary_industry"], "Beauty")
            self.assertIsNone(hair["tertiary_industry"])


if __name__ == "__main__":
    unittest.main()
