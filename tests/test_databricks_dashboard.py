"""Synthetic Spark tests for Yelp Gold variants and dashboard transforms."""

import sys
import unittest
from pathlib import Path

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "databricks_integration" / "scripts"))


@unittest.skipIf(SparkSession is None, "PySpark is not installed")
class DashboardTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("yelp-dashboard-transform-test")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_business_and_monthly_serving_tables(self):
        from data_science_dashboard import (
            build_business_summary,
            build_monthly_summary,
        )

        rows = [
            ("r1", "b1", "Cafe One", "A", "CA", "Restaurants", 5.0, "2021-01", "2021-01-01", 0.8, 0.1, 0.1, 0.1, 0.8, "brand-sample", "Cafe"),
            ("r2", "b1", "Cafe One", "A", "CA", "Restaurants", 1.0, "2021-01", "2021-01-02", -0.8, 0.8, 0.4, 0.7, 0.1, "brand-sample", "Cafe"),
            ("r3", "b2", "Salon Two", "B", "CA", "Hair Salons", 4.0, "2021-02", "2021-02-01", 0.5, 0.1, 0.1, 0.1, 0.7, "brand-sample", "Salon"),
        ]
        columns = [
            "review_id", "business_id", "name", "city", "state", "primary_industry",
            "stars", "year_month", "review_date", "vader_sentiment_score",
            "anger_int_avg", "fear_int_avg", "sadness_int_avg", "joy_int_avg",
            "gold_variant_level", "gold_variant_value",
        ]
        gold = self.spark.createDataFrame(rows, columns)
        business = build_business_summary(gold)
        monthly = build_monthly_summary(gold)

        self.assertEqual(business.count(), 2)
        self.assertEqual(monthly.count(), 2)
        cafe = business.where("gold_variant_value = 'Cafe'").first().asDict()
        self.assertEqual(cafe["review_count"], 2)
        self.assertAlmostEqual(cafe["avg_stars"], 3.0)
        self.assertAlmostEqual(cafe["low_star_rate"], 0.5)
        self.assertIn(cafe["attention_tier"], {"moderate", "high"})
        self.assertTrue(cafe["recommended_focus"])

    def test_bronze_to_silver_canary_sample_is_stable(self):
        from bronze_to_silver import sample_reviews, scope_related_dataset

        reviews = self.spark.createDataFrame(
            [("r1",), ("r2",), ("r3",), ("r4",)],
            ["review_id"],
        )
        first = {
            row.review_id
            for row in sample_reviews(reviews.repartition(1), 2, 42).collect()
        }
        second = {
            row.review_id
            for row in sample_reviews(reviews.repartition(3), 2, 42).collect()
        }
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(sample_reviews(reviews, None, 42).count(), 4)

        selected = self.spark.createDataFrame(
            [("r1", "b1", "u1"), ("r2", "b1", "u2")],
            ["review_id", "business_id", "user_id"],
        )
        businesses = self.spark.createDataFrame(
            [("b1", "kept"), ("b2", "excluded")],
            ["business_id", "name"],
        )
        users = self.spark.createDataFrame(
            [("u1", "kept"), ("u2", "kept"), ("u3", "excluded")],
            ["user_id", "name"],
        )
        scoped_businesses = scope_related_dataset(businesses, "business", selected)
        scoped_users = scope_related_dataset(users, "user", selected)
        self.assertEqual([row.business_id for row in scoped_businesses.collect()], ["b1"])
        self.assertEqual(
            {row.user_id for row in scoped_users.collect()},
            {"u1", "u2"},
        )

    def test_silver_and_gold_sample_arguments_are_independent(self):
        from bronze_to_silver import parse_args as parse_bronze_args
        from silver_to_gold import parse_args as parse_gold_args
        from silver_to_gold import effective_sample_size

        bronze = parse_bronze_args(
            [
                "--input-volume", "/tmp/yelp",
                "--silver-sample-size", "25000",
                "--silver-sample-seed", "11",
            ]
        )
        gold = parse_gold_args(
            [
                "--gold-level", "sample",
                "--gold-sample-size", "5000",
                "--gold-sample-seed", "29",
            ]
        )
        self.assertEqual(bronze.silver_sample_size, 25000)
        self.assertEqual(bronze.silver_sample_seed, 11)
        self.assertEqual(gold.gold_sample_size, 5000)
        self.assertEqual(gold.gold_sample_seed, 29)
        self.assertEqual(effective_sample_size(50000, 25000), 25000)
        self.assertEqual(effective_sample_size(5000, 25000), 5000)

    def test_gold_dependencies_and_deterministic_sampling(self):
        from silver_to_gold import (
            apply_gold_filters,
            build_gold_variant,
            deterministic_sample,
            validate_gold,
        )

        rows = [
            ("r1", "b1", "u1", "Cafe One", "A", "CA", "Restaurants", None, None, 5.0, "2021-01-01", 0.8, "joy", "joy"),
            ("r2", "b1", "u2", "Cafe One", "A", "CA", "Restaurants", "Coffee & Tea", None, 1.0, "2021-01-02", -0.8, "anger", "anger"),
            ("r3", "b2", "u1", "Salon Two", "B", "NV", "Hair Salons", None, None, 4.0, "2021-02-01", 0.5, "joy", "joy"),
            ("r4", "b3", "u3", "Bakery Three", "C", "AZ", "Bakeries", "Restaurants", "Food", 3.0, "2022-01-01", 0.0, "trust", "trust"),
        ]
        columns = [
            "review_id", "business_id", "user_id", "name", "city", "state",
            "primary_industry", "secondary_industry", "tertiary_industry", "stars",
            "review_date", "vader_sentiment_score", "dominant_emotion",
            "primary_emotion_lex",
        ]
        silver = self.spark.createDataFrame(rows, columns)
        filtered = apply_gold_filters(
            silver,
            industries=["Restaurants", "Hair Salons"],
            states=["CA", "NV"],
            start_date="2021-01-01",
            end_date="2021-12-31",
            sentiment_labels=["positive"],
        )
        self.assertEqual({row.review_id for row in filtered.collect()}, {"r1", "r3"})
        business_user = apply_gold_filters(
            silver,
            business_ids=["b1"],
            user_ids=["u2"],
        )
        self.assertEqual([row.review_id for row in business_user.collect()], ["r2"])

        first = [row.review_id for row in deterministic_sample(silver.repartition(1), 3, 17).collect()]
        second = [row.review_id for row in deterministic_sample(silver.repartition(3), 3, 17).collect()]
        self.assertEqual(first, second)

        industry = build_gold_variant(
            silver,
            gold_level="industry",
            sample_size=None,
            sample_seed=17,
            brands=None,
            date_granularity="month",
            emotion_column="dominant_emotion",
            sentiment_column="vader_sentiment_score",
        )
        validation = validate_gold(industry, "industry", None)
        self.assertTrue(validation["passed"])
        self.assertEqual(industry.count(), 3)

        emotion = build_gold_variant(
            silver,
            gold_level="emotion",
            sample_size=None,
            sample_seed=17,
            brands=None,
            date_granularity="month",
            emotion_column="dominant_emotion",
            sentiment_column="vader_sentiment_score",
        )
        self.assertEqual({row.emotion_label for row in emotion.collect()}, {"anger", "joy", "trust"})

        brands = build_gold_variant(
            silver,
            gold_level="brand-sample",
            sample_size=2,
            sample_seed=17,
            brands=["Cafe", "Salon"],
            date_granularity="month",
            emotion_column="dominant_emotion",
            sentiment_column="vader_sentiment_score",
        )
        brand_validation = validate_gold(
            brands,
            "brand-sample",
            2,
            expected_variants=["Cafe", "Salon"],
        )
        self.assertTrue(brand_validation["passed"])
        self.assertEqual(
            {row.gold_variant_value for row in brands.collect()},
            {"Cafe", "Salon"},
        )


if __name__ == "__main__":
    unittest.main()
