"""Configuration-level tests for the three-task Databricks integration."""

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = REPO_ROOT / "databricks_integration" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from common import table_name
from corrected_feature_engineering import output_columns_for


class DatabricksContractTests(unittest.TestCase):
    def test_job_has_ordered_tasks_and_independent_sample_controls(self):
        config = json.loads(
            (REPO_ROOT / "databricks_integration" / "job_config.json").read_text()
        )
        tasks = {task["task_key"]: task for task in config["tasks"]}
        self.assertEqual(
            list(tasks),
            [
                "bronze_to_silver_yelp",
                "silver_to_gold_yelp",
                "data_science_dashboard_yelp",
            ],
        )
        self.assertNotIn("depends_on", tasks["bronze_to_silver_yelp"])
        self.assertEqual(
            tasks["silver_to_gold_yelp"]["depends_on"],
            [{"task_key": "bronze_to_silver_yelp"}],
        )
        self.assertEqual(
            tasks["data_science_dashboard_yelp"]["depends_on"],
            [{"task_key": "silver_to_gold_yelp"}],
        )
        parameters = tasks["bronze_to_silver_yelp"]["spark_python_task"]["parameters"]
        self.assertIn("--google-drive-folder-url", parameters)
        self.assertIn("--google-drive-connection", parameters)
        self.assertIn("--silver-sample-size", parameters)
        self.assertIn("{{job.parameters.silver_sample_size}}", parameters)
        self.assertIn("--lexicons-dir", parameters)
        self.assertIn("--partitions", parameters)

        gold_parameters = tasks["silver_to_gold_yelp"]["spark_python_task"]["parameters"]
        self.assertIn("--gold-level", gold_parameters)
        self.assertIn("brand-sample", gold_parameters)
        self.assertIn("--gold-sample-size", gold_parameters)
        self.assertIn("{{job.parameters.gold_sample_size}}", gold_parameters)
        self.assertEqual(gold_parameters.count("--brand"), 2)

        dashboard_parameters = tasks["data_science_dashboard_yelp"]["spark_python_task"]["parameters"]
        self.assertIn("--gold-table", dashboard_parameters)
        self.assertNotIn("--gold-sample-size", dashboard_parameters)

        defaults = {item["name"]: item["default"] for item in config["parameters"]}
        self.assertEqual(defaults["silver_sample_size"], "25000")
        self.assertEqual(defaults["gold_sample_size"], "5000")
        self.assertTrue(
            config["job_clusters"][0]["new_cluster"]["spark_version"].startswith("17.3")
        )

        for task in tasks.values():
            python_file = task["spark_python_task"]["python_file"]
            self.assertEqual(task["spark_python_task"]["source"], "GIT")
            self.assertTrue((REPO_ROOT / python_file).is_file(), python_file)

    def test_unity_catalog_names_are_restricted(self):
        self.assertEqual(
            table_name("workspace", "default", "yelp_gold_dashboard_variant"),
            "workspace.default.yelp_gold_dashboard_variant",
        )
        with self.assertRaises(ValueError):
            table_name("workspace", "default", "yelp; DROP TABLE reviews")

    def test_dashboard_app_is_deployable_from_its_own_directory(self):
        app_dir = REPO_ROOT / "databricks_integration" / "dashboard_app"
        for name in ("app.py", "app.yaml", "requirements.txt", "README.md"):
            self.assertTrue((app_dir / name).is_file(), name)

    def test_silver_feature_families_are_independently_selectable(self):
        vader_only = output_columns_for(["vader"])
        self.assertEqual(
            vader_only,
            ["vader_sentiment_score", "vader_pos", "vader_neu", "vader_neg"],
        )
        combined = output_columns_for(["domain-lexicons", "transformers"])
        self.assertIn("yelp_sentiment_avg", combined)
        self.assertIn("primary_emotion_lex", combined)
        self.assertIn("hf_sentiment_label", combined)
        self.assertNotIn("vader_sentiment_score", combined)


if __name__ == "__main__":
    unittest.main()
