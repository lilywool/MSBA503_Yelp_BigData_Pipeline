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
    def test_job_has_the_required_ordered_tasks_and_volume(self):
        config = json.loads(
            (REPO_ROOT / "databricks_integration" / "job_config.json").read_text()
        )
        tasks = {task["task_key"]: task for task in config["tasks"]}
        self.assertEqual(
            list(tasks),
            ["bronze_to_silver", "silver_to_gold", "data_science_dashboard"],
        )
        self.assertNotIn("depends_on", tasks["bronze_to_silver"])
        self.assertEqual(
            tasks["silver_to_gold"]["depends_on"],
            [{"task_key": "bronze_to_silver"}],
        )
        self.assertEqual(
            tasks["data_science_dashboard"]["depends_on"],
            [{"task_key": "silver_to_gold"}],
        )
        parameters = tasks["bronze_to_silver"]["spark_python_task"]["parameters"]
        self.assertIn("/Volumes/workspace/default/yelp_academic_raw", parameters)
        self.assertIn("--lexicons-dir", parameters)
        self.assertIn("--partitions", parameters)

        gold_parameters = tasks["silver_to_gold"]["spark_python_task"]["parameters"]
        self.assertIn("--gold-level", gold_parameters)
        self.assertIn("brand-sample", gold_parameters)
        self.assertEqual(gold_parameters.count("--brand"), 2)

        dashboard_parameters = tasks["data_science_dashboard"]["spark_python_task"]["parameters"]
        self.assertIn("--gold-table", dashboard_parameters)
        self.assertNotIn("--sample-size", dashboard_parameters)

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
