"""Configuration-level tests for the three-task Databricks integration."""

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = REPO_ROOT / "databricks_integration" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from common import (
    materialize_frame,
    parse_materialization_schema,
    release_resources,
    resolve_materialization_mode,
    table_name,
)
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
        self.assertIn("--input-volume", parameters)
        self.assertIn("{{job.parameters.input_volume}}", parameters)
        self.assertIn("--lexicons-dir", parameters)
        self.assertIn("{{job.parameters.lexicons_dir}}", parameters)
        self.assertIn("--silver-sample-size", parameters)
        self.assertIn("{{job.parameters.silver_sample_size}}", parameters)
        self.assertNotIn("--google-drive-folder-url", parameters)
        self.assertNotIn("--google-drive-connection", parameters)
        self.assertIn("--partitions", parameters)
        self.assertIn("--materialization-mode", parameters)
        self.assertIn("{{job.parameters.materialization_mode}}", parameters)
        self.assertIn("--materialization-schema", parameters)

        gold_parameters = tasks["silver_to_gold_yelp"]["spark_python_task"]["parameters"]
        self.assertIn("--gold-level", gold_parameters)
        self.assertIn("brand-sample", gold_parameters)
        self.assertIn("--gold-sample-size", gold_parameters)
        self.assertIn("{{job.parameters.gold_sample_size}}", gold_parameters)
        self.assertIn("--materialization-mode", gold_parameters)
        self.assertIn("--materialization-schema", gold_parameters)
        self.assertEqual(gold_parameters.count("--brand"), 2)

        dashboard_parameters = tasks["data_science_dashboard_yelp"]["spark_python_task"]["parameters"]
        self.assertIn("--gold-table", dashboard_parameters)
        self.assertNotIn("--gold-sample-size", dashboard_parameters)

        defaults = {item["name"]: item["default"] for item in config["parameters"]}
        self.assertEqual(defaults["silver_sample_size"], "25000")
        self.assertEqual(defaults["gold_sample_size"], "5000")
        self.assertEqual(defaults["materialization_mode"], "auto")
        self.assertEqual(defaults["materialization_schema"], "workspace.default")
        self.assertEqual(
            defaults["input_volume"],
            "/Volumes/workspace/default/yelp_raw/bronze",
        )
        self.assertEqual(
            defaults["lexicons_dir"],
            "/Volumes/workspace/default/yelp_raw/lexicons",
        )
        serialized = json.dumps(config)
        self.assertIn("/Volumes/workspace/default/yelp_raw/bronze", serialized)
        self.assertIn("/Volumes/workspace/default/yelp_raw/lexicons", serialized)
        self.assertNotIn("job_clusters", config)
        self.assertNotIn("job_cluster_key", serialized)
        self.assertEqual(config["format"], "MULTI_TASK")

        environments = {
            item["environment_key"]: item["spec"] for item in config["environments"]
        }
        self.assertEqual(set(environments), {"yelp_pipeline"})
        serverless = environments["yelp_pipeline"]
        self.assertEqual(serverless["environment_version"], "5")
        self.assertEqual(
            serverless["dependencies"],
            [
                "-r /Workspace/Users/lwool@sandiego.edu/"
                "MSBA503_Yelp_BigData_Pipeline/"
                "databricks_integration/requirements-serverless.txt",
                "/Workspace/Users/lwool@sandiego.edu/"
                "MSBA503_Yelp_BigData_Pipeline",
            ],
        )

        for task in tasks.values():
            self.assertEqual(task["environment_key"], "yelp_pipeline")
            python_file = task["spark_python_task"]["python_file"]
            self.assertEqual(task["spark_python_task"]["source"], "GIT")
            self.assertTrue((REPO_ROOT / python_file).is_file(), python_file)

        bronze_source = (SCRIPTS / "bronze_to_silver.py").read_text(encoding="utf-8")
        gold_source = (SCRIPTS / "silver_to_gold.py").read_text(encoding="utf-8")
        self.assertIn('F.col("_metadata.file_path")', bronze_source)
        self.assertNotIn("F.input_file_name()", bronze_source)
        self.assertNotIn(".persist(", bronze_source)
        self.assertNotIn(".persist(", gold_source)

    def test_serverless_dependencies_do_not_replace_runtime_core_packages(self):
        requirements = (
            REPO_ROOT
            / "databricks_integration"
            / "requirements-serverless.txt"
        ).read_text(encoding="utf-8").lower()
        for package in ("pyspark", "pandas", "numpy", "pyarrow"):
            self.assertNotRegex(requirements, rf"(?m)^\s*{package}\s*[=<>~]")
        self.assertIn("spacy==3.8.16", requirements)
        self.assertIn("en_core_web_sm-3.8.0", requirements)
        self.assertTrue((REPO_ROOT / "pyproject.toml").is_file())

    def test_workspace_python_tasks_run_without_dunder_file(self):
        """Databricks Workspace tasks use exec() and expose `filename` instead."""
        for script_name in (
            "bronze_to_silver.py",
            "silver_to_gold.py",
            "data_science_dashboard.py",
        ):
            script = SCRIPTS / script_name
            namespace = {
                "__name__": f"_workspace_exec_test_{script.stem}",
                "filename": str(script),
            }
            exec(
                compile(script.read_bytes(), str(script), "exec"),
                namespace,
                namespace,
            )
            self.assertNotIn("__file__", namespace)
            self.assertEqual(namespace["SCRIPT_PATH"], script.resolve())

    def test_serverless_managed_arrow_configs_are_optional(self):
        from bronze_to_silver import configure_arrow_runtime

        class ManagedConf:
            def __init__(self):
                self.keys = []

            def set(self, key, value):
                self.keys.append((key, value))
                raise RuntimeError(
                    "[CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION] Configuration "
                    f"{key} is not available"
                )

        conf = ManagedConf()
        result = configure_arrow_runtime(type("Spark", (), {"conf": conf})(), 500)
        self.assertEqual(
            result,
            {
                "spark.sql.execution.arrow.pyspark.enabled": False,
                "spark.sql.execution.arrow.maxRecordsPerBatch": False,
            },
        )
        self.assertEqual(len(conf.keys), 2)

    def test_unexpected_arrow_configuration_errors_still_fail(self):
        from bronze_to_silver import configure_arrow_runtime

        class BrokenConf:
            def set(self, key, value):
                raise RuntimeError("authentication failed")

        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            configure_arrow_runtime(type("Spark", (), {"conf": BrokenConf()})(), 500)

    def test_unity_catalog_names_are_restricted(self):
        self.assertEqual(
            table_name("workspace", "default", "yelp_gold_dashboard_variant"),
            "workspace.default.yelp_gold_dashboard_variant",
        )
        with self.assertRaises(ValueError):
            table_name("workspace", "default", "yelp; DROP TABLE reviews")

        self.assertEqual(
            parse_materialization_schema("workspace.default"),
            ("workspace", "default"),
        )
        with self.assertRaisesRegex(ValueError, "two-part"):
            parse_materialization_schema("workspace")
        self.assertEqual(resolve_materialization_mode("persist"), "persist")
        self.assertEqual(resolve_materialization_mode("delta"), "delta")
        self.assertEqual(resolve_materialization_mode("none"), "none")
        self.assertEqual(resolve_materialization_mode("auto"), "persist")
        with self.assertRaisesRegex(ValueError, "invalid materialization mode"):
            resolve_materialization_mode("cache")

        class FakeWriter:
            def __init__(self):
                self.saved = []

            def format(self, value):
                self.format_name = value
                return self

            def mode(self, value):
                self.mode_name = value
                return self

            def option(self, key, value):
                self.option_value = (key, value)
                return self

            def saveAsTable(self, value):
                self.saved.append(value)

        class FakePlan:
            def __init__(self):
                self.write = FakeWriter()

        class FakeSpark:
            def __init__(self):
                self.sql_calls = []
                self.loaded = []

            def table(self, name):
                self.loaded.append(name)
                return {"table": name}

            def sql(self, statement):
                self.sql_calls.append(statement)

        spark = FakeSpark()
        plan = FakePlan()
        metadata = {
            "materialization": {"mode": "delta", "temporary_tables": []}
        }
        result = materialize_frame(
            spark,
            plan,
            mode="delta",
            materialization_schema="workspace.default",
            stage="silver",
            run_id="abc123",
            metadata=metadata,
            persisted_frames=[],
        )
        scratch = "workspace.default._yelp_silver_abc123"
        self.assertEqual(result, {"table": scratch})
        self.assertEqual(plan.write.saved, [scratch])
        self.assertEqual(metadata["materialization"]["temporary_tables"], [scratch])
        release_resources(spark, [], metadata)
        self.assertEqual(
            spark.sql_calls,
            ["DROP TABLE IF EXISTS `workspace`.`default`.`_yelp_silver_abc123`"],
        )

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
