"""Configuration contract for the EMR-native medallion pipeline."""

import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))


class EmrContractTests(unittest.TestCase):
    def test_cluster_has_ordered_medallion_and_dashboard_steps(self):
        config = json.loads((REPO_ROOT / "aws_emr" / "emr_cluster_config.json").read_text())

        def reject_comment_keys(value):
            if isinstance(value, dict):
                self.assertFalse(
                    any(key.startswith("_") for key in value),
                    "AWS CLI input must not contain documentation-only keys",
                )
                for nested in value.values():
                    reject_comment_keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    reject_comment_keys(nested)

        reject_comment_keys(config)
        self.assertEqual(config["ReleaseLabel"], "emr-7.13.0")
        self.assertFalse(config["Instances"]["KeepJobFlowAliveWhenNoSteps"])
        steps = config["Steps"]
        self.assertEqual(
            [step["Name"] for step in steps],
            [
                "stage-yelp-bronze-parquet",
                "bronze-to-silver-yelp",
                "silver-to-gold-yelp",
                "data-science-dashboard-yelp",
            ],
        )
        for step in steps:
            self.assertEqual(step["ActionOnFailure"], "TERMINATE_CLUSTER")

        bronze_args = steps[0]["HadoopJarStep"]["Args"]
        datasets_index = bronze_args.index("--datasets")
        self.assertEqual(
            bronze_args[datasets_index + 1:datasets_index + 6],
            ["review", "business", "checkin", "tip", "user"],
        )
        silver_args = steps[1]["HadoopJarStep"]["Args"]
        self.assertIn("yelp/tasks/bronze_to_silver.py", " ".join(silver_args))
        self.assertIn("--nlp-component", silver_args)
        self.assertNotIn("transformers", silver_args)
        self.assertIn("--lexicons-dir", silver_args)
        gold_args = steps[2]["HadoopJarStep"]["Args"]
        self.assertIn("--gold-level", gold_args)
        self.assertIn("brand-sample", gold_args)
        dashboard_args = steps[3]["HadoopJarStep"]["Args"]
        self.assertIn("--gold-table", dashboard_args)

        serialized = json.dumps(config).lower()
        for forbidden in ("spark-nlp", "john snow", "maven", "winutils", "java_home"):
            self.assertNotIn(forbidden, serialized)

    def test_emr_entrypoints_are_dedicated_scripts(self):
        scripts = REPO_ROOT / "aws_emr" / "scripts"
        for name in (
            "bronze_to_silver.py",
            "silver_to_gold.py",
            "data_science_dashboard.py",
            "emr_common.py",
        ):
            self.assertTrue((scripts / name).is_file(), name)
        databricks = REPO_ROOT / "databricks_integration" / "scripts"
        self.assertNotEqual(
            (scripts / "bronze_to_silver.py").read_bytes(),
            (databricks / "bronze_to_silver.py").read_bytes(),
        )

    def test_s3_lexicon_directory_preserves_uri_scheme(self):
        from lexicon_cli import LEXICON_FILENAMES, resolve_lexicon_paths

        args = argparse.Namespace(
            lexicons_dir="s3://example-bucket/yelp/lexicons",
            vad_lexicon=None,
            worry_lexicon=None,
            wcst_lexicon=None,
            yelp_lexicon=None,
            nrc_intensity_lexicon=None,
        )
        paths = resolve_lexicon_paths(args)
        for key, filename in LEXICON_FILENAMES.items():
            self.assertEqual(
                paths[key],
                f"s3://example-bucket/yelp/lexicons/{filename}",
            )

    def test_launcher_rejects_placeholders_and_returns_cluster_id(self):
        launcher_path = REPO_ROOT / "aws_emr" / "launch_cluster.py"
        spec = importlib.util.spec_from_file_location("emr_launcher", launcher_path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        unresolved = launcher.unresolved_placeholders(
            {"Instances": [{"InstanceType": "REPLACE_WITH_WORKER"}]}
        )
        self.assertEqual(len(unresolved), 1)

        class FakeClient:
            def __init__(self):
                self.config = None

            def run_job_flow(self, **config):
                self.config = config
                return {"JobFlowId": "j-TEST123"}

        client = FakeClient()
        self.assertEqual(launcher.launch({"Name": "test"}, client), "j-TEST123")
        self.assertEqual(client.config, {"Name": "test"})


if __name__ == "__main__":
    unittest.main()
