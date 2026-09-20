"""Regression tests for the EMR Python-module package builder."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile


REPO_ROOT = Path(__file__).parent.parent
BUILDER_PATH = REPO_ROOT / "aws_emr" / "build_pipeline_package.py"
SPEC = importlib.util.spec_from_file_location("emr_package_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class EmrPackageBuilderTests(unittest.TestCase):
    def test_package_is_complete_and_deterministic(self):
        with tempfile.TemporaryDirectory(prefix="yelp_emr_package_test_") as tmpdir:
            first = Path(tmpdir) / "first.zip"
            second = Path(tmpdir) / "second.zip"
            first_hash = builder.build_package(REPO_ROOT / "pipeline", first)
            second_hash = builder.build_package(REPO_ROOT / "pipeline", second)

            self.assertEqual(first_hash, second_hash)
            with ZipFile(first) as archive:
                names = set(archive.namelist())
            self.assertTrue(builder.REQUIRED_MODULES.issubset(names))
            self.assertIn("spark_feature_engineering.py", names)

            crlf_pipeline = Path(tmpdir) / "crlf_pipeline"
            crlf_pipeline.mkdir()
            for source in (REPO_ROOT / "pipeline").glob("*.py"):
                normalized = source.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                (crlf_pipeline / source.name).write_bytes(
                    normalized.replace(b"\n", b"\r\n")
                )
            crlf_hash = builder.build_package(
                crlf_pipeline, Path(tmpdir) / "crlf.zip"
            )
            self.assertEqual(first_hash, crlf_hash)


if __name__ == "__main__":
    unittest.main()
