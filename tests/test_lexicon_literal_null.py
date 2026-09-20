"""Regression tests for pandas treating the valid lexicon term `null` as NA."""

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from iteration2_lexicons import load_nrc_intensity, load_wcst, load_worry_words
from real_feature_engineering import load_vad_lexicon


class LiteralNullLexiconTests(unittest.TestCase):
    def test_literal_null_survives_every_pandas_lexicon_loader(self):
        with tempfile.TemporaryDirectory(prefix="yelp_null_lexicon_test_") as tmpdir:
            root = Path(tmpdir)
            vad_path = root / "vad.tsv"
            worry_path = root / "worry.tsv"
            wcst_path = root / "wcst.tsv"
            intensity_path = root / "intensity.tsv"

            vad_path.write_text(
                "term\tvalence\tarousal\tdominance\n"
                "null\t0.1\t0.2\t0.3\n"
                "multi word\t0.4\t0.5\t0.6\n",
                encoding="utf-8",
            )
            worry_path.write_text(
                "Term\tMajorityLabel\nnull\t2\ncalm\t1\n",
                encoding="utf-8",
            )
            wcst_path.write_text(
                "term\twarmth (W)\tcompetence (C)\tsociability (S)\ttrust (T)\n"
                "null\t0.1\t0.2\t0.3\t0.4\n",
                encoding="utf-8",
            )
            intensity_path.write_text(
                "null\tfear\t0.75\n",
                encoding="utf-8",
            )

            vad = load_vad_lexicon(str(vad_path))
            worry = load_worry_words(str(worry_path))
            wcst = load_wcst(str(wcst_path))
            intensity = load_nrc_intensity(str(intensity_path))

            self.assertIn("null", vad)
            self.assertNotIn("multi word", vad)
            self.assertEqual(vad["null"]["valence"], 0.1)
            self.assertIn("null", worry)
            self.assertEqual(wcst["null"]["trust"], 0.4)
            self.assertEqual(intensity["null"]["fear"], 0.75)


if __name__ == "__main__":
    unittest.main()
