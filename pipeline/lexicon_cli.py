"""
Shared CLI plumbing for the five lexicon paths every entrypoint needs.

Two ways to point a script at your lexicons:
  1. `--lexicons-dir /path/to/lexicons` - a single folder containing all five
     files under their expected names (see lexicons/README.md).
  2. The five individual flags (`--vad-lexicon`, `--worry-lexicon`, etc.) -
     for when your files live in different places or use different names.

Individual flags always win over `--lexicons-dir` on a per-file basis, so you
can use `--lexicons-dir` for four of them and override just one.
"""

import argparse
from pathlib import Path

# Exact filenames expected inside --lexicons-dir. See lexicons/README.md for
# where to download each one.
LEXICON_FILENAMES = {
    "vad_lexicon_path": "NRC-VAD-Lexicon-v2.1.txt",
    "worry_path": "worrywords-v1.txt",
    "wcst_path": "NRC-WCST-Lexicon-v1.0.txt",
    "yelp_path": "Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt",
    "nrc_intensity_path": "NRC-Emotion-Intensity-Lexicon-v1.txt",
}

# (CLI flag/dest name, key in LEXICON_FILENAMES / lexicon_paths dict)
_ARG_TO_KEY = [
    ("vad_lexicon", "vad_lexicon_path"),
    ("worry_lexicon", "worry_path"),
    ("wcst_lexicon", "wcst_path"),
    ("yelp_lexicon", "yelp_path"),
    ("nrc_intensity_lexicon", "nrc_intensity_path"),
]


def join_resource_path(root: str, filename: str) -> str:
    """Join filesystem and URI roots without collapsing ``scheme://``."""
    if "://" in root:
        return f"{root.rstrip('/')}/{filename}"
    return str(Path(root) / filename)


def add_lexicon_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lexicons-dir", default=None,
        help="Folder containing all 5 lexicon files under their expected names "
             "(see lexicons/README.md). Overridden per-file by the flags below.",
    )
    parser.add_argument("--vad-lexicon", default=None, help="Override just the VAD lexicon path.")
    parser.add_argument("--worry-lexicon", default=None, help="Override just the WorryWords path.")
    parser.add_argument("--wcst-lexicon", default=None, help="Override just the NRC-WCST path.")
    parser.add_argument("--yelp-lexicon", default=None, help="Override just the Yelp AFFLEX-NEGLEX path.")
    parser.add_argument("--nrc-intensity-lexicon", default=None, help="Override just the NRC Emotion Intensity path.")


def resolve_lexicon_paths(args: argparse.Namespace) -> dict:
    """Returns the same 5-key dict every loader/`init_models()` call already
    expects (vad_lexicon_path, worry_path, wcst_path, yelp_path,
    nrc_intensity_path). Raises SystemExit with a clear message naming
    exactly what's missing, rather than a bare argparse "required" error per
    flag."""
    paths, missing = {}, []
    for arg_name, key in _ARG_TO_KEY:
        explicit = getattr(args, arg_name, None)
        if explicit:
            paths[key] = explicit
        elif args.lexicons_dir:
            paths[key] = join_resource_path(
                args.lexicons_dir,
                LEXICON_FILENAMES[key],
            )
        else:
            missing.append(f"--{arg_name.replace('_', '-')}")

    if missing:
        raise SystemExit(
            "Missing lexicon path(s): " + ", ".join(missing) + ".\n"
            "Either pass --lexicons-dir pointing at a folder with all 5 files "
            "under their expected names (see lexicons/README.md), or pass the "
            "flag(s) listed above individually."
        )
    return paths
