"""Build the deterministic ``pipeline_modules.zip`` used by EMR.

Every top-level ``pipeline/*.py`` module is flattened into the archive because
the EMR entry point imports sibling modules by their bare module names.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


REQUIRED_MODULES = {
    "corrected_feature_engineering.py",
    "iteration2_features.py",
    "iteration2_lexicons.py",
    "lexicon_cli.py",
    "real_feature_engineering.py",
    "run_logger.py",
    "spark_validate.py",
}
FIXED_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def build_package(pipeline_dir: Path, output_path: Path) -> str:
    pipeline_dir = pipeline_dir.resolve()
    output_path = output_path.resolve()
    modules = sorted(pipeline_dir.glob("*.py"), key=lambda path: path.name)
    names = {path.name for path in modules}
    missing = sorted(REQUIRED_MODULES - names)
    if missing:
        raise RuntimeError(f"Cannot build package; required modules missing: {missing}")
    if not modules:
        raise RuntimeError(f"No Python modules found in {pipeline_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for module in modules:
            info = ZipInfo(module.name, date_time=FIXED_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            # Git may materialize a checkout with LF or CRLF depending on the
            # host configuration. Normalize source bytes so identical commits
            # produce the same deployment artifact across Windows/Linux clones.
            source = module.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            archive.writestr(info, source)

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"Built {output_path}")
    print(f"Modules: {', '.join(path.name for path in modules)}")
    print(f"SHA256: {digest}")
    return digest


def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build EMR pipeline_modules.zip")
    parser.add_argument("--pipeline-dir", type=Path, default=repo_root / "pipeline")
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "aws_emr" / "dist" / "pipeline_modules.zip",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    build_package(args.pipeline_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
