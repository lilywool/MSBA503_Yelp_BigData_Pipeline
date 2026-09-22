"""Build the deterministic ``pipeline_modules.zip`` used by EMR.

Canonical ``pipeline/*.py`` modules plus the EMR runtime helper are flattened
into the archive because EMR entry points import them by bare module name.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


REQUIRED_PIPELINE_MODULES = {
    "corrected_feature_engineering.py",
    "iteration2_features.py",
    "iteration2_lexicons.py",
    "lexicon_cli.py",
    "medallion_layers.py",
    "real_feature_engineering.py",
    "run_logger.py",
    "spark_validate.py",
}
REQUIRED_TASK_MODULES = {"emr_common.py"}
FIXED_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def build_package(
    pipeline_dir: Path,
    output_path: Path,
    task_modules: tuple[Path, ...] = (),
) -> str:
    pipeline_dir = pipeline_dir.resolve()
    output_path = output_path.resolve()
    modules = sorted(pipeline_dir.glob("*.py"), key=lambda path: path.name)
    task_modules = tuple(Path(path).resolve() for path in task_modules)
    names = {path.name for path in modules}
    missing = sorted(REQUIRED_PIPELINE_MODULES - names)
    if missing:
        raise RuntimeError(f"Cannot build package; required modules missing: {missing}")
    if not modules:
        raise RuntimeError(f"No Python modules found in {pipeline_dir}")
    missing_task_modules = sorted(
        REQUIRED_TASK_MODULES - {path.name for path in task_modules}
    )
    if missing_task_modules:
        raise RuntimeError(
            "Cannot build EMR task package; required modules missing: "
            f"{missing_task_modules}"
        )
    missing_files = [str(path) for path in task_modules if not path.is_file()]
    if missing_files:
        raise RuntimeError(f"Cannot build package; files not found: {missing_files}")

    package_modules = modules + sorted(task_modules, key=lambda path: path.name)
    package_names = [path.name for path in package_modules]
    if len(package_names) != len(set(package_names)):
        raise RuntimeError(f"Duplicate module names in EMR package: {package_names}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for module in package_modules:
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
    print(f"Modules: {', '.join(path.name for path in package_modules)}")
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
    repo_root = Path(__file__).resolve().parents[1]
    build_package(
        args.pipeline_dir,
        args.output,
        task_modules=(
            repo_root / "aws_emr" / "scripts" / "emr_common.py",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
