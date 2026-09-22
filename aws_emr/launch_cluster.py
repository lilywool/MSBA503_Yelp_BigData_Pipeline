"""Launch the transient EMR pipeline from the checked-in RunJobFlow template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PLACEHOLDER = "REPLACE_WITH_"


def unresolved_placeholders(value, path="config") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, nested in value.items():
            found.extend(unresolved_placeholders(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(unresolved_placeholders(nested, f"{path}[{index}]"))
    elif isinstance(value, str) and PLACEHOLDER in value:
        found.append(f"{path}: {value}")
    return found


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    unresolved = unresolved_placeholders(config)
    if unresolved:
        raise ValueError(
            "Resolve every EMR configuration placeholder before launch:\n  - "
            + "\n  - ".join(unresolved)
        )
    return config


def launch(config: dict, client) -> str:
    response = client.run_job_flow(**config)
    cluster_id = response.get("JobFlowId")
    if not cluster_id:
        raise RuntimeError(f"EMR did not return JobFlowId: {response}")
    return cluster_id


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("emr_cluster_config.json"),
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--profile", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit(
            "boto3 is required only for deployment. Run this launcher from AWS "
            "CloudShell or install aws_emr/requirements-deploy.txt in a dedicated "
            "project-local deployment environment."
        ) from exc
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    cluster_id = launch(config, session.client("emr"))
    print(f"Created transient EMR cluster: {cluster_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
