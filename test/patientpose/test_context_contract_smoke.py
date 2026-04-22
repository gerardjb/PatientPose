import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patientpose.cli import build_parser


CONTRACT_PATH = ROOT / "docs" / "refactor_contract.md"
MANIFEST_PATH = ROOT / "config" / "canonical_inputs.json"
TEST_OUTPUT_DIR = ROOT / "tmp" / "test_context_contract_smoke"


def _load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _datasets_by_id(manifest: dict) -> dict[str, dict]:
    return {dataset["id"]: dataset for dataset in manifest["datasets"]}


def _top_level_commands(parser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices.keys())
    raise AssertionError("Top-level CLI parser has no subcommands.")


def test_refactor_contract_and_manifest_match_repo_surface():
    assert CONTRACT_PATH.is_file()
    assert MANIFEST_PATH.is_file()

    manifest = _load_manifest()
    assert manifest["version"] >= 1
    assert manifest["datasets"]
    assert manifest["smokes"]

    top_level_commands = _top_level_commands(build_parser())
    assert set(manifest["command_families"]).issubset(top_level_commands)

    for dataset in manifest["datasets"]:
        if not dataset.get("requires_local_artifacts", False):
            continue
        for label, relative_path in dataset["paths"].items():
            if relative_path is None:
                continue
            resolved_path = ROOT / relative_path
            assert resolved_path.exists(), f"{dataset['id']} missing {label}: {resolved_path}"


def test_manifest_backed_landmark_metric_export_runtime_smoke():
    manifest = _load_manifest()
    smoke = next(item for item in manifest["smokes"] if item["id"] == "finger_tap_landmark_metric_export")
    dataset = _datasets_by_id(manifest)[smoke["dataset"]]

    required_paths = [
        ROOT / dataset["paths"]["camera_csv"],
        ROOT / dataset["paths"]["metadata_json"],
    ]
    if not all(path.exists() for path in required_paths):
        pytest.skip("Canonical fingertap artifacts are not present in this local repo state.")

    defaults = dataset["defaults"]
    trace_output = TEST_OUTPUT_DIR / "finger_tap_trace.csv"
    summary_output = TEST_OUTPUT_DIR / "finger_tap_summary.csv"
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for output_path in (trace_output, summary_output):
        if output_path.exists():
            output_path.unlink()

    command = [
        "report",
        "landmark-metric-export",
        "--project-root",
        str(ROOT),
        "--camera_csv",
        str(ROOT / dataset["paths"]["camera_csv"]),
        "--source",
        defaults["source"],
        "--space",
        defaults["space"],
        "--metric",
        defaults["metric"],
        "--trace-output",
        str(trace_output),
        "--summary-output",
        str(summary_output),
    ]

    if defaults.get("handedness"):
        command.extend(["--handedness", defaults["handedness"]])
    if defaults.get("instance_id") is not None:
        command.extend(["--instance-id", str(defaults["instance_id"])])
    if defaults.get("landmarks"):
        command.extend(["--landmarks", *defaults["landmarks"]])
    if defaults.get("components"):
        command.extend(["--components", *defaults["components"]])

    args = build_parser().parse_args(command)
    args.handler(args)

    assert trace_output.is_file()
    assert summary_output.is_file()

    trace_df = pd.read_csv(trace_output)
    summary_df = pd.read_csv(summary_output)

    assert len(trace_df) >= smoke["expect"]["min_trace_rows"]
    assert not summary_df.empty
    assert float(summary_df.loc[0, "valid_fraction"]) >= smoke["expect"]["min_valid_fraction"]
