from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mocopi import load_mocopi_recording
from mocopi.features import NoCameraPoseDataError
from mocopi.reliability import RELIABILITY_COLUMNS, export_reliability_errors
from mocopi.sync import estimate_camera_to_mocopi_offset
from patientpose.config import resolve_project_paths


def _resolve_cli_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path).resolve()


def add_reliability_export_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument(
        "--motion",
        "--bvh",
        dest="motion_source",
        type=Path,
        required=True,
        help="Path to Mocopi motion source (.bvh, .bin, or session directory).",
    )
    parser.add_argument(
        "--camera_csv",
        type=Path,
        required=True,
        help="Path to camera landmarks CSV (results/OutputCSVs).",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=["l_foot", "r_foot", "l_hand", "r_hand"],
        help="Mocopi joint names to compare.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=["LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_WRIST", "RIGHT_WRIST"],
        help="Camera pose landmarks corresponding to --joints (same order).",
    )
    parser.add_argument(
        "--offset_ms",
        type=float,
        default=None,
        help="Optional camera-to-mocopi offset in ms. If omitted, estimated from r_hand/RIGHT_WRIST.",
    )
    parser.add_argument(
        "--search_ms",
        type=float,
        default=5000.0,
        help="Search range for offset estimation in ms (+/- value).",
    )
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=50.0,
        help="Resampling rate (Hz) used when estimating offset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <project-root>/results/mocopi_camera_reliability.csv.",
    )
    parser.add_argument(
        "--clip-start",
        type=float,
        default=None,
        help="Optional start time (s) to include in offset estimation (to skip early sit/stand).",
    )
    parser.add_argument(
        "--clip-end",
        type=float,
        default=None,
        help="Optional end time (s) to include in offset estimation (to skip late sit/stand).",
    )
    return parser


def build_reliability_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-frame Mocopi vs MediaPipe egocentric errors."
    )
    return add_reliability_export_args(parser)


def compute_or_use_offset(
    seq,
    cam_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
    offset_ms: float | None,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> float:
    return estimate_camera_to_mocopi_offset(
        seq,
        cam_df,
        search_ms,
        rate_hz,
        offset_ms,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
    )


def run_reliability_export(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    motion_source = _resolve_cli_path(args.motion_source, paths.root)
    camera_csv = _resolve_cli_path(args.camera_csv, paths.root)
    output_csv = (
        _resolve_cli_path(args.output, paths.root)
        if args.output is not None
        else (paths.results / "mocopi_camera_reliability.csv").resolve()
    )

    seq = load_mocopi_recording(motion_source)
    cam_df = pd.read_csv(camera_csv)

    try:
        offset_ms = compute_or_use_offset(
            seq,
            cam_df,
            args.search_ms,
            args.rate_hz,
            args.offset_ms,
            clip_start_s=args.clip_start,
            clip_end_s=args.clip_end,
        )
        df_out = export_reliability_errors(seq, cam_df, args.joints, args.landmarks, offset_ms)
    except NoCameraPoseDataError:
        df_out = pd.DataFrame(columns=RELIABILITY_COLUMNS)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    if df_out.empty:
        print(
            f"Wrote empty reliability CSV to {output_csv} because the camera CSV had no usable pose landmarks."
        )
    else:
        print(f"Wrote {len(df_out)} rows to {output_csv}")


def main_reliability_export(argv: list[str] | None = None) -> None:
    parser = build_reliability_export_parser()
    args = parser.parse_args(argv)
    run_reliability_export(args)
