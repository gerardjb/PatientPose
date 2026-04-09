from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mocopi import load_mocopi_recording
from mocopi.features import NoCameraPoseDataError
from mocopi.reliability import (
    RELIABILITY_COLUMNS,
    default_comparison_components,
    ensure_reliability_csv,
    export_reliability_errors,
)
from mocopi.sync import estimate_camera_to_mocopi_offset
from patientpose.config import resolve_project_paths
from patientpose.datasets.discovery import discover_pairs
from patientpose.datasets.models import TrialPair
from patientpose.datasets.roles import parse_camera_role_specs
from patientpose.datasets.session_layout import discover_sessions
from patientpose.landmarks import infer_pose_world_csv


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
        "--camera-space",
        choices=("image", "world"),
        default="world",
        help="Which pose representation to compare against Mocopi.",
    )
    parser.add_argument(
        "--world-csv",
        type=Path,
        default=None,
        help="Optional explicit path to the pose-world CSV used when --camera-space=world.",
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
        help="Output CSV path. Defaults to <project-root>/results/mocopi_camera_reliability_<space>.csv.",
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


def add_reliability_batch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Optional subset of pair ids to process. Legacy mode uses tags like 1a; session mode uses session ids.",
    )
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
    )
    parser.add_argument(
        "--camera-space",
        choices=("image", "world"),
        default="world",
        help="Which pose representation to compare against Mocopi.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write reliability CSVs. Defaults to <project-root>/results/mocopi_reliability.",
    )
    parser.add_argument(
        "--offset_ms",
        type=float,
        default=None,
        help="Optional fixed camera-to-mocopi offset to reuse. If omitted, offsets are estimated per pair.",
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
    return parser


def build_reliability_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-frame Mocopi vs MediaPipe egocentric errors."
    )
    return add_reliability_export_args(parser)


def build_reliability_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch run Mocopi reliability export for discovered Mocopi/camera pairs."
    )
    return add_reliability_batch_args(parser)


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
    projection_camera_csv = camera_csv
    if args.camera_space == "world":
        projection_camera_csv = (
            _resolve_cli_path(args.world_csv, paths.root)
            if args.world_csv is not None
            else infer_pose_world_csv(camera_csv, paths.root)
        )
    output_csv = (
        _resolve_cli_path(args.output, paths.root)
        if args.output is not None
        else (paths.results / f"mocopi_camera_reliability_{args.camera_space}.csv").resolve()
    )

    seq = load_mocopi_recording(motion_source)
    cam_df = pd.read_csv(camera_csv)
    projection_df = pd.read_csv(projection_camera_csv)

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
        df_out = export_reliability_errors(
            seq,
            projection_df,
            args.joints,
            args.landmarks,
            offset_ms,
            camera_space=args.camera_space,
            comparison_components=default_comparison_components(args.camera_space),
        )
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


def _default_landmarks_csv(video_stem: str, project_root: Path) -> Path:
    return (project_root / "results" / "OutputCSVs" / f"landmarks_{video_stem}.csv").resolve()


def _run_reliability_for_pair(
    pair: TrialPair,
    output_dir: Path,
    project_root: Path,
    offset_ms: float | None,
    search_ms: float,
    rate_hz: float,
    *,
    camera_space: str,
) -> None:
    nd_stem = pair.nd_video.stem
    camera_csv = _default_landmarks_csv(nd_stem, project_root)
    if not camera_csv.exists():
        print(f"Skipping {nd_stem}: camera CSV not found at {camera_csv}")
        return
    projection_camera_csv = camera_csv if camera_space == "image" else infer_pose_world_csv(camera_csv, project_root)
    if not projection_camera_csv.exists():
        print(f"Skipping {nd_stem}: {camera_space} camera CSV not found at {projection_camera_csv}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mocopi_camera_reliability_{camera_space}_{nd_stem}.csv"

    print(f"Running reliability export for tag {pair.tag} -> {nd_stem} ({camera_space})")
    output_csv = ensure_reliability_csv(
        pair.motion_source,
        projection_camera_csv,
        output_path,
        offset_ms,
        search_ms,
        rate_hz,
        offset_camera_csv=camera_csv,
        camera_space=camera_space,
        comparison_components=default_comparison_components(camera_space),
    )
    try:
        df = pd.read_csv(output_csv)
    except Exception:
        df = None
    if df is not None and list(df.columns) == RELIABILITY_COLUMNS and df.empty:
        print(f"Skipping metrics for {nd_stem}: camera CSV contains no usable pose landmarks.")


def run_reliability_batch(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    camera_roles = parse_camera_role_specs(args.camera_role)
    pairs = discover_pairs(paths.root, camera_roles=camera_roles)
    if args.tags:
        tags = set(args.tags)
        pairs = [p for p in pairs if p.tag in tags]

    if not pairs:
        sessions = discover_sessions(paths.root)
        if sessions and not camera_roles:
            print(
                "Discovered session folders, but no session pairs were resolved. "
                "Add --camera-role CAMERA_ID=A and --camera-role CAMERA_ID=ND."
            )
        else:
            print("No matching Mocopi/camera pairs found.")
        return

    output_dir = (
        _resolve_cli_path(args.output_dir, paths.root)
        if args.output_dir is not None
        else (paths.results / "mocopi_reliability").resolve()
    )

    for pair in pairs:
        _run_reliability_for_pair(
            pair,
            output_dir,
            paths.root,
            args.offset_ms,
            args.search_ms,
            args.rate_hz,
            camera_space=getattr(args, "camera_space", "world"),
        )


def main_reliability_batch(argv: list[str] | None = None) -> None:
    parser = build_reliability_batch_parser()
    args = parser.parse_args(argv)
    run_reliability_batch(args)
