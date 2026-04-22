from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mocopi.reliability import (
    RELIABILITY_COLUMNS,
    align_pose_counts,
    best_joint_from_reliability,
    default_comparison_components,
    ensure_reliability_csv,
    nd_factor_from_stem,
)
from patientpose.artifacts import ArtifactStore, PairReportArtifacts
from patientpose.config import resolve_project_paths
from patientpose.datasets import discover_pairs, discover_sessions, infer_camera_csv, parse_camera_role_specs
from patientpose.landmarks import infer_pose_world_csv
from patientpose.reporting import (
    body_angle_trace_table,
    compute_body_angle_trace,
    compute_landmark_metric_trace,
    export_landmark_metric_batch,
    metric_trace_table,
    summarize_body_angle_trace,
    summarize_landmark_metric_trace,
)


def add_pair_report_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument("--tags", nargs="+", help="Optional subset of tags to process (e.g., 1a 1b).")
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
    )
    parser.add_argument("--offset_ms", type=float, default=None, help="Optional fixed offset to reuse for both A and ND.")
    parser.add_argument("--search_ms", type=float, default=5000.0, help="Search range for offset estimation.")
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Resample rate for offset estimation.")
    parser.add_argument(
        "--camera-space",
        choices=("image", "world"),
        default="world",
        help="Which camera pose representation to report against Mocopi.",
    )
    parser.add_argument(
        "--plot-component",
        choices=("x", "y", "z"),
        default=None,
        help="Component to plot in the pair trace panels. Defaults to z for world and y for image.",
    )
    parser.add_argument(
        "--clip_start",
        type=float,
        default=None,
        help="Optional start time (s) to include in offset estimation/plots.",
    )
    parser.add_argument(
        "--clip_end",
        type=float,
        default=None,
        help="Optional end time (s) to include in offset estimation/plots.",
    )
    return parser


def add_body_angle_export_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--camera_csv",
        type=Path,
        default=None,
        help="Direct path to the camera landmarks CSV to export.",
    )
    source_group.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Resolved pair tag to export from.",
    )
    parser.add_argument(
        "--camera-side",
        choices=("A", "ND"),
        default="ND",
        help="Which camera side to use when resolving from --tag.",
    )
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
    )
    parser.add_argument(
        "--space",
        choices=("image", "world"),
        default="image",
        help="Which pose representation to analyze when computing body angle.",
    )
    parser.add_argument(
        "--world_csv",
        type=Path,
        default=None,
        help="Optional explicit path to the pose-world CSV. Defaults to the paired preprocess artifact.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.4,
        help="Visibility threshold applied before body-axis estimation.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Centered smoothing window in frames for scale and body axes.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the long-form body-angle trace CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the body-angle summary CSV.",
    )
    return parser


LANDMARK_METRIC_CHOICES = (
    "distance",
    "delta",
    "centroid-distance",
    "angle",
    "thumb-index-distance",
    "pinch-velocity",
)


def _resolve_cli_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path).resolve()


def _add_landmark_metric_option_args(
    parser: argparse.ArgumentParser,
    *,
    include_world_csv: bool,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--source",
        choices=("hand", "pose"),
        default="hand",
        help="Landmark source to extract from the CSV.",
    )
    parser.add_argument(
        "--space",
        choices=("image", "world"),
        default="image",
        help="Which landmark representation to analyze.",
    )
    if include_world_csv:
        parser.add_argument(
            "--world_csv",
            type=Path,
            default=None,
            help="Optional explicit path to the pose-world CSV. Defaults to the paired preprocess artifact.",
        )
    parser.add_argument(
        "--handedness",
        type=str,
        default=None,
        help="Optional handedness filter, e.g. Right or Left.",
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Optional instance id filter for multi-instance landmark sources.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=None,
        help="Landmark names or ids to extract.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Components to use. Choose from x y z.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered rolling smoothing window in frames.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="Optional minimum quality_score. Samples below it are set to NaN before metric computation.",
    )
    parser.add_argument(
        "--metric",
        choices=LANDMARK_METRIC_CHOICES,
        default="distance",
        help="Derived metric to export.",
    )
    parser.add_argument(
        "--delta-component",
        choices=("x", "y", "z"),
        default=None,
        help="Component to use for the delta metric. Defaults to the first component in --components.",
    )
    return parser


def add_landmark_metric_export_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--camera_csv",
        type=Path,
        default=None,
        help="Direct path to the camera landmarks CSV to export.",
    )
    source_group.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Resolved pair tag to export from.",
    )
    parser.add_argument(
        "--camera-side",
        choices=("A", "ND"),
        default="ND",
        help="Which camera side to use when resolving from --tag.",
    )
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
    )
    _add_landmark_metric_option_args(parser, include_world_csv=True)
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the long-form metric trace CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the summary CSV.",
    )
    return parser


def add_landmark_metric_batch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--camera-csvs",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit list of camera landmark CSVs to batch export.",
    )
    source_group.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob pattern, relative to --project-root, for camera landmark CSVs to batch export.",
    )
    _add_landmark_metric_option_args(parser, include_world_csv=False)
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the batch long-form metric trace CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the batch summary CSV.",
    )
    return parser


def build_pair_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-pair reliability plots and ND-A delta summary.")
    return add_pair_report_args(parser)


def _camera_csv_for_video(video_path: Path, project_root: Path) -> Path:
    return (project_root / infer_camera_csv(video_path)).resolve()


def _resolve_landmark_metric_camera_csv(
    *,
    project_root: Path,
    camera_csv: Path | None,
    tag: str | None,
    camera_side: str,
    camera_role_specs: list[str] | None,
) -> Path:
    if camera_csv is not None:
        return _resolve_cli_path(camera_csv, project_root)

    camera_roles = parse_camera_role_specs(camera_role_specs)
    pairs = discover_pairs(project_root, camera_roles=camera_roles)
    pair = next((candidate for candidate in pairs if candidate.tag == tag), None)
    if pair is None:
        sessions = discover_sessions(project_root)
        hint = ""
        if sessions and not camera_roles:
            hint = " Add --camera-role CAMERA_ID=A and --camera-role CAMERA_ID=ND for session data."
        raise SystemExit(f"Tag '{tag}' not found in discovered Mocopi/camera pairs.{hint}")

    video_path = pair.unfiltered_video if camera_side == "A" else pair.nd_video
    return _camera_csv_for_video(video_path, project_root)


def _resolve_batch_camera_csvs(
    *,
    project_root: Path,
    camera_csvs: list[Path] | None,
    pattern: str | None,
) -> list[Path]:
    resolved: list[Path] = []
    if camera_csvs:
        resolved = [_resolve_cli_path(path, project_root) for path in camera_csvs]
    elif pattern:
        resolved = sorted(Path(match).resolve() for match in glob(str(project_root / pattern), recursive=True))
    files = [path for path in resolved if path.is_file()]
    if not files:
        raise SystemExit("No landmark CSVs matched the requested batch inputs.")
    return files


def _pair_plot_path(artifacts: PairReportArtifacts, tag: str, joint: str) -> Path:
    return artifacts.plot_dir / f"pair_{tag}_{joint}.pdf"


def plot_pair(
    artifacts: PairReportArtifacts,
    tag: str,
    nd_csv: Path,
    a_csv: Path,
    nd_cam_csv: Path,
    a_cam_csv: Path,
    offset_nd: float,
    offset_a: float,
    *,
    plot_component: str | None = None,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> None:
    nd_df = pd.read_csv(nd_csv)
    a_df = pd.read_csv(a_csv)

    camera_space = str(a_df["camera_space"].iloc[0]) if "camera_space" in a_df.columns and not a_df.empty else "image"
    if plot_component is None:
        plot_component = "z" if camera_space == "world" else "y"

    joint = best_joint_from_reliability(a_csv, component=plot_component)
    if joint is None:
        print(f"[{tag}] No suitable joint found for plotting.")
        return

    nd_sub = nd_df[nd_df["joint"] == joint]
    a_sub = a_df[a_df["joint"] == joint]
    if nd_sub.empty or a_sub.empty:
        print(f"[{tag}] Missing data for joint {joint}")
        return

    t = nd_sub["time_s"].to_numpy()
    if clip_start_s is not None or clip_end_s is not None:
        if clip_start_s is None:
            clip_start_s = t.min()
        if clip_end_s is None:
            clip_end_s = t.max()
        mask_clip = (t >= clip_start_s) & (t <= clip_end_s)
        nd_sub = nd_sub.loc[mask_clip]
        t = nd_sub["time_s"].to_numpy()

    mocopi_col = {"x": "mocopi_dx", "y": "mocopi_dy", "z": "mocopi_dz"}[plot_component]
    camera_col = {"x": "camera_dx", "y": "camera_dy", "z": "camera_dz"}[plot_component]
    mocopi = nd_sub[mocopi_col].to_numpy()
    nd_traj = nd_sub[camera_col].to_numpy()
    t_a = a_sub["time_s"].to_numpy()
    a_cam = a_sub[camera_col].to_numpy()
    a_traj = np.interp(t, t_a, a_cam, left=np.nan, right=np.nan)

    counts_nd = align_pose_counts(nd_cam_csv, t * 1000.0, offset_nd)
    counts_a = align_pose_counts(a_cam_csv, t * 1000.0, offset_a)

    artifacts.plot_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t, mocopi, label=f"Mocopi d{plot_component}", color="#1f77b4", linewidth=1.5)
    ax1.plot(t, a_traj, label=f"A (unfiltered) d{plot_component}", color="#2ca02c", linewidth=1.2)
    ax1.plot(t, nd_traj, label=f"ND d{plot_component}", color="#d62728", linewidth=1.2)
    ax1.set_ylabel(f"Egocentric d{plot_component}")
    ax1.set_title(f"Tag {tag} - joint {joint} ({camera_space})")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(t, counts_a, label="A pose count", color="#2ca02c", linewidth=1.0)
    ax2.plot(t, counts_nd, label="ND pose count", color="#d62728", linewidth=1.0)
    ax2.set_ylabel("Pose landmarks")
    ax2.set_xlabel("Time (s)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = _pair_plot_path(artifacts, tag, joint)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[{tag}] Saved pair plot to {out_path}")


def run_landmark_metric_export(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    camera_csv_path = _resolve_landmark_metric_camera_csv(
        project_root=paths.root,
        camera_csv=args.camera_csv,
        tag=args.tag,
        camera_side=args.camera_side,
        camera_role_specs=args.camera_role,
    )
    try:
        result = compute_landmark_metric_trace(
            camera_csv_path,
            metric=args.metric,
            source=args.source,
            space=args.space,
            project_root=paths.root,
            world_csv=_resolve_cli_path(args.world_csv, paths.root) if args.world_csv is not None else None,
            handedness=args.handedness,
            instance_id=args.instance_id,
            landmarks=args.landmarks,
            components=args.components,
            delta_component=args.delta_component,
            quality_threshold=args.quality_threshold,
            smooth_window=args.smooth_window,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    artifacts = artifact_store.landmark_metric_report(
        result.stem,
        source=args.source,
        space=args.space,
        metric=args.metric,
    )
    trace_output = _resolve_cli_path(args.trace_output, paths.root) if args.trace_output is not None else artifacts.trace_csv
    summary_output = (
        _resolve_cli_path(args.summary_output, paths.root) if args.summary_output is not None else artifacts.summary_csv
    )

    trace_df = metric_trace_table(result)
    summary_df = summarize_landmark_metric_trace(result)

    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_output, index=False)
    summary_df.to_csv(summary_output, index=False)
    print(f"Wrote landmark metric trace CSV to {trace_output}")
    print(f"Wrote landmark metric summary CSV to {summary_output}")


def run_body_angle_export(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    camera_csv_path = _resolve_landmark_metric_camera_csv(
        project_root=paths.root,
        camera_csv=args.camera_csv,
        tag=args.tag,
        camera_side=args.camera_side,
        camera_role_specs=args.camera_role,
    )
    try:
        result = compute_body_angle_trace(
            camera_csv_path,
            space=args.space,
            project_root=paths.root,
            world_csv=_resolve_cli_path(args.world_csv, paths.root) if args.world_csv is not None else None,
            visibility_threshold=args.visibility_threshold,
            smooth_window=args.smooth_window,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    artifacts = artifact_store.body_angle_report(
        result.stem,
        space=args.space,
    )
    trace_output = _resolve_cli_path(args.trace_output, paths.root) if args.trace_output is not None else artifacts.trace_csv
    summary_output = (
        _resolve_cli_path(args.summary_output, paths.root) if args.summary_output is not None else artifacts.summary_csv
    )

    trace_df = body_angle_trace_table(result)
    summary_df = summarize_body_angle_trace(result)

    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_output, index=False)
    summary_df.to_csv(summary_output, index=False)
    print(f"Wrote body-angle trace CSV to {trace_output}")
    print(f"Wrote body-angle summary CSV to {summary_output}")


def run_landmark_metric_batch(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    camera_csvs = _resolve_batch_camera_csvs(
        project_root=paths.root,
        camera_csvs=args.camera_csvs,
        pattern=args.glob,
    )
    try:
        trace_df, summary_df = export_landmark_metric_batch(
            camera_csvs,
            metric=args.metric,
            source=args.source,
            space=args.space,
            project_root=paths.root,
            handedness=args.handedness,
            instance_id=args.instance_id,
            landmarks=args.landmarks,
            components=args.components,
            delta_component=args.delta_component,
            quality_threshold=args.quality_threshold,
            smooth_window=args.smooth_window,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    artifacts = artifact_store.landmark_metric_batch_report(
        source=args.source,
        space=args.space,
        metric=args.metric,
    )
    trace_output = _resolve_cli_path(args.trace_output, paths.root) if args.trace_output is not None else artifacts.trace_csv
    summary_output = (
        _resolve_cli_path(args.summary_output, paths.root) if args.summary_output is not None else artifacts.summary_csv
    )

    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_output, index=False)
    summary_df.to_csv(summary_output, index=False)
    print(f"Wrote batch landmark metric trace CSV to {trace_output}")
    print(f"Wrote batch landmark metric summary CSV to {summary_output}")


def run_pair_report(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()
    artifacts = artifact_store.pair_report(args.camera_space)

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

    summary_records = []

    for pair in pairs:
        nd_stem = pair.nd_video.stem
        a_stem = pair.unfiltered_video.stem
        nd_camera_csv = _camera_csv_for_video(pair.nd_video, paths.root)
        a_camera_csv = _camera_csv_for_video(pair.unfiltered_video, paths.root)
        nd_projection_csv = nd_camera_csv if args.camera_space == "image" else infer_pose_world_csv(nd_camera_csv, paths.root)
        a_projection_csv = a_camera_csv if args.camera_space == "image" else infer_pose_world_csv(a_camera_csv, paths.root)

        nd_output = artifacts.output_dir / f"mocopi_camera_reliability_{args.camera_space}_{nd_stem}.csv"
        a_output = artifacts.output_dir / f"mocopi_camera_reliability_{args.camera_space}_{a_stem}.csv"

        if not nd_camera_csv.exists() or not a_camera_csv.exists() or not nd_projection_csv.exists() or not a_projection_csv.exists():
            print(
                f"[{pair.tag}] Missing camera CSVs; ND image={nd_camera_csv.exists()} ND {args.camera_space}={nd_projection_csv.exists()} "
                f"A image={a_camera_csv.exists()} A {args.camera_space}={a_projection_csv.exists()}"
            )
            continue

        ensure_reliability_csv(
            pair.motion_source,
            nd_projection_csv,
            nd_output,
            args.offset_ms,
            args.search_ms,
            args.rate_hz,
            offset_camera_csv=nd_camera_csv,
            offset_world_csv=nd_projection_csv if args.camera_space == "world" else None,
            camera_space=args.camera_space,
            comparison_components=default_comparison_components(args.camera_space),
            clip_start_s=args.clip_start,
            clip_end_s=args.clip_end,
        )
        ensure_reliability_csv(
            pair.motion_source,
            a_projection_csv,
            a_output,
            args.offset_ms,
            args.search_ms,
            args.rate_hz,
            offset_camera_csv=a_camera_csv,
            offset_world_csv=a_projection_csv if args.camera_space == "world" else None,
            camera_space=args.camera_space,
            comparison_components=default_comparison_components(args.camera_space),
            clip_start_s=args.clip_start,
            clip_end_s=args.clip_end,
        )

        offset_nd = args.offset_ms if args.offset_ms is not None else None
        offset_a = args.offset_ms if args.offset_ms is not None else None

        nd_df = pd.read_csv(nd_output)
        a_df = pd.read_csv(a_output)
        if (list(nd_df.columns) == RELIABILITY_COLUMNS and nd_df.empty) or (
            list(a_df.columns) == RELIABILITY_COLUMNS and a_df.empty
        ):
            missing = []
            if list(nd_df.columns) == RELIABILITY_COLUMNS and nd_df.empty:
                missing.append("ND")
            if list(a_df.columns) == RELIABILITY_COLUMNS and a_df.empty:
                missing.append("A")
            print(f"[{pair.tag}] No usable pose landmarks for {', '.join(missing)} camera CSV; skipping pair report.")
            continue

        med_nd = nd_df.groupby("joint")["error_2d"].median()
        med_a = a_df.groupby("joint")["error_2d"].median()
        nd_level = nd_factor_from_stem(nd_stem)
        for joint in med_nd.index.intersection(med_a.index):
            err_nd = med_nd[joint]
            err_a = med_a[joint]
            delta = err_nd - err_a
            ratio = err_nd / err_a if err_a and abs(err_a) > 1e-6 else np.nan
            summary_records.append(
                {
                    "tag": pair.tag,
                    "joint": joint,
                    "camera_space": args.camera_space,
                    "nd": nd_level,
                    "error_nd": err_nd,
                    "error_a": err_a,
                    "delta_error": delta,
                    "ratio_error": ratio,
                }
            )

        plot_pair(
            artifacts,
            pair.tag,
            nd_output,
            a_output,
            nd_camera_csv,
            a_camera_csv,
            offset_nd or 0.0,
            offset_a or 0.0,
            plot_component=args.plot_component,
            clip_start_s=args.clip_start,
            clip_end_s=args.clip_end,
        )

    if summary_records:
        summary_df = pd.DataFrame.from_records(summary_records)
        artifacts.output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(artifacts.summary_csv, index=False)
        print(f"Wrote summary to {artifacts.summary_csv}")

        fig, ax = plt.subplots(figsize=(6, 4))
        for joint, sub in summary_df.groupby("joint"):
            sub = sub.sort_values("nd")
            ax.plot(sub["nd"], sub["ratio_error"], marker="o", label=joint)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("ND factor (log2)")
        ax.set_ylabel("Error ratio (ND / A)")
        ax.set_title(f"ND-induced change in body-scale error ({args.camera_space})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(artifacts.ratio_plot)
        plt.close(fig)
        print(f"Saved ratio summary plot to {artifacts.ratio_plot}")
    else:
        print("No summary data produced.")


def main_pair_report(argv: list[str] | None = None) -> None:
    parser = build_pair_report_parser()
    args = parser.parse_args(argv)
    run_pair_report(args)


if __name__ == "__main__":
    main_pair_report()
