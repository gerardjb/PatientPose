from __future__ import annotations

import argparse
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
    ensure_reliability_csv,
    nd_factor_from_stem,
)
from patientpose.artifacts import ArtifactStore, PairReportArtifacts
from patientpose.config import resolve_project_paths
from patientpose.datasets import discover_pairs, discover_sessions, infer_camera_csv, parse_camera_role_specs


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


def build_pair_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-pair reliability plots and ND-A delta summary."
    )
    return add_pair_report_args(parser)


def _camera_csv_for_video(video_path: Path, project_root: Path) -> Path:
    return (project_root / infer_camera_csv(video_path)).resolve()


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
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> None:
    nd_df = pd.read_csv(nd_csv)
    a_df = pd.read_csv(a_csv)

    joint = best_joint_from_reliability(a_csv)
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

    mocopi = nd_sub["mocopi_dy"].to_numpy()
    nd_traj = nd_sub["camera_dy"].to_numpy()
    t_a = a_sub["time_s"].to_numpy()
    a_cam = a_sub["camera_dy"].to_numpy()
    a_traj = np.interp(t, t_a, a_cam, left=np.nan, right=np.nan)

    counts_nd = align_pose_counts(nd_cam_csv, t * 1000.0, offset_nd)
    counts_a = align_pose_counts(a_cam_csv, t * 1000.0, offset_a)

    artifacts.plot_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t, mocopi, label="Mocopi ΔY", color="#1f77b4", linewidth=1.5)
    ax1.plot(t, a_traj, label="A (unfiltered) ΔY", color="#2ca02c", linewidth=1.2)
    ax1.plot(t, nd_traj, label="ND ΔY", color="#d62728", linewidth=1.2)
    ax1.set_ylabel("Egocentric ΔY (body-scale)")
    ax1.set_title(f"Tag {tag} - joint {joint}")
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


def run_pair_report(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()
    artifacts = artifact_store.pair_report()

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

        nd_output = artifacts.output_dir / f"mocopi_camera_reliability_{nd_stem}.csv"
        a_output = artifacts.output_dir / f"mocopi_camera_reliability_{a_stem}.csv"

        if not nd_camera_csv.exists() or not a_camera_csv.exists():
            print(
                f"[{pair.tag}] Missing camera CSVs; ND: {nd_camera_csv.exists()}, A: {a_camera_csv.exists()}"
            )
            continue

        ensure_reliability_csv(
            pair.motion_source,
            nd_camera_csv,
            nd_output,
            args.offset_ms,
            args.search_ms,
            args.rate_hz,
            clip_start_s=args.clip_start,
            clip_end_s=args.clip_end,
        )
        ensure_reliability_csv(
            pair.motion_source,
            a_camera_csv,
            a_output,
            args.offset_ms,
            args.search_ms,
            args.rate_hz,
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
        ax.set_title("ND-induced change in body-scale error (normalized)")
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
