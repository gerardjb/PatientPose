from __future__ import annotations

"""
Generate per-pair Mocopi vs MediaPipe reliability plots and a delta summary.

For each ND/A/Mocopi triplet discovered under sample_data, this script:
  - Ensures a reliability CSV exists for the ND and A recordings (runs mocopi_reliability_export if missing).
  - Finds the joint with the highest |corr| between Mocopi and the A (unfiltered) camera trace.
  - Plots Mocopi vs A vs ND egocentric ΔY over time for that joint, plus visibility (if available) for A and ND.
  - Aggregates median errors for ND and A and writes/plots ND minus A error per joint vs ND level.

Outputs are written to results/mocopi_reliability/.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mocopi.nd_pilot import TrialPair, discover_pairs, discover_sessions, parse_camera_role_specs
from mocopi.reliability import (
    RELIABILITY_COLUMNS,
    nd_factor_from_stem,
    ensure_reliability_csv,
    best_joint_from_reliability,
    align_visibility_series,
    align_pose_counts,
)

RELIABILITY_DIR = Path("results/mocopi_reliability")
PLOT_DIR = RELIABILITY_DIR / "plots"

# Default joint/landmark pairs used by mocopi_reliability_export
PAIR_JOINTS = ["l_foot", "r_foot", "l_hand", "r_hand"]
PAIR_LANDMARKS = ["LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_WRIST", "RIGHT_WRIST"]


def plot_pair(
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

    # Choose best joint from A vs Mocopi
    joint = best_joint_from_reliability(a_csv)
    if joint is None:
        print(f"[{tag}] No suitable joint found for plotting.")
        return

    # Extract time and trajectories
    nd_sub = nd_df[nd_df["joint"] == joint]
    a_sub = a_df[a_df["joint"] == joint]
    if nd_sub.empty or a_sub.empty:
        print(f"[{tag}] Missing data for joint {joint}")
        return

    t = nd_sub["time_s"].to_numpy()
    # Optional clip window
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
    # Interpolate A camera trajectory onto ND/Mocopi time grid
    t_a = a_sub["time_s"].to_numpy()
    a_cam = a_sub["camera_dy"].to_numpy()
    a_traj = np.interp(t, t_a, a_cam, left=np.nan, right=np.nan)

    # Pose landmark counts aligned to the ND/Mocopi timeline
    counts_nd = align_pose_counts(nd_cam_csv, t * 1000.0, offset_nd)
    counts_a = align_pose_counts(a_cam_csv, t * 1000.0, offset_a)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t, mocopi, label="Mocopi ΔY", color="#1f77b4", linewidth=1.5)
    ax1.plot(t, a_traj, label="A (unfiltered) ΔY", color="#2ca02c", linewidth=1.2)
    ax1.plot(t, nd_traj, label="ND ΔY", color="#d62728", linewidth=1.2)
    ax1.set_ylabel("Egocentric ΔY (body-scale)")
    ax1.set_title(f"Tag {tag} – joint {joint}")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(t, counts_a, label="A pose count", color="#2ca02c", linewidth=1.0)
    ax2.plot(t, counts_nd, label="ND pose count", color="#d62728", linewidth=1.0)
    ax2.set_ylabel("Pose landmarks")
    ax2.set_xlabel("Time (s)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = PLOT_DIR / f"pair_{tag}_{joint}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[{tag}] Saved pair plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-pair reliability plots and ND-A delta summary.")
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
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    camera_roles = parse_camera_role_specs(args.camera_role)
    pairs = discover_pairs(base, camera_roles=camera_roles)
    if args.tags:
        tags = set(args.tags)
        pairs = [p for p in pairs if p.tag in tags]
    if not pairs:
        sessions = discover_sessions(base)
        if sessions and not camera_roles:
            print("Discovered session folders, but no session pairs were resolved. Add --camera-role CAMERA_ID=A and --camera-role CAMERA_ID=ND.")
        else:
            print("No matching Mocopi/camera pairs found.")
        return

    summary_records = []

    for pair in pairs:
        nd_stem = pair.nd_video.stem
        a_stem = pair.unfiltered_video.stem
        nd_camera_csv = Path("results/OutputCSVs") / f"landmarks_{nd_stem}.csv"
        a_camera_csv = Path("results/OutputCSVs") / f"landmarks_{a_stem}.csv"

        nd_output = RELIABILITY_DIR / f"mocopi_camera_reliability_{nd_stem}.csv"
        a_output = RELIABILITY_DIR / f"mocopi_camera_reliability_{a_stem}.csv"

        if not nd_camera_csv.exists() or not a_camera_csv.exists():
            print(f"[{pair.tag}] Missing camera CSVs; ND: {nd_camera_csv.exists()}, A: {a_camera_csv.exists()}")
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

        # Offsets may differ if not fixed; recompute based on file contents if needed
        offset_nd = args.offset_ms if args.offset_ms is not None else None
        offset_a = args.offset_ms if args.offset_ms is not None else None

        # Load reliability CSVs
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

        # Median errors per joint
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

        # Plots per pair
        plot_pair(
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
        summary_csv = RELIABILITY_DIR / "nd_delta_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Wrote summary to {summary_csv}")

        # Plot ratio (ND/A) vs ND (log2) per joint
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
        out_plot = RELIABILITY_DIR / "nd_ratio_summary.pdf"
        fig.savefig(out_plot)
        plt.close(fig)
        print(f"Saved ratio summary plot to {out_plot}")
    else:
        print("No summary data produced.")


if __name__ == "__main__":
    main()
