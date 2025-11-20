from __future__ import annotations

"""
Compare Mocopi and camera (MediaPipe) joint trajectories in an egocentric frame.

For a given BVH file and its corresponding camera landmarks CSV, this script:
    - Computes a simple COM-centered (egocentric) frame for Mocopi joints.
    - Computes an analogous COM-centered frame for camera pose landmarks.
    - Aligns the two time series using a specified or estimated offset.
    - Plots both sets of trajectories on shared time axes.

This is intended to answer: “Do the Mocopi gait/arm swing patterns resemble
what the camera sees?”, independent of where the subject is in the room.

Example:
    python -m scripts.mocopi_egocentric_compare \\
        --bvh sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \\
        --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv \\
        --output results/mocopi_camera_egocentric_ND_1a.png
"""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mocopi import (
    load_bvh,
    estimate_time_offset,
)
from mocopi.features import (
    compute_egocentric_positions,
    compute_camera_egocentric_positions,
    resample_feature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Mocopi and camera joint trajectories in an egocentric frame."
    )
    parser.add_argument("--bvh", type=Path, required=True, help="Path to Mocopi BVH file.")
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
        help="Mocopi joint names to visualize (default: l_foot r_foot l_hand r_hand).",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=["LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_WRIST", "RIGHT_WRIST"],
        help="Camera pose landmark names corresponding to the joints (same order).",
    )
    parser.add_argument(
        "--offset_ms",
        type=float,
        default=None,
        help="Optional camera→mocopi offset in ms. If omitted, estimated from features.",
    )
    parser.add_argument(
        "--search_ms",
        type=float,
        default=5000.0,
        help="Search range for offset estimation in ms (± value).",
    )
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=50.0,
        help="Resampling rate (Hz) used when estimating offset.",
    )
    parser.add_argument(
        "--t_start",
        type=float,
        default=None,
        help="Optional plot start time (seconds) in the aligned timeline.",
    )
    parser.add_argument(
        "--t_end",
        type=float,
        default=None,
        help="Optional plot end time (seconds) in the aligned timeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mocopi_camera_egocentric.png"),
        help="Path to output PNG file.",
    )
    return parser.parse_args()


def compute_or_use_offset(
    seq,
    cam_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
    offset_ms: float | None,
) -> float:
    """
    Either reuse a provided offset or compute one from scratch using
    egocentric, scale-normalized r_hand / RIGHT_WRIST vertical motion.
    """
    if offset_ms is not None:
        print(f"Using provided offset_ms={offset_ms:.1f}")
        return offset_ms

    print("Estimating offset via feature correlation...")
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, ["r_hand"])
    if "r_hand" not in mocopi_pos:
        raise RuntimeError("Joint 'r_hand' not found in Mocopi sequence")
    f_m = mocopi_pos["r_hand"][:, 1]  # ΔY in egocentric frame

    t_c_ms, camera_pos = compute_camera_egocentric_positions(cam_df, ["RIGHT_WRIST"])
    if "RIGHT_WRIST" not in camera_pos:
        raise RuntimeError("Landmark 'RIGHT_WRIST' not found in camera CSV")
    f_c = camera_pos["RIGHT_WRIST"][:, 1]  # ΔY in egocentric frame

    t_m_res, f_m_res = resample_feature(t_m_ms, f_m, rate_hz)
    t_c_res, f_c_res = resample_feature(t_c_ms, f_c, rate_hz)

    best_offset, best_score = estimate_time_offset(
        t_m_res,
        f_m_res,
        t_c_res,
        f_c_res,
        search_range_ms=search_ms,
        step_ms=10.0,
    )

    print(f"Estimated offset (camera → mocopi): {best_offset:.1f} ms")
    print(f"Correlation score at best offset:   {best_score:.3f}")

    # Report overlap used
    t_b_shifted = t_c_res + best_offset
    t_min = max(t_m_res[0], t_b_shifted[0])
    t_max = min(t_m_res[-1], t_b_shifted[-1])
    if t_max > t_min:
        mask = (t_m_res >= t_min) & (t_m_res <= t_max)
        overlap_samples = int(mask.sum())
        overlap_seconds = (t_max - t_min) / 1000.0
        print(f"Overlap at best offset: {overlap_seconds:.2f} s over {overlap_samples} samples")
    else:
        print("Overlap at best offset: none (no shared time window)")

    return best_offset


def select_aligned_window(
    t_m_ms: np.ndarray,
    t_c_ms_aligned: np.ndarray,
    t_start: float | None,
    t_end: float | None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Determine a shared time window for plotting in seconds, based on:
        - Mocopi time range
        - Camera (aligned) time range
        - Optional user-specified [t_start, t_end]
    """
    t_m = t_m_ms / 1000.0
    t_c = t_c_ms_aligned / 1000.0

    window_start = max(t_m[0], t_c[0])
    window_end = min(t_m[-1], t_c[-1])

    if t_start is not None:
        window_start = max(window_start, t_start)
    if t_end is not None:
        window_end = min(window_end, t_end)

    if window_end <= window_start:
        raise RuntimeError("No overlapping time window for Mocopi and camera after alignment")

    return np.array([window_start, window_end], dtype=float), (window_start, window_end)


def plot_egocentric_compare(
    t_m_ms: np.ndarray,
    mocopi_pos: dict[str, np.ndarray],
    t_c_ms_aligned: np.ndarray,
    camera_pos: dict[str, np.ndarray],
    joints: Sequence[str],
    landmarks: Sequence[str],
    window: tuple[float, float],
    output_path: Path,
) -> None:
    t_m = t_m_ms / 1000.0
    t_c = t_c_ms_aligned / 1000.0
    t_start, t_end = window

    mask_m = (t_m >= t_start) & (t_m <= t_end)
    mask_c = (t_c >= t_start) & (t_c <= t_end)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_m = axes[0]
    ax_c = axes[1]

    # Mocopi vertical motion (relative Y)
    for name in joints:
        if name not in mocopi_pos:
            continue
        traj = mocopi_pos[name][mask_m]
        ax_m.plot(t_m[mask_m], traj[:, 1], label=name)
    ax_m.set_ylabel("Mocopi ΔY (egocentric)")
    ax_m.set_title("Mocopi egocentric vertical motion")
    ax_m.grid(True, alpha=0.3)
    ax_m.legend(loc="upper right", fontsize=8)

    # Camera vertical motion (relative Y)
    for name in landmarks:
        if name not in camera_pos:
            continue
        traj = camera_pos[name][mask_c]
        ax_c.plot(t_c[mask_c], traj[:, 1], label=name)
    ax_c.set_ylabel("Camera ΔY (egocentric)")
    ax_c.set_xlabel("Time (s, aligned)")
    ax_c.set_title("Camera egocentric vertical motion")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.joints) != len(args.landmarks):
        raise SystemExit("Expected --joints and --landmarks to have the same length")

    seq = load_bvh(args.bvh)
    cam_df = pd.read_csv(args.camera_csv)

    # Egocentric Mocopi positions
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, args.joints)

    # Egocentric camera positions
    t_c_ms_raw, camera_pos = compute_camera_egocentric_positions(cam_df, args.landmarks)

    # Determine or use offset
    offset_ms = compute_or_use_offset(seq, cam_df, args.search_ms, args.rate_hz, args.offset_ms)
    t_c_ms_aligned = t_c_ms_raw + offset_ms

    # Determine overlapping time window in aligned seconds
    window_array, window = select_aligned_window(t_m_ms, t_c_ms_aligned, args.t_start, args.t_end)
    print(f"Aligned plot window: {window[0]:.2f}–{window[1]:.2f} s")

    plot_egocentric_compare(
        t_m_ms,
        mocopi_pos,
        t_c_ms_aligned,
        camera_pos,
        args.joints,
        args.landmarks,
        window,
        args.output,
    )
    print(f"Egocentric comparison plot saved to {args.output}")


if __name__ == "__main__":
    main()
