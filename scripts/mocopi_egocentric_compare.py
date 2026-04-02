from __future__ import annotations

"""
Compare Mocopi and camera (MediaPipe) joint trajectories in an egocentric frame.

For a given Mocopi motion source and its corresponding camera landmarks CSV, this script:
    - Computes a simple COM-centered (egocentric) frame for Mocopi joints.
    - Computes an analogous COM-centered frame for camera pose landmarks.
    - Aligns the two time series using a specified or estimated offset.
    - Plots both sets of trajectories on shared time axes.

This is intended to answer: “Do the Mocopi gait/arm swing patterns resemble
what the camera sees?”, independent of where the subject is in the room.

Example:
    python -m scripts.mocopi_egocentric_compare \\
        --motion sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \\
        --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv \\
        --output results/mocopi_camera_egocentric_ND_1a.png
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mocopi import (
    load_mocopi_recording,
    estimate_camera_to_mocopi_offset,
)
from mocopi.features import (
    NoCameraPoseDataError,
    compute_egocentric_positions,
    compute_camera_egocentric_positions,
)
from mocopi.plots import select_overlap_window, plot_egocentric_compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Mocopi and camera joint trajectories in an egocentric frame."
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


def select_aligned_window(
    t_m_ms: np.ndarray,
    t_c_ms_aligned: np.ndarray,
    t_start: float | None,
    t_end: float | None,
) -> tuple[np.ndarray, tuple[float, float]]:
    window_start, window_end = select_overlap_window(t_m_ms, t_c_ms_aligned, t_start, t_end)
    return np.array([window_start, window_end], dtype=float), (window_start, window_end)


def main() -> None:
    args = parse_args()
    if len(args.joints) != len(args.landmarks):
        raise SystemExit("Expected --joints and --landmarks to have the same length")

    seq = load_mocopi_recording(args.motion_source)
    cam_df = pd.read_csv(args.camera_csv)

    # Egocentric Mocopi positions
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, args.joints)

    # Egocentric camera positions
    try:
        t_c_ms_raw, camera_pos = compute_camera_egocentric_positions(cam_df, args.landmarks)
    except NoCameraPoseDataError as exc:
        raise SystemExit(
            f"Camera CSV has no usable pose landmarks for egocentric comparison: {args.camera_csv}"
        ) from exc

    # Determine or use offset
    try:
        offset_ms = estimate_camera_to_mocopi_offset(
            seq,
            cam_df,
            args.search_ms,
            args.rate_hz,
            args.offset_ms,
        )
    except NoCameraPoseDataError as exc:
        raise SystemExit(
            f"Camera CSV has no usable pose landmarks for offset estimation: {args.camera_csv}"
        ) from exc
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
