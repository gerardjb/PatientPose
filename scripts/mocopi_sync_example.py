from __future__ import annotations

"""
Quick-and-dirty driver to experiment with Mocopi / camera synchronization.

Usage (example):
    python -m scripts.mocopi_sync_example \\
        --bvh sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \\
        --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from mocopi import (
    load_bvh,
    estimate_time_offset,
)
from mocopi.features import (
    resample_feature,
    compute_egocentric_positions,
    compute_camera_egocentric_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mocopi / camera sync demo.")
    parser.add_argument("--bvh", type=Path, required=True, help="Path to Mocopi BVH file.")
    parser.add_argument(
        "--camera_csv",
        type=Path,
        required=True,
        help="Path to camera landmarks CSV (results/OutputCSVs).",
    )
    parser.add_argument(
        "--search_ms",
        type=float,
        default=5000.0,
        help="Search range for offset in milliseconds (± value).",
    )
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=50.0,
        help="Resampling rate in Hz for uniform comparison.",
    )
    args = parser.parse_args()

    seq = load_bvh(args.bvh)
    cam_df = pd.read_csv(args.camera_csv)

    # Use egocentric, scale-normalized vertical motion for alignment
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, ["r_hand"])
    if "r_hand" not in mocopi_pos:
        raise RuntimeError("Joint 'r_hand' not found in Mocopi sequence")
    f_m = mocopi_pos["r_hand"][:, 1]  # ΔY in egocentric frame

    t_c_ms, camera_pos = compute_camera_egocentric_positions(cam_df, ["RIGHT_WRIST"])
    if "RIGHT_WRIST" not in camera_pos:
        raise RuntimeError("Landmark 'RIGHT_WRIST' not found in camera CSV")
    f_c = camera_pos["RIGHT_WRIST"][:, 1]  # ΔY in egocentric frame

    # Resample both to a common grid to stabilize correlation.
    t_m_res, f_m_res = resample_feature(t_m_ms, f_m, args.rate_hz)
    t_c_res, f_c_res = resample_feature(t_c_ms, f_c, args.rate_hz)

    best_offset, best_score = estimate_time_offset(
        t_m_res,
        f_m_res,
        t_c_res,
        f_c_res,
        search_range_ms=args.search_ms,
        step_ms=10.0,
    )

    print(f"Estimated offset (camera → mocopi): {best_offset:.1f} ms")
    print(f"Correlation score at best offset:   {best_score:.3f}")

    # Report how much data actually contributed to that correlation.
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


if __name__ == "__main__":
    main()
