from __future__ import annotations

"""
Quick-and-dirty driver to experiment with Mocopi / camera synchronization.

Usage (example):
    python -m scripts.mocopi_sync_example \\
        --motion sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \\
        --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from mocopi import (
    load_mocopi_recording,
    estimate_camera_to_mocopi_offset,
)
from mocopi.features import NoCameraPoseDataError


def main() -> None:
    parser = argparse.ArgumentParser(description="Mocopi / camera sync demo.")
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
    parser.add_argument(
        "--clip_start",
        type=float,
        default=None,
        help="Optional start time (s) to include in offset estimation (skip early sit/stand).",
    )
    parser.add_argument(
        "--clip_end",
        type=float,
        default=None,
        help="Optional end time (s) to include in offset estimation (skip late sit/stand).",
    )
    args = parser.parse_args()

    seq = load_mocopi_recording(args.motion_source)
    cam_df = pd.read_csv(args.camera_csv)

    try:
        best_offset = estimate_camera_to_mocopi_offset(
            seq,
            cam_df,
            args.search_ms,
            args.rate_hz,
            None,
            clip_start_s=args.clip_start,
            clip_end_s=args.clip_end,
        )
    except NoCameraPoseDataError as exc:
        raise SystemExit(
            f"Camera CSV has no usable pose landmarks, so offset estimation cannot run: {args.camera_csv}"
        ) from exc
    print(f"Estimated offset (camera → mocopi): {best_offset:.1f} ms")


if __name__ == "__main__":
    main()
