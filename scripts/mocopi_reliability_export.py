from __future__ import annotations

"""Export per-frame Mocopi vs MediaPipe egocentric errors."""

import argparse
from pathlib import Path

import pandas as pd

from mocopi import load_bvh
from mocopi.reliability import export_reliability_errors
from mocopi.sync import estimate_camera_to_mocopi_offset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-frame Mocopi vs MediaPipe egocentric errors."
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
        help="Optional camera→mocopi offset in ms. If omitted, estimated from r_hand/RIGHT_WRIST.",
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
        "--output",
        type=Path,
        default=Path("results/mocopi_camera_reliability.csv"),
        help="Output CSV path.",
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
    return parser.parse_args(argv)


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    seq = load_bvh(args.bvh)
    cam_df = pd.read_csv(args.camera_csv)

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"Wrote {len(df_out)} rows to {args.output}")


if __name__ == "__main__":
    main()
