from __future__ import annotations

"""
Render a three-panel video for each A/ND/BVH triplet:
  - Left: unfiltered A_* video with pose overlay
  - Middle: ND_* video with pose overlay
  - Right: Mocopi skeleton (scaled to travel span)

Assumes:
  - Videos in sample_data/ND_pilot (A_* and ND_*).
  - BVH in sample_data/ND_pilot/Re_ Mocopi with matching tag.
  - Camera CSVs in results/OutputCSVs/landmarks_<stem>.csv.

Example:
    python -m scripts.mocopi_triplet_video --tags 1a 1b
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from scripts.mocopi_pair_utils import discover_pairs
from scripts.mocopi_pair_utils import TrialPair
from mocopi import load_bvh, estimate_camera_to_mocopi_offset
from mocopi.visualization import (
    prepare_camera_landmarks,
    draw_camera_skeleton,
    prepare_mocopi_positions,
    draw_mocopi_skeleton,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-panel video for A/ND/BVH triplets.")
    parser.add_argument("--tags", nargs="+", help="Optional subset of tags to process (e.g., 1a 1b).")
    parser.add_argument("--offset_ms", type=float, default=None, help="Optional fixed offset to reuse for both A and ND.")
    parser.add_argument("--search_ms", type=float, default=5000.0, help="Search range for offset estimation.")
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Resample rate for offset estimation.")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=Path("results/OutputVideos/triplets"),
        help="Directory for triplet videos.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    pairs = discover_pairs(base)
    if args.tags:
        tags = set(args.tags)
        pairs = [p for p in pairs if p.tag in tags]
    if not pairs:
        print("No matching ND/A/BVH pairs found under sample_data/ND_pilot.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        nd_stem = pair.nd_video.stem
        a_stem = pair.unfiltered_video.stem
        nd_csv = Path("results/OutputCSVs") / f"landmarks_{nd_stem}.csv"
        a_csv = Path("results/OutputCSVs") / f"landmarks_{a_stem}.csv"
        if not nd_csv.exists() or not a_csv.exists():
            print(f"[{pair.tag}] Missing camera CSVs; skipping.")
            continue

        seq = load_bvh(pair.bvh)
        nd_df = pd.read_csv(nd_csv)
        a_df = pd.read_csv(a_csv)

        if args.offset_ms is not None:
            offset_ms = args.offset_ms
        else:
            offset_ms = estimate_camera_to_mocopi_offset(
                seq,
                nd_df,
                args.search_ms,
                args.rate_hz,
                None,
            )
            print(f"[{pair.tag}] Estimated offset {offset_ms:.1f} ms")

        # Prepare mocopi positions
        t_m_ms, mocopi_positions = prepare_mocopi_positions(seq)

        # Camera landmarks per frame
        nd_landmarks = prepare_camera_landmarks(nd_df)
        a_landmarks = prepare_camera_landmarks(a_df)

        cap_nd = cv2.VideoCapture(str(pair.nd_video))
        cap_a = cv2.VideoCapture(str(pair.unfiltered_video))
        if not cap_nd.isOpened() or not cap_a.isOpened():
            print(f"[{pair.tag}] Could not open videos; skipping.")
            continue

        width_nd = int(cap_nd.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height_nd = int(cap_nd.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        width_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap_nd.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap_nd.get(cv2.CAP_PROP_FRAME_COUNT)) or len(nd_landmarks)
        frame_count = min(frame_count, int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)) or frame_count)

        out_w = width_a + width_nd + width_nd  # mocopi panel same size as ND
        out_h = max(height_a, height_nd)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = args.output_dir / f"triplet_{nd_stem}.avi"
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
        if not out.isOpened():
            print(f"[{pair.tag}] Could not open writer; skipping.")
            continue

        print(f"[{pair.tag}] Rendering {frame_count} frames to {out_path}")
        for frame_idx in range(frame_count):
            ret_nd, frame_nd = cap_nd.read()
            ret_a, frame_a = cap_a.read()
            if not ret_nd or not ret_a:
                break

            left = frame_a.copy()
            lms_a = a_landmarks.get(frame_idx)
            if lms_a:
                draw_camera_skeleton(left, lms_a)

            middle = frame_nd.copy()
            lms_nd = nd_landmarks.get(frame_idx)
            if lms_nd:
                draw_camera_skeleton(middle, lms_nd)

            # Timestamp from ND CSV if available
            t_cam_ms = None
            if lms_nd:
                any_name = next(iter(lms_nd.keys()))
                row = nd_df[(nd_df["frame"] == frame_idx) & (nd_df["landmark_name"] == any_name)]
                if not row.empty:
                    t_cam_ms = float(row["timestamp_ms"].iloc[0])
            if t_cam_ms is None:
                t_cam_ms = frame_idx * 1000.0 / fps
            t_mocopi_ms = t_cam_ms + offset_ms

            right = np.zeros_like(middle)
            draw_mocopi_skeleton(right, mocopi_positions, t_mocopi_ms, t_m_ms)

            combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            combined[:, :width_a] = cv2.resize(left, (width_a, out_h))
            combined[:, width_a : width_a + width_nd] = cv2.resize(middle, (width_nd, out_h))
            combined[:, width_a + width_nd :] = cv2.resize(right, (width_nd, out_h))
            out.write(combined)

        cap_nd.release()
        cap_a.release()
        out.release()
        print(f"[{pair.tag}] Saved triplet video to {out_path}")


if __name__ == "__main__":
    main()
