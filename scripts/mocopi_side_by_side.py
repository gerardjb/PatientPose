from __future__ import annotations

"""
Render a side-by-side video comparing camera keypoints and Mocopi BVH keypoints.

Left panel:
    - Original camera video frames.
    - 2D pose skeleton overlay from a landmarks CSV (results/OutputCSVs).

Right panel:
    - Black background.
    - 2D projection of the Mocopi skeleton, time-aligned via a precomputed
      or estimated offset.

Example:
    python -m scripts.mocopi_side_by_side \\
        --bvh sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \\
        --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv \\
        --video sample_data/ND_1a_20140107_104046.mp4 \\
        --output results/OutputVideos/mocopi_vs_camera_ND_1a.avi
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from mocopi import (
    load_bvh,
    estimate_camera_to_mocopi_offset,
)
from mocopi.visualization import (
    prepare_camera_landmarks,
    draw_camera_skeleton,
    prepare_mocopi_positions,
    draw_mocopi_skeleton,
)


def _infer_video_from_csv(csv_path: Path) -> Path:
    """
    Infer the video path from a landmarks CSV filename.

    Primary target:
        results/OutputVideos/deidentified_ND_1a_20140107_104046.avi

    Fallback if de-identified video is missing:
        sample_data/ND_1a_20140107_104046.mp4
    """
    stem = csv_path.stem
    if stem.startswith("landmarks_"):
        stem = stem[len("landmarks_") :]
    deid = Path("results") / "OutputVideos" / f"deidentified_{stem}.avi"
    if deid.is_file():
        return deid
    return Path("sample_data") / f"{stem}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Mocopi vs camera side-by-side video.")
    parser.add_argument("--bvh", type=Path, required=True, help="Path to Mocopi BVH file.")
    parser.add_argument(
        "--camera_csv",
        type=Path,
        required=True,
        help="Path to camera landmarks CSV (results/OutputCSVs).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to original camera video. If omitted, inferred from CSV name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/OutputVideos/mocopi_vs_camera.avi"),
        help="Path to output side-by-side video.",
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
        help="Resampling rate (Hz) for offset estimation.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to render.",
    )
    args = parser.parse_args()

    bvh_path = args.bvh
    camera_csv_path = args.camera_csv
    video_path = args.video or _infer_video_from_csv(camera_csv_path)
    output_path = args.output

    print(f"BVH:           {bvh_path}")
    print(f"Camera CSV:    {camera_csv_path}")
    print(f"Video source:  {video_path}")
    print(f"Output video:  {output_path}")

    # Load data
    seq = load_bvh(bvh_path)
    cam_df = pd.read_csv(camera_csv_path)

    # Determine offset between Mocopi and camera timelines
    offset_ms = estimate_camera_to_mocopi_offset(
        seq,
        cam_df,
        search_ms=args.search_ms,
        rate_hz=args.rate_hz,
        offset_ms=args.offset_ms,
    )

    camera_by_frame = prepare_camera_landmarks(cam_df)
    t_m_ms, mocopi_positions = prepare_mocopi_positions(seq)

    # Video setup
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(camera_by_frame)

    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    print(f"Rendering {frame_count} frames at {fps:.2f} FPS...")

    # Decide whether to draw an extra skeleton overlay on the left panel:
    # if the source is a de-identified output (already annotated), skip.
    draw_left_overlay = not video_path.name.startswith("deidentified_")

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Left: camera frame (optionally with skeleton overlay)
        left = frame.copy()
        cam_landmarks = camera_by_frame.get(frame_idx)
        if cam_landmarks and draw_left_overlay:
            draw_camera_skeleton(left, cam_landmarks)

        # Compute camera timestamp_ms from CSV if available, otherwise from fps
        if cam_landmarks:
            # Use first row's timestamp for this frame
            # (all landmarks in the frame share the same timestamp_ms)
            any_name = next(iter(cam_landmarks.keys()))
            # Find the row matching this frame and landmark name
            row = cam_df[(cam_df["frame"] == frame_idx) & (cam_df["landmark_name"] == any_name)]
            if not row.empty:
                t_cam_ms = float(row["timestamp_ms"].iloc[0])
            else:
                t_cam_ms = frame_idx * 1000.0 / fps
        else:
            t_cam_ms = frame_idx * 1000.0 / fps

        # Target Mocopi time for this frame: camera timeline shifted by offset
        t_mocopi_ms = t_cam_ms + offset_ms

        # Right: Mocopi skeleton on black background
        right = np.zeros_like(left)
        draw_mocopi_skeleton(right, mocopi_positions, t_mocopi_ms, t_m_ms)

        # Combine side-by-side and write
        combined = np.zeros((height, width * 2, 3), dtype=left.dtype)
        combined[:, :width] = left
        combined[:, width:] = right
        out.write(combined)

    cap.release()
    out.release()
    print("Done.")


if __name__ == "__main__":
    main()
