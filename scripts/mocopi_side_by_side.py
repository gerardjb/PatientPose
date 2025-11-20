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
    estimate_time_offset,
)
from mocopi.features import (
    resample_feature,
    compute_egocentric_positions,
    compute_camera_egocentric_positions,
)


# Simple, readable skeleton definitions
CAMERA_EDGES = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("NOSE", "LEFT_EYE"),
    ("NOSE", "RIGHT_EYE"),
    ("LEFT_EYE", "LEFT_EAR"),
    ("RIGHT_EYE", "RIGHT_EAR"),
]

MOCOPI_JOINTS = [
    "torso_7",
    "neck_1",
    "neck_2",
    "head",
    "l_shoulder",
    "l_up_arm",
    "l_low_arm",
    "l_hand",
    "r_shoulder",
    "r_up_arm",
    "r_low_arm",
    "r_hand",
    "l_up_leg",
    "l_low_leg",
    "l_foot",
    "l_toes",
    "r_up_leg",
    "r_low_leg",
    "r_foot",
    "r_toes",
]

MOCOPI_EDGES = [
    ("torso_7", "neck_1"),
    ("neck_1", "neck_2"),
    ("neck_2", "head"),
    ("torso_7", "l_shoulder"),
    ("l_shoulder", "l_up_arm"),
    ("l_up_arm", "l_low_arm"),
    ("l_low_arm", "l_hand"),
    ("torso_7", "r_shoulder"),
    ("r_shoulder", "r_up_arm"),
    ("r_up_arm", "r_low_arm"),
    ("r_low_arm", "r_hand"),
    ("torso_7", "l_up_leg"),
    ("l_up_leg", "l_low_leg"),
    ("l_low_leg", "l_foot"),
    ("l_foot", "l_toes"),
    ("torso_7", "r_up_leg"),
    ("r_up_leg", "r_low_leg"),
    ("r_low_leg", "r_foot"),
    ("r_foot", "r_toes"),
]


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


def _compute_or_use_offset(
    bvh_path: Path,
    camera_csv_path: Path,
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
    seq = load_bvh(bvh_path)
    cam_df = pd.read_csv(camera_csv_path)

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

    return best_offset


def _prepare_camera_landmarks(cam_df: pd.DataFrame) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Pre-group camera landmarks by frame for quick lookup.

    Returns:
        frame_index -> {landmark_name: (x_norm, y_norm)}
    """
    per_frame: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for _, row in cam_df.iterrows():
        frame_idx = int(row["frame"])
        lm_name = str(row["landmark_name"])
        x = float(row["x"])
        y = float(row["y"])
        per_frame.setdefault(frame_idx, {})[lm_name] = (x, y)
    return per_frame


def _draw_camera_skeleton(
    frame: np.ndarray,
    landmarks: Dict[str, Tuple[float, float]],
) -> None:
    """Draw a simple pose skeleton onto the camera frame (in-place)."""
    h, w = frame.shape[:2]

    # Convert to pixel coords
    pts: Dict[str, Tuple[int, int]] = {}
    for name, (xn, yn) in landmarks.items():
        cx = int(xn * w)
        cy = int(yn * h)
        pts[name] = (cx, cy)

    # Draw edges
    for a, b in CAMERA_EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    # Draw joints
    for p in pts.values():
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


def _prepare_mocopi_positions(seq, joints: list[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Precompute Mocopi joint positions over time and global scaling extents.

    Returns:
        (timestamps_ms, joint_positions)
        where joint_positions[joint_name] -> (num_frames, 3) array.
    """
    timestamps_ms = seq.timestamps_ms()
    joint_positions: Dict[str, np.ndarray] = {}

    xs = []
    ys = []

    for j in joints:
        if j not in seq.joint_names:
            continue
        pos = seq.joint_positions(j)
        joint_positions[j] = pos
        xs.append(pos[:, 0])
        ys.append(pos[:, 1])

    if xs and ys:
        xs_all = np.concatenate(xs)
        ys_all = np.concatenate(ys)
        x_min, x_max = float(xs_all.min()), float(xs_all.max())
        y_min, y_max = float(ys_all.min()), float(ys_all.max())
    else:
        x_min = y_min = -1.0
        x_max = y_max = 1.0

    # Store extents for later scaling
    joint_positions["_extents"] = np.array([[x_min, x_max, y_min, y_max]], dtype=float)
    return timestamps_ms, joint_positions


def _draw_mocopi_skeleton(
    canvas: np.ndarray,
    joints_positions: Dict[str, np.ndarray],
    t_mocopi_ms: float,
    timestamps_ms: np.ndarray,
) -> None:
    """
    Draw Mocopi skeleton onto the provided canvas (in-place).

    t_mocopi_ms:
        Target time in Mocopi timeline at which to sample joint positions.
    """
    h, w = canvas.shape[:2]
    extents = joints_positions.get("_extents")
    if extents is None:
        return
    x_min, x_max, y_min, y_max = extents[0]
    if x_max <= x_min or y_max <= y_min:
        return

    # Find nearest Mocopi frame for the requested time
    idx = int(np.searchsorted(timestamps_ms, t_mocopi_ms))
    if idx <= 0 or idx >= len(timestamps_ms):
        return

    pts: Dict[str, Tuple[int, int]] = {}
    for name in MOCOPI_JOINTS:
        if name not in joints_positions:
            continue
        pos = joints_positions[name]
        if idx >= pos.shape[0]:
            continue
        x, y = float(pos[idx, 0]), float(pos[idx, 1])

        # Normalize into [0, 1] range and map to canvas
        xn = (x - x_min) / (x_max - x_min + 1e-6)
        yn = (y - y_min) / (y_max - y_min + 1e-6)
        yn = 1.0 - yn  # Flip vertical so up is up

        cx = int(xn * (w * 0.8) + w * 0.1)
        cy = int(yn * (h * 0.8) + h * 0.1)
        pts[name] = (cx, cy)

    # Draw edges
    for a, b in MOCOPI_EDGES:
        if a in pts and b in pts:
            cv2.line(canvas, pts[a], pts[b], (0, 255, 255), 2)

    # Draw joints
    for p in pts.values():
        cv2.circle(canvas, p, 3, (255, 0, 0), -1)


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

    # Determine offset between Mocopi and camera timelines
    offset_ms = _compute_or_use_offset(
        bvh_path,
        camera_csv_path,
        search_ms=args.search_ms,
        rate_hz=args.rate_hz,
        offset_ms=args.offset_ms,
    )

    # Load data
    seq = load_bvh(bvh_path)
    cam_df = pd.read_csv(camera_csv_path)
    camera_by_frame = _prepare_camera_landmarks(cam_df)
    t_m_ms, mocopi_positions = _prepare_mocopi_positions(seq, MOCOPI_JOINTS)

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
            _draw_camera_skeleton(left, cam_landmarks)

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
        _draw_mocopi_skeleton(right, mocopi_positions, t_mocopi_ms, t_m_ms)

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
