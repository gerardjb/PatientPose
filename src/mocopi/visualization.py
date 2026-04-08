from __future__ import annotations

"""Skeleton prep and drawing helpers shared by Mocopi video scripts."""

from typing import Dict, Tuple, List

import cv2
import numpy as np

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
    "root",
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


def prepare_camera_landmarks(cam_df) -> Dict[int, Dict[str, Tuple[float, float]]]:
    per_frame: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for _, row in cam_df.iterrows():
        frame_idx = int(row["frame"])
        lm_name = str(row["landmark_name"])
        x = float(row["x"])
        y = float(row["y"])
        per_frame.setdefault(frame_idx, {})[lm_name] = (x, y)
    return per_frame


def draw_camera_skeleton(frame: np.ndarray, landmarks: Dict[str, Tuple[float, float]]) -> None:
    h, w = frame.shape[:2]
    pts: Dict[str, Tuple[int, int]] = {}
    for name, (xn, yn) in landmarks.items():
        cx = int(xn * w)
        cy = int(yn * h)
        pts[name] = (cx, cy)
    for a, b in CAMERA_EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts.values():
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


def prepare_mocopi_positions(seq, joints: List[str] | None = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if joints is None:
        joints = MOCOPI_JOINTS

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

    joint_positions["_extents"] = np.array([[x_min, x_max, y_min, y_max]], dtype=float)
    if "root" in joint_positions:
        root_pos = joint_positions["root"]
        walk_min = float(root_pos[:, 0].min())
        walk_max = float(root_pos[:, 0].max())
    else:
        walk_min, walk_max = x_min, x_max
    joint_positions["_walk_range"] = np.array([[walk_min, walk_max]], dtype=float)

    center_name = None
    if "torso_7" in joint_positions:
        center_name = "torso_7"
    elif "root" in joint_positions:
        center_name = "root"
    else:
        for joint_name in joints:
            if joint_name in joint_positions:
                center_name = joint_name
                break

    if center_name is not None:
        center_pos = joint_positions[center_name]
        centered_xs = []
        centered_ys = []
        for joint_name in joints:
            pos = joint_positions.get(joint_name)
            if pos is None:
                continue
            frame_count = min(pos.shape[0], center_pos.shape[0])
            if frame_count <= 0:
                continue
            centered_xs.append(pos[:frame_count, 0] - center_pos[:frame_count, 0])
            centered_ys.append(pos[:frame_count, 1] - center_pos[:frame_count, 1])

        if centered_xs and centered_ys:
            max_abs_x = max(float(np.abs(np.concatenate(centered_xs)).max()), 1.0)
            max_abs_y = max(float(np.abs(np.concatenate(centered_ys)).max()), 1.0)
        else:
            max_abs_x = max_abs_y = 1.0

        joint_positions["_center_joint_name"] = center_name
        joint_positions["_centered_limits"] = np.array([[max_abs_x, max_abs_y]], dtype=float)

    return timestamps_ms, joint_positions


def draw_mocopi_skeleton(
    canvas: np.ndarray,
    joints_positions: Dict[str, np.ndarray],
    t_mocopi_ms: float,
    timestamps_ms: np.ndarray,
    *,
    view: str = "walk-range",
) -> None:
    h, w = canvas.shape[:2]
    extents = joints_positions.get("_extents")
    if extents is None:
        return
    x_min, x_max, y_min, y_max = extents[0]
    walk_range = joints_positions.get("_walk_range")
    if walk_range is not None:
        walk_min, walk_max = walk_range[0]
    else:
        walk_min, walk_max = x_min, x_max
    if walk_max <= walk_min or y_max <= y_min:
        return

    idx = int(np.searchsorted(timestamps_ms, t_mocopi_ms))
    if idx >= len(timestamps_ms):
        return
    if idx < 0:
        idx = 0

    pts: Dict[str, Tuple[int, int]] = {}
    center_joint_name = joints_positions.get("_center_joint_name")
    centered_limits = joints_positions.get("_centered_limits")
    center_pos = None
    if isinstance(center_joint_name, str):
        center_pos = joints_positions.get(center_joint_name)

    for name in MOCOPI_JOINTS:
        if name not in joints_positions:
            continue
        pos = joints_positions[name]
        if idx >= pos.shape[0]:
            continue
        x, y = float(pos[idx, 0]), float(pos[idx, 1])

        if view == "body-centered" and center_pos is not None and centered_limits is not None and idx < center_pos.shape[0]:
            x_limit, y_limit = centered_limits[0]
            center_x = float(center_pos[idx, 0])
            center_y = float(center_pos[idx, 1])
            xn = 0.5 + 0.5 * ((x - center_x) / (x_limit + 1e-6))
            yn = 0.5 - 0.5 * ((y - center_y) / (y_limit + 1e-6))
        else:
            xn = (x - walk_min) / (walk_max - walk_min + 1e-6)
            yn = 1.0 - ((y - y_min) / (y_max - y_min + 1e-6))

        xn = float(np.clip(xn, 0.0, 1.0))
        yn = float(np.clip(yn, 0.0, 1.0))
        cx = int(xn * (w * 0.8) + w * 0.1)
        cy = int(yn * (h * 0.8) + h * 0.1)
        pts[name] = (cx, cy)
    for a, b in MOCOPI_EDGES:
        if a in pts and b in pts:
            cv2.line(canvas, pts[a], pts[b], (0, 255, 255), 2)
    for p in pts.values():
        cv2.circle(canvas, p, 3, (255, 0, 0), -1)


__all__ = [
    "CAMERA_EDGES",
    "MOCOPI_JOINTS",
    "MOCOPI_EDGES",
    "prepare_camera_landmarks",
    "draw_camera_skeleton",
    "prepare_mocopi_positions",
    "draw_mocopi_skeleton",
]
