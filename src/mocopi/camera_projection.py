from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .features import NoCameraPoseDataError

DEFAULT_ORIGIN_LANDMARKS = ("LEFT_HIP", "RIGHT_HIP")
DEFAULT_SCALE_PAIRS = (
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "LEFT_SHOULDER"),
    ("RIGHT_HIP", "RIGHT_SHOULDER"),
)
DEFAULT_LATERAL_AXIS_PAIR = ("LEFT_HIP", "RIGHT_HIP")
DEFAULT_VERTICAL_AXIS_PAIRS = (
    ("LEFT_HIP", "LEFT_SHOULDER"),
    ("RIGHT_HIP", "RIGHT_SHOULDER"),
)


@dataclass(frozen=True)
class CameraProjectionConfig:
    visibility_threshold: float | None = 0.4
    smooth_window: int = 7
    rotate_to_body_frame: bool = False
    origin_landmarks: tuple[str, ...] = DEFAULT_ORIGIN_LANDMARKS
    scale_pairs: tuple[tuple[str, str], ...] = DEFAULT_SCALE_PAIRS
    lateral_axis_pair: tuple[str, str] = DEFAULT_LATERAL_AXIS_PAIR
    vertical_axis_pairs: tuple[tuple[str, str], ...] = DEFAULT_VERTICAL_AXIS_PAIRS


@dataclass(frozen=True)
class CameraProjectionResult:
    frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    positions: dict[str, np.ndarray]
    image_points: dict[str, np.ndarray]
    image_depths: dict[str, np.ndarray]
    valid_mask: dict[str, np.ndarray]
    origin_xy: np.ndarray
    origin_z: np.ndarray
    scale: np.ndarray
    body_x_axis: np.ndarray
    body_y_axis: np.ndarray


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr.copy()
    half_window = max(0, window // 2)
    out = arr.copy()
    for idx in range(arr.shape[0]):
        lo = max(0, idx - half_window)
        hi = min(arr.shape[0], idx + half_window + 1)
        chunk = arr[lo:hi]
        finite = np.isfinite(chunk)
        out[idx] = float(np.mean(chunk[finite])) if np.any(finite) else np.nan
    return out


def _smooth_2d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array to smooth")
    return np.column_stack([_smooth_1d(arr[:, dim], window) for dim in range(arr.shape[1])])


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-6)
    out[valid] = arr[valid] / norms[valid]
    return out


def _nanmedian_or_default(values: np.ndarray, default: float = 1.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.median(finite))


def _mean_points(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs, ys = zip(*points)
    return float(np.mean(xs)), float(np.mean(ys))


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (float((a[0] + b[0]) * 0.5), float((a[1] + b[1]) * 0.5))


def _orthogonalize_axes(x_axis: np.ndarray, y_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_axis, dtype=float)

    x = _normalize_vectors(x)
    y = _normalize_vectors(y)

    for idx in range(x.shape[0]):
        if not np.all(np.isfinite(x[idx])):
            x[idx] = np.array([1.0, 0.0], dtype=float)
        if not np.all(np.isfinite(y[idx])):
            y[idx] = np.array([0.0, -1.0], dtype=float)

        y_proj = y[idx] - float(np.dot(y[idx], x[idx])) * x[idx]
        y_norm = np.linalg.norm(y_proj)
        if not np.isfinite(y_norm) or y_norm <= 1e-6:
            y_proj = np.array([-x[idx, 1], x[idx, 0]], dtype=float)
            y_norm = np.linalg.norm(y_proj)
        y[idx] = y_proj / max(y_norm, 1e-6)

        if y[idx, 1] > 0.0:
            y[idx] *= -1.0

    return x, y


def compute_camera_projection(
    df: pd.DataFrame,
    landmark_names: Sequence[str],
    config: CameraProjectionConfig | None = None,
) -> CameraProjectionResult:
    if config is None:
        config = CameraProjectionConfig()

    pose_df = df[df["source"] == "pose"].copy()
    if pose_df.empty:
        raise NoCameraPoseDataError("No pose landmarks found in camera CSV")

    has_visibility = "visibility" in pose_df.columns
    frame_indices = np.array(sorted(pose_df["frame"].unique()), dtype=int)
    timestamps_ms: list[float] = []
    origin_xy = np.full((len(frame_indices), 2), np.nan, dtype=float)
    origin_z = np.full(len(frame_indices), np.nan, dtype=float)
    scale = np.full(len(frame_indices), np.nan, dtype=float)
    body_x_axis = np.full((len(frame_indices), 2), np.nan, dtype=float)
    body_y_axis = np.full((len(frame_indices), 2), np.nan, dtype=float)
    image_points = {
        name: np.full((len(frame_indices), 2), np.nan, dtype=float)
        for name in landmark_names
    }
    image_depths = {
        name: np.full(len(frame_indices), np.nan, dtype=float)
        for name in landmark_names
    }

    for row_idx, frame_idx in enumerate(frame_indices):
        sub = pose_df[pose_df["frame"] == frame_idx]
        if sub.empty:
            continue

        timestamps_ms.append(float(sub["timestamp_ms"].iloc[0]))

        by_name_xy: dict[str, tuple[float, float]] = {}
        by_name_z: dict[str, float] = {}
        for _, row in sub.iterrows():
            visibility = float(row["visibility"]) if has_visibility else 1.0
            if config.visibility_threshold is not None:
                if np.isnan(visibility) or visibility < config.visibility_threshold:
                    continue
            landmark_name = str(row["landmark_name"])
            by_name_xy[landmark_name] = (float(row["x"]), float(row["y"]))
            by_name_z[landmark_name] = float(row["z"]) if "z" in row.index else np.nan

        origin_candidates = [by_name_xy[name] for name in config.origin_landmarks if name in by_name_xy]
        if not origin_candidates:
            origin_candidates = list(by_name_xy.values())
        if origin_candidates:
            origin_xy[row_idx] = _mean_points(origin_candidates)

        origin_depth_candidates = [by_name_z[name] for name in config.origin_landmarks if name in by_name_z]
        if not origin_depth_candidates:
            origin_depth_candidates = list(by_name_z.values())
        finite_depths = np.asarray(origin_depth_candidates, dtype=float)
        finite_depths = finite_depths[np.isfinite(finite_depths)]
        if finite_depths.size:
            origin_z[row_idx] = float(np.mean(finite_depths))

        scale_values = []
        for left_name, right_name in config.scale_pairs:
            if left_name in by_name_xy and right_name in by_name_xy:
                left_pt = np.asarray(by_name_xy[left_name], dtype=float)
                right_pt = np.asarray(by_name_xy[right_name], dtype=float)
                distance = float(np.linalg.norm(right_pt - left_pt))
                if np.isfinite(distance) and distance > 1e-6:
                    scale_values.append(distance)
        if scale_values:
            scale[row_idx] = float(np.mean(scale_values))
        elif np.all(np.isfinite(origin_xy[row_idx])) and by_name_xy:
            offsets = np.asarray(list(by_name_xy.values()), dtype=float) - origin_xy[row_idx]
            distances = np.linalg.norm(offsets, axis=1)
            finite = distances[np.isfinite(distances)]
            if finite.size:
                scale[row_idx] = float(np.mean(finite))

        if config.lateral_axis_pair[0] in by_name_xy and config.lateral_axis_pair[1] in by_name_xy:
            left_pt = np.asarray(by_name_xy[config.lateral_axis_pair[0]], dtype=float)
            right_pt = np.asarray(by_name_xy[config.lateral_axis_pair[1]], dtype=float)
            body_x_axis[row_idx] = right_pt - left_pt
        else:
            body_x_axis[row_idx] = np.array([1.0, 0.0], dtype=float)

        vertical_vectors = []
        for hip_name, shoulder_name in config.vertical_axis_pairs:
            if hip_name in by_name_xy and shoulder_name in by_name_xy:
                hip_pt = np.asarray(by_name_xy[hip_name], dtype=float)
                shoulder_pt = np.asarray(by_name_xy[shoulder_name], dtype=float)
                vertical_vectors.append(shoulder_pt - hip_pt)
        if vertical_vectors:
            body_y_axis[row_idx] = np.mean(np.stack(vertical_vectors, axis=0), axis=0)
        else:
            body_y_axis[row_idx] = np.array([0.0, -1.0], dtype=float)

        for landmark_name in landmark_names:
            if landmark_name in by_name_xy:
                image_points[landmark_name][row_idx] = by_name_xy[landmark_name]
            if landmark_name in by_name_z:
                image_depths[landmark_name][row_idx] = by_name_z[landmark_name]

    timestamps_arr = np.asarray(timestamps_ms, dtype=float)
    if timestamps_arr.size != frame_indices.size:
        timestamps_arr = frame_indices.astype(float)

    smooth_window = max(1, int(config.smooth_window))
    origin_xy = _smooth_2d(origin_xy, smooth_window)
    scale = _smooth_1d(scale, smooth_window)
    fill_scale = _nanmedian_or_default(scale, default=1.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fill_scale)
    body_x_axis = _smooth_2d(body_x_axis, smooth_window)
    body_y_axis = _smooth_2d(body_y_axis, smooth_window)
    body_x_axis, body_y_axis = _orthogonalize_axes(body_x_axis, body_y_axis)

    positions: dict[str, np.ndarray] = {}
    valid_mask: dict[str, np.ndarray] = {}

    for landmark_name, pts in image_points.items():
        projected = np.full((pts.shape[0], 3), np.nan, dtype=float)
        valid = np.all(np.isfinite(pts), axis=1) & np.all(np.isfinite(origin_xy), axis=1) & np.isfinite(scale)
        rel = pts - origin_xy
        if config.rotate_to_body_frame:
            projected[valid, 0] = np.einsum("ij,ij->i", rel[valid], body_x_axis[valid]) / scale[valid]
            projected[valid, 1] = np.einsum("ij,ij->i", rel[valid], body_y_axis[valid]) / scale[valid]
        else:
            projected[valid, 0] = rel[valid, 0] / scale[valid]
            projected[valid, 1] = -rel[valid, 1] / scale[valid]
        depths = image_depths[landmark_name]
        depth_valid = valid & np.isfinite(depths) & np.isfinite(origin_z)
        projected[depth_valid, 2] = (depths[depth_valid] - origin_z[depth_valid]) / scale[depth_valid]
        positions[landmark_name] = projected
        valid_mask[landmark_name] = valid

    return CameraProjectionResult(
        frame_indices=frame_indices,
        timestamps_ms=timestamps_arr,
        positions=positions,
        image_points=image_points,
        image_depths=image_depths,
        valid_mask=valid_mask,
        origin_xy=origin_xy,
        origin_z=origin_z,
        scale=scale,
        body_x_axis=body_x_axis,
        body_y_axis=body_y_axis,
    )


__all__ = [
    "CameraProjectionConfig",
    "CameraProjectionResult",
    "compute_camera_projection",
]
