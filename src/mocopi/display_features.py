from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .camera_projection import CameraProjectionResult


LOWER_LIMB_SUFFIXES = ("HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX")
DISTAL_FOOT_SUFFIXES = ("ANKLE", "HEEL", "FOOT_INDEX")
WEIGHTED_LOWER_LIMB = {
    "HIP": 0.05,
    "KNEE": 0.10,
    "ANKLE": 0.22,
    "HEEL": 0.28,
    "FOOT_INDEX": 0.35,
}


@dataclass(frozen=True)
class CameraDisplayFeatureConfig:
    mode: str = "auto"
    smooth_window: int = 7


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr.copy()
    radius = max(0, window // 2)
    out = arr.copy()
    for idx in range(arr.shape[0]):
        lo = max(0, idx - radius)
        hi = min(arr.shape[0], idx + radius + 1)
        chunk = arr[lo:hi]
        finite = np.isfinite(chunk)
        out[idx] = float(np.mean(chunk[finite])) if np.any(finite) else np.nan
    return out


def _smooth_positions(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected position traces to be 2D")
    if window <= 1:
        return arr.copy()
    return np.column_stack([_smooth_1d(arr[:, dim], window) for dim in range(arr.shape[1])])


def _infer_side(landmark_name: str) -> str | None:
    upper = landmark_name.upper()
    if upper.startswith("LEFT_"):
        return "LEFT"
    if upper.startswith("RIGHT_"):
        return "RIGHT"
    return None


def _landmark_is_lower_limb(landmark_name: str) -> bool:
    upper = landmark_name.upper()
    return any(upper.endswith(suffix) for suffix in LOWER_LIMB_SUFFIXES)


def _candidate_lower_limb_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_{suffix}" for suffix in ("HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX"))


def _candidate_distal_foot_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_{suffix}" for suffix in DISTAL_FOOT_SUFFIXES)


def _nanmean_stack(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("Expected at least one array to average")
    stack = np.stack(arrays, axis=0).astype(float)
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    totals = np.where(valid, stack, 0.0).sum(axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=float)
    mask = counts > 0
    out[mask] = totals[mask] / counts[mask]
    return out


def _weighted_nanmean(named_arrays: list[tuple[str, np.ndarray]], weights: dict[str, float]) -> np.ndarray:
    if not named_arrays:
        raise ValueError("Expected at least one named array to average")
    stack = np.stack([np.asarray(arr, dtype=float) for _, arr in named_arrays], axis=0)
    weight_vec = np.asarray([float(weights.get(name, 0.0)) for name, _ in named_arrays], dtype=float)
    if not np.any(weight_vec > 0):
        return _nanmean_stack([arr for _, arr in named_arrays])

    valid = np.isfinite(stack)
    weighted = np.where(valid, stack * weight_vec[:, None, None], 0.0)
    total_weight = np.where(valid, weight_vec[:, None, None], 0.0).sum(axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=float)
    mask = total_weight > 0
    out[mask] = weighted.sum(axis=0)[mask] / total_weight[mask]
    return out


def _raw_traces(
    projection: CameraProjectionResult,
    landmark_names: Sequence[str],
    *,
    smooth_window: int,
) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {}
    for landmark_name in landmark_names:
        if landmark_name not in projection.positions:
            continue
        traces[landmark_name] = _smooth_positions(projection.positions[landmark_name], smooth_window)
    return traces


def _lower_limb_composite_traces(
    projection: CameraProjectionResult,
    landmark_names: Sequence[str],
    *,
    smooth_window: int,
) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {}
    seen_sides: set[str] = set()

    for landmark_name in landmark_names:
        side = _infer_side(landmark_name)
        if side is None or side in seen_sides:
            continue
        seen_sides.add(side)
        arrays = [
            projection.positions[name]
            for name in _candidate_lower_limb_names(side)
            if name in projection.positions
        ]
        if not arrays and landmark_name in projection.positions:
            arrays = [projection.positions[landmark_name]]
        if not arrays:
            continue
        traces[f"{side}_LOWER_LIMB"] = _smooth_positions(_nanmean_stack(arrays), smooth_window)

    if not traces:
        return _raw_traces(projection, landmark_names, smooth_window=smooth_window)
    return traces


def _distal_foot_composite_traces(
    projection: CameraProjectionResult,
    landmark_names: Sequence[str],
    *,
    smooth_window: int,
) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {}
    seen_sides: set[str] = set()

    for landmark_name in landmark_names:
        side = _infer_side(landmark_name)
        if side is None or side in seen_sides:
            continue
        seen_sides.add(side)
        arrays = [
            projection.positions[name]
            for name in _candidate_distal_foot_names(side)
            if name in projection.positions
        ]
        if not arrays and landmark_name in projection.positions:
            arrays = [projection.positions[landmark_name]]
        if not arrays:
            continue
        traces[f"{side}_DISTAL_FOOT"] = _smooth_positions(_nanmean_stack(arrays), smooth_window)

    if not traces:
        return _raw_traces(projection, landmark_names, smooth_window=smooth_window)
    return traces


def _weighted_lower_limb_traces(
    projection: CameraProjectionResult,
    landmark_names: Sequence[str],
    *,
    smooth_window: int,
) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {}
    seen_sides: set[str] = set()

    for landmark_name in landmark_names:
        side = _infer_side(landmark_name)
        if side is None or side in seen_sides:
            continue
        seen_sides.add(side)
        named_arrays = []
        for suffix in ("HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX"):
            full_name = f"{side}_{suffix}"
            if full_name in projection.positions:
                named_arrays.append((suffix, projection.positions[full_name]))
        if not named_arrays and landmark_name in projection.positions:
            named_arrays = [(landmark_name.split("_", 1)[-1], projection.positions[landmark_name])]
        if not named_arrays:
            continue
        traces[f"{side}_WEIGHTED_LOWER_LIMB"] = _smooth_positions(
            _weighted_nanmean(named_arrays, WEIGHTED_LOWER_LIMB),
            smooth_window,
        )

    if not traces:
        return _raw_traces(projection, landmark_names, smooth_window=smooth_window)
    return traces


def build_camera_display_traces(
    projection: CameraProjectionResult,
    landmark_names: Sequence[str],
    config: CameraDisplayFeatureConfig | None = None,
) -> dict[str, np.ndarray]:
    if config is None:
        config = CameraDisplayFeatureConfig()

    mode = config.mode
    if mode == "auto":
        lower_limb = bool(landmark_names) and all(_landmark_is_lower_limb(name) for name in landmark_names)
        mode = "lower-limb-composite" if lower_limb else "raw"

    if mode == "raw":
        return _raw_traces(projection, landmark_names, smooth_window=config.smooth_window)
    if mode == "lower-limb-composite":
        return _lower_limb_composite_traces(projection, landmark_names, smooth_window=config.smooth_window)
    if mode == "distal-foot-composite":
        return _distal_foot_composite_traces(projection, landmark_names, smooth_window=config.smooth_window)
    if mode == "weighted-lower-limb":
        return _weighted_lower_limb_traces(projection, landmark_names, smooth_window=config.smooth_window)

    raise ValueError(f"Unsupported camera display feature mode: {config.mode!r}")


__all__ = [
    "CameraDisplayFeatureConfig",
    "build_camera_display_traces",
]
