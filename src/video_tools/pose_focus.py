"""Utilities for tracking pose focus regions across frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from mediapipe.tasks.components.containers import NormalizedLandmark, Rect
except ModuleNotFoundError:  # pragma: no cover - fallback for older mediapipe layouts
    from mediapipe.tasks.python.components.containers import NormalizedLandmark, Rect


@dataclass
class PoseFocusHint:
    """Summary of a representative pose observation for ROI seeding."""

    frame_index: int
    rotation_code: int | None
    bbox: Tuple[float, float, float, float]
    score: float


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _expand_bbox(bbox: Tuple[float, float, float, float], amount: float) -> Tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    expand_x = width * amount
    expand_y = height * amount
    return (
        _clip(left - expand_x),
        _clip(top - expand_y),
        _clip(right + expand_x),
        _clip(bottom + expand_y),
    )


def bbox_from_landmarks(landmarks: Sequence[NormalizedLandmark], margin: float = 0.05) -> Tuple[float, float, float, float]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    left = _clip(min(xs) - margin)
    right = _clip(max(xs) + margin)
    top = _clip(min(ys) - margin)
    bottom = _clip(max(ys) + margin)
    return left, top, right, bottom


class PoseFocusTracker:
    """Maintains a normalized ROI rectangle derived from an anchor observation."""

    def __init__(
        self,
        hint: PoseFocusHint,
        max_failures_before_reset: int = 10,
        expansion_rate: float = 0.25,
        smooth_factor: float = 0.8,
    ) -> None:
        self._hint = hint
        self._current_bbox = list(hint.bbox)
        self._failure_count = 0
        self._max_failures = max_failures_before_reset
        self._expansion_rate = expansion_rate
        self._smooth_factor = smooth_factor

    def region_rect(self) -> Optional[Rect]:
        if self._current_bbox is None:
            return None
        left, top, right, bottom = self._current_bbox
        return Rect(left=float(left), top=float(top), right=float(right), bottom=float(bottom))

    def register_success(self, landmarks: Sequence[NormalizedLandmark]) -> None:
        self._failure_count = 0
        new_bbox = bbox_from_landmarks(landmarks)
        if self._current_bbox is None:
            self._current_bbox = list(new_bbox)
            return
        sm = self._smooth_factor
        left = sm * self._current_bbox[0] + (1 - sm) * new_bbox[0]
        top = sm * self._current_bbox[1] + (1 - sm) * new_bbox[1]
        right = sm * self._current_bbox[2] + (1 - sm) * new_bbox[2]
        bottom = sm * self._current_bbox[3] + (1 - sm) * new_bbox[3]
        self._current_bbox = [_clip(left), _clip(top), _clip(right), _clip(bottom)]

    def register_failure(self) -> None:
        self._failure_count += 1
        if self._current_bbox is None:
            return
        if self._failure_count >= self._max_failures:
            self._current_bbox = [0.0, 0.0, 1.0, 1.0]
            return
        left, top, right, bottom = self._current_bbox
        width = max(1e-3, right - left)
        height = max(1e-3, bottom - top)
        expand_x = width * self._expansion_rate
        expand_y = height * self._expansion_rate
        self._current_bbox = [
            _clip(left - expand_x),
            _clip(top - expand_y),
            _clip(right + expand_x),
            _clip(bottom + expand_y),
        ]

    def current_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        if self._current_bbox is None:
            return None
        return tuple(self._current_bbox)


def crop_frame_from_bbox(
    frame_rgb: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, CropTransform] | None:
    height, width = frame_rgb.shape[:2]
    left = max(0, int(round(bbox[0] * width)))
    top = max(0, int(round(bbox[1] * height)))
    right = min(width, int(round(bbox[2] * width)))
    bottom = min(height, int(round(bbox[3] * height)))
    if right - left < 5 or bottom - top < 5:
        return None
    crop = frame_rgb[top:bottom, left:right]
    transform = CropTransform(
        left=left,
        top=top,
        width=right - left,
        height=bottom - top,
        frame_width=width,
        frame_height=height,
    )
    return crop, transform


def remap_landmarks_from_crop(
    landmarks: Sequence[NormalizedLandmark],
    transform: CropTransform,
) -> Sequence[NormalizedLandmark]:
    if transform.width <= 0 or transform.height <= 0:
        return landmarks
    for landmark in landmarks:
        full_x = (transform.left + landmark.x * transform.width) / transform.frame_width
        full_y = (transform.top + landmark.y * transform.height) / transform.frame_height
        landmark.x = _clip(full_x)
        landmark.y = _clip(full_y)
    return landmarks


__all__ = [
    "CropTransform",
    "PoseFocusHint",
    "PoseFocusTracker",
    "bbox_from_landmarks",
    "crop_frame_from_bbox",
    "remap_landmarks_from_crop",
]


__all__ = ["PoseFocusHint", "PoseFocusTracker", "bbox_from_landmarks"]
@dataclass
class CropTransform:
    """Metadata linking a cropped region back to the full frame."""

    left: int
    top: int
    width: int
    height: int
    frame_width: int
    frame_height: int
