"""Frame sampling utilities supporting orientation analysis."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List

import cv2
import numpy as np


@dataclass
class FrameSample:
    """Container describing a sampled frame."""

    frame_index: int
    timestamp_ms: int
    frame_bgr: np.ndarray
    brightness: float
    contrast: float
    motion_score: float


@dataclass
class FrameSamplerConfig:
    """Configuration for FrameSampler."""

    max_samples: int = 150
    primary_stride: int = 15
    rewind_radius: int = 3
    min_brightness: float = 25.0
    max_brightness: float = 240.0
    min_contrast: float = 10.0
    motion_threshold: float = 0.5


class FrameSampler:
    """Provides staggered sampling and neighbor rewinds for a capture stream."""

    def __init__(self, cap: cv2.VideoCapture, fps: float, config: FrameSamplerConfig | None = None) -> None:
        self.cap = cap
        self.fps = fps or 30.0
        self.config = config or FrameSamplerConfig()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._samples_taken = 0
        self._buffer_limit = max(5, self.config.rewind_radius * 4)
        self._buffer_indices: Deque[int] = deque()
        self._buffer_map: Dict[int, FrameSample] = {}
        self._prev_lowres: np.ndarray | None = None
        self._last_index_read = -1
        self.quick_rejects = 0

    def uniform_samples(self) -> Iterable[FrameSample]:
        """Yield uniformly staggered samples until cap or budget is exhausted."""
        stride = self._compute_stride()
        if self.total_frames > 0:
            indices = range(0, self.total_frames, stride)
        else:
            indices = (idx * stride for idx in range(self.config.max_samples))

        for frame_index in indices:
            if not self._can_take_more():
                break
            sample = self._read_frame_at(int(frame_index))
            if sample is None:
                break
            if self._is_viable(sample):
                yield sample
            else:
                self.quick_rejects += 1

    def request_neighbors(self, center_index: int, radius: int, apply_filters: bool = False) -> List[FrameSample]:
        """Return cached or freshly sampled frames within +/- radius of the anchor."""
        neighbors: List[FrameSample] = []
        for offset in range(-radius, radius + 1):
            frame_index = center_index + offset
            if offset == 0:
                continue
            sample = self._buffer_map.get(frame_index)
            if sample is None:
                sample = self._read_frame_at(frame_index)
            if sample is None:
                continue
            if apply_filters and not self._is_viable(sample):
                continue
            neighbors.append(sample)
        return neighbors

    def _remember_sample(self, sample: FrameSample) -> None:
        self._buffer_indices.append(sample.frame_index)
        self._buffer_map[sample.frame_index] = sample
        while len(self._buffer_indices) > self._buffer_limit:
            old_index = self._buffer_indices.popleft()
            self._buffer_map.pop(old_index, None)

    def _compute_stride(self) -> int:
        if self.total_frames > 0 and self.total_frames < self.config.max_samples:
            return 1
        return max(1, self.config.primary_stride)

    def _calc_metrics(self, frame_bgr: np.ndarray) -> tuple[float, float, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        lowres = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        lowres = lowres.astype(np.float32)
        if self._prev_lowres is None:
            motion = 0.0
        else:
            motion = float(np.mean(np.abs(lowres - self._prev_lowres)))
        self._prev_lowres = lowres
        return brightness, contrast, motion

    def _is_viable(self, sample: FrameSample) -> bool:
        if sample.brightness < self.config.min_brightness or sample.brightness > self.config.max_brightness:
            return False
        if sample.contrast < self.config.min_contrast and sample.motion_score < self.config.motion_threshold:
            return False
        return True

    def _read_frame_at(self, frame_index: int) -> FrameSample | None:
        if frame_index < 0:
            return None
        if self.total_frames and frame_index >= self.total_frames:
            return None
        if not self._can_take_more():
            return None

        if frame_index != self._last_index_read + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        self._last_index_read = frame_index

        timestamp_ms = int((frame_index / self.fps) * 1000.0)
        brightness, contrast, motion = self._calc_metrics(frame)
        sample = FrameSample(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            frame_bgr=frame,
            brightness=brightness,
            contrast=contrast,
            motion_score=motion,
        )
        self._remember_sample(sample)
        self._samples_taken += 1
        return sample

    def _can_take_more(self) -> bool:
        return self._samples_taken < self.config.max_samples


__all__ = ["FrameSample", "FrameSampler", "FrameSamplerConfig"]
