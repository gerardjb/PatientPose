from pathlib import Path
import sys
import importlib

import numpy as np
import pytest

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from video_tools.frame_sampling import FrameSample
from video_tools.orientation_focus import (
    OrientationAnalyzer,
    OrientationAnalyzerConfig,
    PoseObservation,
)
import video_tools.orientation_focus as orientation_focus_module
from video_tools.pose_quality import PoseQuality, PoseQualityScorer, PoseQualityScorerConfig


def _load_pose_module():
    try:
        return importlib.import_module("mediapipe.solutions.pose")
    except ModuleNotFoundError:
        return importlib.import_module("mediapipe.python.solutions.pose")


try:
    _POSE_MODULE = _load_pose_module()
except ModuleNotFoundError:
    pytest.skip("MediaPipe pose solution not available", allow_module_level=True)


PoseLandmark = _POSE_MODULE.PoseLandmark


class FakeLandmark:
    def __init__(self, x: float, y: float, visibility: float = 0.9) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


def test_pose_quality_scorer_identifies_good_pose() -> None:
    scorer = PoseQualityScorer(PoseQualityScorerConfig())
    landmarks = [FakeLandmark(0.5, 0.6) for _ in range(len(PoseLandmark))]

    landmarks[PoseLandmark.LEFT_SHOULDER.value] = FakeLandmark(0.4, 0.45)
    landmarks[PoseLandmark.RIGHT_SHOULDER.value] = FakeLandmark(0.6, 0.46)
    landmarks[PoseLandmark.NOSE.value] = FakeLandmark(0.5, 0.25)
    landmarks[PoseLandmark.LEFT_HIP.value] = FakeLandmark(0.45, 0.7)
    landmarks[PoseLandmark.RIGHT_HIP.value] = FakeLandmark(0.55, 0.7)

    quality = scorer.score_landmarks(landmarks)
    assert quality is not None
    assert quality.is_good
    assert quality.score < 30.0


def test_orientation_analyzer_uses_anchor_neighborhood(monkeypatch, tmp_path) -> None:
    rotation_codes = [None, cv2.ROTATE_90_CLOCKWISE]
    config = OrientationAnalyzerConfig(
        rotation_codes=rotation_codes,
        max_scan_frames=6,
        primary_stride=1,
        anchor_radius=1,
        debug_enabled=False,
    )

    def make_sample(index: int) -> FrameSample:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return FrameSample(
            frame_index=index,
            timestamp_ms=index * 33,
            frame_bgr=frame,
            brightness=80.0 + index,
            contrast=15.0,
            motion_score=1.0,
        )

    sample_initial = make_sample(0)
    sample_anchor = make_sample(5)
    sample_neighbor_before = make_sample(4)
    sample_neighbor_after = make_sample(6)

    class FakeSampler:
        def __init__(self) -> None:
            self.quick_rejects = 0
            self._requested = False

        def uniform_samples(self):
            yield sample_initial
            yield sample_anchor

        def request_neighbors(self, center_index: int, radius: int, apply_filters: bool = False):
            if self._requested:
                return []
            self._requested = True
            assert center_index == sample_anchor.frame_index
            assert radius == 1
            return [sample_neighbor_before, sample_neighbor_after]

    fake_sampler = FakeSampler()

    class DummyPoseMarker:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_pose_marker_factory(_path):
        return DummyPoseMarker()

    def fake_capture(*_args, **_kwargs):
        class DummyCapture:
            def isOpened(self):
                return True

            def get(self, _prop):
                return 30.0

            def release(self):
                return None

        return DummyCapture()

    monkeypatch.setattr(orientation_focus_module.cv2, "VideoCapture", fake_capture)

    def fake_evaluate_sample(self, pose_marker, sample, scorer, prefer_rotation=None):
        if sample.frame_index == sample_initial.frame_index:
            rotation = rotation_codes[0]
            score = 120.0
            is_good = False
        else:
            rotation = rotation_codes[1]
            score = 5.0
            is_good = True

        quality = PoseQuality(
            score=score,
            is_good=is_good,
            angle_deg=10.0,
            vertical_deg=5.0,
            visibility_mean=0.9,
            landmark_count=30,
            confidence_sum=25.0,
            nose_above_shoulders=True,
        )
        observation = PoseObservation(
            frame_index=sample.frame_index,
            timestamp_ms=sample.timestamp_ms,
            rotation_code=rotation,
            quality=quality,
            detection_confidence=quality.visibility_mean,
            brightness=sample.brightness,
            contrast=sample.contrast,
            motion_score=sample.motion_score,
            source="original",
        )
        return [observation]

    monkeypatch.setattr(OrientationAnalyzer, "_evaluate_sample", fake_evaluate_sample)

    analyzer = OrientationAnalyzer(
        config,
        sampler_factory=lambda cap, fps, cfg: fake_sampler,
        pose_marker_factory=fake_pose_marker_factory,
    )

    video_path = tmp_path / "dummy.mp4"
    pose_model = tmp_path / "model.task"
    decision = analyzer.analyze(video_path, pose_model)

    assert decision.rotation_code == cv2.ROTATE_90_CLOCKWISE
    assert decision.reason.startswith("vote")
