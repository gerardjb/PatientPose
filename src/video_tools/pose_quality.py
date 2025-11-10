"""Pose quality scoring utilities used by orientation analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

try:
    from mediapipe.solutions.pose import PoseLandmark
except ModuleNotFoundError:  # Fallback for environments without mediapipe.solutions package
    from mediapipe.python.solutions.pose import PoseLandmark


@dataclass
class PoseQuality:
    """Container summarizing the quality of a detected pose instance."""

    score: float
    is_good: bool
    angle_deg: float
    vertical_deg: float
    visibility_mean: float
    landmark_count: int
    confidence_sum: float
    nose_above_shoulders: bool


@dataclass
class PoseQualityScorerConfig:
    """Thresholds steering the pose quality calculation."""

    max_angle_for_good: float = 25.0
    max_vertical_tilt_deg: float = 35.0
    min_visibility_mean: float = 0.45
    min_landmark_count: int = 20
    good_pose_score_threshold: float = 75.0
    nose_penalty: float = 90.0
    visibility_penalty_scale: float = 180.0
    missing_landmark_penalty: float = 3.0
    visibility_presence_threshold: float = 0.2


class PoseQualityScorer:
    """Scores pose landmarks using heuristics tuned for orientation inference."""

    def __init__(self, config: PoseQualityScorerConfig | None = None) -> None:
        self.config = config or PoseQualityScorerConfig()

    def score(self, pose_result: Any) -> PoseQuality | None:
        """Score a MediaPipe PoseLandmarkerResult."""
        if not getattr(pose_result, "pose_landmarks", None):
            return None
        return self.score_landmarks(pose_result.pose_landmarks[0])

    def score_landmarks(self, landmarks: Sequence[Any]) -> PoseQuality | None:
        """Score a raw sequence of landmarks."""
        try:
            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
            nose = landmarks[PoseLandmark.NOSE.value]
            left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]
        except (IndexError, AttributeError):
            return None

        dx = right_shoulder.x - left_shoulder.x
        dy = right_shoulder.y - left_shoulder.y
        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        # Catches near-vertical angles
        angle_deg = min(abs(angle_deg), abs(abs(angle_deg) - 180.0))


        hip_center_x = (left_hip.x + right_hip.x) * 0.5
        hip_center_y = (left_hip.y + right_hip.y) * 0.5
        nose_dx = nose.x - hip_center_x
        nose_dy = hip_center_y - nose.y
        vertical_deg = math.degrees(math.atan2(abs(nose_dx), abs(nose_dy) + 1e-6))

        visibility_values: List[float] = []
        confidence_sum = 0.0
        present_count = 0
        for landmark in landmarks:
            visibility = float(getattr(landmark, "visibility", 0.0) or 0.0)
            visibility_values.append(visibility)
            confidence_sum += max(0.0, visibility)
            if visibility >= self.config.visibility_presence_threshold:
                present_count += 1

        if not visibility_values:
            return None

        visibility_mean = sum(visibility_values) / len(visibility_values)
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) * 0.5
        nose_above_shoulders = nose.y < shoulder_mid_y

        angle_penalty = max(0.0, angle_deg - self.config.max_angle_for_good)
        vertical_penalty = max(0.0, vertical_deg - self.config.max_vertical_tilt_deg)
        nose_penalty = 0.0 if nose_above_shoulders else self.config.nose_penalty
        visibility_penalty = max(0.0, self.config.min_visibility_mean - visibility_mean) * self.config.visibility_penalty_scale
        coverage_penalty = max(0, self.config.min_landmark_count - present_count) * self.config.missing_landmark_penalty

        score = angle_penalty + vertical_penalty + nose_penalty + visibility_penalty + coverage_penalty
        is_good = (
            angle_deg <= self.config.max_angle_for_good
            and vertical_deg <= self.config.max_vertical_tilt_deg
            and visibility_mean >= self.config.min_visibility_mean
            and present_count >= self.config.min_landmark_count
            and nose_above_shoulders
            and score <= self.config.good_pose_score_threshold
        )

        return PoseQuality(
            score=score,
            is_good=is_good,
            angle_deg=angle_deg,
            vertical_deg=vertical_deg,
            visibility_mean=visibility_mean,
            landmark_count=present_count,
            confidence_sum=confidence_sum,
            nose_above_shoulders=nose_above_shoulders,
        )


__all__ = [
    "PoseQuality",
    "PoseQualityScorer",
    "PoseQualityScorerConfig",
]
