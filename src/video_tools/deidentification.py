# -*- coding: utf-8 -*-
"""
Utilities for de-identifying video frames.
"""
import math
from typing import Iterable, Sequence

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.framework.formats import landmark_pb2

PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
PoseLandmark = mp.solutions.pose.PoseLandmark

FACE_LANDMARK_INDICES: Sequence[int] = [
    PoseLandmark.NOSE.value,
    PoseLandmark.LEFT_EYE_INNER.value,
    PoseLandmark.LEFT_EYE.value,
    PoseLandmark.LEFT_EYE_OUTER.value,
    PoseLandmark.RIGHT_EYE_INNER.value,
    PoseLandmark.RIGHT_EYE.value,
    PoseLandmark.RIGHT_EYE_OUTER.value,
    PoseLandmark.MOUTH_LEFT.value,
    PoseLandmark.MOUTH_RIGHT.value,
]
LEFT_EYE_OUTER_INDEX = PoseLandmark.LEFT_EYE_OUTER.value
RIGHT_EYE_OUTER_INDEX = PoseLandmark.RIGHT_EYE_OUTER.value


def _collect_landmark_coords(
    landmarks: Sequence[landmark_pb2.NormalizedLandmark],
    indices: Iterable[int],
) -> Sequence[tuple[float, float]]:
    coords = []
    for idx in indices:
        if idx >= len(landmarks):
            return []
        lm = landmarks[idx]
        coords.append((lm.x, lm.y))
    return coords


def blur_face_with_pose(
    frame_rgb: np.ndarray, pose_result: PoseLandmarkerResult, square_scale: float = 2.0
) -> np.ndarray:
    """
    Blacks out a square region around the face using pose landmarks.

    Args:
        frame_rgb: Frame in RGB format.
        pose_result: Pose detection result for the frame.
        square_scale: Multiplier applied to the inter-eye distance to set the box size.

    Returns:
        A copy of the frame with the face region anonymized.
    """
    if not pose_result.pose_landmarks:
        return frame_rgb

    annotated = np.copy(frame_rgb)
    height, width, _ = annotated.shape
    landmarks = pose_result.pose_landmarks[0]

    face_coords = _collect_landmark_coords(landmarks, FACE_LANDMARK_INDICES)
    if not face_coords:
        return frame_rgb

    center_x_norm = float(np.mean([coord[0] for coord in face_coords]))
    center_y_norm = float(np.mean([coord[1] for coord in face_coords]))
    center_x = int(center_x_norm * width)
    center_y = int(center_y_norm * height)

    if LEFT_EYE_OUTER_INDEX >= len(landmarks) or RIGHT_EYE_OUTER_INDEX >= len(landmarks):
        return frame_rgb

    left_eye = landmarks[LEFT_EYE_OUTER_INDEX]
    right_eye = landmarks[RIGHT_EYE_OUTER_INDEX]

    eye_distance = math.sqrt(
        (left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2
    )
    side_length = max(10, int(square_scale * eye_distance * max(width, height)))
    half_side = side_length // 2

    xmin = max(0, center_x - half_side)
    ymin = max(0, center_y - half_side)
    xmax = min(width, center_x + half_side)
    ymax = min(height, center_y + half_side)

    if xmin < xmax and ymin < ymax:
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 0, 0), thickness=-1)

    return annotated
