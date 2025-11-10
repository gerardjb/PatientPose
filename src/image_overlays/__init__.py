"""
Utilities for visual overlays that annotate MediaPipe detections on frames.

Centralizing these helpers keeps drawing logic consistent across scripts and
makes it easier to evolve the visual language without touching every caller.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2
import numpy as np

os.environ.setdefault("MEDIAPIPE_SKIP_AUDIO", "1")

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

DrawingSpec = mp.solutions.drawing_utils.DrawingSpec
PoseConnections = mp.solutions.pose.POSE_CONNECTIONS
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

INDEX_FINGER_TIP_INDEX = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value
THUMB_TIP_INDEX = mp.solutions.hands.HandLandmark.THUMB_TIP.value

__all__ = [
    "draw_right_hand_fingertips",
    "draw_pose_landmarks",
]


def draw_right_hand_fingertips(
    frame_rgb: np.ndarray,
    hand_result: Optional[HandLandmarkerResult],
    *,
    radius: int = 3,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Overlay the detected right-hand index and thumb tips using solid circles.
    """
    if not hand_result or not hand_result.hand_landmarks or not hand_result.handedness:
        return frame_rgb

    annotated = np.copy(frame_rgb)
    height, width, _ = annotated.shape

    for handedness_list, landmarks in zip(hand_result.handedness, hand_result.hand_landmarks):
        if not handedness_list:
            continue
        if handedness_list[0].category_name.lower() != "right":
            continue

        for idx in (INDEX_FINGER_TIP_INDEX, THUMB_TIP_INDEX):
            if idx >= len(landmarks):
                continue
            landmark = landmarks[idx]
            cx = int(landmark.x * width)
            cy = int(landmark.y * height)
            if 0 <= cx < width and 0 <= cy < height:
                cv2.circle(annotated, (cx, cy), radius, color, thickness=-1)

    return annotated


def draw_pose_landmarks(
    frame_rgb: np.ndarray,
    pose_result: Optional[PoseLandmarkerResult],
    *,
    landmark_style: Optional[DrawingSpec | dict] = None,
    connection_style: Optional[DrawingSpec | dict] = None,
) -> np.ndarray:
    """
    Draw full-body pose landmarks and connections for every detected person.
    """
    if not pose_result or not pose_result.pose_landmarks:
        return frame_rgb

    annotated = np.copy(frame_rgb)

    if landmark_style is None:
        landmark_style = (
            drawing_styles.get_default_pose_landmarks_style()
            if hasattr(drawing_styles, "get_default_pose_landmarks_style")
            else DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

    if connection_style is None:
        connection_style = (
            drawing_styles.get_default_pose_connections_style()
            if hasattr(drawing_styles, "get_default_pose_connections_style")
            else DrawingSpec(color=(0, 138, 255), thickness=2, circle_radius=2)
        )

    for normalized_landmarks in pose_result.pose_landmarks:
        landmark_list = (
            normalized_landmarks
            if hasattr(normalized_landmarks, "landmark")
            else _to_landmark_list(normalized_landmarks)
        )
        drawing_utils.draw_landmarks(
            annotated,
            landmark_list,
            PoseConnections,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )

    return annotated
def _to_landmark_list(landmarks: Sequence) -> landmark_pb2.NormalizedLandmarkList:
    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        proto.landmark.add(x=lm.x, y=lm.y, z=getattr(lm, "z", 0.0), visibility=getattr(lm, "visibility", 0.0))
    return proto
