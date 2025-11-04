# -*- coding: utf-8 -*-
"""
Utilities for inferring the upright orientation of video frames.
"""
from pathlib import Path
import math
from typing import List

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
PoseLandmark = mp.solutions.pose.PoseLandmark

ORIENTATION_ROTATIONS: List[int | None] = [
    None,
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_90_COUNTERCLOCKWISE,
    cv2.ROTATE_180,
]


def rotate_frame(frame: np.ndarray, rotation_code: int | None) -> np.ndarray:
    """Return the frame rotated according to the provided code."""
    if rotation_code is None:
        return frame
    return cv2.rotate(frame, rotation_code)


def _orientation_score(pose_result: PoseLandmarkerResult) -> float:
    """Calculate a score representing how upright the pose appears."""
    if not pose_result.pose_landmarks:
        return float("inf")

    landmarks = pose_result.pose_landmarks[0]
    try:
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[PoseLandmark.NOSE.value]
    except IndexError:
        return float("inf")

    dx = right_shoulder.x - left_shoulder.x
    dy = right_shoulder.y - left_shoulder.y
    angle_deg = math.degrees(math.atan2(dy, dx))
    angle_score = min(abs(angle_deg), abs(abs(angle_deg) - 180.0))

    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) * 0.5
    nose_penalty = 0.0 if nose.y < shoulder_mid_y else 90.0

    return angle_score + nose_penalty


def infer_orientation_from_frame(frame_bgr: np.ndarray, pose_model_path: Path) -> int | None:
    """
    Infer the upright orientation for a single frame using pose landmarks.
    Returns the rotation code that best aligns the person upright.
    """
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=VisionRunningMode.IMAGE,
    )
    with PoseLandmarker.create_from_options(options) as pose_marker:
        best_rotation = None
        best_score = float("inf")

        for rotation_code in ORIENTATION_ROTATIONS:
            rotated = rotate_frame(frame_bgr, rotation_code)
            frame_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = pose_marker.detect(mp_image)
            score = _orientation_score(result)
            if score < best_score:
                best_score = score
                best_rotation = rotation_code

    if best_score == float("inf"):
        return None
    return best_rotation


def _rotation_label(rotation_code: int | None) -> str:
    mapping = {
        None: "no rotation",
        cv2.ROTATE_90_CLOCKWISE: "90° clockwise",
        cv2.ROTATE_90_COUNTERCLOCKWISE: "90° counter-clockwise",
        cv2.ROTATE_180: "180°",
    }
    return mapping.get(rotation_code, "unknown rotation")


def determine_rotation_code(
    video_path: Path, pose_model_path: Path, rotate_flag: bool, auto_orient: bool
) -> int | None:
    """
    Determine the rotation code to use for processing a video.
    Respects manual overrides and optionally infers rotation from the first frame.
    """
    if rotate_flag:
        if auto_orient:
            print("Manual rotation flag overrides auto-orientation request.")
        print("Using manual 90° clockwise rotation.")
        return cv2.ROTATE_90_CLOCKWISE

    if not auto_orient:
        return None

    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("Warning: Unable to read first frame for auto-orientation; defaulting to no rotation.")
        return None

    inferred = infer_orientation_from_frame(first_frame, pose_model_path)
    if inferred is None:
        print("Auto-orientation could not determine rotation; defaulting to no rotation.")
        return None

    print(f"Auto-orientation selected {_rotation_label(inferred)}.")
    return inferred


__all__ = [
    "determine_rotation_code",
    "infer_orientation_from_frame",
    "rotate_frame",
]
