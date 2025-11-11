# -*- coding: utf-8 -*-
"""
Utilities for inferring the upright orientation of video frames.
"""
from pathlib import Path
from typing import List, Tuple
import os

import cv2
import numpy as np

os.environ.setdefault("MEDIAPIPE_SKIP_AUDIO", "1")

import mediapipe as mp

from .orientation_focus import OrientationAnalyzer, OrientationAnalyzerConfig
from .pose_focus import PoseFocusHint
from .pose_quality import PoseQuality, PoseQualityScorer

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

ORIENTATION_ROTATIONS: List[int | None] = [
    None,
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_90_COUNTERCLOCKWISE,
    cv2.ROTATE_180,
]

ORIENTATION_SAMPLE_LIMIT = 150  # frames to scan before giving up


def rotate_frame(frame: np.ndarray, rotation_code: int | None) -> np.ndarray:
    """Return the frame rotated according to the provided code."""
    if rotation_code is None:
        return frame
    return cv2.rotate(frame, rotation_code)


def infer_orientation_from_frame(frame_bgr: np.ndarray, pose_model_path: Path) -> Tuple[int | None, str]:
    """
    Infer the upright orientation for a single frame using pose landmarks.

    Returns a tuple of (rotation_code, source_label) where source_label is one of:
        - "pose-original": pose detection succeeded on the original frame.
        - "pose-upscaled": detection only succeeded after a 2x upscale fallback.
        - "none": no pose detected even after fallback.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, "none"

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=VisionRunningMode.IMAGE,
    )
    scorer = PoseQualityScorer()
    with PoseLandmarker.create_from_options(options) as pose_marker:
        rotation, quality = _best_rotation_for_frame(pose_marker, frame_bgr, scorer)
        if rotation is not None and quality is not None:
            return rotation, "pose-original"

        upscaled = cv2.resize(frame_bgr, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        rotation_up, quality_up = _best_rotation_for_frame(pose_marker, upscaled, scorer)
        if rotation_up is not None and quality_up is not None:
            return rotation_up, "pose-upscaled"

    return None, "none"


def _best_rotation_for_frame(
    pose_marker: PoseLandmarker, frame_bgr: np.ndarray, scorer: PoseQualityScorer
) -> Tuple[int | None, PoseQuality | None]:
    best_rotation = None
    best_quality = None
    for rotation_code in ORIENTATION_ROTATIONS:
        rotated = rotate_frame(frame_bgr, rotation_code)
        frame_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = pose_marker.detect(mp_image)
        if not result.pose_landmarks:
            continue
        quality = scorer.score(result)
        if quality is None:
            continue
        if best_quality is None or quality.score < best_quality.score:
            best_quality = quality
            best_rotation = rotation_code
    return best_rotation, best_quality


def _rotation_label(rotation_code: int | None) -> str:
    mapping = {
        None: "no rotation",
        cv2.ROTATE_90_CLOCKWISE: "90° clockwise",
        cv2.ROTATE_90_COUNTERCLOCKWISE: "90° counter-clockwise",
        cv2.ROTATE_180: "180°",
    }
    return mapping.get(rotation_code, "unknown rotation")


def determine_rotation_code(
    video_path: Path,
    pose_model_path: Path,
    rotate_flag: bool,
    auto_orient: bool,
    orientation_max_scan: int | None = None,
    orientation_debug: bool = False,
    orientation_debug_dir: Path | None = None,
    orientation_good_target: int | None = None,
    orientation_min_detections: int | None = None,
    return_details: bool = False,
) -> int | None | tuple[int | None, PoseFocusHint | None]:
    """
    Determine the rotation code to use for processing a video.
    Respects manual overrides and optionally infers rotation from the first frame.
    """
    if rotate_flag:
        if auto_orient:
            print("Manual rotation flag overrides auto-orientation request.")
        print("Using manual 90° clockwise rotation.")
        if return_details:
            return cv2.ROTATE_90_CLOCKWISE, None
        return cv2.ROTATE_90_CLOCKWISE

    if not auto_orient:
        return (None, None) if return_details else None

    max_scan = orientation_max_scan or ORIENTATION_SAMPLE_LIMIT
    analyzer_config = OrientationAnalyzerConfig(
        rotation_codes=ORIENTATION_ROTATIONS,
        max_scan_frames=max_scan,
        primary_stride=15,
        anchor_radius=3,
        debug_enabled=orientation_debug,
        debug_output_dir=orientation_debug_dir,
        good_pose_target=orientation_good_target or 5,
        min_detected_rotations=orientation_min_detections or 2,
    )
    analyzer = OrientationAnalyzer(analyzer_config)
    decision = analyzer.analyze(video_path, pose_model_path)

    if decision.rotation_code is None:
        print("Auto-orientation found orientation was correct, landmark hints available") if return_details else print("Auto-Orientation could not resolve orientation, no landmark hints available.")
        _maybe_report_debug_path(video_path, orientation_debug, orientation_debug_dir)
        return (None, decision.focus_hint) if return_details else None

    label = _rotation_label(decision.rotation_code)
    reason = decision.reason
    if reason.startswith("vote:"):
        print(f"Auto-orientation selected {label} after pose voting ({reason}).")
    elif reason == "anchor":
        print(f"Auto-orientation selected {label} anchored on the first good pose instance.")
    elif reason == "best-score":
        print(f"Auto-orientation selected {label} based on the best scored pose.")
    else:
        print(f"Auto-orientation selected {label} without pose confidence.")

    _maybe_report_debug_path(video_path, orientation_debug, orientation_debug_dir)
    if return_details:
        return decision.rotation_code, decision.focus_hint
    return decision.rotation_code


def _maybe_report_debug_path(video_path: Path, debug_enabled: bool, debug_dir: Path | None) -> None:
    if not debug_enabled or not debug_dir:
        return
    debug_path = debug_dir / f"{video_path.stem}_orientation.json"
    if debug_path.exists():
        print(f"Orientation debug saved to {debug_path}")


__all__ = [
    "determine_rotation_code",
    "infer_orientation_from_frame",
    "rotate_frame",
]
