#!/usr/bin/env python
"""
Probe pose quality metrics for specific frames and rotations.

This script inspects a small set of “bad” and “good” frame indices of the
PXL_20251106_173128817_identifiable_and_rotated.mp4 sample by running the
PoseLandmarker and reporting the detailed PoseQuality breakdown for every
tested rotation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from video_tools.orientation import ORIENTATION_ROTATIONS, rotate_frame
from video_tools.pose_quality import PoseQuality, PoseQualityScorer

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
DRAWING_UTILS = mp.solutions.drawing_utils
LANDMARK_STYLE = DRAWING_UTILS.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
CONNECTION_STYLE = DRAWING_UTILS.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)


ROTATION_LABELS = {
    None: "no rotation",
    cv2.ROTATE_90_CLOCKWISE: "90° clockwise",
    cv2.ROTATE_90_COUNTERCLOCKWISE: "90° counter-clockwise",
    cv2.ROTATE_180: "180°",
}


@dataclass
class RotationResult:
    rotation_code: int | None
    detected: bool
    quality: PoseQuality | None
    source: str
    landmarks: List[object] | None = None


def parse_args() -> argparse.Namespace:
    default_video = REPO_ROOT / "sample_data" / "PXL_20251106_173128817_identifiable_and_rotated.mp4"
    default_model = REPO_ROOT / "models" / "pose_landmarker.task"
    parser = argparse.ArgumentParser(
        description="Inspect pose quality metrics for selected frames and rotations."
    )
    parser.add_argument("--video", type=Path, default=default_video, help="Video to inspect.")
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=default_model,
        help="Path to pose_landmarker.task.",
    )
    parser.add_argument(
        "--bad-indices",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30],
        help="Zero-based indices expected to be low quality.",
    )
    parser.add_argument(
        "--good-indices",
        type=int,
        nargs="+",
        default=[75, 85, 95],
        help="Zero-based indices expected to be high quality.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="For debugging: sample every Nth frame around each requested index.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Number of neighboring frames to include on each side of the target index.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=REPO_ROOT / "results" / "pose_quality_debug.mp4",
        help="Optional path to write annotated frames (set empty to skip).",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=4.0,
        help="Frame rate for the debug video (if enabled).",
    )
    return parser.parse_args()


def rotation_label(rotation_code: int | None) -> str:
    return ROTATION_LABELS.get(rotation_code, str(rotation_code))


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    if frame_index < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


def evaluate_frame(
    pose_marker: PoseLandmarker,
    frame_bgr: np.ndarray,
    scorer: PoseQualityScorer,
    rotations: Sequence[int | None],
) -> List[RotationResult]:
    results: List[RotationResult] = []
    for rotation_code in rotations:
        rotated = rotate_frame(frame_bgr, rotation_code)
        frame_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        landmarker_result = pose_marker.detect(mp_image)
        if not landmarker_result.pose_landmarks:
            results.append(
                RotationResult(rotation_code=rotation_code, detected=False, quality=None, source="original")
            )
            continue
        quality = scorer.score(landmarker_result)
        results.append(
            RotationResult(
                rotation_code=rotation_code,
                detected=True,
                quality=quality,
                source="original",
                landmarks=landmarker_result.pose_landmarks[0],
            )
        )
    return results


def print_rotation_results(frame_index: int, expectation: str, results: Iterable[RotationResult]) -> None:
    print(f"\n=== Frame {frame_index} ({expectation}) ===")
    for result in results:
        label = rotation_label(result.rotation_code)
        if not result.detected or result.quality is None:
            print(f"  {label:<18} -> pose not detected")
            continue
        q = result.quality
        print(
            f"  {label:<18} -> detected"
            f" | score={q.score:6.2f}"
            f" | good={str(q.is_good):>5}"
            f" | angle={q.angle_deg:5.1f}°"
            f" | vertical={q.vertical_deg:5.1f}°"
            f" | visibility_mean={q.visibility_mean:4.2f}"
            f" | landmarks={q.landmark_count:3d}"
            f" | nose_above={q.nose_above_shoulders}"
        )


def gather_indices(indices: Sequence[int], window: int, stride: int) -> List[int]:
    expanded: List[int] = []
    for idx in indices:
        for offset in range(-window, window + 1, stride):
            expanded.append(idx + offset)
    return sorted(set(expanded))


def inverse_rotation_code(rotation_code: int | None) -> int | None:
    if rotation_code == cv2.ROTATE_90_CLOCKWISE:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if rotation_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.ROTATE_90_CLOCKWISE
    return rotation_code


def _landmarks_to_proto(landmarks: Sequence[object]) -> landmark_pb2.NormalizedLandmarkList:
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(getattr(lm, "z", 0.0)),
                visibility=float(getattr(lm, "visibility", 0.0)),
            )
            for lm in landmarks
        ]
    )
    return proto


def draw_pose_on_frame(frame_bgr: np.ndarray, result: RotationResult, frame_index: int) -> np.ndarray:
    overlay = frame_bgr.copy()
    if result.detected and result.landmarks:
        proto = _landmarks_to_proto(result.landmarks)
        DRAWING_UTILS.draw_landmarks(
            overlay,
            proto,
            POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_STYLE,
            connection_drawing_spec=CONNECTION_STYLE,
        )

    text_color = (255, 255, 255)
    lines = [
        f"Frame {frame_index}",
        f"Rotation: {rotation_label(result.rotation_code)}",
    ]
    if result.quality:
        lines.append(f"Score: {result.quality.score:.2f} | Good: {result.quality.is_good}")
        lines.append(
            f"Angle: {result.quality.angle_deg:.1f}° | Vertical: {result.quality.vertical_deg:.1f}°"
        )
        lines.append(
            f"Visibility: {result.quality.visibility_mean:.2f} | Landmarks: {result.quality.landmark_count}"
        )
    else:
        lines.append("Pose not detected")

    for i, line in enumerate(lines):
        origin = (10, 30 + i * 30)
        cv2.putText(
            overlay,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def annotate_and_restore(frame_bgr: np.ndarray, result: RotationResult, frame_index: int) -> np.ndarray:
    rotated = rotate_frame(frame_bgr, result.rotation_code)
    annotated = draw_pose_on_frame(rotated, result, frame_index)
    inverse_code = inverse_rotation_code(result.rotation_code)
    restored = rotate_frame(annotated, inverse_code)
    return restored


def ensure_video_writer(
    writer: cv2.VideoWriter | None,
    output_path: Path,
    frame_shape: tuple[int, int, int],
    fps: float,
) -> cv2.VideoWriter:
    if writer is not None and writer.isOpened():
        return writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer at {output_path}")
    return writer


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found at {args.video}")
    if not args.pose_model.is_file():
        raise FileNotFoundError(f"Pose model not found at {args.pose_model}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {args.video}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.pose_model)),
        running_mode=VisionRunningMode.IMAGE,
    )
    scorer = PoseQualityScorer()

    bad_indices = gather_indices(args.bad_indices, args.window, args.stride)
    good_indices = gather_indices(args.good_indices, args.window, args.stride)

    writer: cv2.VideoWriter | None = None
    output_video_path = Path(args.output_video) if args.output_video else None

    try:
        with PoseLandmarker.create_from_options(options) as pose_marker:
            for expectation, indices in (("bad", bad_indices), ("good", good_indices)):
                for frame_index in indices:
                    frame = read_frame(cap, frame_index)
                    if frame is None:
                        print(f"\n=== Frame {frame_index} ({expectation}) ===")
                        print("  Unable to read frame from video.")
                        continue
                    results = evaluate_frame(pose_marker, frame, scorer, ORIENTATION_ROTATIONS)
                    print_rotation_results(frame_index, expectation, results)
                    if output_video_path:
                        writer = ensure_video_writer(writer, output_video_path, frame.shape, args.output_fps)
                        for result in results:
                            annotated = annotate_and_restore(frame, result, frame_index)
                            writer.write(annotated)
    finally:
        cap.release()
        if writer:
            writer.release()


if __name__ == "__main__":
    main()
