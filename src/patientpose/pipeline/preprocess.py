from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

os.environ.setdefault("MEDIAPIPE_SKIP_AUDIO", "1")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from analysis_tools.landmark_utils import extract_landmarks_for_frame
from image_overlays import draw_pose_landmarks
from patientpose.artifacts import ArtifactStore, PreprocessVideoArtifacts
from patientpose.config import ProjectPaths, resolve_project_paths
from video_tools import blur_face_with_pose, determine_rotation_code, rotate_frame
from video_tools.pose_focus import (
    PoseFocusHint,
    PoseFocusTracker,
    crop_frame_from_bbox,
    remap_landmarks_from_crop,
)
from video_tools.pose_quality import PoseQualityScorer

DEFAULT_VIDEO_NAME = "20250408_fingerTap_decrement.mp4"
HAND_MODEL_FILENAME = "hand_landmarker.task"
POSE_MODEL_FILENAME = "pose_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LANDMARK_COLUMNS = [
    "frame",
    "timestamp_ms",
    "source",
    "instance_id",
    "handedness",
    "landmark_id",
    "landmark_name",
    "x",
    "y",
    "z",
    "visibility",
]
FRAME_SUMMARY_COLUMNS = [
    "frame",
    "timestamp_ms",
    "pose_detected",
    "num_pose_landmarks",
    "hand_detected",
    "num_hand_landmarks",
    "pose_quality_score",
]


def _pose_quality_value(pose_quality) -> float:
    if pose_quality is None:
        return float("nan")
    for attr in ("score", "overall_score", "value"):
        value = getattr(pose_quality, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def add_preprocess_video_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=False,
        type=str,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        help="Directory containing sample videos. Defaults to <project-root>/sample_data.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing MediaPipe models. Defaults to <project-root>/models.",
    )
    parser.add_argument(
        "-r",
        "--rotate",
        action="store_true",
        help="Force a 90 degree clockwise rotation before processing.",
    )
    parser.add_argument(
        "--auto-orient",
        action="store_false",
        help="Attempt to infer the upright orientation from the first frame.",
    )
    parser.add_argument(
        "--orientation-max-scan",
        type=int,
        help="Maximum number of frames to scan while auto-orienting (default 150).",
    )
    parser.add_argument(
        "--orientation-debug",
        action="store_true",
        help="Enable verbose orientation diagnostics and JSON summaries.",
    )
    parser.add_argument(
        "--orientation-good-target",
        type=int,
        help="Number of good pose frames required before locking orientation (default 5).",
    )
    parser.add_argument(
        "--orientation-min-detections",
        type=int,
        help="Minimum number of rotations that must detect a pose on a frame for it to count (default 2).",
    )
    return parser


def build_preprocess_video_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process video while de-identifying faces.")
    return add_preprocess_video_args(parser)


def resolve_video_input(
    *,
    filename: str | None,
    video_dir: str | None,
    paths: ProjectPaths,
) -> Path:
    if filename:
        return Path(filename).resolve()
    base_dir = Path(video_dir).resolve() if video_dir else paths.sample_data
    return (base_dir / DEFAULT_VIDEO_NAME).resolve()


def resolve_model_paths(model_dir: str | None, paths: ProjectPaths) -> tuple[Path, Path]:
    resolved_model_dir = Path(model_dir).resolve() if model_dir else paths.models
    return resolved_model_dir / HAND_MODEL_FILENAME, resolved_model_dir / POSE_MODEL_FILENAME


def process_video(
    video_path: Path,
    hand_model_path: Path,
    pose_model_path: Path,
    rotation_code: int | None,
    artifacts: PreprocessVideoArtifacts,
    *,
    pose_focus_hint: PoseFocusHint | None = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if rotation_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        output_width = original_height
        output_height = original_width
    else:
        output_width = original_width
        output_height = original_height

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(artifacts.annotated_video), fourcc, fps, (output_width, output_height))
    plain_writer = cv2.VideoWriter(str(artifacts.plain_video), fourcc, fps, (output_width, output_height))

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )
    pose_video_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=VisionRunningMode.VIDEO,
    )
    pose_image_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=VisionRunningMode.IMAGE,
    )

    frame_index = 0
    all_landmarks: List[dict] = []
    frame_summaries: List[dict] = []

    focus_tracker = PoseFocusTracker(pose_focus_hint) if pose_focus_hint else None
    pose_quality_scorer = PoseQualityScorer()

    with HandLandmarker.create_from_options(hand_options) as handmarker, PoseLandmarker.create_from_options(
        pose_video_options
    ) as posemarker, PoseLandmarker.create_from_options(pose_image_options) as posemarker_image:
        print("Hand and pose landmarkers initialized.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("End of video reached.")
                else:
                    print("Failed to read frame due to an issue with the video file.")
                break

            frame = rotate_frame(frame, rotation_code)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * (1000.0 / fps))
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            hand_result = handmarker.detect_for_video(mp_frame, timestamp_ms)
            pose_result = posemarker.detect_for_video(mp_frame, timestamp_ms)
            pose_quality = pose_quality_scorer.score(pose_result) if pose_result.pose_landmarks else None

            if (not pose_result.pose_landmarks or not (pose_quality and pose_quality.is_good)) and focus_tracker:
                bbox = focus_tracker.current_bbox()
                if bbox:
                    crop_data = crop_frame_from_bbox(frame_rgb, bbox)
                    if crop_data:
                        crop_rgb, transform = crop_data
                        crop_rgb = np.ascontiguousarray(crop_rgb)
                        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                        crop_result = posemarker_image.detect(mp_crop)
                        if crop_result.pose_landmarks:
                            remap_landmarks_from_crop(crop_result.pose_landmarks[0], transform)
                            pose_result = crop_result
                            pose_quality = pose_quality_scorer.score(pose_result)

            if focus_tracker:
                if pose_result.pose_landmarks:
                    focus_tracker.register_success(pose_result.pose_landmarks[0])
                else:
                    focus_tracker.register_failure()

            frame_landmarks = extract_landmarks_for_frame(frame_index, timestamp_ms, hand_result, pose_result)
            all_landmarks.extend(frame_landmarks)
            num_pose_landmarks = 0
            if pose_result.pose_landmarks:
                num_pose_landmarks = sum(len(landmarks) for landmarks in pose_result.pose_landmarks)
            num_hand_landmarks = 0
            if hand_result.hand_landmarks:
                num_hand_landmarks = sum(len(landmarks) for landmarks in hand_result.hand_landmarks)
            frame_summaries.append(
                {
                    "frame": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "pose_detected": bool(pose_result.pose_landmarks),
                    "num_pose_landmarks": int(num_pose_landmarks),
                    "hand_detected": bool(hand_result.hand_landmarks),
                    "num_hand_landmarks": int(num_hand_landmarks),
                    "pose_quality_score": _pose_quality_value(pose_quality),
                }
            )

            anonymized_frame = blur_face_with_pose(frame_rgb, pose_result)
            plain_writer.write(cv2.cvtColor(anonymized_frame, cv2.COLOR_RGB2BGR))
            annotated_frame = draw_pose_landmarks(anonymized_frame, pose_result)
            writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            frame_index += 1
            if frame_index % 50 == 0:
                print(f"Processed {frame_index} frames.")

    cap.release()
    writer.release()
    plain_writer.release()

    landmarks_df = pd.DataFrame(all_landmarks, columns=LANDMARK_COLUMNS)
    landmarks_df.to_csv(artifacts.landmarks_csv, index=False)
    if all_landmarks:
        print(f"Landmark data saved to {artifacts.landmarks_csv}")
    else:
        print(f"No landmarks detected; wrote empty landmark CSV to {artifacts.landmarks_csv}")

    frame_summary_df = pd.DataFrame(frame_summaries, columns=FRAME_SUMMARY_COLUMNS)
    frame_summary_df.to_csv(artifacts.frame_summary_csv, index=False)
    print(f"Frame summary saved to {artifacts.frame_summary_csv}")

    print(f"De-identified video saved to {artifacts.annotated_video}")
    print(f"De-identified video without keypoints saved to {artifacts.plain_video}")


def run_preprocess_video(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    hand_model_path, pose_model_path = resolve_model_paths(args.model_dir, paths)
    if not hand_model_path.is_file():
        raise FileNotFoundError(f"Hand model file not found at {hand_model_path}")
    if not pose_model_path.is_file():
        raise FileNotFoundError(f"Pose model file not found at {pose_model_path}")

    video_path = resolve_video_input(filename=args.filename, video_dir=args.video_dir, paths=paths)
    if args.filename is None:
        print(f"Using default video file provided, {video_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found at {video_path}")

    rotation_code, pose_focus_hint = determine_rotation_code(
        video_path,
        pose_model_path,
        args.rotate,
        args.auto_orient,
        orientation_max_scan=args.orientation_max_scan,
        orientation_debug=args.orientation_debug,
        orientation_debug_dir=paths.orientation_debug,
        orientation_good_target=args.orientation_good_target,
        orientation_min_detections=args.orientation_min_detections,
        return_details=True,
    )

    artifacts = artifact_store.preprocess_video(video_path)
    process_video(
        video_path,
        hand_model_path,
        pose_model_path,
        rotation_code,
        artifacts,
        pose_focus_hint=pose_focus_hint,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_preprocess_video_parser()
    args = parser.parse_args(argv)
    run_preprocess_video(args)


if __name__ == "__main__":
    main()
