# -*- coding: utf-8 -*-
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
from video_tools import blur_face_with_pose, determine_rotation_code, rotate_frame
from video_tools.pose_focus import (
    PoseFocusHint,
    PoseFocusTracker,
    crop_frame_from_bbox,
    remap_landmarks_from_crop,
)
from video_tools.pose_quality import PoseQualityScorer

# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results"
VIDEO_DIR = OUTPUT_DIR / "OutputVideos"
CSV_DIR = OUTPUT_DIR / "OutputCSVs"
DEFAULT_SAMPLE_DIR = BASE_DIR / "sample_data"
DEFAULT_MODEL_DIR = BASE_DIR / "models"
DEFAULT_VIDEO_NAME = "20250408_fingerTap_decrement.mp4"
HAND_MODEL_FILENAME = "hand_landmarker.task"
POSE_MODEL_FILENAME = "pose_landmarker.task"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe task shortcuts
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def process_video(
    video_path: Path,
    hand_model_path: Path,
    pose_model_path: Path,
    rotation_code: int | None,
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

    video_name_tag = video_path.stem
    output_video_path = VIDEO_DIR / f"deidentified_{video_name_tag}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (output_width, output_height))

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
            pose_quality = (
                pose_quality_scorer.score(pose_result) if pose_result.pose_landmarks else None
            )

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

            anonymized_frame = blur_face_with_pose(frame_rgb, pose_result)
            annotated_frame = draw_pose_landmarks(anonymized_frame, pose_result)
            writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            frame_index += 1
            if frame_index % 50 == 0:
                print(f"Processed {frame_index} frames.")

    cap.release()
    writer.release()

    if all_landmarks:
        landmarks_df = pd.DataFrame(all_landmarks)
        output_csv_path = CSV_DIR / f"landmarks_{video_name_tag}.csv"
        landmarks_df.to_csv(output_csv_path, index=False)
        print(f"Landmark data saved to {output_csv_path}")

    print(f"De-identified video saved to {output_video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process video while de-identifying faces.")
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
        help=f"Directory containing sample videos. Defaults to {DEFAULT_SAMPLE_DIR}.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help=f"Directory containing MediaPipe models. Defaults to {DEFAULT_MODEL_DIR}.",
    )
    parser.add_argument(
        "-r",
        "--rotate",
        action="store_true",
        help="Force a 90° clockwise rotation before processing.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else DEFAULT_MODEL_DIR
    video_dir = Path(args.video_dir).resolve() if args.video_dir else DEFAULT_SAMPLE_DIR

    hand_model_path = model_dir / HAND_MODEL_FILENAME
    pose_model_path = model_dir / POSE_MODEL_FILENAME
    if not hand_model_path.is_file():
        raise FileNotFoundError(f"Hand model file not found at {hand_model_path}")
    if not pose_model_path.is_file():
        raise FileNotFoundError(f"Pose model file not found at {pose_model_path}")

    if args.filename:
        video_path = Path(args.filename).resolve()
    else:
        video_path = (video_dir / DEFAULT_VIDEO_NAME).resolve()
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
        orientation_debug_dir=OUTPUT_DIR / "orientation_debug",
        orientation_good_target=args.orientation_good_target,
        orientation_min_detections=args.orientation_min_detections,
        return_details=True,
    )
    process_video(video_path, hand_model_path, pose_model_path, rotation_code, pose_focus_hint=pose_focus_hint)


if __name__ == "__main__":
    main()
