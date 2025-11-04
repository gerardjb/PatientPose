# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from analysis_tools.landmark_utils import (
    extract_landmarks_for_frame,
    INDEX_FINGER_TIP_INDEX,
    THUMB_TIP_INDEX,
)
from video_tools import blur_face_with_pose, determine_rotation_code, rotate_frame

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


def draw_right_hand_fingertips(
    frame_rgb: np.ndarray, hand_result: mp.tasks.vision.HandLandmarkerResult
) -> np.ndarray:
    """Overlay right-hand index and thumb tips as small white circles."""
    if not hand_result.hand_landmarks or not hand_result.handedness:
        return frame_rgb

    annotated = np.copy(frame_rgb)
    height, width, _ = annotated.shape

    for handedness_list, landmarks in zip(hand_result.handedness, hand_result.hand_landmarks):
        if not handedness_list:
            continue
        if handedness_list[0].category_name.lower() != "right":
            continue

        for landmark_idx, landmark in enumerate(landmarks):
            if landmark_idx not in (INDEX_FINGER_TIP_INDEX, THUMB_TIP_INDEX):
                continue

            cx = int(landmark.x * width)
            cy = int(landmark.y * height)
            if 0 <= cx < width and 0 <= cy < height:
                cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1)

    return annotated


def process_video(
    video_path: Path, hand_model_path: Path, pose_model_path: Path, rotation_code: int | None
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
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=VisionRunningMode.VIDEO,
    )

    frame_index = 0
    all_landmarks: List[dict] = []

    with HandLandmarker.create_from_options(hand_options) as handmarker, PoseLandmarker.create_from_options(
        pose_options
    ) as posemarker:
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

            frame_landmarks = extract_landmarks_for_frame(frame_index, timestamp_ms, hand_result, pose_result)
            all_landmarks.extend(frame_landmarks)

            anonymized_frame = blur_face_with_pose(frame_rgb, pose_result)
            annotated_frame = draw_right_hand_fingertips(anonymized_frame, hand_result)
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

    rotation_code = determine_rotation_code(video_path, pose_model_path, args.rotate, args.auto_orient)
    process_video(video_path, hand_model_path, pose_model_path, rotation_code)


if __name__ == "__main__":
    main()
