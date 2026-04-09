from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

os.environ.setdefault("MEDIAPIPE_SKIP_AUDIO", "1")

import cv2
import matplotlib
import mediapipe as mp
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_tools.landmark_utils import (
    INDEX_FINGER_TIP_INDEX,
    THUMB_TIP_INDEX,
    extract_pose_world_landmarks_for_frame,
    extract_landmarks_for_frame,
)
from image_overlays import draw_pose_landmarks
from patientpose.artifacts import ArtifactStore, PreprocessVideoArtifacts, QualityVideoArtifacts
from patientpose.config import ProjectPaths, resolve_project_paths
from patientpose.landmarks import (
    FRAME_SUMMARY_COLUMNS,
    IMAGE_LANDMARK_COLUMNS,
    QUALITY_LANDMARK_COLUMNS,
    WORLD_LANDMARK_COLUMNS,
)
from video_tools import blur_face_with_pose, determine_rotation_code, rotate_frame
from video_tools.image_quality_utils import (
    calculate_confidence_score,
    calculate_local_laplacian_variance,
    calculate_local_motion,
)
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
QUALITY_PATCH_SIZE = 21
QUALITY_VIDEO_FPS = 5

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

ROTATION_LABELS = {
    None: "none",
    cv2.ROTATE_90_CLOCKWISE: "90cw",
    cv2.ROTATE_90_COUNTERCLOCKWISE: "90ccw",
    cv2.ROTATE_180: "180",
}


def _pose_quality_value(pose_quality) -> float:
    if pose_quality is None:
        return float("nan")
    for attr in ("score", "overall_score", "value"):
        value = getattr(pose_quality, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def _rotation_label(rotation_code: int | None) -> str:
    return ROTATION_LABELS.get(rotation_code, "unknown")


def _crop_summary_fields(crop_transform) -> dict[str, float]:
    if crop_transform is None:
        return {
            "crop_left": float("nan"),
            "crop_top": float("nan"),
            "crop_width": float("nan"),
            "crop_height": float("nan"),
            "crop_frame_width": float("nan"),
            "crop_frame_height": float("nan"),
            "crop_scale_x": float("nan"),
            "crop_scale_y": float("nan"),
            "crop_scale": float("nan"),
        }
    return {
        "crop_left": float(crop_transform.left),
        "crop_top": float(crop_transform.top),
        "crop_width": float(crop_transform.width),
        "crop_height": float(crop_transform.height),
        "crop_frame_width": float(crop_transform.frame_width),
        "crop_frame_height": float(crop_transform.frame_height),
        "crop_scale_x": float(crop_transform.scale_x),
        "crop_scale_y": float(crop_transform.scale_y),
        "crop_scale": float(crop_transform.scale),
    }


def _write_processing_metadata(
    *,
    video_path: Path,
    rotation_code: int | None,
    metadata_path: Path,
    mode: str,
    orientation_source: str,
    landmarks_csv: Path,
    pose_world_csv: Path | None,
    annotated_video: Path,
    plain_video: Path,
    extra: dict | None = None,
) -> None:
    payload = {
        "video_stem": video_path.stem,
        "source_video": str(video_path.resolve()),
        "mode": mode,
        "orientation_source": orientation_source,
        "rotation_label": _rotation_label(rotation_code),
        "landmarks_csv": str(landmarks_csv.resolve()),
        "pose_world_csv": str(pose_world_csv.resolve()) if pose_world_csv is not None else None,
        "annotated_video": str(annotated_video.resolve()),
        "plain_video": str(plain_video.resolve()),
    }
    if extra:
        payload.update(extra)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2))
    print(f"Processing metadata saved to {metadata_path}")


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


def add_preprocess_quality_video_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
        help="Rotate the video 90 degrees clockwise before processing.",
    )
    return parser


def build_preprocess_quality_video_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process video into quality-visualization artifacts.")
    return add_preprocess_quality_video_args(parser)


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


def resolve_hand_model_path(model_dir: str | None, paths: ProjectPaths) -> Path:
    resolved_model_dir = Path(model_dir).resolve() if model_dir else paths.models
    return resolved_model_dir / HAND_MODEL_FILENAME


def process_video(
    video_path: Path,
    hand_model_path: Path,
    pose_model_path: Path,
    rotation_code: int | None,
    artifacts: PreprocessVideoArtifacts,
    *,
    pose_focus_hint: PoseFocusHint | None = None,
    orientation_source: str = "auto",
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
    all_world_landmarks: List[dict] = []
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

            pose_source = "full_frame"
            crop_transform = None

            if (not pose_result.pose_landmarks or not (pose_quality and pose_quality.is_good)) and focus_tracker:
                bbox = focus_tracker.current_bbox()
                if bbox:
                    crop_data = crop_frame_from_bbox(frame_rgb, bbox)
                    if crop_data:
                        crop_rgb, crop_transform = crop_data
                        crop_rgb = np.ascontiguousarray(crop_rgb)
                        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                        crop_result = posemarker_image.detect(mp_crop)
                        if crop_result.pose_landmarks:
                            remap_landmarks_from_crop(crop_result.pose_landmarks[0], crop_transform)
                            pose_result = crop_result
                            pose_quality = pose_quality_scorer.score(pose_result)
                            pose_source = "focus_crop"

            if focus_tracker:
                if pose_result.pose_landmarks:
                    focus_tracker.register_success(pose_result.pose_landmarks[0])
                else:
                    focus_tracker.register_failure()

            frame_landmarks = extract_landmarks_for_frame(
                frame_index,
                timestamp_ms,
                hand_result,
                pose_result,
                pose_source=pose_source,
                crop_transform=crop_transform,
            )
            frame_world_landmarks = extract_pose_world_landmarks_for_frame(
                frame_index,
                timestamp_ms,
                pose_result,
                pose_source=pose_source,
                crop_transform=crop_transform,
            )
            all_landmarks.extend(frame_landmarks)
            all_world_landmarks.extend(frame_world_landmarks)
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
                    "pose_source": pose_source if pose_result.pose_landmarks else "missing",
                    **_crop_summary_fields(crop_transform if pose_source == "focus_crop" else None),
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

    landmarks_df = pd.DataFrame(all_landmarks, columns=IMAGE_LANDMARK_COLUMNS)
    landmarks_df.to_csv(artifacts.landmarks_csv, index=False)
    if all_landmarks:
        print(f"Landmark data saved to {artifacts.landmarks_csv}")
    else:
        print(f"No landmarks detected; wrote empty landmark CSV to {artifacts.landmarks_csv}")

    world_landmarks_df = pd.DataFrame(all_world_landmarks, columns=WORLD_LANDMARK_COLUMNS)
    world_landmarks_df.to_csv(artifacts.pose_world_csv, index=False)
    if all_world_landmarks:
        print(f"World-space pose data saved to {artifacts.pose_world_csv}")
    else:
        print(f"No world-space pose data detected; wrote empty pose-world CSV to {artifacts.pose_world_csv}")

    frame_summary_df = pd.DataFrame(frame_summaries, columns=FRAME_SUMMARY_COLUMNS)
    frame_summary_df.to_csv(artifacts.frame_summary_csv, index=False)
    print(f"Frame summary saved to {artifacts.frame_summary_csv}")

    _write_processing_metadata(
        video_path=video_path,
        rotation_code=rotation_code,
        metadata_path=artifacts.metadata_json,
        mode="preprocess-video",
        orientation_source=orientation_source,
        landmarks_csv=artifacts.landmarks_csv,
        pose_world_csv=artifacts.pose_world_csv,
        annotated_video=artifacts.annotated_video,
        plain_video=artifacts.plain_video,
        extra={"frame_summary_csv": str(artifacts.frame_summary_csv.resolve())},
    )

    print(f"De-identified video saved to {artifacts.annotated_video}")
    print(f"De-identified video without keypoints saved to {artifacts.plain_video}")


def _save_quality_plots(
    landmarks_df: pd.DataFrame,
    artifacts: QualityVideoArtifacts,
    video_name_tag: str,
) -> None:
    if landmarks_df.empty:
        print("Skipping plotting: Landmark DataFrame is empty.")
        return

    right_hand_df = landmarks_df[
        (landmarks_df["source"] == "hand") & (landmarks_df["handedness"] == "Right")
    ].copy()

    index_tip_df = right_hand_df[right_hand_df["landmark_id"] == INDEX_FINGER_TIP_INDEX]
    thumb_tip_df = right_hand_df[right_hand_df["landmark_id"] == THUMB_TIP_INDEX]

    if not index_tip_df.empty or not thumb_tip_df.empty:
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        if not index_tip_df.empty:
            ax1.plot(
                index_tip_df["frame"],
                index_tip_df["y"],
                label="Right Index Tip Y",
                color="red",
                marker=".",
                linestyle="-",
                markersize=4,
            )
        if not thumb_tip_df.empty:
            ax1.plot(
                thumb_tip_df["frame"],
                thumb_tip_df["y"],
                label="Right Thumb Tip Y",
                color="blue",
                marker=".",
                linestyle="-",
                markersize=4,
            )
        ax1.set_xlabel("Frame Number", fontsize=12)
        ax1.set_ylabel("Normalized Y Position", fontsize=12)
        ax1.set_title(f"Right Fingertip Y Position Over Time ({video_name_tag})", fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.invert_yaxis()
        fig1.savefig(artifacts.position_plot, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"Position plot saved successfully as: {artifacts.position_plot}")
    else:
        print("Skipping position plot: No data found for right index or thumb tip.")

    quality_cols = ["laplacian_variance", "mean_motion_diff", "quality_score"]
    if all(col in landmarks_df.columns for col in quality_cols) and (
        not index_tip_df.empty or not thumb_tip_df.empty
    ):
        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        if not index_tip_df.empty:
            axes2[0].plot(
                index_tip_df["frame"],
                index_tip_df["laplacian_variance"],
                label="Index Tip",
                color="red",
                marker=".",
                linestyle="-",
                markersize=3,
                alpha=0.7,
            )
        if not thumb_tip_df.empty:
            axes2[0].plot(
                thumb_tip_df["frame"],
                thumb_tip_df["laplacian_variance"],
                label="Thumb Tip",
                color="blue",
                marker=".",
                linestyle="-",
                markersize=3,
                alpha=0.7,
            )
        axes2[0].set_ylabel("Laplacian Var\n(Sharpness)", fontsize=10)
        axes2[0].set_title(f"Landmark Quality Metrics Over Time ({video_name_tag})", fontsize=14)
        axes2[0].legend(fontsize=9)
        axes2[0].grid(True, linestyle="--", alpha=0.6)

        if not index_tip_df.empty:
            axes2[1].plot(
                index_tip_df["frame"],
                index_tip_df["mean_motion_diff"],
                label="Index Tip",
                color="red",
                marker=".",
                linestyle="-",
                markersize=3,
                alpha=0.7,
            )
        if not thumb_tip_df.empty:
            axes2[1].plot(
                thumb_tip_df["frame"],
                thumb_tip_df["mean_motion_diff"],
                label="Thumb Tip",
                color="blue",
                marker=".",
                linestyle="-",
                markersize=3,
                alpha=0.7,
            )
        axes2[1].set_ylabel("Mean Motion Diff\n(Motion)", fontsize=10)
        axes2[1].grid(True, linestyle="--", alpha=0.6)

        if not index_tip_df.empty:
            axes2[2].plot(
                index_tip_df["frame"],
                index_tip_df["quality_score"],
                label="Index Tip",
                color="red",
                marker=".",
                linestyle="-",
                markersize=3,
                alpha=0.7,
            )
        if not thumb_tip_df.empty:
            axes2[2].plot(
                thumb_tip_df["frame"],
                thumb_tip_df["quality_score"],
                label="Thumb Tip",
                color="blue",
                marker=".",
                linestyle="-",
                markersize=3,
                alpha=0.7,
            )
        axes2[2].set_ylabel("Quality Score", fontsize=10)
        axes2[2].set_xlabel("Frame Number", fontsize=12)
        axes2[2].grid(True, linestyle="--", alpha=0.6)

        print(f"Mean quality score for index tip: {index_tip_df['quality_score'].mean()}")
        print(f"Mean quality score for thumb tip: {thumb_tip_df['quality_score'].mean()}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig2.savefig(artifacts.quality_plot, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Quality plot saved successfully as: {artifacts.quality_plot}")
    else:
        print("Skipping quality plot: Quality columns not found in DataFrame or no landmark data.")


def process_quality_video(
    video_path: Path,
    hand_model_path: Path,
    rotation_code: int | None,
    artifacts: QualityVideoArtifacts,
    *,
    orientation_source: str = "manual",
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if rotation_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        output_width = original_height
        output_height = original_width
    else:
        output_width = original_width
        output_height = original_height

    fourcc_quality = cv2.VideoWriter_fourcc(*"MJPG")
    quality_video_writer = cv2.VideoWriter(
        str(artifacts.annotated_video),
        fourcc_quality,
        QUALITY_VIDEO_FPS,
        (output_width, output_height),
    )
    quality_video_plain_writer = cv2.VideoWriter(
        str(artifacts.plain_video),
        fourcc_quality,
        QUALITY_VIDEO_FPS,
        (output_width, output_height),
    )

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    all_landmarks_data: List[dict] = []
    previous_gray_frame = None
    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with HandLandmarker.create_from_options(hand_options) as handmarker:
        print("Hand Landmarker created successfully.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("End of video reached.")
                else:
                    print("Failed to read frame due to an issue with the video file.")
                break

            rotated_frame = rotate_frame(frame, rotation_code)
            frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_index * (1000.0 / fps))

            hand_result = handmarker.detect_for_video(mp_frame, timestamp_ms)

            current_gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            diff_image_norm = None
            laplacian_img_abs_norm = None
            landmark_quality_metrics: dict[tuple[int, int], dict[str, float]] = {}

            if previous_gray_frame is not None:
                diff_image = cv2.absdiff(current_gray_frame, previous_gray_frame)
                diff_image_norm = cv2.normalize(
                    diff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

                laplacian_img = cv2.Laplacian(current_gray_frame, cv2.CV_64F)
                laplacian_img_abs = np.absolute(laplacian_img)
                laplacian_img_abs_norm = cv2.normalize(
                    laplacian_img_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

                if hand_result.hand_landmarks:
                    h, w = current_gray_frame.shape
                    for instance_idx, landmarks in enumerate(hand_result.hand_landmarks):
                        for landmark_idx, landmark in enumerate(landmarks):
                            cx = int(landmark.x * w)
                            cy = int(landmark.y * h)
                            blur_score = calculate_local_laplacian_variance(
                                laplacian_img_abs_norm, (cx, cy), QUALITY_PATCH_SIZE
                            )
                            motion_score = calculate_local_motion(
                                diff_image_norm, (cx, cy), QUALITY_PATCH_SIZE
                            )
                            conf_score = calculate_confidence_score(
                                blur_score, motion_score, method="inverse_motion_sharpness"
                            )
                            landmark_quality_metrics[(instance_idx, landmark_idx)] = {
                                "laplacian_variance": blur_score,
                                "mean_motion_diff": motion_score,
                                "quality_score": conf_score,
                            }

            frame_landmarks = extract_landmarks_for_frame(frame_index, timestamp_ms, hand_result, None)
            for landmark_dict in frame_landmarks:
                if landmark_dict["source"] == "hand" and previous_gray_frame is not None:
                    key = (landmark_dict["instance_id"], landmark_dict["landmark_id"])
                    metrics = landmark_quality_metrics.get(key)
                    if metrics:
                        landmark_dict["laplacian_variance"] = metrics["laplacian_variance"]
                        landmark_dict["mean_motion_diff"] = metrics["mean_motion_diff"]
                        landmark_dict["quality_score"] = metrics["quality_score"]
                    else:
                        landmark_dict["laplacian_variance"] = np.nan
                        landmark_dict["mean_motion_diff"] = np.nan
                        landmark_dict["quality_score"] = np.nan
                else:
                    landmark_dict["laplacian_variance"] = np.nan
                    landmark_dict["mean_motion_diff"] = np.nan
                    landmark_dict["quality_score"] = np.nan
            all_landmarks_data.extend(frame_landmarks)

            composite_frame = cv2.cvtColor(current_gray_frame, cv2.COLOR_GRAY2BGR)
            composite_frame[:, :, 0] = current_gray_frame
            composite_frame[:, :, 1] = (
                laplacian_img_abs_norm.astype(np.uint8)
                if laplacian_img_abs_norm is not None
                else current_gray_frame
            )
            composite_frame[:, :, 2] = (
                diff_image_norm.astype(np.uint8) if diff_image_norm is not None else current_gray_frame
            )

            quality_video_plain_writer.write(composite_frame.copy())

            if hand_result.hand_landmarks and hand_result.handedness:
                h_vis, w_vis = composite_frame.shape[:2]
                if len(hand_result.handedness) == len(hand_result.hand_landmarks):
                    for handedness_list, landmarks in zip(
                        hand_result.handedness, hand_result.hand_landmarks
                    ):
                        if not handedness_list:
                            continue
                        hand_label = handedness_list[0].category_name
                        if hand_label.lower() != "right":
                            continue
                        for landmark_idx, landmark in enumerate(landmarks):
                            if landmark_idx not in {INDEX_FINGER_TIP_INDEX, THUMB_TIP_INDEX}:
                                continue
                            cx_vis = int(landmark.x * w_vis)
                            cy_vis = int(landmark.y * h_vis)
                            if 0 <= cx_vis < w_vis and 0 <= cy_vis < h_vis:
                                cv2.circle(composite_frame, (cx_vis, cy_vis), 3, (255, 255, 255), -1)

            quality_video_writer.write(composite_frame)
            previous_gray_frame = current_gray_frame.copy()

            frame_index += 1
            if frame_index % 50 == 0:
                print(f"Processed {frame_index} frames.")

    cap.release()
    quality_video_writer.release()
    quality_video_plain_writer.release()

    landmarks_df = pd.DataFrame(all_landmarks_data, columns=QUALITY_LANDMARK_COLUMNS)
    landmarks_df.to_csv(artifacts.landmarks_csv, index=False)
    if all_landmarks_data:
        print(
            f"Landmark data with quality metrics saved successfully to: {artifacts.landmarks_csv}"
        )
    else:
        print(f"No landmark data was collected; wrote empty landmark CSV to: {artifacts.landmarks_csv}")

    pd.DataFrame(columns=WORLD_LANDMARK_COLUMNS).to_csv(artifacts.pose_world_csv, index=False)
    print(f"Wrote empty pose-world CSV to: {artifacts.pose_world_csv}")

    _write_processing_metadata(
        video_path=video_path,
        rotation_code=rotation_code,
        metadata_path=artifacts.metadata_json,
        mode="preprocess-quality-video",
        orientation_source=orientation_source,
        landmarks_csv=artifacts.landmarks_csv,
        pose_world_csv=artifacts.pose_world_csv,
        annotated_video=artifacts.annotated_video,
        plain_video=artifacts.plain_video,
        extra={
            "position_plot": str(artifacts.position_plot.resolve()),
            "quality_plot": str(artifacts.quality_plot.resolve()),
        },
    )

    print(f"Quality video with keypoints saved to: {artifacts.annotated_video}")
    print(f"Quality video without keypoints saved to: {artifacts.plain_video}")
    print("Video processing finished. Generating plots...")

    _save_quality_plots(landmarks_df, artifacts, video_path.stem)


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
    orientation_source = "manual-90cw" if args.rotate else ("auto" if args.auto_orient else "none")
    process_video(
        video_path,
        hand_model_path,
        pose_model_path,
        rotation_code,
        artifacts,
        pose_focus_hint=pose_focus_hint,
        orientation_source=orientation_source,
    )


def run_preprocess_quality_video(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    hand_model_path = resolve_hand_model_path(args.model_dir, paths)
    if not hand_model_path.is_file():
        raise FileNotFoundError(f"Hand model file not found at {hand_model_path}")

    video_path = resolve_video_input(filename=args.filename, video_dir=args.video_dir, paths=paths)
    if args.filename is None:
        print(f"Using default video file provided, {video_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found at {video_path}")

    rotation_code = cv2.ROTATE_90_CLOCKWISE if args.rotate else None
    artifacts = artifact_store.preprocess_quality_video(video_path)
    orientation_source = "manual-90cw" if args.rotate else "none"
    process_quality_video(
        video_path,
        hand_model_path,
        rotation_code,
        artifacts,
        orientation_source=orientation_source,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_preprocess_video_parser()
    args = parser.parse_args(argv)
    run_preprocess_video(args)


def main_quality_video(argv: list[str] | None = None) -> None:
    parser = build_preprocess_quality_video_parser()
    args = parser.parse_args(argv)
    run_preprocess_quality_video(args)


if __name__ == "__main__":
    main()
