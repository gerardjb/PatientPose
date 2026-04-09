from __future__ import annotations

IMAGE_LANDMARK_COLUMNS = [
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
    "coordinate_space",
    "pose_source",
    "crop_left",
    "crop_top",
    "crop_width",
    "crop_height",
    "crop_frame_width",
    "crop_frame_height",
    "crop_scale_x",
    "crop_scale_y",
    "crop_scale",
]

WORLD_LANDMARK_COLUMNS = [
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
    "coordinate_space",
    "pose_source",
    "crop_left",
    "crop_top",
    "crop_width",
    "crop_height",
    "crop_frame_width",
    "crop_frame_height",
    "crop_scale_x",
    "crop_scale_y",
    "crop_scale",
]

FRAME_SUMMARY_COLUMNS = [
    "frame",
    "timestamp_ms",
    "pose_detected",
    "num_pose_landmarks",
    "hand_detected",
    "num_hand_landmarks",
    "pose_quality_score",
    "pose_source",
    "crop_left",
    "crop_top",
    "crop_width",
    "crop_height",
    "crop_frame_width",
    "crop_frame_height",
    "crop_scale_x",
    "crop_scale_y",
    "crop_scale",
]

QUALITY_LANDMARK_COLUMNS = [
    *IMAGE_LANDMARK_COLUMNS,
    "laplacian_variance",
    "mean_motion_diff",
    "quality_score",
]

IMAGE_SPACE = "image"
WORLD_SPACE = "world"


__all__ = [
    "FRAME_SUMMARY_COLUMNS",
    "IMAGE_LANDMARK_COLUMNS",
    "IMAGE_SPACE",
    "QUALITY_LANDMARK_COLUMNS",
    "WORLD_LANDMARK_COLUMNS",
    "WORLD_SPACE",
]
