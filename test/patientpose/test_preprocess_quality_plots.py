from pathlib import Path

import numpy as np
import pandas as pd

from src.patientpose.artifacts import QualityVideoArtifacts
from src.patientpose.pipeline.preprocess import _save_quality_plots


TEST_OUTPUT_DIR = Path("tmp") / "test_preprocess_quality_plots"


def _quality_artifacts() -> QualityVideoArtifacts:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return QualityVideoArtifacts(
        annotated_video=TEST_OUTPUT_DIR / "quality_vis.avi",
        plain_video=TEST_OUTPUT_DIR / "quality_vis_no_keypoints.avi",
        landmarks_csv=TEST_OUTPUT_DIR / "landmarks.csv",
        pose_world_csv=TEST_OUTPUT_DIR / "pose_world.csv",
        position_plot=TEST_OUTPUT_DIR / "fingertip_position.png",
        quality_plot=TEST_OUTPUT_DIR / "fingertip_quality.png",
        metadata_json=TEST_OUTPUT_DIR / "metadata.json",
    )


def _sample_quality_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": 0,
                "timestamp_ms": 0,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 8,
                "landmark_name": "INDEX_FINGER_TIP",
                "x": 0.40,
                "y": 0.50,
                "z": 0.60,
                "visibility": np.nan,
                "coordinate_space": "image",
                "pose_source": "missing",
                "crop_left": np.nan,
                "crop_top": np.nan,
                "crop_width": np.nan,
                "crop_height": np.nan,
                "crop_frame_width": np.nan,
                "crop_frame_height": np.nan,
                "crop_scale_x": np.nan,
                "crop_scale_y": np.nan,
                "crop_scale": np.nan,
                "laplacian_variance": 1.2,
                "mean_motion_diff": 0.2,
                "quality_score": 0.8,
            },
            {
                "frame": 0,
                "timestamp_ms": 0,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 4,
                "landmark_name": "THUMB_TIP",
                "x": 0.10,
                "y": 0.20,
                "z": 0.30,
                "visibility": np.nan,
                "coordinate_space": "image",
                "pose_source": "missing",
                "crop_left": np.nan,
                "crop_top": np.nan,
                "crop_width": np.nan,
                "crop_height": np.nan,
                "crop_frame_width": np.nan,
                "crop_frame_height": np.nan,
                "crop_scale_x": np.nan,
                "crop_scale_y": np.nan,
                "crop_scale": np.nan,
                "laplacian_variance": 1.4,
                "mean_motion_diff": 0.1,
                "quality_score": 0.9,
            },
            {
                "frame": 1,
                "timestamp_ms": 33,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 8,
                "landmark_name": "INDEX_FINGER_TIP",
                "x": 0.41,
                "y": 0.52,
                "z": 0.61,
                "visibility": np.nan,
                "coordinate_space": "image",
                "pose_source": "missing",
                "crop_left": np.nan,
                "crop_top": np.nan,
                "crop_width": np.nan,
                "crop_height": np.nan,
                "crop_frame_width": np.nan,
                "crop_frame_height": np.nan,
                "crop_scale_x": np.nan,
                "crop_scale_y": np.nan,
                "crop_scale": np.nan,
                "laplacian_variance": 1.1,
                "mean_motion_diff": 0.25,
                "quality_score": 0.82,
            },
            {
                "frame": 1,
                "timestamp_ms": 33,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 4,
                "landmark_name": "THUMB_TIP",
                "x": 0.11,
                "y": 0.22,
                "z": 0.31,
                "visibility": np.nan,
                "coordinate_space": "image",
                "pose_source": "missing",
                "crop_left": np.nan,
                "crop_top": np.nan,
                "crop_width": np.nan,
                "crop_height": np.nan,
                "crop_frame_width": np.nan,
                "crop_frame_height": np.nan,
                "crop_scale_x": np.nan,
                "crop_scale_y": np.nan,
                "crop_scale": np.nan,
                "laplacian_variance": 1.3,
                "mean_motion_diff": 0.15,
                "quality_score": 0.91,
            },
        ]
    )


def test_save_quality_plots_writes_expected_outputs():
    artifacts = _quality_artifacts()
    for path in (artifacts.position_plot, artifacts.quality_plot):
        if path.exists():
            path.unlink()

    _save_quality_plots(_sample_quality_df(), artifacts, "synthetic")

    assert artifacts.position_plot.is_file()
    assert artifacts.position_plot.stat().st_size > 0
    assert artifacts.quality_plot.is_file()
    assert artifacts.quality_plot.stat().st_size > 0
