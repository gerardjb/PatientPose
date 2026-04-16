from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.patientpose.pipeline.diagnostics import run_landmark_metric_plot, run_landmark_traces


TEST_OUTPUT_DIR = Path("tmp") / "test_landmark_diagnostics_pipeline"


def _sample_landmark_csv() -> Path:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TEST_OUTPUT_DIR / "landmarks_hand_sample.csv"
    df = pd.DataFrame(
        [
            {
                "frame": 0,
                "timestamp_ms": 0,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 4,
                "landmark_name": "THUMB_TIP",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
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
                "quality_score": 0.9,
            },
            {
                "frame": 0,
                "timestamp_ms": 0,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 8,
                "landmark_name": "INDEX_FINGER_TIP",
                "x": 3.0,
                "y": 4.0,
                "z": 0.0,
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
                "quality_score": 0.95,
            },
            {
                "frame": 1,
                "timestamp_ms": 33,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 4,
                "landmark_name": "THUMB_TIP",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
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
                "quality_score": 0.88,
            },
            {
                "frame": 1,
                "timestamp_ms": 33,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 8,
                "landmark_name": "INDEX_FINGER_TIP",
                "x": 6.0,
                "y": 8.0,
                "z": 0.0,
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
                "quality_score": 0.92,
            },
        ]
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def test_run_landmark_traces_writes_output():
    csv_path = _sample_landmark_csv()
    output_path = TEST_OUTPUT_DIR / "landmark_traces.png"
    if output_path.exists():
        output_path.unlink()

    args = SimpleNamespace(
        project_root=Path("."),
        camera_csv=csv_path,
        tag=None,
        camera_side="ND",
        camera_role=None,
        source="hand",
        space="image",
        world_csv=None,
        handedness="Right",
        instance_id=0,
        landmarks=["THUMB_TIP", "INDEX_FINGER_TIP"],
        components=["x", "y"],
        smooth_window=1,
        quality_threshold=None,
        output=output_path,
    )

    run_landmark_traces(args)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_run_landmark_metric_plot_writes_output():
    csv_path = _sample_landmark_csv()
    output_path = TEST_OUTPUT_DIR / "landmark_metric.png"
    if output_path.exists():
        output_path.unlink()

    args = SimpleNamespace(
        project_root=Path("."),
        camera_csv=csv_path,
        tag=None,
        camera_side="ND",
        camera_role=None,
        source="hand",
        space="image",
        world_csv=None,
        handedness="Right",
        instance_id=0,
        landmarks=["THUMB_TIP", "INDEX_FINGER_TIP"],
        components=["x", "y"],
        smooth_window=1,
        quality_threshold=None,
        metric="distance",
        delta_component=None,
        output=output_path,
    )

    run_landmark_metric_plot(args)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
