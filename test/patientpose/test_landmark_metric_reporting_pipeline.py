from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.patientpose.pipeline.reporting import run_landmark_metric_batch, run_landmark_metric_export


TEST_OUTPUT_DIR = Path("tmp") / "test_landmark_metric_reporting_pipeline"


def _write_hand_csv(path: Path, *, scale: float) -> Path:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
                "x": 3.0 * scale,
                "y": 4.0 * scale,
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
                "x": 6.0 * scale,
                "y": 8.0 * scale,
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
    df.to_csv(path, index=False)
    return path


def test_run_landmark_metric_export_writes_trace_and_summary_csvs():
    csv_path = _write_hand_csv(TEST_OUTPUT_DIR / "export_a.csv", scale=1.0)
    trace_output = TEST_OUTPUT_DIR / "export_trace.csv"
    summary_output = TEST_OUTPUT_DIR / "export_summary.csv"
    for output in (trace_output, summary_output):
        if output.exists():
            output.unlink()

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
        landmarks=None,
        components=["x", "y"],
        smooth_window=1,
        quality_threshold=None,
        metric="thumb-index-distance",
        delta_component=None,
        trace_output=trace_output,
        summary_output=summary_output,
    )

    run_landmark_metric_export(args)

    assert trace_output.is_file()
    assert summary_output.is_file()
    trace_df = pd.read_csv(trace_output)
    summary_df = pd.read_csv(summary_output)
    assert len(trace_df) == 2
    assert len(summary_df) == 1
    assert summary_df.loc[0, "mean"] == 7.5


def test_run_landmark_metric_batch_writes_trace_and_summary_csvs_from_glob():
    _write_hand_csv(TEST_OUTPUT_DIR / "batch_a.csv", scale=1.0)
    _write_hand_csv(TEST_OUTPUT_DIR / "batch_b.csv", scale=2.0)
    trace_output = TEST_OUTPUT_DIR / "batch_trace.csv"
    summary_output = TEST_OUTPUT_DIR / "batch_summary.csv"
    for output in (trace_output, summary_output):
        if output.exists():
            output.unlink()

    args = SimpleNamespace(
        project_root=Path("."),
        camera_csvs=None,
        glob="tmp/test_landmark_metric_reporting_pipeline/batch_*.csv",
        source="hand",
        space="image",
        handedness="Right",
        instance_id=0,
        landmarks=None,
        components=["x", "y"],
        smooth_window=1,
        quality_threshold=None,
        metric="thumb-index-distance",
        delta_component=None,
        trace_output=trace_output,
        summary_output=summary_output,
    )

    run_landmark_metric_batch(args)

    assert trace_output.is_file()
    assert summary_output.is_file()
    trace_df = pd.read_csv(trace_output)
    summary_df = pd.read_csv(summary_output)
    assert len(trace_df) == 4
    assert len(summary_df) == 2
    assert set(summary_df["stem"]) == {"batch_a", "batch_b"}
