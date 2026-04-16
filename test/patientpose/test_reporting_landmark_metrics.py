from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.patientpose.reporting.landmark_metrics import (
    SUMMARY_TABLE_COLUMNS,
    TRACE_TABLE_COLUMNS,
    compute_landmark_metric_trace,
    export_landmark_metric_batch,
    metric_trace_table,
    summarize_landmark_metric_trace,
)


TEST_OUTPUT_DIR = Path("tmp") / "test_reporting_landmark_metrics"


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


def test_compute_landmark_metric_trace_returns_expected_trace():
    csv_path = _write_hand_csv(TEST_OUTPUT_DIR / "trace_a.csv", scale=1.0)

    result = compute_landmark_metric_trace(
        csv_path,
        metric="thumb-index-distance",
        source="hand",
        handedness="Right",
    )

    assert result.stem == "trace_a"
    assert result.metric_label == "thumb-index distance"
    assert result.landmarks == ("THUMB_TIP", "INDEX_FINGER_TIP")
    assert result.components == ("x", "y")
    assert result.metric_df["value"].tolist() == pytest.approx([5.0, 10.0])

    trace_df = metric_trace_table(result)
    assert tuple(trace_df.columns) == TRACE_TABLE_COLUMNS
    assert trace_df["requested_metric"].tolist() == ["thumb-index-distance", "thumb-index-distance"]


def test_summarize_landmark_metric_trace_returns_expected_stats():
    csv_path = _write_hand_csv(TEST_OUTPUT_DIR / "trace_b.csv", scale=1.0)
    result = compute_landmark_metric_trace(
        csv_path,
        metric="thumb-index-distance",
        source="hand",
        handedness="Right",
    )

    summary_df = summarize_landmark_metric_trace(result)

    assert tuple(summary_df.columns) == SUMMARY_TABLE_COLUMNS
    row = summary_df.iloc[0]
    assert row["n_total"] == 2
    assert row["n_valid"] == 2
    assert row["valid_fraction"] == pytest.approx(1.0)
    assert row["duration_s"] == pytest.approx(0.033)
    assert row["mean"] == pytest.approx(7.5)
    assert row["median"] == pytest.approx(7.5)
    assert row["min"] == pytest.approx(5.0)
    assert row["max"] == pytest.approx(10.0)


def test_export_landmark_metric_batch_concatenates_traces_and_summaries():
    csv_a = _write_hand_csv(TEST_OUTPUT_DIR / "batch_a.csv", scale=1.0)
    csv_b = _write_hand_csv(TEST_OUTPUT_DIR / "batch_b.csv", scale=2.0)

    trace_df, summary_df = export_landmark_metric_batch(
        [csv_a, csv_b],
        metric="thumb-index-distance",
        source="hand",
        handedness="Right",
    )

    assert tuple(trace_df.columns) == TRACE_TABLE_COLUMNS
    assert tuple(summary_df.columns) == SUMMARY_TABLE_COLUMNS
    assert len(trace_df) == 4
    assert len(summary_df) == 2
    assert set(summary_df["stem"]) == {"batch_a", "batch_b"}
    means = dict(zip(summary_df["stem"], summary_df["mean"]))
    assert means["batch_a"] == pytest.approx(7.5)
    assert means["batch_b"] == pytest.approx(15.0)
