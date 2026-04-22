from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.patientpose.reporting.body_angle import (
    BODY_ANGLE_SUMMARY_COLUMNS,
    BODY_ANGLE_TRACE_COLUMNS,
    body_angle_trace_table,
    compute_body_angle_trace,
    summarize_body_angle_trace,
)


TEST_OUTPUT_DIR = Path("tmp") / "test_reporting_body_angle"


def _write_pose_csv(path: Path) -> Path:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        (0, 0, "LEFT_HIP", 0.0, 1.0),
        (0, 0, "RIGHT_HIP", 1.0, 1.0),
        (0, 0, "LEFT_SHOULDER", 0.0, 0.0),
        (0, 0, "RIGHT_SHOULDER", 1.0, 0.0),
        (1, 33, "LEFT_HIP", 0.0, 1.0),
        (1, 33, "RIGHT_HIP", 1.0, 2.0),
        (1, 33, "LEFT_SHOULDER", 1.0, 0.0),
        (1, 33, "RIGHT_SHOULDER", 2.0, 1.0),
    ]
    df = pd.DataFrame(
        [
            {
                "frame": frame,
                "timestamp_ms": timestamp_ms,
                "source": "pose",
                "instance_id": 0,
                "handedness": "N/A",
                "landmark_id": landmark_idx,
                "landmark_name": landmark_name,
                "x": x,
                "y": y,
                "z": 0.0,
                "visibility": 0.99,
                "coordinate_space": "image",
                "pose_source": "full_frame",
                "crop_left": np.nan,
                "crop_top": np.nan,
                "crop_width": np.nan,
                "crop_height": np.nan,
                "crop_frame_width": np.nan,
                "crop_frame_height": np.nan,
                "crop_scale_x": np.nan,
                "crop_scale_y": np.nan,
                "crop_scale": np.nan,
            }
            for landmark_idx, (frame, timestamp_ms, landmark_name, x, y) in enumerate(rows)
        ]
    )
    df.to_csv(path, index=False)
    return path


def test_compute_body_angle_trace_returns_expected_angles():
    csv_path = _write_pose_csv(TEST_OUTPUT_DIR / "body_angle.csv")

    result = compute_body_angle_trace(
        csv_path,
        space="image",
        visibility_threshold=0.4,
        smooth_window=1,
    )

    assert result.stem == "body_angle"
    assert result.space == "image"
    assert result.trace_df["body_angle_deg"].tolist() == pytest.approx([-90.0, -45.0])

    trace_df = body_angle_trace_table(result)
    assert tuple(trace_df.columns) == BODY_ANGLE_TRACE_COLUMNS
    assert trace_df["smooth_window"].tolist() == [1, 1]


def test_summarize_body_angle_trace_returns_expected_stats():
    csv_path = _write_pose_csv(TEST_OUTPUT_DIR / "body_angle_summary.csv")
    result = compute_body_angle_trace(
        csv_path,
        space="image",
        visibility_threshold=0.4,
        smooth_window=1,
    )

    summary_df = summarize_body_angle_trace(result)

    assert tuple(summary_df.columns) == BODY_ANGLE_SUMMARY_COLUMNS
    row = summary_df.iloc[0]
    assert row["n_total"] == 2
    assert row["n_valid"] == 2
    assert row["valid_fraction"] == pytest.approx(1.0)
    assert row["duration_s"] == pytest.approx(0.033)
    assert row["mean"] == pytest.approx(-67.5)
    assert row["median"] == pytest.approx(-67.5)
    assert row["min"] == pytest.approx(-90.0)
    assert row["max"] == pytest.approx(-45.0)
