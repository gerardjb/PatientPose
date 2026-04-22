from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.patientpose.pipeline.reporting import run_body_angle_export


TEST_OUTPUT_DIR = Path("tmp") / "test_body_angle_reporting_pipeline"


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


def test_run_body_angle_export_writes_trace_and_summary_csvs():
    csv_path = _write_pose_csv(TEST_OUTPUT_DIR / "export_body_angle.csv")
    trace_output = TEST_OUTPUT_DIR / "body_angle_trace.csv"
    summary_output = TEST_OUTPUT_DIR / "body_angle_summary.csv"
    for output in (trace_output, summary_output):
        if output.exists():
            output.unlink()

    args = SimpleNamespace(
        project_root=Path("."),
        camera_csv=csv_path,
        tag=None,
        camera_side="ND",
        camera_role=None,
        space="image",
        world_csv=None,
        visibility_threshold=0.4,
        smooth_window=1,
        trace_output=trace_output,
        summary_output=summary_output,
    )

    run_body_angle_export(args)

    assert trace_output.is_file()
    assert summary_output.is_file()
    trace_df = pd.read_csv(trace_output)
    summary_df = pd.read_csv(summary_output)
    assert len(trace_df) == 2
    assert len(summary_df) == 1
    assert trace_df["body_angle_deg"].tolist() == [-90.0, -45.0]
    assert summary_df.loc[0, "mean"] == -67.5
