import numpy as np
import pandas as pd

from src.patientpose.landmarks import build_landmark_series
from src.patientpose.metrics import (
    centroid,
    pairwise_delta,
    pairwise_distance,
    pinch_velocity,
    point_to_centroid_distance,
    segment_angle,
    thumb_index_distance,
)


def _series(frame_values: list[tuple[int, int, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "timestamp_ms": timestamp_ms,
                "x": x,
                "y": y,
                "z": z,
            }
            for frame, timestamp_ms, x, y, z in frame_values
        ]
    )


def _sample_landmark_df() -> pd.DataFrame:
    return pd.DataFrame(
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
                "quality_score": 0.8,
            },
            {
                "frame": 1,
                "timestamp_ms": 100,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 4,
                "landmark_name": "THUMB_TIP",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "visibility": np.nan,
                "quality_score": 0.95,
            },
            {
                "frame": 1,
                "timestamp_ms": 100,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 8,
                "landmark_name": "INDEX_FINGER_TIP",
                "x": 6.0,
                "y": 8.0,
                "z": 0.0,
                "visibility": np.nan,
                "quality_score": 0.85,
            },
            {
                "frame": 2,
                "timestamp_ms": 200,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 4,
                "landmark_name": "THUMB_TIP",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "visibility": np.nan,
                "quality_score": 0.96,
            },
        ]
    )


def test_pairwise_distance_and_delta_match_expected_geometry():
    series_a = _series([(0, 0, 0.0, 0.0, 0.0), (1, 10, 1.0, 1.0, 1.0)])
    series_b = _series([(0, 0, 3.0, 4.0, 0.0), (1, 10, 4.0, 5.0, 1.0)])

    distance_df = pairwise_distance(series_a, series_b, components=("x", "y"))
    delta_df = pairwise_delta(series_a, series_b, component="x")

    assert list(distance_df["value"]) == [5.0, 5.0]
    assert list(delta_df["value"]) == [-3.0, -3.0]


def test_centroid_and_point_to_centroid_distance_are_computed_per_frame():
    left = _series([(0, 0, 0.0, 0.0, 0.0), (1, 10, 2.0, 0.0, 0.0)])
    right = _series([(0, 0, 2.0, 0.0, 0.0), (1, 10, 4.0, 0.0, 0.0)])
    center = centroid([left, right], components=("x", "y"))

    assert list(center["x"]) == [1.0, 3.0]
    assert list(center["y"]) == [0.0, 0.0]

    point = _series([(0, 0, 1.0, 2.0, 0.0), (1, 10, 3.0, 2.0, 0.0)])
    distance_df = point_to_centroid_distance(point, center, components=("x", "y"))
    assert np.allclose(distance_df["value"].to_numpy(dtype=float), np.array([2.0, 2.0]))


def test_segment_angle_returns_degrees():
    series_a = _series([(0, 0, 0.0, 0.0, 0.0)])
    series_b = _series([(0, 0, 1.0, 0.0, 0.0)])
    series_c = _series([(0, 0, 0.0, 0.0, 0.0)])
    series_d = _series([(0, 0, 0.0, 1.0, 0.0)])

    angle_df = segment_angle(series_a, series_b, series_c, series_d, components=("x", "y"))
    assert angle_df.loc[0, "value"] == 90.0


def test_thumb_index_distance_and_pinch_velocity_work_with_time_series_output():
    df = _sample_landmark_df()
    thumb = build_landmark_series(
        df,
        "THUMB_TIP",
        source="hand",
        handedness="Right",
        components=("x", "y"),
    )
    index_tip = build_landmark_series(
        df,
        "INDEX_FINGER_TIP",
        source="hand",
        handedness="Right",
        components=("x", "y"),
    )

    distance_df = thumb_index_distance(thumb, index_tip, components=("x", "y"))
    velocity_df = pinch_velocity(thumb, index_tip, components=("x", "y"))

    assert list(distance_df["value"][:2]) == [5.0, 10.0]
    assert np.isnan(distance_df.loc[2, "value"])
    assert np.isnan(velocity_df.loc[0, "value"])
    assert velocity_df.loc[1, "value"] == 50.0
    assert np.isnan(velocity_df.loc[2, "value"])
