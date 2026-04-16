import numpy as np
import pandas as pd

from src.patientpose.landmarks.time_series import (
    build_landmark_series,
    build_multi_landmark_series,
    resolve_landmark_id,
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
                "x": 0.10,
                "y": 0.20,
                "z": 0.30,
                "visibility": np.nan,
                "quality_score": 0.90,
            },
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
                "quality_score": 0.80,
            },
            {
                "frame": 0,
                "timestamp_ms": 0,
                "source": "hand",
                "instance_id": 1,
                "handedness": "Left",
                "landmark_id": 8,
                "landmark_name": "INDEX_FINGER_TIP",
                "x": 9.40,
                "y": 9.50,
                "z": 9.60,
                "visibility": np.nan,
                "quality_score": 0.10,
            },
            {
                "frame": 1,
                "timestamp_ms": 33,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 8,
                "landmark_name": "INDEX_FINGER_TIP",
                "x": 0.45,
                "y": 0.55,
                "z": 0.65,
                "visibility": np.nan,
                "quality_score": 0.85,
            },
            {
                "frame": 2,
                "timestamp_ms": 66,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 4,
                "landmark_name": "THUMB_TIP",
                "x": 0.12,
                "y": 0.22,
                "z": 0.32,
                "visibility": np.nan,
                "quality_score": 0.95,
            },
            {
                "frame": 2,
                "timestamp_ms": 66,
                "source": "hand",
                "instance_id": 0,
                "handedness": "Right",
                "landmark_id": 8,
                "landmark_name": "INDEX_FINGER_TIP",
                "x": 0.47,
                "y": 0.57,
                "z": 0.67,
                "visibility": np.nan,
                "quality_score": 0.88,
            },
            {
                "frame": 2,
                "timestamp_ms": 66,
                "source": "pose",
                "instance_id": 0,
                "handedness": None,
                "landmark_id": 27,
                "landmark_name": "LEFT_ANKLE",
                "x": 0.70,
                "y": 0.80,
                "z": 0.90,
                "visibility": 0.99,
                "quality_score": np.nan,
            },
        ]
    )


def test_resolve_landmark_id_matches_name_case_insensitively():
    df = _sample_landmark_df()
    assert resolve_landmark_id("index_finger_tip", df) == 8


def test_build_landmark_series_filters_by_source_and_handedness():
    df = _sample_landmark_df()
    series = build_landmark_series(
        df,
        "INDEX_FINGER_TIP",
        source="hand",
        handedness="Right",
        components=("x", "y"),
        extra_columns=("quality_score",),
    )

    assert list(series["frame"]) == [0, 1, 2]
    assert list(series["timestamp_ms"]) == [0, 33, 66]
    assert list(series["landmark_id"].unique()) == [8]
    assert list(series["landmark_name"].unique()) == ["INDEX_FINGER_TIP"]
    assert series.loc[0, "x"] == 0.40
    assert series.loc[0, "y"] == 0.50
    assert series.loc[0, "quality_score"] == 0.80
    assert not np.isclose(series.loc[0, "x"], 9.40)


def test_build_multi_landmark_series_uses_shared_frame_index_and_fills_missing_rows():
    df = _sample_landmark_df()
    series_map = build_multi_landmark_series(
        df,
        ["THUMB_TIP", "INDEX_FINGER_TIP"],
        source="hand",
        handedness="Right",
        components=("x",),
    )

    thumb = series_map["THUMB_TIP"]
    index_tip = series_map["INDEX_FINGER_TIP"]

    assert list(thumb["frame"]) == [0, 1, 2]
    assert list(index_tip["frame"]) == [0, 1, 2]
    assert thumb.loc[0, "x"] == 0.10
    assert np.isnan(thumb.loc[1, "x"])
    assert thumb.loc[2, "x"] == 0.12
    assert index_tip.loc[1, "x"] == 0.45


def test_build_landmark_series_adds_requested_missing_columns_as_nan():
    df = _sample_landmark_df()
    series = build_landmark_series(
        df,
        "LEFT_ANKLE",
        source="pose",
        components=("x", "visibility"),
        extra_columns=("quality_score",),
    )

    assert list(series["frame"]) == [2]
    assert series.loc[0, "x"] == 0.70
    assert series.loc[0, "visibility"] == 0.99
    assert np.isnan(series.loc[0, "quality_score"])
