from __future__ import annotations

import numpy as np
import pandas as pd

from .landmark_metrics import pairwise_distance


def thumb_index_distance(
    thumb_series: pd.DataFrame,
    index_series: pd.DataFrame,
    *,
    components: tuple[str, ...] = ("x", "y"),
) -> pd.DataFrame:
    return pairwise_distance(
        thumb_series,
        index_series,
        components=components,
        metric_name=f"thumb_index_distance_{''.join(components)}",
    )


def fingertip_span(
    fingertip_a: pd.DataFrame,
    fingertip_b: pd.DataFrame,
    *,
    components: tuple[str, ...] = ("x", "y"),
    label: str = "fingertip_span",
) -> pd.DataFrame:
    return pairwise_distance(
        fingertip_a,
        fingertip_b,
        components=components,
        metric_name=f"{label}_{''.join(components)}",
    )


def pinch_velocity(
    thumb_series: pd.DataFrame,
    index_series: pd.DataFrame,
    *,
    components: tuple[str, ...] = ("x", "y"),
    absolute: bool = False,
) -> pd.DataFrame:
    distance_df = thumb_index_distance(thumb_series, index_series, components=components)
    timestamps_ms = distance_df["timestamp_ms"].to_numpy(dtype=float)
    distance = distance_df["value"].to_numpy(dtype=float)

    velocity = np.full(len(distance_df), np.nan, dtype=float)
    if len(distance_df) >= 2:
        dt_seconds = np.diff(timestamps_ms) / 1000.0
        dd = np.diff(distance)
        valid = np.isfinite(dd) & np.isfinite(dt_seconds) & (dt_seconds != 0.0)
        velocity_steps = np.full(len(dd), np.nan, dtype=float)
        velocity_steps[valid] = dd[valid] / dt_seconds[valid]
        velocity[1:] = velocity_steps
    if absolute:
        velocity = np.abs(velocity)

    out = distance_df.loc[:, ["frame", "timestamp_ms"]].copy()
    out["metric"] = "pinch_velocity_abs" if absolute else "pinch_velocity"
    out["value"] = velocity
    return out


__all__ = [
    "fingertip_span",
    "pinch_velocity",
    "thumb_index_distance",
]
