from __future__ import annotations

"""Data-quality metrics derived from camera landmark CSVs."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def count_visible_landmarks(
    df_or_path: pd.DataFrame | str | Path,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Count how many pose landmarks are above a visibility threshold per timestamp.
    """
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, (str, Path)) else df_or_path
    if "visibility" not in df.columns:
        pose_df = df[df["source"] == "pose"]
        timestamps = pose_df["timestamp_ms"].to_numpy(dtype=float)
        counts = np.zeros_like(timestamps, dtype=float)
        totals = np.ones_like(timestamps, dtype=float)
        return timestamps, counts, totals

    pose_df = df[df["source"] == "pose"].copy()
    pose_df["vis_ok"] = pose_df["visibility"] >= threshold
    grouped = (
        pose_df.groupby("timestamp_ms")
        .agg(vis_ok_sum=("vis_ok", "sum"), total=("vis_ok", "count"))
        .reset_index()
    )
    timestamps = grouped["timestamp_ms"].to_numpy(dtype=float)
    counts = grouped["vis_ok_sum"].to_numpy(dtype=float)
    totals = grouped["total"].to_numpy(dtype=float)
    return timestamps, counts, totals


def visibility_percent(
    df_or_path: pd.DataFrame | str | Path,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Percentage of landmarks above the given visibility threshold per timestamp.
    """
    t_ms, counts, totals = count_visible_landmarks(df_or_path, threshold)
    percent = np.where(totals > 0, (counts / totals) * 100.0, 0.0)
    percent = np.nan_to_num(percent, nan=0.0, posinf=0.0, neginf=0.0)
    return t_ms, percent


__all__ = ["count_visible_landmarks", "visibility_percent"]
