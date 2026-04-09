from __future__ import annotations

import pandas as pd

from .schema import IMAGE_SPACE, WORLD_SPACE


def pose_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "source" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["source"] == "pose"].copy()


def pose_image_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = pose_rows(df)
    if "coordinate_space" in out.columns:
        out = out[out["coordinate_space"].fillna(IMAGE_SPACE) == IMAGE_SPACE]
    return out


def pose_world_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = pose_rows(df)
    if "coordinate_space" in out.columns:
        out = out[out["coordinate_space"].fillna(WORLD_SPACE) == WORLD_SPACE]
    return out


__all__ = [
    "pose_image_rows",
    "pose_rows",
    "pose_world_rows",
]
