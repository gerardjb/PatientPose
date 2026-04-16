from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


INDEX_COLUMNS = ("frame", "timestamp_ms")


def _normalize_text(value: object) -> str:
    return str(value).strip().upper()


def _dedupe_columns(values: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def filter_landmark_rows(
    df: pd.DataFrame,
    *,
    source: str | None = None,
    handedness: str | None = None,
    instance_id: int | None = None,
    coordinate_space: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if source is not None and "source" in out.columns:
        out = out[out["source"].astype(str).str.upper() == _normalize_text(source)]
    if handedness is not None and "handedness" in out.columns:
        out = out[out["handedness"].astype(str).str.upper() == _normalize_text(handedness)]
    if instance_id is not None and "instance_id" in out.columns:
        out = out[out["instance_id"] == instance_id]
    if coordinate_space is not None and "coordinate_space" in out.columns:
        out = out[out["coordinate_space"].astype(str).str.upper() == _normalize_text(coordinate_space)]
    return out


def resolve_landmark_id(
    landmark: int | str,
    df: pd.DataFrame,
    *,
    source: str | None = None,
    handedness: str | None = None,
    instance_id: int | None = None,
    coordinate_space: str | None = None,
) -> int:
    if isinstance(landmark, int):
        return landmark

    text = landmark.strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)

    normalized = _normalize_text(text)
    search_dfs = [
        filter_landmark_rows(
            df,
            source=source,
            handedness=handedness,
            instance_id=instance_id,
            coordinate_space=coordinate_space,
        ),
        df,
    ]
    for search_df in search_dfs:
        if "landmark_name" not in search_df.columns or "landmark_id" not in search_df.columns:
            continue
        matches = search_df[
            search_df["landmark_name"].astype(str).str.upper() == normalized
        ]["landmark_id"].dropna()
        if not matches.empty:
            return int(matches.iloc[0])

    raise KeyError(f"Could not resolve landmark '{landmark}' from the provided landmark table.")


def resolve_landmark_name(landmark_id: int, df: pd.DataFrame) -> str:
    if "landmark_name" in df.columns and "landmark_id" in df.columns:
        matches = df[df["landmark_id"] == landmark_id]["landmark_name"].dropna()
        if not matches.empty:
            return str(matches.iloc[0])
    return f"LANDMARK_{landmark_id}"


def build_frame_index(
    df: pd.DataFrame,
    *,
    source: str | None = None,
    handedness: str | None = None,
    instance_id: int | None = None,
    coordinate_space: str | None = None,
) -> pd.DataFrame:
    filtered = filter_landmark_rows(
        df,
        source=source,
        handedness=handedness,
        instance_id=instance_id,
        coordinate_space=coordinate_space,
    )
    if not set(INDEX_COLUMNS).issubset(filtered.columns):
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    out = (
        filtered.loc[:, list(INDEX_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(INDEX_COLUMNS))
        .reset_index(drop=True)
    )
    return out


def build_landmark_series(
    df: pd.DataFrame,
    landmark: int | str,
    *,
    source: str | None = None,
    handedness: str | None = None,
    instance_id: int | None = None,
    coordinate_space: str | None = None,
    components: Sequence[str] = ("x", "y", "z"),
    extra_columns: Sequence[str] = (),
    frame_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    landmark_id = resolve_landmark_id(
        landmark,
        df,
        source=source,
        handedness=handedness,
        instance_id=instance_id,
        coordinate_space=coordinate_space,
    )
    landmark_name = resolve_landmark_name(landmark_id, df)

    filtered = filter_landmark_rows(
        df,
        source=source,
        handedness=handedness,
        instance_id=instance_id,
        coordinate_space=coordinate_space,
    )
    if frame_index is None:
        base_index = build_frame_index(
            filtered,
            source=None,
            handedness=None,
            instance_id=None,
            coordinate_space=None,
        )
    else:
        base_index = frame_index.loc[:, list(INDEX_COLUMNS)].copy()

    value_columns = _dedupe_columns([*components, *extra_columns])
    selected_rows = filtered[filtered["landmark_id"] == landmark_id].copy()
    available_value_columns = [column for column in value_columns if column in selected_rows.columns]

    if not selected_rows.empty:
        selected_rows = (
            selected_rows.loc[:, [*INDEX_COLUMNS, *available_value_columns]]
            .sort_values(list(INDEX_COLUMNS))
            .groupby(list(INDEX_COLUMNS), as_index=False)
            .first()
        )
    else:
        selected_rows = pd.DataFrame(columns=[*INDEX_COLUMNS, *available_value_columns])

    out = base_index.merge(
        selected_rows,
        on=list(INDEX_COLUMNS),
        how="left",
        sort=True,
    )
    for column in value_columns:
        if column not in out.columns:
            out[column] = np.nan

    out.insert(2, "landmark_id", landmark_id)
    out.insert(3, "landmark_name", landmark_name)
    return out.loc[:, [*INDEX_COLUMNS, "landmark_id", "landmark_name", *value_columns]]


def build_multi_landmark_series(
    df: pd.DataFrame,
    landmarks: Sequence[int | str],
    *,
    source: str | None = None,
    handedness: str | None = None,
    instance_id: int | None = None,
    coordinate_space: str | None = None,
    components: Sequence[str] = ("x", "y", "z"),
    extra_columns: Sequence[str] = (),
) -> dict[str, pd.DataFrame]:
    filtered = filter_landmark_rows(
        df,
        source=source,
        handedness=handedness,
        instance_id=instance_id,
        coordinate_space=coordinate_space,
    )
    shared_index = build_frame_index(filtered)
    out: dict[str, pd.DataFrame] = {}
    for landmark in landmarks:
        landmark_id = resolve_landmark_id(
            landmark,
            df,
            source=source,
            handedness=handedness,
            instance_id=instance_id,
            coordinate_space=coordinate_space,
        )
        landmark_name = resolve_landmark_name(landmark_id, df)
        out[landmark_name] = build_landmark_series(
            df,
            landmark_id,
            source=source,
            handedness=handedness,
            instance_id=instance_id,
            coordinate_space=coordinate_space,
            components=components,
            extra_columns=extra_columns,
            frame_index=shared_index,
        )
    return out


__all__ = [
    "INDEX_COLUMNS",
    "build_frame_index",
    "build_landmark_series",
    "build_multi_landmark_series",
    "filter_landmark_rows",
    "resolve_landmark_id",
    "resolve_landmark_name",
]
