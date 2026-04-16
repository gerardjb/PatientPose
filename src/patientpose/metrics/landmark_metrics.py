from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from patientpose.landmarks import INDEX_COLUMNS


def _merge_series(
    series_a: pd.DataFrame,
    series_b: pd.DataFrame,
    *,
    components: Sequence[str],
) -> pd.DataFrame:
    columns_a = [*INDEX_COLUMNS, *components]
    columns_b = [*INDEX_COLUMNS, *components]
    renamed_a = series_a.loc[:, columns_a].rename(columns={component: f"{component}_a" for component in components})
    renamed_b = series_b.loc[:, columns_b].rename(columns={component: f"{component}_b" for component in components})
    return renamed_a.merge(renamed_b, on=list(INDEX_COLUMNS), how="outer", sort=True)


def _result_frame(index_df: pd.DataFrame, value: np.ndarray, *, metric: str) -> pd.DataFrame:
    out = index_df.loc[:, list(INDEX_COLUMNS)].copy()
    out["metric"] = metric
    out["value"] = value
    return out


def pairwise_delta(
    series_a: pd.DataFrame,
    series_b: pd.DataFrame,
    *,
    component: str = "x",
    metric_name: str | None = None,
) -> pd.DataFrame:
    merged = _merge_series(series_a, series_b, components=(component,))
    value = merged[f"{component}_a"].to_numpy(dtype=float) - merged[f"{component}_b"].to_numpy(dtype=float)
    return _result_frame(merged, value, metric=metric_name or f"delta_{component}")


def pairwise_distance(
    series_a: pd.DataFrame,
    series_b: pd.DataFrame,
    *,
    components: Sequence[str] = ("x", "y"),
    metric_name: str | None = None,
) -> pd.DataFrame:
    merged = _merge_series(series_a, series_b, components=components)
    squared_sum = np.zeros(len(merged), dtype=float)
    valid_mask = np.ones(len(merged), dtype=bool)
    for component in components:
        a = merged[f"{component}_a"].to_numpy(dtype=float)
        b = merged[f"{component}_b"].to_numpy(dtype=float)
        valid_component = np.isfinite(a) & np.isfinite(b)
        valid_mask &= valid_component
        diff = a - b
        squared_sum += np.where(valid_component, diff * diff, 0.0)
    value = np.where(valid_mask, np.sqrt(squared_sum), np.nan)
    label = metric_name or f"distance_{''.join(components)}"
    return _result_frame(merged, value, metric=label)


def centroid(
    series_list: Sequence[pd.DataFrame],
    *,
    components: Sequence[str] = ("x", "y"),
) -> pd.DataFrame:
    if not series_list:
        raise ValueError("series_list must contain at least one landmark series.")

    base = series_list[0].loc[:, list(INDEX_COLUMNS)].copy()
    for series in series_list[1:]:
        base = base.merge(series.loc[:, list(INDEX_COLUMNS)], on=list(INDEX_COLUMNS), how="outer", sort=True)
    base = base.drop_duplicates().sort_values(list(INDEX_COLUMNS)).reset_index(drop=True)

    out = base.copy()
    for component in components:
        stacked = []
        for series in series_list:
            component_df = base.merge(
                series.loc[:, [*INDEX_COLUMNS, component]],
                on=list(INDEX_COLUMNS),
                how="left",
                sort=True,
            )
            stacked.append(component_df[component].to_numpy(dtype=float))
        values = np.vstack(stacked)
        out[component] = np.nanmean(values, axis=0)
    return out


def point_to_centroid_distance(
    point_series: pd.DataFrame,
    centroid_series: pd.DataFrame,
    *,
    components: Sequence[str] = ("x", "y"),
    metric_name: str | None = None,
) -> pd.DataFrame:
    return pairwise_distance(
        point_series,
        centroid_series,
        components=components,
        metric_name=metric_name or f"point_to_centroid_distance_{''.join(components)}",
    )


def segment_angle(
    series_a: pd.DataFrame,
    series_b: pd.DataFrame,
    series_c: pd.DataFrame,
    series_d: pd.DataFrame,
    *,
    components: Sequence[str] = ("x", "y"),
    degrees: bool = True,
    metric_name: str | None = None,
) -> pd.DataFrame:
    if not components:
        raise ValueError("components must contain at least one axis.")

    base = series_a.loc[:, list(INDEX_COLUMNS)].copy()
    for series in (series_b, series_c, series_d):
        base = base.merge(series.loc[:, list(INDEX_COLUMNS)], on=list(INDEX_COLUMNS), how="outer", sort=True)
    base = base.drop_duplicates().sort_values(list(INDEX_COLUMNS)).reset_index(drop=True)

    vec_ab = []
    vec_cd = []
    valid_mask = np.ones(len(base), dtype=bool)
    for component in components:
        a_df = base.merge(series_a.loc[:, [*INDEX_COLUMNS, component]], on=list(INDEX_COLUMNS), how="left", sort=True)
        b_df = base.merge(series_b.loc[:, [*INDEX_COLUMNS, component]], on=list(INDEX_COLUMNS), how="left", sort=True)
        c_df = base.merge(series_c.loc[:, [*INDEX_COLUMNS, component]], on=list(INDEX_COLUMNS), how="left", sort=True)
        d_df = base.merge(series_d.loc[:, [*INDEX_COLUMNS, component]], on=list(INDEX_COLUMNS), how="left", sort=True)
        a = a_df[component].to_numpy(dtype=float)
        b = b_df[component].to_numpy(dtype=float)
        c = c_df[component].to_numpy(dtype=float)
        d = d_df[component].to_numpy(dtype=float)
        component_valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
        valid_mask &= component_valid
        vec_ab.append(np.where(component_valid, b - a, 0.0))
        vec_cd.append(np.where(component_valid, d - c, 0.0))

    vec_ab_arr = np.vstack(vec_ab).T
    vec_cd_arr = np.vstack(vec_cd).T

    dot = np.sum(vec_ab_arr * vec_cd_arr, axis=1)
    norm_ab = np.linalg.norm(vec_ab_arr, axis=1)
    norm_cd = np.linalg.norm(vec_cd_arr, axis=1)
    nonzero = (norm_ab > 0.0) & (norm_cd > 0.0)
    valid_mask &= nonzero

    cos_theta = np.ones(len(base), dtype=float)
    safe_denominator = norm_ab * norm_cd
    cos_theta[valid_mask] = np.clip(dot[valid_mask] / safe_denominator[valid_mask], -1.0, 1.0)
    angle = np.full(len(base), np.nan, dtype=float)
    angle[valid_mask] = np.arccos(cos_theta[valid_mask])
    if degrees:
        angle = np.degrees(angle)

    label = metric_name or ("segment_angle_deg" if degrees else "segment_angle_rad")
    return _result_frame(base, angle, metric=label)


__all__ = [
    "centroid",
    "pairwise_delta",
    "pairwise_distance",
    "point_to_centroid_distance",
    "segment_angle",
]
