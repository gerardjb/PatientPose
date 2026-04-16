from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from patientpose.landmarks import build_multi_landmark_series, landmark_stem_from_image_csv, load_landmark_views
from patientpose.metrics import (
    centroid,
    pairwise_delta,
    pairwise_distance,
    pinch_velocity,
    point_to_centroid_distance,
    segment_angle,
    thumb_index_distance,
)


TRACE_TABLE_COLUMNS = (
    "input_csv",
    "stem",
    "requested_metric",
    "metric_label",
    "source",
    "space",
    "handedness",
    "instance_id",
    "landmarks",
    "components",
    "delta_component",
    "quality_threshold",
    "smooth_window",
    "frame",
    "timestamp_ms",
    "metric",
    "value",
)

SUMMARY_TABLE_COLUMNS = (
    "input_csv",
    "stem",
    "requested_metric",
    "metric_label",
    "source",
    "space",
    "handedness",
    "instance_id",
    "landmarks",
    "components",
    "delta_component",
    "quality_threshold",
    "smooth_window",
    "n_total",
    "n_valid",
    "valid_fraction",
    "duration_s",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p05",
    "p25",
    "p75",
    "p95",
)


@dataclass(frozen=True)
class LandmarkMetricTraceResult:
    input_csv: Path
    stem: str
    requested_metric: str
    metric_label: str
    source: str
    space: str
    handedness: str | None
    instance_id: int | None
    landmarks: tuple[str, ...]
    components: tuple[str, ...]
    delta_component: str | None
    quality_threshold: float | None
    smooth_window: int
    metric_df: pd.DataFrame


def _normalize_components(components: Sequence[str] | None, *, default: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for component in components or default:
        value = str(component).lower()
        if value not in {"x", "y", "z"}:
            raise ValueError(f"Unknown component '{component}'. Choose from x, y, z.")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _apply_quality_threshold_and_smoothing(
    series_df: pd.DataFrame,
    *,
    columns: Sequence[str],
    smooth_window: int,
    quality_threshold: float | None,
) -> pd.DataFrame:
    out = series_df.copy()
    if quality_threshold is not None and "quality_score" in out.columns:
        low_quality = out["quality_score"].notna() & (out["quality_score"] < quality_threshold)
        out.loc[low_quality, list(columns)] = np.nan
    if smooth_window > 1:
        for column in columns:
            if column in out.columns:
                out[column] = out[column].rolling(window=smooth_window, center=True, min_periods=1).mean()
    return out


def _build_landmark_series_map(
    df: pd.DataFrame,
    *,
    landmarks: Sequence[str],
    source: str,
    handedness: str | None,
    instance_id: int | None,
    components: Sequence[str],
    quality_threshold: float | None,
    smooth_window: int,
) -> dict[str, pd.DataFrame]:
    if not landmarks:
        raise ValueError("--landmarks is required for landmark metric computation.")
    extra_columns = ("quality_score",) if "quality_score" in df.columns else ()
    series_map = build_multi_landmark_series(
        df,
        landmarks,
        source=source,
        handedness=handedness,
        instance_id=instance_id,
        components=components,
        extra_columns=extra_columns,
    )
    return {
        label: _apply_quality_threshold_and_smoothing(
            series_df,
            columns=components,
            smooth_window=smooth_window,
            quality_threshold=quality_threshold,
        )
        for label, series_df in series_map.items()
    }


def _resolve_metric_landmarks(metric: str, landmarks: Sequence[str] | None) -> tuple[str, ...]:
    if landmarks:
        return tuple(str(landmark) for landmark in landmarks)
    if metric in {"thumb-index-distance", "pinch-velocity"}:
        return ("THUMB_TIP", "INDEX_FINGER_TIP")
    raise ValueError(f"--landmarks is required for metric '{metric}'.")


def _compute_metric_trace(
    *,
    metric: str,
    series_map: dict[str, pd.DataFrame],
    components: Sequence[str],
    delta_component: str | None,
) -> tuple[pd.DataFrame, str]:
    names = list(series_map.keys())
    if metric == "distance":
        if len(names) != 2:
            raise ValueError("Metric 'distance' requires exactly 2 landmarks.")
        metric_df = pairwise_distance(series_map[names[0]], series_map[names[1]], components=components)
        return metric_df, f"{names[0]} vs {names[1]} distance"
    if metric == "delta":
        if len(names) != 2:
            raise ValueError("Metric 'delta' requires exactly 2 landmarks.")
        component = delta_component or components[0]
        metric_df = pairwise_delta(series_map[names[0]], series_map[names[1]], component=component)
        return metric_df, f"{names[0]} - {names[1]} ({component})"
    if metric == "centroid-distance":
        if len(names) < 2:
            raise ValueError("Metric 'centroid-distance' requires at least 2 landmarks.")
        point_name = names[0]
        centroid_df = centroid([series_map[name] for name in names[1:]], components=components)
        metric_df = point_to_centroid_distance(
            series_map[point_name],
            centroid_df,
            components=components,
        )
        return metric_df, f"{point_name} to centroid({', '.join(names[1:])})"
    if metric == "angle":
        if len(names) != 4:
            raise ValueError("Metric 'angle' requires exactly 4 landmarks.")
        metric_df = segment_angle(
            series_map[names[0]],
            series_map[names[1]],
            series_map[names[2]],
            series_map[names[3]],
            components=components,
        )
        return metric_df, f"angle({names[0]}-{names[1]}, {names[2]}-{names[3]})"
    if metric == "thumb-index-distance":
        if len(names) != 2:
            raise ValueError("Metric 'thumb-index-distance' requires exactly 2 landmarks.")
        metric_df = thumb_index_distance(series_map[names[0]], series_map[names[1]], components=tuple(components))
        return metric_df, "thumb-index distance"
    if metric == "pinch-velocity":
        if len(names) != 2:
            raise ValueError("Metric 'pinch-velocity' requires exactly 2 landmarks.")
        metric_df = pinch_velocity(series_map[names[0]], series_map[names[1]], components=tuple(components))
        return metric_df, "pinch velocity"
    raise ValueError(f"Unsupported metric '{metric}'.")


def compute_landmark_metric_trace(
    camera_csv: Path,
    *,
    metric: str,
    source: str = "hand",
    space: str = "image",
    project_root: Path | None = None,
    world_csv: Path | None = None,
    handedness: str | None = None,
    instance_id: int | None = None,
    landmarks: Sequence[str] | None = None,
    components: Sequence[str] | None = None,
    delta_component: str | None = None,
    quality_threshold: float | None = None,
    smooth_window: int = 1,
) -> LandmarkMetricTraceResult:
    resolved_camera_csv = camera_csv.resolve()
    landmark_views = load_landmark_views(
        resolved_camera_csv,
        project_root=project_root,
        world_csv=world_csv,
        require_world=space == "world",
    )
    source_df = landmark_views.world_df if space == "world" else landmark_views.image_df
    if source_df is None:
        raise FileNotFoundError(f"No {space}-space landmark data available for {resolved_camera_csv}.")

    normalized_components = _normalize_components(components, default=("x", "y"))
    resolved_landmarks = _resolve_metric_landmarks(metric, landmarks)
    series_map = _build_landmark_series_map(
        source_df,
        landmarks=resolved_landmarks,
        source=source,
        handedness=handedness,
        instance_id=instance_id,
        components=normalized_components,
        quality_threshold=quality_threshold,
        smooth_window=smooth_window,
    )
    metric_df, metric_label = _compute_metric_trace(
        metric=metric,
        series_map=series_map,
        components=normalized_components,
        delta_component=delta_component,
    )
    return LandmarkMetricTraceResult(
        input_csv=resolved_camera_csv,
        stem=landmark_stem_from_image_csv(resolved_camera_csv),
        requested_metric=metric,
        metric_label=metric_label,
        source=source,
        space=space,
        handedness=handedness,
        instance_id=instance_id,
        landmarks=tuple(series_map.keys()),
        components=normalized_components,
        delta_component=delta_component,
        quality_threshold=quality_threshold,
        smooth_window=smooth_window,
        metric_df=metric_df,
    )


def metric_trace_table(result: LandmarkMetricTraceResult) -> pd.DataFrame:
    out = result.metric_df.copy()
    out.insert(0, "smooth_window", result.smooth_window)
    out.insert(0, "quality_threshold", result.quality_threshold)
    out.insert(0, "delta_component", result.delta_component)
    out.insert(0, "components", ",".join(result.components))
    out.insert(0, "landmarks", ",".join(result.landmarks))
    out.insert(0, "instance_id", result.instance_id)
    out.insert(0, "handedness", result.handedness)
    out.insert(0, "space", result.space)
    out.insert(0, "source", result.source)
    out.insert(0, "metric_label", result.metric_label)
    out.insert(0, "requested_metric", result.requested_metric)
    out.insert(0, "stem", result.stem)
    out.insert(0, "input_csv", str(result.input_csv))
    return out.loc[:, list(TRACE_TABLE_COLUMNS)]


def summarize_landmark_metric_trace(result: LandmarkMetricTraceResult) -> pd.DataFrame:
    values = pd.to_numeric(result.metric_df.get("value"), errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_numeric(result.metric_df.get("timestamp_ms"), errors="coerce").to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    finite_timestamps = timestamps[np.isfinite(timestamps)]

    if finite_values.size:
        mean = float(np.mean(finite_values))
        median = float(np.median(finite_values))
        std = float(np.std(finite_values, ddof=0))
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        p05, p25, p75, p95 = (float(np.percentile(finite_values, percentile)) for percentile in (5, 25, 75, 95))
    else:
        mean = median = std = value_min = value_max = p05 = p25 = p75 = p95 = np.nan

    duration_s = np.nan
    if finite_timestamps.size:
        duration_s = float((np.max(finite_timestamps) - np.min(finite_timestamps)) / 1000.0)

    record = {
        "input_csv": str(result.input_csv),
        "stem": result.stem,
        "requested_metric": result.requested_metric,
        "metric_label": result.metric_label,
        "source": result.source,
        "space": result.space,
        "handedness": result.handedness,
        "instance_id": result.instance_id,
        "landmarks": ",".join(result.landmarks),
        "components": ",".join(result.components),
        "delta_component": result.delta_component,
        "quality_threshold": result.quality_threshold,
        "smooth_window": result.smooth_window,
        "n_total": int(len(result.metric_df)),
        "n_valid": int(finite_values.size),
        "valid_fraction": float(finite_values.size / len(result.metric_df)) if len(result.metric_df) else np.nan,
        "duration_s": duration_s,
        "mean": mean,
        "median": median,
        "std": std,
        "min": value_min,
        "max": value_max,
        "p05": p05,
        "p25": p25,
        "p75": p75,
        "p95": p95,
    }
    return pd.DataFrame([record], columns=list(SUMMARY_TABLE_COLUMNS))


def export_landmark_metric_batch(
    camera_csvs: Iterable[Path],
    *,
    metric: str,
    source: str = "hand",
    space: str = "image",
    project_root: Path | None = None,
    world_csv: Path | None = None,
    handedness: str | None = None,
    instance_id: int | None = None,
    landmarks: Sequence[str] | None = None,
    components: Sequence[str] | None = None,
    delta_component: str | None = None,
    quality_threshold: float | None = None,
    smooth_window: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    for camera_csv in camera_csvs:
        result = compute_landmark_metric_trace(
            camera_csv,
            metric=metric,
            source=source,
            space=space,
            project_root=project_root,
            world_csv=world_csv,
            handedness=handedness,
            instance_id=instance_id,
            landmarks=landmarks,
            components=components,
            delta_component=delta_component,
            quality_threshold=quality_threshold,
            smooth_window=smooth_window,
        )
        trace_tables.append(metric_trace_table(result))
        summary_tables.append(summarize_landmark_metric_trace(result))

    trace_df = (
        pd.concat(trace_tables, ignore_index=True)
        if trace_tables
        else pd.DataFrame(columns=list(TRACE_TABLE_COLUMNS))
    )
    summary_df = (
        pd.concat(summary_tables, ignore_index=True)
        if summary_tables
        else pd.DataFrame(columns=list(SUMMARY_TABLE_COLUMNS))
    )
    return trace_df, summary_df


__all__ = [
    "LandmarkMetricTraceResult",
    "SUMMARY_TABLE_COLUMNS",
    "TRACE_TABLE_COLUMNS",
    "compute_landmark_metric_trace",
    "export_landmark_metric_batch",
    "metric_trace_table",
    "summarize_landmark_metric_trace",
]
