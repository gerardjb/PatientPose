from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mocopi import CameraProjectionConfig, compute_camera_projection
from patientpose.landmarks import landmark_stem_from_image_csv, load_landmark_views


BODY_ANGLE_TRACE_COLUMNS = (
    "input_csv",
    "stem",
    "space",
    "visibility_threshold",
    "smooth_window",
    "frame",
    "timestamp_ms",
    "body_angle_deg",
    "scale",
    "body_x_axis_x",
    "body_x_axis_y",
    "body_y_axis_x",
    "body_y_axis_y",
)

BODY_ANGLE_SUMMARY_COLUMNS = (
    "input_csv",
    "stem",
    "space",
    "visibility_threshold",
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
class BodyAngleTraceResult:
    input_csv: Path
    stem: str
    space: str
    visibility_threshold: float | None
    smooth_window: int
    trace_df: pd.DataFrame


def compute_body_angle_trace(
    camera_csv: Path,
    *,
    space: str = "image",
    project_root: Path | None = None,
    world_csv: Path | None = None,
    visibility_threshold: float | None = 0.4,
    smooth_window: int = 7,
) -> BodyAngleTraceResult:
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

    projection = compute_camera_projection(
        source_df,
        ("LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"),
        CameraProjectionConfig(
            space=space,
            visibility_threshold=visibility_threshold,
            smooth_window=smooth_window,
            rotate_to_body_frame=False,
        ),
    )

    body_angle_deg = np.degrees(np.arctan2(projection.body_y_axis[:, 1], projection.body_y_axis[:, 0]))
    trace_df = pd.DataFrame(
        {
            "frame": projection.frame_indices,
            "timestamp_ms": projection.timestamps_ms,
            "body_angle_deg": body_angle_deg,
            "scale": projection.scale,
            "body_x_axis_x": projection.body_x_axis[:, 0],
            "body_x_axis_y": projection.body_x_axis[:, 1],
            "body_y_axis_x": projection.body_y_axis[:, 0],
            "body_y_axis_y": projection.body_y_axis[:, 1],
        }
    )

    return BodyAngleTraceResult(
        input_csv=resolved_camera_csv,
        stem=landmark_stem_from_image_csv(resolved_camera_csv),
        space=space,
        visibility_threshold=visibility_threshold,
        smooth_window=smooth_window,
        trace_df=trace_df,
    )


def body_angle_trace_table(result: BodyAngleTraceResult) -> pd.DataFrame:
    out = result.trace_df.copy()
    out.insert(0, "smooth_window", result.smooth_window)
    out.insert(0, "visibility_threshold", result.visibility_threshold)
    out.insert(0, "space", result.space)
    out.insert(0, "stem", result.stem)
    out.insert(0, "input_csv", str(result.input_csv))
    return out.loc[:, list(BODY_ANGLE_TRACE_COLUMNS)]


def summarize_body_angle_trace(result: BodyAngleTraceResult) -> pd.DataFrame:
    values = pd.to_numeric(result.trace_df.get("body_angle_deg"), errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_numeric(result.trace_df.get("timestamp_ms"), errors="coerce").to_numpy(dtype=float)
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
        "space": result.space,
        "visibility_threshold": result.visibility_threshold,
        "smooth_window": result.smooth_window,
        "n_total": int(len(result.trace_df)),
        "n_valid": int(finite_values.size),
        "valid_fraction": float(finite_values.size / len(result.trace_df)) if len(result.trace_df) else np.nan,
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
    return pd.DataFrame([record], columns=list(BODY_ANGLE_SUMMARY_COLUMNS))


__all__ = [
    "BODY_ANGLE_SUMMARY_COLUMNS",
    "BODY_ANGLE_TRACE_COLUMNS",
    "BodyAngleTraceResult",
    "body_angle_trace_table",
    "compute_body_angle_trace",
    "summarize_body_angle_trace",
]
