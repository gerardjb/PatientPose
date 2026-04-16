from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mocopi.visualization import CAMERA_EDGES, normalized_point_to_pixel


DIAGNOSTIC_COLORS = [
    (72, 214, 118),
    (54, 151, 255),
    (255, 88, 88),
    (255, 203, 67),
]


def _color_for_index(index: int) -> tuple[int, int, int]:
    return DIAGNOSTIC_COLORS[index % len(DIAGNOSTIC_COLORS)]


def _frame_scale(height: int) -> float:
    return max(height / 1080.0, 0.9)


def _trace_limits(metric_df: pd.DataFrame) -> tuple[float, float]:
    if "value" not in metric_df.columns:
        return -1.0, 1.0
    values = metric_df["value"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * 0.1, 0.5)
        return lo - pad, hi + pad
    pad = max((hi - lo) * 0.08, 1e-6)
    return lo - pad, hi + pad


def _draw_pose_context(
    frame: np.ndarray,
    pose_landmarks: dict[str, tuple[float, float]],
    *,
    rotation_code: int | None,
) -> None:
    pts: dict[str, tuple[int, int]] = {}
    for name, (xn, yn) in pose_landmarks.items():
        if not np.isfinite(xn) or not np.isfinite(yn):
            continue
        pts[name] = normalized_point_to_pixel(
            frame.shape,
            (float(xn), float(yn)),
            rotation_code=rotation_code,
        )

    for name_a, name_b in CAMERA_EDGES:
        if name_a in pts and name_b in pts:
            cv2.line(frame, pts[name_a], pts[name_b], (150, 150, 150), 1, cv2.LINE_AA)
    for point in pts.values():
        cv2.circle(frame, point, 2, (200, 200, 200), -1)


def _draw_selected_landmarks(
    frame: np.ndarray,
    *,
    row_idx: int | None,
    landmark_series_map: dict[str, pd.DataFrame],
    scale: float,
    rotation_code: int | None,
) -> None:
    if row_idx is None:
        return

    radius = max(6, int(round(8 * scale)))
    outline = max(2, int(round(2 * scale)))
    font_scale = 0.52 * scale

    for landmark_idx, (label, series_df) in enumerate(landmark_series_map.items()):
        if row_idx >= len(series_df):
            continue
        row = series_df.iloc[row_idx]
        x = float(row.get("x", np.nan))
        y = float(row.get("y", np.nan))
        color = _color_for_index(landmark_idx)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        cx, cy = normalized_point_to_pixel(
            frame.shape,
            (x, y),
            rotation_code=rotation_code,
        )
        cv2.circle(frame, (cx, cy), radius, color, -1)
        cv2.circle(frame, (cx, cy), radius + outline + 1, (20, 20, 20), outline)
        cv2.putText(
            frame,
            label,
            (cx + radius + 6, cy - radius - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, outline),
            cv2.LINE_AA,
        )


def _draw_metric_trace_panel(
    frame: np.ndarray,
    *,
    metric_df: pd.DataFrame,
    metric_label: str,
    row_idx: int | None,
    current_time_ms: float,
    trace_window_seconds: float,
    scale: float,
) -> None:
    if metric_df.empty:
        return

    height, width = frame.shape[:2]
    panel_h = max(160, int(round(height * 0.24)))
    x0 = max(16, int(round(20 * scale)))
    y0 = height - panel_h - max(14, int(round(18 * scale)))
    x1 = width - x0
    y1 = y0 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (220, 220, 220), 2)

    inner_pad = max(12, int(round(14 * scale)))
    label_pad = max(52, int(round(60 * scale)))
    title_h = max(30, int(round(34 * scale)))
    plot_x0 = x0 + label_pad
    plot_x1 = x1 - inner_pad
    plot_y0 = y0 + title_h
    plot_y1 = y1 - inner_pad

    cv2.putText(
        frame,
        metric_label,
        (x0 + inner_pad, y0 + max(20, int(round(26 * scale)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62 * scale,
        (240, 240, 240),
        max(1, int(round(2 * scale))),
        cv2.LINE_AA,
    )

    y_min, y_max = _trace_limits(metric_df)
    if y_max <= y_min:
        y_max = y_min + 1.0

    cv2.rectangle(frame, (plot_x0, plot_y0), (plot_x1, plot_y1), (48, 48, 48), 1)

    if y_min < 0.0 < y_max:
        zero_y = int(round(plot_y1 - ((0.0 - y_min) / (y_max - y_min)) * (plot_y1 - plot_y0)))
        cv2.line(frame, (plot_x0, zero_y), (plot_x1, zero_y), (110, 110, 110), 1)

    cv2.putText(
        frame,
        f"{y_max:.2f}",
        (x0 + inner_pad, plot_y0 + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46 * scale,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{y_min:.2f}",
        (x0 + inner_pad, plot_y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46 * scale,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    window_ms = max(trace_window_seconds * 1000.0, 250.0)
    window_start = current_time_ms - window_ms
    timestamps = metric_df["timestamp_ms"].to_numpy(dtype=float)
    values = metric_df["value"].to_numpy(dtype=float)
    mask = np.isfinite(timestamps) & np.isfinite(values) & (timestamps >= window_start) & (timestamps <= current_time_ms)
    if np.any(mask):
        selected_times = timestamps[mask]
        selected_values = values[mask]
        x_span = max(current_time_ms - window_start, 1.0)
        polyline: list[tuple[int, int]] = []
        for time_ms, value in zip(selected_times, selected_values):
            px = int(round(plot_x0 + ((time_ms - window_start) / x_span) * (plot_x1 - plot_x0)))
            py = int(round(plot_y1 - ((value - y_min) / (y_max - y_min)) * (plot_y1 - plot_y0)))
            polyline.append((px, py))
        for start, end in zip(polyline[:-1], polyline[1:]):
            cv2.line(frame, start, end, (54, 151, 255), 2, cv2.LINE_AA)
        if polyline:
            cv2.circle(frame, polyline[-1], max(4, int(round(5 * scale))), (255, 203, 67), -1)

    current_value = np.nan
    if row_idx is not None and row_idx < len(metric_df):
        current_value = float(metric_df.iloc[row_idx].get("value", np.nan))
    value_text = f"value={current_value:+.3f}" if np.isfinite(current_value) else "value=missing"
    cv2.putText(
        frame,
        value_text,
        (plot_x1 - max(180, int(round(210 * scale))), y0 + max(20, int(round(26 * scale)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * scale,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{trace_window_seconds:.1f}s window",
        (plot_x0, plot_y1 + max(14, int(round(18 * scale)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44 * scale,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def render_landmark_overlay_video(
    *,
    video_path: Path,
    landmark_series_map: dict[str, pd.DataFrame],
    metric_df: pd.DataFrame,
    metric_label: str,
    output_path: Path,
    rotation_code: int | None,
    max_frames: int | None,
    title: str,
    overlay_label: str,
    trace_window_seconds: float = 3.0,
    pose_landmarks_by_frame: dict[int, dict[str, tuple[float, float]]] | None = None,
) -> None:
    if not landmark_series_map:
        raise ValueError("landmark_series_map must contain at least one series.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if rotation_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        width = raw_height
        height = raw_width
    else:
        width = raw_width
        height = raw_height
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_limit = max_frames if max_frames is not None else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {output_path}")

    first_landmark_df = next(iter(landmark_series_map.values()))
    landmark_frame_to_row = {
        int(frame_idx): row_idx
        for row_idx, frame_idx in enumerate(first_landmark_df["frame"].to_numpy(dtype=int))
    }
    metric_frame_to_row = {
        int(frame_idx): row_idx
        for row_idx, frame_idx in enumerate(metric_df["frame"].to_numpy(dtype=int))
    } if "frame" in metric_df.columns else {}

    scale = _frame_scale(height)
    header_scale = 0.84 * scale
    header_thickness = max(2, int(round(2.0 * scale)))
    subtitle_scale = 0.58 * scale

    frame_idx = 0
    while True:
        if frame_limit is not None and frame_idx >= frame_limit:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        landmark_row_idx = landmark_frame_to_row.get(frame_idx)
        metric_row_idx = metric_frame_to_row.get(frame_idx)
        pose_landmarks = pose_landmarks_by_frame.get(frame_idx, {}) if pose_landmarks_by_frame is not None else {}
        if pose_landmarks:
            _draw_pose_context(frame, pose_landmarks, rotation_code=rotation_code)

        _draw_selected_landmarks(
            frame,
            row_idx=landmark_row_idx,
            landmark_series_map=landmark_series_map,
            scale=scale,
            rotation_code=rotation_code,
        )

        if metric_row_idx is not None and metric_row_idx < len(metric_df):
            current_time_ms = float(metric_df.iloc[metric_row_idx].get("timestamp_ms", frame_idx * 1000.0 / fps))
        elif landmark_row_idx is not None and landmark_row_idx < len(first_landmark_df):
            current_time_ms = float(first_landmark_df.iloc[landmark_row_idx].get("timestamp_ms", frame_idx * 1000.0 / fps))
        else:
            current_time_ms = frame_idx * 1000.0 / fps

        _draw_metric_trace_panel(
            frame,
            metric_df=metric_df,
            metric_label=metric_label,
            row_idx=metric_row_idx,
            current_time_ms=current_time_ms,
            trace_window_seconds=trace_window_seconds,
            scale=scale,
        )

        cv2.putText(
            frame,
            title,
            (24, max(30, int(round(38 * scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            header_scale,
            (240, 240, 240),
            header_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{overlay_label} | Frame {frame_idx} | t={current_time_ms / 1000.0:.2f}s",
            (24, max(56, int(round(72 * scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            subtitle_scale,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


__all__ = ["render_landmark_overlay_video"]
