from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mocopi.camera_projection import CameraProjectionResult
from mocopi.visualization import CAMERA_EDGES


DIAGNOSTIC_COLORS = [
    (72, 214, 118),
    (54, 151, 255),
    (255, 88, 88),
    (255, 203, 67),
]

COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def _color_for_index(index: int) -> tuple[int, int, int]:
    return DIAGNOSTIC_COLORS[index % len(DIAGNOSTIC_COLORS)]


def _normalize_components(
    components: Sequence[str] | None,
    *,
    max_components: int | None = None,
) -> list[str]:
    normalized: list[str] = []
    for component in components or ("x", "y"):
        comp = str(component).lower()
        if comp not in COMPONENT_INDEX:
            raise ValueError(f"Unknown projection component: {component!r}")
        if comp not in normalized:
            normalized.append(comp)
    if not normalized:
        normalized = ["x", "y"]
    if max_components is not None:
        normalized = normalized[:max_components]
    return normalized


def prepare_pose_landmarks_by_frame(
    df: pd.DataFrame,
    *,
    visibility_threshold: float | None = None,
) -> dict[int, dict[str, tuple[float, float]]]:
    pose_df = df[df["source"] == "pose"].copy()
    if pose_df.empty:
        return {}

    has_visibility = "visibility" in pose_df.columns
    per_frame: dict[int, dict[str, tuple[float, float]]] = {}
    for _, row in pose_df.iterrows():
        if visibility_threshold is not None and has_visibility:
            visibility = float(row["visibility"])
            if np.isnan(visibility) or visibility < visibility_threshold:
                continue
        frame_idx = int(row["frame"])
        landmark_name = str(row["landmark_name"])
        x = float(row["x"])
        y = float(row["y"])
        per_frame.setdefault(frame_idx, {})[landmark_name] = (x, y)
    return per_frame


def _draw_full_pose_skeleton(
    frame: np.ndarray,
    pose_landmarks: dict[str, tuple[float, float]],
) -> None:
    h, w = frame.shape[:2]
    pts: dict[str, tuple[int, int]] = {}
    for name, (xn, yn) in pose_landmarks.items():
        if not np.isfinite(xn) or not np.isfinite(yn):
            continue
        pts[name] = (int(xn * w), int(yn * h))

    for name_a, name_b in CAMERA_EDGES:
        if name_a in pts and name_b in pts:
            cv2.line(frame, pts[name_a], pts[name_b], (160, 160, 160), 1, cv2.LINE_AA)
    for point in pts.values():
        cv2.circle(frame, point, 2, (195, 195, 195), -1)


def _draw_component_trace_panel(
    frame: np.ndarray,
    *,
    result: CameraProjectionResult,
    row_idx: int,
    landmarks: Sequence[str],
    component_names: Sequence[str],
    x0: int,
    y0: int,
    width: int,
    height: int,
    trail_length: int,
    component_limits: dict[str, float],
    scale_factor: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (210, 210, 210), 2)

    title_h = max(32, int(34 * scale_factor))
    inner_pad = max(12, int(14 * scale_factor))
    gap = max(10, int(12 * scale_factor))
    label_pad = max(26, int(30 * scale_factor))
    bottom_pad = max(14, int(16 * scale_factor))
    num_plots = max(1, len(component_names))
    plot_h = max(48, (height - title_h - bottom_pad - gap * max(num_plots - 1, 0)) // num_plots)
    plot_w = max(100, width - label_pad - inner_pad * 2)
    plot_x0 = x0 + label_pad + inner_pad
    plot_x1 = plot_x0 + plot_w

    cv2.putText(
        frame,
        "Projected components",
        (x0 + inner_pad, y0 + max(24, int(28 * scale_factor))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64 * scale_factor,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    def draw_subplot(
        label: str,
        panel_y0: int,
        panel_y1: int,
        component_limit: float,
    ) -> None:
        component_index = COMPONENT_INDEX[label]
        center_y = (panel_y0 + panel_y1) // 2
        cv2.rectangle(frame, (plot_x0, panel_y0), (plot_x1, panel_y1), (48, 48, 48), 1)
        cv2.line(frame, (plot_x0, center_y), (plot_x1, center_y), (100, 100, 100), 1)
        cv2.putText(
            frame,
            f"d{label}",
            (x0 + inner_pad, center_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 * scale_factor,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        lo = max(0, row_idx - trail_length + 1)
        trail_times = result.timestamps_ms[lo : row_idx + 1]
        if trail_times.size <= 1:
            x_min = 0.0
            x_span = 1.0
        else:
            x_min = float(trail_times[0])
            x_span = max(float(trail_times[-1] - trail_times[0]), 1.0)

        amplitude = max(component_limit, 0.5)
        y_center = 0.5 * (panel_y0 + panel_y1)
        y_radius = 0.42 * max(panel_y1 - panel_y0, 1)

        for landmark_idx, landmark_name in enumerate(landmarks):
            if landmark_name not in result.positions:
                continue
            color = _color_for_index(landmark_idx)
            series = result.positions[landmark_name][lo : row_idx + 1, component_index]
            polyline: list[tuple[int, int]] = []
            for sample_idx, value in enumerate(series):
                if not np.isfinite(value):
                    continue
                t_ms = float(trail_times[sample_idx]) if sample_idx < trail_times.size else x_min
                px = int(plot_x0 + ((t_ms - x_min) / x_span) * plot_w)
                py = int(y_center - (float(value) / amplitude) * y_radius)
                polyline.append((px, py))
            for start, end in zip(polyline[:-1], polyline[1:]):
                cv2.line(frame, start, end, color, 2, cv2.LINE_AA)
            if polyline:
                cv2.circle(frame, polyline[-1], 4, color, -1)

    panel_y0 = y0 + title_h
    for idx, component_name in enumerate(component_names):
        current_y0 = panel_y0 + idx * (plot_h + gap)
        current_y1 = current_y0 + plot_h
        draw_subplot(
            component_name,
            current_y0,
            current_y1,
            component_limits.get(component_name, 0.5),
        )


def plot_projection_components(
    result: CameraProjectionResult,
    landmarks: Sequence[str],
    output_path: Path,
    *,
    title: str,
    components: Sequence[str] | None = None,
) -> None:
    component_names = _normalize_components(components)
    times_s = result.timestamps_ms / 1000.0
    fig, axes = plt.subplots(len(component_names) + 1, 1, figsize=(9, 2.1 * len(component_names) + 2.2), sharex=True)
    component_axes = axes[:-1]
    ax_state = axes[-1]

    for idx, landmark_name in enumerate(landmarks):
        if landmark_name not in result.positions:
            continue
        series = result.positions[landmark_name]
        color = np.array(_color_for_index(idx), dtype=float) / 255.0
        for component_name, axis in zip(component_names, component_axes):
            axis.plot(
                times_s,
                series[:, COMPONENT_INDEX[component_name]],
                label=landmark_name,
                color=color,
            )

    body_angle_deg = np.degrees(np.arctan2(result.body_y_axis[:, 1], result.body_y_axis[:, 0]))
    ax_state.plot(times_s, result.scale, color="black", label="scale")
    ax_state_2 = ax_state.twinx()
    ax_state_2.plot(times_s, body_angle_deg, color="#aa3377", alpha=0.7, label="body angle")

    for component_name, axis in zip(component_names, component_axes):
        axis.set_ylabel(f"d{component_name}")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right", fontsize=8, frameon=False)
    ax_state.set_ylabel("scale")
    ax_state_2.set_ylabel("angle (deg)")
    ax_state.set_xlabel("Time (s)")

    ax_state.grid(alpha=0.3)
    ax_state.legend(loc="upper left", fontsize=8, frameon=False)
    ax_state_2.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_projection_overlay_video(
    *,
    video_path: Path,
    result: CameraProjectionResult,
    pose_landmarks_by_frame: dict[int, dict[str, tuple[float, float]]],
    landmarks: Sequence[str],
    output_path: Path,
    rotation_code: int | None,
    max_frames: int | None,
    title: str,
    projection_frame_label: str,
    trace_components: Sequence[str] | None = None,
    trail_length: int = 60,
) -> None:
    component_names = _normalize_components(trace_components, max_components=3)
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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(result.frame_indices)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {output_path}")

    frame_to_row = {int(frame_idx): row_idx for row_idx, frame_idx in enumerate(result.frame_indices)}
    positions_stack = []
    for landmark_name in landmarks:
        if landmark_name in result.positions:
            positions_stack.append(result.positions[landmark_name])
    if positions_stack:
        all_positions = np.concatenate(positions_stack, axis=0)
        component_limits = {}
        for name in component_names:
            values = all_positions[:, COMPONENT_INDEX[name]]
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                component_limits[name] = max(float(np.max(np.abs(finite_values))), 0.5)
            else:
                component_limits[name] = 1.0
    else:
        component_limits = {name: 1.0 for name in component_names}

    header_scale = max(height / 1080.0, 0.9) * 0.85
    header_thickness = max(2, int(round(2.0 * max(height / 1080.0, 0.9))))

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        row_idx = frame_to_row.get(frame_idx)
        if row_idx is not None:
            pose_landmarks = pose_landmarks_by_frame.get(frame_idx, {})
            if pose_landmarks:
                _draw_full_pose_skeleton(frame, pose_landmarks)

            origin = result.origin_xy[row_idx]
            if np.all(np.isfinite(origin)):
                ox = int(origin[0] * width)
                oy = int(origin[1] * height)
                cv2.circle(frame, (ox, oy), 6, (0, 220, 255), -1)
                scale_norm = float(result.scale[row_idx]) if np.isfinite(result.scale[row_idx]) else 0.1
                axis_scale = max(0.06, 0.9 * scale_norm)
                x_axis = result.body_x_axis[row_idx]
                y_axis = result.body_y_axis[row_idx]
                x_tip = (
                    int((origin[0] + x_axis[0] * axis_scale) * width),
                    int((origin[1] + x_axis[1] * axis_scale) * height),
                )
                y_tip = (
                    int((origin[0] + y_axis[0] * axis_scale) * width),
                    int((origin[1] + y_axis[1] * axis_scale) * height),
                )
                cv2.arrowedLine(frame, (ox, oy), x_tip, (0, 64, 255), 3, tipLength=0.18)
                cv2.arrowedLine(frame, (ox, oy), y_tip, (72, 214, 118), 3, tipLength=0.18)

            for landmark_idx, landmark_name in enumerate(landmarks):
                color = _color_for_index(landmark_idx)
                raw_pt = pose_landmarks.get(landmark_name)
                if raw_pt is not None and np.all(np.isfinite(raw_pt)):
                    raw_cx = int(raw_pt[0] * width)
                    raw_cy = int(raw_pt[1] * height)
                    used_in_projection = False
                    if landmark_name in result.valid_mask and row_idx < len(result.valid_mask[landmark_name]):
                        used_in_projection = bool(result.valid_mask[landmark_name][row_idx])
                    if used_in_projection:
                        cv2.circle(frame, (raw_cx, raw_cy), 8, color, -1)
                        cv2.circle(frame, (raw_cx, raw_cy), 11, (20, 20, 20), 2)
                    else:
                        cv2.circle(frame, (raw_cx, raw_cy), 9, color, 2)
                        cv2.line(frame, (raw_cx - 6, raw_cy - 6), (raw_cx + 6, raw_cy + 6), color, 2, cv2.LINE_AA)
                        cv2.line(frame, (raw_cx - 6, raw_cy + 6), (raw_cx + 6, raw_cy - 6), color, 2, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        landmark_name,
                        (raw_cx + 10, raw_cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * max(height / 1080.0, 0.9),
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                if landmark_name not in result.image_points:
                    continue
                pt = result.image_points[landmark_name][row_idx]
                if np.all(np.isfinite(pt)):
                    cx = int(pt[0] * width)
                    cy = int(pt[1] * height)
                    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

            inset_w = max(280, int(0.33 * width))
            inset_h = max(240, int(0.33 * height))
            inset_x0 = width - inset_w - 20
            inset_y0 = 20
            _draw_component_trace_panel(
                frame,
                result=result,
                row_idx=row_idx,
                landmarks=landmarks,
                component_names=component_names,
                x0=inset_x0,
                y0=inset_y0,
                width=inset_w,
                height=inset_h,
                trail_length=trail_length,
                component_limits=component_limits,
                scale_factor=max(height / 1080.0, 0.9),
            )

            for landmark_idx, landmark_name in enumerate(landmarks):
                if landmark_name not in result.positions:
                    continue
                color = _color_for_index(landmark_idx)
                current = result.positions[landmark_name][row_idx]
                component_terms = []
                for component_name in component_names:
                    value = current[COMPONENT_INDEX[component_name]]
                    if np.isfinite(value):
                        component_terms.append(f"d{component_name}={value:+.2f}")
                if component_terms:
                    value_text = f"{landmark_name}: " + " ".join(component_terms)
                else:
                    value_text = f"{landmark_name}: missing"
                cv2.putText(
                    frame,
                    value_text,
                    (24, height - 80 + landmark_idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * max(height / 1080.0, 0.9),
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                "Filled = used in projection | X = raw point excluded by threshold",
                (24, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54 * max(height / 1080.0, 0.9),
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            title,
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            header_scale,
            (240, 240, 240),
            header_thickness,
            cv2.LINE_AA,
        )
        header_time_ms = (
            float(result.timestamps_ms[row_idx])
            if row_idx is not None and row_idx < len(result.timestamps_ms)
            else (frame_idx * 1000.0 / fps)
        )
        cv2.putText(
            frame,
            f"Projection frame: {projection_frame_label} | Frame {frame_idx} | t={header_time_ms / 1000.0:.2f}s",
            (24, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 * max(height / 1080.0, 0.9),
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    cap.release()
    writer.release()


__all__ = [
    "plot_projection_components",
    "prepare_pose_landmarks_by_frame",
    "render_projection_overlay_video",
]
