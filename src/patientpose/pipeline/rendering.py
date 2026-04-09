from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from mocopi import (
    estimate_camera_to_camera_offset,
    estimate_camera_to_mocopi_offset,
    load_mocopi_recording,
    nd_factor_from_stem,
    visibility_percent,
)
from mocopi.features import (
    NoCameraPoseDataError,
    compute_egocentric_positions,
)
from mocopi.camera_projection import CameraProjectionConfig, compute_camera_projection
from mocopi.visualization import (
    draw_camera_skeleton,
    draw_mocopi_skeleton,
    prepare_camera_landmarks,
    prepare_mocopi_positions,
)
from patientpose.artifacts import ArtifactStore
from patientpose.config import resolve_project_paths
from patientpose.datasets import discover_pairs, discover_sessions, infer_camera_csv, parse_camera_role_specs
from patientpose.landmarks import infer_pose_world_csv
from video_tools import determine_rotation_code, rotate_frame


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".m4v", ".mkv"}
POSE_MODEL_FILENAME = "pose_landmarker.task"
VIDEO_ROTATION_CHOICES = {
    "auto": None,
    "none": None,
    "90cw": cv2.ROTATE_90_CLOCKWISE,
    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}
PROCESSED_VIDEO_PREFIXES = (
    "deidentified_",
    "deidentified_no_keypoints_",
    "quality_vis_",
    "quality_vis_no_keypoints_",
)
PANEL_CAMERA_COLOR = (72, 214, 118)
PANEL_MOCOPI_COLOR = (65, 189, 255)
PANEL_A_COLOR = (72, 214, 118)
PANEL_ND_COLOR = (54, 151, 255)
COLOR_MOCOPI = "#000000"
COLOR_A = "#1d4f8a"
COLOR_ND = "#ff00ff"
DEFAULT_FOURPANEL_JOINTS = ["l_foot", "r_foot"]
DEFAULT_FOURPANEL_LANDMARKS = ["LEFT_ANKLE", "RIGHT_ANKLE"]
COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def add_side_by_side_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument(
        "--motion",
        "--bvh",
        dest="motion_source",
        type=Path,
        required=True,
        help="Path to Mocopi motion source (.bvh, .bin, or session directory).",
    )
    parser.add_argument(
        "--camera_csv",
        type=Path,
        required=True,
        help="Path to camera landmarks CSV (results/OutputCSVs).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Explicit path to the camera-panel video. Overrides --camera-panel-source inference.",
    )
    parser.add_argument(
        "--camera-panel-source",
        choices=("auto", "deidentified", "deidentified-no-keypoints", "raw"),
        default="auto",
        help=(
            "What to show in the left camera panel when --video is not given. "
            "'auto' prefers deidentified_no_keypoints, then deidentified, then raw."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output side-by-side video. Defaults to <project-root>/results/OutputVideos/mocopi_vs_camera.avi.",
    )
    parser.add_argument(
        "--offset_ms",
        type=float,
        default=None,
        help="Optional camera-to-mocopi offset in ms. If omitted, estimated from features.",
    )
    parser.add_argument(
        "--search_ms",
        type=float,
        default=5000.0,
        help="Search range for offset estimation in ms (+/- value).",
    )
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=50.0,
        help="Resampling rate (Hz) for offset estimation.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to render.",
    )
    parser.add_argument(
        "--video-rotation",
        choices=tuple(VIDEO_ROTATION_CHOICES.keys()),
        default="auto",
        help=(
            "Rotation to apply to the camera video before drawing overlays. "
            "Use 'auto' to run the same pose-based orientation inference used during preprocessing."
        ),
    )
    parser.add_argument(
        "--orientation-max-scan",
        type=int,
        default=None,
        help="Maximum number of frames to scan when --video-rotation=auto.",
    )
    parser.add_argument(
        "--mocopi-view",
        choices=("body-centered", "walk-range"),
        default="body-centered",
        help="How to frame the Mocopi panel. 'body-centered' keeps the skeleton centered; 'walk-range' shows progress across the full trajectory.",
    )
    return parser


def add_triplet_video_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument("--tags", nargs="+", help="Optional subset of tags to process.")
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
    )
    parser.add_argument(
        "--camera-panel-source",
        choices=("auto", "deidentified", "deidentified-no-keypoints", "raw"),
        default="auto",
        help="What to show for the A/ND camera panels. 'auto' prefers deidentified, then deidentified-no-keypoints, then raw.",
    )
    parser.add_argument(
        "--offset_ms",
        type=float,
        default=None,
        help="Optional fixed offset to reuse for both A and ND.",
    )
    parser.add_argument("--search_ms", type=float, default=5000.0, help="Search range for offset estimation.")
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Resampling rate for offset estimation.")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for triplet videos. Defaults to <project-root>/results/OutputVideos/triplets.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to render per triplet.",
    )
    parser.add_argument(
        "--video-rotation",
        choices=tuple(VIDEO_ROTATION_CHOICES.keys()),
        default="auto",
        help="Rotation to apply to raw camera panel videos before drawing overlays.",
    )
    parser.add_argument(
        "--orientation-max-scan",
        type=int,
        default=None,
        help="Maximum number of frames to scan when --video-rotation=auto.",
    )
    parser.add_argument(
        "--mocopi-view",
        choices=("body-centered", "walk-range"),
        default="body-centered",
        help="How to frame the Mocopi panel.",
    )
    return parser


def add_fourpanel_triplet_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument("--tag", required=True, help="Triplet tag to plot.")
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=DEFAULT_FOURPANEL_JOINTS,
        help="Mocopi joint names to plot.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=DEFAULT_FOURPANEL_LANDMARKS,
        help="Camera pose landmarks to plot in the same order.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.4,
        help="Visibility threshold for camera landmarks.",
    )
    parser.add_argument(
        "--camera-space",
        choices=("image", "world"),
        default="world",
        help="Which camera pose representation to plot.",
    )
    parser.add_argument(
        "--plot-component",
        choices=("x", "y", "z"),
        default=None,
        help="Projected component to plot. Defaults to z for world and y for image.",
    )
    parser.add_argument(
        "--offset-ms",
        type=float,
        default=None,
        help="Optional fixed camera-to-mocopi offset (ms) to apply before plotting. If omitted, estimated by cross-correlation.",
    )
    parser.add_argument("--search_ms", type=float, default=5000.0, help="Search range for offset estimation.")
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Resampling rate for offset estimation.")
    parser.add_argument("--x-min", type=float, default=None, help="Optional minimum time (s) for all x-axes.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional maximum time (s) for all x-axes.")
    parser.add_argument("--y-dy-min", type=float, default=None, help="Optional minimum dY for motion panels.")
    parser.add_argument("--y-dy-max", type=float, default=None, help="Optional maximum dY for motion panels.")
    parser.add_argument(
        "--y-count-min",
        type=float,
        default=None,
        help="Optional minimum for the visibility-count panel.",
    )
    parser.add_argument(
        "--y-count-max",
        type=float,
        default=None,
        help="Optional maximum for the visibility-count panel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the PDF plot.",
    )
    return parser


def build_side_by_side_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Mocopi vs camera side-by-side video.")
    return add_side_by_side_args(parser)


def build_triplet_video_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render three-panel A/ND/Mocopi triplet videos.")
    return add_triplet_video_args(parser)


def build_fourpanel_triplet_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render four-panel egocentric plots for a triplet.")
    return add_fourpanel_triplet_args(parser)


def _resolve_cli_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path).resolve()


def _csv_stem(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.startswith("landmarks_"):
        stem = stem[len("landmarks_") :]
    return stem


def _infer_raw_video_from_csv(csv_path: Path, project_root: Path, metadata: dict | None = None) -> Path:
    if metadata and metadata.get("source_video"):
        metadata_path = Path(metadata["source_video"]).resolve()
        if metadata_path.is_file():
            return metadata_path

    stem = _csv_stem(csv_path)

    legacy_path = (project_root / "sample_data" / f"{stem}.mp4").resolve()
    if legacy_path.is_file():
        return legacy_path

    sample_dir = (project_root / "sample_data").resolve()
    for candidate in sample_dir.rglob(f"{stem}.*"):
        if candidate.suffix.lower() in VIDEO_SUFFIXES:
            return candidate.resolve()
    return legacy_path


def _infer_metadata_from_csv(csv_path: Path, project_root: Path) -> Path:
    stem = _csv_stem(csv_path)
    return (project_root / "results" / "OutputCSVs" / f"landmarks_metadata_{stem}.json").resolve()


def _infer_processed_video_from_csv(
    csv_path: Path,
    project_root: Path,
    variant: str,
    metadata: dict | None = None,
) -> Path | None:
    if variant == "deidentified":
        metadata_key = "annotated_video"
        prefix = "deidentified_"
    elif variant == "deidentified-no-keypoints":
        metadata_key = "plain_video"
        prefix = "deidentified_no_keypoints_"
    else:
        raise ValueError(f"Unsupported processed video variant: {variant}")

    if metadata and metadata.get(metadata_key):
        metadata_path = Path(metadata[metadata_key]).resolve()
        if metadata_path.is_file():
            return metadata_path

    stem = _csv_stem(csv_path)
    candidate = (project_root / "results" / "OutputVideos" / f"{prefix}{stem}.avi").resolve()
    if candidate.is_file():
        return candidate
    return None


def _timestamp_for_frame(
    cam_df: pd.DataFrame,
    frame_idx: int,
    cam_landmarks: dict[str, tuple[float, float]] | None,
    fps: float,
) -> float:
    if cam_landmarks:
        any_name = next(iter(cam_landmarks.keys()))
        row = cam_df[(cam_df["frame"] == frame_idx) & (cam_df["landmark_name"] == any_name)]
        if not row.empty:
            return float(row["timestamp_ms"].iloc[0])
    return frame_idx * 1000.0 / fps


def _build_frame_timestamps(cam_df: pd.DataFrame, frame_count: int, fps: float) -> np.ndarray:
    step_ms = 1000.0 / max(fps, 1e-6)
    base = np.arange(frame_count, dtype=float) * step_ms
    if frame_count <= 0 or cam_df.empty or "frame" not in cam_df.columns or "timestamp_ms" not in cam_df.columns:
        return base

    frame_ts = (
        cam_df[["frame", "timestamp_ms"]]
        .dropna()
        .drop_duplicates(subset=["frame"])
        .sort_values("frame")
    )
    if frame_ts.empty:
        return base

    frames = frame_ts["frame"].to_numpy(dtype=int)
    timestamps = frame_ts["timestamp_ms"].to_numpy(dtype=float)
    valid = (frames >= 0) & (frames < frame_count) & np.isfinite(timestamps)
    frames = frames[valid]
    timestamps = timestamps[valid]
    if frames.size == 0:
        return base

    offset_ms = float(timestamps[0] - frames[0] * step_ms)
    full = base + offset_ms
    if frames.size == 1:
        return full

    interp_frames = np.arange(frames[0], frames[-1] + 1, dtype=float)
    full[frames[0] : frames[-1] + 1] = np.interp(interp_frames, frames.astype(float), timestamps)
    return full


def _nearest_frame_index(frame_timestamps_ms: np.ndarray, target_ms: float) -> int:
    if frame_timestamps_ms.size == 0:
        return 0
    idx = int(np.searchsorted(frame_timestamps_ms, target_ms))
    if idx <= 0:
        return 0
    if idx >= frame_timestamps_ms.size:
        return int(frame_timestamps_ms.size - 1)
    prev_idx = idx - 1
    if abs(frame_timestamps_ms[idx] - target_ms) < abs(frame_timestamps_ms[prev_idx] - target_ms):
        return idx
    return prev_idx


def _read_frame_at_index(
    cap: cv2.VideoCapture,
    target_idx: int,
    *,
    rotation_code: int | None,
    last_idx: int | None,
) -> tuple[bool, np.ndarray | None, int | None]:
    if target_idx < 0:
        target_idx = 0

    if last_idx is None or target_idx <= last_idx or target_idx > last_idx + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        last_idx = target_idx - 1

    frame: np.ndarray | None = None
    ret = False
    while last_idx < target_idx:
        ret, frame = cap.read()
        if not ret:
            return False, None, last_idx
        last_idx += 1

    if frame is None:
        ret, frame = cap.read()
        if not ret:
            return False, None, last_idx
        last_idx += 1

    return True, rotate_frame(frame, rotation_code), last_idx


def _rotation_label(rotation_code: int | None) -> str:
    labels = {
        None: "none",
        cv2.ROTATE_90_CLOCKWISE: "90cw",
        cv2.ROTATE_90_COUNTERCLOCKWISE: "90ccw",
        cv2.ROTATE_180: "180",
    }
    return labels.get(rotation_code, "unknown")


def _rotation_code_from_label(label: str | None) -> tuple[bool, int | None]:
    if label is None:
        return False, None
    normalized = label.strip().lower()
    if normalized not in VIDEO_ROTATION_CHOICES:
        return False, None
    return True, VIDEO_ROTATION_CHOICES[normalized]


def _load_processing_metadata(camera_csv_path: Path, project_root: Path) -> dict | None:
    metadata_path = _infer_metadata_from_csv(camera_csv_path, project_root)
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_camera_panel_video(
    *,
    camera_csv_path: Path,
    explicit_video_path: Path | None,
    project_root: Path,
    panel_source: str,
) -> tuple[Path, str]:
    if explicit_video_path is not None:
        return explicit_video_path, "explicit"

    metadata = _load_processing_metadata(camera_csv_path, project_root)

    if panel_source == "auto":
        annotated_video = _infer_processed_video_from_csv(
            camera_csv_path, project_root, "deidentified", metadata
        )
        if annotated_video is not None:
            return annotated_video, "deidentified"
        plain_video = _infer_processed_video_from_csv(
            camera_csv_path, project_root, "deidentified-no-keypoints", metadata
        )
        if plain_video is not None:
            return plain_video, "deidentified-no-keypoints"
        return _infer_raw_video_from_csv(camera_csv_path, project_root, metadata), "raw"

    if panel_source in {"deidentified", "deidentified-no-keypoints"}:
        processed_video = _infer_processed_video_from_csv(camera_csv_path, project_root, panel_source, metadata)
        if processed_video is None:
            raise FileNotFoundError(
                f"Could not find a {panel_source} camera-panel video for {camera_csv_path}. "
                "Run preprocess first or pass --video explicitly."
            )
        return processed_video, panel_source

    return _infer_raw_video_from_csv(camera_csv_path, project_root, metadata), "raw"


def _camera_csv_for_video(video_path: Path, project_root: Path) -> Path:
    return (project_root / infer_camera_csv(video_path)).resolve()


def _resolve_pairs_for_render(
    *,
    project_root: Path,
    tags: Sequence[str] | None,
    camera_role_specs: Sequence[str] | None,
) -> list:
    camera_roles = parse_camera_role_specs(camera_role_specs)
    pairs = discover_pairs(project_root, camera_roles=camera_roles)
    if tags:
        tag_set = set(tags)
        pairs = [pair for pair in pairs if pair.tag in tag_set]
    if not pairs:
        sessions = discover_sessions(project_root)
        if sessions and not camera_roles:
            print(
                "Discovered session folders, but no session pairs were resolved. "
                "Add --camera-role CAMERA_ID=A and --camera-role CAMERA_ID=ND."
            )
        else:
            print("No matching Mocopi/camera pairs found.")
    return pairs


def _resolve_fourpanel_offsets(
    seq,
    *,
    nd_image_df: pd.DataFrame,
    a_image_df: pd.DataFrame,
    offset_ms: float | None,
    search_ms: float,
    rate_hz: float,
) -> tuple[float, float, str]:
    if offset_ms is not None:
        fixed = float(offset_ms)
        return fixed, fixed, f"offset_fixed_{fixed:.1f}ms"

    nd_offset_ms = estimate_camera_to_mocopi_offset(
        seq,
        nd_image_df,
        search_ms,
        rate_hz,
        None,
    )
    a_offset_ms = estimate_camera_to_mocopi_offset(
        seq,
        a_image_df,
        search_ms,
        rate_hz,
        None,
    )
    return (
        float(nd_offset_ms),
        float(a_offset_ms),
        f"offset_auto_ND{nd_offset_ms:.1f}ms_A{a_offset_ms:.1f}ms",
    )


def _estimate_direct_camera_offset(
    *,
    reference_df: pd.DataFrame,
    moving_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
) -> float | None:
    try:
        return estimate_camera_to_camera_offset(
            reference_df,
            moving_df,
            search_ms=search_ms,
            rate_hz=rate_hz,
        )
    except (NoCameraPoseDataError, RuntimeError, ValueError):
        return None


def _rotated_video_size(width: int, height: int, rotation_code: int | None) -> tuple[int, int]:
    if rotation_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        return height, width
    return width, height


def _resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    if src_h == target_height:
        return frame
    scale = target_height / max(src_h, 1)
    target_width = max(1, int(round(src_w * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _fourpanel_auto_ylim_from_window(
    t_ms: np.ndarray,
    traces: dict[str, np.ndarray],
    xlim: tuple[float, float],
    component_index: int,
    padding: float = 0.05,
) -> tuple[float, float] | None:
    t_s = t_ms / 1000.0
    mask = (t_s >= xlim[0]) & (t_s <= xlim[1])
    if not np.any(mask):
        return None
    ys: list[float] = []
    for arr in traces.values():
        if arr is None or arr.shape[0] != t_s.shape[0]:
            continue
        y_vals = arr[:, component_index][mask]
        y_vals = y_vals[np.isfinite(y_vals)]
        if y_vals.size:
            ys.append(float(np.min(y_vals)))
            ys.append(float(np.max(y_vals)))
    if not ys:
        return None
    y_min = min(ys)
    y_max = max(ys)
    if y_max == y_min:
        y_pad = max(1e-3, abs(y_max) * padding)
        return y_min - y_pad, y_max + y_pad
    span = y_max - y_min
    pad = span * padding
    return y_min - pad, y_max + pad


def _ensure_percent_trace(
    t_ms: np.ndarray,
    perc: np.ndarray,
    xlim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if t_ms is None or perc is None or len(t_ms) == 0 or len(perc) == 0:
        t_fill = np.array([xlim[0] * 1000.0, xlim[1] * 1000.0], dtype=float)
        p_fill = np.zeros_like(t_fill, dtype=float)
        return t_fill, p_fill
    if np.all(~np.isfinite(perc)):
        t_fill = np.array([xlim[0] * 1000.0, xlim[1] * 1000.0], dtype=float)
        p_fill = np.zeros_like(t_fill, dtype=float)
        return t_fill, p_fill
    return t_ms, perc


def _plot_fourpanel_traces(
    ax,
    t_ms: np.ndarray,
    traces: dict[str, np.ndarray],
    label_text: str,
    xlim: tuple[float, float],
    label_color: str,
    component_index: int,
    component_label: str,
) -> None:
    t_s = t_ms / 1000.0
    for name, arr in traces.items():
        if arr is None or arr.shape[0] != t_s.shape[0]:
            continue
        is_right = name.lower().startswith("r_") or name.lower().startswith("right")
        linestyle = "--" if is_right else "-"
        ax.plot(t_s, arr[:, component_index], label=name, color=COLOR_MOCOPI, linestyle=linestyle)
    ax.set_ylabel(
        f"{label_text}\nd{component_label}",
        color=label_color,
        fontsize=12,
        rotation=0,
        ha="right",
        va="top",
        labelpad=20,
    )
    ax.grid(False)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlim(xlim)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _is_processed_video(video_path: Path) -> bool:
    return video_path.name.startswith(PROCESSED_VIDEO_PREFIXES)


def _resolve_video_rotation_code(
    video_path: Path,
    camera_csv_path: Path,
    paths,
    rotation_mode: str,
    orientation_max_scan: int | None,
) -> tuple[int | None, str]:
    if rotation_mode != "auto":
        return VIDEO_ROTATION_CHOICES[rotation_mode], "explicit"

    if _is_processed_video(video_path):
        return None, "processed-video"

    metadata = _load_processing_metadata(camera_csv_path, paths.root)
    if metadata is not None:
        found_rotation, rotation_code = _rotation_code_from_label(metadata.get("rotation_label"))
        if metadata.get("source_video") and found_rotation:
            return rotation_code, f"metadata:{metadata.get('orientation_source', 'unknown')}"

    pose_model_path = paths.models / POSE_MODEL_FILENAME
    if not pose_model_path.is_file():
        raise FileNotFoundError(
            f"Pose model file not found at {pose_model_path}; needed for --video-rotation auto."
        )

    return (
        determine_rotation_code(
            video_path,
            pose_model_path,
            rotate_flag=False,
            auto_orient=True,
            orientation_max_scan=orientation_max_scan,
        ),
        "inferred",
    )


def _draw_panel_chrome(
    panel: np.ndarray,
    *,
    title: str,
    subtitle: str,
    accent_color: tuple[int, int, int],
) -> None:
    h, w = panel.shape[:2]
    scale_factor = max(h / 1080.0, 0.9)
    title_scale = 1.1 * scale_factor
    subtitle_scale = 0.76 * scale_factor
    title_thickness = max(2, int(round(2.0 * scale_factor)))
    subtitle_thickness = max(1, int(round(1.5 * scale_factor)))
    title_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)
    subtitle_size, _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, subtitle_thickness)
    top_pad = max(18, int(0.018 * h))
    row_gap = max(16, int(0.016 * h))
    bottom_pad = max(16, int(0.016 * h))
    header_h = max(
        100,
        int(
            top_pad
            + title_size[1]
            + row_gap
            + subtitle_size[1]
            + bottom_pad
        ),
    )
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, header_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, panel, 0.28, 0.0, panel)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), accent_color, 2)
    cv2.line(panel, (0, header_h), (w - 1, header_h), accent_color, 2)
    text_x = max(20, int(0.018 * w))
    title_y = top_pad + title_size[1]
    subtitle_y = title_y + row_gap + subtitle_size[1]
    cv2.putText(
        panel,
        title,
        (text_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        accent_color,
        title_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        subtitle,
        (text_x, subtitle_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        subtitle_scale,
        (240, 240, 240),
        subtitle_thickness,
        cv2.LINE_AA,
    )


def _draw_combined_chrome(
    combined: np.ndarray,
    *,
    panel_width: int,
    frame_idx: int,
    camera_time_ms: float,
    mocopi_time_ms: float,
    offset_ms: float | None = None,
    offset_text: str | None = None,
) -> None:
    h = combined.shape[0]
    cv2.line(combined, (panel_width, 0), (panel_width, h - 1), (200, 200, 200), 2)
    if offset_text is None:
        offset_text = f"offset {float(offset_ms or 0.0):+0.0f} ms"
    footer = (
        f"Frame {frame_idx} | camera {camera_time_ms / 1000.0:0.2f}s | "
        f"mocopi {mocopi_time_ms / 1000.0:0.2f}s | {offset_text}"
    )
    scale_factor = max(h / 1080.0, 0.9)
    footer_scale = 0.72 * scale_factor
    footer_thickness = max(1, int(round(1.5 * scale_factor)))
    footer_size, _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, footer_scale, footer_thickness)
    footer_x = max(18, int(0.012 * combined.shape[1]))
    footer_y = h - max(20, int(0.02 * h))
    cv2.putText(
        combined,
        footer,
        (footer_x, max(footer_size[1] + 8, footer_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        footer_scale,
        (225, 225, 225),
        footer_thickness,
        cv2.LINE_AA,
    )


def run_side_by_side(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    motion_source = _resolve_cli_path(args.motion_source, paths.root)
    camera_csv_path = _resolve_cli_path(args.camera_csv, paths.root)
    explicit_video_path = _resolve_cli_path(args.video, paths.root) if args.video is not None else None
    video_path, camera_panel_source = _resolve_camera_panel_video(
        camera_csv_path=camera_csv_path,
        explicit_video_path=explicit_video_path,
        project_root=paths.root,
        panel_source=args.camera_panel_source,
    )
    output_path = (
        _resolve_cli_path(args.output, paths.root)
        if args.output is not None
        else artifact_store.side_by_side().output_video.resolve()
    )

    print(f"Motion source: {motion_source}")
    print(f"Camera CSV:    {camera_csv_path}")
    print(f"Video source:  {video_path}")
    print(f"Camera panel:  {camera_panel_source}")
    print(f"Output video:  {output_path}")

    seq = load_mocopi_recording(motion_source)
    cam_df = pd.read_csv(camera_csv_path)

    try:
        offset_ms = estimate_camera_to_mocopi_offset(
            seq,
            cam_df,
            search_ms=args.search_ms,
            rate_hz=args.rate_hz,
            offset_ms=args.offset_ms,
        )
    except NoCameraPoseDataError as exc:
        raise SystemExit(
            f"Camera CSV has no usable pose landmarks, so offset estimation cannot run: {camera_csv_path}"
        ) from exc

    camera_by_frame = prepare_camera_landmarks(cam_df)
    t_m_ms, mocopi_positions = prepare_mocopi_positions(seq)

    video_rotation_code, rotation_source = _resolve_video_rotation_code(
        video_path,
        camera_csv_path,
        paths,
        args.video_rotation,
        args.orientation_max_scan,
    )
    print(f"Video rotation: {_rotation_label(video_rotation_code)} ({rotation_source})")
    print(f"Mocopi view:    {args.mocopi_view}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if video_rotation_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        width = raw_height
        height = raw_width
    else:
        width = raw_width
        height = raw_height
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(camera_by_frame)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    print(f"Rendering {frame_count} frames at {fps:.2f} FPS...")

    draw_left_overlay = not video_path.name.startswith("deidentified_")

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate_frame(frame, video_rotation_code)

        left = frame.copy()
        cam_landmarks = camera_by_frame.get(frame_idx)
        if cam_landmarks and draw_left_overlay:
            draw_camera_skeleton(left, cam_landmarks)

        t_cam_ms = _timestamp_for_frame(cam_df, frame_idx, cam_landmarks, fps)
        t_mocopi_ms = t_cam_ms + offset_ms

        right = np.zeros_like(left)
        draw_mocopi_skeleton(right, mocopi_positions, t_mocopi_ms, t_m_ms, view=args.mocopi_view)
        if camera_panel_source == "deidentified-no-keypoints":
            left_title = "Camera (deidentified)"
            left_subtitle = f"t={t_cam_ms / 1000.0:0.2f}s | no keypoints"
        elif camera_panel_source == "deidentified":
            left_title = "Camera (deidentified)"
            left_subtitle = f"t={t_cam_ms / 1000.0:0.2f}s | annotated"
        elif camera_panel_source == "raw":
            left_title = "Camera (raw)"
            left_subtitle = f"t={t_cam_ms / 1000.0:0.2f}s | source video"
        else:
            left_title = "Camera"
            left_subtitle = f"t={t_cam_ms / 1000.0:0.2f}s | explicit"
        _draw_panel_chrome(
            left,
            title=left_title,
            subtitle=left_subtitle,
            accent_color=PANEL_CAMERA_COLOR,
        )
        _draw_panel_chrome(
            right,
            title="Mocopi",
            subtitle=f"t={t_mocopi_ms / 1000.0:0.2f}s | {args.mocopi_view}",
            accent_color=PANEL_MOCOPI_COLOR,
        )

        combined = np.zeros((height, width * 2, 3), dtype=left.dtype)
        combined[:, :width] = left
        combined[:, width:] = right
        _draw_combined_chrome(
            combined,
            panel_width=width,
            frame_idx=frame_idx,
            camera_time_ms=t_cam_ms,
            mocopi_time_ms=t_mocopi_ms,
            offset_ms=offset_ms,
        )
        out.write(combined)

    cap.release()
    out.release()
    print("Done.")


def run_triplet_video(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    pairs = _resolve_pairs_for_render(
        project_root=paths.root,
        tags=args.tags,
        camera_role_specs=args.camera_role,
    )
    if not pairs:
        return

    for pair in pairs:
        nd_camera_csv = _camera_csv_for_video(pair.nd_video, paths.root)
        a_camera_csv = _camera_csv_for_video(pair.unfiltered_video, paths.root)
        if not nd_camera_csv.exists() or not a_camera_csv.exists():
            print(f"[{pair.tag}] Missing camera CSVs; skipping.")
            continue

        seq = load_mocopi_recording(pair.motion_source)
        nd_df = pd.read_csv(nd_camera_csv)
        a_df = pd.read_csv(a_camera_csv)
        direct_a_to_nd_offset_ms = _estimate_direct_camera_offset(
            reference_df=nd_df,
            moving_df=a_df,
            search_ms=args.search_ms,
            rate_hz=args.rate_hz,
        )

        if args.offset_ms is not None:
            nd_offset_ms = float(args.offset_ms)
            a_offset_ms = float(args.offset_ms)
        else:
            try:
                nd_offset_ms = estimate_camera_to_mocopi_offset(
                    seq,
                    nd_df,
                    args.search_ms,
                    args.rate_hz,
                    None,
                )
                a_offset_ms = estimate_camera_to_mocopi_offset(
                    seq,
                    a_df,
                    args.search_ms,
                    args.rate_hz,
                    None,
                )
            except NoCameraPoseDataError:
                print(f"[{pair.tag}] A or ND camera CSV has no usable pose landmarks; skipping triplet video.")
                continue
            if direct_a_to_nd_offset_ms is None:
                print(f"[{pair.tag}] Estimated offsets ND={nd_offset_ms:.1f} ms, A={a_offset_ms:.1f} ms")
            else:
                print(
                    f"[{pair.tag}] Estimated offsets ND={nd_offset_ms:.1f} ms, A={a_offset_ms:.1f} ms, "
                    f"A->ND={direct_a_to_nd_offset_ms:.1f} ms"
                )

        t_m_ms, mocopi_positions = prepare_mocopi_positions(seq)
        nd_landmarks = prepare_camera_landmarks(nd_df)
        a_landmarks = prepare_camera_landmarks(a_df)

        explicit_output_dir = _resolve_cli_path(args.output_dir, paths.root) if args.output_dir is not None else None
        artifacts = artifact_store.triplet_video(pair.tag)
        output_dir = explicit_output_dir if explicit_output_dir is not None else artifacts.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"triplet_{pair.tag}.avi"

        a_video_path, a_source = _resolve_camera_panel_video(
            camera_csv_path=a_camera_csv,
            explicit_video_path=None,
            project_root=paths.root,
            panel_source=args.camera_panel_source,
        )
        nd_video_path, nd_source = _resolve_camera_panel_video(
            camera_csv_path=nd_camera_csv,
            explicit_video_path=None,
            project_root=paths.root,
            panel_source=args.camera_panel_source,
        )

        a_rotation_code, a_rotation_source = _resolve_video_rotation_code(
            a_video_path,
            a_camera_csv,
            paths,
            args.video_rotation,
            args.orientation_max_scan,
        )
        nd_rotation_code, nd_rotation_source = _resolve_video_rotation_code(
            nd_video_path,
            nd_camera_csv,
            paths,
            args.video_rotation,
            args.orientation_max_scan,
        )

        cap_a = cv2.VideoCapture(str(a_video_path))
        cap_nd = cv2.VideoCapture(str(nd_video_path))
        if not cap_a.isOpened() or not cap_nd.isOpened():
            print(f"[{pair.tag}] Could not open camera panel videos; skipping.")
            cap_a.release()
            cap_nd.release()
            continue

        raw_width_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        raw_height_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        raw_width_nd = int(cap_nd.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        raw_height_nd = int(cap_nd.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        width_a, height_a = _rotated_video_size(raw_width_a, raw_height_a, a_rotation_code)
        width_nd, height_nd = _rotated_video_size(raw_width_nd, raw_height_nd, nd_rotation_code)

        fps_nd = cap_nd.get(cv2.CAP_PROP_FPS) or 30.0
        fps_a = cap_a.get(cv2.CAP_PROP_FPS) or fps_nd
        fps = fps_nd or fps_a or 30.0
        frame_count_nd = int(cap_nd.get(cv2.CAP_PROP_FRAME_COUNT)) or max(len(nd_landmarks), 1)
        frame_count_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)) or max(len(a_landmarks), 1)
        nd_frame_timestamps_ms = _build_frame_timestamps(nd_df, frame_count_nd, fps_nd)
        a_frame_timestamps_ms = _build_frame_timestamps(a_df, frame_count_a, fps_a)
        nd_mocopi_timestamps_ms = nd_frame_timestamps_ms + nd_offset_ms
        a_mocopi_timestamps_ms = a_frame_timestamps_ms + a_offset_ms
        common_start_ms = max(
            float(t_m_ms[0]),
            float(nd_mocopi_timestamps_ms[0]),
            float(a_mocopi_timestamps_ms[0]),
        )
        common_end_ms = min(
            float(t_m_ms[-1]),
            float(nd_mocopi_timestamps_ms[-1]),
            float(a_mocopi_timestamps_ms[-1]),
        )
        render_nd_indices = np.flatnonzero(
            (nd_mocopi_timestamps_ms >= common_start_ms) & (nd_mocopi_timestamps_ms <= common_end_ms)
        )
        if render_nd_indices.size == 0:
            print(f"[{pair.tag}] No overlapping A/ND/Mocopi timeline after offset estimation; skipping.")
            cap_a.release()
            cap_nd.release()
            continue
        if args.max_frames is not None:
            render_nd_indices = render_nd_indices[: args.max_frames]
        frame_count = int(render_nd_indices.size)

        panel_height = max(height_a, height_nd)
        panel_width_a = max(1, int(round(width_a * (panel_height / max(height_a, 1)))))
        panel_width_nd = max(1, int(round(width_nd * (panel_height / max(height_nd, 1)))))
        mocopi_width = max(panel_width_a, panel_width_nd)
        out_w = panel_width_a + panel_width_nd + mocopi_width
        out_h = panel_height

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
        if not out.isOpened():
            print(f"[{pair.tag}] Could not open writer; skipping.")
            cap_a.release()
            cap_nd.release()
            continue

        print(f"[{pair.tag}] A panel: {a_video_path} ({a_source}, {_rotation_label(a_rotation_code)} / {a_rotation_source})")
        print(f"[{pair.tag}] ND panel: {nd_video_path} ({nd_source}, {_rotation_label(nd_rotation_code)} / {nd_rotation_source})")
        print(f"[{pair.tag}] Rendering {frame_count} frames to {output_path}")

        draw_overlay_a = not _is_processed_video(a_video_path)
        draw_overlay_nd = not _is_processed_video(nd_video_path)
        last_a_idx: int | None = None
        last_nd_idx: int | None = None
        if direct_a_to_nd_offset_ms is None:
            offset_text = f"offset ND {nd_offset_ms:+0.0f} ms | A {a_offset_ms:+0.0f} ms"
        else:
            offset_text = (
                f"offset ND {nd_offset_ms:+0.0f} ms | A {a_offset_ms:+0.0f} ms | "
                f"A->ND {direct_a_to_nd_offset_ms:+0.0f} ms"
            )

        for output_idx, nd_frame_idx in enumerate(render_nd_indices):
            t_nd_ms = float(nd_frame_timestamps_ms[nd_frame_idx])
            t_mocopi_ms = float(nd_mocopi_timestamps_ms[nd_frame_idx])
            target_a_time_ms = t_mocopi_ms - a_offset_ms
            a_frame_idx = _nearest_frame_index(a_frame_timestamps_ms, target_a_time_ms)

            ret_nd, frame_nd, last_nd_idx = _read_frame_at_index(
                cap_nd,
                int(nd_frame_idx),
                rotation_code=nd_rotation_code,
                last_idx=last_nd_idx,
            )
            ret_a, frame_a, last_a_idx = _read_frame_at_index(
                cap_a,
                int(a_frame_idx),
                rotation_code=a_rotation_code,
                last_idx=last_a_idx,
            )
            if not ret_a or not ret_nd:
                break

            left = frame_a.copy()
            middle = frame_nd.copy()
            lms_a = a_landmarks.get(int(a_frame_idx))
            lms_nd = nd_landmarks.get(int(nd_frame_idx))
            if lms_a and draw_overlay_a:
                draw_camera_skeleton(left, lms_a)
            if lms_nd and draw_overlay_nd:
                draw_camera_skeleton(middle, lms_nd)

            t_a_ms = float(a_frame_timestamps_ms[a_frame_idx])

            right = np.zeros((panel_height, mocopi_width, 3), dtype=np.uint8)
            draw_mocopi_skeleton(right, mocopi_positions, t_mocopi_ms, t_m_ms, view=args.mocopi_view)

            left_resized = _resize_to_height(left, panel_height)
            middle_resized = _resize_to_height(middle, panel_height)

            left_subtitle = f"t={t_a_ms / 1000.0:0.2f}s | {a_source}"
            nd_subtitle = f"t={t_nd_ms / 1000.0:0.2f}s | {nd_source}"
            _draw_panel_chrome(
                left_resized,
                title="Camera A",
                subtitle=left_subtitle,
                accent_color=PANEL_A_COLOR,
            )
            _draw_panel_chrome(
                middle_resized,
                title="Camera ND",
                subtitle=nd_subtitle,
                accent_color=PANEL_ND_COLOR,
            )
            _draw_panel_chrome(
                right,
                title="Mocopi",
                subtitle=f"t={t_mocopi_ms / 1000.0:0.2f}s | {args.mocopi_view}",
                accent_color=PANEL_MOCOPI_COLOR,
            )

            combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            combined[:, :panel_width_a] = cv2.resize(left_resized, (panel_width_a, out_h))
            combined[:, panel_width_a : panel_width_a + panel_width_nd] = cv2.resize(
                middle_resized, (panel_width_nd, out_h)
            )
            combined[:, panel_width_a + panel_width_nd :] = right
            cv2.line(combined, (panel_width_a, 0), (panel_width_a, out_h - 1), (200, 200, 200), 2)
            cv2.line(
                combined,
                (panel_width_a + panel_width_nd, 0),
                (panel_width_a + panel_width_nd, out_h - 1),
                (200, 200, 200),
                2,
            )
            _draw_combined_chrome(
                combined,
                panel_width=panel_width_a,
                frame_idx=output_idx,
                camera_time_ms=t_nd_ms,
                mocopi_time_ms=t_mocopi_ms,
                offset_text=offset_text,
            )
            out.write(combined)

        cap_a.release()
        cap_nd.release()
        out.release()
        print(f"[{pair.tag}] Saved triplet video to {output_path}")


def run_fourpanel_triplet(args: argparse.Namespace) -> None:
    if len(args.joints) != len(args.landmarks):
        raise SystemExit("Expected --joints and --landmarks to have the same length")

    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    pairs = _resolve_pairs_for_render(
        project_root=paths.root,
        tags=[args.tag],
        camera_role_specs=args.camera_role,
    )
    if not pairs:
        raise SystemExit(f"Tag '{args.tag}' not found in discovered Mocopi/camera pairs.")
    pair = pairs[0]

    nd_camera_csv = _camera_csv_for_video(pair.nd_video, paths.root)
    a_camera_csv = _camera_csv_for_video(pair.unfiltered_video, paths.root)
    nd_projection_csv = nd_camera_csv if args.camera_space == "image" else infer_pose_world_csv(nd_camera_csv, paths.root)
    a_projection_csv = a_camera_csv if args.camera_space == "image" else infer_pose_world_csv(a_camera_csv, paths.root)
    if not nd_camera_csv.exists() or not a_camera_csv.exists() or not nd_projection_csv.exists() or not a_projection_csv.exists():
        raise SystemExit(f"Missing camera CSVs for tag={args.tag} (ND: {nd_camera_csv}, A: {a_camera_csv})")

    seq = load_mocopi_recording(pair.motion_source)
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, args.joints)

    nd_image_df = pd.read_csv(nd_camera_csv)
    a_image_df = pd.read_csv(a_camera_csv)
    nd_df = pd.read_csv(nd_projection_csv)
    a_df = pd.read_csv(a_projection_csv)
    direct_a_to_nd_offset_ms = _estimate_direct_camera_offset(
        reference_df=nd_image_df,
        moving_df=a_image_df,
        search_ms=args.search_ms,
        rate_hz=args.rate_hz,
    )
    try:
        nd_offset_ms, a_offset_ms, offset_label = _resolve_fourpanel_offsets(
            seq,
            nd_image_df=nd_image_df,
            a_image_df=a_image_df,
            offset_ms=args.offset_ms,
            search_ms=args.search_ms,
            rate_hz=args.rate_hz,
        )
    except NoCameraPoseDataError as exc:
        raise SystemExit("Cannot estimate four-panel offset because one camera CSV has no usable image-space pose landmarks.") from exc
    except RuntimeError as exc:
        raise SystemExit(f"Could not estimate four-panel offset: {exc}") from exc

    plot_component = args.plot_component or ("z" if args.camera_space == "world" else "y")
    component_index = COMPONENT_INDEX[plot_component]
    try:
        nd_projection = compute_camera_projection(
            nd_df,
            args.landmarks,
            CameraProjectionConfig(
                space=args.camera_space,
                visibility_threshold=args.visibility_threshold,
                rotate_to_body_frame=(args.camera_space == "image"),
            ),
        )
    except NoCameraPoseDataError as exc:
        raise SystemExit(f"ND camera CSV has no usable {args.camera_space} pose landmarks: {nd_projection_csv}") from exc
    try:
        a_projection = compute_camera_projection(
            a_df,
            args.landmarks,
            CameraProjectionConfig(
                space=args.camera_space,
                visibility_threshold=args.visibility_threshold,
                rotate_to_body_frame=(args.camera_space == "image"),
            ),
        )
    except NoCameraPoseDataError as exc:
        raise SystemExit(f"A camera CSV has no usable {args.camera_space} pose landmarks: {a_projection_csv}") from exc
    t_nd_ms, nd_pos = nd_projection.timestamps_ms, nd_projection.positions
    t_a_ms, a_pos = a_projection.timestamps_ms, a_projection.positions

    t_nd_ms = t_nd_ms + nd_offset_ms
    t_a_ms = t_a_ms + a_offset_ms

    t_nd_count, nd_percent = visibility_percent(nd_image_df, args.visibility_threshold)
    t_a_count, a_percent = visibility_percent(a_image_df, args.visibility_threshold)
    t_nd_count = t_nd_count + nd_offset_ms
    t_a_count = t_a_count + a_offset_ms

    fig, axes = plt.subplots(4, 1, figsize=(6, 4), sharex=True)
    ax_mocopi, ax_a, ax_nd, ax_vis = axes

    time_arrays = []
    for arr in (t_m_ms, t_a_ms, t_nd_ms, t_a_count, t_nd_count):
        if arr is not None and len(arr) > 0:
            time_arrays.append(arr / 1000.0)
    if not time_arrays:
        raise SystemExit("No timestamps available to set x-limits.")
    global_min = float(min(np.min(arr) for arr in time_arrays))
    global_max = float(max(np.max(arr) for arr in time_arrays))
    x_lo = args.x_min if args.x_min is not None else global_min
    x_hi = args.x_max if args.x_max is not None else global_max
    if x_hi <= x_lo:
        raise SystemExit("Invalid x-limits: x-max must be greater than x-min.")
    xlim = (x_lo, x_hi)

    mocopi_traces = {k: mocopi_pos.get(k) for k in args.joints}
    a_traces = {k: a_pos.get(k) for k in args.landmarks}
    nd_traces = {k: nd_pos.get(k) for k in args.landmarks}

    nd_factor = nd_factor_from_stem(pair.nd_video.stem)
    nd_label_text = f"Video ND = {nd_factor:g}" if nd_factor is not None else "Video ND = ?"

    t_a_count, a_percent = _ensure_percent_trace(t_a_count, a_percent, xlim)
    t_nd_count, nd_percent = _ensure_percent_trace(t_nd_count, nd_percent, xlim)

    _plot_fourpanel_traces(ax_mocopi, t_m_ms, mocopi_traces, "Mocopi", xlim, COLOR_MOCOPI, component_index, plot_component)
    _plot_fourpanel_traces(ax_a, t_a_ms, a_traces, "Video ND\n0", xlim, COLOR_A, component_index, plot_component)
    _plot_fourpanel_traces(
        ax_nd,
        t_nd_ms,
        nd_traces,
        nd_label_text.replace("Video ND = ", "Video ND\n"),
        xlim,
        COLOR_ND,
        component_index,
        plot_component,
    )

    for ax, traces, t_ms in (
        (ax_mocopi, mocopi_traces, t_m_ms),
        (ax_a, a_traces, t_a_ms),
        (ax_nd, nd_traces, t_nd_ms),
    ):
        if args.y_dy_min is not None or args.y_dy_max is not None:
            ymin = args.y_dy_min if args.y_dy_min is not None else ax.get_ylim()[0]
            ymax = args.y_dy_max if args.y_dy_max is not None else ax.get_ylim()[1]
        else:
            yl = _fourpanel_auto_ylim_from_window(t_ms, traces, xlim, component_index)
            if yl is None:
                ymin, ymax = ax.get_ylim()
            else:
                ymin, ymax = yl
        ax.set_ylim(ymin, ymax)

    base_t = np.array([xlim[0], xlim[1]], dtype=float)
    ax_vis.plot(base_t, np.zeros_like(base_t), color=COLOR_A, linestyle=":", linewidth=1.0)
    ax_vis.plot(base_t, np.zeros_like(base_t), color=COLOR_ND, linestyle=":", linewidth=1.0)
    mask_a = np.isfinite(a_percent)
    if np.any(mask_a):
        ax_vis.plot(t_a_count[mask_a] / 1000.0, a_percent[mask_a], label="Video ND = 0", color=COLOR_A)
    mask_nd = np.isfinite(nd_percent)
    if np.any(mask_nd):
        ax_vis.plot(t_nd_count[mask_nd] / 1000.0, nd_percent[mask_nd], label=nd_label_text, color=COLOR_ND)
    ax_vis.set_ylabel("Visible\n keypoints (%)", color=COLOR_MOCOPI)
    ax_vis.grid(alpha=0.3)
    ax_vis.legend(fontsize=8, frameon=False)
    ax_vis.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_vis.set_xlim(xlim)
    for spine in ax_vis.spines.values():
        spine.set_visible(False)
    if args.y_count_min is not None or args.y_count_max is not None:
        ymin = args.y_count_min if args.y_count_min is not None else ax_vis.get_ylim()[0]
        ymax = args.y_count_max if args.y_count_max is not None else ax_vis.get_ylim()[1]
        ax_vis.set_ylim(ymin, ymax)
    else:
        counts_stack = []
        for t_ms, counts in ((t_a_count, a_percent), (t_nd_count, nd_percent)):
            if t_ms is None or counts is None or len(t_ms) != len(counts):
                continue
            t_s = t_ms / 1000.0
            mask = (t_s >= xlim[0]) & (t_s <= xlim[1])
            if not np.any(mask):
                continue
            vals = counts[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                counts_stack.extend([float(np.min(vals)), float(np.max(vals))])
        if counts_stack:
            cmin = min(counts_stack)
            cmax = max(counts_stack)
            if cmax == cmin:
                pad = max(1.0, cmax * 0.05)
                ax_vis.set_ylim(cmin - pad, cmax + pad)
            else:
                span = cmax - cmin
                pad = span * 0.05
                ax_vis.set_ylim(cmin - pad, cmax + pad)
        else:
            ax_vis.set_ylim(-1.0, 1.0)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    if args.output is None:
        output_path = artifact_store.fourpanel_triplet(
            pair.tag,
            camera_space=args.camera_space,
            component=plot_component,
            offset_label=offset_label,
            visibility_threshold=args.visibility_threshold,
        ).output_plot
    else:
        output_path = _resolve_cli_path(args.output, paths.root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(
        f"Saved four-panel plot to {output_path} "
        f"(camera_space={args.camera_space}, component=d{plot_component}, "
        f"ND_offset_ms={nd_offset_ms:.1f}, A_offset_ms={a_offset_ms:.1f}, "
        f"A_to_ND_offset_ms={direct_a_to_nd_offset_ms if direct_a_to_nd_offset_ms is not None else float('nan'):.1f}, "
        f"visibility_threshold={args.visibility_threshold:.2f})"
    )


def main_side_by_side(argv: list[str] | None = None) -> None:
    parser = build_side_by_side_parser()
    args = parser.parse_args(argv)
    run_side_by_side(args)


def main_triplet_video(argv: list[str] | None = None) -> None:
    parser = build_triplet_video_parser()
    args = parser.parse_args(argv)
    run_triplet_video(args)


def main_fourpanel_triplet(argv: list[str] | None = None) -> None:
    parser = build_fourpanel_triplet_parser()
    args = parser.parse_args(argv)
    run_fourpanel_triplet(args)
