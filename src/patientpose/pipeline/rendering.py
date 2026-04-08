from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mocopi import estimate_camera_to_mocopi_offset, load_mocopi_recording
from mocopi.features import NoCameraPoseDataError
from mocopi.visualization import (
    draw_camera_skeleton,
    draw_mocopi_skeleton,
    prepare_camera_landmarks,
    prepare_mocopi_positions,
)
from patientpose.artifacts import ArtifactStore
from patientpose.config import resolve_project_paths
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


def build_side_by_side_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Mocopi vs camera side-by-side video.")
    return add_side_by_side_args(parser)


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
    offset_ms: float,
) -> None:
    h = combined.shape[0]
    cv2.line(combined, (panel_width, 0), (panel_width, h - 1), (200, 200, 200), 2)
    footer = (
        f"Frame {frame_idx} | camera {camera_time_ms / 1000.0:0.2f}s | "
        f"mocopi {mocopi_time_ms / 1000.0:0.2f}s | offset {offset_ms:+0.0f} ms"
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


def main_side_by_side(argv: list[str] | None = None) -> None:
    parser = build_side_by_side_parser()
    args = parser.parse_args(argv)
    run_side_by_side(args)
