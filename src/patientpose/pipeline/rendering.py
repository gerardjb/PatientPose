from __future__ import annotations

import argparse
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
        help="Path to original camera video. If omitted, inferred from the CSV name.",
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


def _infer_video_from_csv(csv_path: Path, project_root: Path) -> Path:
    """
    Infer the video path from a landmarks CSV filename.

    Primary target:
        <project-root>/results/OutputVideos/deidentified_<stem>.avi

    Fallback if the de-identified video is missing:
        search under <project-root>/sample_data/ for a video with the same stem
    """
    stem = csv_path.stem
    if stem.startswith("landmarks_"):
        stem = stem[len("landmarks_") :]

    deid = (project_root / "results" / "OutputVideos" / f"deidentified_{stem}.avi").resolve()
    if deid.is_file():
        return deid

    legacy_path = (project_root / "sample_data" / f"{stem}.mp4").resolve()
    if legacy_path.is_file():
        return legacy_path

    sample_dir = (project_root / "sample_data").resolve()
    for candidate in sample_dir.rglob(f"{stem}.*"):
        if candidate.suffix.lower() in VIDEO_SUFFIXES:
            return candidate.resolve()
    return legacy_path


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


def _is_processed_video(video_path: Path) -> bool:
    return video_path.name.startswith(PROCESSED_VIDEO_PREFIXES)


def _resolve_video_rotation_code(
    video_path: Path,
    paths,
    rotation_mode: str,
    orientation_max_scan: int | None,
) -> int | None:
    if rotation_mode != "auto":
        return VIDEO_ROTATION_CHOICES[rotation_mode]

    if _is_processed_video(video_path):
        return None

    pose_model_path = paths.models / POSE_MODEL_FILENAME
    if not pose_model_path.is_file():
        raise FileNotFoundError(
            f"Pose model file not found at {pose_model_path}; needed for --video-rotation auto."
        )

    return determine_rotation_code(
        video_path,
        pose_model_path,
        rotate_flag=False,
        auto_orient=True,
        orientation_max_scan=orientation_max_scan,
    )


def run_side_by_side(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    motion_source = _resolve_cli_path(args.motion_source, paths.root)
    camera_csv_path = _resolve_cli_path(args.camera_csv, paths.root)
    video_path = _resolve_cli_path(args.video, paths.root) if args.video is not None else _infer_video_from_csv(
        camera_csv_path, paths.root
    )
    output_path = (
        _resolve_cli_path(args.output, paths.root)
        if args.output is not None
        else artifact_store.side_by_side().output_video.resolve()
    )

    print(f"Motion source: {motion_source}")
    print(f"Camera CSV:    {camera_csv_path}")
    print(f"Video source:  {video_path}")
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

    video_rotation_code = _resolve_video_rotation_code(
        video_path,
        paths,
        args.video_rotation,
        args.orientation_max_scan,
    )
    print(f"Video rotation: {_rotation_label(video_rotation_code)}")
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

        combined = np.zeros((height, width * 2, 3), dtype=left.dtype)
        combined[:, :width] = left
        combined[:, width:] = right
        out.write(combined)

    cap.release()
    out.release()
    print("Done.")


def main_side_by_side(argv: list[str] | None = None) -> None:
    parser = build_side_by_side_parser()
    args = parser.parse_args(argv)
    run_side_by_side(args)
