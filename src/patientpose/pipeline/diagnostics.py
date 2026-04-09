from __future__ import annotations

import argparse
from pathlib import Path

from mocopi import CameraProjectionConfig, compute_camera_projection
import pandas as pd
from patientpose.artifacts import ArtifactStore
from patientpose.config import resolve_project_paths
from patientpose.datasets import discover_pairs, discover_sessions, infer_camera_csv, parse_camera_role_specs
from patientpose.diagnostics import (
    plot_projection_components,
    prepare_pose_landmarks_by_frame,
    render_projection_overlay_video,
)
from patientpose.landmarks import load_landmark_views
from patientpose.pipeline.rendering import (
    _resolve_camera_panel_video,
    _resolve_cli_path,
    _resolve_video_rotation_code,
)


def add_egocentric_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PatientPose repo root. Defaults to the nearest parent containing pyproject.toml.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--camera_csv",
        type=Path,
        default=None,
        help="Direct path to the camera landmarks CSV to diagnose.",
    )
    source_group.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Resolved pair tag to diagnose.",
    )
    parser.add_argument(
        "--camera-side",
        choices=("A", "ND"),
        default="ND",
        help="Which camera side to use when resolving from --tag.",
    )
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=["LEFT_ANKLE", "RIGHT_ANKLE"],
        help="Pose landmarks to project and visualize.",
    )
    parser.add_argument(
        "--space",
        choices=("image", "world"),
        default="world",
        help="Which pose representation to analyze.",
    )
    parser.add_argument(
        "--world_csv",
        type=Path,
        default=None,
        help="Optional explicit path to the pose-world CSV. Defaults to the paired preprocess artifact.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Projected components to visualize. Choose from x y z.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.4,
        help="Visibility threshold applied before projection.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Smoothing window in frames for origin, scale, and body axes.",
    )
    parser.add_argument(
        "--body-frame",
        action="store_true",
        help="Rotate dx/dy into a body-aligned frame using hips/torso axes.",
    )
    return parser


def add_egocentric_plot_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_egocentric_common_args(parser)
    parser.set_defaults(components=["y", "z"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path for the components plot.",
    )
    return parser


def add_egocentric_video_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_egocentric_common_args(parser)
    parser.set_defaults(components=["y", "z"])
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Explicit path to the camera-panel video. Overrides --camera-panel-source.",
    )
    parser.add_argument(
        "--camera-panel-source",
        choices=("auto", "deidentified", "deidentified-no-keypoints", "raw"),
        default="deidentified-no-keypoints",
        help="What to show under the diagnostic overlays when --video is not given.",
    )
    parser.add_argument(
        "--video-rotation",
        choices=("auto", "none", "90cw", "90ccw", "180"),
        default="auto",
        help="Rotation to apply to raw camera videos before drawing overlays.",
    )
    parser.add_argument(
        "--orientation-max-scan",
        type=int,
        default=None,
        help="Maximum number of frames to scan when --video-rotation=auto.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to render.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path for the overlay video.",
    )
    return parser


def _camera_csv_for_video(video_path: Path, project_root: Path) -> Path:
    return (project_root / infer_camera_csv(video_path)).resolve()


def _resolve_camera_inputs(
    *,
    project_root: Path,
    camera_csv: Path | None,
    tag: str | None,
    camera_side: str,
    camera_role_specs: list[str] | None,
) -> tuple[Path, Path | None, str]:
    if camera_csv is not None:
        resolved_csv = _resolve_cli_path(camera_csv, project_root)
        return resolved_csv, None, resolved_csv.stem.replace("landmarks_", "")

    camera_roles = parse_camera_role_specs(camera_role_specs)
    pairs = discover_pairs(project_root, camera_roles=camera_roles)
    pair = next((candidate for candidate in pairs if candidate.tag == tag), None)
    if pair is None:
        sessions = discover_sessions(project_root)
        hint = ""
        if sessions and not camera_roles:
            hint = " Add --camera-role CAMERA_ID=A and --camera-role CAMERA_ID=ND for session data."
        raise SystemExit(f"Tag '{tag}' not found in discovered Mocopi/camera pairs.{hint}")

    video_path = pair.unfiltered_video if camera_side == "A" else pair.nd_video
    resolved_csv = _camera_csv_for_video(video_path, project_root)
    return resolved_csv, video_path, video_path.stem


def _projection_frame_label(body_frame: bool) -> str:
    return "body" if body_frame else "image"


def run_egocentric_plot(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    camera_csv_path, _, label = _resolve_camera_inputs(
        project_root=paths.root,
        camera_csv=args.camera_csv,
        tag=args.tag,
        camera_side=args.camera_side,
        camera_role_specs=args.camera_role,
    )
    landmark_views = load_landmark_views(
        camera_csv_path,
        project_root=paths.root,
        world_csv=_resolve_cli_path(args.world_csv, paths.root) if args.world_csv is not None else None,
        require_world=args.space == "world",
    )
    image_df = landmark_views.image_df
    pose_landmarks_by_frame = prepare_pose_landmarks_by_frame(image_df, visibility_threshold=None)
    projection_df = landmark_views.world_df if args.space == "world" else image_df
    if projection_df is None:
        raise SystemExit(f"No {args.space}-space landmark data available for {camera_csv_path}.")
    projection = compute_camera_projection(
        projection_df,
        args.landmarks,
        CameraProjectionConfig(
            space=args.space,
            visibility_threshold=args.visibility_threshold,
            smooth_window=args.smooth_window,
            rotate_to_body_frame=args.body_frame,
        ),
    )

    frame_mode = _projection_frame_label(args.body_frame)
    default_output = artifact_store.egocentric_diagnostics(label, frame_mode, args.space).components_plot
    output_path = _resolve_cli_path(args.output, paths.root) if args.output is not None else default_output
    plot_projection_components(
        projection,
        args.landmarks,
        output_path,
        title=f"Egocentric components: {label} ({args.space}, {frame_mode})",
        components=args.components,
    )
    print(f"Saved egocentric components plot to {output_path}")


def run_egocentric_video(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()

    camera_csv_path, resolved_video_path, label = _resolve_camera_inputs(
        project_root=paths.root,
        camera_csv=args.camera_csv,
        tag=args.tag,
        camera_side=args.camera_side,
        camera_role_specs=args.camera_role,
    )

    explicit_video_path = _resolve_cli_path(args.video, paths.root) if args.video is not None else resolved_video_path
    video_path, panel_source = _resolve_camera_panel_video(
        camera_csv_path=camera_csv_path,
        explicit_video_path=explicit_video_path,
        project_root=paths.root,
        panel_source=args.camera_panel_source,
    )
    rotation_code, rotation_source = _resolve_video_rotation_code(
        video_path,
        camera_csv_path,
        paths,
        args.video_rotation,
        args.orientation_max_scan,
    )

    landmark_views = load_landmark_views(
        camera_csv_path,
        project_root=paths.root,
        world_csv=_resolve_cli_path(args.world_csv, paths.root) if args.world_csv is not None else None,
        require_world=args.space == "world",
    )
    image_df = landmark_views.image_df
    projection_df = landmark_views.world_df if args.space == "world" else image_df
    if projection_df is None:
        raise SystemExit(f"No {args.space}-space landmark data available for {camera_csv_path}.")
    pose_landmarks_by_frame = prepare_pose_landmarks_by_frame(image_df, visibility_threshold=None)
    projection = compute_camera_projection(
        projection_df,
        args.landmarks,
        CameraProjectionConfig(
            space=args.space,
            visibility_threshold=args.visibility_threshold,
            smooth_window=args.smooth_window,
            rotate_to_body_frame=args.body_frame,
        ),
    )

    frame_mode = _projection_frame_label(args.body_frame)
    default_output = artifact_store.egocentric_diagnostics(label, frame_mode, args.space).overlay_video
    output_path = _resolve_cli_path(args.output, paths.root) if args.output is not None else default_output
    title = f"Egocentric debug: {label}"
    render_projection_overlay_video(
        video_path=video_path,
        result=projection,
        pose_landmarks_by_frame=pose_landmarks_by_frame,
        landmarks=args.landmarks,
        output_path=output_path,
        rotation_code=rotation_code,
        max_frames=args.max_frames,
        title=title,
        projection_frame_label=f"{args.space} | {frame_mode} | {panel_source} | {rotation_source}",
        trace_components=args.components,
    )
    print(f"Saved egocentric overlay video to {output_path}")


__all__ = [
    "add_egocentric_plot_args",
    "add_egocentric_video_args",
    "run_egocentric_plot",
    "run_egocentric_video",
]
