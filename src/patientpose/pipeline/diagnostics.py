from __future__ import annotations

import argparse
from pathlib import Path

from mocopi import CameraProjectionConfig, compute_camera_projection
import numpy as np
import pandas as pd
from patientpose.artifacts import ArtifactStore
from patientpose.config import resolve_project_paths
from patientpose.datasets import discover_pairs, discover_sessions, infer_camera_csv, parse_camera_role_specs
from patientpose.diagnostics import (
    TraceSpec,
    plot_landmark_components,
    plot_metric_trace,
    plot_projection_components,
    prepare_pose_landmarks_by_frame,
    render_landmark_overlay_video,
    render_projection_overlay_video,
)
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


def add_landmark_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
        "--source",
        choices=("hand", "pose"),
        default="hand",
        help="Landmark source to extract from the CSV.",
    )
    parser.add_argument(
        "--space",
        choices=("image", "world"),
        default="image",
        help="Which landmark representation to analyze.",
    )
    parser.add_argument(
        "--world_csv",
        type=Path,
        default=None,
        help="Optional explicit path to the pose-world CSV. Defaults to the paired preprocess artifact.",
    )
    parser.add_argument(
        "--handedness",
        type=str,
        default=None,
        help="Optional handedness filter, e.g. Right or Left.",
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Optional instance id filter for multi-instance landmark sources.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=None,
        help="Landmark names or ids to extract.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Components to use. Choose from x y z.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered rolling smoothing window in frames.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="Optional minimum quality_score. Samples below it are set to NaN before plotting/metrics.",
    )
    return parser


def add_landmark_traces_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_landmark_common_args(parser)
    parser.set_defaults(components=["x", "y"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path for the landmark trace plot.",
    )
    return parser


def add_landmark_metric_plot_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_landmark_common_args(parser)
    parser.set_defaults(components=["x", "y"])
    parser.add_argument(
        "--metric",
        choices=(
            "distance",
            "delta",
            "centroid-distance",
            "angle",
            "thumb-index-distance",
            "pinch-velocity",
        ),
        default="distance",
        help="Derived metric to compute from the selected landmarks.",
    )
    parser.add_argument(
        "--delta-component",
        choices=("x", "y", "z"),
        default=None,
        help="Component to use for the delta metric. Defaults to the first component in --components.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path for the metric plot.",
    )
    return parser


def add_landmark_overlay_video_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_landmark_common_args(parser)
    parser.set_defaults(components=["x", "y"])
    parser.add_argument(
        "--metric",
        choices=(
            "distance",
            "delta",
            "centroid-distance",
            "angle",
            "thumb-index-distance",
            "pinch-velocity",
        ),
        default="distance",
        help="Derived metric to render in the rolling trace panel.",
    )
    parser.add_argument(
        "--delta-component",
        choices=("x", "y", "z"),
        default=None,
        help="Component to use for the delta metric. Defaults to the first component in --components.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Explicit path to the camera-panel video. Overrides --camera-panel-source.",
    )
    parser.add_argument(
        "--camera-panel-source",
        choices=("auto", "deidentified", "deidentified-no-keypoints", "raw"),
        default="auto",
        help="What to show under the landmark overlays when --video is not given.",
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
        "--trace-window-seconds",
        type=float,
        default=3.0,
        help="Rolling history window for the bottom trace strip.",
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


def _resolve_landmark_df(
    *,
    paths,
    camera_csv_path: Path,
    space: str,
    world_csv: Path | None,
) -> tuple[pd.DataFrame, str]:
    landmark_views = load_landmark_views(
        camera_csv_path,
        project_root=paths.root,
        world_csv=_resolve_cli_path(world_csv, paths.root) if world_csv is not None else None,
        require_world=space == "world",
    )
    df = landmark_views.world_df if space == "world" else landmark_views.image_df
    if df is None:
        raise SystemExit(f"No {space}-space landmark data available for {camera_csv_path}.")
    return df, landmark_stem_from_image_csv(camera_csv_path)


def _normalize_components(components: list[str] | None, *, default: list[str]) -> list[str]:
    normalized: list[str] = []
    for component in components or default:
        value = str(component).lower()
        if value not in {"x", "y", "z"}:
            raise SystemExit(f"Unknown component '{component}'. Choose from x, y, z.")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _apply_quality_threshold_and_smoothing(
    series_df: pd.DataFrame,
    *,
    columns: list[str],
    smooth_window: int,
    quality_threshold: float | None,
) -> pd.DataFrame:
    out = series_df.copy()
    if quality_threshold is not None and "quality_score" in out.columns:
        low_quality = out["quality_score"].notna() & (out["quality_score"] < quality_threshold)
        out.loc[low_quality, columns] = np.nan
    if smooth_window > 1:
        for column in columns:
            if column in out.columns:
                out[column] = out[column].rolling(window=smooth_window, center=True, min_periods=1).mean()
    return out


def _build_landmark_series_map(
    df: pd.DataFrame,
    *,
    landmarks: list[str] | None,
    source: str,
    handedness: str | None,
    instance_id: int | None,
    components: list[str],
    quality_threshold: float | None,
    smooth_window: int,
) -> dict[str, pd.DataFrame]:
    if not landmarks:
        raise SystemExit("--landmarks is required for landmark trace and metric diagnostics.")
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


def _resolve_metric_landmarks(metric: str, landmarks: list[str] | None) -> list[str]:
    if landmarks:
        return landmarks
    if metric in {"thumb-index-distance", "pinch-velocity"}:
        return ["THUMB_TIP", "INDEX_FINGER_TIP"]
    raise SystemExit(f"--landmarks is required for metric '{metric}'.")


def _compute_metric_trace(
    *,
    metric: str,
    series_map: dict[str, pd.DataFrame],
    components: list[str],
    delta_component: str | None,
) -> tuple[pd.DataFrame, str]:
    names = list(series_map.keys())
    if metric == "distance":
        if len(names) != 2:
            raise SystemExit("Metric 'distance' requires exactly 2 landmarks.")
        metric_df = pairwise_distance(series_map[names[0]], series_map[names[1]], components=components)
        return metric_df, f"{names[0]} vs {names[1]} distance"
    if metric == "delta":
        if len(names) != 2:
            raise SystemExit("Metric 'delta' requires exactly 2 landmarks.")
        component = delta_component or components[0]
        metric_df = pairwise_delta(series_map[names[0]], series_map[names[1]], component=component)
        return metric_df, f"{names[0]} - {names[1]} ({component})"
    if metric == "centroid-distance":
        if len(names) < 2:
            raise SystemExit("Metric 'centroid-distance' requires at least 2 landmarks.")
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
            raise SystemExit("Metric 'angle' requires exactly 4 landmarks.")
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
            raise SystemExit("Metric 'thumb-index-distance' requires exactly 2 landmarks.")
        metric_df = thumb_index_distance(series_map[names[0]], series_map[names[1]], components=tuple(components))
        return metric_df, "thumb-index distance"
    if metric == "pinch-velocity":
        if len(names) != 2:
            raise SystemExit("Metric 'pinch-velocity' requires exactly 2 landmarks.")
        metric_df = pinch_velocity(series_map[names[0]], series_map[names[1]], components=tuple(components))
        return metric_df, "pinch velocity"
    raise SystemExit(f"Unsupported metric '{metric}'.")


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


def run_landmark_traces(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()
    camera_csv_path, _, _ = _resolve_camera_inputs(
        project_root=paths.root,
        camera_csv=args.camera_csv,
        tag=args.tag,
        camera_side=args.camera_side,
        camera_role_specs=args.camera_role,
    )
    df, stem = _resolve_landmark_df(
        paths=paths,
        camera_csv_path=camera_csv_path,
        space=args.space,
        world_csv=args.world_csv,
    )
    components = _normalize_components(args.components, default=["x", "y"])
    series_map = _build_landmark_series_map(
        df,
        landmarks=args.landmarks,
        source=args.source,
        handedness=args.handedness,
        instance_id=args.instance_id,
        components=components,
        quality_threshold=args.quality_threshold,
        smooth_window=args.smooth_window,
    )
    output_path = (
        _resolve_cli_path(args.output, paths.root)
        if args.output is not None
        else artifact_store.landmark_trace_diagnostics(
            stem,
            source=args.source,
            space=args.space,
            components=tuple(components),
        ).output_plot
    )
    plot_landmark_components(
        series_map,
        output_path,
        title=f"Landmark traces: {stem} ({args.source}, {args.space})",
        components=components,
    )
    print(f"Saved landmark trace plot to {output_path}")


def run_landmark_metric_plot(args: argparse.Namespace) -> None:
    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()
    camera_csv_path, _, _ = _resolve_camera_inputs(
        project_root=paths.root,
        camera_csv=args.camera_csv,
        tag=args.tag,
        camera_side=args.camera_side,
        camera_role_specs=args.camera_role,
    )
    df, stem = _resolve_landmark_df(
        paths=paths,
        camera_csv_path=camera_csv_path,
        space=args.space,
        world_csv=args.world_csv,
    )
    components = _normalize_components(args.components, default=["x", "y"])
    landmarks = _resolve_metric_landmarks(args.metric, args.landmarks)
    series_map = _build_landmark_series_map(
        df,
        landmarks=landmarks,
        source=args.source,
        handedness=args.handedness,
        instance_id=args.instance_id,
        components=components,
        quality_threshold=args.quality_threshold,
        smooth_window=args.smooth_window,
    )
    metric_df, label = _compute_metric_trace(
        metric=args.metric,
        series_map=series_map,
        components=components,
        delta_component=args.delta_component,
    )
    output_path = (
        _resolve_cli_path(args.output, paths.root)
        if args.output is not None
        else artifact_store.landmark_metric_diagnostics(
            stem,
            source=args.source,
            space=args.space,
            metric=args.metric,
        ).output_plot
    )
    plot_metric_trace(
        [TraceSpec(label=label, df=metric_df)],
        output_path,
        title=f"Landmark metric: {stem} ({args.metric})",
        y_label=metric_df["metric"].iloc[0] if not metric_df.empty else args.metric,
    )
    print(f"Saved landmark metric plot to {output_path}")


def run_landmark_overlay_video(args: argparse.Namespace) -> None:
    if args.source == "hand" and args.space == "world":
        raise SystemExit("Hand landmarks currently support overlay video only in image space.")

    paths = resolve_project_paths(args.project_root)
    artifact_store = ArtifactStore(paths)
    artifact_store.ensure_standard_dirs()
    camera_csv_path, resolved_video_path, _ = _resolve_camera_inputs(
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
    metric_df_source = landmark_views.world_df if args.space == "world" else landmark_views.image_df
    if metric_df_source is None:
        raise SystemExit(f"No {args.space}-space landmark data available for {camera_csv_path}.")
    image_df = landmark_views.image_df
    stem = landmark_stem_from_image_csv(camera_csv_path)
    components = _normalize_components(args.components, default=["x", "y"])
    landmarks = _resolve_metric_landmarks(args.metric, args.landmarks)

    computation_series_map = _build_landmark_series_map(
        metric_df_source,
        landmarks=landmarks,
        source=args.source,
        handedness=args.handedness,
        instance_id=args.instance_id,
        components=components,
        quality_threshold=args.quality_threshold,
        smooth_window=args.smooth_window,
    )
    overlay_series_map = _build_landmark_series_map(
        image_df,
        landmarks=landmarks,
        source=args.source,
        handedness=args.handedness,
        instance_id=args.instance_id,
        components=["x", "y"],
        quality_threshold=args.quality_threshold,
        smooth_window=1,
    )
    metric_df, label = _compute_metric_trace(
        metric=args.metric,
        series_map=computation_series_map,
        components=components,
        delta_component=args.delta_component,
    )

    output_path = (
        _resolve_cli_path(args.output, paths.root)
        if args.output is not None
        else artifact_store.landmark_overlay_diagnostics(
            stem,
            source=args.source,
            space=args.space,
            metric=args.metric,
        ).output_video
    )
    pose_landmarks_by_frame = (
        prepare_pose_landmarks_by_frame(image_df, visibility_threshold=None)
        if args.source == "pose"
        else {}
    )
    render_landmark_overlay_video(
        video_path=video_path,
        landmark_series_map=overlay_series_map,
        metric_df=metric_df,
        metric_label=label,
        output_path=output_path,
        rotation_code=rotation_code,
        max_frames=args.max_frames,
        title=f"Landmark overlay: {stem}",
        overlay_label=f"{args.source} | {args.space} | {panel_source} | {rotation_source}",
        trace_window_seconds=args.trace_window_seconds,
        pose_landmarks_by_frame=pose_landmarks_by_frame,
    )
    print(f"Saved landmark overlay video to {output_path}")


__all__ = [
    "add_egocentric_plot_args",
    "add_egocentric_video_args",
    "add_landmark_overlay_video_args",
    "add_landmark_metric_plot_args",
    "add_landmark_traces_args",
    "run_egocentric_plot",
    "run_egocentric_video",
    "run_landmark_overlay_video",
    "run_landmark_metric_plot",
    "run_landmark_traces",
]
