"""
Lightweight utilities for working with Mocopi motion capture data.

This module focuses on:
    - Parsing Mocopi BVH files and session BIN recordings into a structured time-series representation.
    - Exporting joint trajectories in a landmarks-like tabular format.
    - Computing simple features for synchronizing with camera-based keypoints.
"""

from .bvh_io import MocopiSequence, load_bvh, mocopi_to_frame_table
from .recording_io import load_mocopi_bin, load_mocopi_recording, resolve_mocopi_source
from .features import (
    MocopiFeatureConfig,
    CameraFeatureConfig,
    extract_mocopi_feature,
    extract_camera_feature,
)
from .sync import estimate_time_offset, clean_feature_samples, estimate_camera_to_mocopi_offset
from .nd_pilot import (
    CameraRecording,
    CaptureSession,
    TrialPair,
    discover_pairs,
    discover_sessions,
    infer_camera_csv,
    pair_for_tag,
    parse_camera_role_specs,
    resolve_session_pair,
)
from .reliability import (
    SCALE_REF_JOINTS,
    compute_body_scale_series,
    export_reliability_errors,
    ensure_reliability_csv,
    nd_factor_from_stem,
    best_joint_from_reliability,
    align_visibility_series,
    align_pose_counts,
    joint_medians,
    nd_error_summary,
    get_aligned_traces,
)
from .camera_metrics import count_visible_landmarks, visibility_percent
from .visualization import (
    CAMERA_EDGES,
    MOCOPI_EDGES,
    MOCOPI_JOINTS,
    prepare_camera_landmarks,
    draw_camera_skeleton,
    prepare_mocopi_positions,
    draw_mocopi_skeleton,
)
from .plots import select_overlap_window, plot_egocentric_compare, plot_feet_centered

__all__ = [
    "MocopiSequence",
    "load_bvh",
    "load_mocopi_bin",
    "load_mocopi_recording",
    "resolve_mocopi_source",
    "mocopi_to_frame_table",
    "MocopiFeatureConfig",
    "CameraFeatureConfig",
    "extract_mocopi_feature",
    "extract_camera_feature",
    "estimate_time_offset",
    "clean_feature_samples",
    "estimate_camera_to_mocopi_offset",
    "CameraRecording",
    "CaptureSession",
    "TrialPair",
    "discover_pairs",
    "discover_sessions",
    "infer_camera_csv",
    "pair_for_tag",
    "parse_camera_role_specs",
    "resolve_session_pair",
    "SCALE_REF_JOINTS",
    "compute_body_scale_series",
    "export_reliability_errors",
    "ensure_reliability_csv",
    "nd_factor_from_stem",
    "best_joint_from_reliability",
    "align_visibility_series",
    "align_pose_counts",
    "joint_medians",
    "nd_error_summary",
    "get_aligned_traces",
    "count_visible_landmarks",
    "visibility_percent",
    "CAMERA_EDGES",
    "MOCOPI_EDGES",
    "MOCOPI_JOINTS",
    "prepare_camera_landmarks",
    "draw_camera_skeleton",
    "prepare_mocopi_positions",
    "draw_mocopi_skeleton",
    "select_overlap_window",
    "plot_egocentric_compare",
    "plot_feet_centered",
]
