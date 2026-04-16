"""Diagnostics helpers for PatientPose workflows."""

from .egocentric import (
    plot_projection_components,
    prepare_pose_landmarks_by_frame,
    render_projection_overlay_video,
)
from .landmark_overlay import render_landmark_overlay_video
from .landmark_traces import (
    TraceSpec,
    plot_landmark_components,
    plot_metric_panels,
    plot_metric_trace,
    plot_pairwise_landmark_comparison,
)

__all__ = [
    "TraceSpec",
    "plot_landmark_components",
    "plot_metric_panels",
    "plot_metric_trace",
    "plot_pairwise_landmark_comparison",
    "plot_projection_components",
    "render_landmark_overlay_video",
    "prepare_pose_landmarks_by_frame",
    "render_projection_overlay_video",
]
