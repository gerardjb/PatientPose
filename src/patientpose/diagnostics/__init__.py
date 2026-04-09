"""Diagnostics helpers for PatientPose workflows."""

from .egocentric import (
    plot_projection_components,
    prepare_pose_landmarks_by_frame,
    render_projection_overlay_video,
)

__all__ = [
    "plot_projection_components",
    "prepare_pose_landmarks_by_frame",
    "render_projection_overlay_video",
]
