"""
Lightweight utilities for working with Mocopi BVH motion capture data.

This module focuses on:
    - Parsing Mocopi-style BVH files into a structured time-series representation.
    - Exporting joint trajectories in a landmarks-like tabular format.
    - Computing simple features for synchronizing with camera-based keypoints.
"""

from .bvh_io import MocopiSequence, load_bvh, mocopi_to_frame_table
from .features import (
    MocopiFeatureConfig,
    CameraFeatureConfig,
    extract_mocopi_feature,
    extract_camera_feature,
)
from .sync import estimate_time_offset

__all__ = [
    "MocopiSequence",
    "load_bvh",
    "mocopi_to_frame_table",
    "MocopiFeatureConfig",
    "CameraFeatureConfig",
    "extract_mocopi_feature",
    "extract_camera_feature",
    "estimate_time_offset",
]

