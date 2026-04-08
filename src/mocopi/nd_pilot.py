from __future__ import annotations

"""
Compatibility wrapper for dataset discovery helpers.

The implementation now lives under ``patientpose.datasets`` so that repo layout,
session discovery, and camera-role resolution sit in the workflow layer instead
of the mocopi domain package. This module re-exports the existing names to avoid
breaking downstream imports during the migration.
"""

from patientpose.datasets.discovery import discover_pairs, pair_for_tag, session_lookup
from patientpose.datasets.models import CameraRecording, CaptureSession, TrialPair
from patientpose.datasets.roles import parse_camera_role_specs
from patientpose.datasets.session_layout import discover_sessions, infer_camera_csv, resolve_session_pair

__all__ = [
    "CameraRecording",
    "CaptureSession",
    "TrialPair",
    "discover_sessions",
    "discover_pairs",
    "pair_for_tag",
    "parse_camera_role_specs",
    "resolve_session_pair",
    "infer_camera_csv",
    "session_lookup",
]
