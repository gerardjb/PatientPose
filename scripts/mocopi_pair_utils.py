from __future__ import annotations

"""Compatibility shim for the session-aware mocopi discovery helpers."""

from mocopi.nd_pilot import (
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

__all__ = [
    "CameraRecording",
    "CaptureSession",
    "TrialPair",
    "discover_pairs",
    "discover_sessions",
    "infer_camera_csv",
    "pair_for_tag",
    "parse_camera_role_specs",
    "resolve_session_pair",
]
