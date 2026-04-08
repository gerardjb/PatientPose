"""Dataset discovery and normalization for PatientPose workflows."""

from .discovery import discover_pairs, pair_for_tag, session_lookup
from .models import CameraRecording, CaptureSession, TrialPair
from .roles import parse_camera_role_specs
from .session_layout import discover_sessions, infer_camera_csv, resolve_session_pair

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
    "session_lookup",
]
