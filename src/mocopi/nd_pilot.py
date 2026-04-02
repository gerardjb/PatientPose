from __future__ import annotations

"""
Compatibility helpers for discovering legacy ND-pilot pairs and newer session folders.

The original Mocopi scripts assumed a single dataset layout:
    sample_data/ND_pilot/A_<tag>_*.mp4
    sample_data/ND_pilot/ND_<tag>_*.mp4
    sample_data/ND_pilot/Re_ Mocopi/MCPM_*_<tag>.bvh

Newer captures are stored by session, for example:
    sample_data/<session_id>/session_log.jsonl
    sample_data/<session_id>/<session_id>_mocopi.bin
    sample_data/<session_id>/phone_<camera_id>/VID_*.mp4

This module exposes a unified discovery surface while preserving the original
`TrialPair` API expected by the existing scripts.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".m4v", ".mkv"}
ROLE_ALIASES = {
    "A": "A",
    "UNFILTERED": "A",
    "ND": "ND",
}


@dataclass(frozen=True)
class CameraRecording:
    camera_id: str
    video_path: Path
    csv_path: Path
    role: str | None = None


@dataclass
class CaptureSession:
    session_id: str
    session_dir: Path
    motion_source: Path
    session_log_path: Path | None
    recordings: list[CameraRecording]
    metadata_events: list[dict] = field(default_factory=list)

    @property
    def available_camera_ids(self) -> list[str]:
        return sorted({recording.camera_id for recording in self.recordings})


@dataclass
class TrialPair:
    tag: str
    unfiltered_video: Path
    nd_video: Path
    motion_source: Path
    mode: str = "legacy"
    session_id: str | None = None

    @property
    def bvh(self) -> Path:
        # Backward-compatible alias kept for scripts that still refer to pair.bvh.
        return self.motion_source


def _normalize_role(role: str) -> str:
    normalized = ROLE_ALIASES.get(role.strip().upper())
    if normalized is None:
        allowed = ", ".join(sorted(ROLE_ALIASES))
        raise ValueError(f"Unknown camera role {role!r}; expected one of: {allowed}")
    return normalized


def parse_camera_role_specs(specs: Sequence[str] | None) -> dict[str, str]:
    """
    Parse CLI-friendly role mappings like:
        192.168.50.162=A
        phone_192.168.50.171=ND
    """
    mapping: dict[str, str] = {}
    if not specs:
        return mapping

    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid camera-role mapping {spec!r}; expected CAMERA_ID=ROLE")
        camera_id, role = spec.split("=", 1)
        camera_id = camera_id.strip()
        if not camera_id:
            raise ValueError(f"Invalid camera-role mapping {spec!r}; camera id is empty")
        mapping[camera_id] = _normalize_role(role)
    return mapping


def infer_camera_csv(video_path: Path) -> Path:
    return Path("results") / "OutputCSVs" / f"landmarks_{video_path.stem}.csv"


def _load_session_events(session_log_path: Path | None) -> list[dict]:
    if session_log_path is None or not session_log_path.is_file():
        return []

    events: list[dict] = []
    for line in session_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def discover_sessions(base: Path) -> list[CaptureSession]:
    base = base.resolve()
    sample_dir = base / "sample_data"
    if not sample_dir.is_dir():
        return []

    sessions: list[CaptureSession] = []
    for session_dir in sorted(sample_dir.iterdir()):
        if not session_dir.is_dir() or session_dir.name == "ND_pilot":
            continue

        motion_candidates = sorted(session_dir.glob("*_mocopi.bin"))
        if not motion_candidates:
            continue

        session_log_path = session_dir / "session_log.jsonl"
        recordings: list[CameraRecording] = []
        for camera_dir in sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("phone_")):
            camera_id = camera_dir.name[len("phone_") :]
            for video_path in sorted(p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES):
                recordings.append(
                    CameraRecording(
                        camera_id=camera_id,
                        video_path=video_path,
                        csv_path=infer_camera_csv(video_path),
                    )
                )

        sessions.append(
            CaptureSession(
                session_id=session_dir.name,
                session_dir=session_dir,
                motion_source=motion_candidates[0],
                session_log_path=session_log_path if session_log_path.exists() else None,
                recordings=recordings,
                metadata_events=_load_session_events(session_log_path if session_log_path.exists() else None),
            )
        )

    return sessions


def _discover_legacy_pairs(base: Path) -> list[TrialPair]:
    sample_dir = base / "sample_data"
    nd_dir = sample_dir / "ND_pilot"
    bvh_dir = nd_dir / "Re_ Mocopi"
    if not nd_dir.is_dir() or not bvh_dir.is_dir():
        return []

    bvh_map: dict[str, Path] = {}
    for bvh in bvh_dir.glob("MCPM_*_*.bvh"):
        tag = bvh.stem.split("_")[-1]
        bvh_map[tag] = bvh

    pairs: list[TrialPair] = []
    for nd_video in sorted(nd_dir.glob("ND_*.mp4")):
        parts = nd_video.stem.split("_")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag not in bvh_map:
            continue
        candidates = sorted(nd_dir.glob(f"A_{tag}_*.mp4"))
        if not candidates:
            continue
        pairs.append(
            TrialPair(
                tag=tag,
                unfiltered_video=candidates[0],
                nd_video=nd_video,
                motion_source=bvh_map[tag],
                mode="legacy",
            )
        )
    return pairs


def _recording_matches_camera(recording: CameraRecording, mapping_key: str) -> bool:
    return mapping_key in {
        recording.camera_id,
        f"phone_{recording.camera_id}",
        recording.video_path.parent.name,
    }


def resolve_session_pair(
    session: CaptureSession,
    camera_roles: dict[str, str],
) -> TrialPair:
    if not camera_roles:
        available = ", ".join(session.available_camera_ids) or "<none>"
        raise ValueError(
            f"Session {session.session_id!r} requires explicit camera-role mappings; "
            f"available camera ids: {available}"
        )

    selected: dict[str, CameraRecording] = {}
    for recording in session.recordings:
        for mapping_key, role in camera_roles.items():
            if _recording_matches_camera(recording, mapping_key):
                if role in selected:
                    raise ValueError(
                        f"Session {session.session_id!r} mapped multiple recordings to role {role!r}"
                    )
                selected[role] = CameraRecording(
                    camera_id=recording.camera_id,
                    video_path=recording.video_path,
                    csv_path=recording.csv_path,
                    role=role,
                )
                break

    missing_roles = [role for role in ("A", "ND") if role not in selected]
    if missing_roles:
        available = ", ".join(session.available_camera_ids) or "<none>"
        raise ValueError(
            f"Session {session.session_id!r} is missing role(s) {', '.join(missing_roles)}; "
            f"available camera ids: {available}"
        )

    return TrialPair(
        tag=session.session_id,
        unfiltered_video=selected["A"].video_path,
        nd_video=selected["ND"].video_path,
        motion_source=session.motion_source,
        mode="session",
        session_id=session.session_id,
    )


def discover_pairs(
    base: Path,
    camera_roles: dict[str, str] | None = None,
    include_legacy: bool = True,
) -> list[TrialPair]:
    base = base.resolve()
    pairs: list[TrialPair] = []
    if include_legacy:
        pairs.extend(_discover_legacy_pairs(base))

    if camera_roles:
        for session in discover_sessions(base):
            pairs.append(resolve_session_pair(session, camera_roles))

    return pairs


def pair_for_tag(
    base: Path,
    tag: str,
    camera_roles: dict[str, str] | None = None,
    include_legacy: bool = True,
) -> Optional[TrialPair]:
    for pair in discover_pairs(base, camera_roles=camera_roles, include_legacy=include_legacy):
        if pair.tag == tag:
            return pair
    return None


def session_lookup(base: Path) -> dict[str, CaptureSession]:
    return {session.session_id: session for session in discover_sessions(base)}


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
