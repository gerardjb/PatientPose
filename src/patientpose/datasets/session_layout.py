from __future__ import annotations

import json
from pathlib import Path

from .models import CameraRecording, CaptureSession, TrialPair


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".m4v", ".mkv"}


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
        for camera_dir in sorted(
            p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("phone_")
        ):
            camera_id = camera_dir.name[len("phone_") :]
            for video_path in sorted(
                p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
            ):
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
                metadata_events=_load_session_events(
                    session_log_path if session_log_path.exists() else None
                ),
            )
        )

    return sessions


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
