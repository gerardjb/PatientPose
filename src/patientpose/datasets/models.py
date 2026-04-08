from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
