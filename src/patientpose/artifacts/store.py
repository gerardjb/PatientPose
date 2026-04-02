from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from patientpose.config import ProjectPaths


@dataclass(frozen=True)
class PreprocessVideoArtifacts:
    annotated_video: Path
    plain_video: Path
    landmarks_csv: Path
    frame_summary_csv: Path


class ArtifactStore:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def ensure_standard_dirs(self) -> None:
        self.paths.output_videos.mkdir(parents=True, exist_ok=True)
        self.paths.output_csvs.mkdir(parents=True, exist_ok=True)
        self.paths.output_plots.mkdir(parents=True, exist_ok=True)
        self.paths.orientation_debug.mkdir(parents=True, exist_ok=True)

    def preprocess_video(self, video_path: Path) -> PreprocessVideoArtifacts:
        stem = video_path.stem
        return PreprocessVideoArtifacts(
            annotated_video=self.paths.output_videos / f"deidentified_{stem}.avi",
            plain_video=self.paths.output_videos / f"deidentified_no_keypoints_{stem}.avi",
            landmarks_csv=self.paths.output_csvs / f"landmarks_{stem}.csv",
            frame_summary_csv=self.paths.output_csvs / f"landmarks_summary_{stem}.csv",
        )
