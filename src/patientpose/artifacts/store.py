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


@dataclass(frozen=True)
class QualityVideoArtifacts:
    annotated_video: Path
    plain_video: Path
    landmarks_csv: Path
    position_plot: Path
    quality_plot: Path


@dataclass(frozen=True)
class PairReportArtifacts:
    output_dir: Path
    plot_dir: Path
    summary_csv: Path
    ratio_plot: Path


@dataclass(frozen=True)
class SideBySideArtifacts:
    output_video: Path


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

    def preprocess_quality_video(self, video_path: Path) -> QualityVideoArtifacts:
        stem = video_path.stem
        return QualityVideoArtifacts(
            annotated_video=self.paths.output_videos / f"quality_vis_{stem}.avi",
            plain_video=self.paths.output_videos / f"quality_vis_no_keypoints_{stem}.avi",
            landmarks_csv=self.paths.output_csvs / f"landmarks_{stem}.csv",
            position_plot=self.paths.output_plots / f"fingertip_position_{stem}.png",
            quality_plot=self.paths.output_plots / f"fingertip_quality_{stem}.png",
        )

    def pair_report(self) -> PairReportArtifacts:
        output_dir = self.paths.results / "mocopi_reliability"
        return PairReportArtifacts(
            output_dir=output_dir,
            plot_dir=output_dir / "plots",
            summary_csv=output_dir / "nd_delta_summary.csv",
            ratio_plot=output_dir / "nd_ratio_summary.pdf",
        )

    def side_by_side(self) -> SideBySideArtifacts:
        return SideBySideArtifacts(
            output_video=self.paths.output_videos / "mocopi_vs_camera.avi",
        )
