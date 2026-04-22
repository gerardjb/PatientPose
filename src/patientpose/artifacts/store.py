from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from patientpose.config import ProjectPaths


@dataclass(frozen=True)
class PreprocessVideoArtifacts:
    annotated_video: Path
    plain_video: Path
    landmarks_csv: Path
    pose_world_csv: Path
    frame_summary_csv: Path
    metadata_json: Path


@dataclass(frozen=True)
class QualityVideoArtifacts:
    annotated_video: Path
    plain_video: Path
    landmarks_csv: Path
    pose_world_csv: Path
    position_plot: Path
    quality_plot: Path
    metadata_json: Path


@dataclass(frozen=True)
class PairReportArtifacts:
    output_dir: Path
    plot_dir: Path
    summary_csv: Path
    ratio_plot: Path


@dataclass(frozen=True)
class SideBySideArtifacts:
    output_video: Path


@dataclass(frozen=True)
class TripletVideoArtifacts:
    output_dir: Path
    output_video: Path


@dataclass(frozen=True)
class FourpanelTripletArtifacts:
    output_plot: Path


@dataclass(frozen=True)
class EgocentricDiagnosticsArtifacts:
    output_dir: Path
    components_plot: Path
    overlay_video: Path


@dataclass(frozen=True)
class LandmarkTraceDiagnosticsArtifacts:
    output_dir: Path
    output_plot: Path


@dataclass(frozen=True)
class LandmarkMetricDiagnosticsArtifacts:
    output_dir: Path
    output_plot: Path


@dataclass(frozen=True)
class LandmarkOverlayDiagnosticsArtifacts:
    output_dir: Path
    output_video: Path


@dataclass(frozen=True)
class LandmarkMetricReportArtifacts:
    output_dir: Path
    trace_csv: Path
    summary_csv: Path


@dataclass(frozen=True)
class BodyAngleReportArtifacts:
    output_dir: Path
    trace_csv: Path
    summary_csv: Path


class ArtifactStore:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def ensure_standard_dirs(self) -> None:
        self.paths.output_videos.mkdir(parents=True, exist_ok=True)
        self.paths.output_csvs.mkdir(parents=True, exist_ok=True)
        self.paths.output_plots.mkdir(parents=True, exist_ok=True)
        self.paths.orientation_debug.mkdir(parents=True, exist_ok=True)
        (self.paths.results / "Diagnostics" / "egocentric").mkdir(parents=True, exist_ok=True)
        (self.paths.results / "Diagnostics" / "landmarks").mkdir(parents=True, exist_ok=True)
        (self.paths.results / "Reports" / "landmark_metrics").mkdir(parents=True, exist_ok=True)
        (self.paths.results / "Reports" / "body_angle").mkdir(parents=True, exist_ok=True)

    def preprocess_video(self, video_path: Path) -> PreprocessVideoArtifacts:
        stem = video_path.stem
        return PreprocessVideoArtifacts(
            annotated_video=self.paths.output_videos / f"deidentified_{stem}.avi",
            plain_video=self.paths.output_videos / f"deidentified_no_keypoints_{stem}.avi",
            landmarks_csv=self.paths.output_csvs / f"landmarks_{stem}.csv",
            pose_world_csv=self.paths.output_csvs / f"pose_world_{stem}.csv",
            frame_summary_csv=self.paths.output_csvs / f"landmarks_summary_{stem}.csv",
            metadata_json=self.paths.output_csvs / f"landmarks_metadata_{stem}.json",
        )

    def preprocess_quality_video(self, video_path: Path) -> QualityVideoArtifacts:
        stem = video_path.stem
        return QualityVideoArtifacts(
            annotated_video=self.paths.output_videos / f"quality_vis_{stem}.avi",
            plain_video=self.paths.output_videos / f"quality_vis_no_keypoints_{stem}.avi",
            landmarks_csv=self.paths.output_csvs / f"landmarks_{stem}.csv",
            pose_world_csv=self.paths.output_csvs / f"pose_world_{stem}.csv",
            position_plot=self.paths.output_plots / f"fingertip_position_{stem}.png",
            quality_plot=self.paths.output_plots / f"fingertip_quality_{stem}.png",
            metadata_json=self.paths.output_csvs / f"landmarks_metadata_{stem}.json",
        )

    def pair_report(self, camera_space: str) -> PairReportArtifacts:
        output_dir = self.paths.results / "mocopi_reliability"
        return PairReportArtifacts(
            output_dir=output_dir,
            plot_dir=output_dir / "plots" / camera_space,
            summary_csv=output_dir / f"nd_delta_summary_{camera_space}.csv",
            ratio_plot=output_dir / f"nd_ratio_summary_{camera_space}.pdf",
        )

    def side_by_side(self) -> SideBySideArtifacts:
        return SideBySideArtifacts(
            output_video=self.paths.output_videos / "mocopi_vs_camera.avi",
        )

    def triplet_video(self, tag: str) -> TripletVideoArtifacts:
        output_dir = self.paths.output_videos / "triplets"
        return TripletVideoArtifacts(
            output_dir=output_dir,
            output_video=output_dir / f"triplet_{tag}.avi",
        )

    def fourpanel_triplet(
        self,
        tag: str,
        *,
        camera_space: str,
        component: str,
        offset_label: str,
        visibility_threshold: float,
    ) -> FourpanelTripletArtifacts:
        return FourpanelTripletArtifacts(
            output_plot=self.paths.output_plots
            / f"fourpanel_{tag}_{camera_space}_d{component}_{offset_label}_vis_{visibility_threshold:.2f}.pdf",
        )

    def egocentric_diagnostics(self, stem: str, frame_mode: str, space: str) -> EgocentricDiagnosticsArtifacts:
        output_dir = self.paths.results / "Diagnostics" / "egocentric"
        return EgocentricDiagnosticsArtifacts(
            output_dir=output_dir,
            components_plot=output_dir / f"{stem}_{space}_{frame_mode}_components.pdf",
            overlay_video=output_dir / f"{stem}_{space}_{frame_mode}_overlay.avi",
        )

    def landmark_trace_diagnostics(
        self,
        stem: str,
        *,
        source: str,
        space: str,
        components: tuple[str, ...],
    ) -> LandmarkTraceDiagnosticsArtifacts:
        output_dir = self.paths.results / "Diagnostics" / "landmarks"
        component_label = "_".join(components)
        return LandmarkTraceDiagnosticsArtifacts(
            output_dir=output_dir,
            output_plot=output_dir / f"{stem}_{source}_{space}_{component_label}_traces.png",
        )

    def landmark_metric_diagnostics(
        self,
        stem: str,
        *,
        source: str,
        space: str,
        metric: str,
    ) -> LandmarkMetricDiagnosticsArtifacts:
        output_dir = self.paths.results / "Diagnostics" / "landmarks"
        return LandmarkMetricDiagnosticsArtifacts(
            output_dir=output_dir,
            output_plot=output_dir / f"{stem}_{source}_{space}_{metric}.png",
        )

    def landmark_overlay_diagnostics(
        self,
        stem: str,
        *,
        source: str,
        space: str,
        metric: str,
    ) -> LandmarkOverlayDiagnosticsArtifacts:
        output_dir = self.paths.results / "Diagnostics" / "landmarks"
        return LandmarkOverlayDiagnosticsArtifacts(
            output_dir=output_dir,
            output_video=output_dir / f"{stem}_{source}_{space}_{metric}_overlay.avi",
        )

    def landmark_metric_report(
        self,
        stem: str,
        *,
        source: str,
        space: str,
        metric: str,
    ) -> LandmarkMetricReportArtifacts:
        output_dir = self.paths.results / "Reports" / "landmark_metrics"
        base = f"{stem}_{source}_{space}_{metric}"
        return LandmarkMetricReportArtifacts(
            output_dir=output_dir,
            trace_csv=output_dir / f"{base}_trace.csv",
            summary_csv=output_dir / f"{base}_summary.csv",
        )

    def landmark_metric_batch_report(
        self,
        *,
        source: str,
        space: str,
        metric: str,
    ) -> LandmarkMetricReportArtifacts:
        output_dir = self.paths.results / "Reports" / "landmark_metrics"
        base = f"batch_{source}_{space}_{metric}"
        return LandmarkMetricReportArtifacts(
            output_dir=output_dir,
            trace_csv=output_dir / f"{base}_traces.csv",
            summary_csv=output_dir / f"{base}_summary.csv",
        )

    def body_angle_report(
        self,
        stem: str,
        *,
        space: str,
    ) -> BodyAngleReportArtifacts:
        output_dir = self.paths.results / "Reports" / "body_angle"
        base = f"{stem}_{space}_body-angle"
        return BodyAngleReportArtifacts(
            output_dir=output_dir,
            trace_csv=output_dir / f"{base}_trace.csv",
            summary_csv=output_dir / f"{base}_summary.csv",
        )
