"""Artifact path helpers for PatientPose workflows."""

from .store import (
    ArtifactStore,
    BodyAngleReportArtifacts,
    EgocentricDiagnosticsArtifacts,
    FourpanelTripletArtifacts,
    LandmarkMetricDiagnosticsArtifacts,
    LandmarkMetricReportArtifacts,
    LandmarkOverlayDiagnosticsArtifacts,
    LandmarkTraceDiagnosticsArtifacts,
    PairReportArtifacts,
    PreprocessVideoArtifacts,
    QualityVideoArtifacts,
    SideBySideArtifacts,
    TripletVideoArtifacts,
)

__all__ = [
    "ArtifactStore",
    "BodyAngleReportArtifacts",
    "EgocentricDiagnosticsArtifacts",
    "FourpanelTripletArtifacts",
    "LandmarkMetricDiagnosticsArtifacts",
    "LandmarkMetricReportArtifacts",
    "LandmarkOverlayDiagnosticsArtifacts",
    "LandmarkTraceDiagnosticsArtifacts",
    "PairReportArtifacts",
    "PreprocessVideoArtifacts",
    "QualityVideoArtifacts",
    "SideBySideArtifacts",
    "TripletVideoArtifacts",
]
