"""Artifact path helpers for PatientPose workflows."""

from .store import (
    ArtifactStore,
    EgocentricDiagnosticsArtifacts,
    FourpanelTripletArtifacts,
    LandmarkMetricDiagnosticsArtifacts,
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
    "EgocentricDiagnosticsArtifacts",
    "FourpanelTripletArtifacts",
    "LandmarkMetricDiagnosticsArtifacts",
    "LandmarkOverlayDiagnosticsArtifacts",
    "LandmarkTraceDiagnosticsArtifacts",
    "PairReportArtifacts",
    "PreprocessVideoArtifacts",
    "QualityVideoArtifacts",
    "SideBySideArtifacts",
    "TripletVideoArtifacts",
]
