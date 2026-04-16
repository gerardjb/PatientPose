"""Artifact path helpers for PatientPose workflows."""

from .store import (
    ArtifactStore,
    EgocentricDiagnosticsArtifacts,
    FourpanelTripletArtifacts,
    LandmarkMetricDiagnosticsArtifacts,
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
    "LandmarkTraceDiagnosticsArtifacts",
    "PairReportArtifacts",
    "PreprocessVideoArtifacts",
    "QualityVideoArtifacts",
    "SideBySideArtifacts",
    "TripletVideoArtifacts",
]
