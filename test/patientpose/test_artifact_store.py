from pathlib import Path

from src.patientpose.artifacts import ArtifactStore
from src.patientpose.config.paths import ProjectPaths


def _paths(root: Path) -> ProjectPaths:
    results = root / "results"
    return ProjectPaths(
        root=root,
        sample_data=root / "sample_data",
        results=results,
        output_videos=results / "OutputVideos",
        output_csvs=results / "OutputCSVs",
        output_plots=results / "OutputPlots",
        models=root / "models",
        orientation_debug=results / "orientation_debug",
    )


def test_landmark_diagnostics_artifact_paths_are_stable():
    root = Path("C:/repo")
    store = ArtifactStore(_paths(root))

    trace_artifacts = store.landmark_trace_diagnostics(
        "fingertap",
        source="hand",
        space="image",
        components=("x", "y"),
    )
    metric_artifacts = store.landmark_metric_diagnostics(
        "fingertap",
        source="hand",
        space="image",
        metric="thumb-index-distance",
    )
    overlay_artifacts = store.landmark_overlay_diagnostics(
        "fingertap",
        source="hand",
        space="image",
        metric="thumb-index-distance",
    )
    report_artifacts = store.landmark_metric_report(
        "fingertap",
        source="hand",
        space="image",
        metric="thumb-index-distance",
    )
    batch_report_artifacts = store.landmark_metric_batch_report(
        source="hand",
        space="image",
        metric="thumb-index-distance",
    )
    body_angle_artifacts = store.body_angle_report(
        "fingertap",
        space="image",
    )

    assert trace_artifacts.output_dir == root / "results" / "Diagnostics" / "landmarks"
    assert trace_artifacts.output_plot == root / "results" / "Diagnostics" / "landmarks" / "fingertap_hand_image_x_y_traces.png"
    assert metric_artifacts.output_plot == root / "results" / "Diagnostics" / "landmarks" / "fingertap_hand_image_thumb-index-distance.png"
    assert overlay_artifacts.output_video == root / "results" / "Diagnostics" / "landmarks" / "fingertap_hand_image_thumb-index-distance_overlay.avi"
    assert report_artifacts.trace_csv == root / "results" / "Reports" / "landmark_metrics" / "fingertap_hand_image_thumb-index-distance_trace.csv"
    assert report_artifacts.summary_csv == root / "results" / "Reports" / "landmark_metrics" / "fingertap_hand_image_thumb-index-distance_summary.csv"
    assert batch_report_artifacts.trace_csv == root / "results" / "Reports" / "landmark_metrics" / "batch_hand_image_thumb-index-distance_traces.csv"
    assert batch_report_artifacts.summary_csv == root / "results" / "Reports" / "landmark_metrics" / "batch_hand_image_thumb-index-distance_summary.csv"
    assert body_angle_artifacts.trace_csv == root / "results" / "Reports" / "body_angle" / "fingertap_image_body-angle_trace.csv"
    assert body_angle_artifacts.summary_csv == root / "results" / "Reports" / "body_angle" / "fingertap_image_body-angle_summary.csv"
