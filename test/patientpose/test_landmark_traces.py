from pathlib import Path

import pandas as pd

from src.patientpose.diagnostics.landmark_traces import (
    TraceSpec,
    plot_landmark_components,
    plot_metric_panels,
    plot_metric_trace,
    plot_pairwise_landmark_comparison,
)


TEST_OUTPUT_DIR = Path("tmp") / "test_landmark_traces"


def _series(values: list[tuple[int, int, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "timestamp_ms": timestamp_ms,
                "x": x,
                "y": y,
                "z": z,
            }
            for frame, timestamp_ms, x, y, z in values
        ]
    )


def _metric(values: list[tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "timestamp_ms": timestamp_ms,
                "value": value,
            }
            for frame, timestamp_ms, value in values
        ]
    )


def _assert_nonempty_file(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0


def _output_path(filename: str) -> Path:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = TEST_OUTPUT_DIR / filename
    if path.exists():
        path.unlink()
    return path


def test_plot_landmark_components_writes_png():
    output_path = _output_path("landmark_components.png")
    plot_landmark_components(
        {
            "THUMB_TIP": _series([(0, 0, 0.1, 0.2, 0.3), (1, 33, 0.2, 0.3, 0.4)]),
            "INDEX_FINGER_TIP": _series([(0, 0, 0.4, 0.5, 0.6), (1, 33, 0.5, 0.6, 0.7)]),
        },
        output_path,
        title="Landmark components",
        components=("x", "y"),
        invert_components=("y",),
    )
    _assert_nonempty_file(output_path)


def test_plot_metric_trace_writes_png():
    output_path = _output_path("metric_trace.png")
    plot_metric_trace(
        [
            TraceSpec(label="thumb-index distance", df=_metric([(0, 0, 1.0), (1, 33, 2.0)])),
        ],
        output_path,
        title="Metric trace",
        y_label="distance",
    )
    _assert_nonempty_file(output_path)


def test_plot_metric_panels_writes_png():
    output_path = _output_path("metric_panels.png")
    plot_metric_panels(
        [
            (
                "Position",
                [
                    TraceSpec(label="index", df=_metric([(0, 0, 0.3), (1, 33, 0.35)])),
                    TraceSpec(label="thumb", df=_metric([(0, 0, 0.4), (1, 33, 0.45)])),
                ],
            ),
            (
                "Quality",
                [
                    TraceSpec(label="index", df=_metric([(0, 0, 0.8), (1, 33, 0.82)])),
                    TraceSpec(label="thumb", df=_metric([(0, 0, 0.9), (1, 33, 0.91)])),
                ],
            ),
        ],
        output_path,
        title="Metric panels",
        y_labels=("position", "quality"),
    )
    _assert_nonempty_file(output_path)


def test_plot_pairwise_landmark_comparison_writes_png():
    output_path = _output_path("pairwise_landmarks.png")
    plot_pairwise_landmark_comparison(
        _series([(0, 0, 0.1, 0.2, 0.3), (1, 33, 0.2, 0.3, 0.4)]),
        _series([(0, 0, 0.4, 0.5, 0.6), (1, 33, 0.5, 0.6, 0.7)]),
        output_path,
        title="Pairwise comparison",
        components=("x", "y"),
        label_a="thumb",
        label_b="index",
    )
    _assert_nonempty_file(output_path)
