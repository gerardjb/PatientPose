from __future__ import annotations

import argparse

from patientpose.pipeline.analyze import (
    add_reliability_batch_args,
    add_reliability_export_args,
    run_reliability_batch,
    run_reliability_export,
)
from patientpose.pipeline.diagnostics import (
    add_egocentric_plot_args,
    add_egocentric_video_args,
    add_landmark_overlay_video_args,
    add_landmark_metric_plot_args,
    add_landmark_traces_args,
    run_egocentric_plot,
    run_egocentric_video,
    run_landmark_overlay_video,
    run_landmark_metric_plot,
    run_landmark_traces,
)
from patientpose.pipeline.preprocess import (
    add_preprocess_quality_video_args,
    add_preprocess_video_args,
    run_preprocess_quality_video,
    run_preprocess_video,
)
from patientpose.pipeline.rendering import (
    add_fourpanel_triplet_args,
    add_side_by_side_args,
    add_triplet_video_args,
    run_fourpanel_triplet,
    run_side_by_side,
    run_triplet_video,
)
from patientpose.pipeline.reporting import (
    add_body_angle_export_args,
    add_landmark_metric_batch_args,
    add_landmark_metric_export_args,
    add_pair_report_args,
    run_body_angle_export,
    run_landmark_metric_batch,
    run_landmark_metric_export,
    run_pair_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patientpose",
        description="PatientPose workflow CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess videos into deidentified AVIs and landmark CSVs.",
    )
    preprocess_subparsers = preprocess_parser.add_subparsers(dest="preprocess_command", required=True)

    preprocess_video_parser = preprocess_subparsers.add_parser(
        "video",
        help="Run the standard video preprocessing workflow.",
    )
    add_preprocess_video_args(preprocess_video_parser)
    preprocess_video_parser.set_defaults(handler=run_preprocess_video)

    preprocess_quality_parser = preprocess_subparsers.add_parser(
        "quality-video",
        help="Run the quality-visualization preprocessing workflow.",
    )
    add_preprocess_quality_video_args(preprocess_quality_parser)
    preprocess_quality_parser.set_defaults(handler=run_preprocess_quality_video)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run mocopi/camera analysis workflows.",
    )
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command", required=True)

    analyze_reliability_parser = analyze_subparsers.add_parser(
        "reliability",
        help="Export per-frame Mocopi vs camera reliability errors.",
    )
    add_reliability_export_args(analyze_reliability_parser)
    analyze_reliability_parser.set_defaults(handler=run_reliability_export)

    analyze_reliability_batch_parser = analyze_subparsers.add_parser(
        "reliability-batch",
        help="Batch-run reliability export across discovered pairs.",
    )
    add_reliability_batch_args(analyze_reliability_batch_parser)
    analyze_reliability_batch_parser.set_defaults(handler=run_reliability_batch)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate summary reports and plots.",
    )
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)

    report_pair_parser = report_subparsers.add_parser(
        "pair-report",
        help="Generate per-pair reliability plots and ND-A summary outputs.",
    )
    add_pair_report_args(report_pair_parser)
    report_pair_parser.set_defaults(handler=run_pair_report)

    report_body_angle_parser = report_subparsers.add_parser(
        "body-angle-export",
        help="Export the egocentric body-angle trace and summary CSVs for one camera CSV.",
    )
    add_body_angle_export_args(report_body_angle_parser)
    report_body_angle_parser.set_defaults(handler=run_body_angle_export)

    report_landmark_export_parser = report_subparsers.add_parser(
        "landmark-metric-export",
        help="Export structured landmark metric trace and summary CSVs for one camera CSV.",
    )
    add_landmark_metric_export_args(report_landmark_export_parser)
    report_landmark_export_parser.set_defaults(handler=run_landmark_metric_export)

    report_landmark_batch_parser = report_subparsers.add_parser(
        "landmark-metric-batch",
        help="Batch export structured landmark metric trace and summary CSVs.",
    )
    add_landmark_metric_batch_args(report_landmark_batch_parser)
    report_landmark_batch_parser.set_defaults(handler=run_landmark_metric_batch)

    render_parser = subparsers.add_parser(
        "render",
        help="Render visualization outputs from processed artifacts.",
    )
    render_subparsers = render_parser.add_subparsers(dest="render_command", required=True)

    render_side_by_side_parser = render_subparsers.add_parser(
        "side-by-side",
        help="Render a camera-vs-mocopi side-by-side video.",
    )
    add_side_by_side_args(render_side_by_side_parser)
    render_side_by_side_parser.set_defaults(handler=run_side_by_side)

    render_triplet_parser = render_subparsers.add_parser(
        "triplet-video",
        help="Render three-panel A/ND/Mocopi triplet videos.",
    )
    add_triplet_video_args(render_triplet_parser)
    render_triplet_parser.set_defaults(handler=run_triplet_video)

    render_fourpanel_parser = render_subparsers.add_parser(
        "fourpanel-triplet",
        help="Render four-panel egocentric plots for a resolved triplet.",
    )
    add_fourpanel_triplet_args(render_fourpanel_parser)
    render_fourpanel_parser.set_defaults(handler=run_fourpanel_triplet)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Run troubleshooting and diagnostics workflows.",
    )
    diagnose_subparsers = diagnose_parser.add_subparsers(dest="diagnose_command", required=True)

    diagnose_plot_parser = diagnose_subparsers.add_parser(
        "egocentric-plot",
        help="Plot dx/dy component traces and projection state for a camera CSV.",
    )
    add_egocentric_plot_args(diagnose_plot_parser)
    diagnose_plot_parser.set_defaults(handler=run_egocentric_plot)

    diagnose_video_parser = diagnose_subparsers.add_parser(
        "egocentric-video",
        help="Render an overlay video showing body-frame axes and projected dx/dy trails.",
    )
    add_egocentric_video_args(diagnose_video_parser)
    diagnose_video_parser.set_defaults(handler=run_egocentric_video)

    diagnose_landmark_traces_parser = diagnose_subparsers.add_parser(
        "landmark-traces",
        help="Plot arbitrary landmark component traces over time from a camera CSV.",
    )
    add_landmark_traces_args(diagnose_landmark_traces_parser)
    diagnose_landmark_traces_parser.set_defaults(handler=run_landmark_traces)

    diagnose_landmark_metric_parser = diagnose_subparsers.add_parser(
        "landmark-metric-plot",
        help="Plot a derived metric over arbitrary landmark sets from a camera CSV.",
    )
    add_landmark_metric_plot_args(diagnose_landmark_metric_parser)
    diagnose_landmark_metric_parser.set_defaults(handler=run_landmark_metric_plot)

    diagnose_landmark_overlay_parser = diagnose_subparsers.add_parser(
        "landmark-overlay-video",
        help="Render a video overlay with selected landmarks and a rolling derived-metric trace.",
    )
    add_landmark_overlay_video_args(diagnose_landmark_overlay_parser)
    diagnose_landmark_overlay_parser.set_defaults(handler=run_landmark_overlay_video)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
