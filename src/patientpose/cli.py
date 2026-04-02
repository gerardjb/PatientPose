from __future__ import annotations

import argparse

from patientpose.pipeline.preprocess import add_preprocess_video_args, run_preprocess_video


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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
