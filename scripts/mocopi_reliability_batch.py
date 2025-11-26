from __future__ import annotations

"""
Batch runner for mocopi_reliability_export using the current ND_pilot naming scheme.

For each ND_* video in sample_data that has:
    - a matching A_* unfiltered video, and
    - a matching BVH in sample_data/ND_pilot/Re_ Mocopi with the same tag,
this script calls mocopi_reliability_export to produce a per-frame error CSV.

Example:
    python -m scripts.mocopi_reliability_batch
    python -m scripts.mocopi_reliability_batch --tags 1a 1b
"""

import argparse
from pathlib import Path

from mocopi.nd_pilot import TrialPair, discover_pairs
from mocopi.reliability import ensure_reliability_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run mocopi_reliability_export for ND_pilot pairs.")
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Optional subset of tags to process (e.g., 1a 1b). If omitted, process all discovered pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mocopi_reliability"),
        help="Directory to write reliability CSVs.",
    )
    parser.add_argument(
        "--offset_ms",
        type=float,
        default=None,
        help="Optional fixed camera→mocopi offset to reuse. If omitted, offsets are estimated per pair.",
    )
    parser.add_argument(
        "--search_ms",
        type=float,
        default=5000.0,
        help="Search range for offset estimation in ms (± value).",
    )
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=50.0,
        help="Resampling rate (Hz) used when estimating offset.",
    )
    return parser.parse_args()


def run_for_pair(pair: TrialPair, output_dir: Path, offset_ms: float | None, search_ms: float, rate_hz: float) -> None:
    """
    Invoke mocopi_reliability_export for a single ND/A/BVH triplet.
    """
    nd_stem = pair.nd_video.stem  # e.g., ND_1a_...
    camera_csv = Path("results/OutputCSVs") / f"landmarks_{nd_stem}.csv"
    if not camera_csv.exists():
        print(f"Skipping {nd_stem}: camera CSV not found at {camera_csv}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mocopi_camera_reliability_{nd_stem}.csv"

    print(f"Running reliability export for tag {pair.tag} -> {nd_stem}")
    ensure_reliability_csv(
        pair.bvh,
        camera_csv,
        output_path,
        offset_ms,
        search_ms,
        rate_hz,
    )


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    pairs = discover_pairs(base)
    if args.tags:
        tags = {t for t in args.tags}
        pairs = [p for p in pairs if p.tag in tags]

    if not pairs:
        print("No matching ND/A/BVH pairs found under sample_data/ND_pilot.")
        return

    for pair in pairs:
        run_for_pair(pair, args.output_dir, args.offset_ms, args.search_ms, args.rate_hz)


if __name__ == "__main__":
    main()
