from __future__ import annotations

"""
Batch runner for mocopi_reliability_export across legacy ND_pilot pairs and session folders.

Example:
    python -m scripts.mocopi_reliability_batch
    python -m scripts.mocopi_reliability_batch --tags 1a 1b
"""

import argparse
from pathlib import Path

import pandas as pd

from mocopi.nd_pilot import TrialPair, discover_pairs, discover_sessions, parse_camera_role_specs
from mocopi.reliability import RELIABILITY_COLUMNS, ensure_reliability_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run mocopi_reliability_export for discovered Mocopi/camera pairs.")
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Optional subset of pair ids to process. Legacy mode uses tags like 1a; session mode uses session ids.",
    )
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
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
    Invoke mocopi_reliability_export for a single ND/A/Mocopi triplet.
    """
    nd_stem = pair.nd_video.stem  # e.g., ND_1a_...
    camera_csv = Path("results/OutputCSVs") / f"landmarks_{nd_stem}.csv"
    if not camera_csv.exists():
        print(f"Skipping {nd_stem}: camera CSV not found at {camera_csv}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mocopi_camera_reliability_{nd_stem}.csv"

    print(f"Running reliability export for tag {pair.tag} -> {nd_stem}")
    output_csv = ensure_reliability_csv(
        pair.motion_source,
        camera_csv,
        output_path,
        offset_ms,
        search_ms,
        rate_hz,
    )
    df = None
    try:
        df = pd.read_csv(output_csv)
    except Exception:
        df = None
    if df is not None and list(df.columns) == RELIABILITY_COLUMNS and df.empty:
        print(f"Skipping metrics for {nd_stem}: camera CSV contains no usable pose landmarks.")


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    camera_roles = parse_camera_role_specs(args.camera_role)
    pairs = discover_pairs(base, camera_roles=camera_roles)
    if args.tags:
        tags = {t for t in args.tags}
        pairs = [p for p in pairs if p.tag in tags]

    if not pairs:
        sessions = discover_sessions(base)
        if sessions and not camera_roles:
            print("Discovered session folders, but no session pairs were resolved. Add --camera-role CAMERA_ID=A and --camera-role CAMERA_ID=ND.")
        else:
            print("No matching Mocopi/camera pairs found.")
        return

    for pair in pairs:
        run_for_pair(pair, args.output_dir, args.offset_ms, args.search_ms, args.rate_hz)


if __name__ == "__main__":
    main()
