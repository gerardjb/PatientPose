from __future__ import annotations

"""Plot Mocopi vs MediaPipe egocentric errors from the reliability export CSV."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mocopi.reliability import joint_medians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot error_2d medians from reliability export.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to mocopi_camera_reliability CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mocopi_reliability_plot.pdf"),
        help="Output path for the plot (vector-friendly).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="error_2d",
        help="Column to plot (default: error_2d).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(f"Metric '{args.metric}' not found in CSV columns: {df.columns}")

    # Drop NaN/inf values for the selected metric to avoid skewed medians.
    df_clean = df[pd.to_numeric(df[args.metric], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).notna()]
    if df_clean.empty:
        raise SystemExit(f"No finite values found for metric '{args.metric}' in {args.csv}")

    grouped = joint_medians(df_clean, args.metric).sort_values("joint")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grouped["joint"], grouped[args.metric], color="#4C78A8")
    ax.set_ylabel(f"Median {args.metric} (body-scale normalized)")
    ax.set_xlabel("Joint")
    ax.set_title("Mocopi vs MediaPipe egocentric error")
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
