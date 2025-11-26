from __future__ import annotations

"""Plot median error per joint across ND levels (log2 scale)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mocopi.reliability import nd_error_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot median error vs ND level across reliability CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of reliability CSVs annotated with ND, e.g., ND=2:file.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mocopi_nd_summary.pdf"),
        help="Output plot path (vector-friendly).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="error_2d",
        help="Metric column to use (default: error_2d).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    items = []
    for item in args.inputs:
        if ":" in item:
            nd_str, csv_path = item.split(":", 1)
        elif "=" in item:
            nd_str, csv_path = item.split("=", 1)
        else:
            raise SystemExit("Each input must be ND=<value>:<csv_path>")
        nd_level = float(nd_str.replace("ND", "").replace("=", ""))
        df = pd.read_csv(csv_path)
        if args.metric not in df.columns:
            raise SystemExit(f"Metric {args.metric} not in {csv_path}")
        items.append((nd_level, df))

    df_all = nd_error_summary(items, args.metric)
    if df_all.empty:
        raise SystemExit("No data to plot")

    fig, ax = plt.subplots(figsize=(6, 4))
    for joint, sub in df_all.groupby("joint"):
        sub = sub.sort_values("nd")
        ax.plot(sub["nd"], sub[args.metric], marker="o", label=joint)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("ND factor (log2)")
    ax.set_ylabel(f"Median {args.metric} (body-scale normalized)")
    ax.set_title("Error vs ND level")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
