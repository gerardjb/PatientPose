from __future__ import annotations

"""
Symmetry and mapping diagnostics for Mocopi vs MediaPipe egocentric trajectories.

For each A/ND/BVH triplet under sample_data/ND_pilot, this script:
  - Aligns Mocopi and camera egocentric ΔY trajectories for feet and hands.
  - Computes same-side vs cross-side correlations (e.g., l_foot↔LEFT_ANKLE vs l_foot↔RIGHT_ANKLE).
  - Checks correlation sign for each mapping.
  - Plots Mocopi and MediaPipe feet trajectories (l/r) over time for visual phase comparison.

Outputs:
  - Text correlation table to stdout.
  - Per-tag, per-condition (A/ND) PDFs under results/mocopi_reliability/symmetry/.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mocopi.nd_pilot import discover_pairs
from mocopi.reliability import (
    SCALE_REF_JOINTS,
    get_aligned_traces,
)

PAIR_JOINTS = ["l_foot", "r_foot", "l_hand", "r_hand"]
PAIR_LANDMARKS = ["LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_WRIST", "RIGHT_WRIST"]

OUT_DIR = Path("results/mocopi_reliability/symmetry")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symmetry diagnostics for Mocopi vs MediaPipe.")
    parser.add_argument("--tags", nargs="+", help="Optional subset of tags to process (e.g., 1a 1b).")
    parser.add_argument(
        "--offset_ms",
        type=float,
        default=None,
        help="Optional fixed offset to reuse instead of estimating per condition.",
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
        help="Resample rate (Hz) for offset estimation.",
    )
    parser.add_argument(
        "--clip_start",
        type=float,
        default=None,
        help="Optional start time (s) to include in offset estimation/plots.",
    )
    parser.add_argument(
        "--clip_end",
        type=float,
        default=None,
        help="Optional end time (s) to include in offset estimation/plots.",
    )
    return parser.parse_args()


def corr_safe(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 30:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def report_correlations(label: str, t_s: np.ndarray, mocopi_y: Dict[str, np.ndarray], camera_y: Dict[str, np.ndarray]) -> None:
    print(f"\n=== Correlations for {label} ===")
    for j, lm in zip(PAIR_JOINTS, PAIR_LANDMARKS):
        if j not in mocopi_y or lm not in camera_y:
            continue
        same = corr_safe(mocopi_y[j], camera_y[lm])
        # Cross-lateral counterpart
        if "l_" in j:
            lm_cross = lm.replace("LEFT_", "RIGHT_")
        elif "r_" in j:
            lm_cross = lm.replace("RIGHT_", "LEFT_")
        else:
            lm_cross = None
        cross = None
        if lm_cross and lm_cross in camera_y:
            cross = corr_safe(mocopi_y[j], camera_y[lm_cross])

        print(f"{j} ↔ {lm}: same-side corr = {same:.3f}" if same is not None else f"{j} ↔ {lm}: same-side corr = NA")
        if cross is not None:
            print(f"{j} ↔ {lm_cross}: cross-side corr = {cross:.3f}")


def plot_feet(label: str, t_s: np.ndarray, mocopi_y: Dict[str, np.ndarray], camera_y: Dict[str, np.ndarray]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lf_m = mocopi_y.get("l_foot")
    rf_m = mocopi_y.get("r_foot")
    la_c = camera_y.get("LEFT_ANKLE")
    ra_c = camera_y.get("RIGHT_ANKLE")

    # Build a mask of any valid samples to define the mid-trial window
    mask_any = np.zeros_like(t_s, dtype=bool)
    for arr in (lf_m, rf_m, la_c, ra_c):
        if arr is not None:
            mask_any |= np.isfinite(arr)
    if mask_any.sum() < 10:
        print(f"{label}: insufficient valid samples to plot.")
        return

    t_valid = t_s[mask_any]
    t_lo, t_hi = t_valid.min(), t_valid.max()
    t_mid_lo = t_lo + 0.25 * (t_hi - t_lo)
    t_mid_hi = t_lo + 0.75 * (t_hi - t_lo)
    window_mask = (t_s >= t_mid_lo) & (t_s <= t_mid_hi)

    def center_trace(trace: np.ndarray | None) -> np.ndarray | None:
        if trace is None:
            return None
        mask = window_mask & np.isfinite(trace)
        if mask.sum() == 0:
            return trace
        mean = float(trace[mask].mean())
        return trace - mean

    lf_m_c = center_trace(lf_m)
    rf_m_c = center_trace(rf_m)
    la_c_c = center_trace(la_c)
    ra_c_c = center_trace(ra_c)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if lf_m_c is not None:
        ax.plot(t_s, lf_m_c, label="Mocopi L foot", color="#1f77b4", linestyle="-")
    if rf_m_c is not None:
        ax.plot(t_s, rf_m_c, label="Mocopi R foot", color="#ff7f0e", linestyle="-")
    if la_c_c is not None:
        ax.plot(t_s, la_c_c, label="Camera LEFT_ANKLE", color="#1f77b4", linestyle="--")
    if ra_c_c is not None:
        ax.plot(t_s, ra_c_c, label="Camera RIGHT_ANKLE", color="#ff7f0e", linestyle="--")

    ax.set_ylabel("Centered ΔY (body-scale)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{label} – Mocopi vs MediaPipe feet (centered mid-trial)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = OUT_DIR / f"feet_{label.replace(' ', '_')}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved feet plot to {out_path}")


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    pairs = discover_pairs(base)
    if args.tags:
        tags = set(args.tags)
        pairs = [p for p in pairs if p.tag in tags]
    if not pairs:
        print("No matching ND/A/BVH pairs found under sample_data/ND_pilot.")
        return

    for pair in pairs:
        # ND condition
        nd_stem = pair.nd_video.stem
        nd_csv = Path("results/OutputCSVs") / f"landmarks_{nd_stem}.csv"
        if nd_csv.exists():
            t_s, mocopi_y, camera_y, off_nd = get_aligned_traces(
                pair.bvh,
                nd_csv,
                PAIR_JOINTS,
                PAIR_LANDMARKS,
                args.search_ms,
                args.rate_hz,
                args.offset_ms,
                args.clip_start,
                args.clip_end,
            )
            label = f"{pair.tag} ND"
            report_correlations(label, t_s, mocopi_y, camera_y)
            plot_feet(label, t_s, mocopi_y, camera_y)
        else:
            print(f"[{pair.tag}] ND camera CSV {nd_csv} missing; skipping ND diagnostics.")

        # A (unfiltered) condition
        a_stem = pair.unfiltered_video.stem
        a_csv = Path("results/OutputCSVs") / f"landmarks_{a_stem}.csv"
        if a_csv.exists():
            t_s, mocopi_y, camera_y, off_a = get_aligned_traces(
                pair.bvh,
                a_csv,
                PAIR_JOINTS,
                PAIR_LANDMARKS,
                args.search_ms,
                args.rate_hz,
                args.offset_ms,
                args.clip_start,
                args.clip_end,
            )
            label = f"{pair.tag} A"
            report_correlations(label, t_s, mocopi_y, camera_y)
            plot_feet(label, t_s, mocopi_y, camera_y)
        else:
            print(f"[{pair.tag}] A camera CSV {a_csv} missing; skipping A diagnostics.")


if __name__ == "__main__":
    main()
