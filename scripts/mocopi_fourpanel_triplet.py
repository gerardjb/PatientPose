from __future__ import annotations

"""
Plot egocentric trajectories for a tagged A/ND/Mocopi triplet on four axes:
  1) Mocopi egocentric ΔY for feet joints (defaults to l_foot/r_foot)
  2) Camera A (ND=0) egocentric ΔY for corresponding landmarks
  3) Camera ND egocentric ΔY for corresponding landmarks
  4) Percent of pose keypoints above the visibility threshold over time (A and ND)

Example:
    python -m scripts.mocopi_fourpanel_triplet \\
        --tag 1a \\
        --visibility-threshold 0.8 \\
        --offset-ms 0.0 \\
        --output results/fourpanel_1a.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from mocopi import (
    load_mocopi_recording,
    nd_factor_from_stem,
    visibility_percent,
)
from mocopi.features import (
    NoCameraPoseDataError,
    compute_egocentric_positions,
    compute_camera_egocentric_positions,
)
from mocopi.nd_pilot import discover_pairs, discover_sessions, parse_camera_role_specs

COLOR_MOCOPI = "#000000"
COLOR_A = "#1d4f8a"  # brightened blue
COLOR_ND = "#ff00ff"  # magenta

DEFAULT_JOINTS = ["l_foot", "r_foot"]
DEFAULT_LANDMARKS = ["LEFT_ANKLE", "RIGHT_ANKLE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four-panel egocentric plot for Mocopi vs camera A/ND (no temporal offset)."
    )
    parser.add_argument("--tag", required=True, help="Triplet tag to plot (e.g., 1a).")
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        help="Session-mode camera mapping in the form CAMERA_ID=ROLE, where ROLE is A or ND.",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=DEFAULT_JOINTS,
        help="Mocopi joint names to plot (default: l_foot r_foot).",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=DEFAULT_LANDMARKS,
        help="Camera pose landmarks to plot in the same order (default: LEFT_ANKLE RIGHT_ANKLE).",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.8,
        help="Visibility threshold for camera landmarks (default: 0.8).",
    )
    parser.add_argument(
        "--offset-ms",
        type=float,
        default=0.0,
        help="Optional camera→mocopi offset (ms) to apply to camera timelines before plotting.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional minimum time (s) for all x-axes. Defaults to min of all traces.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional maximum time (s) for all x-axes. Defaults to max of all traces.",
    )
    parser.add_argument(
        "--y-dy-min",
        type=float,
        default=None,
        help="Optional minimum ΔY for motion panels.",
    )
    parser.add_argument(
        "--y-dy-max",
        type=float,
        default=None,
        help="Optional maximum ΔY for motion panels.",
    )
    parser.add_argument(
        "--y-count-min",
        type=float,
        default=None,
        help="Optional minimum for visibility-count panel.",
    )
    parser.add_argument(
        "--y-count-max",
        type=float,
        default=None,
        help="Optional maximum for visibility-count panel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fourpanel_triplet.pdf"),
        help="Output path for the PDF/PNG.",
    )
    return parser.parse_args()


def _plot_traces(
    ax,
    t_ms: np.ndarray,
    traces: dict[str, np.ndarray],
    label_text: str,
    xlim: tuple[float, float],
    label_color: str,
):
    t_s = t_ms / 1000.0
    for name, arr in traces.items():
        if arr is None:
            continue
        if arr.shape[0] != t_s.shape[0]:
            continue
        is_right = name.lower().startswith("r_") or name.lower().startswith("right")
        linestyle = "--" if is_right else "-"
        ax.plot(t_s, arr[:, 1], label=name, color=COLOR_MOCOPI, linestyle=linestyle)
    ax.set_ylabel(
        label_text,
        color=label_color,
        fontsize=12,
        rotation=0,
        ha="right",
        va="top",
        labelpad=20,
    )
    ax.grid(False)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlim(xlim)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _auto_ylim_from_window(
    t_ms: np.ndarray,
    traces: dict[str, np.ndarray],
    xlim: tuple[float, float],
    padding: float = 0.05,
) -> tuple[float, float] | None:
    t_s = t_ms / 1000.0
    mask = (t_s >= xlim[0]) & (t_s <= xlim[1])
    if not np.any(mask):
        return None
    ys: list[float] = []
    for arr in traces.values():
        if arr is None or arr.shape[0] != t_s.shape[0]:
            continue
        y_vals = arr[:, 1][mask]
        y_vals = y_vals[np.isfinite(y_vals)]
        if y_vals.size:
            ys.append(float(np.min(y_vals)))
            ys.append(float(np.max(y_vals)))
    if not ys:
        return None
    y_min = min(ys)
    y_max = max(ys)
    if y_max == y_min:
        y_pad = max(1e-3, abs(y_max) * padding)
        return y_min - y_pad, y_max + y_pad
    span = y_max - y_min
    pad = span * padding
    return y_min - pad, y_max + pad


def _ensure_percent_trace(
    t_ms: np.ndarray,
    perc: np.ndarray,
    xlim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure we have something to plot; if no timestamps/values, return a 0% line over xlim.
    """
    if t_ms is None or perc is None or len(t_ms) == 0 or len(perc) == 0:
        t_fill = np.array([xlim[0] * 1000.0, xlim[1] * 1000.0], dtype=float)
        p_fill = np.zeros_like(t_fill, dtype=float)
        return t_fill, p_fill
    if np.all(~np.isfinite(perc)):
        t_fill = np.array([xlim[0] * 1000.0, xlim[1] * 1000.0], dtype=float)
        p_fill = np.zeros_like(t_fill, dtype=float)
        return t_fill, p_fill
    return t_ms, perc


def main() -> None:
    args = parse_args()
    if len(args.joints) != len(args.landmarks):
        raise SystemExit("Expected --joints and --landmarks to have the same length")

    default_output = Path("results/fourpanel_triplet.pdf")
    if args.output == default_output and args.tag:
        args.output = default_output.with_name(f"fourpanel_{args.tag}.pdf")

    base = Path(__file__).resolve().parent.parent
    camera_roles = parse_camera_role_specs(args.camera_role)
    pair = next((p for p in discover_pairs(base, camera_roles=camera_roles) if p.tag == args.tag), None)
    if pair is None:
        hint = ""
        if discover_sessions(base) and not camera_roles:
            hint = " Add --camera-role CAMERA_ID=A and --camera-role CAMERA_ID=ND for session data."
        raise SystemExit(f"Tag '{args.tag}' not found in discovered Mocopi/camera pairs.{hint}")

    nd_csv = Path("results/OutputCSVs") / f"landmarks_{pair.nd_video.stem}.csv"
    a_csv = Path("results/OutputCSVs") / f"landmarks_{pair.unfiltered_video.stem}.csv"
    if not nd_csv.exists() or not a_csv.exists():
        raise SystemExit(f"Missing camera CSVs for tag={args.tag} (ND: {nd_csv}, A: {a_csv})")

    # Mocopi egocentric
    seq = load_mocopi_recording(pair.motion_source)
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, args.joints)

    # Camera egocentric
    nd_df = pd.read_csv(nd_csv)
    a_df = pd.read_csv(a_csv)
    try:
        t_nd_ms, nd_pos = compute_camera_egocentric_positions(
            nd_df, args.landmarks, visibility_threshold=args.visibility_threshold
        )
    except NoCameraPoseDataError as exc:
        raise SystemExit(f"ND camera CSV has no usable pose landmarks: {nd_csv}") from exc
    try:
        t_a_ms, a_pos = compute_camera_egocentric_positions(
            a_df, args.landmarks, visibility_threshold=args.visibility_threshold
        )
    except NoCameraPoseDataError as exc:
        raise SystemExit(f"A camera CSV has no usable pose landmarks: {a_csv}") from exc
    if args.offset_ms:
        t_nd_ms = t_nd_ms + args.offset_ms
        t_a_ms = t_a_ms + args.offset_ms

    # Rebase all timelines so that the provided offset becomes time zero.
    if args.offset_ms:
        t_shift = args.offset_ms
        t_m_ms = t_m_ms - t_shift
        t_nd_ms = t_nd_ms - t_shift
        t_a_ms = t_a_ms - t_shift

    # Count visible keypoints per timestamp and convert to percentage.
    t_nd_count, nd_percent = visibility_percent(nd_df, args.visibility_threshold)
    t_a_count, a_percent = visibility_percent(a_df, args.visibility_threshold)
    if args.offset_ms:
        t_nd_count = t_nd_count + args.offset_ms
        t_a_count = t_a_count + args.offset_ms
        t_nd_count = t_nd_count - t_shift
        t_a_count = t_a_count - t_shift

    # Build figure
    fig, axes = plt.subplots(4, 1, figsize=(6, 4), sharex=True)
    ax_mocopi, ax_a, ax_nd, ax_vis = axes

    # Determine global x-limits (seconds) to align zero across panels.
    time_arrays = []
    for arr in (t_m_ms, t_a_ms, t_nd_ms, t_a_count, t_nd_count):
        if arr is not None and len(arr) > 0:
            time_arrays.append(arr / 1000.0)
    if not time_arrays:
        raise SystemExit("No timestamps available to set x-limits.")
    global_min = float(min(np.min(arr) for arr in time_arrays))
    global_max = float(max(np.max(arr) for arr in time_arrays))
    x_lo = args.x_min if args.x_min is not None else global_min
    x_hi = args.x_max if args.x_max is not None else global_max
    if x_hi <= x_lo:
        raise SystemExit("Invalid x-limits: x-max must be greater than x-min.")
    xlim = (x_lo, x_hi)

    mocopi_traces = {k: mocopi_pos.get(k) for k in args.joints}
    a_traces = {k: a_pos.get(k) for k in args.landmarks}
    nd_traces = {k: nd_pos.get(k) for k in args.landmarks}

    nd_factor = nd_factor_from_stem(pair.nd_video.stem)
    nd_label_text = f"Video ND = {nd_factor:g}" if nd_factor is not None else "Video ND = ?"

    # Ensure percent traces have data to plot (fill zeros if empty).
    t_a_count, a_percent = _ensure_percent_trace(t_a_count, a_percent, xlim)
    t_nd_count, nd_percent = _ensure_percent_trace(t_nd_count, nd_percent, xlim)

    _plot_traces(
        ax_mocopi,
        t_m_ms,
        mocopi_traces,
        "Mocopi",
        xlim,
        COLOR_MOCOPI,
    )
    _plot_traces(
        ax_a,
        t_a_ms,
        a_traces,
        "Video ND\n0",
        xlim,
        COLOR_A,
    )
    _plot_traces(
        ax_nd,
        t_nd_ms,
        nd_traces,
        nd_label_text.replace("Video ND = ", "Video ND\n"),
        xlim,
        COLOR_ND,
    )

    # Auto-scale Y for motion panels within the x-window unless overridden.
    for ax, traces, t_ms in (
        (ax_mocopi, mocopi_traces, t_m_ms),
        (ax_a, a_traces, t_a_ms),
        (ax_nd, nd_traces, t_nd_ms),
    ):
        if args.y_dy_min is not None or args.y_dy_max is not None:
            ymin = args.y_dy_min if args.y_dy_min is not None else ax.get_ylim()[0]
            ymax = args.y_dy_max if args.y_dy_max is not None else ax.get_ylim()[1]
        else:
            yl = _auto_ylim_from_window(t_ms, traces, xlim)
            if yl is None:
                ymin, ymax = ax.get_ylim()
            else:
                ymin, ymax = yl
        ax.set_ylim(ymin, ymax)

    # Visibility counts (%). Always show a zero baseline across the current window.
    base_t = np.array([xlim[0], xlim[1]], dtype=float)
    ax_vis.plot(base_t, np.zeros_like(base_t), color=COLOR_A, linestyle=":", linewidth=1.0)
    ax_vis.plot(base_t, np.zeros_like(base_t), color=COLOR_ND, linestyle=":", linewidth=1.0)
    mask_a = np.isfinite(a_percent)
    if np.any(mask_a):
        ax_vis.plot(t_a_count[mask_a] / 1000.0, a_percent[mask_a], label="Video ND = 0", color=COLOR_A)
    mask_nd = np.isfinite(nd_percent)
    if np.any(mask_nd):
        ax_vis.plot(t_nd_count[mask_nd] / 1000.0, nd_percent[mask_nd], label=nd_label_text, color=COLOR_ND)
    ax_vis.set_title("")
    ax_vis.set_ylabel("Visible\n keypoints (%)", color=COLOR_MOCOPI)
    ax_vis.set_xlabel("")
    ax_vis.grid(alpha=0.3)
    ax_vis.legend(fontsize=8, frameon=False)
    ax_vis.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_vis.set_xlim(xlim)
    for spine in ax_vis.spines.values():
        spine.set_visible(False)
    if args.y_count_min is not None or args.y_count_max is not None:
        ymin = args.y_count_min if args.y_count_min is not None else ax_vis.get_ylim()[0]
        ymax = args.y_count_max if args.y_count_max is not None else ax_vis.get_ylim()[1]
        ax_vis.set_ylim(ymin, ymax)
    else:
        counts_stack = []
        for t_ms, counts in ((t_a_count, a_percent), (t_nd_count, nd_percent)):
            if t_ms is None or counts is None or len(t_ms) != len(counts):
                continue
            t_s = t_ms / 1000.0
            mask = (t_s >= xlim[0]) & (t_s <= xlim[1])
            if not np.any(mask):
                continue
            vals = counts[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                counts_stack.append(float(np.min(vals)))
                counts_stack.append(float(np.max(vals)))
        if counts_stack:
            cmin = min(counts_stack)
            cmax = max(counts_stack)
            if cmax == cmin:
                pad = max(1.0, cmax * 0.05)
                ax_vis.set_ylim(cmin - pad, cmax + pad)
            else:
                span = cmax - cmin
                pad = span * 0.05
                ax_vis.set_ylim(cmin - pad, cmax + pad)
        else:
            # No data at all; show a flat 0–1% band.
            ax_vis.set_ylim(-1.0, 1.0)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    offset_tag = f"offset_{args.offset_ms:.1f}ms"
    vis_tag = f"vis_{args.visibility_threshold:.2f}"
    out_path = args.output.with_name(f"{args.output.stem}_{offset_tag}_{vis_tag}{args.output.suffix}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"Saved four-panel plot to {out_path} "
        f"(offset_ms={args.offset_ms:.1f}, visibility_threshold={args.visibility_threshold:.2f})"
    )


if __name__ == "__main__":
    main()
