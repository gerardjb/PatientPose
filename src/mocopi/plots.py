from __future__ import annotations

"""Reusable matplotlib plots for Mocopi diagnostics."""

from pathlib import Path
from typing import Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def select_overlap_window(
    t_m_ms: np.ndarray,
    t_c_ms: np.ndarray,
    t_start: float | None = None,
    t_end: float | None = None,
) -> Tuple[float, float]:
    """
    Determine a shared time window (seconds) given two aligned timelines.
    """
    t_m = t_m_ms / 1000.0
    t_c = t_c_ms / 1000.0

    window_start = max(t_m[0], t_c[0])
    window_end = min(t_m[-1], t_c[-1])

    if t_start is not None:
        window_start = max(window_start, t_start)
    if t_end is not None:
        window_end = min(window_end, t_end)

    if window_end <= window_start:
        raise RuntimeError("No overlapping time window for Mocopi and camera after alignment")

    return window_start, window_end


def plot_egocentric_compare(
    t_m_ms: np.ndarray,
    mocopi_pos: dict[str, np.ndarray],
    t_c_ms_aligned: np.ndarray,
    camera_pos: dict[str, np.ndarray],
    joints: Sequence[str],
    landmarks: Sequence[str],
    window: tuple[float, float],
    output_path: Path,
) -> None:
    t_m = t_m_ms / 1000.0
    t_c = t_c_ms_aligned / 1000.0
    t_start, t_end = window

    mask_m = (t_m >= t_start) & (t_m <= t_end)
    mask_c = (t_c >= t_start) & (t_c <= t_end)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_m = axes[0]
    ax_c = axes[1]

    for name in joints:
        if name not in mocopi_pos:
            continue
        traj = mocopi_pos[name][mask_m]
        ax_m.plot(t_m[mask_m], traj[:, 1], label=name)
    ax_m.set_ylabel("Mocopi ΔY (egocentric)")
    ax_m.set_title("Mocopi egocentric vertical motion")
    ax_m.grid(True, alpha=0.3)
    ax_m.legend(loc="upper right", fontsize=8)

    for name in landmarks:
        if name not in camera_pos:
            continue
        traj = camera_pos[name][mask_c]
        ax_c.plot(t_c[mask_c], traj[:, 1], label=name)
    ax_c.set_ylabel("Camera ΔY (egocentric)")
    ax_c.set_xlabel("Time (s, aligned)")
    ax_c.set_title("Camera egocentric vertical motion")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feet_centered(
    label: str,
    t_s: np.ndarray,
    mocopi_y: dict[str, np.ndarray],
    camera_y: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    lf_m = mocopi_y.get("l_foot")
    rf_m = mocopi_y.get("r_foot")
    la_c = camera_y.get("LEFT_ANKLE")
    ra_c = camera_y.get("RIGHT_ANKLE")

    mask_any = np.zeros_like(t_s, dtype=bool)
    for arr in (lf_m, rf_m, la_c, ra_c):
        if arr is not None:
            mask_any |= np.isfinite(arr)
    if mask_any.sum() < 10:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


__all__ = ["select_overlap_window", "plot_egocentric_compare", "plot_feet_centered"]
