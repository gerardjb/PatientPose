from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_time_offset(
    t_a_ms: np.ndarray,
    f_a: np.ndarray,
    t_b_ms: np.ndarray,
    f_b: np.ndarray,
    search_range_ms: float,
    step_ms: float = 10.0,
) -> Tuple[float, float]:
    """
    Estimate time offset between two 1D features using cross-correlation over a limited search range.

    Args:
        t_a_ms: timestamps for feature A (e.g., mocopi) in ms, increasing.
        f_a: values for feature A.
        t_b_ms: timestamps for feature B (e.g., camera) in ms, increasing.
        f_b: values for feature B.
        search_range_ms: maximum absolute offset to consider (+/-) in ms.
        step_ms: step size for offset search in ms.

    Returns:
        (best_offset_ms, best_score)
        where best_offset_ms is applied such that:
            t_b_shifted = t_b_ms + best_offset_ms
        aligns feature B onto feature A.
    """
    if len(t_a_ms) == 0 or len(t_b_ms) == 0:
        raise ValueError("Cannot estimate offset on empty features")

    t_a = np.asarray(t_a_ms, dtype=float)
    v_a = (np.asarray(f_a, dtype=float) - np.mean(f_a)) / (np.std(f_a) + 1e-6)

    t_b = np.asarray(t_b_ms, dtype=float)
    v_b_raw = (np.asarray(f_b, dtype=float) - np.mean(f_b)) / (np.std(f_b) + 1e-6)

    offsets = np.arange(-search_range_ms, search_range_ms + step_ms, step_ms, dtype=float)

    best_score = 0.0
    best_abs_score = -np.inf
    best_offset = 0.0

    for offset in offsets:
        t_b_shifted = t_b + offset

        # Interpolate B onto A's timeline over overlapping range.
        t_min = max(t_a[0], t_b_shifted[0])
        t_max = min(t_a[-1], t_b_shifted[-1])
        if t_max <= t_min:
            continue

        mask = (t_a >= t_min) & (t_a <= t_max)
        if not np.any(mask):
            continue

        t_overlap = t_a[mask]
        v_a_overlap = v_a[mask]

        v_b_interp = np.interp(t_overlap, t_b_shifted, v_b_raw)
        if v_b_interp.size < 3:
            continue

        # Compute normalized correlation coefficient.
        num = float(np.sum(v_a_overlap * v_b_interp))
        denom = float(
            np.sqrt(np.sum(v_a_overlap**2) * np.sum(v_b_interp**2)) + 1e-6
        )
        score = num / denom

        # Use the offset that maximizes correlation magnitude, but keep the sign
        # so callers can distinguish positive vs negative correlation.
        if np.isfinite(score) and abs(score) > best_abs_score:
            best_abs_score = abs(score)
            best_score = score
            best_offset = float(offset)

    if not np.isfinite(best_abs_score):
        raise RuntimeError("Unable to compute a finite correlation score for any offset")

    return best_offset, best_score
