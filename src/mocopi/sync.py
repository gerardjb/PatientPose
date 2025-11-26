from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .features import (
    compute_camera_egocentric_positions,
    compute_egocentric_positions,
    resample_feature,
)


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


def clean_feature_samples(
    t_ms: np.ndarray,
    values: np.ndarray,
    label: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove non-finite samples from a feature and ensure enough remain for correlation.
    """
    t_arr = np.asarray(t_ms, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    mask = np.isfinite(t_arr) & np.isfinite(v_arr)
    t_arr = t_arr[mask]
    v_arr = v_arr[mask]
    if t_arr.size < 3:
        msg = f"Not enough finite samples after cleaning"
        if label:
            msg += f" for {label}"
        raise RuntimeError(msg)
    return t_arr, v_arr


def estimate_camera_to_mocopi_offset(
    seq,
    cam_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
    offset_ms: float | None = None,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> float:
    """
    Estimate or reuse the camera→mocopi offset using egocentric, scale-normalized
    r_hand vs RIGHT_WRIST vertical motion.
    """
    if offset_ms is not None:
        return float(offset_ms)

    # Mocopi egocentric r_hand
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, ["r_hand"])
    if "r_hand" not in mocopi_pos:
        raise RuntimeError("Joint 'r_hand' not found in Mocopi sequence")
    f_m = mocopi_pos["r_hand"][:, 1]
    t_m_ms, f_m = clean_feature_samples(t_m_ms, f_m, "mocopi r_hand")

    # Camera egocentric RIGHT_WRIST
    t_c_ms, camera_pos = compute_camera_egocentric_positions(cam_df, ["RIGHT_WRIST"])
    if "RIGHT_WRIST" not in camera_pos:
        raise RuntimeError("Landmark 'RIGHT_WRIST' not found in camera CSV")
    f_c = camera_pos["RIGHT_WRIST"][:, 1]
    t_c_ms, f_c = clean_feature_samples(t_c_ms, f_c, "camera RIGHT_WRIST")

    t_m_res, f_m_res = resample_feature(t_m_ms, f_m, rate_hz)
    t_c_res, f_c_res = resample_feature(t_c_ms, f_c, rate_hz)

    # Align lengths defensively
    t_c_res = np.asarray(t_c_res)
    f_c_res = np.asarray(f_c_res)
    if t_c_res.shape != f_c_res.shape:
        n = min(len(t_c_res), len(f_c_res))
        t_c_res = t_c_res[:n]
        f_c_res = f_c_res[:n]

    if clip_start_s is not None or clip_end_s is not None:
        c_start = clip_start_s if clip_start_s is not None else t_m_res[0] / 1000.0
        c_end = clip_end_s if clip_end_s is not None else t_m_res[-1] / 1000.0
        mask = (t_m_res >= c_start * 1000.0) & (t_m_res <= c_end * 1000.0)
        if mask.sum() > 10:
            t_m_res = t_m_res[mask]
            f_m_res = f_m_res[mask]
            f_c_res = np.interp(t_m_res, t_c_res, f_c_res)
            t_c_res = t_m_res.copy()

    best_offset, _ = estimate_time_offset(
        t_m_res,
        f_m_res,
        t_c_res,
        f_c_res,
        search_range_ms=search_ms,
        step_ms=10.0,
    )
    return float(best_offset)
