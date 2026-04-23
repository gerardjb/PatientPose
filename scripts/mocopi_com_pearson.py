from __future__ import annotations

"""
Standalone Mocopi-vs-camera trace correlation trial.

This script is intentionally kept under scripts/ while the workflow is being
evaluated. It compares a selected Mocopi trace with a selected MediaPipe camera
trace derived from camera_projection.py:

    - Mocopi trace: COM, a single joint, the midpoint of two joints, or a
      left-right difference.
    - Camera trace: projection origin, a single landmark, a midpoint of two
      landmarks, or a left-right difference; either projected or raw.
    - Offset: camera timestamps are shifted by offset_ms, matching sync.py:
          t_camera_aligned = t_camera + offset_ms
    - Offset search: weighted cross-correlation, optionally emphasizing frames
      where MediaPipe appears front-facing/high-quality.
    - Optional detrending removes slow baseline shifts before alignment and
      correlation while preserving the raw traces for plotting.

Example:
    python -m scripts.mocopi_com_pearson ^
      --motion sample_data/ND_pilot/Re_Mocopi/MCPM_20251112_135620_1a.bvh ^
      --camera_csv results/OutputCSVs/landmarks_VID_20260215_042925.061.csv ^
      --camera-space world ^
      --component z
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mocopi import CameraProjectionConfig, compute_camera_projection, load_mocopi_recording
from mocopi.features import compute_center_of_mass, resample_feature
from patientpose.landmarks import landmark_stem_from_image_csv, load_landmark_views


DEFAULT_OUTPUT_DIR = Path("results/mocopi_reliability/com_pearson")
DEFAULT_ORIGIN_LANDMARKS = ("LEFT_HIP", "RIGHT_HIP")
DEFAULT_CAMERA_TRACE_LANDMARKS = ("LEFT_ANKLE", "RIGHT_ANKLE")
DEFAULT_MOCOPI_TRACE_JOINTS = ("l_foot", "r_foot")
DEFAULT_SYNC_MOCOPI_JOINTS = ("l_foot", "r_foot")
DEFAULT_SYNC_CAMERA_LANDMARKS = ("LEFT_ANKLE", "RIGHT_ANKLE")


@dataclass(frozen=True)
class OffsetScanResult:
    offsets_ms: np.ndarray
    scores: np.ndarray
    metrics: np.ndarray
    n_effective: np.ndarray
    coverage: np.ndarray
    best_offset_ms: float
    best_score: float
    best_metric: float
    peak_margin: float


@dataclass(frozen=True)
class AlignmentResult:
    time_s: np.ndarray
    mocopi_trace: np.ndarray
    camera_trace: np.ndarray
    front_facing_score: np.ndarray
    front_facing_used: np.ndarray
    offset_ms: float
    full_mocopi_time_s: np.ndarray
    full_mocopi_trace: np.ndarray
    full_camera_time_s: np.ndarray
    full_camera_trace: np.ndarray
    full_front_time_s: np.ndarray
    full_front_score: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Mocopi-vs-MediaPipe COM trace correlation with "
            "front-facing weighted offset search."
        )
    )
    parser.add_argument(
        "--motion",
        "--bvh",
        dest="motion_source",
        type=Path,
        required=True,
        help="Path to Mocopi motion source (.bvh, .bin, or session directory).",
    )
    parser.add_argument(
        "--camera_csv",
        type=Path,
        required=True,
        help="Path to image-space camera landmarks CSV.",
    )
    parser.add_argument(
        "--world_csv",
        type=Path,
        default=None,
        help="Optional pose_world CSV. Inferred from metadata/results when omitted.",
    )
    parser.add_argument(
        "--camera-space",
        choices=("image", "world"),
        default="world",
        help="Camera coordinate space used for the camera COM proxy.",
    )
    parser.add_argument(
        "--component",
        choices=("x", "y", "z"),
        default=None,
        help="Component for both traces. Defaults to z for world and y for image.",
    )
    parser.add_argument(
        "--mocopi-component",
        choices=("x", "y", "z"),
        default=None,
        help="Optional Mocopi component override.",
    )
    parser.add_argument(
        "--mocopi-trace",
        choices=("com", "single", "midpoint", "difference"),
        default="com",
        help="Mocopi trace definition.",
    )
    parser.add_argument(
        "--mocopi-joints",
        nargs="+",
        default=list(DEFAULT_MOCOPI_TRACE_JOINTS),
        help=(
            "Mocopi joints used when --mocopi-trace is single, midpoint, or difference. "
            "For gait, l_foot r_foot is the default."
        ),
    )
    parser.add_argument(
        "--camera-component",
        choices=("x", "y", "z"),
        default=None,
        help="Optional camera component override.",
    )
    parser.add_argument(
        "--mocopi-com-joints",
        nargs="+",
        default=None,
        help="Optional Mocopi joints for the COM proxy. Defaults to core joints.",
    )
    parser.add_argument(
        "--camera-trace",
        choices=(
            "origin",
            "midpoint",
            "difference",
            "single",
            "raw-midpoint",
            "raw-difference",
            "raw-single",
        ),
        default="origin",
        help=(
            "Camera trace definition. origin uses the projection origin. midpoint/difference/single "
            "use projected body-scale-relative landmark positions. raw-midpoint/raw-difference/raw-single "
            "use the raw selected landmark coordinate component from the chosen camera space."
        ),
    )
    parser.add_argument(
        "--camera-landmarks",
        nargs="+",
        default=list(DEFAULT_CAMERA_TRACE_LANDMARKS),
        help=(
            "Camera landmarks used when --camera-trace is midpoint, difference, or single. "
            "For distal gait traces, use LEFT_ANKLE RIGHT_ANKLE."
        ),
    )
    parser.add_argument(
        "--origin-landmarks",
        nargs="+",
        default=list(DEFAULT_ORIGIN_LANDMARKS),
        help="Camera landmarks used as the projection origin.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.4,
        help="MediaPipe visibility threshold for camera projection.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Projection smoothing window.",
    )
    parser.add_argument(
        "--offset-ms",
        type=float,
        default=None,
        help="Optional fixed camera-to-Mocopi offset. If omitted, estimate by cross-correlation.",
    )
    parser.add_argument(
        "--search-ms",
        type=float,
        default=5000.0,
        help="Offset search range in ms, plus/minus this value.",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=10.0,
        help="Offset scan step in ms.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=50.0,
        help="Resampling rate used during offset search.",
    )
    parser.add_argument(
        "--front-weight-mode",
        choices=("soft", "hard", "none"),
        default="soft",
        help=(
            "How front-facing scores affect offset search: soft weights by score, "
            "hard keeps only frames above threshold, none ignores the score."
        ),
    )
    parser.add_argument(
        "--front-facing-threshold",
        type=float,
        default=0.5,
        help="Threshold used by hard front-facing mode and front-only Pearson output.",
    )
    parser.add_argument(
        "--front-window",
        nargs=2,
        type=float,
        action="append",
        metavar=("START_S", "END_S"),
        default=None,
        help=(
            "Camera-timeline window that is visually confirmed front-facing. "
            "May be repeated. When provided, automatic front-facing scores are "
            "zeroed outside these windows."
        ),
    )
    parser.add_argument(
        "--front-segment",
        choices=("all", "first"),
        default="all",
        help=(
            "How to use connected front-score runs above threshold. "
            "all keeps every run; first keeps only the first connected run."
        ),
    )
    parser.add_argument(
        "--front-segment-trim-start-s",
        type=float,
        default=0.0,
        help="Seconds to trim from the start of each kept front-score run.",
    )
    parser.add_argument(
        "--front-segment-trim-end-s",
        type=float,
        default=0.0,
        help="Seconds to trim from the end of each kept front-score run.",
    )
    parser.add_argument(
        "--correlation-mode",
        choices=("absolute", "positive", "negative"),
        default="absolute",
        help="How to pick the best offset from signed cross-correlation scores.",
    )
    parser.add_argument(
        "--sync-signal",
        choices=("trace", "left-right-difference"),
        default="trace",
        help=(
            "Signal used for offset estimation. trace uses the selected report traces; "
            "left-right-difference uses left-right gait difference signals."
        ),
    )
    parser.add_argument(
        "--sync-mocopi-joints",
        nargs=2,
        default=None,
        metavar=("LEFT_JOINT", "RIGHT_JOINT"),
        help="Optional Mocopi joints used when --sync-signal left-right-difference.",
    )
    parser.add_argument(
        "--sync-camera-landmarks",
        nargs=2,
        default=None,
        metavar=("LEFT_LANDMARK", "RIGHT_LANDMARK"),
        help="Optional camera landmarks used when --sync-signal left-right-difference.",
    )
    parser.add_argument(
        "--detrend",
        choices=("none", "rolling-mean", "rolling-median"),
        default="none",
        help="Optional detrending applied before offset estimation and Pearson calculation.",
    )
    parser.add_argument(
        "--detrend-window-s",
        type=float,
        default=1.5,
        help="Window in seconds used for rolling detrending.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum valid/effective samples needed for a correlation.",
    )
    parser.add_argument(
        "--peak-separation-ms",
        type=float,
        default=150.0,
        help="Offset distance used when estimating peak margin.",
    )
    parser.add_argument(
        "--clip-start",
        type=float,
        default=None,
        help="Optional start time in seconds on the Mocopi/reference timeline.",
    )
    parser.add_argument(
        "--clip-end",
        type=float,
        default=None,
        help="Optional end time in seconds on the Mocopi/reference timeline.",
    )
    parser.add_argument(
        "--invert-camera",
        action="store_true",
        help="Multiply the camera trace by -1 before alignment/correlation.",
    )
    parser.add_argument(
        "--invert-mocopi",
        action="store_true",
        help="Multiply the Mocopi trace by -1 before alignment/correlation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for default outputs.",
    )
    parser.add_argument("--trace-output", type=Path, default=None, help="Aligned trace CSV path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Summary CSV path.")
    parser.add_argument("--scan-output", type=Path, default=None, help="Offset scan CSV path.")
    parser.add_argument("--plot-output", type=Path, default=None, help="Aligned trace PNG path.")
    return parser.parse_args()


def component_index(component: str) -> int:
    comp = component.lower()
    if comp == "x":
        return 0
    if comp == "y":
        return 1
    if comp == "z":
        return 2
    raise ValueError(f"Unsupported component: {component!r}")


def clean_series(t_ms: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_ms, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    if t.size == 0:
        return t, v

    unique_t, inverse = np.unique(t, return_inverse=True)
    if unique_t.size == t.size:
        return t, v

    out = np.full(unique_t.shape, np.nan, dtype=float)
    for idx in range(unique_t.size):
        vals = v[inverse == idx]
        finite = vals[np.isfinite(vals)]
        out[idx] = float(np.mean(finite)) if finite.size else np.nan
    return clean_series(unique_t, out)


def aggregate_trace_values(arrays: list[np.ndarray], trace_kind: str, *, context: str) -> np.ndarray:
    if not arrays:
        raise RuntimeError(f"No arrays available to aggregate for {context}")
    kind = trace_kind.removeprefix("raw-")
    if kind == "single":
        return np.asarray(arrays[0], dtype=float)
    if kind == "midpoint":
        stacked = np.column_stack([np.asarray(arr, dtype=float) for arr in arrays])
        return np.nanmean(stacked, axis=1)
    if kind == "difference":
        if len(arrays) != 2:
            raise RuntimeError(f"{context} difference requires exactly two inputs")
        first = np.asarray(arrays[0], dtype=float)
        second = np.asarray(arrays[1], dtype=float)
        out = first - second
        out[~np.isfinite(first) | ~np.isfinite(second)] = np.nan
        return out
    raise ValueError(f"Unsupported trace kind for {context}: {trace_kind!r}")


def _infer_window_samples(t_ms: np.ndarray, window_s: float) -> int:
    if window_s <= 0.0:
        return 1
    t = np.asarray(t_ms, dtype=float)
    if t.size < 2:
        return 1
    dt = np.diff(t)
    finite = dt[np.isfinite(dt) & (dt > 0.0)]
    if finite.size == 0:
        return 1
    median_dt_s = float(np.median(finite) / 1000.0)
    if median_dt_s <= 0.0:
        return 1
    window = max(1, int(round(window_s / median_dt_s)))
    if window % 2 == 0:
        window += 1
    return window


def _rolling_stat(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1 or mode == "none":
        return arr.copy()
    radius = max(0, window // 2)
    out = np.full(arr.shape, np.nan, dtype=float)
    for idx in range(arr.size):
        lo = max(0, idx - radius)
        hi = min(arr.size, idx + radius + 1)
        chunk = arr[lo:hi]
        finite = chunk[np.isfinite(chunk)]
        if finite.size == 0:
            continue
        if mode == "rolling-mean":
            out[idx] = float(np.mean(finite))
        elif mode == "rolling-median":
            out[idx] = float(np.median(finite))
        else:
            raise ValueError(f"Unsupported rolling mode: {mode!r}")
    return out


def detrend_series(
    t_ms: np.ndarray,
    values: np.ndarray,
    mode: str,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if mode == "none":
        return arr.copy(), np.full(arr.shape, np.nan, dtype=float)
    window = _infer_window_samples(t_ms, window_s)
    trend = _rolling_stat(arr, window, mode)
    detrended = arr.copy()
    mask = np.isfinite(arr) & np.isfinite(trend)
    detrended[mask] = arr[mask] - trend[mask]
    detrended[~mask] = np.nan
    return detrended, trend


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return out
    mean = float(np.mean(arr[finite]))
    std = float(np.std(arr[finite]))
    if std <= 1e-12:
        return out
    out[finite] = (arr[finite] - mean) / std
    return out


def zscore_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_ref = np.isfinite(ref)
    if finite_ref.sum() < 2:
        return out
    mean = float(np.mean(ref[finite_ref]))
    std = float(np.std(ref[finite_ref]))
    if std <= 1e-12:
        return out
    finite = np.isfinite(arr)
    out[finite] = (arr[finite] - mean) / std
    return out


def standard_pearson(x: np.ndarray, y: np.ndarray, min_samples: int) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_samples:
        return np.nan, n
    x_valid = np.asarray(x[mask], dtype=float)
    y_valid = np.asarray(y[mask], dtype=float)
    if float(np.std(x_valid)) <= 1e-12 or float(np.std(y_valid)) <= 1e-12:
        return np.nan, n
    return float(np.corrcoef(x_valid, y_valid)[0, 1]), n


def weighted_corr(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    min_samples: int,
) -> tuple[float, float, int]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(w_arr) & (w_arr > 0.0)
    n = int(mask.sum())
    if n < min_samples:
        return np.nan, 0.0, n

    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    w_valid = w_arr[mask]
    w_sum = float(np.sum(w_valid))
    if w_sum <= 1e-12:
        return np.nan, 0.0, n

    n_eff = float(w_sum**2 / (np.sum(w_valid**2) + 1e-12))
    if n_eff < min_samples:
        return np.nan, n_eff, n

    x_mean = float(np.sum(w_valid * x_valid) / w_sum)
    y_mean = float(np.sum(w_valid * y_valid) / w_sum)
    x_centered = x_valid - x_mean
    y_centered = y_valid - y_mean
    cov = float(np.sum(w_valid * x_centered * y_centered) / w_sum)
    x_var = float(np.sum(w_valid * x_centered**2) / w_sum)
    y_var = float(np.sum(w_valid * y_centered**2) / w_sum)
    denom = float(np.sqrt(x_var * y_var))
    if denom <= 1e-12:
        return np.nan, n_eff, n
    return cov / denom, n_eff, n


def metric_from_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "absolute":
        return np.abs(scores)
    if mode == "positive":
        return scores.copy()
    if mode == "negative":
        return -scores
    raise ValueError(f"Unsupported correlation mode: {mode!r}")


def best_offset_from_scan(
    offsets_ms: np.ndarray,
    scores: np.ndarray,
    mode: str,
    peak_separation_ms: float,
) -> tuple[float, float, float, float, np.ndarray]:
    metrics = metric_from_scores(scores, mode)
    finite = np.isfinite(metrics)
    if not np.any(finite):
        raise RuntimeError("No finite offset scores; cannot estimate camera-to-Mocopi offset")

    best_idx = int(np.nanargmax(metrics))
    best_offset = float(offsets_ms[best_idx])
    best_score = float(scores[best_idx])
    best_metric = float(metrics[best_idx])

    competing = metrics.copy()
    competing[np.abs(offsets_ms - best_offset) <= peak_separation_ms] = np.nan
    second_best = float(np.nanmax(competing)) if np.any(np.isfinite(competing)) else 0.0
    peak_margin = max(0.0, best_metric - second_best)
    return best_offset, best_score, best_metric, peak_margin, metrics


def front_weights_for_mode(
    front_scores: np.ndarray,
    mode: str,
    threshold: float,
) -> np.ndarray:
    if mode == "none":
        return np.ones_like(front_scores, dtype=float)
    clipped = np.clip(np.asarray(front_scores, dtype=float), 0.0, 1.0)
    if mode == "soft":
        return clipped
    if mode == "hard":
        return (clipped >= threshold).astype(float)
    raise ValueError(f"Unsupported front weight mode: {mode!r}")


def apply_front_windows(
    t_front_ms: np.ndarray,
    scores: np.ndarray,
    windows_s: Sequence[Sequence[float]] | None,
) -> np.ndarray:
    if not windows_s:
        return np.asarray(scores, dtype=float)

    t_s = np.asarray(t_front_ms, dtype=float) / 1000.0
    window_mask = np.zeros(t_s.shape, dtype=bool)
    for start_s, end_s in windows_s:
        start = float(start_s)
        end = float(end_s)
        if end <= start:
            raise ValueError(f"Invalid --front-window {start:g} {end:g}: END_S must be greater than START_S")
        window_mask |= (t_s >= start) & (t_s <= end)

    gated = np.asarray(scores, dtype=float).copy()
    gated[~window_mask] = 0.0
    return gated


def apply_front_segment_selection(
    t_front_ms: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    mode: str,
) -> tuple[np.ndarray, float | None, float | None]:
    arr = np.asarray(scores, dtype=float)
    if mode == "all":
        return arr.copy(), None, None

    if mode != "first":
        raise ValueError(f"Unsupported front segment mode: {mode!r}")

    mask = np.isfinite(arr) & (arr >= threshold)
    if not np.any(mask):
        return np.zeros_like(arr, dtype=float), None, None

    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    first_start = int(starts[0])
    first_end = int(ends[0])

    gated = np.zeros_like(arr, dtype=float)
    gated[first_start : first_end + 1] = arr[first_start : first_end + 1]
    return (
        gated,
        float(t_front_ms[first_start] / 1000.0),
        float(t_front_ms[first_end] / 1000.0),
    )


def apply_front_segment_trims(
    t_front_ms: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    trim_start_s: float,
    trim_end_s: float,
    *,
    segment_mode: str,
) -> tuple[np.ndarray, float | None, float | None]:
    arr = np.asarray(scores, dtype=float)
    if trim_start_s < 0.0 or trim_end_s < 0.0:
        raise ValueError("Front segment trims must be non-negative")
    if trim_start_s == 0.0 and trim_end_s == 0.0:
        if segment_mode == "first":
            mask = np.isfinite(arr) & (arr >= threshold)
            if np.any(mask):
                starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
                ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
                return arr.copy(), float(t_front_ms[int(starts[0])] / 1000.0), float(t_front_ms[int(ends[0])] / 1000.0)
        return arr.copy(), None, None

    mask = np.isfinite(arr) & (arr >= threshold)
    if not np.any(mask):
        return np.zeros_like(arr, dtype=float), None, None

    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    gated = np.zeros_like(arr, dtype=float)
    effective_start_s: float | None = None
    effective_end_s: float | None = None

    for start_idx, end_idx in zip(starts, ends):
        start_time_s = float(t_front_ms[int(start_idx)] / 1000.0)
        end_time_s = float(t_front_ms[int(end_idx)] / 1000.0)
        kept_mask = (
            (t_front_ms >= (start_time_s + trim_start_s) * 1000.0)
            & (t_front_ms <= (end_time_s - trim_end_s) * 1000.0)
        )
        kept_mask &= np.arange(arr.size) >= int(start_idx)
        kept_mask &= np.arange(arr.size) <= int(end_idx)
        if not np.any(kept_mask):
            continue
        gated[kept_mask] = arr[kept_mask]
        if effective_start_s is None:
            kept_indices = np.flatnonzero(kept_mask)
            effective_start_s = float(t_front_ms[int(kept_indices[0])] / 1000.0)
            effective_end_s = float(t_front_ms[int(kept_indices[-1])] / 1000.0)

    return gated, effective_start_s, effective_end_s


def scan_weighted_offsets(
    t_ref_ms: np.ndarray,
    ref_values: np.ndarray,
    t_camera_ms: np.ndarray,
    camera_values: np.ndarray,
    camera_front_scores: np.ndarray,
    *,
    search_ms: float,
    step_ms: float,
    front_weight_mode: str,
    front_facing_threshold: float,
    correlation_mode: str,
    min_samples: int,
    peak_separation_ms: float,
    clip_start_s: float | None,
    clip_end_s: float | None,
) -> OffsetScanResult:
    offsets = np.arange(-search_ms, search_ms + step_ms, step_ms, dtype=float)
    scores = np.full(offsets.shape, np.nan, dtype=float)
    n_effective = np.zeros(offsets.shape, dtype=float)
    coverage = np.zeros(offsets.shape, dtype=float)

    ref_mask = np.isfinite(t_ref_ms) & np.isfinite(ref_values)
    if clip_start_s is not None:
        ref_mask &= t_ref_ms >= float(clip_start_s) * 1000.0
    if clip_end_s is not None:
        ref_mask &= t_ref_ms <= float(clip_end_s) * 1000.0

    for idx, offset in enumerate(offsets):
        t_camera_shifted = t_camera_ms + offset
        t_min = max(float(np.nanmin(t_ref_ms[ref_mask])), float(np.nanmin(t_camera_shifted)))
        t_max = min(float(np.nanmax(t_ref_ms[ref_mask])), float(np.nanmax(t_camera_shifted)))
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            continue

        mask = ref_mask & (t_ref_ms >= t_min) & (t_ref_ms <= t_max)
        if int(mask.sum()) < min_samples:
            continue

        t_overlap = t_ref_ms[mask]
        ref_overlap = ref_values[mask]
        camera_interp = np.interp(t_overlap, t_camera_shifted, camera_values, left=np.nan, right=np.nan)
        front_interp = np.interp(t_overlap, t_camera_shifted, camera_front_scores, left=0.0, right=0.0)
        weights = front_weights_for_mode(front_interp, front_weight_mode, front_facing_threshold)

        score, n_eff, n_raw = weighted_corr(ref_overlap, camera_interp, weights, min_samples)
        scores[idx] = score
        n_effective[idx] = n_eff
        coverage[idx] = float(n_raw / max(1, int(mask.sum())))

    best_offset, best_score, best_metric, peak_margin, metrics = best_offset_from_scan(
        offsets,
        scores,
        correlation_mode,
        peak_separation_ms,
    )
    return OffsetScanResult(
        offsets_ms=offsets,
        scores=scores,
        metrics=metrics,
        n_effective=n_effective,
        coverage=coverage,
        best_offset_ms=best_offset,
        best_score=best_score,
        best_metric=best_metric,
        peak_margin=peak_margin,
    )


def compute_front_facing_scores(image_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pose = image_df[image_df["source"] == "pose"].copy()
    if "coordinate_space" in pose.columns:
        pose = pose[pose["coordinate_space"].fillna("image") == "image"].copy()
    if pose.empty:
        raise RuntimeError("Image CSV contains no image-space pose rows for front-facing scoring")

    grouped_time = pose.groupby("frame", sort=True)["timestamp_ms"].first()
    frames = grouped_time.index.to_numpy()
    timestamps = grouped_time.to_numpy(dtype=float)

    x_table = pose.pivot_table(index="frame", columns="landmark_name", values="x", aggfunc="mean").reindex(frames)

    def pair_x_series(left_name: str, right_name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if left_name not in x_table.columns or right_name not in x_table.columns:
            return None, None
        return (
            x_table[left_name].to_numpy(dtype=float),
            x_table[right_name].to_numpy(dtype=float),
        )

    def pair_order_score(left_x: np.ndarray | None, right_x: np.ndarray | None) -> np.ndarray:
        if left_x is None or right_x is None:
            return np.full(frames.shape, np.nan, dtype=float)
        dx = np.asarray(left_x, dtype=float) - np.asarray(right_x, dtype=float)
        finite = dx[np.isfinite(dx)]
        if finite.size == 0:
            return np.full(frames.shape, np.nan, dtype=float)
        scale = float(np.median(np.abs(finite)))
        scale = max(scale * 0.20, 1e-6)
        return 0.5 * (1.0 + np.tanh(dx / scale))

    def pair_span_score(left_x: np.ndarray | None, right_x: np.ndarray | None) -> np.ndarray:
        if left_x is None or right_x is None:
            return np.full(frames.shape, np.nan, dtype=float)
        span = np.abs(np.asarray(left_x, dtype=float) - np.asarray(right_x, dtype=float))
        finite = span[np.isfinite(span) & (span > 1e-9)]
        if finite.size == 0:
            return np.full(frames.shape, np.nan, dtype=float)
        median_span = float(np.median(finite))
        return np.clip(span / max(median_span, 1e-9), 0.0, 1.0)

    left_shoulder_x, right_shoulder_x = pair_x_series("LEFT_SHOULDER", "RIGHT_SHOULDER")
    left_hip_x, right_hip_x = pair_x_series("LEFT_HIP", "RIGHT_HIP")

    shoulder_order = pair_order_score(left_shoulder_x, right_shoulder_x)
    shoulder_span = pair_span_score(left_shoulder_x, right_shoulder_x)
    hip_order = pair_order_score(left_hip_x, right_hip_x)
    hip_span = pair_span_score(left_hip_x, right_hip_x)

    components = [
        (np.clip(shoulder_order, 0.0, 1.0), 0.50),
        (np.clip(shoulder_span, 0.0, 1.0), 0.20),
        (np.clip(hip_order, 0.0, 1.0), 0.20),
        (np.clip(hip_span, 0.0, 1.0), 0.10),
    ]
    numerator = np.zeros(frames.shape, dtype=float)
    denominator = np.zeros(frames.shape, dtype=float)
    for values, weight in components:
        finite = np.isfinite(values)
        numerator[finite] += weight * values[finite]
        denominator[finite] += weight

    scores = np.zeros(frames.shape, dtype=float)
    valid = denominator > 0.0
    scores[valid] = numerator[valid] / denominator[valid]
    return timestamps, np.clip(scores, 0.0, 1.0)


def camera_landmark_component(
    projection,
    landmark_name: str,
    component: str,
    camera_space: str,
    *,
    raw: bool,
) -> np.ndarray:
    if raw:
        if component == "z":
            if landmark_name not in projection.image_depths:
                raise RuntimeError(f"Camera projection did not include landmark depth: {landmark_name}")
            return projection.image_depths[landmark_name].copy()
        if landmark_name not in projection.image_points:
            raise RuntimeError(f"Camera projection did not include landmark point: {landmark_name}")
        values = projection.image_points[landmark_name][:, component_index(component)].copy()
        if camera_space == "image" and component == "y":
            values *= -1.0
        return values

    if landmark_name not in projection.positions:
        raise RuntimeError(f"Camera projection did not include landmark position: {landmark_name}")
    return projection.positions[landmark_name][:, component_index(component)].copy()


def extract_mocopi_trace(
    seq,
    trace_kind: str,
    joints: Sequence[str],
    *,
    mocopi_component: str,
    com_joints: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    t_ms = seq.timestamps_ms()
    comp_idx = component_index(mocopi_component)

    if trace_kind == "com":
        positions = compute_center_of_mass(seq, com_joints)
        return clean_series(t_ms, positions[:, comp_idx].astype(float))

    arrays = []
    missing = []
    for joint_name in joints:
        if joint_name not in seq.joint_names:
            missing.append(joint_name)
            continue
        arrays.append(seq.joint_positions(joint_name)[:, comp_idx].astype(float))
    if missing:
        raise RuntimeError(f"Mocopi sequence is missing joints: {', '.join(missing)}")

    values = aggregate_trace_values(arrays, trace_kind, context="Mocopi trace")
    return clean_series(t_ms, values)


def extract_camera_trace(
    df: pd.DataFrame,
    origin_landmarks: Sequence[str],
    camera_landmarks: Sequence[str],
    *,
    camera_trace: str,
    camera_space: str,
    camera_component: str,
    visibility_threshold: float | None,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    projection_landmarks = list(dict.fromkeys([*origin_landmarks, *camera_landmarks]))
    projection = compute_camera_projection(
        df,
        projection_landmarks,
        CameraProjectionConfig(
            space=camera_space,
            visibility_threshold=visibility_threshold,
            smooth_window=smooth_window,
            rotate_to_body_frame=False,
            origin_landmarks=tuple(origin_landmarks),
        ),
    )

    if camera_trace == "origin" and camera_component == "z":
        values = projection.origin_z.copy()
    elif camera_trace == "origin":
        values = projection.origin_xy[:, component_index(camera_component)].copy()
        if camera_space == "image" and camera_component == "y":
            values *= -1.0
    else:
        arrays = []
        raw = camera_trace.startswith("raw-")
        for landmark_name in camera_landmarks:
            arrays.append(
                camera_landmark_component(
                    projection,
                    landmark_name,
                    camera_component,
                    camera_space,
                    raw=raw,
                )
            )
        values = aggregate_trace_values(arrays, camera_trace, context="camera trace")
    return clean_series(projection.timestamps_ms, values)


def build_default_paths(args: argparse.Namespace, mocopi_component: str, camera_component: str) -> tuple[Path, Path, Path, Path]:
    camera_stem = landmark_stem_from_image_csv(args.camera_csv)
    motion_stem = args.motion_source.stem or "mocopi"
    trace_label = args.camera_trace
    if args.camera_trace != "origin":
        trace_label += "_" + "-".join(args.camera_landmarks)
    mocopi_label = args.mocopi_trace
    if args.mocopi_trace != "com":
        mocopi_label += "_" + "-".join(args.mocopi_joints)
    label_parts = [
        motion_stem,
        camera_stem,
        args.camera_space,
        trace_label,
        f"mocopi-{mocopi_label}",
        f"c{camera_component}",
        f"m{mocopi_component}",
    ]
    if args.sync_signal != "trace":
        label_parts.append("sync-lrdiff")
    if args.detrend != "none":
        label_parts.append(f"dt-{args.detrend}-{args.detrend_window_s:g}s")
    label = "_".join(label_parts)
    output_dir = args.output_dir
    candidate_trace = output_dir / f"{label}_aligned_traces.csv"
    candidate_abs_len = len(str((Path.cwd() / candidate_trace).resolve(strict=False)))
    if len(label) > 90 or candidate_abs_len > 220:
        digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
        label = "_".join(
            [
                motion_stem,
                camera_stem,
                args.camera_space,
                f"{digest}",
                f"c{camera_component}",
                f"m{mocopi_component}",
            ]
        )
    trace_output = args.trace_output or output_dir / f"{label}_aligned_traces.csv"
    summary_output = args.summary_output or output_dir / f"{label}_summary.csv"
    scan_output = args.scan_output or output_dir / f"{label}_offset_scan.csv"
    plot_output = args.plot_output or output_dir / f"{label}_aligned_traces.png"
    return trace_output, summary_output, scan_output, plot_output


def describe_mocopi_trace(trace_kind: str, mocopi_joints: Sequence[str]) -> str:
    if trace_kind == "com":
        return "COM"
    if trace_kind == "single":
        return f"single {mocopi_joints[0]}"
    return f"{trace_kind} {'/'.join(mocopi_joints)}"


def describe_camera_trace(trace_kind: str, camera_landmarks: Sequence[str]) -> str:
    if trace_kind == "origin":
        return "origin"
    if trace_kind == "single":
        return f"single {camera_landmarks[0]}"
    return f"{trace_kind} {'/'.join(camera_landmarks)}"


def align_series_pair(
    t_ref_ms: np.ndarray,
    ref_values: np.ndarray,
    t_camera_ms: np.ndarray,
    camera_values: np.ndarray,
    offset_ms: float,
    *,
    clip_start_s: float | None,
    clip_end_s: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_camera_aligned = t_camera_ms + offset_ms
    t_min = max(float(np.nanmin(t_ref_ms)), float(np.nanmin(t_camera_aligned)))
    t_max = min(float(np.nanmax(t_ref_ms)), float(np.nanmax(t_camera_aligned)))
    if clip_start_s is not None:
        t_min = max(t_min, float(clip_start_s) * 1000.0)
    if clip_end_s is not None:
        t_max = min(t_max, float(clip_end_s) * 1000.0)
    if t_max <= t_min:
        raise RuntimeError("No overlapping time window after applying offset")

    mask = (t_ref_ms >= t_min) & (t_ref_ms <= t_max) & np.isfinite(ref_values)
    t_overlap = t_ref_ms[mask]
    ref_overlap = ref_values[mask]
    camera_overlap = np.interp(t_overlap, t_camera_aligned, camera_values, left=np.nan, right=np.nan)
    return t_overlap, ref_overlap, camera_overlap


def align_traces(
    t_m_ms: np.ndarray,
    mocopi_values: np.ndarray,
    t_camera_ms: np.ndarray,
    camera_values: np.ndarray,
    t_front_ms: np.ndarray,
    front_scores: np.ndarray,
    offset_ms: float,
    *,
    front_facing_threshold: float,
    clip_start_s: float | None,
    clip_end_s: float | None,
) -> AlignmentResult:
    t_camera_aligned = t_camera_ms + offset_ms
    t_overlap, mocopi_overlap, camera_overlap = align_series_pair(
        t_m_ms,
        mocopi_values,
        t_camera_ms,
        camera_values,
        offset_ms,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
    )
    front_overlap = np.interp(t_overlap, t_front_ms + offset_ms, front_scores, left=0.0, right=0.0)
    return AlignmentResult(
        time_s=t_overlap / 1000.0,
        mocopi_trace=mocopi_overlap,
        camera_trace=camera_overlap,
        front_facing_score=front_overlap,
        front_facing_used=front_overlap >= front_facing_threshold,
        offset_ms=offset_ms,
        full_mocopi_time_s=t_m_ms / 1000.0,
        full_mocopi_trace=mocopi_values,
        full_camera_time_s=t_camera_aligned / 1000.0,
        full_camera_trace=camera_values,
        full_front_time_s=(t_front_ms + offset_ms) / 1000.0,
        full_front_score=front_scores,
    )


def save_outputs(
    alignment: AlignmentResult,
    aligned_mocopi_eval: np.ndarray,
    aligned_camera_eval: np.ndarray,
    full_mocopi_eval_trace: np.ndarray,
    full_camera_eval_trace: np.ndarray,
    aligned_sync_mocopi: np.ndarray,
    aligned_sync_camera: np.ndarray,
    scan: OffsetScanResult | None,
    args: argparse.Namespace,
    *,
    mocopi_component: str,
    camera_component: str,
    camera_space: str,
    mocopi_trace_label: str,
    camera_trace_label: str,
    sync_trace_label: str,
    front_segment_start_s: float | None,
    front_segment_end_s: float | None,
    trace_output: Path,
    summary_output: Path,
    scan_output: Path,
    plot_output: Path,
) -> None:
    pearson_all, n_all = standard_pearson(aligned_mocopi_eval, aligned_camera_eval, args.min_samples)
    front_mask = (
        np.isfinite(aligned_mocopi_eval)
        & np.isfinite(aligned_camera_eval)
        & alignment.front_facing_used
    )
    pearson_front, n_front = standard_pearson(
        aligned_mocopi_eval[front_mask],
        aligned_camera_eval[front_mask],
        args.min_samples,
    )
    weighted_r, n_eff, n_weighted_raw = weighted_corr(
        aligned_mocopi_eval,
        aligned_camera_eval,
        front_weights_for_mode(
            alignment.front_facing_score,
            "soft",
            args.front_facing_threshold,
        ),
        args.min_samples,
    )

    trace_df = pd.DataFrame(
        {
            "time_s": alignment.time_s,
            "mocopi_trace": alignment.mocopi_trace,
            "camera_trace": alignment.camera_trace,
            "mocopi_eval_trace": aligned_mocopi_eval,
            "camera_eval_trace": aligned_camera_eval,
            "mocopi_zscore": zscore(alignment.mocopi_trace),
            "camera_zscore": zscore(alignment.camera_trace),
            "mocopi_eval_zscore": zscore(aligned_mocopi_eval),
            "camera_eval_zscore": zscore(aligned_camera_eval),
            "sync_mocopi_trace": aligned_sync_mocopi,
            "sync_camera_trace": aligned_sync_camera,
            "front_facing_score": alignment.front_facing_score,
            "front_facing_used": alignment.front_facing_used,
            "offset_ms": alignment.offset_ms,
        }
    )
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_output, index=False)

    if scan is not None:
        scan_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "offset_ms": scan.offsets_ms,
                "signed_correlation": scan.scores,
                "selection_metric": scan.metrics,
                "n_effective": scan.n_effective,
                "coverage": scan.coverage,
            }
        ).to_csv(scan_output, index=False)
    else:
        scan_output = Path("")

    finite_front = np.isfinite(alignment.front_facing_score)
    front_coverage = (
        float(np.mean(alignment.front_facing_used[finite_front])) if np.any(finite_front) else np.nan
    )
    offset_at_search_boundary = False
    if scan is not None:
        boundary_distance = min(
            abs(scan.best_offset_ms - float(np.nanmin(scan.offsets_ms))),
            abs(scan.best_offset_ms - float(np.nanmax(scan.offsets_ms))),
        )
        offset_at_search_boundary = bool(boundary_distance <= max(1e-9, args.step_ms))
    summary = {
        "motion_source": str(args.motion_source),
        "camera_csv": str(args.camera_csv),
        "world_csv": str(args.world_csv) if args.world_csv is not None else "",
        "camera_space": camera_space,
        "mocopi_trace": args.mocopi_trace,
        "mocopi_joints": " ".join(args.mocopi_joints),
        "camera_trace": args.camera_trace,
        "camera_landmarks": " ".join(args.camera_landmarks),
        "origin_landmarks": " ".join(args.origin_landmarks),
        "mocopi_trace_label": mocopi_trace_label,
        "camera_trace_label": camera_trace_label,
        "sync_signal": args.sync_signal,
        "sync_trace_label": sync_trace_label,
        "sync_mocopi_joints": " ".join(args.sync_mocopi_joints or []),
        "sync_camera_landmarks": " ".join(args.sync_camera_landmarks or []),
        "detrend": args.detrend,
        "detrend_window_s": args.detrend_window_s,
        "camera_component": camera_component,
        "mocopi_component": mocopi_component,
        "offset_ms": alignment.offset_ms,
        "offset_source": "fixed" if args.offset_ms is not None else "estimated",
        "offset_score": scan.best_score if scan is not None else np.nan,
        "offset_metric": scan.best_metric if scan is not None else np.nan,
        "offset_peak_margin": scan.peak_margin if scan is not None else np.nan,
        "offset_at_search_boundary": offset_at_search_boundary,
        "front_weight_mode": args.front_weight_mode,
        "front_facing_threshold": args.front_facing_threshold,
        "front_windows_s": repr(args.front_window or []),
        "front_segment_trim_start_s": args.front_segment_trim_start_s,
        "front_segment_trim_end_s": args.front_segment_trim_end_s,
        "front_segment": args.front_segment,
        "front_segment_start_s": front_segment_start_s,
        "front_segment_end_s": front_segment_end_s,
        "front_coverage": front_coverage,
        "pearson_r_all": pearson_all,
        "n_all": n_all,
        "pearson_r_front": pearson_front,
        "n_front": n_front,
        "weighted_r_front_score": weighted_r,
        "n_effective_front_score": n_eff,
        "n_weighted_raw": n_weighted_raw,
        "trace_output": str(trace_output),
        "scan_output": str(scan_output),
        "plot_output": str(plot_output),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(summary_output, index=False)

    plot_alignment(
        alignment,
        scan,
        plot_output,
        pearson_all=pearson_all,
        pearson_front=pearson_front,
        weighted_r=weighted_r,
        mocopi_component=mocopi_component,
        camera_component=camera_component,
        camera_space=camera_space,
        mocopi_trace_label=mocopi_trace_label,
        camera_trace_label=camera_trace_label,
        aligned_mocopi_eval=aligned_mocopi_eval,
        aligned_camera_eval=aligned_camera_eval,
        full_mocopi_eval_trace=full_mocopi_eval_trace,
        full_camera_eval_trace=full_camera_eval_trace,
        detrend_label=args.detrend,
        sync_trace_label=sync_trace_label,
        front_facing_threshold=args.front_facing_threshold,
    )

    print(f"Aligned trace CSV: {trace_output}")
    print(f"Summary CSV: {summary_output}")
    if scan is not None:
        print(f"Offset scan CSV: {scan_output}")
        if offset_at_search_boundary:
            print("Warning: selected offset is at the search boundary; consider increasing --search-ms.")
    print(f"Aligned trace plot: {plot_output}")


def plot_alignment(
    alignment: AlignmentResult,
    scan: OffsetScanResult | None,
    output_path: Path,
    *,
    pearson_all: float,
    pearson_front: float,
    weighted_r: float,
    mocopi_component: str,
    camera_component: str,
    camera_space: str,
    mocopi_trace_label: str,
    camera_trace_label: str,
    aligned_mocopi_eval: np.ndarray,
    aligned_camera_eval: np.ndarray,
    full_mocopi_eval_trace: np.ndarray,
    full_camera_eval_trace: np.ndarray,
    detrend_label: str,
    sync_trace_label: str,
    front_facing_threshold: float,
) -> None:
    n_rows = 3 if scan is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 8 if scan is not None else 6), sharex=False)
    axes = list(np.atleast_1d(axes))

    ax_full = axes[0]
    aligned_start_s = float(np.nanmin(alignment.time_s))
    aligned_end_s = float(np.nanmax(alignment.time_s))
    full_x_min = float(
        np.nanmin(
            [
                np.nanmin(alignment.full_mocopi_time_s),
                np.nanmin(alignment.full_camera_time_s),
            ]
        )
    )
    full_x_max = float(
        np.nanmax(
            [
                np.nanmax(alignment.full_mocopi_time_s),
                np.nanmax(alignment.full_camera_time_s),
            ]
        )
    )
    full_mocopi_z = zscore(alignment.full_mocopi_trace)
    full_camera_z = zscore(alignment.full_camera_trace)
    ax_full.plot(
        alignment.full_mocopi_time_s,
        full_mocopi_z,
        label=f"Mocopi {mocopi_trace_label} {mocopi_component} full trace",
        color="#1f77b4",
        linewidth=1.0,
        alpha=0.85,
    )
    ax_full.plot(
        alignment.full_camera_time_s,
        full_camera_z,
        label=f"Camera {camera_space} {camera_component} {camera_trace_label} full trace, offset applied",
        color="#d62728",
        linewidth=1.0,
        alpha=0.75,
    )
    y_min, y_max = ax_full.get_ylim()
    ax_full.axvspan(
        aligned_start_s,
        aligned_end_s,
        color="#ffbf00",
        alpha=0.18,
        label="aligned window used below",
    )
    ax_full.set_ylim(y_min, y_max)
    ax_full.set_xlim(full_x_min, full_x_max)
    ax_full.set_ylabel("Full trace (z-score)")
    ax_full.set_xlabel("Time (s, Mocopi timeline after offset)")
    ax_full.set_title("Full camera and Mocopi output traces")
    ax_full.grid(alpha=0.3)
    ax_full_front = ax_full.twinx()
    ax_full_front.plot(
        alignment.full_front_time_s,
        alignment.full_front_score,
        color="#2ca02c",
        linewidth=1.0,
        alpha=0.85,
        label="front_facing_score",
    )
    ax_full_front.axhline(
        front_facing_threshold,
        color="#2ca02c",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label=f"front threshold = {front_facing_threshold:g}",
    )
    ax_full_front.set_ylim(0.0, 1.05)
    ax_full_front.set_ylabel("Front-facing score", color="#2ca02c")
    ax_full_front.tick_params(axis="y", colors="#2ca02c")
    lines_left, labels_left = ax_full.get_legend_handles_labels()
    lines_right, labels_right = ax_full_front.get_legend_handles_labels()
    ax_full.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right", fontsize=8)

    ax = axes[1]

    t = alignment.time_s
    full_mocopi_eval_z = zscore(full_mocopi_eval_trace)
    full_camera_eval_z = zscore(full_camera_eval_trace)
    mocopi_z = zscore_against(aligned_mocopi_eval, full_mocopi_eval_trace)
    camera_z = zscore_against(aligned_camera_eval, full_camera_eval_trace)
    ax.plot(
        alignment.full_mocopi_time_s,
        full_mocopi_eval_z,
        label="Mocopi full eval trace",
        color="#1f77b4",
        linewidth=0.8,
        alpha=0.22,
    )
    ax.plot(
        alignment.full_camera_time_s,
        full_camera_eval_z,
        label="Camera full eval trace, offset applied",
        color="#d62728",
        linewidth=0.8,
        alpha=0.22,
    )
    ax.axvspan(
        aligned_start_s,
        aligned_end_s,
        color="#ffbf00",
        alpha=0.12,
        label="aligned window",
    )
    ax.plot(t, mocopi_z, label=f"Mocopi {mocopi_trace_label} {mocopi_component} (eval z-score)", color="#1f77b4", linewidth=1.4)
    ax.plot(
        t,
        camera_z,
        label=f"Camera {camera_trace_label} {camera_space} {camera_component} (eval z-score)",
        color="#d62728",
        linewidth=1.2,
        alpha=0.85,
    )
    if np.any(alignment.front_facing_used):
        y_min, y_max = ax.get_ylim()
        ax.fill_between(
            t,
            y_min,
            y_max,
            where=alignment.front_facing_used,
            color="#2ca02c",
            alpha=0.10,
            step="mid",
            label="front-facing/high-quality mask",
        )
        ax.set_ylim(y_min, y_max)
    ax.set_xlim(full_x_min, full_x_max)
    ax.set_ylabel("Trace (z-score)")
    ax.set_xlabel("Time (s, Mocopi timeline)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        "Aligned eval traces: "
        f"offset={alignment.offset_ms:.1f} ms, "
        f"Pearson r(front)={pearson_front:.3f}, "
        f"r(all)={pearson_all:.3f}, "
        f"weighted r={weighted_r:.3f}, "
        f"detrend={detrend_label}, "
        f"sync={sync_trace_label}"
    )

    if scan is not None:
        ax_scan = axes[2]
        ax_scan.plot(scan.offsets_ms, scan.scores, label="signed correlation", color="#444444", linewidth=1.0)
        ax_scan.plot(scan.offsets_ms, scan.metrics, label="selection metric", color="#9467bd", linewidth=1.0, alpha=0.8)
        ax_scan.axvline(scan.best_offset_ms, color="#d62728", linestyle="--", linewidth=1.2, label="selected offset")
        ax_scan.set_xlabel("Camera offset applied to timestamps (ms)")
        ax_scan.set_ylabel("Cross-correlation")
        ax_scan.grid(alpha=0.3)
        ax_scan.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_component = args.component or ("z" if args.camera_space == "world" else "y")
    mocopi_component = args.mocopi_component or base_component
    camera_component = args.camera_component or base_component
    mocopi_trace_label = describe_mocopi_trace(args.mocopi_trace, args.mocopi_joints)
    camera_trace_label = describe_camera_trace(args.camera_trace, args.camera_landmarks)

    views = load_landmark_views(
        args.camera_csv,
        world_csv=args.world_csv,
        require_world=args.camera_space == "world",
    )
    args.world_csv = views.world_csv
    projection_df = views.world_df if args.camera_space == "world" else views.image_df
    if projection_df is None:
        raise SystemExit("World-space camera projection requested, but no pose_world CSV was found")

    seq = load_mocopi_recording(args.motion_source)
    t_m_ms, mocopi_values = extract_mocopi_trace(
        seq,
        args.mocopi_trace,
        args.mocopi_joints,
        mocopi_component=mocopi_component,
        com_joints=args.mocopi_com_joints,
    )
    if args.invert_mocopi:
        mocopi_values *= -1.0
    if t_m_ms.size < args.min_samples:
        raise SystemExit("Not enough finite Mocopi trace samples")
    mocopi_eval_values, _ = detrend_series(t_m_ms, mocopi_values, args.detrend, args.detrend_window_s)

    t_camera_ms, camera_values = extract_camera_trace(
        projection_df,
        args.origin_landmarks,
        args.camera_landmarks,
        camera_trace=args.camera_trace,
        camera_space=args.camera_space,
        camera_component=camera_component,
        visibility_threshold=args.visibility_threshold,
        smooth_window=args.smooth_window,
    )
    if args.invert_camera:
        camera_values *= -1.0
    if t_camera_ms.size < args.min_samples:
        raise SystemExit("Not enough finite camera trace samples")
    camera_eval_values, _ = detrend_series(t_camera_ms, camera_values, args.detrend, args.detrend_window_s)

    t_front_ms, front_scores = compute_front_facing_scores(views.image_df)
    t_front_ms, front_scores = clean_series(t_front_ms, front_scores)
    if t_front_ms.size < args.min_samples:
        raise SystemExit("Not enough image-space pose samples for front-facing scoring")
    try:
        front_scores = apply_front_windows(t_front_ms, front_scores, args.front_window)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    front_scores, front_segment_start_s, front_segment_end_s = apply_front_segment_selection(
        t_front_ms,
        front_scores,
        args.front_facing_threshold,
        args.front_segment,
    )
    try:
        front_scores, front_segment_start_s, front_segment_end_s = apply_front_segment_trims(
            t_front_ms,
            front_scores,
            args.front_facing_threshold,
            args.front_segment_trim_start_s,
            args.front_segment_trim_end_s,
            segment_mode=args.front_segment,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.front_segment != "all":
        if front_segment_start_s is None or front_segment_end_s is None:
            print("Warning: no connected front-score segment passed threshold; front mask is empty.")
        else:
            print(
                "Selected first front-score segment: "
                f"{front_segment_start_s:.2f}s to {front_segment_end_s:.2f}s"
            )

    sync_trace_label = "report trace"
    if args.sync_signal == "trace":
        t_sync_m_ms = t_m_ms
        mocopi_sync_values = mocopi_eval_values
        t_sync_camera_ms = t_camera_ms
        camera_sync_values = camera_eval_values
    else:
        sync_mocopi_joints = tuple(args.sync_mocopi_joints or DEFAULT_SYNC_MOCOPI_JOINTS)
        sync_camera_landmarks = tuple(args.sync_camera_landmarks or DEFAULT_SYNC_CAMERA_LANDMARKS)
        t_sync_m_ms, mocopi_sync_raw = extract_mocopi_trace(
            seq,
            "difference",
            sync_mocopi_joints,
            mocopi_component=mocopi_component,
            com_joints=args.mocopi_com_joints,
        )
        if args.invert_mocopi:
            mocopi_sync_raw *= -1.0
        sync_camera_trace = "raw-difference" if args.camera_trace.startswith("raw-") else "difference"
        t_sync_camera_ms, camera_sync_raw = extract_camera_trace(
            projection_df,
            args.origin_landmarks,
            sync_camera_landmarks,
            camera_trace=sync_camera_trace,
            camera_space=args.camera_space,
            camera_component=camera_component,
            visibility_threshold=args.visibility_threshold,
            smooth_window=args.smooth_window,
        )
        if args.invert_camera:
            camera_sync_raw *= -1.0
        mocopi_sync_values, _ = detrend_series(t_sync_m_ms, mocopi_sync_raw, args.detrend, args.detrend_window_s)
        camera_sync_values, _ = detrend_series(t_sync_camera_ms, camera_sync_raw, args.detrend, args.detrend_window_s)
        args.sync_mocopi_joints = list(sync_mocopi_joints)
        args.sync_camera_landmarks = list(sync_camera_landmarks)
        sync_trace_label = f"left-right difference {sync_mocopi_joints[0]}-{sync_mocopi_joints[1]} / {sync_camera_landmarks[0]}-{sync_camera_landmarks[1]}"

    t_m_scan, mocopi_scan = resample_feature(t_sync_m_ms, mocopi_sync_values, args.rate_hz)
    t_camera_scan, camera_scan = resample_feature(t_sync_camera_ms, camera_sync_values, args.rate_hz)
    front_scan = np.interp(t_camera_scan, t_front_ms, front_scores, left=0.0, right=0.0)

    scan = None
    if args.offset_ms is None:
        scan = scan_weighted_offsets(
            t_m_scan,
            mocopi_scan,
            t_camera_scan,
            camera_scan,
            front_scan,
            search_ms=args.search_ms,
            step_ms=args.step_ms,
            front_weight_mode=args.front_weight_mode,
            front_facing_threshold=args.front_facing_threshold,
            correlation_mode=args.correlation_mode,
            min_samples=args.min_samples,
            peak_separation_ms=args.peak_separation_ms,
            clip_start_s=args.clip_start,
            clip_end_s=args.clip_end,
        )
        offset_ms = scan.best_offset_ms
        print(
            "Estimated offset: "
            f"{offset_ms:.1f} ms "
            f"(score={scan.best_score:.3f}, metric={scan.best_metric:.3f}, "
            f"peak_margin={scan.peak_margin:.3f})"
        )
    else:
        offset_ms = float(args.offset_ms)
        print(f"Using fixed offset: {offset_ms:.1f} ms")

    alignment = align_traces(
        t_m_ms,
        mocopi_values,
        t_camera_ms,
        camera_values,
        t_front_ms,
        front_scores,
        offset_ms,
        front_facing_threshold=args.front_facing_threshold,
        clip_start_s=args.clip_start,
        clip_end_s=args.clip_end,
    )
    _, aligned_mocopi_eval, aligned_camera_eval = align_series_pair(
        t_m_ms,
        mocopi_eval_values,
        t_camera_ms,
        camera_eval_values,
        offset_ms,
        clip_start_s=args.clip_start,
        clip_end_s=args.clip_end,
    )
    aligned_sync_mocopi = np.interp(
        alignment.time_s * 1000.0,
        t_sync_m_ms,
        mocopi_sync_values,
        left=np.nan,
        right=np.nan,
    )
    aligned_sync_camera = np.interp(
        alignment.time_s * 1000.0,
        t_sync_camera_ms + offset_ms,
        camera_sync_values,
        left=np.nan,
        right=np.nan,
    )

    trace_output, summary_output, scan_output, plot_output = build_default_paths(
        args,
        mocopi_component,
        camera_component,
    )
    save_outputs(
        alignment,
        aligned_mocopi_eval,
        aligned_camera_eval,
        mocopi_eval_values,
        camera_eval_values,
        aligned_sync_mocopi,
        aligned_sync_camera,
        scan,
        args,
        mocopi_component=mocopi_component,
        camera_component=camera_component,
        camera_space=args.camera_space,
        mocopi_trace_label=mocopi_trace_label,
        camera_trace_label=camera_trace_label,
        sync_trace_label=sync_trace_label,
        front_segment_start_s=front_segment_start_s,
        front_segment_end_s=front_segment_end_s,
        trace_output=trace_output,
        summary_output=summary_output,
        scan_output=scan_output,
        plot_output=plot_output,
    )


if __name__ == "__main__":
    main()
