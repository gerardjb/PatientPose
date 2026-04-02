from __future__ import annotations

"""
Reliability utilities for Mocopi vs camera (MediaPipe) comparisons.

This module centralizes scale normalization, offset estimation, and per-frame
error exports that were previously duplicated across scripts.
"""

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .features import (
    NoCameraPoseDataError,
    compute_camera_egocentric_positions,
    compute_egocentric_positions,
)
from .sync import clean_feature_samples, estimate_camera_to_mocopi_offset
from .recording_io import load_mocopi_recording

SCALE_REF_JOINTS = ["l_up_leg", "r_up_leg", "l_shoulder", "r_shoulder"]
RELIABILITY_COLUMNS = [
    "time_s",
    "joint",
    "landmark",
    "error_2d",
    "mocopi_dx",
    "mocopi_dy",
    "camera_dx",
    "camera_dy",
]


def compute_body_scale_series(
    mocopi_pos: dict[str, np.ndarray],
    ref_names: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Return a per-frame body scale so Mocopi matches the camera normalization.
    """
    if ref_names is None:
        ref_names = SCALE_REF_JOINTS
    ref_arrays = [mocopi_pos[name][:, :2] for name in ref_names if name in mocopi_pos]
    if not ref_arrays:
        ref_arrays = [arr[:, :2] for arr in mocopi_pos.values()]
    if not ref_arrays:
        raise RuntimeError("No Mocopi joints available to compute scale")

    ref_stack = np.stack(ref_arrays, axis=0)
    scales = np.linalg.norm(ref_stack, axis=2).mean(axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    return scales


def export_reliability_errors(
    seq,
    cam_df: pd.DataFrame,
    joints: Sequence[str],
    landmarks: Sequence[str],
    offset_ms: float,
) -> pd.DataFrame:
    """
    Compute per-frame Mocopi vs camera egocentric errors for joint/landmark pairs.
    """
    if len(joints) != len(landmarks):
        raise ValueError("Expected joints and landmarks to have the same length")

    request_joints = list(dict.fromkeys([*joints, *SCALE_REF_JOINTS]))
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, request_joints)
    t_c_ms, camera_pos = compute_camera_egocentric_positions(cam_df, landmarks)

    t_c_aligned_ms = t_c_ms + offset_ms
    scales = compute_body_scale_series(mocopi_pos)

    records: list[dict] = []

    for j_name, lm_name in zip(joints, landmarks):
        if j_name not in mocopi_pos or lm_name not in camera_pos:
            continue

        m_traj = mocopi_pos[j_name]
        c_traj = camera_pos[lm_name]

        cx = np.interp(t_m_ms, t_c_aligned_ms, c_traj[:, 0], left=np.nan, right=np.nan)
        cy = np.interp(t_m_ms, t_c_aligned_ms, c_traj[:, 1], left=np.nan, right=np.nan)

        mx = m_traj[:, 0] / scales
        my = m_traj[:, 1] / scales

        dx = mx - cx
        dy = my - cy
        err = np.sqrt(dx**2 + dy**2)

        for t_ms, err_i, mx_i, my_i, cx_i, cy_i in zip(t_m_ms, err, mx, my, cx, cy):
            if np.isnan(err_i):
                continue
            records.append(
                {
                    "time_s": t_ms / 1000.0,
                    "joint": j_name,
                    "landmark": lm_name,
                    "error_2d": float(err_i),
                    "mocopi_dx": float(mx_i),
                    "mocopi_dy": float(my_i),
                    "camera_dx": float(cx_i),
                    "camera_dy": float(cy_i),
                }
            )

    return pd.DataFrame.from_records(records)


def ensure_reliability_csv(
    motion_source: Path,
    camera_csv: Path,
    output_csv: Path,
    offset_ms: float | None,
    search_ms: float,
    rate_hz: float,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> Path:
    """
    Guarantee a reliability CSV exists on disk, computing offset if needed.
    """
    if output_csv.exists():
        return output_csv

    seq = load_mocopi_recording(motion_source)
    cam_df = pd.read_csv(camera_csv)
    try:
        offset_used = estimate_camera_to_mocopi_offset(
            seq,
            cam_df,
            search_ms,
            rate_hz,
            offset_ms,
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
        )
        df_out = export_reliability_errors(
            seq,
            cam_df,
            ["l_foot", "r_foot", "l_hand", "r_hand"],
            ["LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_WRIST", "RIGHT_WRIST"],
            offset_used,
        )
    except NoCameraPoseDataError:
        df_out = pd.DataFrame(columns=RELIABILITY_COLUMNS)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return output_csv


def nd_factor_from_stem(stem: str) -> float:
    letter = stem.split("_")[1][-1] if "_" in stem else stem[-1]
    mapping = {"a": 2.0, "b": 4.0, "c": 8.0, "d": 16.0}
    return mapping.get(letter.lower(), 1.0)


def best_joint_from_reliability(df_or_path: Path | pd.DataFrame) -> str | None:
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, (str, Path)) else df_or_path
    best_joint = None
    best_abs_corr = -np.inf
    for joint, sub in df.groupby("joint"):
        moc = sub["mocopi_dy"]
        cam = sub["camera_dy"]
        mask = moc.notna() & cam.notna()
        if mask.sum() < 10:
            continue
        corr = np.corrcoef(moc[mask], cam[mask])[0, 1]
        if np.isfinite(corr) and abs(corr) > best_abs_corr:
            best_abs_corr = abs(corr)
            best_joint = joint
    return best_joint


def align_visibility_series(
    cam_csv: Path | pd.DataFrame,
    landmark: str,
    target_times_ms: np.ndarray,
    offset_ms: float,
) -> np.ndarray:
    df = pd.read_csv(cam_csv) if isinstance(cam_csv, (str, Path)) else cam_csv
    if "visibility" not in df.columns:
        return np.full_like(target_times_ms, np.nan, dtype=float)
    sub = df[(df["source"] == "pose") & (df["landmark_name"] == landmark)]
    if sub.empty:
        return np.full_like(target_times_ms, np.nan, dtype=float)
    ts = sub["timestamp_ms"].to_numpy(dtype=float) + offset_ms
    vis = sub["visibility"].to_numpy(dtype=float)
    return np.interp(target_times_ms, ts, vis, left=np.nan, right=np.nan)


def align_pose_counts(
    cam_csv: Path | pd.DataFrame,
    target_times_ms: np.ndarray,
    offset_ms: float,
) -> np.ndarray:
    df = pd.read_csv(cam_csv) if isinstance(cam_csv, (str, Path)) else cam_csv
    sub = df[df["source"] == "pose"]
    if sub.empty or "timestamp_ms" not in sub.columns:
        return np.full_like(target_times_ms, np.nan, dtype=float)
    counts = sub.groupby("timestamp_ms").size().reset_index(name="count")
    ts = counts["timestamp_ms"].to_numpy(dtype=float) + offset_ms
    cnt = counts["count"].to_numpy(dtype=float)
    return np.interp(target_times_ms, ts, cnt, left=np.nan, right=np.nan)


def joint_medians(df: pd.DataFrame, metric: str = "error_2d") -> pd.DataFrame:
    metric_series = pd.to_numeric(df[metric], errors="coerce")
    finite_mask = metric_series.replace([float("inf"), float("-inf")], pd.NA).notna()
    df_clean = df[finite_mask]
    grouped = df_clean.groupby("joint")[metric].median().reset_index()
    return grouped


def nd_error_summary(
    items: Iterable[tuple[float, pd.DataFrame]],
    metric: str = "error_2d",
) -> pd.DataFrame:
    records: list[dict] = []
    for nd_level, df in items:
        med = joint_medians(df, metric)
        for _, row in med.iterrows():
            records.append({"nd": nd_level, "joint": row["joint"], metric: row[metric]})
    return pd.DataFrame.from_records(records)


def get_aligned_traces(
    motion_source: Path,
    camera_csv: Path,
    joints: Sequence[str],
    landmarks: Sequence[str],
    search_ms: float,
    rate_hz: float,
    offset_ms: float | None,
    clip_start_s: float | None,
    clip_end_s: float | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """
    Return aligned egocentric ΔY traces for Mocopi joints and camera landmarks.

    Returns:
        t_s, mocopi_y, camera_y, offset_used
        where t_s is Mocopi time in seconds, and dicts map names to ΔY arrays.
    """
    seq = load_mocopi_recording(motion_source)
    cam_df = pd.read_csv(camera_csv)

    needed_joints = list(dict.fromkeys([*joints, *SCALE_REF_JOINTS]))
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, needed_joints)
    scales = compute_body_scale_series(mocopi_pos)

    mocopi_y: dict[str, np.ndarray] = {}
    for j in joints:
        if j not in mocopi_pos:
            continue
        mocopi_y[j] = mocopi_pos[j][:, 1] / scales

    t_c_ms, camera_pos = compute_camera_egocentric_positions(cam_df, landmarks)

    if offset_ms is None:
        offset_used = estimate_camera_to_mocopi_offset(
            seq,
            cam_df,
            search_ms,
            rate_hz,
            None,
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
        )
    else:
        offset_used = offset_ms

    t_c_aligned = t_c_ms + offset_used
    camera_y: dict[str, np.ndarray] = {}
    for lm in landmarks:
        if lm not in camera_pos:
            continue
        camera_y[lm] = np.interp(
            t_m_ms,
            t_c_aligned,
            camera_pos[lm][:, 1],
            left=np.nan,
            right=np.nan,
        )

    t_s = t_m_ms / 1000.0
    if clip_start_s is not None or clip_end_s is not None:
        if clip_start_s is None:
            clip_start_s = float(t_s.min())
        if clip_end_s is None:
            clip_end_s = float(t_s.max())
        mask = (t_s >= clip_start_s) & (t_s <= clip_end_s)
        t_s = t_s[mask]
        for k in list(mocopi_y.keys()):
            mocopi_y[k] = mocopi_y[k][mask]
        for k in list(camera_y.keys()):
            camera_y[k] = camera_y[k][mask]

    return t_s, mocopi_y, camera_y, offset_used


__all__ = [
    "SCALE_REF_JOINTS",
    "RELIABILITY_COLUMNS",
    "compute_body_scale_series",
    "export_reliability_errors",
    "ensure_reliability_csv",
    "nd_factor_from_stem",
    "best_joint_from_reliability",
    "align_visibility_series",
    "align_pose_counts",
    "joint_medians",
    "nd_error_summary",
    "get_aligned_traces",
]
