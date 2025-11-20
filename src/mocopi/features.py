from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .bvh_io import MocopiSequence


@dataclass
class MocopiFeatureConfig:
    """Configuration for a 1D feature extracted from Mocopi joint positions."""

    joint_name: str
    component: str = "y"  # one of "x", "y", "z"
    normalize_by: str | None = None  # joint_name to use for scale, not yet used


@dataclass
class CameraFeatureConfig:
    """Configuration for a 1D feature extracted from camera landmark CSVs."""

    landmark_name: str
    source: str = "pose"  # "pose" or "hand"
    component: str = "y"  # "x", "y", "z"
    normalize_by_pair: tuple[str, str] | None = None  # e.g. ("LEFT_SHOULDER", "RIGHT_SHOULDER")


def _component_index(component: str) -> int:
    comp = component.lower()
    if comp == "x":
        return 0
    if comp == "y":
        return 1
    if comp == "z":
        return 2
    raise ValueError(f"Unknown component: {component!r}")


def extract_mocopi_feature(seq: MocopiSequence, config: MocopiFeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a 1D feature from a MocopiSequence.

    Returns:
        timestamps_ms, feature_values
    """
    positions = seq.joint_positions(config.joint_name)
    comp_idx = _component_index(config.component)
    values = positions[:, comp_idx].astype(float)
    timestamps = seq.timestamps_ms()
    return timestamps, values


def extract_camera_feature(
    df: pd.DataFrame,
    config: CameraFeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a 1D feature from a landmarks CSV.

    Expects a table with columns:
        frame, timestamp_ms, source, landmark_name, x, y, z
    (matching results/OutputCSVs structure).
    """
    df_source = df[
        (df["source"] == config.source)
        & (df["landmark_name"] == config.landmark_name)
    ].copy()
    if df_source.empty:
        raise ValueError(
            f"No rows found for source={config.source!r}, landmark_name={config.landmark_name!r}"
        )

    comp_idx = _component_index(config.component)
    comp_col = {0: "x", 1: "y", 2: "z"}[comp_idx]

    # Aggregate over instances if needed (e.g., multiple persons) by averaging.
    grouped = df_source.groupby("frame", as_index=False).agg(
        {
            "timestamp_ms": "first",
            comp_col: "mean",
        }
    )

    timestamps = grouped["timestamp_ms"].to_numpy(dtype=float)
    values = grouped[comp_col].to_numpy(dtype=float)
    return timestamps, values


def resample_feature(
    timestamps_ms: Sequence[float],
    values: Sequence[float],
    target_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample an arbitrary 1D feature to a uniform sampling grid using linear interpolation.
    """
    if len(timestamps_ms) == 0:
        raise ValueError("Cannot resample empty feature")

    t = np.asarray(timestamps_ms, dtype=float)
    v = np.asarray(values, dtype=float)

    t0 = float(t[0])
    t1 = float(t[-1])
    if t1 <= t0:
        raise ValueError("Non-increasing timestamps for feature")

    dt = 1000.0 / float(target_rate_hz)
    grid = np.arange(t0, t1, dt, dtype=float)

    # np.interp handles 1D linear interpolation
    resampled = np.interp(grid, t, v)
    return grid, resampled


def compute_center_of_mass(
    seq: MocopiSequence,
    joint_names: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Compute a simple per-frame center-of-mass proxy from Mocopi joints.

    By default this uses a small set of torso/hip joints if available,
    otherwise it falls back to all joints in the sequence.
    """
    if joint_names is None:
        # Prefer a consistent body core if present.
        preferred = [
            "root",
            "torso_3",
            "torso_5",
            "torso_7",
            "l_up_leg",
            "r_up_leg",
        ]
        joint_names = [j for j in preferred if j in seq.joint_names] or seq.joint_names

    positions = []
    for name in joint_names:
        positions.append(seq.joint_positions(name))

    if not positions:
        raise ValueError("No valid joints available to compute center-of-mass")

    stacked = np.stack(positions, axis=0)  # (J, T, 3)
    com = stacked.mean(axis=0)  # (T, 3)
    return com


def compute_egocentric_positions(
    seq: MocopiSequence,
    joint_names: Sequence[str],
    com_joint_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute per-frame joint positions in an egocentric frame.

    This subtracts a simple center-of-mass proxy from each requested joint's
    3D world-space position so that the body core stays approximately fixed
    near the origin over time.

    Returns:
        timestamps_ms, positions_rel
        where positions_rel[name] has shape (num_frames, 3).
    """
    com = compute_center_of_mass(seq, com_joint_names)
    timestamps = seq.timestamps_ms()

    positions_rel: dict[str, np.ndarray] = {}
    for name in joint_names:
        if name not in seq.joint_names:
            continue
        world = seq.joint_positions(name)
        positions_rel[name] = world - com

    return timestamps, positions_rel


def compute_camera_egocentric_positions(
    df: pd.DataFrame,
    landmark_names: Sequence[str],
    com_landmarks: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute per-frame egocentric 2D positions for selected camera landmarks.

    Uses pose landmarks only. For each frame, computes a COM proxy in image
    coordinates, subtracts it from each requested landmark's (x, y), and then
    normalizes by a per-frame body scale so that distance to the camera does
    not dominate the trajectories. The vertical component is flipped so that
    increasing values correspond to motion "upward" in the image, to match
    the Mocopi convention where +Y is roughly up.
    """
    pose_df = df[df["source"] == "pose"].copy()
    if pose_df.empty:
        raise ValueError("No pose landmarks found in camera CSV")

    if com_landmarks is None:
        core = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        com_landmarks = core

    frames = sorted(pose_df["frame"].unique())
    timestamps: list[float] = []
    series: dict[str, list[list[float]]] = {name: [] for name in landmark_names}

    for frame_idx in frames:
        sub = pose_df[pose_df["frame"] == frame_idx]
        if sub.empty:
            continue

        timestamps.append(float(sub["timestamp_ms"].iloc[0]))

        by_name: dict[str, tuple[float, float]] = {}
        for _, row in sub.iterrows():
            by_name[str(row["landmark_name"])] = (float(row["x"]), float(row["y"]))

        com_candidates = [by_name[name] for name in com_landmarks if name in by_name]
        if not com_candidates:
            com_candidates = list(by_name.values())
        if not com_candidates:
            # No landmarks at all in this frame; treat as zeros.
            com_x = com_y = 0.0
            scale = 1.0
        else:
            xs, ys = zip(*com_candidates)
            com_x = float(np.mean(xs))
            com_y = float(np.mean(ys))
            dists = np.sqrt((np.asarray(xs) - com_x) ** 2 + (np.asarray(ys) - com_y) ** 2)
            scale = float(np.mean(dists)) if dists.size else 1.0
            if scale <= 1e-6:
                scale = 1.0

        for name in landmark_names:
            if name in by_name:
                x, y = by_name[name]
                # Flip Y so that up is positive, matching Mocopi's +Y convention.
                series[name].append(
                    [(x - com_x) / scale, -(y - com_y) / scale]
                )
            else:
                series[name].append([np.nan, np.nan])

    timestamps_ms = np.asarray(timestamps, dtype=float)
    positions_rel: dict[str, np.ndarray] = {
        name: np.asarray(vals, dtype=float) for name, vals in series.items()
    }
    return timestamps_ms, positions_rel
