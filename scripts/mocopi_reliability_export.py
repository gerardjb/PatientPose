from __future__ import annotations

"""Export per-frame Mocopi vs MediaPipe egocentric errors."""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mocopi import load_bvh, estimate_time_offset
from mocopi.features import (
    compute_egocentric_positions,
    compute_camera_egocentric_positions,
    resample_feature,
)

SCALE_REF_JOINTS = ["l_up_leg", "r_up_leg", "l_shoulder", "r_shoulder"]


def _compute_mocopi_scale_series(
    mocopi_pos: dict[str, np.ndarray],
    ref_names: Sequence[str] = SCALE_REF_JOINTS,
) -> np.ndarray:
    """Return a per-frame body scale so Mocopi matches the camera normalization."""

    ref_arrays = [mocopi_pos[name][:, :2] for name in ref_names if name in mocopi_pos]
    if not ref_arrays:
        ref_arrays = [arr[:, :2] for arr in mocopi_pos.values()]
    if not ref_arrays:
        raise RuntimeError("No Mocopi joints available to compute scale")

    ref_stack = np.stack(ref_arrays, axis=0)
    scales = np.linalg.norm(ref_stack, axis=2).mean(axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    return scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-frame Mocopi vs MediaPipe egocentric errors."
    )
    parser.add_argument("--bvh", type=Path, required=True, help="Path to Mocopi BVH file.")
    parser.add_argument(
        "--camera_csv",
        type=Path,
        required=True,
        help="Path to camera landmarks CSV (results/OutputCSVs).",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=["l_foot", "r_foot", "l_hand", "r_hand"],
        help="Mocopi joint names to compare.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=["LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_WRIST", "RIGHT_WRIST"],
        help="Camera pose landmarks corresponding to --joints (same order).",
    )
    parser.add_argument(
        "--offset_ms",
        type=float,
        default=None,
        help="Optional camera→mocopi offset in ms. If omitted, estimated from r_hand/RIGHT_WRIST.",
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
        help="Resampling rate (Hz) used when estimating offset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mocopi_camera_reliability.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def compute_or_use_offset(
    seq,
    cam_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
    offset_ms: float | None,
) -> float:
    """Use provided offset or estimate it from egocentric r_hand/RIGHT_WRIST."""
    if offset_ms is not None:
        print(f"Using provided offset_ms={offset_ms:.1f}")
        return offset_ms

    print("Estimating offset via egocentric r_hand/RIGHT_WRIST...")
    needed_joints = list(dict.fromkeys(["r_hand", *SCALE_REF_JOINTS]))
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, needed_joints)
    if "r_hand" not in mocopi_pos:
        raise RuntimeError("Joint 'r_hand' not found in Mocopi sequence")
    scales = _compute_mocopi_scale_series(mocopi_pos)
    f_m = mocopi_pos["r_hand"][:, 1] / scales

    t_c_ms, camera_pos = compute_camera_egocentric_positions(cam_df, ["RIGHT_WRIST"])
    if "RIGHT_WRIST" not in camera_pos:
        raise RuntimeError("Landmark 'RIGHT_WRIST' not found in camera CSV")
    f_c = camera_pos["RIGHT_WRIST"][:, 1]

    t_m_res, f_m_res = resample_feature(t_m_ms, f_m, rate_hz)
    t_c_res, f_c_res = resample_feature(t_c_ms, f_c, rate_hz)

    best_offset, best_score = estimate_time_offset(
        t_m_res,
        f_m_res,
        t_c_res,
        f_c_res,
        search_range_ms=search_ms,
        step_ms=10.0,
    )

    print(f"Estimated offset (camera → mocopi): {best_offset:.1f} ms")
    print(f"Correlation score at best offset:   {best_score:.3f}")
    return best_offset


def export_errors(
    seq,
    cam_df: pd.DataFrame,
    joints: Sequence[str],
    landmarks: Sequence[str],
    offset_ms: float,
    output_path: Path,
) -> None:
    if len(joints) != len(landmarks):
        raise ValueError("Expected --joints and --landmarks to have the same length")

    request_joints = list(dict.fromkeys([*joints, *SCALE_REF_JOINTS]))
    t_m_ms, mocopi_pos = compute_egocentric_positions(seq, request_joints)
    t_c_ms, camera_pos = compute_camera_egocentric_positions(cam_df, landmarks)

    t_c_aligned_ms = t_c_ms + offset_ms
    scales = _compute_mocopi_scale_series(mocopi_pos)

    records: list[dict] = []

    for j_name, lm_name in zip(joints, landmarks):
        if j_name not in mocopi_pos or lm_name not in camera_pos:
            print(f"Skipping pair {j_name} ↔ {lm_name}: missing data")
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

        for t_ms, err_i, mx_i, my_i, cx_i, cy_i in zip(
            t_m_ms, err, mx, my, cx, cy
        ):
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

    df_out = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Wrote {len(df_out)} rows to {output_path}")


def main() -> None:
    args = parse_args()

    seq = load_bvh(args.bvh)
    cam_df = pd.read_csv(args.camera_csv)

    offset_ms = compute_or_use_offset(seq, cam_df, args.search_ms, args.rate_hz, args.offset_ms)
    export_errors(seq, cam_df, args.joints, args.landmarks, offset_ms, args.output)


if __name__ == "__main__":
    main()
