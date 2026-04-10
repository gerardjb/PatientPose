from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from .camera_projection import CameraProjectionConfig, compute_camera_projection
from .features import NoCameraPoseDataError, compute_egocentric_positions, resample_feature


FeatureKind = Literal["single", "midpoint", "difference"]
RelationKind = Literal["positive", "negative", "either"]
TaskKind = Literal["auto", "gait", "hand"]


@dataclass(frozen=True)
class SyncCandidate:
    name: str
    camera_landmarks: tuple[str, ...]
    mocopi_joints: tuple[str, ...]
    feature_kind: FeatureKind
    component: str
    camera_space: str = "image"
    expected_relation: RelationKind = "positive"
    prior_weight: float = 1.0
    smooth_window: int = 7


@dataclass(frozen=True)
class SyncEvaluation:
    candidate_name: str
    offset_ms: float
    signed_correlation: float
    metric_score: float
    peak_margin: float
    coverage: float
    overlap_ratio: float
    amplitude: float
    jitter_ratio: float
    final_score: float
    camera_space: str
    component: str
    feature_kind: FeatureKind
    expected_relation: RelationKind


def _component_index(component: str) -> int:
    comp = component.lower()
    if comp == "x":
        return 0
    if comp == "y":
        return 1
    if comp == "z":
        return 2
    raise ValueError(f"Unknown component: {component!r}")


def scan_time_offset_scores(
    t_a_ms: np.ndarray,
    f_a: np.ndarray,
    t_b_ms: np.ndarray,
    f_b: np.ndarray,
    search_range_ms: float,
    step_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the scanned offsets and normalized correlation scores for each offset.

    Offsets are applied such that:
        t_b_shifted = t_b_ms + offset
    """
    if len(t_a_ms) == 0 or len(t_b_ms) == 0:
        raise ValueError("Cannot estimate offset on empty features")

    t_a = np.asarray(t_a_ms, dtype=float)
    v_a = np.asarray(f_a, dtype=float)
    t_b = np.asarray(t_b_ms, dtype=float)
    v_b_raw = np.asarray(f_b, dtype=float)

    offsets = np.arange(-search_range_ms, search_range_ms + step_ms, step_ms, dtype=float)
    scores = np.full(offsets.shape, np.nan, dtype=float)

    v_a = (v_a - np.mean(v_a)) / (np.std(v_a) + 1e-6)
    v_b_raw = (v_b_raw - np.mean(v_b_raw)) / (np.std(v_b_raw) + 1e-6)

    for idx, offset in enumerate(offsets):
        t_b_shifted = t_b + offset
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

        num = float(np.sum(v_a_overlap * v_b_interp))
        denom = float(np.sqrt(np.sum(v_a_overlap**2) * np.sum(v_b_interp**2)) + 1e-6)
        scores[idx] = num / denom

    return offsets, scores


def estimate_time_offset(
    t_a_ms: np.ndarray,
    f_a: np.ndarray,
    t_b_ms: np.ndarray,
    f_b: np.ndarray,
    search_range_ms: float,
    step_ms: float = 10.0,
) -> Tuple[float, float]:
    """
    Estimate time offset between two 1D features using cross-correlation.

    Returns:
        (best_offset_ms, best_score)
        where best_offset_ms is applied such that:
            t_b_shifted = t_b_ms + best_offset_ms
        aligns feature B onto feature A.
    """
    offsets, scores = scan_time_offset_scores(
        t_a_ms,
        f_a,
        t_b_ms,
        f_b,
        search_range_ms=search_range_ms,
        step_ms=step_ms,
    )
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise RuntimeError("Unable to compute a finite correlation score for any offset")
    best_idx = int(np.nanargmax(np.abs(scores)))
    return float(offsets[best_idx]), float(scores[best_idx])


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
        msg = "Not enough finite samples after cleaning"
        if label:
            msg += f" for {label}"
        raise RuntimeError(msg)
    return t_arr, v_arr


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr.copy()
    radius = max(0, int(window) // 2)
    out = np.empty_like(arr, dtype=float)
    for idx in range(arr.size):
        lo = max(0, idx - radius)
        hi = min(arr.size, idx + radius + 1)
        out[idx] = float(np.mean(arr[lo:hi]))
    return out


def _robust_amplitude(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return 0.0
    return float(np.percentile(finite, 95.0) - np.percentile(finite, 5.0))


def _compute_jitter_ratio(values: np.ndarray, window: int) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 5:
        return 1.0
    smooth = _moving_average(arr, window)
    residual = arr - smooth
    smooth_std = float(np.std(smooth))
    residual_std = float(np.std(residual))
    return residual_std / (smooth_std + 1e-6)


def _metric_scores(scores: np.ndarray, relation: RelationKind) -> np.ndarray:
    if relation == "positive":
        return scores.copy()
    if relation == "negative":
        return -scores
    if relation == "either":
        return np.abs(scores)
    raise ValueError(f"Unsupported relation kind: {relation!r}")


def _best_offset_from_scores(
    offsets: np.ndarray,
    scores: np.ndarray,
    relation: RelationKind,
    *,
    separation_ms: float = 150.0,
) -> tuple[float, float, float, float]:
    metrics = _metric_scores(scores, relation)
    finite = np.isfinite(metrics)
    if not np.any(finite):
        raise RuntimeError("Unable to compute a finite correlation score for any offset")

    best_idx = int(np.nanargmax(metrics))
    best_offset = float(offsets[best_idx])
    best_signed = float(scores[best_idx])
    best_metric = float(metrics[best_idx])

    suppress = np.abs(offsets - best_offset) <= separation_ms
    competing = metrics.copy()
    competing[suppress] = np.nan
    second_metric = float(np.nanmax(competing)) if np.any(np.isfinite(competing)) else 0.0
    peak_margin = max(0.0, best_metric - second_metric)
    return best_offset, best_signed, best_metric, peak_margin


def _aggregate_feature(arrays: list[np.ndarray], kind: FeatureKind) -> np.ndarray:
    if not arrays:
        raise RuntimeError("No feature arrays available to aggregate")

    if kind == "single":
        return np.asarray(arrays[0], dtype=float)

    if kind == "midpoint":
        stacked = np.column_stack([np.asarray(arr, dtype=float) for arr in arrays])
        return np.nanmean(stacked, axis=1)

    if kind == "difference":
        if len(arrays) != 2:
            raise RuntimeError("Difference features require exactly two arrays")
        left = np.asarray(arrays[0], dtype=float)
        right = np.asarray(arrays[1], dtype=float)
        diff = left - right
        invalid = ~np.isfinite(left) | ~np.isfinite(right)
        diff[invalid] = np.nan
        return diff

    raise ValueError(f"Unsupported feature kind: {kind!r}")


def _candidate_bank(task: TaskKind = "auto") -> list[SyncCandidate]:
    gait_candidates = [
        SyncCandidate("feet_mid_world_z", ("LEFT_ANKLE", "RIGHT_ANKLE"), ("l_foot", "r_foot"), "midpoint", "z", "world", "positive", 1.30),
        SyncCandidate("feet_diff_world_z", ("LEFT_ANKLE", "RIGHT_ANKLE"), ("l_foot", "r_foot"), "difference", "z", "world", "positive", 1.25),
        SyncCandidate("knees_mid_world_z", ("LEFT_KNEE", "RIGHT_KNEE"), ("l_low_leg", "r_low_leg"), "midpoint", "z", "world", "positive", 1.18),
        SyncCandidate("knees_diff_world_z", ("LEFT_KNEE", "RIGHT_KNEE"), ("l_low_leg", "r_low_leg"), "difference", "z", "world", "positive", 1.14),
        SyncCandidate("hips_mid_world_z", ("LEFT_HIP", "RIGHT_HIP"), ("l_up_leg", "r_up_leg"), "midpoint", "z", "world", "positive", 1.05),
        SyncCandidate("feet_mid_world_y", ("LEFT_ANKLE", "RIGHT_ANKLE"), ("l_foot", "r_foot"), "midpoint", "y", "world", "positive", 0.95),
        SyncCandidate("feet_diff_world_y", ("LEFT_ANKLE", "RIGHT_ANKLE"), ("l_foot", "r_foot"), "difference", "y", "world", "positive", 0.92),
        SyncCandidate("ankles_mid_image_y", ("LEFT_ANKLE", "RIGHT_ANKLE"), ("l_foot", "r_foot"), "midpoint", "y", "image", "positive", 0.75),
        SyncCandidate("ankles_diff_image_y", ("LEFT_ANKLE", "RIGHT_ANKLE"), ("l_foot", "r_foot"), "difference", "y", "image", "positive", 0.70),
    ]
    hand_candidates = [
        SyncCandidate("right_wrist_image_y", ("RIGHT_WRIST",), ("r_hand",), "single", "y", "image", "positive", 1.20),
        SyncCandidate("left_wrist_image_y", ("LEFT_WRIST",), ("l_hand",), "single", "y", "image", "positive", 1.15),
        SyncCandidate("wrists_mid_image_y", ("LEFT_WRIST", "RIGHT_WRIST"), ("l_hand", "r_hand"), "midpoint", "y", "image", "positive", 1.05),
        SyncCandidate("right_wrist_world_z", ("RIGHT_WRIST",), ("r_hand",), "single", "z", "world", "positive", 0.95),
        SyncCandidate("left_wrist_world_z", ("LEFT_WRIST",), ("l_hand",), "single", "z", "world", "positive", 0.90),
    ]
    if task == "gait":
        return gait_candidates + hand_candidates
    if task == "hand":
        return hand_candidates + gait_candidates
    return gait_candidates + hand_candidates


def _build_camera_projection_cache(
    image_df: pd.DataFrame,
    world_df: pd.DataFrame | None,
    candidates: list[SyncCandidate],
) -> dict[str, object]:
    cache: dict[str, object] = {}
    for camera_space in sorted({candidate.camera_space for candidate in candidates}):
        if camera_space == "world":
            if world_df is None:
                continue
            df_space = world_df
        else:
            df_space = image_df

        landmark_names = sorted(
            {
                landmark_name
                for candidate in candidates
                if candidate.camera_space == camera_space
                for landmark_name in candidate.camera_landmarks
            }
        )
        if not landmark_names:
            continue
        try:
            cache[camera_space] = compute_camera_projection(
                df_space,
                landmark_names,
                CameraProjectionConfig(
                    space=camera_space,
                    visibility_threshold=0.4,
                    smooth_window=7,
                    rotate_to_body_frame=False,
                ),
            )
        except NoCameraPoseDataError:
            continue
    return cache


def _build_mocopi_position_cache(seq, candidates: list[SyncCandidate]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    joint_names = sorted(
        {
            joint_name
            for candidate in candidates
            for joint_name in candidate.mocopi_joints
        }
    )
    return compute_egocentric_positions(seq, joint_names)


def _extract_camera_candidate_feature(
    projection_cache: dict[str, object],
    candidate: SyncCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate.camera_space not in projection_cache:
        raise RuntimeError(f"Camera space {candidate.camera_space!r} is not available for candidate {candidate.name}")

    projection = projection_cache[candidate.camera_space]
    comp_idx = _component_index(candidate.component)
    arrays = []
    for landmark_name in candidate.camera_landmarks:
        if landmark_name not in projection.positions:
            raise RuntimeError(f"Landmark {landmark_name!r} not available for candidate {candidate.name}")
        arrays.append(projection.positions[landmark_name][:, comp_idx])
    return projection.timestamps_ms, _aggregate_feature(arrays, candidate.feature_kind)


def _extract_mocopi_candidate_feature(
    timestamps_ms: np.ndarray,
    mocopi_positions: dict[str, np.ndarray],
    candidate: SyncCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    comp_idx = _component_index(candidate.component)
    arrays = []
    for joint_name in candidate.mocopi_joints:
        if joint_name not in mocopi_positions:
            raise RuntimeError(f"Joint {joint_name!r} not available for candidate {candidate.name}")
        arrays.append(mocopi_positions[joint_name][:, comp_idx])
    return timestamps_ms, _aggregate_feature(arrays, candidate.feature_kind)


def _evaluate_resampled_features(
    candidate: SyncCandidate,
    t_ref_ms: np.ndarray,
    ref_values: np.ndarray,
    t_moving_ms: np.ndarray,
    moving_values: np.ndarray,
    *,
    search_ms: float,
    rate_hz: float,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> SyncEvaluation:
    raw_ref = np.asarray(ref_values, dtype=float)
    raw_moving = np.asarray(moving_values, dtype=float)
    coverage = min(
        float(np.isfinite(raw_ref).mean()) if raw_ref.size else 0.0,
        float(np.isfinite(raw_moving).mean()) if raw_moving.size else 0.0,
    )

    t_ref_ms, ref_values = clean_feature_samples(t_ref_ms, ref_values, f"{candidate.name} reference")
    t_moving_ms, moving_values = clean_feature_samples(t_moving_ms, moving_values, f"{candidate.name} moving")

    t_ref_res, ref_res = resample_feature(t_ref_ms, ref_values, rate_hz)
    t_moving_res, moving_res = resample_feature(t_moving_ms, moving_values, rate_hz)

    if clip_start_s is not None or clip_end_s is not None:
        clip_lo = clip_start_s * 1000.0 if clip_start_s is not None else float(t_ref_res[0])
        clip_hi = clip_end_s * 1000.0 if clip_end_s is not None else float(t_ref_res[-1])
        mask_ref = (t_ref_res >= clip_lo) & (t_ref_res <= clip_hi)
        if np.count_nonzero(mask_ref) > 10:
            t_ref_res = t_ref_res[mask_ref]
            ref_res = ref_res[mask_ref]

    smooth_ref = _moving_average(ref_res, candidate.smooth_window)
    smooth_moving = _moving_average(moving_res, candidate.smooth_window)

    amplitude = min(_robust_amplitude(smooth_ref), _robust_amplitude(smooth_moving))
    amp_score = float(np.tanh(amplitude / 0.35))
    jitter_ratio = max(
        _compute_jitter_ratio(ref_res, candidate.smooth_window),
        _compute_jitter_ratio(moving_res, candidate.smooth_window),
    )
    jitter_score = 1.0 / (1.0 + jitter_ratio)

    offsets, scores = scan_time_offset_scores(
        t_ref_res,
        smooth_ref,
        t_moving_res,
        smooth_moving,
        search_range_ms=search_ms,
        step_ms=10.0,
    )
    best_offset, signed_score, metric_score, peak_margin = _best_offset_from_scores(
        offsets,
        scores,
        candidate.expected_relation,
    )

    t_moving_shifted = t_moving_res + best_offset
    overlap_lo = max(float(t_ref_res[0]), float(t_moving_shifted[0]))
    overlap_hi = min(float(t_ref_res[-1]), float(t_moving_shifted[-1]))
    ref_duration = max(float(t_ref_res[-1] - t_ref_res[0]), 1.0)
    moving_duration = max(float(t_moving_res[-1] - t_moving_res[0]), 1.0)
    overlap_ratio = max(0.0, overlap_hi - overlap_lo) / max(1.0, min(ref_duration, moving_duration))

    peak_score = 0.75 * metric_score + 0.25 * min(1.0, peak_margin / 0.2)
    final_score = candidate.prior_weight * coverage * overlap_ratio * amp_score * jitter_score * max(0.0, peak_score)

    return SyncEvaluation(
        candidate_name=candidate.name,
        offset_ms=best_offset,
        signed_correlation=signed_score,
        metric_score=metric_score,
        peak_margin=peak_margin,
        coverage=coverage,
        overlap_ratio=overlap_ratio,
        amplitude=amplitude,
        jitter_ratio=jitter_ratio,
        final_score=final_score,
        camera_space=candidate.camera_space,
        component=candidate.component,
        feature_kind=candidate.feature_kind,
        expected_relation=candidate.expected_relation,
    )


def select_camera_to_mocopi_sync(
    seq,
    image_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
    *,
    offset_ms: float | None = None,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
    world_df: pd.DataFrame | None = None,
    task: TaskKind = "auto",
) -> SyncEvaluation:
    if offset_ms is not None:
        return SyncEvaluation(
            candidate_name="fixed_offset",
            offset_ms=float(offset_ms),
            signed_correlation=np.nan,
            metric_score=np.nan,
            peak_margin=np.nan,
            coverage=np.nan,
            overlap_ratio=np.nan,
            amplitude=np.nan,
            jitter_ratio=np.nan,
            final_score=np.nan,
            camera_space="fixed",
            component="na",
            feature_kind="single",
            expected_relation="either",
        )

    candidates = _candidate_bank(task)
    projection_cache = _build_camera_projection_cache(image_df, world_df, candidates)
    t_m_ms, mocopi_positions = _build_mocopi_position_cache(seq, candidates)

    evaluations: list[SyncEvaluation] = []
    for candidate in candidates:
        try:
            t_camera_ms, camera_feature = _extract_camera_candidate_feature(projection_cache, candidate)
            t_mocopi_ms, mocopi_feature = _extract_mocopi_candidate_feature(t_m_ms, mocopi_positions, candidate)
            evaluations.append(
                _evaluate_resampled_features(
                    candidate,
                    t_mocopi_ms,
                    mocopi_feature,
                    t_camera_ms,
                    camera_feature,
                    search_ms=search_ms,
                    rate_hz=rate_hz,
                    clip_start_s=clip_start_s,
                    clip_end_s=clip_end_s,
                )
            )
        except (RuntimeError, ValueError, NoCameraPoseDataError):
            continue

    if not evaluations:
        raise RuntimeError("No valid sync candidates were available for camera-to-Mocopi alignment")

    return max(evaluations, key=lambda evaluation: evaluation.final_score)


def select_camera_to_camera_sync(
    ref_image_df: pd.DataFrame,
    moving_image_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
    *,
    offset_ms: float | None = None,
    ref_world_df: pd.DataFrame | None = None,
    moving_world_df: pd.DataFrame | None = None,
    task: TaskKind = "auto",
) -> SyncEvaluation:
    if offset_ms is not None:
        return SyncEvaluation(
            candidate_name="fixed_offset",
            offset_ms=float(offset_ms),
            signed_correlation=np.nan,
            metric_score=np.nan,
            peak_margin=np.nan,
            coverage=np.nan,
            overlap_ratio=np.nan,
            amplitude=np.nan,
            jitter_ratio=np.nan,
            final_score=np.nan,
            camera_space="fixed",
            component="na",
            feature_kind="single",
            expected_relation="either",
        )

    candidates = _candidate_bank(task)
    ref_cache = _build_camera_projection_cache(ref_image_df, ref_world_df, candidates)
    moving_cache = _build_camera_projection_cache(moving_image_df, moving_world_df, candidates)

    evaluations: list[SyncEvaluation] = []
    for candidate in candidates:
        try:
            t_ref_ms, ref_feature = _extract_camera_candidate_feature(ref_cache, candidate)
            t_moving_ms, moving_feature = _extract_camera_candidate_feature(moving_cache, candidate)
            evaluations.append(
                _evaluate_resampled_features(
                    candidate,
                    t_ref_ms,
                    ref_feature,
                    t_moving_ms,
                    moving_feature,
                    search_ms=search_ms,
                    rate_hz=rate_hz,
                )
            )
        except (RuntimeError, ValueError, NoCameraPoseDataError):
            continue

    if not evaluations:
        raise RuntimeError("No valid sync candidates were available for camera-to-camera alignment")

    return max(evaluations, key=lambda evaluation: evaluation.final_score)


def estimate_camera_to_mocopi_offset(
    seq,
    cam_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
    offset_ms: float | None = None,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
    *,
    world_df: pd.DataFrame | None = None,
    task: TaskKind = "auto",
) -> float:
    """
    Estimate or reuse the camera-to-Mocopi offset using the best scored sync candidate.
    """
    evaluation = select_camera_to_mocopi_sync(
        seq,
        cam_df,
        search_ms,
        rate_hz,
        offset_ms=offset_ms,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        world_df=world_df,
        task=task,
    )
    return float(evaluation.offset_ms)


def estimate_camera_to_camera_offset(
    cam_ref_df: pd.DataFrame,
    cam_moving_df: pd.DataFrame,
    search_ms: float,
    rate_hz: float,
    offset_ms: float | None = None,
    *,
    ref_world_df: pd.DataFrame | None = None,
    moving_world_df: pd.DataFrame | None = None,
    task: TaskKind = "auto",
) -> float:
    """
    Estimate or reuse the moving-camera to reference-camera offset using the best scored sync candidate.

    The returned value is applied such that:
        t_moving_shifted = t_moving_ms + offset_ms
    aligns the moving camera onto the reference camera.
    """
    evaluation = select_camera_to_camera_sync(
        cam_ref_df,
        cam_moving_df,
        search_ms,
        rate_hz,
        offset_ms=offset_ms,
        ref_world_df=ref_world_df,
        moving_world_df=moving_world_df,
        task=task,
    )
    return float(evaluation.offset_ms)

