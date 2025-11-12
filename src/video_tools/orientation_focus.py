"""Orientation analyzer that concentrates on high-quality pose detections."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from .frame_sampling import FrameSample, FrameSampler, FrameSamplerConfig
from .pose_focus import PoseFocusHint, bbox_from_landmarks
from .pose_quality import PoseQuality, PoseQualityScorer, PoseQualityScorerConfig

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class PoseObservation:
    """Pose detection metadata for a specific frame/rotation pairing."""

    frame_index: int
    timestamp_ms: int
    rotation_code: int | None
    quality: PoseQuality
    detection_confidence: float
    brightness: float
    contrast: float
    motion_score: float
    source: str
    landmarks: List[landmark_pb2.NormalizedLandmark] | None


@dataclass
class OrientationDecision:
    """Final orientation inference and the supporting observations."""

    rotation_code: int | None
    reason: str
    observations: List[PoseObservation] = field(default_factory=list)
    debug_entries: List[dict] = field(default_factory=list)
    focus_hint: PoseFocusHint | None = None


class TemporalAnchorBuffer:
    """Stores the first good pose and nearby observations for confirmation."""

    def __init__(self, radius: int) -> None:
        self.radius = radius
        self.anchor: PoseObservation | None = None
        self._neighbors: Dict[int, PoseObservation] = {}

    def record(self, observation: PoseObservation) -> None:
        if self.anchor is None and observation.quality.is_good:
            self.anchor = observation
            self._neighbors[observation.frame_index] = observation
            return
        if self.anchor and abs(observation.frame_index - self.anchor.frame_index) <= self.radius:
            self._neighbors[observation.frame_index] = observation

    def need_more_neighbors(self) -> bool:
        if not self.anchor:
            return False
        expected = self.radius * 2 + 1
        return len(self._neighbors) < expected

    def neighbors(self) -> List[PoseObservation]:
        return list(self._neighbors.values())


@dataclass
class OrientationAnalyzerConfig:
    """Configuration container for OrientationAnalyzer."""

    rotation_codes: Sequence[int | None]
    max_scan_frames: int = 150
    primary_stride: int = 5
    anchor_radius: int = 3
    vote_ratio_threshold: float = 1.6
    min_vote_weight: float = 0.15
    use_upscale_fallback: bool = True
    good_pose_target: int = 5
    min_detected_rotations: int = 2
    min_good_frame_support: int = 2
    debug_enabled: bool = False
    debug_output_dir: Path | None = None
    quality_config: PoseQualityScorerConfig = field(default_factory=PoseQualityScorerConfig)
    sampler_config: FrameSamplerConfig = field(default_factory=FrameSamplerConfig)

    def __post_init__(self) -> None:
        self.sampler_config.max_samples = self.max_scan_frames
        self.sampler_config.primary_stride = self.primary_stride
        self.sampler_config.rewind_radius = self.anchor_radius


class OrientationAnalyzer:
    """High-level orchestrator for orientation inference with pose anchors."""

    def __init__(
        self,
        config: OrientationAnalyzerConfig,
        sampler_factory: Callable[[cv2.VideoCapture, float, OrientationAnalyzerConfig], FrameSampler] | None = None,
        pose_marker_factory: Callable[[Path], PoseLandmarker] | None = None,
    ) -> None:
        self.config = config
        self._sampler_factory = sampler_factory or self._default_sampler_factory
        self._pose_marker_factory = pose_marker_factory or self._default_pose_marker_factory

    def analyze(self, video_path: Path, pose_model_path: Path) -> OrientationDecision:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sampler = self._sampler_factory(cap, fps, self.config)
        scorer = PoseQualityScorer(self.config.quality_config)

        rotation_votes: Dict[int | None, List[PoseObservation]] = defaultdict(list)
        good_pose_counts: Dict[int | None, int] = defaultdict(int)
        good_frame_support: Dict[int | None, set[int]] = defaultdict(set)
        detections_seen: set[int | None] = set()
        first_good_observation: Dict[int | None, PoseObservation] = {}
        anchor_buffer = TemporalAnchorBuffer(self.config.anchor_radius)
        processed_frames: set[int] = set()
        all_observations: List[PoseObservation] = []
        debug_entries: List[dict] = []
        best_observation: PoseObservation | None = None

        pose_marker_cm = self._pose_marker_factory(pose_model_path)
        try:
            with pose_marker_cm as pose_marker:
                if pose_marker is None:
                    raise RuntimeError("Pose landmarker could not be initialized.")
                for sample in sampler.uniform_samples():
                    if sample.frame_index in processed_frames:
                        continue
                    processed_frames.add(sample.frame_index)
                    sample_observations = self._evaluate_sample(pose_marker, sample, scorer)
                    valid_observations = [
                        obs for obs in sample_observations if obs.landmarks
                    ]
                    detected_rotations = {obs.rotation_code for obs in valid_observations}
                    if len(detected_rotations) < self.config.min_detected_rotations:
                        continue
                    all_observations.extend(valid_observations)
                    triggered, good_pose_rotation = self._register_observations(
                        valid_observations,
                        rotation_votes,
                        anchor_buffer,
                        good_pose_counts,
                        good_frame_support,
                        first_good_observation,
                        detections_seen,
                    )
                    best_observation = self._pick_best(best_observation, valid_observations)
                    if self.config.debug_enabled:
                        debug_entries.extend(self._observations_to_debug(valid_observations, sample))

                    neighbor_triggered = False
                    neighbor_rotation: int | None = None
                    if anchor_buffer.anchor and anchor_buffer.need_more_neighbors():
                        neighbors = sampler.request_neighbors(anchor_buffer.anchor.frame_index, self.config.anchor_radius)
                        for neighbor in neighbors:
                            if neighbor.frame_index in processed_frames:
                                continue
                            processed_frames.add(neighbor.frame_index)
                            neighbor_observations = self._evaluate_sample(
                                pose_marker,
                                neighbor,
                                scorer,
                                prefer_rotation=anchor_buffer.anchor.rotation_code,
                            )
                            valid_neighbor_observations = [
                                obs for obs in neighbor_observations if obs.landmarks
                            ]
                            detected_neighbor_rotations = {obs.rotation_code for obs in valid_neighbor_observations}
                            if len(detected_neighbor_rotations) < self.config.min_detected_rotations:
                                continue
                            all_observations.extend(valid_neighbor_observations)
                            triggered, good_pose_rotation = self._register_observations(
                                valid_neighbor_observations,
                                rotation_votes,
                                anchor_buffer,
                                good_pose_counts,
                                good_frame_support,
                                first_good_observation,
                                detections_seen,
                            )
                            if triggered and not neighbor_triggered:
                                neighbor_triggered = True
                                neighbor_rotation = good_pose_rotation
                            best_observation = self._pick_best(best_observation, valid_neighbor_observations)
                            if self.config.debug_enabled:
                                debug_entries.extend(self._observations_to_debug(valid_neighbor_observations, neighbor))
                    if neighbor_triggered:
                        triggered = True
                        good_pose_rotation = neighbor_rotation

                    if triggered:
                        return self._finalize_decision(
                            good_pose_rotation,
                            "good-target",
                            all_observations,
                            debug_entries,
                            sampler,
                            video_path,
                            first_good_observation,
                        )
        finally:
            cap.release()

        if len(detections_seen) < self.config.min_detected_rotations:
            decision = OrientationDecision(
                rotation_code=None,
                reason="insufficient-detections",
                observations=all_observations,
                debug_entries=debug_entries,
                focus_hint=None,
            )
        elif anchor_buffer.anchor:
            rotation_code = anchor_buffer.anchor.rotation_code
            decision = OrientationDecision(
                rotation_code=rotation_code,
                reason="anchor",
                observations=all_observations,
                debug_entries=debug_entries,
                focus_hint=self._derive_focus_hint(rotation_code, all_observations, first_good_observation),
            )
        elif best_observation:
            rotation_code = best_observation.rotation_code
            decision = OrientationDecision(
                rotation_code=rotation_code,
                reason="best-score",
                observations=all_observations,
                debug_entries=debug_entries,
                focus_hint=self._derive_focus_hint(rotation_code, all_observations, first_good_observation),
            )
        else:
            decision = OrientationDecision(
                rotation_code=None,
                reason="no-pose",
                observations=all_observations,
                debug_entries=debug_entries,
                focus_hint=None,
            )

        self._write_debug_payload(decision, sampler, video_path)
        return decision

    def _evaluate_sample(
        self,
        pose_marker: PoseLandmarker,
        sample: FrameSample,
        scorer: PoseQualityScorer,
        prefer_rotation: int | None = None,
    ) -> List[PoseObservation]:
        rotation_order = self._rotation_order(prefer_rotation)
        observations = self._evaluate_frame_rotations(pose_marker, sample, scorer, rotation_order, source="original")
        if observations:
            return observations
        if not self.config.use_upscale_fallback:
            return []

        upscaled = cv2.resize(sample.frame_bgr, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        return self._evaluate_frame_rotations(pose_marker, sample, scorer, rotation_order, source="upscaled", frame_override=upscaled)

    def _evaluate_frame_rotations(
        self,
        pose_marker: PoseLandmarker,
        sample: FrameSample,
        scorer: PoseQualityScorer,
        rotations: Sequence[int | None],
        source: str,
        frame_override: np.ndarray | None = None,
    ) -> List[PoseObservation]:
        frame_bgr = frame_override if frame_override is not None else sample.frame_bgr
        observations: List[PoseObservation] = []
        for rotation_code in rotations:
            rotated = self._rotate_frame(frame_bgr, rotation_code)
            frame_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = pose_marker.detect(mp_image)
            if not result.pose_landmarks:
                continue
            quality = scorer.score(result)
            if quality is None:
                continue
            observation = PoseObservation(
                frame_index=sample.frame_index,
                timestamp_ms=sample.timestamp_ms,
                rotation_code=rotation_code,
                quality=quality,
                detection_confidence=quality.visibility_mean,
                brightness=sample.brightness,
                contrast=sample.contrast,
                motion_score=sample.motion_score,
                source=source,
                landmarks=list(result.pose_landmarks[0]),
            )
            observations.append(observation)
        return observations

    def _register_observations(
        self,
        observations: Iterable[PoseObservation],
        rotation_votes: Dict[int | None, List[PoseObservation]],
        anchor_buffer: TemporalAnchorBuffer,
        good_pose_counts: Dict[int | None, int],
        good_frame_support: Dict[int | None, set[int]],
        first_good_observation: Dict[int | None, PoseObservation],
        detections_seen: set[int | None],
    ) -> tuple[bool, int | None]:
        triggered: bool | False = False
        trigger_rotation: int | None = None
        detected_rotations = {obs.rotation_code for obs in observations if obs.landmarks}
        global_detection_update = bool(detected_rotations)
        if global_detection_update:
            detections_seen.update(detected_rotations)
        eligible = len(detected_rotations) >= self.config.min_detected_rotations
        for obs in observations:
            rotation_votes[obs.rotation_code].append(obs)
            anchor_buffer.record(obs)
            if obs.quality.is_good and eligible:
                if obs.rotation_code not in first_good_observation:
                    first_good_observation[obs.rotation_code] = obs
                good_frame_support[obs.rotation_code].add(obs.frame_index)
                good_pose_counts[obs.rotation_code] += 1
                if (
                    triggered is False
                    and len(good_frame_support[obs.rotation_code]) >= self.config.min_good_frame_support
                    and good_pose_counts[obs.rotation_code] >= self.config.good_pose_target
                ):
                    triggered = True
                    trigger_rotation = obs.rotation_code
        return (triggered, trigger_rotation)

    def _evaluate_votes(self, rotation_votes: Dict[int | None, List[PoseObservation]]) -> tuple[int | None, float] | None:
        if not rotation_votes:
            return None
        vote_scores: Dict[int | None, float] = {}
        for rotation_code, observations in rotation_votes.items():
            weight = sum(1.0 / (obs.quality.score + 1.0) for obs in observations)
            vote_scores[rotation_code] = weight
        best_rotation = max(vote_scores, key=vote_scores.get)
        best_score = vote_scores[best_rotation]
        if best_score < self.config.min_vote_weight:
            return None
        second_score = max((score for rot, score in vote_scores.items() if rot != best_rotation), default=0.0)
        ratio = best_score / (second_score + 1e-6)
        if ratio >= self.config.vote_ratio_threshold:
            return best_rotation, ratio
        return None

    def _rotation_order(self, prefer_rotation: int | None) -> List[int | None]:
        rotations = list(self.config.rotation_codes)
        if prefer_rotation in rotations:
            rotations.remove(prefer_rotation)
            rotations.insert(0, prefer_rotation)
        return rotations

    @staticmethod
    def _rotate_frame(frame_bgr: np.ndarray, rotation_code: int | None) -> np.ndarray:
        if rotation_code is None:
            return frame_bgr
        return cv2.rotate(frame_bgr, rotation_code)

    @staticmethod
    def _pick_best(
        current_best: PoseObservation | None, new_observations: Iterable[PoseObservation]
    ) -> PoseObservation | None:
        best = current_best
        for obs in new_observations:
            if best is None or obs.quality.score < best.quality.score:
                best = obs
        return best

    def _observations_to_debug(self, observations: Iterable[PoseObservation], sample: FrameSample) -> List[dict]:
        debug_rows = []
        for obs in observations:
            debug_rows.append(
                {
                    "frame_index": obs.frame_index,
                    "timestamp_ms": obs.timestamp_ms,
                    "rotation_code": obs.rotation_code,
                    "score": obs.quality.score,
                    "is_good": obs.quality.is_good,
                    "visibility_mean": obs.quality.visibility_mean,
                    "brightness": sample.brightness,
                    "contrast": sample.contrast,
                    "motion_score": sample.motion_score,
                    "source": obs.source,
                }
            )
        return debug_rows

    def _write_debug_payload(
        self,
        decision: OrientationDecision,
        sampler: FrameSampler,
        video_path: Path,
    ) -> None:
        if not self.config.debug_enabled or not self.config.debug_output_dir:
            return
        payload = {
            "video": str(video_path),
            "decision": {
                "rotation_code": decision.rotation_code,
                "reason": decision.reason,
                "focus_hint": {
                    "frame_index": decision.focus_hint.frame_index,
                    "rotation_code": decision.focus_hint.rotation_code,
                    "bbox": decision.focus_hint.bbox,
                    "score": decision.focus_hint.score,
                }
                if decision.focus_hint
                else None,
            },
            "quick_rejects": getattr(sampler, "quick_rejects", 0),
            "observations": decision.debug_entries,
        }
        debug_dir = self.config.debug_output_dir
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{video_path.stem}_orientation.json"
        debug_path.write_text(json.dumps(payload, indent=2))

    def _finalize_decision(
        self,
        rotation_code: int | None,
        reason: str,
        observations: List[PoseObservation],
        debug_entries: List[dict],
        sampler: FrameSampler,
        video_path: Path,
        first_good_observation: Dict[int | None, PoseObservation],
    ) -> OrientationDecision:
        focus_hint = self._derive_focus_hint(rotation_code, observations, first_good_observation)
        decision = OrientationDecision(
            rotation_code=rotation_code,
            reason=reason,
            observations=observations,
            debug_entries=debug_entries,
            focus_hint=focus_hint,
        )
        self._write_debug_payload(decision, sampler, video_path)
        return decision

    @staticmethod
    def _default_sampler_factory(
        cap: cv2.VideoCapture, fps: float, config: OrientationAnalyzerConfig
    ) -> FrameSampler:
        sampler_config = FrameSamplerConfig(
            max_samples=config.max_scan_frames,
            primary_stride=config.primary_stride,
            rewind_radius=config.anchor_radius,
            min_brightness=config.sampler_config.min_brightness,
            max_brightness=config.sampler_config.max_brightness,
            min_contrast=config.sampler_config.min_contrast,
            motion_threshold=config.sampler_config.motion_threshold,
        )
        return FrameSampler(cap, fps, sampler_config)

    @staticmethod
    def _default_pose_marker_factory(pose_model_path: Path) -> PoseLandmarker:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=VisionRunningMode.IMAGE,
        )
        return PoseLandmarker.create_from_options(options)

    def _derive_focus_hint(
        self,
        rotation_code: int | None,
        observations: List[PoseObservation],
        first_good_observation: Dict[int | None, PoseObservation],
    ) -> PoseFocusHint | None:
        if rotation_code in first_good_observation:
            observation = first_good_observation[rotation_code]
            bbox = bbox_from_landmarks(observation.landmarks) if observation.landmarks else None
            if bbox:
                return PoseFocusHint(
                    frame_index=observation.frame_index,
                    rotation_code=rotation_code,
                    bbox=bbox,
                    score=observation.quality.score,
                )
        candidates = [
            obs for obs in observations if obs.rotation_code == rotation_code and obs.landmarks
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda obs: obs.quality.score)
        bbox = bbox_from_landmarks(best.landmarks)
        return PoseFocusHint(
            frame_index=best.frame_index,
            rotation_code=rotation_code,
            bbox=bbox,
            score=best.quality.score,
        )


__all__ = [
    "OrientationAnalyzer",
    "OrientationAnalyzerConfig",
    "OrientationDecision",
    "PoseObservation",
]
