# -*- coding: utf-8 -*-
"""
Utilities for de-identifying video frames.

Pose landmarks are combined with OpenCV's YuNet face detector to anonymize every
visible participant, even when heads are turned or multiple people appear. When the
base-scale detector misses a distant subject, a multi-scale fallback rescans the frame.
"""
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

os.environ.setdefault("MEDIAPIPE_SKIP_AUDIO", "1")

import mediapipe as mp

from mediapipe.framework.formats import landmark_pb2

PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
PoseLandmark = mp.solutions.pose.PoseLandmark

FACE_LANDMARK_INDICES: Sequence[int] = [
    PoseLandmark.NOSE.value,
    PoseLandmark.LEFT_EYE_INNER.value,
    PoseLandmark.LEFT_EYE.value,
    PoseLandmark.LEFT_EYE_OUTER.value,
    PoseLandmark.RIGHT_EYE_INNER.value,
    PoseLandmark.RIGHT_EYE.value,
    PoseLandmark.RIGHT_EYE_OUTER.value,
    PoseLandmark.MOUTH_LEFT.value,
    PoseLandmark.MOUTH_RIGHT.value,
    PoseLandmark.LEFT_EAR.value,
    PoseLandmark.RIGHT_EAR.value,
]

POSE_MATCH_DISTANCE_FACTOR = 0.18  # fraction of max dimension used for pose/det match
YUNET_SCORE_THRESHOLD = 0.6
YUNET_NMS_THRESHOLD = 0.3
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MULTISCALE_FACTORS: Tuple[float, ...] = (1.5, 2.0)
MULTISCALE_IOU_THRESHOLD = 0.35

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
YUNET_MODEL_PATH = MODEL_DIR / YUNET_MODEL_FILENAME

_YUNET_DETECTOR: Optional[cv2.FaceDetectorYN] = None


def _collect_landmark_coords(
    landmarks: Sequence[landmark_pb2.NormalizedLandmark],
    indices: Iterable[int],
) -> Sequence[tuple[float, float, float]]:
    coords = []
    for idx in indices:
        if idx >= len(landmarks):
            return []
        lm = landmarks[idx]
        coords.append((lm.x, lm.y, lm.z))
    return coords


def _get_yunet_detector(frame_shape: Tuple[int, int, int]) -> cv2.FaceDetectorYN:
    global _YUNET_DETECTOR
    height, width, _ = frame_shape
    if not YUNET_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YuNet model not found at {YUNET_MODEL_PATH}. "
            "Download it from the OpenCV Zoo and place it in the models/ directory."
        )

    if _YUNET_DETECTOR is None:
        _YUNET_DETECTOR = cv2.FaceDetectorYN_create(
            str(YUNET_MODEL_PATH),
            "",
            (width, height),
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
            top_k=5000,
        )
    else:
        _YUNET_DETECTOR.setInputSize((width, height))
    return _YUNET_DETECTOR


def _run_yunet(frame_rgb: np.ndarray) -> Optional[np.ndarray]:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    detector = _get_yunet_detector(frame_rgb.shape)
    _, detections = detector.detect(frame_bgr)
    return detections


def _clip_box(
    box: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    frame_height, frame_width, _ = frame_shape
    xmin = int(max(0, min(frame_width, box[0])))
    ymin = int(max(0, min(frame_height, box[1])))
    xmax = int(max(0, min(frame_width, box[2])))
    ymax = int(max(0, min(frame_height, box[3])))
    return xmin, ymin, xmax, ymax


def _detections_to_boxes(
    detections: Optional[np.ndarray],
    frame_shape: Tuple[int, int, int],
    scale: float = 1.0,
) -> List[Tuple[int, int, int, int]]:
    if detections is None:
        return []
    boxes: List[Tuple[int, int, int, int]] = []
    inv_scale = 1.0 / scale
    for det in detections:
        x, y, w, h = det[:4]
        xmin = int(x * inv_scale)
        ymin = int(y * inv_scale)
        xmax = int((x + w) * inv_scale)
        ymax = int((y + h) * inv_scale)
        boxes.append(_clip_box((xmin, ymin, xmax, ymax), frame_shape))
    return boxes


def _detect_faces_yunet(frame_rgb: np.ndarray) -> List[dict]:
    if frame_rgb.size == 0:
        return []
    detections = _run_yunet(frame_rgb)
    boxes = _detections_to_boxes(detections, frame_rgb.shape, scale=1.0)
    return [{"bbox": box} for box in boxes]


def _compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _merge_boxes_iou(
    boxes: List[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int, int],
    threshold: float = MULTISCALE_IOU_THRESHOLD,
) -> List[Tuple[int, int, int, int]]:
    merged: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        box = _clip_box(box, frame_shape)
        matched = False
        for idx, existing in enumerate(merged):
            if _compute_iou(box, existing) >= threshold:
                xmin = min(box[0], existing[0])
                ymin = min(box[1], existing[1])
                xmax = max(box[2], existing[2])
                ymax = max(box[3], existing[3])
                merged[idx] = _clip_box((xmin, ymin, xmax, ymax), frame_shape)
                matched = True
                break
        if not matched:
            merged.append(box)
    return merged


def _detect_faces_multiscale(frame_rgb: np.ndarray) -> List[dict]:
    if frame_rgb.size == 0:
        return []

    boxes: List[Tuple[int, int, int, int]] = []
    for scale in MULTISCALE_FACTORS:
        scaled = cv2.resize(
            frame_rgb,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
        detections = _run_yunet(scaled)
        boxes.extend(_detections_to_boxes(detections, frame_rgb.shape, scale=scale))

    if not boxes:
        return []

    merged = _merge_boxes_iou(boxes, frame_rgb.shape)
    return [{"bbox": box} for box in merged]


def _pose_face_bounds(
    pose_landmarks: Sequence[landmark_pb2.NormalizedLandmark],
    frame_shape: Tuple[int, int, int],
    depth_padding_px: float = 0.0,
) -> Tuple[int, int, int, int] | None:
    """
    Build a bounding rectangle by projecting the face landmark prism into XY space.
    """
    coords = _collect_landmark_coords(pose_landmarks, FACE_LANDMARK_INDICES)
    if not coords:
        return None

    height, width, _ = frame_shape
    xs = [pt[0] * width for pt in coords]
    ys = [pt[1] * height for pt in coords]
    zs = [pt[2] * width for pt in coords]  # MediaPipe z is normalized to image width

    depth_span = (max(zs) - min(zs)) if zs else 0.0
    pad = max(depth_padding_px, depth_span * 0.6)

    xmin = int(max(0, min(xs) - pad))
    xmax = int(min(width, max(xs) + pad))
    ymin = int(max(0, min(ys) - pad))
    ymax = int(min(height, max(ys) + pad))

    if xmin >= xmax or ymin >= ymax:
        return None
    return xmin, ymin, xmax, ymax


def _expand_box(
    box: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    scale: float = 1.25,
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return box

    pad_x = int(width * (scale - 1.0) * 0.05)
    pad_y = int(height * (scale - 1.0) * 0.05)

    frame_height, frame_width, _ = frame_shape
    expanded = (
        max(0, xmin - pad_x),
        max(0, ymin - pad_y),
        min(frame_width, xmax + pad_x),
        min(frame_height, ymax + pad_y),
    )
    return expanded


def _match_pose_face(
    detection_center: Tuple[float, float],
    pose_boxes: List[dict],
    frame_shape: Tuple[int, int, int],
) -> Tuple[int, int, int, int] | None:
    frame_height, frame_width, _ = frame_shape
    max_dim = max(frame_height, frame_width)
    max_distance = max_dim * POSE_MATCH_DISTANCE_FACTOR

    best = None
    best_dist = float("inf")
    for pose_face in pose_boxes:
        pose_center = pose_face["center"]
        dist = math.dist(detection_center, pose_center)
        if dist < best_dist:
            best = pose_face
            best_dist = dist

    if best and best_dist <= max_distance:
        best["assigned"] = True
        return best["bbox"]
    return None


def _pose_faces(
    pose_result: PoseLandmarkerResult, frame_shape: Tuple[int, int, int]
) -> List[dict]:
    if not pose_result or not pose_result.pose_landmarks:
        return []

    faces: List[dict] = []
    for landmarks in pose_result.pose_landmarks:
        bounds = _pose_face_bounds(landmarks, frame_shape)
        if not bounds:
            continue
        xmin, ymin, xmax, ymax = bounds
        center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
        faces.append({"bbox": bounds, "center": center, "assigned": False})
    return faces


def _mask_region(
    frame: np.ndarray, bounds: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 0, 0)
) -> None:
    xmin, ymin, xmax, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        return
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=-1)


def blur_face_with_pose(
    frame_rgb: np.ndarray, pose_result: PoseLandmarkerResult, square_scale: float = 2.0
) -> np.ndarray:
    """
    Anonymize all detected faces in the frame.

    - Pose landmarks take precedence; when they are present we rely on their
      projected face prisms to minimize detector overhead.
    - YuNet only runs when no pose landmarks are available; if the base
      resolution misses a face the detector retries on upscaled frames before
      giving up.
    - Pose landmarks, when present, expand the mask using a projected 3D prism
      so turning heads remain covered.

    Args:
        frame_rgb: Frame in RGB format.
        pose_result: Pose detection result for the frame (optional for refinement).
        square_scale: Retained for backwards compatibility; used as a padding
            multiplier for RetinaFace boxes when pose data is absent.
    Returns:
        annotated: Annotated frame with faces masked out.
    """
    if frame_rgb.size == 0:
        return frame_rgb

    annotated = np.copy(frame_rgb)
    frame_shape = annotated.shape

    masks: List[Tuple[int, int, int, int]] = []

    pose_faces = _pose_faces(pose_result, frame_shape)
    if pose_faces:
        # Pose landmarks already give us the per-person regions; skip YuNet for this frame.
        masks.extend(face["bbox"] for face in pose_faces)
    else:
        yunet_detections = _detect_faces_yunet(frame_rgb)
        if not yunet_detections:
            yunet_detections = _detect_faces_multiscale(frame_rgb)

        for detection in yunet_detections:
            box = detection["bbox"]
            masks.append(_expand_box(box, frame_shape, scale=square_scale))

    for bounds in masks:
        _mask_region(annotated, bounds)

    return annotated
