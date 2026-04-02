from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

from .bvh_io import MocopiSequence, load_bvh


_CONTAINER_BOXES = {"head", "sndf", "skdf", "bons", "bndt", "fram", "btrs", "btdt"}
_SESSION_JOINT_NAMES = [
    "root",
    "torso_1",
    "torso_2",
    "torso_3",
    "torso_4",
    "torso_5",
    "torso_6",
    "torso_7",
    "neck_1",
    "neck_2",
    "head",
    "l_shoulder",
    "l_up_arm",
    "l_low_arm",
    "l_hand",
    "r_shoulder",
    "r_up_arm",
    "r_low_arm",
    "r_hand",
    "l_up_leg",
    "l_low_leg",
    "l_foot",
    "l_toes",
    "r_up_leg",
    "r_low_leg",
    "r_foot",
    "r_toes",
]
_SESSION_PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    7,
    11,
    12,
    13,
    7,
    15,
    16,
    17,
    0,
    19,
    20,
    21,
    0,
    23,
    24,
    25,
]


@dataclass
class _FrameData:
    frame_number: int
    timestamp_s: float
    joint_ids: list[int]
    rotations: np.ndarray
    positions: np.ndarray


def _iter_boxes(message: bytes, start: int = 0, end: int | None = None) -> Iterable[tuple[str, int, int]]:
    if end is None:
        end = len(message)
    offset = start
    while offset + 8 <= end:
        size = struct.unpack_from("<i", message, offset)[0]
        if size <= 0:
            break
        data_start = offset + 8
        data_end = data_start + size
        if data_end > end:
            break
        box_type = message[offset + 4 : offset + 8].decode("ascii", errors="ignore")
        yield box_type, data_start, data_end
        if box_type in _CONTAINER_BOXES:
            yield from _iter_boxes(message, data_start, data_end)
        offset = data_end


def _parse_skeleton_message(message: bytes) -> tuple[list[int], list[int]] | None:
    joint_ids: list[int] = []
    parent_ids: list[int] = []
    has_skeleton = False
    for box_type, data_start, _ in _iter_boxes(message):
        if box_type == "skdf":
            has_skeleton = True
        elif box_type == "bnid":
            joint_ids.append(int(struct.unpack_from("<h", message, data_start)[0]))
        elif box_type == "pbid":
            parent_ids.append(int(struct.unpack_from("<h", message, data_start)[0]))
    if not has_skeleton or not joint_ids:
        return None
    return joint_ids, parent_ids


def _parse_frame_message(message: bytes) -> _FrameData | None:
    frame_number: int | None = None
    timestamp_s: float | None = None
    joint_ids: list[int] = []
    rotations: list[list[float]] = []
    positions: list[list[float]] = []
    has_frame = False

    for box_type, data_start, _ in _iter_boxes(message):
        if box_type == "fram":
            has_frame = True
        elif box_type == "fnum":
            frame_number = int(struct.unpack_from("<i", message, data_start)[0])
        elif box_type == "time":
            timestamp_s = float(struct.unpack_from("<f", message, data_start)[0])
        elif box_type == "bnid":
            joint_ids.append(int(struct.unpack_from("<h", message, data_start)[0]))
        elif box_type == "tran":
            values = struct.unpack_from("<7f", message, data_start)
            rotations.append([values[0], values[1], values[2], values[3]])
            positions.append([values[4], values[5], values[6]])

    if not has_frame or frame_number is None or timestamp_s is None or not joint_ids:
        return None

    return _FrameData(
        frame_number=frame_number,
        timestamp_s=timestamp_s,
        joint_ids=joint_ids,
        rotations=np.asarray(rotations, dtype=float),
        positions=np.asarray(positions, dtype=float),
    )


def _read_mocopi_packets(raw: bytes) -> tuple[tuple[list[int], list[int]] | None, list[_FrameData]]:
    offset = 0
    skeleton: tuple[list[int], list[int]] | None = None
    frames: list[_FrameData] = []

    while offset + 12 <= len(raw):
        size = struct.unpack_from("<i", raw, offset + 8)[0]
        if size <= 0 or offset + 12 + size > len(raw):
            break
        message = raw[offset + 12 : offset + 12 + size]
        offset += 12 + size

        if skeleton is None:
            parsed_skeleton = _parse_skeleton_message(message)
            if parsed_skeleton is not None:
                skeleton = parsed_skeleton

        parsed_frame = _parse_frame_message(message)
        if parsed_frame is not None:
            frames.append(parsed_frame)

    return skeleton, frames


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion.astype(float)
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-8:
        return np.eye(3, dtype=float)
    x, y, z, w = quaternion / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=float,
    )


def _session_joint_names(joint_ids: list[int]) -> list[str]:
    if joint_ids == list(range(len(_SESSION_JOINT_NAMES))):
        return list(_SESSION_JOINT_NAMES)
    return [f"joint_{joint_id}" for joint_id in joint_ids]


def _build_world_positions(
    joint_ids: list[int],
    parent_ids: list[int],
    frames: list[_FrameData],
    joint_names: list[str],
) -> dict[str, np.ndarray]:
    num_frames = len(frames)
    num_joints = len(joint_ids)
    positions = np.zeros((num_frames, num_joints, 3), dtype=float)

    joint_index = {joint_id: idx for idx, joint_id in enumerate(joint_ids)}
    parent_indices = [joint_index.get(parent_id, -1) for parent_id in parent_ids]

    for frame_idx, frame in enumerate(frames):
        local_positions = np.asarray(frame.positions, dtype=float)
        local_rotations = np.asarray(frame.rotations, dtype=float)
        world_positions = np.zeros((num_joints, 3), dtype=float)
        world_rotations = np.zeros((num_joints, 4), dtype=float)
        world_rotations[:, 3] = 1.0

        for joint_idx in range(num_joints):
            parent_idx = parent_indices[joint_idx]
            local_pos = local_positions[joint_idx]
            local_rot = local_rotations[joint_idx]
            if parent_idx < 0:
                world_positions[joint_idx] = local_pos
                world_rotations[joint_idx] = local_rot
                continue

            parent_rot = world_rotations[parent_idx]
            rotated_offset = _quaternion_to_matrix(parent_rot) @ local_pos
            world_positions[joint_idx] = world_positions[parent_idx] + rotated_offset
            world_rotations[joint_idx] = _quaternion_multiply(parent_rot, local_rot)

        positions[frame_idx] = world_positions

    return {name: positions[:, idx, :] for idx, name in enumerate(joint_names)}


def load_mocopi_bin(path: str | Path) -> MocopiSequence:
    path = Path(path)
    raw = path.read_bytes()
    skeleton, frames = _read_mocopi_packets(raw)
    if not frames:
        raise ValueError(f"No frame data found in Mocopi BIN file: {path}")

    frames.sort(key=lambda frame: frame.frame_number)
    if skeleton is None:
        joint_ids = list(frames[0].joint_ids)
        parent_ids = list(_SESSION_PARENTS[: len(joint_ids)])
    else:
        joint_ids, parent_ids = skeleton

    joint_names = _session_joint_names(joint_ids)
    timestamps_ms = (np.asarray([frame.timestamp_s for frame in frames], dtype=float) - frames[0].timestamp_s) * 1000.0
    if len(timestamps_ms) > 1:
        frame_time = float(np.median(np.diff(timestamps_ms))) / 1000.0
    else:
        frame_time = 1.0 / 50.0

    world_positions = _build_world_positions(joint_ids, parent_ids, frames, joint_names)
    num_joints = len(joint_names)
    channel_counts = [6] * num_joints
    channel_starts = [idx * 6 for idx in range(num_joints)]
    dummy_frames = np.zeros((len(frames), num_joints * 6), dtype=float)

    return MocopiSequence(
        joint_names=joint_names,
        channel_counts=channel_counts,
        frame_time=frame_time,
        frames=dummy_frames,
        metadata={
            "source_path": str(path),
            "source_format": "mocopi_bin",
            "frames": str(len(frames)),
            "frame_time": str(frame_time),
        },
        parents=list(parent_ids),
        offsets=np.zeros((num_joints, 3), dtype=float),
        channel_starts=channel_starts,
        timestamp_override_ms=timestamps_ms,
        joint_position_overrides=world_positions,
    )


def resolve_mocopi_source(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        bvh_candidates = sorted(path.glob("*.bvh"))
        if bvh_candidates:
            return bvh_candidates[0]
        bin_candidates = sorted(path.glob("*_mocopi.bin"))
        if bin_candidates:
            return bin_candidates[0]
        raise FileNotFoundError(f"No Mocopi motion source found under {path}")
    return path


def load_mocopi_recording(path: str | Path) -> MocopiSequence:
    resolved = resolve_mocopi_source(path)
    suffix = resolved.suffix.lower()
    if suffix == ".bvh":
        return load_bvh(resolved)
    if suffix == ".bin":
        return load_mocopi_bin(resolved)
    raise ValueError(f"Unsupported Mocopi motion source: {resolved}")


__all__ = [
    "load_mocopi_bin",
    "load_mocopi_recording",
    "resolve_mocopi_source",
]
