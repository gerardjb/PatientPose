from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MocopiSequence:
    """Container for Mocopi BVH motion data."""

    joint_names: List[str]
    channel_counts: List[int]
    frame_time: float
    frames: np.ndarray  # shape: (num_frames, total_channels)
    metadata: Dict[str, str]
    parents: List[int]
    offsets: np.ndarray  # shape: (num_joints, 3)
    channel_starts: List[int]

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def total_channels(self) -> int:
        return int(self.frames.shape[1])

    def timestamps_ms(self) -> np.ndarray:
        """Return timestamps in milliseconds for each frame."""
        return np.arange(self.num_frames, dtype=float) * self.frame_time * 1000.0

    def joint_index(self, joint_name: str) -> int:
        """Return the index of the named joint."""
        if joint_name not in self.joint_names:
            raise KeyError(f"Unknown Mocopi joint: {joint_name}")
        return self.joint_names.index(joint_name)

    def joint_channel_range(self, joint_name: str) -> Tuple[int, int]:
        """Return the [start, end) channel indices for a joint in the raw channel array."""
        idx = self.joint_index(joint_name)
        start = int(self.channel_starts[idx])
        count = int(self.channel_counts[idx])
        return start, start + count

    def joint_position_channels(self, joint_name: str) -> Tuple[int, int, int]:
        """
        Return indices for the X/Y/Z position channels of a joint.

        Assumes Mocopi-style BVH where the first six channels for each
        joint (including root) are:
            Xposition, Yposition, Zposition, Zrotation, Xrotation, Yrotation
        """
        start, end = self.joint_channel_range(joint_name)
        if end - start < 3:
            raise ValueError(f"Joint {joint_name} does not have 3 position channels")
        return start + 0, start + 1, start + 2

    def joint_positions(self, joint_name: str) -> np.ndarray:
        """
        Return approximate world-space XYZ positions for a joint as array of shape (num_frames, 3).

        This performs a simple forward-kinematics pass using:
            - Root translation channels.
            - Per-joint offsets.
            - Per-joint Z/X/Y rotation channels.

        It is intended for synchronization feature extraction rather than precise rendering.
        """
        joint_idx = self.joint_index(joint_name)
        return _forward_kinematics_joint(self, joint_idx)


def _parse_bvh_header(
    lines: List[str],
) -> Tuple[List[str], List[int], int, List[int], np.ndarray, List[int]]:
    """
    Parse the HIERARCHY section to extract joint metadata.

    Returns:
        joint_names, channel_counts, motion_start_index, parents, offsets, channel_starts
    """
    joint_names: List[str] = []
    channel_counts: List[int] = []
    parents: List[int] = []
    offsets_list: List[Tuple[float, float, float]] = []
    channel_starts: List[int] = []

    total_channels = 0

    i = 0
    n = len(lines)

    # Expect first line "HIERARCHY"
    while i < n and not lines[i].strip().upper().startswith("HIERARCHY"):
        i += 1
    if i >= n:
        raise ValueError("BVH file missing HIERARCHY section")
    i += 1

    parent_stack: List[int] = []
    current_joint: int | None = None
    last_token: str | None = None
    in_end_site = False

    # Traverse hierarchy until we reach "MOTION"
    while i < n:
        line = lines[i].strip()
        if line.upper().startswith("MOTION"):
            break

        if line.startswith("ROOT") or line.startswith("JOINT"):
            parts = line.split()
            if len(parts) >= 2:
                joint_name = parts[1]
                parent_idx = parent_stack[-1] if parent_stack else -1
                joint_idx = len(joint_names)
                joint_names.append(joint_name)
                parents.append(parent_idx)
                offsets_list.append((0.0, 0.0, 0.0))
                channel_counts.append(0)
                channel_starts.append(total_channels)
                current_joint = joint_idx
                last_token = "JOINT"
                in_end_site = False
        elif line.startswith("End Site"):
            current_joint = None
            last_token = "ENDSITE"
            in_end_site = True
        elif line.startswith("OFFSET") and current_joint is not None:
            parts = line.split()
            if len(parts) >= 4:
                offsets_list[current_joint] = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
        elif line.startswith("CHANNELS") and current_joint is not None:
            parts = line.split()
            try:
                count = int(parts[1])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed CHANNELS line: {line}")
            channel_counts[current_joint] = count
            total_channels += count
        elif line.startswith("{"):
            # Only push when opening a joint block, not an End Site.
            if last_token == "JOINT" and current_joint is not None:
                parent_stack.append(current_joint)
        elif line.startswith("}"):
            if in_end_site:
                # Closing an End Site block; do not pop parent joint.
                in_end_site = False
            else:
                if parent_stack:
                    parent_stack.pop()

        i += 1

    if len(joint_names) != len(channel_counts):
        # Mocopi BVH appears to assign CHANNELS to every joint including root.
        raise ValueError(
            f"Mismatch between joint count ({len(joint_names)}) "
            f"and channel count entries ({len(channel_counts)})"
        )

    motion_start_index = i
    offsets = np.asarray(offsets_list, dtype=float)
    return joint_names, channel_counts, motion_start_index, parents, offsets, channel_starts


def load_bvh(path: str | Path) -> MocopiSequence:
    """
    Load a Mocopi BVH file and return a MocopiSequence.

    This parser is tailored to the Mocopi ND pilot files and assumes:
        - Every ROOT / JOINT has a CHANNELS line.
        - The MOTION section contains:
            Frames: <int>
            Frame Time: <float>
            followed by one line per frame with all channel values.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Normalize newlines and split; keep simple for now.
    lines = text.splitlines()

    (
        joint_names,
        channel_counts,
        motion_start_idx,
        parents,
        offsets,
        channel_starts,
    ) = _parse_bvh_header(lines)

    # Find Frames / Frame Time
    i = motion_start_idx
    while i < len(lines) and not lines[i].strip().startswith("Frames:"):
        i += 1
    if i >= len(lines):
        raise ValueError("Could not find 'Frames:' line in BVH MOTION section")

    frames_line = lines[i].strip()
    try:
        num_frames = int(frames_line.split()[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed Frames line: {frames_line}") from exc

    i += 1
    if i >= len(lines) or not lines[i].strip().startswith("Frame Time:"):
        raise ValueError("Could not find 'Frame Time:' line in BVH MOTION section")
    frame_time_line = lines[i].strip()
    try:
        frame_time = float(frame_time_line.split()[2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed Frame Time line: {frame_time_line}") from exc

    # Remaining lines are motion data
    motion_lines = [ln.strip() for ln in lines[i + 1 :] if ln.strip()]
    if len(motion_lines) < num_frames:
        raise ValueError(
            f"Expected at least {num_frames} motion lines, found {len(motion_lines)}"
        )

    total_channels = sum(channel_counts)
    data = np.zeros((num_frames, total_channels), dtype=float)

    for frame_idx in range(num_frames):
        parts = motion_lines[frame_idx].split()
        if len(parts) < total_channels:
            raise ValueError(
                f"Frame {frame_idx} has {len(parts)} values, expected {total_channels}"
            )
        data[frame_idx, :] = np.asarray(parts[:total_channels], dtype=float)

    metadata = {
        "source_path": str(path),
        "frames": str(num_frames),
        "frame_time": str(frame_time),
    }

    return MocopiSequence(
        joint_names=joint_names,
        channel_counts=channel_counts,
        frame_time=frame_time,
        frames=data,
        metadata=metadata,
        parents=parents,
        offsets=offsets,
        channel_starts=channel_starts,
    )


def mocopi_to_frame_table(
    seq: MocopiSequence,
    joint_subset: List[str] | None = None,
) -> pd.DataFrame:
    """
    Convert a MocopiSequence into a long-form frame table similar to landmark CSVs.

    Columns:
        frame, timestamp_ms, source, joint_name, x, y, z
    """
    if joint_subset is None:
        joints = seq.joint_names
    else:
        joints = [j for j in joint_subset if j in seq.joint_names]

    timestamps = seq.timestamps_ms()
    records: List[Dict[str, float | int | str]] = []

    for frame_idx in range(seq.num_frames):
        t_ms = float(timestamps[frame_idx])
        for joint in joints:
            x, y, z = seq.joint_positions(joint)[frame_idx]
            records.append(
                {
                    "frame": frame_idx,
                    "timestamp_ms": t_ms,
                    "source": "mocopi",
                    "joint_name": joint,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                }
            )

    return pd.DataFrame.from_records(records)


def _rotation_matrix_zxy(z_deg: float, x_deg: float, y_deg: float) -> np.ndarray:
    """Construct a rotation matrix from BVH-style Z/X/Y Euler angles (degrees)."""
    z = np.deg2rad(z_deg)
    x = np.deg2rad(x_deg)
    y = np.deg2rad(y_deg)

    cz, sz = np.cos(z), np.sin(z)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)

    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)

    # Apply Z, then X, then Y: R = Ry * Rx * Rz
    return ry @ rx @ rz


def _forward_kinematics_joint(seq: MocopiSequence, joint_idx: int) -> np.ndarray:
    """
    Compute approximate world positions for a single joint across all frames.

    This walks the chain from root to the target joint using offsets and
    Z/X/Y rotation channels. Only root translation channels are applied.
    """
    num_frames = seq.num_frames
    positions = np.zeros((num_frames, 3), dtype=float)

    # Build ancestor chain from root to joint_idx (inclusive)
    chain: List[int] = []
    cur = joint_idx
    while cur != -1:
        chain.append(cur)
        cur = seq.parents[cur]
    chain = list(reversed(chain))

    root_idx = chain[0]
    root_start = seq.channel_starts[root_idx]

    # Indices for root translation and rotation channels
    root_tx = root_start + 0
    root_ty = root_start + 1
    root_tz = root_start + 2
    root_rz = root_start + 3
    root_rx = root_start + 4
    root_ry = root_start + 5

    for frame_idx in range(num_frames):
        channels = seq.frames[frame_idx]

        # Root translation and rotation
        root_pos = np.array(
            [channels[root_tx], channels[root_ty], channels[root_tz]], dtype=float
        )
        root_rot = _rotation_matrix_zxy(
            channels[root_rz], channels[root_rx], channels[root_ry]
        )

        pos = root_pos
        rot = root_rot

        # Walk down the chain from the joint after root
        for j in chain[1:]:
            offset = seq.offsets[j]
            pos = pos + rot @ offset

            start = seq.channel_starts[j]
            if seq.channel_counts[j] >= 6:
                rz = channels[start + 3]
                rx = channels[start + 4]
                ry = channels[start + 5]
                local_rot = _rotation_matrix_zxy(rz, rx, ry)
                rot = rot @ local_rot

        positions[frame_idx] = pos

    return positions
