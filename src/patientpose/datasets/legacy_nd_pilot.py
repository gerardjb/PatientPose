from __future__ import annotations

from pathlib import Path

from .models import TrialPair


def discover_legacy_pairs(base: Path) -> list[TrialPair]:
    sample_dir = base / "sample_data"
    nd_dir = sample_dir / "ND_pilot"
    bvh_dir = nd_dir / "Re_ Mocopi"
    if not nd_dir.is_dir() or not bvh_dir.is_dir():
        return []

    bvh_map: dict[str, Path] = {}
    for bvh in bvh_dir.glob("MCPM_*_*.bvh"):
        tag = bvh.stem.split("_")[-1]
        bvh_map[tag] = bvh

    pairs: list[TrialPair] = []
    for nd_video in sorted(nd_dir.glob("ND_*.mp4")):
        parts = nd_video.stem.split("_")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag not in bvh_map:
            continue
        candidates = sorted(nd_dir.glob(f"A_{tag}_*.mp4"))
        if not candidates:
            continue
        pairs.append(
            TrialPair(
                tag=tag,
                unfiltered_video=candidates[0],
                nd_video=nd_video,
                motion_source=bvh_map[tag],
                mode="legacy",
            )
        )
    return pairs
