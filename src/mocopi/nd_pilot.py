from __future__ import annotations

"""
Utilities for discovering ND-pilot A/ND/BVH triplets.

These helpers centralize the dataset-specific naming assumptions used across
the Mocopi analysis scripts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TrialPair:
    tag: str
    unfiltered_video: Path
    nd_video: Path
    bvh: Path


def discover_pairs(base: Path) -> List[TrialPair]:
    """
    Discover A_* and ND_* video pairs and match them to Mocopi BVH by suffix tag.

    Assumptions (ND pilot):
        - Unfiltered videos: sample_data/A_<tag>.mp4 (e.g., A_1a_20140107_141140.mp4)
        - ND videos:        sample_data/ND_<tag>.mp4 (e.g., ND_1a_20140107_104046.mp4)
        - Mocopi BVH:       sample_data/ND_pilot/'Re_ Mocopi'/MCPM_*_<tag>.bvh
    """
    base = base.resolve()
    sample_dir = base / "sample_data"
    nd_dir = sample_dir / "ND_pilot"
    bvh_dir = nd_dir / "Re_ Mocopi"

    bvh_map = {}
    for bvh in bvh_dir.glob("MCPM_*_*.bvh"):
        tag = bvh.stem.split("_")[-1]
        bvh_map[tag] = bvh

    pairs: List[TrialPair] = []
    for nd_video in nd_dir.glob("ND_*.mp4"):
        parts = nd_video.stem.split("_")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag not in bvh_map:
            continue
        candidates = sorted(nd_dir.glob(f"A_{tag}_*.mp4"))
        if not candidates:
            continue
        unfiltered = candidates[0]
        pairs.append(
            TrialPair(
                tag=tag,
                unfiltered_video=unfiltered,
                nd_video=nd_video,
                bvh=bvh_map[tag],
            )
        )
    return pairs


def pair_for_tag(base: Path, tag: str) -> Optional[TrialPair]:
    """
    Return the TrialPair for a specific tag (e.g., "1a") if present.
    """
    for pair in discover_pairs(base):
        if pair.tag == tag:
            return pair
    return None


__all__ = ["TrialPair", "discover_pairs", "pair_for_tag"]
