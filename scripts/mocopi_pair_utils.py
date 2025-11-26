from __future__ import annotations

"""
Helpers to pair unfiltered/ND video trials with matching Mocopi BVH files.

Assumptions:
    - Unfiltered videos: sample_data/A_<tag>.mp4 (e.g., A_1a_20140107_141140.mp4)
    - ND videos:        sample_data/ND_<tag>.mp4 (e.g., ND_1a_20140107_104046.mp4)
    - Mocopi BVH:       sample_data/ND_pilot/'Re_ Mocopi'/MCPM_*_<tag>.bvh

This utility parses tags (e.g., "1a") and finds matching paths.
"""

from mocopi.nd_pilot import TrialPair, discover_pairs, pair_for_tag

__all__ = ["TrialPair", "discover_pairs", "pair_for_tag"]
