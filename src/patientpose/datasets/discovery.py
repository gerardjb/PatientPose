from __future__ import annotations

from pathlib import Path
from typing import Optional

from .legacy_nd_pilot import discover_legacy_pairs
from .models import CaptureSession, TrialPair
from .session_layout import discover_sessions, resolve_session_pair


def discover_pairs(
    base: Path,
    camera_roles: dict[str, str] | None = None,
    include_legacy: bool = True,
) -> list[TrialPair]:
    base = base.resolve()
    pairs: list[TrialPair] = []
    if include_legacy:
        pairs.extend(discover_legacy_pairs(base))

    if camera_roles:
        for session in discover_sessions(base):
            pairs.append(resolve_session_pair(session, camera_roles))

    return pairs


def pair_for_tag(
    base: Path,
    tag: str,
    camera_roles: dict[str, str] | None = None,
    include_legacy: bool = True,
) -> Optional[TrialPair]:
    for pair in discover_pairs(base, camera_roles=camera_roles, include_legacy=include_legacy):
        if pair.tag == tag:
            return pair
    return None


def session_lookup(base: Path) -> dict[str, CaptureSession]:
    return {session.session_id: session for session in discover_sessions(base)}
