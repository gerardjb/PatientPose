from __future__ import annotations

from typing import Sequence


ROLE_ALIASES = {
    "A": "A",
    "UNFILTERED": "A",
    "ND": "ND",
}


def _normalize_role(role: str) -> str:
    normalized = ROLE_ALIASES.get(role.strip().upper())
    if normalized is None:
        allowed = ", ".join(sorted(ROLE_ALIASES))
        raise ValueError(f"Unknown camera role {role!r}; expected one of: {allowed}")
    return normalized


def parse_camera_role_specs(specs: Sequence[str] | None) -> dict[str, str]:
    """
    Parse CLI-friendly role mappings like:
        192.168.50.162=A
        phone_192.168.50.171=ND
    """
    mapping: dict[str, str] = {}
    if not specs:
        return mapping

    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid camera-role mapping {spec!r}; expected CAMERA_ID=ROLE")
        camera_id, role = spec.split("=", 1)
        camera_id = camera_id.strip()
        if not camera_id:
            raise ValueError(f"Invalid camera-role mapping {spec!r}; camera id is empty")
        mapping[camera_id] = _normalize_role(role)
    return mapping
