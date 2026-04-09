from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LandmarkViews:
    image_csv: Path
    image_df: pd.DataFrame
    metadata_path: Path | None
    metadata: dict | None
    world_csv: Path | None = None
    world_df: pd.DataFrame | None = None


def landmark_stem_from_image_csv(image_csv: Path) -> str:
    stem = image_csv.stem
    if stem.startswith("landmarks_"):
        return stem[len("landmarks_") :]
    return stem


def infer_metadata_json(image_csv: Path, project_root: Path | None = None) -> Path:
    stem = landmark_stem_from_image_csv(image_csv)
    base = project_root if project_root is not None else image_csv.parent.parent.parent
    return (base / "results" / "OutputCSVs" / f"landmarks_metadata_{stem}.json").resolve()


def infer_pose_world_csv(image_csv: Path, project_root: Path | None = None) -> Path:
    stem = landmark_stem_from_image_csv(image_csv)
    base = project_root if project_root is not None else image_csv.parent.parent.parent
    return (base / "results" / "OutputCSVs" / f"pose_world_{stem}.csv").resolve()


def load_processing_metadata(image_csv: Path, project_root: Path | None = None) -> tuple[Path | None, dict | None]:
    metadata_path = infer_metadata_json(image_csv, project_root)
    if not metadata_path.is_file():
        return None, None
    try:
        return metadata_path, json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return metadata_path, None


def load_landmark_views(
    image_csv: Path,
    *,
    project_root: Path | None = None,
    world_csv: Path | None = None,
    require_world: bool = False,
) -> LandmarkViews:
    resolved_image_csv = image_csv.resolve()
    image_df = pd.read_csv(resolved_image_csv)
    metadata_path, metadata = load_processing_metadata(resolved_image_csv, project_root)

    resolved_world_csv = world_csv.resolve() if world_csv is not None else None
    if resolved_world_csv is None and metadata is not None and metadata.get("pose_world_csv"):
        candidate = Path(metadata["pose_world_csv"]).resolve()
        if candidate.is_file():
            resolved_world_csv = candidate
    if resolved_world_csv is None:
        candidate = infer_pose_world_csv(resolved_image_csv, project_root)
        if candidate.is_file():
            resolved_world_csv = candidate

    world_df = None
    if resolved_world_csv is not None and resolved_world_csv.is_file():
        world_df = pd.read_csv(resolved_world_csv)
    elif require_world:
        raise FileNotFoundError(
            f"Could not find pose-world landmarks CSV for {resolved_image_csv}. "
            "Run preprocess with the new world-landmark output first."
        )

    return LandmarkViews(
        image_csv=resolved_image_csv,
        image_df=image_df,
        world_csv=resolved_world_csv,
        world_df=world_df,
        metadata_path=metadata_path,
        metadata=metadata,
    )


__all__ = [
    "LandmarkViews",
    "infer_metadata_json",
    "infer_pose_world_csv",
    "landmark_stem_from_image_csv",
    "load_landmark_views",
    "load_processing_metadata",
]
