from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    sample_data: Path
    results: Path
    output_videos: Path
    output_csvs: Path
    output_plots: Path
    models: Path
    orientation_debug: Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Resolve the PatientPose project root from an explicit path or the current working directory.

    The packaged CLI should not depend on the install location of the Python package, because
    that breaks for non-editable installs. Treat the repo root as a workflow concern instead.
    """
    candidate = (start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise FileNotFoundError(
        f"Could not locate the PatientPose project root from {candidate}. "
        "Pass --project-root explicitly or run the command from inside the repo."
    )


def resolve_project_paths(project_root: Path | None = None) -> ProjectPaths:
    root = find_project_root(project_root)
    results = root / "results"
    return ProjectPaths(
        root=root,
        sample_data=root / "sample_data",
        results=results,
        output_videos=results / "OutputVideos",
        output_csvs=results / "OutputCSVs",
        output_plots=results / "OutputPlots",
        models=root / "models",
        orientation_debug=results / "orientation_debug",
    )
