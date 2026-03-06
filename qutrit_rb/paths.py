"""Project-relative path helpers for scripts, plots, and result artifacts."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


def resolve_project_path(path_like: str, prefer_results: bool = False) -> Path:
    """
    Resolve relative paths against the repository root.

    Existing explicit paths win. When `prefer_results=True`, bare relative
    names also map into `results/` so older checkpoint names still resolve
    after artifacts are moved under that directory.
    """
    path = Path(path_like)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    repo_candidate = PROJECT_ROOT / path

    if cwd_candidate.exists():
        return cwd_candidate
    if repo_candidate.exists():
        return repo_candidate

    if prefer_results:
        results_candidate = (
            PROJECT_ROOT / path
            if path.parts and path.parts[0] == "results"
            else RESULTS_ROOT / path
        )
        if results_candidate.exists():
            return results_candidate
        return results_candidate

    return repo_candidate
