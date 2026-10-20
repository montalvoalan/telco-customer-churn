"""Small path helpers shared by project entry points."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    return Path(__file__).resolve().parents[2]


def default_mlflow_uri(project_root: Path | None = None) -> str:
    """Return the local SQLite tracking URI used by training."""
    root = project_root or get_project_root()
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}"
