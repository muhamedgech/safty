"""Load the YAML config and resolve its paths relative to the project root."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read the config file and turn every entry under `paths` into an absolute Path."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"] = {name: resolve(p) for name, p in config["paths"].items()}
    return config


def resolve(path: str | Path) -> Path:
    """Relative paths are relative to the project root, not the current directory."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
