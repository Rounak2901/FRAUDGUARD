"""Load and expose project configuration from config.yaml and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config and apply a few environment variable overrides."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Environment overrides (API / DB)
    if os.getenv("FRAUD_GUARD_DB_URL"):
        cfg.setdefault("api", {})["db_url"] = os.environ["FRAUD_GUARD_DB_URL"]
    if os.getenv("FRAUD_GUARD_PORT"):
        cfg.setdefault("api", {})["port"] = int(os.environ["FRAUD_GUARD_PORT"])
    if os.getenv("FRAUD_GUARD_SEED"):
        cfg.setdefault("project", {})["seed"] = int(os.environ["FRAUD_GUARD_SEED"])

    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a paths.* entry relative to the project root."""
    rel = cfg["paths"][key]
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create data / model / figure directories if missing."""
    for key in ("raw_dir", "processed_dir", "models_dir", "figures_dir"):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
