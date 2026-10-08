"""Configuration loading.

The YAML file in configs/config.yaml is the single source of thresholds and
weights. A few values can be overridden from the environment so deployments do
not need to edit the file (MODEL_PATH, SIH_CONFIG, SIH_STORAGE, SIH_POSTGIS_DSN).
"""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """dict with attribute access and project-root-relative path resolution."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def path(self, value: str | os.PathLike) -> Path:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("SIH_CONFIG", DEFAULT_CONFIG_PATH))
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if os.environ.get("MODEL_PATH"):
        data.setdefault("segmentation", {})["model_path"] = os.environ["MODEL_PATH"]
    if os.environ.get("SIH_STORAGE"):
        data.setdefault("storage", {})["backend"] = os.environ["SIH_STORAGE"]
    if os.environ.get("SIH_POSTGIS_DSN"):
        data.setdefault("storage", {})["postgis_dsn"] = os.environ["SIH_POSTGIS_DSN"]
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
