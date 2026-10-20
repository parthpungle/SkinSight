"""Load ml/config.yaml and resolve data paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = os.environ.get("SKINSIGHT_DATA", cfg["data_root"])
    return cfg


def data_path(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path for an entry under cfg['paths']."""
    return Path(cfg["data_root"]) / cfg["paths"][key]
