"""Loads the served checkpoint for each model variant once at startup."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ml.models import MODEL_NAMES, build_model

log = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    id: str
    name: str
    ready: bool
    model: torch.nn.Module | None = None
    ckpt: dict = field(default_factory=dict)
    run: str | None = None


def _variant(run: str) -> str:
    return run.rsplit("_s", 1)[0]


def _served_runs(artifacts: Path) -> dict[str, str]:
    """variant -> run. Prefer the evaluation's choice (best val AUROC); else the lowest seed on disk."""
    summary = artifacts / "metrics" / "summary.json"
    if summary.is_file():
        served = json.loads(summary.read_text(encoding="utf-8")).get("served_checkpoints", {})
        if served:
            return served
    runs: dict[str, str] = {}
    for p in sorted((artifacts / "models").glob("*.pth")):
        if p.stem.endswith("_smoke"):
            continue
        runs.setdefault(_variant(p.stem), p.stem)
    return runs


def load_registry(artifacts: Path, device: torch.device) -> dict[str, ModelEntry]:
    registry: dict[str, ModelEntry] = {}
    served = _served_runs(artifacts)
    for model_id in MODEL_NAMES:  # always list the three core models, ready or not
        registry[model_id] = ModelEntry(model_id, MODEL_NAMES[model_id], ready=False)
    for variant, run in served.items():
        path = artifacts / "models" / f"{run}.pth"
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model = build_model(ckpt["model_id"], pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
            model.to(device).eval()
        except Exception:
            log.exception("could not load checkpoint for %s", variant)
            continue
        name = MODEL_NAMES[ckpt["model_id"]] + (" + colour constancy" if ckpt.get("color_constancy") else "")
        ckpt.pop("state_dict")
        registry[variant] = ModelEntry(variant, name, True, model, ckpt, run)
    return registry
