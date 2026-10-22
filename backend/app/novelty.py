"""Saved novelty-analysis outputs for the Novelty tab (read-only; never recomputes metrics)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

CAM_FILE = re.compile(r"^cam_in_mask_(?P<run>.+)_(?P<split>val|test)\.csv$")
OUTCOMES = ("TP", "FP", "TN", "FN")


def _read_json(path: Path) -> dict | list | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def outcome_enrichment(cam: pd.DataFrame) -> dict:
    """Median size-normalised Grad-CAM enrichment (cam_in_mask / mask_area_frac) per TP/FP/TN/FN."""
    outcome = pd.Series("TN", index=cam.index)
    outcome[(cam.true_label == 1) & (cam.pred_label == 1)] = "TP"
    outcome[(cam.true_label == 0) & (cam.pred_label == 1)] = "FP"
    outcome[(cam.true_label == 1) & (cam.pred_label == 0)] = "FN"
    enrich = cam["cam_in_mask"] / cam["mask_area_frac"].clip(lower=1e-3)
    out = {}
    for o in OUTCOMES:
        sel = enrich[outcome == o]
        out[o] = {"median": float(sel.median()) if len(sel) else None, "n": int(len(sel))}
    return out


def load_novelty(artifacts: Path) -> dict:
    nov = artifacts / "novelty"
    test = _read_json(nov / "stratified_test.json")
    if test is None:
        return {"available": False}
    enrichment: dict = {"val": {}, "test": {}}
    for p in sorted(nov.glob("cam_in_mask_*.csv")):
        m = CAM_FILE.match(p.name)
        if m:
            enrichment[m["split"]][m["run"]] = outcome_enrichment(pd.read_csv(p))
    return {
        "available": True,
        "stratified_test": test,
        "stratified_val": _read_json(nov / "stratified_val.json"),
        "enrichment": enrichment,
        "verdicts": _read_json(nov / "verdicts.json") or [],
    }
