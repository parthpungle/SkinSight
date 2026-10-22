"""Read-only index of test-split images with lesion scores and the reference model's saved score."""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

from ml.stratified_analysis import assign_tertiles

log = logging.getLogger(__name__)
IMAGE_ID = re.compile(r"^ISIC_\d{7}$")
REFERENCE_VARIANT = "efficientnet_b0"
FIELDS = ("image_id", "lesion_id", "dx", "label", "border_haziness", "haziness_tertile", "lab_sd",
          "colour_tertile", "contrast_L", "hair_frac", "ref_score", "ref_pred", "ref_threshold")


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


class SampleStore:
    """Joins manifest (test split), lesion features and the served reference model's test predictions."""

    def __init__(self, data_root: Path, splits_dir: Path, artifacts: Path):
        self.data_root = data_root
        self.available = False
        self.reference_run: str | None = None
        self.df = pd.DataFrame()
        manifest = splits_dir / "manifest_split.csv"
        feats_path = artifacts / "novelty" / "lesion_features.csv"
        if not manifest.is_file():
            return
        try:
            m = pd.read_csv(manifest)
            test = m[m["split"] == "test"][["image_id", "lesion_id", "dx", "label", "image_path"]].copy()
            if feats_path.is_file():
                f = pd.read_csv(feats_path)
                train = f[f["split"] == "train"]
                ft = f[f["split"] == "test"].copy()
                ft["haziness_tertile"] = assign_tertiles(ft["border_haziness"], train["border_haziness"]).astype(str)
                ft["colour_tertile"] = assign_tertiles(ft["lab_sd"], train["lab_sd"]).astype(str)
                cols = ["image_id", "border_haziness", "haziness_tertile", "lab_sd", "colour_tertile",
                        "contrast_L", "hair_frac"]
                test = test.merge(ft[cols], on="image_id", how="left")
            self._attach_reference(test, artifacts)
            self.df = test.sort_values("image_id").reset_index(drop=True)
            self.ids = set(self.df["image_id"])
            self.available = True
        except Exception:
            log.exception("could not build the sample index")

    def _attach_reference(self, test: pd.DataFrame, artifacts: Path) -> None:
        summary = artifacts / "metrics" / "summary.json"
        preds = artifacts / "metrics" / "test_predictions.csv"
        for c in ("ref_score", "ref_pred", "ref_threshold"):
            test[c] = None
        if not (summary.is_file() and preds.is_file()):
            return
        run = json.loads(summary.read_text(encoding="utf-8")).get("served_checkpoints", {}).get(REFERENCE_VARIANT)
        if not run:
            return
        p = pd.read_csv(preds)
        p = p[p["run"] == run].set_index("image_id")
        self.reference_run = run
        test["ref_score"] = test["image_id"].map(p["score_malignant"])
        test["ref_pred"] = test["image_id"].map(p["pred_label"])
        if "threshold" in p.columns:
            test["ref_threshold"] = test["image_id"].map(p["threshold"])

    def _row(self, r: pd.Series) -> dict:
        return {k: _clean(r.get(k)) for k in FIELDS}

    def query(self, label: int | None = None, dx: str | None = None, haziness_tertile: str | None = None,
              colour_tertile: str | None = None, disagree: bool = False, offset: int = 0, limit: int = 50) -> dict:
        df = self.df
        if label is not None:
            df = df[df["label"] == label]
        if dx:
            df = df[df["dx"] == dx]
        if haziness_tertile and "haziness_tertile" in df:
            df = df[df["haziness_tertile"] == haziness_tertile]
        if colour_tertile and "colour_tertile" in df:
            df = df[df["colour_tertile"] == colour_tertile]
        if disagree:
            df = df[df["ref_pred"].notna() & (df["ref_pred"] != df["label"])]
        page = df.iloc[offset:offset + limit]
        return {"available": True, "total": int(len(df)), "offset": offset, "limit": limit,
                "reference_run": self.reference_run,
                "dx_values": sorted(self.df["dx"].unique().tolist()),
                "items": [self._row(r) for _, r in page.iterrows()]}

    def get(self, image_id: str) -> dict | None:
        if not IMAGE_ID.match(image_id) or image_id not in self.ids:
            return None
        return self._row(self.df[self.df["image_id"] == image_id].iloc[0])

    def image_path(self, image_id: str) -> Path | None:
        if not IMAGE_ID.match(image_id) or image_id not in self.ids:
            return None
        rel = self.df.loc[self.df["image_id"] == image_id, "image_path"].iloc[0]
        path = (self.data_root / rel).resolve()
        return path if path.is_file() and path.is_relative_to(self.data_root.resolve()) else None
