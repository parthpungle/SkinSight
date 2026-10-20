"""Manual rubric used to VALIDATE the automatic lesion scores (not as the primary method).

Rubric (annotate from the image only, blind to label and prediction):
  border_clarity : 0 = clear, 1 = partly hazy, 2 = highly indistinct
  uniformity     : 0 = relatively homogeneous, 1 = mixed, 2 = strongly inhomogeneous
  artifacts      : any of hair / illumination / low_contrast / other (semicolon-separated), or blank
Record annotator name and date; if two annotators label the same subset, report agreement (Cohen's kappa).

Usage:
    python -m ml.rubric sample --n 100          # writes artifacts/novelty/rubric_template.csv (val split)
    python -m ml.rubric validate --file artifacts/novelty/rubric_annotations.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from ml.config import data_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=123)
    v = sub.add_parser("validate")
    v.add_argument("--file", required=True)
    args = parser.parse_args()

    cfg = load_config()
    root, out = Path(cfg["data_root"]), data_path(cfg, "artifacts") / "novelty"
    feats = pd.read_csv(out / "lesion_features.csv")

    if args.cmd == "sample":
        manifest = pd.read_csv(data_path(cfg, "splits") / "manifest_split.csv")
        val = feats[feats["split"] == "val"].dropna(subset=["border_haziness", "lab_sd"])
        pick = val.sample(n=min(args.n, len(val)), random_state=args.seed)
        tpl = pick[["image_id"]].merge(manifest[["image_id", "image_path"]], on="image_id")
        tpl["image_path"] = tpl["image_path"].map(lambda p: str(root / p))
        for c in ("border_clarity", "uniformity", "artifacts", "annotator", "date"):
            tpl[c] = ""
        tpl.to_csv(out / "rubric_template.csv", index=False)
        print(f"wrote {out / 'rubric_template.csv'} ({len(tpl)} images). Labels/predictions are deliberately omitted.")
        return

    ann = pd.read_csv(args.file).merge(feats, on="image_id")
    ann = ann.dropna(subset=["border_clarity", "uniformity"])
    res = {}
    for manual, auto in (("border_clarity", "border_haziness"), ("border_clarity", "border_width_px"),
                         ("border_clarity", "contrast_L"), ("uniformity", "lab_sd"), ("uniformity", "ab_entropy")):
        rho, p = spearmanr(ann[manual], ann[auto], nan_policy="omit")
        res[f"{manual}~{auto}"] = {"spearman_rho": float(rho), "p_value": float(p), "n": int(len(ann))}
    (out / "rubric_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
