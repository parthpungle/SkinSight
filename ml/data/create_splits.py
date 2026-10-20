"""Lesion-grouped, label-stratified train/val/test split.

Usage:
    python -m ml.data.create_splits

Reads <data_root>/splits/manifest.csv, writes manifest_split.csv and split_report.json.
Splitting is done on lesions (not images) so no lesion appears in two splits.
HAM10000 has no patient_id, so patient-level leakage cannot be ruled out (documented limitation).
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import data_path, load_config


def split_by_group(df: pd.DataFrame, group_col: str, label_col: str,
                   val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    """Return df with a `split` column; every group lands in exactly one split."""
    lesions = df.groupby(group_col)[label_col].first().reset_index()
    trainval, test = train_test_split(lesions, test_size=test_frac, stratify=lesions[label_col], random_state=seed)
    train, val = train_test_split(trainval, test_size=val_frac / (1 - test_frac),
                                  stratify=trainval[label_col], random_state=seed)
    assignment = {g: s for s, part in (("train", train), ("val", val), ("test", test)) for g in part[group_col]}
    out = df.copy()
    out["split"] = out[group_col].map(assignment)
    return out


def main() -> None:
    cfg = load_config()
    sp = cfg["split"]
    splits_dir = data_path(cfg, "splits")
    df = pd.read_csv(splits_dir / "manifest.csv")

    out = split_by_group(df, sp["group_by"], "label", sp["val"], sp["test"], sp["seed"])
    out.to_csv(splits_dir / "manifest_split.csv", index=False)

    report = {"seed": sp["seed"], "group_by": sp["group_by"], "splits": {}}
    for name, part in out.groupby("split"):
        report["splits"][name] = {
            "n_images": int(len(part)),
            "n_lesions": int(part[sp["group_by"]].nunique()),
            "n_malignant": int(part["label"].sum()),
            "malignant_frac": round(float(part["label"].mean()), 4),
        }
    (splits_dir / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
