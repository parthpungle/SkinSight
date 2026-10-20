"""Build and validate the HAM10000 manifest.

Usage:
    python -m ml.data.build_manifest [--skip-decode-check]

Writes <data_root>/splits/manifest.csv and manifest_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ml.config import data_path, load_config


def map_labels(df: pd.DataFrame, malignant: list[str], benign: list[str]) -> pd.DataFrame:
    """Add the binary `label` column (1 = malignant). Unknown diagnoses raise."""
    unknown = set(df["dx"]) - set(malignant) - set(benign)
    if unknown:
        raise ValueError(f"Unmapped diagnoses: {sorted(unknown)}")
    df = df.copy()
    df["label"] = df["dx"].isin(malignant).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-decode-check", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    root = Path(cfg["data_root"])
    images_dir = data_path(cfg, "images")
    masks_dir = data_path(cfg, "masks")
    out_dir = data_path(cfg, "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(data_path(cfg, "metadata"))
    required = {"lesion_id", "image_id", "dx"}
    missing_cols = required - set(meta.columns)
    if missing_cols:
        raise ValueError(f"Metadata missing columns: {sorted(missing_cols)}")
    if meta["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values in metadata")

    df = map_labels(meta, cfg["dataset"]["malignant"], cfg["dataset"]["benign"])

    # Lesions with several images must share one diagnosis, otherwise grouping is ambiguous.
    dx_per_lesion = df.groupby("lesion_id")["dx"].nunique()
    inconsistent = dx_per_lesion[dx_per_lesion > 1].index.tolist()
    if inconsistent:
        raise ValueError(f"{len(inconsistent)} lesions have conflicting diagnoses, e.g. {inconsistent[:5]}")

    df["image_path"] = df["image_id"].map(lambda i: (images_dir / f"{i}.jpg").relative_to(root).as_posix())
    df["mask_path"] = df["image_id"].map(
        lambda i: (masks_dir / f"{i}_segmentation.png").relative_to(root).as_posix()
    )

    df["image_exists"] = df["image_path"].map(lambda p: (root / p).is_file())
    df["mask_exists"] = df["mask_path"].map(lambda p: (root / p).is_file())

    widths, heights, readable = [], [], []
    for p, exists in tqdm(zip(df["image_path"], df["image_exists"]), total=len(df), desc="checking images"):
        if not exists or args.skip_decode_check:
            widths.append(None), heights.append(None), readable.append(exists)
            continue
        try:
            with Image.open(root / p) as im:
                im.verify()
            with Image.open(root / p) as im:
                widths.append(im.width), heights.append(im.height)
            readable.append(True)
        except Exception:
            widths.append(None), heights.append(None), readable.append(False)
    df["width"], df["height"], df["readable"] = widths, heights, readable

    bad = df[~df["image_exists"] | ~df["readable"]]
    df = df[df["image_exists"] & df["readable"]]

    cols = ["image_id", "lesion_id", "dx", "dx_type", "label", "image_path", "mask_path", "mask_exists",
            "width", "height", "age", "sex", "localization"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_dir / "manifest.csv", index=False)

    report = {
        "dataset": cfg["dataset"]["name"],
        "n_images": int(len(df)),
        "n_lesions": int(df["lesion_id"].nunique()),
        "n_excluded_missing_or_unreadable": int(len(bad)),
        "excluded_ids": bad["image_id"].tolist()[:50],
        "n_missing_masks": int((~df["mask_exists"]).sum()),
        "class_counts": {cfg["dataset"]["class_names"][k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "dx_counts": {k: int(v) for k, v in df["dx"].value_counts().items()},
        "image_sizes": {f"{w}x{h}": int(n) for (w, h), n in df.groupby(["width", "height"]).size().items()},
        "label_mapping": {"malignant": cfg["dataset"]["malignant"], "benign": cfg["dataset"]["benign"]},
    }
    (out_dir / "manifest_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
