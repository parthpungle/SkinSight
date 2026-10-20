"""Grad-CAM (pytorch-grad-cam) helpers and the CAM-inside-lesion analysis.

Grad-CAM is a qualitative, gradient-based visualisation. It is not a segmentation, not a clinical
explanation, and not proof of what the model "reasons" about.

Usage:
    python -m ml.explainability --run efficientnet_b0_s0 [--n-examples 12]
Writes artifacts/novelty/cam_in_mask_<run>.csv and sample overlays under artifacts/figures/gradcam/<run>/.
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

from ml.config import data_path, load_config
from ml.lesion_features import read_mask
from ml.models import gradcam_target_layer

CAM_OPACITY = 0.45  # weight of the heatmap in the overlay; colormap = OpenCV JET (blue low, red high)


def gradcam_map(model: torch.nn.Module, model_id: str, x: torch.Tensor, target_class: int) -> np.ndarray:
    """Return a (H, W) Grad-CAM map in [0, 1] for one image batch x of shape (1, 3, H, W)."""
    with GradCAM(model=model, target_layers=[gradcam_target_layer(model, model_id)]) as cam:
        maps = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])
    m = maps[0]
    if not np.isfinite(m).all():
        raise ValueError("Grad-CAM produced non-finite values")
    return m


def overlay(rgb01: np.ndarray, cam_map: np.ndarray) -> np.ndarray:
    return show_cam_on_image(rgb01, cam_map, use_rgb=True, image_weight=1 - CAM_OPACITY)


def to_png_base64(rgb_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def cam_in_mask_fraction(cam_map: np.ndarray, mask: np.ndarray) -> float:
    """Share of total CAM energy that falls inside the lesion mask (same resolution)."""
    total = float(cam_map.sum())
    return float((cam_map * mask).sum() / total) if total > 0 else float("nan")


def main() -> None:
    from ml.evaluate import load_checkpoint
    from ml.preprocessing import display_image, preprocess_for_inference
    from ml.train import get_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n-examples", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config()
    root, art = Path(cfg["data_root"]), data_path(cfg, "artifacts")
    device = get_device()
    model, ckpt = load_checkpoint(art / "models" / f"{args.run}.pth", device)
    cc, size, thr = bool(ckpt.get("color_constancy")), ckpt["image_size"], ckpt["val_threshold"]
    df = pd.read_csv(data_path(cfg, "splits") / "manifest_split.csv")
    df = df[(df["split"] == args.split) & df["mask_exists"]].reset_index(drop=True)

    fig_dir = art / "figures" / "gradcam" / args.run
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, cams = [], {}
    for r in tqdm(df.itertuples(), total=len(df), desc=f"grad-cam {args.run}"):
        x = preprocess_for_inference(root / r.image_path, cc).to(device)
        with torch.no_grad():
            score = float(torch.softmax(model(x), 1)[0, 1])
        pred = int(score >= thr)
        cmap = gradcam_map(model, ckpt["model_id"], x, pred)
        mask = read_mask(root / r.mask_path, (size, size))
        rows.append({"image_id": r.image_id, "true_label": r.label, "pred_label": pred, "score_malignant": score,
                     "cam_in_mask": cam_in_mask_fraction(cmap, mask), "mask_area_frac": float(mask.mean())})
        cams[r.image_id] = cmap.astype(np.float16)

    res = pd.DataFrame(rows)
    # Save a few overlays per outcome (TP/TN/FP/FN) for qualitative review.
    res["outcome"] = np.select(
        [(res.true_label == 1) & (res.pred_label == 1), (res.true_label == 0) & (res.pred_label == 0),
         (res.true_label == 0) & (res.pred_label == 1)], ["TP", "TN", "FP"], "FN")
    paths = dict(zip(df["image_id"], df["image_path"]))
    for outcome, part in res.groupby("outcome"):
        for image_id in part["image_id"].head(max(1, args.n_examples // 4)):
            rgb01 = display_image(root / paths[image_id], size)
            overlay_img = overlay(rgb01, cams[image_id].astype(np.float32))
            Image.fromarray(np.hstack([(rgb01 * 255).astype(np.uint8), overlay_img])).save(
                fig_dir / f"{outcome}_{image_id}.png")
    out = art / "novelty"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"cam_in_mask_{args.run}_{args.split}.csv", index=False)
    print(res.groupby("outcome")["cam_in_mask"].describe().round(3))


if __name__ == "__main__":
    main()
