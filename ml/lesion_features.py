"""Automatic lesion descriptors for the novelty analysis (fixed before any model was evaluated).

All measurements are taken on a hair-suppressed copy of the image (DullRazor-style black-hat +
inpainting). Hair removal is used for MEASUREMENT ONLY; model inputs are never hair-removed.

Primary scores (higher = harder case)
  border_haziness : 1 / edge_strength, where edge_strength is the peak |grad L*| (L* units per px)
                    within BAND_PX of each contour pixel, medianed along the contour. An edge is weak
                    when the transition is wide OR lesion/skin contrast is low; both are "indistinct".
  lab_sd          : total colour standard deviation inside the mask in CIELAB,
                    sqrt(var L* + var a* + var b*). Larger = more inhomogeneous.
Secondary
  border_width_px : contrast_L / edge_strength, the transition width (Gaussian edge: ~2.5 sigma).
                    Version 1 used this as the primary score; inspecting the training-set extremes
                    showed it calls faint, low-contrast lesions "sharp" (ratio of two small numbers),
                    so it was demoted before any evaluation results were seen.
  contrast_L      : |L_inner - L_outer| (low contrast is another form of indistinct border).
  ab_entropy      : Shannon entropy (bits) of the 16x16 a*b* histogram inside the mask.
  hair_frac       : fraction of pixels flagged as hair (artifact covariate).

Usage:
    python -m ml.lesion_features            # writes artifacts/novelty/lesion_features.csv
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ml.config import data_path, load_config

BAND_PX = 6          # search radius around each contour pixel for the peak gradient
RING_INNER = (8, 25) # distance range (px) inside the contour for the inner ring
RING_OUTER = (8, 25) # distance range (px) outside the contour for the outer ring
MIN_EDGE = 0.05      # L* units per px; floor so border_haziness stays finite


HAIR_MIN_LENGTH = 40    # px: long side of the component's rotated bounding box
HAIR_MAX_WIDTH = 6.0    # px: strand width, estimated as 4 x mean distance-to-edge inside the component
HAIR_MIN_ASPECT = 4.0   # a single strand is elongated...
HAIR_MAX_FILL = 0.25    # ...or several crossing strands leave their rotated box mostly empty
# Tuned for precision over recall: wider kernels / looser filters started erasing pigment network
# and small lesions on training images. Known limitation: very dense or thick hair is under-detected.


def hair_mask(rgb: np.ndarray) -> np.ndarray:
    """Long, thin dark structures (DullRazor black-hat + per-component shape filter).

    Short dots/globules (< HAIR_MIN_LENGTH), thick blobs (width) and dense pigment network
    (neither elongated nor sparse) are kept OUT of the mask: they are lesion texture that
    lab_sd is meant to measure.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    _, cand = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    dist = cv2.distanceTransform(cand, cv2.DIST_L2, 3)
    mean_dist = np.bincount(labels.ravel(), weights=dist.ravel(), minlength=n) / np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) < HAIR_MIN_LENGTH * 0.7 or 4 * mean_dist[i] > HAIR_MAX_WIDTH:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
        if long_side < HAIR_MIN_LENGTH:
            continue
        keep[i] = long_side / short_side >= HAIR_MIN_ASPECT or area / (long_side * short_side) <= HAIR_MAX_FILL
    mask = (keep[labels] * 255).astype(np.uint8)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def remove_hair(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Inpaint hair (measurement only). Returns (image, hair_frac)."""
    hair = hair_mask(rgb)
    if not hair.any():
        return rgb, 0.0
    return cv2.inpaint(rgb, hair, 3, cv2.INPAINT_TELEA), float((hair > 0).mean())


def compute_features(rgb: np.ndarray, mask: np.ndarray, hair_removal: bool = True) -> dict[str, float]:
    """rgb: HxWx3 uint8, mask: HxW bool (lesion = True). Returns NaNs if the mask is unusable."""
    h, w = mask.shape
    out = {"mask_area_frac": float(mask.mean()), "mask_touches_frame": bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())}
    nan = dict(border_haziness=np.nan, edge_strength=np.nan, border_width_px=np.nan, contrast_L=np.nan,
               lab_sd=np.nan, ab_entropy=np.nan, hair_frac=np.nan)
    if mask.sum() < 100 or (~mask).sum() < 100:
        return {**out, **nan}

    hair_frac = 0.0
    if hair_removal:
        rgb, hair_frac = remove_hair(rgb)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)  # L 0..100, a/b ~ -127..127
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)

    m8 = mask.astype(np.uint8)
    d_in = cv2.distanceTransform(m8, cv2.DIST_L2, 5)          # distance to background, inside lesion
    d_out = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5)     # distance to lesion, outside
    # Ignore pixels near the image frame (dermoscope vignette, masks clipped by the frame).
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]) < 10

    contour = mask & (d_in <= 1.5) & ~frame
    inner = mask & (d_in >= RING_INNER[0]) & (d_in <= RING_INNER[1]) & ~frame
    outer = ~mask & (d_out >= RING_OUTER[0]) & (d_out <= RING_OUTER[1]) & ~frame
    if contour.sum() < 30 or inner.sum() < 50 or outer.sum() < 50:
        peak, border_width, contrast = np.nan, np.nan, np.nan
    else:
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0   # Sobel 3x3 -> per-pixel units
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        grad = np.sqrt(gx ** 2 + gy ** 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND_PX + 1, 2 * BAND_PX + 1))
        peak = float(np.median(cv2.dilate(grad, kernel)[contour]))
        contrast = float(abs(L[inner].mean() - L[outer].mean()))
        border_width = contrast / max(peak, MIN_EDGE)

    pix = lab[mask]
    lab_sd = float(np.sqrt(pix.var(axis=0).sum()))
    hist, _, _ = np.histogram2d(pix[:, 1], pix[:, 2], bins=16, range=[[-64, 64], [-64, 64]])
    p = hist.ravel() / max(hist.sum(), 1)
    p = p[p > 0]
    ab_entropy = float(-(p * np.log2(p)).sum())
    haziness = 1.0 / max(peak, MIN_EDGE) if peak == peak else np.nan
    return {**out, "border_haziness": float(haziness), "edge_strength": float(peak),
            "border_width_px": float(border_width), "contrast_L": float(contrast),
            "lab_sd": lab_sd, "ab_entropy": ab_entropy, "hair_frac": hair_frac}


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def _row_features(args: tuple[str, str, str]) -> dict:
    image_id, img_path, mask_path = args
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = read_mask(Path(mask_path), rgb.shape[:2])
    return {"image_id": image_id, **compute_features(rgb, mask)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cfg = load_config()
    root = Path(cfg["data_root"])
    df = pd.read_csv(data_path(cfg, "splits") / "manifest_split.csv")
    df = df[df["mask_exists"]]
    jobs = [(r.image_id, str(root / r.image_path), str(root / r.mask_path)) for r in df.itertuples()]
    with Pool(args.workers) as pool:
        rows = list(tqdm(pool.imap(_row_features, jobs, chunksize=32), total=len(jobs), desc="lesion features"))
    feats = pd.DataFrame(rows).merge(df[["image_id", "lesion_id", "split", "label", "dx"]], on="image_id")
    out = data_path(cfg, "artifacts") / "novelty"
    out.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out / "lesion_features.csv", index=False)
    print(feats.groupby("split")[["border_haziness", "lab_sd", "border_width_px", "contrast_L", "hair_frac"]]
          .describe().T.round(3))


if __name__ == "__main__":
    main()
