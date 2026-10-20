"""Evaluate saved checkpoints on the untouched test split.

Usage:
    python -m ml.evaluate                 # every non-smoke checkpoint in artifacts/models
    python -m ml.evaluate --runs efficientnet_b0_s0 mobilenet_v2_s0

For each run the operating threshold is the one chosen on validation (stored in the checkpoint);
nothing about the test set feeds back into model or threshold choice.

Outputs (under <data_root>/artifacts/metrics):
    summary.json            per-run test metrics with lesion-level bootstrap CIs, per-model seed
                            aggregates, and the served checkpoint per model (best val AUROC)
    test_predictions.csv    per-image scores for every run
    val_predictions.csv     per-image validation scores (for val-only analysis decisions)
    confusion_<run>.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.config import data_path, load_config
from ml.datasets import ManifestDataset, load_split
from ml.metrics import metrics_with_ci
from ml.models import MODEL_NAMES, build_model, count_parameters
from ml.preprocessing import eval_transform, preprocess_for_inference
from ml.train import get_device, predict_scores


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["model_id"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def measure_latency(model: torch.nn.Module, image_path: Path, device: torch.device, cc: bool, n: int = 50) -> float:
    """Median single-image latency (ms) including decode + preprocessing, after 5 warm-up runs."""
    times = []
    with torch.inference_mode():
        for i in range(n + 5):
            t0 = time.perf_counter()
            x = preprocess_for_inference(image_path, cc).to(device)
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 5:
                times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def plot_confusion(m: dict, title: str, out: Path) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["benign", "malignant"])
    ax.set_yticks([0, 1], ["benign", "malignant"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config()
    root = Path(cfg["data_root"])
    art = data_path(cfg, "artifacts")
    out_dir = art / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    n_boot = cfg["evaluate"]["bootstrap_resamples"]

    ckpts = sorted((art / "models").glob("*.pth"))
    ckpts = [p for p in ckpts if not p.stem.endswith("_smoke")]
    if args.runs:
        ckpts = [p for p in ckpts if p.stem in set(args.runs)]
    if not ckpts:
        raise SystemExit("No checkpoints found; train first.")

    manifest = data_path(cfg, "splits") / "manifest_split.csv"
    test_df, val_df = load_split(manifest, "test"), load_split(manifest, "val")
    groups = test_df["lesion_id"].to_numpy()
    runs, pred_frames, val_frames = {}, [], []
    for path in ckpts:
        model, ckpt = load_checkpoint(path, device)
        cc = bool(ckpt.get("color_constancy", False))

        def scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            loader = DataLoader(ManifestDataset(df, root, eval_transform(ckpt["image_size"], cc)),
                                batch_size=64, num_workers=cfg["train"]["num_workers"])
            return predict_scores(model, loader, device, amp=False)

        # Validation scores are saved for val-only decisions (stratified analysis, choice of fix).
        yv, sv = scores(val_df)
        val_frames.append(pd.DataFrame({"image_id": val_df["image_id"], "lesion_id": val_df["lesion_id"],
                                        "true_label": yv, "score_malignant": sv, "run": path.stem,
                                        "threshold": ckpt["val_threshold"]}))
        y, s = scores(test_df)
        m = metrics_with_ci(y, s, groups, ckpt["val_threshold"], n_resamples=n_boot)
        hist_path = art / "runs" / path.stem / "history.json"
        hist = json.loads(hist_path.read_text()) if hist_path.is_file() else {}
        m.update({
            "run": path.stem, "model_id": ckpt["model_id"], "model_name": MODEL_NAMES[ckpt["model_id"]],
            "seed": ckpt["seed"], "color_constancy": cc, "best_epoch": ckpt["epoch"],
            "val_auroc": ckpt["val_auroc"], "n_params": count_parameters(model),
            "artifact_mb": round(path.stat().st_size / 2**20, 2),
            "train_seconds": hist.get("train_seconds"),
            "latency_ms_median": measure_latency(model, root / test_df["image_path"][0], device, cc),
            "latency_method": f"median of 50 single-image runs on {device} incl. decode+preprocess",
        })
        runs[path.stem] = m
        plot_confusion(m, f"{path.stem} (thr={m['threshold']:.2f})", out_dir / f"confusion_{path.stem}.png")
        pred_frames.append(pd.DataFrame({
            "image_id": test_df["image_id"], "lesion_id": test_df["lesion_id"], "true_label": y,
            "score_malignant": s, "pred_label": (s >= m["threshold"]).astype(int), "run": path.stem,
            "threshold": m["threshold"],
        }))
        print(f"{path.stem}: AUROC={m['auroc']:.4f} sens={m['sensitivity']:.3f} spec={m['specificity']:.3f}")

    # Aggregate seeds per model variant and pick the served checkpoint by validation AUROC.
    variants: dict[str, list[dict]] = {}
    for m in runs.values():
        variants.setdefault(m["run"].rsplit("_s", 1)[0], []).append(m)
    aggregate, served = {}, {}
    for v, ms in variants.items():
        aggregate[v] = {k: {"mean": float(np.mean([x[k] for x in ms])), "sd": float(np.std([x[k] for x in ms], ddof=1)) if len(ms) > 1 else None}
                        for k in ("auroc", "sensitivity", "specificity", "balanced_accuracy", "f1")}
        aggregate[v]["n_seeds"] = len(ms)
        best = max(ms, key=lambda x: x["val_auroc"])
        served[v] = best["run"]

    summary = {
        "dataset": cfg["dataset"]["name"], "positive_class": "malignant",
        "split_seed": cfg["split"]["seed"], "n_test_images": int(len(test_df)),
        "n_test_lesions": int(test_df["lesion_id"].nunique()),
        "threshold_rule": "Youden J on validation", "ci_method": f"lesion-level percentile bootstrap, {n_boot} resamples",
        "score_semantics": "uncalibrated_model_score",
        "runs": runs, "aggregate": aggregate, "served_checkpoints": served,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.concat(pred_frames).to_csv(out_dir / "test_predictions.csv", index=False)
    pd.concat(val_frames).to_csv(out_dir / "val_predictions.csv", index=False)
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
