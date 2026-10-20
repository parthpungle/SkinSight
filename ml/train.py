"""Train one model for one seed; select the best epoch by validation AUROC.

Usage:
    python -m ml.train --model efficientnet_b0 --seed 0
    python -m ml.train --model small_cnn --seed 0 --smoke        # quick pipeline check
    python -m ml.train --model efficientnet_b0 --seed 0 --color-constancy   # novelty fix run

Outputs (under <data_root>/artifacts):
    models/<run>.pth               weights + metadata (loadable with weights_only=True)
    runs/<run>/history.json        per-epoch train loss and validation metrics
The test split is never touched here.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from ml.config import data_path, load_config
from ml.datasets import ManifestDataset, load_split
from ml.metrics import binary_metrics, youden_threshold
from ml.models import CLASS_NAMES, MODEL_IDS, backbone_parameters, build_model, count_parameters, head_parameters
from ml.preprocessing import IMAGENET_MEAN, IMAGENET_STD, eval_transform, train_transform


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_name(model_id: str, seed: int, color_constancy: bool) -> str:
    return f"{model_id}{'_cc' if color_constancy else ''}_s{seed}"


@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, malignant softmax scores)."""
    model.eval()
    ys, ss = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        ss.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ss)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in backbone_parameters(model):
        p.requires_grad = trainable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--color-constancy", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 20 batches: pipeline check only")
    args = parser.parse_args()

    cfg = load_config()
    tcfg, mcfg = cfg["train"], cfg["train"]["models"][args.model]
    set_seed(args.seed)
    device = get_device()
    amp = bool(tcfg["amp"]) and device.type == "cuda"
    root = Path(cfg["data_root"])
    manifest = data_path(cfg, "splits") / "manifest_split.csv"
    art = data_path(cfg, "artifacts")
    name = run_name(args.model, args.seed, args.color_constancy) + ("_smoke" if args.smoke else "")
    (art / "models").mkdir(parents=True, exist_ok=True)
    run_dir = art / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_split(manifest, "train"), load_split(manifest, "val")
    size = cfg["preprocess"]["image_size"]
    train_ds = ManifestDataset(train_df, root, train_transform(size, args.color_constancy))
    val_ds = ManifestDataset(val_df, root, eval_transform(size, args.color_constancy))
    nw = tcfg["num_workers"]
    loader_kw = dict(batch_size=tcfg["batch_size"], num_workers=nw, pin_memory=device.type == "cuda",
                     persistent_workers=nw > 0)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(args.model, pretrained=True).to(device)
    pretrained = args.model != "small_cnn"
    if pretrained:
        optimizer = torch.optim.AdamW([
            {"params": backbone_parameters(model), "lr": mcfg["backbone_lr"]},
            {"params": head_parameters(model), "lr": mcfg["head_lr"]},
        ], weight_decay=mcfg["weight_decay"])
        freeze_epochs = mcfg["freeze_epochs"]
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
        freeze_epochs = 0

    max_epochs = 1 if args.smoke else (args.epochs or tcfg["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    weight = None
    if tcfg["class_weighted_loss"]:
        counts = np.bincount(train_df["label"], minlength=2)
        weight = torch.tensor(len(train_df) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    print(f"run={name} device={device} amp={amp} params={count_parameters(model):,} "
          f"train={len(train_ds)} val={len(val_ds)} class_weights={None if weight is None else weight.tolist()}")

    history, best_auroc, best_epoch, bad_epochs = [], -1.0, -1, 0
    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        if pretrained:
            set_backbone_trainable(model, epoch > freeze_epochs)
        model.train()
        t0, losses = time.time(), []
        for step, (x, y) in enumerate(train_loader):
            if args.smoke and step >= 20:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        y_val, s_val = predict_scores(model, val_loader, device, amp)
        thr = youden_threshold(y_val, s_val)
        vm = binary_metrics(y_val, s_val, thr)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc": vm["auroc"],
               "val_sensitivity": vm["sensitivity"], "val_specificity": vm["specificity"],
               "val_threshold": thr, "backbone_frozen": pretrained and epoch <= freeze_epochs,
               "epoch_seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec))

        if vm["auroc"] is not None and vm["auroc"] > best_auroc:
            best_auroc, best_epoch, bad_epochs = vm["auroc"], epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_id": args.model,
                "class_names": list(CLASS_NAMES),
                "image_size": size,
                "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
                "color_constancy": args.color_constancy,
                "seed": args.seed,
                "epoch": epoch,
                "val_auroc": vm["auroc"],
                "val_threshold": thr,
            }, art / "models" / f"{name}.pth")
        else:
            bad_epochs += 1
            if epoch > freeze_epochs and bad_epochs >= tcfg["early_stopping_patience"]:
                print(f"early stop at epoch {epoch}")
                break

    meta = {
        "run": name, "model_id": args.model, "seed": args.seed, "color_constancy": args.color_constancy,
        "best_epoch": best_epoch, "best_val_auroc": best_auroc,
        "train_seconds": round(time.time() - t_start, 1),
        "n_params": count_parameters(model), "config": cfg,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None},
        "history": history,
    }
    (run_dir / "history.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"best epoch {best_epoch} val_auroc={best_auroc:.4f}")


if __name__ == "__main__":
    main()
