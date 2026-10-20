"""H1/H2: performance by border-haziness and colour-inhomogeneity tertiles.

Tertile cut-points are fitted on the TRAIN split's lesion features, then applied to val/test.
Run on --split val to choose the fix; run on --split test once for the final report.

Usage:
    python -m ml.stratified_analysis --split val
    python -m ml.stratified_analysis --split test --compare efficientnet_b0 efficientnet_b0_cc

Writes artifacts/novelty/stratified_<split>.json and a figure per score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.config import data_path, load_config
from ml.metrics import metrics_with_ci, paired_bootstrap_delta

SCORES = {"border_haziness": "Border haziness (1 / edge strength)", "lab_sd": "Colour inhomogeneity (Lab SD)"}
TERTILES = ["low", "mid", "high"]


def assign_tertiles(values: pd.Series, train_values: pd.Series) -> pd.Series:
    q1, q2 = np.nanquantile(train_values, [1 / 3, 2 / 3])
    return pd.cut(values, [-np.inf, q1, q2, np.inf], labels=TERTILES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--runs", nargs="*", default=None, help="default: all runs in the predictions file")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "FIX"), default=None,
                        help="variant names, e.g. efficientnet_b0 efficientnet_b0_cc (paired per seed)")
    args = parser.parse_args()

    cfg = load_config()
    art = data_path(cfg, "artifacts")
    n_boot = cfg["evaluate"]["bootstrap_resamples"]
    feats = pd.read_csv(art / "novelty" / "lesion_features.csv")
    preds = pd.read_csv(art / "metrics" / f"{args.split}_predictions.csv")
    train_feats = feats[feats["split"] == "train"]

    feats = feats[feats["split"] == args.split].copy()
    for s in SCORES:
        feats[f"{s}_tertile"] = assign_tertiles(feats[s], train_feats[s])
    df = preds.merge(feats.drop(columns=["lesion_id"]), on="image_id", how="inner")
    runs = args.runs or sorted(df["run"].unique())

    result: dict = {"split": args.split, "cutpoints_from": "train", "scores": {}, "comparisons": {}}
    for s in SCORES:
        q = np.nanquantile(train_feats[s], [1 / 3, 2 / 3]).tolist()
        result["scores"][s] = {"cutpoints": q, "runs": {}}
        for run in runs:
            part = df[df["run"] == run]
            thr = float(part["threshold"].iloc[0])
            per = {}
            for t in TERTILES:
                p = part[part[f"{s}_tertile"] == t]
                if len(p) == 0:
                    continue
                per[t] = metrics_with_ci(p["true_label"].to_numpy(), p["score_malignant"].to_numpy(),
                                         p["lesion_id"].to_numpy(), thr, n_resamples=n_boot)
                cam = art / "novelty" / f"cam_in_mask_{run}_{args.split}.csv"
                if cam.is_file():
                    c = pd.read_csv(cam).merge(p[["image_id"]], on="image_id")
                    per[t]["cam_in_mask_mean"] = float(c["cam_in_mask"].mean()) if len(c) else None
                    # Size-normalised: 1.0 = CAM energy spread like a random map; > 1 = concentrated on the lesion.
                    enrich = c["cam_in_mask"] / c["mask_area_frac"].clip(lower=1e-3)
                    per[t]["cam_enrichment_median"] = float(enrich.median()) if len(c) else None
            result["scores"][s]["runs"][run] = per

    if args.compare:
        base, fix = args.compare
        for s in SCORES:
            comp = {}
            seeds = sorted({r.rsplit("_s", 1)[1] for r in runs if r.rsplit("_s", 1)[0] == base}
                           & {r.rsplit("_s", 1)[1] for r in runs if r.rsplit("_s", 1)[0] == fix})
            for seed in seeds:
                a = df[df["run"] == f"{base}_s{seed}"].set_index("image_id")
                b = df[df["run"] == f"{fix}_s{seed}"].set_index("image_id").loc[a.index]
                for t in TERTILES + ["all"]:
                    sel = np.ones(len(a), bool) if t == "all" else (a[f"{s}_tertile"] == t).to_numpy()
                    for key in ("auroc", "sensitivity"):
                        comp.setdefault(t, {}).setdefault(key, {})[f"s{seed}"] = paired_bootstrap_delta(
                            a["true_label"].to_numpy()[sel], a["score_malignant"].to_numpy()[sel],
                            b["score_malignant"].to_numpy()[sel], a["lesion_id"].to_numpy()[sel],
                            float(a["threshold"].iloc[0]), float(b["threshold"].iloc[0]), key, n_boot)
            result["comparisons"][s] = {"base": base, "fix": fix, "by_tertile": comp}

    out = art / "novelty"
    (out / f"stratified_{args.split}.json").write_text(json.dumps(result, indent=2))

    # Figure: AUROC and sensitivity per tertile for each run, with CIs.
    for s, label in SCORES.items():
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
        for ax, key in zip(axes, ("auroc", "sensitivity")):
            for i, run in enumerate(runs):
                per = result["scores"][s]["runs"][run]
                xs = np.array([TERTILES.index(t) for t in per]) + (i - len(runs) / 2) * 0.08
                ys, lo, hi = [], [], []
                for t in per:  # undefined metrics (e.g. AUROC with one class) are plotted as gaps
                    v, (c_lo, c_hi) = per[t][key], per[t]["ci95"][key]
                    ys.append(np.nan if v is None else v)
                    lo.append(0.0 if v is None or c_lo is None else v - c_lo)
                    hi.append(0.0 if v is None or c_hi is None else c_hi - v)
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=run, linewidth=1)
            ax.set_xticks(range(3), [f"{t}" for t in TERTILES])
            ax.set_xlabel(f"{label} tertile")
            ax.set_ylabel(key)
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=7, loc="best")
        fig.suptitle(f"{args.split}: performance by {label}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"stratified_{args.split}_{s}.png", dpi=150)
        plt.close(fig)
    print(json.dumps({s: {r: {t: round(v["auroc"], 3) if v["auroc"] else None for t, v in per.items()}
                          for r, per in result["scores"][s]["runs"].items()} for s in SCORES}, indent=1))


if __name__ == "__main__":
    main()
