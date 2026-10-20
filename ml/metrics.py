"""Binary metrics (malignant = positive), Youden threshold, and lesion-level bootstrap CIs."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _safe_div(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (choose on validation only)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, score)
    best = int(np.argmax(tpr - fpr))
    return float(min(thr[best], 1.0))


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    prec = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    auroc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else None
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)), "n_pos": int(y_true.sum()), "n_neg": int((1 - y_true).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
        "balanced_accuracy": (sens + spec) / 2 if sens is not None and spec is not None else None,
        "auroc": auroc,
    }


def bootstrap_ci(groups: np.ndarray, fn: Callable[[np.ndarray], float | None],
                 n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Percentile CI resampling *groups* (lesions) with replacement.

    `fn` receives an array of row indices and returns a statistic (or None if undefined).
    """
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    stats = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[g] for g in pick])
        v = fn(idx)
        if v is not None and np.isfinite(v):
            stats.append(v)
    if not stats:
        return None, None
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def metrics_with_ci(y_true: np.ndarray, score: np.ndarray, groups: np.ndarray, threshold: float,
                    keys: tuple[str, ...] = ("auroc", "sensitivity", "specificity", "balanced_accuracy"),
                    n_resamples: int = 1000, seed: int = 0) -> dict:
    y_true, score = np.asarray(y_true), np.asarray(score)
    point = binary_metrics(y_true, score, threshold)
    ci = {}
    for k in keys:
        lo, hi = bootstrap_ci(groups, lambda idx, k=k: binary_metrics(y_true[idx], score[idx], threshold)[k],
                              n_resamples, seed)
        ci[k] = [lo, hi]
    point["ci95"] = ci
    return point


def paired_bootstrap_delta(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, groups: np.ndarray,
                           thr_a: float, thr_b: float, key: str = "auroc",
                           n_resamples: int = 1000, seed: int = 0) -> dict:
    """CI for metric(B) - metric(A) on the same resampled lesions."""
    y_true, score_a, score_b = map(np.asarray, (y_true, score_a, score_b))

    def delta(idx: np.ndarray) -> float | None:
        a = binary_metrics(y_true[idx], score_a[idx], thr_a)[key]
        b = binary_metrics(y_true[idx], score_b[idx], thr_b)[key]
        return None if a is None or b is None else b - a

    point = delta(np.arange(len(y_true)))
    lo, hi = bootstrap_ci(groups, delta, n_resamples, seed)
    return {"metric": key, "delta": point, "ci95": [lo, hi]}
