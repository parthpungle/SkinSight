import numpy as np

from ml.metrics import binary_metrics, bootstrap_ci, paired_bootstrap_delta, youden_threshold


def test_known_confusion_matrix():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    m = binary_metrics(y, s, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["sensitivity"] == 0.5
    assert abs(m["specificity"] - 2 / 3) < 1e-9


def test_single_class_does_not_crash():
    m = binary_metrics(np.zeros(4, int), np.array([0.1, 0.2, 0.7, 0.4]), 0.5)
    assert m["sensitivity"] is None
    assert m["auroc"] is None
    assert m["specificity"] == 0.75


def test_youden_separable():
    y = np.array([0, 0, 1, 1])
    t = youden_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.clip(y * 0.6 + rng.random(300) * 0.5, 0, 1)
    groups = np.arange(300) // 2
    lo, hi = bootstrap_ci(groups, lambda idx: binary_metrics(y[idx], s[idx], 0.5)["auroc"], 200)
    point = binary_metrics(y, s, 0.5)["auroc"]
    assert lo <= point <= hi


def test_paired_delta_zero_for_identical_models():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 100)
    s = rng.random(100)
    d = paired_bootstrap_delta(y, s, s, np.arange(100), 0.5, 0.5, n_resamples=50)
    assert d["delta"] == 0 and d["ci95"] == [0.0, 0.0]
