import cv2
import numpy as np

from ml.lesion_features import compute_features, remove_hair


def _disc_image(blur_sigma: float, inner=(60, 40, 30), outer=(200, 160, 150), noise: float = 0.0):
    h, w = 450, 600
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (300, 225), 120, 1, -1)
    img = np.where(mask[..., None] == 1, np.array(inner, np.float32), np.array(outer, np.float32))
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise:
        img = img + np.random.default_rng(0).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask.astype(bool)


def _add_hair(img: np.ndarray, n: int = 25) -> np.ndarray:
    out = img.copy()
    rng = np.random.default_rng(1)
    for _ in range(n):
        p1 = tuple(int(v) for v in rng.integers(0, (600, 450)))
        p2 = tuple(int(v) for v in rng.integers(0, (600, 450)))
        cv2.line(out, p1, p2, (20, 15, 10), 2)
    return out


def test_haziness_increases_with_blur():
    vals = [compute_features(*_disc_image(s))["border_haziness"] for s in (1, 3, 6, 10)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_low_contrast_border_counts_as_hazier():
    faint = compute_features(*_disc_image(2.0, inner=(185, 148, 140)))
    strong = compute_features(*_disc_image(2.0, inner=(60, 40, 30)))
    assert faint["border_haziness"] > 3 * strong["border_haziness"]


def test_width_tracks_blur_and_is_contrast_invariant():
    widths = [compute_features(*_disc_image(s))["border_width_px"] for s in (1, 3, 6, 10)]
    assert all(a < b for a, b in zip(widths, widths[1:]))
    lo = compute_features(*_disc_image(4.0, inner=(150, 120, 110)))["border_width_px"]
    hi = compute_features(*_disc_image(4.0, inner=(40, 30, 20)))["border_width_px"]
    assert 0.6 < lo / hi < 1.6


def test_border_tolerates_small_mask_offset():
    img, mask = _disc_image(3.0)
    shrunk = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)  # ~3 px inward
    a = compute_features(img, mask)["border_haziness"]
    b = compute_features(img, shrunk)["border_haziness"]
    assert 0.7 < a / b < 1.4


def test_hair_does_not_make_a_faint_border_look_sharp():
    img, mask = _disc_image(6.0, inner=(170, 135, 125))
    clean = compute_features(img, mask)["border_haziness"]
    hairy_no_removal = compute_features(_add_hair(img), mask, hair_removal=False)["border_haziness"]
    hairy_removed = compute_features(_add_hair(img), mask)["border_haziness"]
    assert hairy_no_removal < 0.6 * clean           # the confound the removal step exists for
    assert abs(hairy_removed - clean) / clean < 0.35


def test_remove_hair_flags_lines_not_lesion():
    img, _ = _disc_image(2.0)
    assert remove_hair(img)[1] < 0.01
    assert remove_hair(_add_hair(img))[1] > 0.02


def test_dots_and_network_are_not_hair():
    """Pigment dots/globules and a short mesh inside the lesion must survive hair removal."""
    img, _ = _disc_image(2.0, inner=(150, 110, 90))
    rng = np.random.default_rng(3)
    for _ in range(300):
        x, y = (int(v) for v in rng.integers((200, 125), (400, 325)))
        cv2.circle(img, (x, y), int(rng.integers(2, 5)), (60, 40, 30), -1)
    for x in range(220, 380, 12):   # short network strands (< HAIR_MIN_LENGTH)
        cv2.line(img, (x, 200), (x + 8, 230), (70, 50, 40), 2)
    assert remove_hair(img)[1] < 0.005


def test_uniform_lesion_less_inhomogeneous_than_noisy():
    flat = compute_features(*_disc_image(1.0), hair_removal=False)
    noisy = compute_features(*_disc_image(1.0, noise=25), hair_removal=False)
    assert noisy["lab_sd"] > flat["lab_sd"]


def test_empty_mask_returns_nan():
    img, _ = _disc_image(1.0)
    f = compute_features(img, np.zeros(img.shape[:2], bool))
    assert np.isnan(f["border_haziness"]) and np.isnan(f["lab_sd"])
