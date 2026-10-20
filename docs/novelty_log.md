# Novelty analysis log

This log records decisions in the order they were made. Everything in sections 1–3 was written **before** any test-set stratified analysis or colour-constancy test result existed. The baseline test metrics in `summary.json` had been computed, but they were not split by tertile and played no part in any decision below.

## 1. Score definitions (2026-10-20, before any model was evaluated)

- Primary border score: `border_haziness` = 1 / peak edge strength, measured on a hair-suppressed image. v1 (transition width) was demoted after inspecting the training-set extremes; see the ADR addendum.
- Primary colour score: `lab_sd`, the total CIELAB SD inside the mask.
- Tertile cut-points are fitted on the train split: haziness 0.592 / 0.826, `lab_sd` 10.80 / 14.20.

## 2. Validation findings (2026-10-20, ~19:15)

Source: `artifacts/novelty/stratified_val.json`, 1,469 val images. Ranges below are lesion-level bootstrap 95% CIs.

**Colour inhomogeneity: H1 supported on validation.** Across all 9 baseline runs, AUROC falls from the most uniform to the most inhomogeneous tertile:

| Model | Uniform tertile AUROC | Inhomogeneous tertile AUROC |
|---|---|---|
| EfficientNet-B0 | 0.967–0.972 | 0.872–0.888 |
| MobileNetV2 | 0.955–0.975 | 0.853–0.872 |
| Small CNN | 0.922–0.932 | 0.779–0.791 |

Specificity falls too (EfficientNet-B0: 0.86 → 0.73). For EfficientNet-B0 s0 the CIs don't overlap: uniform [0.95–0.98] vs inhomogeneous [0.85–0.92].

**Border haziness: H1 not supported; the effect runs the other way.** The *clearest*-border tertile is the hardest (EfficientNet-B0 AUROC 0.870–0.883, against 0.956–0.962 for the hazy tertile). An exploratory val-only cross-tab (EfficientNet-B0, mean score over 3 seeds) suggests confounding:
- 59% of clear-border lesions (265/446) are also in the most inhomogeneous colour tertile. That combined cell is the hardest, at AUROC 0.856.
- The clear-border tertile holds 84 melanomas, against 30 in the hazy tertile. Its median lesion/skin contrast is 19.5 L*, against 6.4 for the hazy tertile. "Clear border" here largely means dark, high-contrast lesions, where melanoma vs dark nevus is the hard distinction.
- Within the high-`lab_sd` level, a gap remains across haziness (0.856 / 0.957 / 0.913), but the cells are small (n = 65–265).

**Grad-CAM (val, EfficientNet-B0 s0, size-normalised enrichment = CAM share in lesion ÷ lesion area share; 1 = chance):**

| Outcome | Enrichment (median) |
|---|---|
| False positive | 0.58 |
| True positive | 1.15 |
| True negative | 2.62 |
| False negative | 4.45 |

- TP and FP share a target class, so they compare directly: when the model wrongly calls a benign lesion malignant, the evidence comes mostly from outside the lesion. The three saved FP overlays all have their hot-spot on the adjacent skin.
- The small CNN's enrichment is about 1.1–1.5, so it barely prefers the lesion over the rest of the image.
- The raw (unnormalised) in-mask fraction falls toward clear borders only because those lesions are smaller. That was a size artifact, and normalising removes it.

## 3. Fix decision (2026-10-20, ~19:20, validation only)

Pre-specified rule (ADR D4.5): use Shades-of-Gray colour constancy, unless validation shows the hazy-border gap to be the larger one, in which case use subgroup reweighting.

The colour gap is the clear, hypothesis-direction effect, and the border effect is reversed. So:
- **Fix:** Shades-of-Gray colour constancy (p = 6), applied after the resize in both training and inference.
- **Backbone:** EfficientNet-B0, the best mean val AUROC (0.935 vs MobileNetV2 0.929).
- **Seeds:** 3 (0, 1, 2), with everything else identical to the baseline.
- **Pre-specified test comparison:** paired lesion-level bootstrap Δ (fix − baseline) of AUROC and sensitivity, per seed, overall and per tertile, reported once.

Expectation, stated in advance: colour constancy corrects global illumination and colour cast, not within-lesion colour variation. It may therefore *not* close the inhomogeneity gap. A null result will be reported as such.

Implementation note: the first fix run applied colour constancy before the resize (35 ms/img, CPU-bound). It was stopped after about 3 epochs of seed 0, the step was moved after the resize (4.9 ms/img), and all 3 seeds were restarted from scratch. No result from the stopped run was used.

## 4. Test results (single final run, 2026-10-20 ~20:25)

Sources: `artifacts/novelty/stratified_test.json` and `artifacts/novelty/cam_in_mask_*_test.csv`. Test set: 1,485 images, 286 malignant. Ranges are lesion-level bootstrap 95% CIs; Δ = fix − baseline, paired by seed.

**H1 (colour inhomogeneity): supported, and it replicates validation.**

| Run | Uniform AUROC | Mid AUROC | Inhomogeneous AUROC |
|---|---|---|---|
| EfficientNet-B0 s0 | 0.956 | 0.888 | 0.852 [0.80–0.90] |
| EfficientNet-B0 s1 | 0.949 | 0.898 | 0.853 |
| EfficientNet-B0 s2 | 0.952 | 0.895 | 0.870 |
| MobileNetV2 s2 | 0.969 | 0.904 | 0.862 |
| Small CNN s2 | 0.926 | 0.882 | 0.768 |

Specificity falls in the inhomogeneous tertile (EfficientNet-B0 s0: 0.86 → 0.75). That tertile is also the most malignant-rich (145/476 vs 65/474), so this is an association, not a cause.

**H1 (border haziness): not supported; the direction is reversed, as on validation.** The clear-border tertile is hardest:

| Run | Clear AUROC | Mid AUROC | Hazy AUROC |
|---|---|---|---|
| EfficientNet-B0 (3 seeds) | 0.833–0.848 | 0.913–0.919 | 0.937–0.950 |
| MobileNetV2 s2 | 0.848 | 0.927 | 0.958 |
| Small CNN s2 | 0.780 | 0.878 | 0.923 |

Section 2 gives the likely explanation: dark, high-contrast, colour-varied lesions, and a melanoma-rich tertile.

**H2 (colour-constancy fix): not supported.**

| Subset | ΔAUROC s0 | ΔAUROC s1 | ΔAUROC s2 |
|---|---|---|---|
| Overall | −0.010 [−0.027, +0.006] | −0.001 [−0.016, +0.013] | −0.010 [−0.024, +0.004] |
| Inhomogeneous tertile | −0.029 [−0.066, +0.004] | +0.001 [−0.027, +0.031] | −0.032 [−0.059, −0.006] |

Seed-mean test AUROC is 0.896 ± 0.005 for the fix against 0.903 ± 0.004 for the baseline. The sensitivity changes, for example s2 overall −0.136 [−0.186, −0.087], come from where the validation-chosen threshold landed, not from better ranking. This matches the expectation recorded in section 3: correcting the illuminant does not address colour variation *within* the lesion.

**Grad-CAM (size-normalised enrichment, median; 1 = chance), test split:**

| Run | TP | FP | TN | FN |
|---|---|---|---|---|
| EfficientNet-B0 s0 | 1.14 | **0.49** | 2.82 | 3.83 |
| EfficientNet-B0 + CC s1 | 1.13 | **0.48** | 3.27 | 3.32 |
| MobileNetV2 s2 | 1.84 | 1.27 | 2.40 | 2.58 |
| Small CNN s2 | 1.89 | 2.08 | 0.88 | 0.04 |

- The EfficientNet-B0 finding replicates validation (FP 0.58 → 0.49): its false-malignant calls rest mostly on image regions *outside* the lesion.
- For MobileNetV2, false positives are less lesion-focused than true positives (1.27 vs 1.84) but still above chance.
- The small CNN shows a different pattern: its *benign* calls are not lesion-focused.
- So the off-lesion false-positive behaviour is architecture-specific. Grad-CAM is a qualitative tool, so this is a lead to investigate (surrounding sun-damaged skin, vignette, rulers), not proof of a shortcut.

## 5. Status of the manual rubric

`artifacts/novelty/rubric_template.csv` (100 random val images, labels hidden) is waiting for annotation. Until it is annotated, the claim that `border_haziness` and `lab_sd` match human judgement of "hazy" and "inhomogeneous" rests only on the visual inspection of the training-set extremes.
