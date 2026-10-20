# SkinSight

> **Research prototype only.** SkinSight is an educational machine-learning project, not a medical device. Its outputs
> may be wrong and are not a diagnosis or a measure of clinical risk. Do not use it to make healthcare decisions;
> consult a qualified healthcare professional about skin concerns.

Binary (benign vs malignant) dermoscopy classification, built as a leakage-free re-examination of
Magdy et al., *Performance Enhancement of Skin Cancer Classification Using Computer Vision*, IEEE Access 2023.
The project adds a quantitative study of **hazy lesion borders** and **inhomogeneous colour** as failure modes.

Design decisions and the evaluation of the original implementation guide are in
[docs/ADR-001-plan-evaluation.md](docs/ADR-001-plan-evaluation.md).

## What is compared

| Model | Notes |
|---|---|
| `small_cnn` | Paper-style CNN from scratch: 5 conv layers (32/32/64/128/128) + BN, GAP, one FC |
| `efficientnet_b0` | ImageNet weights; head-only for 2 epochs, then full fine-tune |
| `mobilenet_v2` | Same protocol as EfficientNet-B0 |
| `*_cc` | Same model trained with Shades-of-Gray colour constancy (novelty fix candidate) |

Each model is trained with 3 seeds. Results are reported as mean ± SD, and the served checkpoint has lesion-level bootstrap CIs.

## Novelty: border haziness and colour inhomogeneity

These were pre-specified before any test results were seen.

- **H1:** sensitivity and AUROC are lower in the haziest-border and most-inhomogeneous tertiles.
- **H2:** one targeted fix, chosen on validation, narrows that gap without hurting overall AUROC.

Scores are computed automatically from the HAM10000 lesion masks ([ml/lesion_features.py](ml/lesion_features.py)). They are measured on a hair-suppressed copy of the image; the model inputs are never hair-removed.

- `border_haziness` is 1 ÷ edge strength, where edge strength is the peak |∇L*| near the lesion contour (median along the contour). A border comes out weak, and so hazy, when the transition is wide *or* when lesion and skin have similar lightness.
- `lab_sd` is the total CIELAB colour SD inside the lesion. Higher means more inhomogeneous colour.

Version 1 of the border score was a transition width (contrast ÷ gradient). Before any model was evaluated, inspecting the training-set extremes showed that it rated faint, hairy lesions as *sharp*, because it divides one small number by another and hair edges inflate the gradient. It was demoted to a secondary score (`border_width_px`), and hair suppression was added. Tests in [tests/test_lesion_features.py](tests/test_lesion_features.py) cover both failure cases.

A manual rubric on 100 random validation images ([ml/rubric.py](ml/rubric.py)) checks that these scores agree with human judgement (Spearman ρ). Grad-CAM is quantified as the share of CAM energy inside the lesion mask, per tertile.

## Data

- **Dataset:** HAM10000 (Tschandl et al., Harvard Dataverse, doi:10.7910/DVN/DBW86T), licensed **CC BY-NC 4.0**. Cite Tschandl et al. 2018 (Sci. Data) and, for the masks, Tschandl et al. 2020 (Nat. Med.). The data is not redistributed in this repo.
- **Label mapping:** malignant = `mel, bcc, akiec`; benign = `nv, bkl, df, vasc`. This gives about 20% malignant. The paper used a balanced 2,000/2,000 set, so accuracy cannot be compared with it directly.
- **Split:** 70/15/15 by `lesion_id`, stratified by label, seed 42. No lesion appears in two splits. HAM10000 has no patient ID, so leakage between patients cannot be ruled out.
- **Preprocessing** ([ml/preprocessing.py](ml/preprocessing.py)):
  - The whole image is resized to 224×224. There is no center crop, so lesion borders are kept.
  - ImageNet mean/std normalisation is used for all models.
  - Training augmentation is horizontal/vertical flips and ±15° rotation.
  - There is no hair removal or median filter.

## Setup (Windows, RTX 4050 6 GB)

Data, artifacts and the virtual environment live in `C:\skinsight_data`, outside OneDrive. To use another location, set `SKINSIGHT_DATA`.

```powershell
py -3.12 -m venv C:\skinsight_data\.venv
C:\skinsight_data\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Download the four HAM10000 files from the Dataverse page into `C:\skinsight_data\raw\zips`, then extract them:

```powershell
python -m ml.data.extract
```

## Pipeline

Run everything from the repository root:

```powershell
python -m ml.data.build_manifest            # validate files and labels -> splits/manifest.csv
python -m ml.data.create_splits             # lesion-grouped split    -> splits/manifest_split.csv
python -m ml.train --model small_cnn --seed 0 --smoke   # pipeline check
python -m ml.train --model efficientnet_b0 --seed 0     # repeat for each model x seeds 0,1,2
python -m ml.evaluate                       # test metrics + CIs -> artifacts/metrics/summary.json
python -m ml.lesion_features                # border / colour scores from masks
python -m ml.explainability --run efficientnet_b0_s0 --split val
python -m ml.stratified_analysis --split val            # choose the fix on VALIDATION
python -m ml.train --model efficientnet_b0 --seed 0 --color-constancy   # x 3 seeds
python -m ml.evaluate
python -m ml.stratified_analysis --split test --compare efficientnet_b0 efficientnet_b0_cc   # once
```

`scripts/train_all.ps1` runs every training job in sequence.

## App

```powershell
uvicorn backend.app.main:app --reload       # API: http://127.0.0.1:8000/docs
cd frontend; copy .env.example .env; npm install; npm run dev   # UI: http://localhost:5173
```

The endpoints are `GET /health`, `GET /models`, `POST /predict` (multipart: `file`, `model_id`, `include_gradcam`) and `GET /metrics`. `/metrics` returns saved evaluation output only.

## Tests

```powershell
pytest
```

The tests cover:

- split leakage
- the label mapping
- preprocessing determinism, and identical tensors from the API and evaluation paths
- model output shapes and Grad-CAM layers
- metric edge cases
- the lesion-feature behaviour on synthetic discs
- API validation and its error codes

## Results

All numbers come from `C:\skinsight_data\artifacts\metrics\summary.json` and `artifacts\novelty\stratified_test.json`. The test set is HAM10000 with 1,485 images from 1,121 lesions, 286 of them malignant. The threshold was chosen on validation (Youden's J). Values are mean ± SD over 3 seeds.

| Model | Params | Test AUROC | Sensitivity | Specificity | Balanced acc. |
|---|---|---|---|---|---|
| Small CNN (paper-style) | 0.25 M | 0.858 ± 0.013 | 0.855 ± 0.009 | 0.715 ± 0.015 | 0.785 ± 0.011 |
| EfficientNet-B0 | 4.01 M | 0.903 ± 0.004 | 0.824 ± 0.012 | 0.825 ± 0.007 | 0.824 ± 0.003 |
| MobileNetV2 | 2.23 M | 0.905 ± 0.007 | 0.823 ± 0.041 | 0.820 ± 0.024 | 0.821 ± 0.009 |
| EfficientNet-B0 + colour constancy | 4.01 M | 0.896 ± 0.005 | 0.796 ± 0.084 | 0.814 ± 0.065 | 0.805 ± 0.013 |

- The two pretrained models are indistinguishable on test.
- Both beat the from-scratch CNN by about 0.05 AUROC.
- Single-image latency on the RTX 4050 is 10–20 ms, including decoding.

The paper reports around 99% accuracy. That figure is **not comparable**: the paper used a balanced 4,000-image subset and a random image-level split with no validation set, and its GWO search tuned on the test set.

### Novelty findings

The full log, with the decisions made before the test set was used, is in [docs/novelty_log.md](docs/novelty_log.md).

1. **Colour inhomogeneity is associated with worse performance (H1 supported).** For EfficientNet-B0, AUROC is about 0.95 in the most uniform colour tertile and 0.85–0.87 in the most inhomogeneous one. Every model shows the same drop on validation and on test.
2. **Hazy borders are *not* the hard cases (H1 not supported, the direction is reversed).** Clear-border lesions are hardest, at 0.83–0.85 vs 0.94–0.95 for hazy ones. The validation analysis points to confounding: clear borders mark dark, high-contrast, colour-varied lesions, and that tertile holds most of the melanomas.
3. **Colour-constancy preprocessing does not help (H2 not supported).** The paired ΔAUROC overall is −0.010 to −0.001, with every CI containing 0. In the inhomogeneous tertile it ranges from −0.032 to +0.001.
4. **EfficientNet-B0's false positives rely on off-lesion regions.** Median Grad-CAM enrichment is 0.49 for false positives against 1.14 for true positives, and this replicates from validation to test. The pattern is architecture-specific: MobileNetV2 shows a weaker version, and the small CNN a different one.

## Limitations

- This is a single public dataset, mostly from fair-skinned European populations, so performance will not transfer to other populations or devices without evidence.
- Model scores are uncalibrated. The decision threshold is Youden's J on validation.
- Grad-CAM is a qualitative visualisation, not a segmentation and not a clinical rationale.
- Border and colour scores depend on the quality of the provided masks. The tertile analyses show association, not causation.
- Hair suppression for the measurements is tuned for precision. It no longer erases pigment network, but very dense or thick hair is under-detected, so the scores for those images are less reliable. `hair_frac` is saved as a covariate.
- The reference paper's published numbers are not targets. They come from a different data subset and a random split with no validation set.
