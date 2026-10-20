# ADR-001: SkinSight — evaluation of the implementation guide and revised plan

**Status:** Proposed  **Date:** 2026-10-20  **Deciders:** Parth (student), course supervisor

## Context

- **Source plan:** `SkinSight_AI_Agent_Implementation_Guide (1).md`. It covers a binary benign/malignant dermoscopy classifier with three models (small CNN, EfficientNet-B0, MobileNetV2), Grad-CAM, an error analysis of hazy borders and inhomogeneous appearance, and a React + Tailwind + FastAPI dashboard.
- **Reference paper:** Magdy et al., IEEE Access 2023. It was built in MATLAB on a CPU. It used 4,000 balanced ISIC images (2,000/2,000) with a **random 80/20 split and no validation set**. Preprocessing was resize to 250, median filter and hair removal; segmentation was tried and dropped. It tested 10 pretrained CNNs (5 epochs, batch 32, lr 1e-4) and a small CNN (5 conv layers: 32/32/64/128/128 + BN, 97.75%). Its headline claims are **KNN-PDNN** (KNN with k=10 on CNN features, >99%) and **AlexGWO** (GWO tunes batch size and lr, 99%). GWO's objective is test accuracy, so the test set is used for tuning.
- **Constraints:** RTX 4050 Laptop 6 GB (driver 617.42, confirmed working). No deployment. Student timeline.
- **Repo state:** `backend/` and `frontend/` exist but are empty, and there is one commit.
- **Decisions you made:** FastAPI + **minimal** React; the novelty goes as far as **measure + one fix**; KNN-PDNN stays **optional**.

## Verdict on the md plan

**Overall: sound and honest, but unbalanced.** Its research-integrity rules are its best part: leakage-free grouped splits, no test-set tuning, careful score semantics, and Grad-CAM caveats. They fix the paper's real weaknesses, so keep them all. It has four problems:

1. **The novelty is under-specified.** "Investigate errors" through a manual rubric over FP/FN cases is descriptive. It only looks at errors, so you can't compute error *rates* per category. It is subjective and has no hypothesis.
2. **The dataset choice is left open.** The novelty, the leakage rules and the label mapping all depend on it.
3. **Effort goes to the wrong places.** About 40% of the spec covers web hardening and UI (CORS, decompression bombs, 413/415 handling, a Research page, Recharts) for an app that will never be deployed.
4. **There are concrete technical gaps** (listed below), and some of them would quietly confound the novelty.

## Decision: changes to the plan

### D1. Dataset → HAM10000 (ISIC 2018), lesion-grouped split
- 10,015 images, each with a `lesion_id`, so grouped splitting is possible. There is no patient_id, so record that as a limitation.
- **Public lesion segmentation masks exist for HAM10000** (Tschandl et al., Harvard Dataverse). The novelty measurements need them. Verify the download before Phase 1. Fallback: the ISIC 2018 Task 1 masks (2,594-image subset).
- Binary mapping: malignant = `mel, bcc, akiec`; benign = `nv, bkl, df, vasc`. That gives about 1,954 malignant and 8,061 benign, roughly 20% positive. Document that this is imbalanced, unlike the paper's balanced set, so accuracy can't be compared with the paper.
- Split 70/15/15, grouped by `lesion_id` and stratified by label, with a fixed seed.

### D2. Technical corrections to the md
| Issue in md | Fix |
|---|---|
| Uses the torchvision "weights transforms" (resize 256 → **center-crop 224**). The crop cuts off lesion borders, which **confounds the border-haziness study** | Resize the whole image to 224×224 for every model. Keep ImageNet mean/std for the pretrained models. |
| Batch 8–16 is too conservative | Batch 32 with AMP fits comfortably for B0 and MobileNetV2 at 224 on 6 GB. Fall back to 16 if you hit OOM. |
| A single run per model; differences will be inside the noise | **3 seeds** per model (cheap). Report mean ± sd. |
| Defers CIs | **Lesion-level bootstrap** 95% CIs (1,000 resamples), plus a paired bootstrap for model differences. About 30 lines of code. |
| Early-stopping metric and threshold are unspecified | Early-stop on **val AUROC**. Choose the operating threshold on val (Youden's J). Report test metrics at that threshold and also AUROC. |
| Accuracy-led metrics on imbalanced data | Lead with AUROC, sensitivity, specificity and balanced accuracy. Use class-weighted CE (one approach, documented). |
| Says only "small CNN" | Copy the **paper's CNN** (5 conv: 32/32/64/128/128 + BN + FC). That gives a direct link to the paper. |
| Decodes JPEGs from disk on every epoch, on Windows | **Pre-resize once** to a 256-px cache. Use `num_workers=4`, `persistent_workers=True`, and an `if __name__ == "__main__"` guard. |
| A custom Grad-CAM implementation | Use the `grad-cam` (pytorch-grad-cam) package. Target layers: `features[-1]` for both B0 and MobileNetV2, and the last conv block for the small CNN. |
| A large test list | Keep the essentials: split leakage, **preprocessing parity** (the API tensor matches the eval tensor exactly), metric zero-division, and an API smoke test. |

### D3. Environment issues (found during inspection, fix before Phase 2)
- **Mismatched PyTorch builds:** the global Python 3.14 has `torch 2.11.0+cu130` with `torchvision 0.26.0+cpu`. Create a project **venv on Python 3.12**, which has broader wheel support (opencv, grad-cam). Install torch and torchvision from the *same* CUDA index URL. Check `torch.cuda.is_available()` and that a torchvision op runs on CUDA.
- **The project lives under OneDrive.** Syncing 10k images, checkpoints, `.venv` and `node_modules` will be slow and can lock files. Put `data/` and `artifacts/` outside OneDrive (for example `C:\skinsight_data`), set the path in `ml/config.yaml`, or move the repo to `C:\dev\SkinSight`.
- Node 24 is fine for Vite.

### D4. Novelty: "measure + one fix" (replaces the manual rubric as the main method)
Hypotheses, written down **before** looking at test results:
- **H1:** Sensitivity and AUROC are lower on the hazy-border and highly inhomogeneous subgroups.
- **H2:** One targeted fix narrows that gap without hurting overall AUROC.

Method:
1. **Automatic scores from the masks** (`ml/lesion_features.py`):
   - *Border sharpness:* the mean gradient magnitude of Lab L across a ±k px band around the mask contour, divided by the inner-ring vs outer-ring L contrast. A low value means a hazy border.
   - *Colour inhomogeneity:* within-mask Lab standard deviation, plus a count of colour clusters (k-means, the ABCD "C" criterion). Pick one score as primary in advance.
2. **Validate the scores:** keep the md's manual rubric, but use it on a random ~100-image subset of val, not just errors. Report the Spearman correlation between the rubric and the scores. This turns the rubric into a validation check rather than the whole method.
3. **Stratified evaluation:** split the test set into tertiles of each score, using cut-points fitted on train. Report sensitivity, specificity and AUROC per tertile with bootstrap CIs.
4. **Explainability link:** compute the Grad-CAM energy fraction inside the lesion mask per tertile. Does the model look outside the lesion when the border is hazy? This makes Grad-CAM quantitative, not just pictures.
5. **One fix**, chosen using **val** stratified results and applied to the best backbone on val, over 3 seeds:
   - Default candidate: **Shades-of-Gray colour constancy** preprocessing. It is established in the ISIC literature, it is a single function, and it has to be applied in both training and inference.
   - Alternative if val shows the hazy-border gap is the bigger one: **subgroup-aware reweighting** using the train-split border scores.
   - Then report the test results for that fix once.
6. Ablation framing: baseline vs fix, overall and per tertile, with a paired bootstrap Δ.

What the claims can say: "associated with". Not "causes".

### D5. App scope (FastAPI + minimal React)
- Backend endpoints: `/health`, `/models`, `/predict` (file, model_id, include_gradcam) and `/metrics`, which reads saved JSON. Models load once at startup. Keep a 10 MB limit, a Pillow decode check and `Image.MAX_IMAGE_PIXELS`. Drop the rest of the hardening list.
- Frontend: Vite + React with Tailwind v4 through `@tailwindcss/vite` (single-line setup). Two pages:
  - **Analyze:** upload and preview, model select, scores labelled "model score (uncalibrated)", and original + Grad-CAM side by side.
  - **Results:** the metrics table from `/metrics`, the stratified tertile table, and static figure PNGs.
- Show the disclaimer on both pages.
- Dropped: the Dashboard/Research split, Recharts, `/compare` and a frontend test suite.

### D6. Kept optional / out of scope
- **KNN-PDNN** (optional, after the core is done): extract features from the trained or ImageNet backbones, then run sklearn KNN with k=10 on the same grouped split. It costs about an hour, and it would test whether the paper's >99% survives without leakage.
- **AlexGWO:** out of scope. 10 agents × 50 iterations means about 500 trainings, and its objective leaks the test set.
- Hair removal and median filtering are out of scope for the core.

## Options considered (major forks)

| Fork | Chosen | Rejected | Why |
|---|---|---|---|
| Dataset | HAM10000 + masks | ISIC 2020 (33k images, 1.8% malignant, no masks) / an ad-hoc 4k ISIC subset like the paper | lesion_id plus masks make the novelty measurable; ISIC 2020's extreme imbalance hurts small-model training |
| Novelty method | Automatic scores + stratified eval + one fix | A manual FP/FN rubric | You can compute rates, it is reproducible and hypothesis-driven, and the rubric is kept as validation |
| UI | FastAPI + minimal React | Full md dashboard / Gradio | Your choice; it trims about half the web work |

## Consequences
- **Easier:** defending results (CIs, seeds, no leakage), a clear contribution statement, and less web code.
- **Harder:** you need the mask download and the lesion-feature code (about 1–2 days), and the fix experiment adds 3 more training runs.
- **Revisit:** if the masks are unavailable, restrict the novelty analysis to the ISIC 2018 Task 1 subset. If H1 doesn't hold, report it as a negative result, which is still valid.

## Revised phase order (research before app)

0. **Env:** Python 3.12 venv with matched torch/torchvision CUDA, data dir outside OneDrive, `.gitignore`, `ml/config.yaml`.
1. **Data:** `ml/data/build_manifest.py` (HAM metadata, binary map, mask paths, checks) and `ml/data/create_splits.py` (grouped and stratified). Test: no lesion_id overlap between splits.
2. **Preprocessing + models:** `ml/preprocessing.py` (one source, no center-crop), `ml/models.py` (paper CNN, B0, MobileNetV2 with 2-logit heads), and a resize cache script. Test: forward-pass shapes.
3. **Training:** `ml/train.py` with config, AMP, class-weighted CE, early stop on val AUROC, best checkpoint + metadata, 3 seeds. Do a smoke run first.
4. **Evaluation:** `ml/evaluate.py` produces val-chosen thresholds, test metrics with bootstrap CIs, `summary.json`, `test_predictions.csv` and confusion matrices.
5. **Grad-CAM:** `ml/explainability.py` (pytorch-grad-cam) plus the CAM-in-mask fraction.
6. **Novelty:**
   - `ml/lesion_features.py` computes the scores.
   - `ml/stratified_analysis.py` produces the tertile tables and figures, then score validation against the manual subset.
   - Choose the fix on **val**, retrain it, and evaluate on test once.
7. **API:** `backend/app/` with routes and services that import `ml/preprocessing.py` and `ml/models.py` (no duplication), plus pytest smoke tests.
8. **Frontend:** Analyze and Results pages.
9. **README:** setup, dataset terms, commands, results tables traced to artifacts, limitations, disclaimer. KNN-PDNN only if there is time.

## Verification
- `pytest tests/` covers leakage, preprocessing parity, metric edge cases and API smoke.
- Check `nvidia-smi` during a smoke epoch: VRAM below 6 GB and GPU utilisation high. If utilisation is low, the dataloader is the bottleneck.
- Run evaluation from a clean process using only the saved checkpoints and check that the metrics match `summary.json`.
- Send the same image to `/predict` with curl and through the React UI and confirm the scores are identical.
- Every number in the README or UI must trace to a file in `artifacts/metrics/`.

## Addendum: implementation deviations (2026-10-20)

- **Resize cache dropped (D2).** All HAM10000 images are 600×450, so decoding them is cheap. More importantly, a lossy cache would make the training/evaluation tensors differ from API inference. Every path now calls `ml/preprocessing.py` directly, and `tests/test_models_preprocessing.py::test_inference_path_matches_eval_path` asserts the tensors are bit-identical.
- **Border score revised before any evaluation (D4).** The v1 transition-width score rated faint and hairy lesions as "sharp" when the training-set extremes were inspected. The primary score is now `border_haziness` = 1 / peak edge strength, measured on a hair-suppressed copy of the image (measurement only). `border_width_px` is kept as a secondary score. No validation or test model outputs had been computed when this change was made.
- **Environment.** A venv at `C:\skinsight_data\.venv` (Python 3.12.10) with `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128`. Data and artifacts live in `C:\skinsight_data`, outside OneDrive.
- **Measured cost.** On the RTX 4050, EfficientNet-B0 at batch 32 with AMP peaks at 1.5 GB VRAM and takes about 22–32 s per epoch.
- **Hair suppression for measurement.** The black-hat detector is followed by a per-component shape filter: length ≥ 40 px, width ≤ 6 px, and either aspect ≥ 4 or fill ≤ 0.25. An unfiltered DullRazor flagged a median of 18% of pixels, mostly pigment network. A two-scale variant with looser limits erased small lesions. The conservative setting was chosen by inspecting training images only.
