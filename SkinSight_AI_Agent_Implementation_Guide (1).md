# SkinSight — AI Agent Implementation Guide

> **Purpose:** This document is the implementation source of truth for an AI coding agent building SkinSight. Read this file fully before modifying code. Follow the phases in order, keep the implementation simple, and do not invent results or claim medical validity.
>
> **Project type:** Educational computer-vision research prototype for binary classification of dermoscopic skin-lesion images.
>
> **Core research direction:** Rebuild a compact, explainable pipeline; compare transfer learning against a lightweight CNN; investigate errors associated with hazy lesion borders and inhomogeneous lesion appearance.
>
> **Important limitation:** SkinSight is not a medical device and must not be presented as a diagnostic service. Predictions and model scores are experimental outputs, not clinical risk estimates. The interface must encourage users to seek qualified medical advice for concerns about a lesion.

---

## 1. Agent Operating Instructions

1. Read this document before coding and treat it as the project specification.
2. Inspect the existing repository before creating or replacing files. Preserve working code unless there is a clear reason to change it.
3. Implement one phase at a time. After each phase:
   - run the relevant checks;
   - fix errors before moving on;
   - summarize changed files and commands run;
   - list any blockers and assumptions.
4. Do not fabricate datasets, labels, metrics, training logs, model weights, screenshots, or test results.
5. Do not report a model as trained until training actually ran and its artifacts exist.
6. Never use the test set for hyperparameter tuning, early stopping, threshold selection, or repeated model selection. Use training/validation data for those choices and evaluate on the held-out test set at the end.
7. Fit preprocessing statistics only on the training split where applicable. Apply augmentation only to training images.
8. Avoid patient/lesion leakage. If patient or lesion identifiers are available, group splits by those identifiers. If identifiers are unavailable, document that limitation.
9. Keep all model preprocessing identical between validation/test and API inference.
10. Make model outputs and UI labels precise: a softmax/sigmoid score is a model score, not a probability of clinical malignancy unless calibrated and appropriately validated.
11. Never expose local file paths, stack traces, secrets, or full server internals in user-facing API errors.
12. Do not add cloud deployment, authentication, a database, complex microservices, or a separate job queue unless explicitly requested. The first goal is a reliable local demo.
13. Use readable, typed, modular code; avoid overengineering and unnecessary abstractions.
14. If a dependency or command is not available, inspect the environment and explain the issue rather than silently substituting an incompatible package.
15. If implementation requirements conflict with this document, flag the conflict and ask before making a major architectural change.

---

## 2. Project Goals

### Primary goals

- Classify a dermoscopic image into **benign** or **malignant** using a binary classifier.
- Compare three models using a consistent evaluation protocol:
  1. a small CNN trained from scratch;
  2. EfficientNet-B0 transfer learning;
  3. MobileNetV2 transfer learning.
- Provide an explainability view using Grad-CAM for supported models.
- Investigate errors associated with:
  - hazy or indistinct lesion borders;
  - inhomogeneous appearance (uneven color, pigmentation, or texture).
- Provide a polished React + Tailwind CSS web dashboard backed by FastAPI.
- Record evaluation results, training configuration, and known limitations so results can be reproduced.

### Out of scope for the first version

- Clinical diagnosis or treatment recommendations.
- Claiming that Grad-CAM proves causal reasoning.
- Automated medical triage or urgency decisions.
- Training a new model every time a user uploads an image.
- Real-time lesion monitoring across patient visits.
- Patient accounts, patient-history integration, or a production medical records system.
- Cloud deployment, paid services, database infrastructure, or authentication unless explicitly requested.
- Grey Wolf Optimization / AlexGWO reproduction in the initial implementation.
- KNN over deep features unless the core three-model pipeline is complete and there is time for an optional experiment.

### Definition of done

A first usable version is complete when:

- The data manifest and train/validation/test split are reproducible.
- All three models have been trained (or failures are clearly documented) and evaluated on the same untouched test split.
- Metrics are saved to disk and can be displayed in the dashboard.
- FastAPI can load a selected saved model and process a valid image.
- React can upload an image and show the returned prediction, score, and explanation when available.
- The UI has loading, error, and empty states.
- Failure examples can be reviewed and categorized using a documented rubric.
- Automated tests cover preprocessing, API validation, and key inference behavior.
- The README documents setup, data handling, model training, API startup, frontend startup, limitations, and reproducibility.

---

## 3. System Architecture

### High-level data flow

```text
                 OFFLINE RESEARCH / TRAINING
  Dataset + labels
        |
        v
  Data audit + manifest + grouped split
        |
        v
  Shared preprocessing + train-only augmentation
        |
        +----------------+------------------+
        |                |                  |
        v                v                  v
    Small CNN       EfficientNet-B0     MobileNetV2
    from scratch    transfer learning   transfer learning
        |                |                  |
        +----------------+------------------+
                         |
                         v
           Validation + final test evaluation
                         |
                         v
        Metrics JSON + confusion matrices + error CSV
                         |
                         v
                 Saved model artifacts
                         |
                         v
                  ONLINE / LOCAL DEMO

  React + Tailwind CSS (browser)
        |
        | HTTP multipart/form-data / JSON
        v
  FastAPI
    - validate upload
    - decode image
    - call shared preprocessing
    - load selected model
    - predict
    - generate Grad-CAM if requested/supported
    - return JSON + heatmap representation
        |
        v
  PyTorch inference service + saved weights
```

### Architectural rules

- **Training is offline.** Training scripts create weights and evaluation artifacts; the API only loads and uses them.
- **One source of truth for preprocessing.** Put shared inference preprocessing in a module used by both evaluation and the API.
- **Separate research code from web code.** The API should call reusable inference functions, not duplicate model logic inside route handlers.
- **Use one API process for the initial local app.** Do not create separate services for each model.
- **Load weights once at startup** when practical. Avoid loading a model for every request.
- **Use CPU/GPU selection dynamically.** CUDA for supported NVIDIA GPUs, MPS for compatible Apple Silicon operations, otherwise CPU.
- **Keep metrics static after evaluation.** `/metrics` should return saved metrics rather than run evaluation on each request.
- **Keep API output explicit.** Return the selected model, predicted class, score semantics, and explanation availability.
- **The browser should not access model files directly.** All inference goes through FastAPI.

### Main technology stack

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React + Vite | SPA, pages and components |
| Styling | Tailwind CSS | Responsive design and visual system |
| Charts | Recharts (optional) | Model metric visualizations |
| API | FastAPI + Uvicorn | HTTP routes and validation |
| Validation | Pydantic | Request/response schemas |
| ML | PyTorch + torchvision | Dataset loading, models and inference |
| Images | Pillow; OpenCV only where needed | Safe image decode and processing |
| Metrics | scikit-learn | Metrics, confusion matrix, ROC-AUC |
| Explainability | Grad-CAM implementation or a vetted compatible package | Heatmap generation |
| Tests | pytest; frontend test tool if configured | Regression checks |
| Version control | Git | Code and documentation |

Keep the dependency set small. Pin or record working dependency versions in the environment files after testing the chosen versions.

---

## 4. Repository Structure

Use this structure unless an existing repository has a strong, working convention. Keep data and generated artifacts out of ordinary Git history when large or restricted.

```text
SkinSight/
├── README.md
├── .gitignore
├── .env.example
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── vite.config.js
│   ├── postcss.config.js             # if required by chosen Tailwind version
│   ├── tailwind.config.js             # if required by chosen Tailwind version
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       ├── services/
│       │   └── api.js
│       ├── components/
│       │   ├── AppHeader.jsx
│       │   ├── ImageUploader.jsx
│       │   ├── PredictionPanel.jsx
│       │   ├── GradCamViewer.jsx
│       │   ├── ModelComparison.jsx
│       │   ├── MetricsCards.jsx
│       │   ├── LoadingState.jsx
│       │   ├── ErrorMessage.jsx
│       │   └── MedicalDisclaimer.jsx
│       └── pages/
│           ├── DashboardPage.jsx
│           ├── AnalyzePage.jsx
│           └── ResearchPage.jsx
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── core/
│       │   ├── config.py
│       │   └── device.py
│       ├── api/
│       │   └── routes/
│       │       ├── health.py
│       │       ├── predict.py
│       │       ├── models.py
│       │       └── metrics.py
│       ├── schemas/
│       │   └── prediction.py
│       └── services/
│           ├── model_registry.py
│           ├── inference_service.py
│           ├── image_service.py
│           └── explainability_service.py
├── ml/
│   ├── config.yaml
│   ├── data/
│   │   ├── build_manifest.py
│   │   └── create_splits.py
│   ├── datasets.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   └── explainability.py
├── data/
│   ├── raw/                           # not committed
│   ├── processed/                     # not committed unless tiny/allowed
│   └── splits/                        # split manifests; check data terms
├── artifacts/
│   ├── models/                        # weights; usually not committed
│   ├── metrics/
│   └── figures/
└── tests/
    ├── test_preprocessing.py
    ├── test_inference.py
    └── test_api.py
```

The precise file layout may be simplified if the repository is new. Do not create empty placeholder files for every item at once; add files as each phase needs them.

---

## 5. Dataset and Data Governance

### Source

Start with the ISIC Archive or an appropriate ISIC Challenge dataset. Before downloading, inspect the dataset terms, challenge rules, label definitions, and any restrictions on redistribution. Do not assume that every ISIC collection uses identical labels or class definitions.

### Data requirements

1. Store images locally under `data/raw/` or configure a dataset root outside the repository.
2. Build a manifest such as CSV with at least:
   - relative image path or stable image ID;
   - original label;
   - mapped binary label;
   - patient/lesion ID if available;
   - source/collection ID if available;
   - split assignment.
3. Define the binary mapping explicitly and validate it. Exclude ambiguous labels unless the chosen dataset has a documented rule for handling them.
4. Check missing files, duplicate IDs, unreadable images, class counts, image dimensions, and label inconsistencies.
5. Prefer patient- or lesion-grouped splitting when identifiers exist. No image from one patient/lesion may appear in more than one split.
6. A reasonable starting split is 70% train, 15% validation, 15% test, adjusted for group constraints and class balance.
7. Save the split seed and manifests. Make the final test split fixed and untouched during model selection.
8. Perform augmentation only after a training sample has been selected. Never augment validation or test images.
9. Do not upload private/restricted images or patient information to external services.
10. Do not commit the dataset to Git unless the license and size clearly permit it.

### Preprocessing

- Decode safely and convert to RGB.
- Resize/crop consistently to the model input dimensions, starting at 224 × 224.
- Use the normalization expected by each pretrained backbone. For a fair comparison, document any model-specific normalization; do not incorrectly apply one backbone's normalization to all models.
- Use deterministic preprocessing for validation, test and inference.
- Apply only modest training augmentations (for example horizontal/vertical flips and small rotations) where suitable. Document the choices.
- Do not implement hair removal, median filtering, lesion segmentation, or other complex preprocessing by default. Add them only as a controlled experiment, since they can alter information and make comparisons harder.
- Ensure the API and evaluation scripts use the same resizing, color conversion, and normalization logic.

---

## 6. Model Design and Training

### Model A: Small CNN from scratch

Create a compact CNN with several convolutional blocks, activation functions, pooling, and a small classification head. Keep the architecture understandable and lightweight. Avoid claiming it is state of the art. Record the parameter count and training settings.

### Model B: EfficientNet-B0 transfer learning

- Initialize from an appropriate pretrained torchvision weight set.
- Replace the final classification layer with a binary-classification head.
- First freeze the feature extractor and train the new head.
- If validation performance warrants it, unfreeze selected later blocks and fine-tune with a lower learning rate.
- Use the preprocessing associated with the selected pretrained weights.

### Model C: MobileNetV2 transfer learning

Follow the same transfer-learning protocol and fair comparison principles as EfficientNet-B0. Replace the classifier head and use its matching pretrained-weight preprocessing.

### Training configuration

Start with configurable, not hard-coded, values in `ml/config.yaml`. Reasonable starting points, not guaranteed optimal:

- Input size: 224 × 224.
- Batch size: 8–16 on an RTX 4050 with 6 GB VRAM; reduce if memory errors occur.
- Optimizer: AdamW.
- Initial head learning rate: around `1e-3`.
- Fine-tuning backbone learning rate: around `1e-5`.
- Epochs: start with 5–10, then adjust based on validation curves.
- Early stopping and checkpoint selection based on validation metrics only.
- Set random seeds where practical and record any nondeterminism.
- Use mixed precision on CUDA only when supported and tested; keep a safe fallback.
- Save best checkpoint, epoch history, configuration, and software/device details.
- Handle class imbalance through documented sampling or class weights if needed. Do not assume the dataset is balanced.
- Do not pick a model or threshold by looking at the test results.

### Device selection

Use a helper that chooses an available device:

```python
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Do not assume CUDA works on Apple Silicon. MPS is the Apple GPU path; CUDA is for supported NVIDIA GPUs. Test all selected operations on the chosen device and allow CPU fallback where feasible.

### Saved model artifacts

For each model, save enough information to reconstruct the exact inference behavior:
- state dict;
- architecture/model identifier;
- class-to-index mapping;
- input size and normalization;
- training configuration;
- selected checkpoint epoch;
- package versions if feasible.

Prefer a weights-only state dict plus a trusted, code-defined architecture. Never load arbitrary user-provided model files. Keep filenames deterministic, for example:
- `artifacts/models/small_cnn_best.pth`
- `artifacts/models/efficientnet_b0_best.pth`
- `artifacts/models/mobilenet_v2_best.pth`

---

## 7. Evaluation Protocol

Evaluate each model on the exact same held-out test samples. Report at least:

- accuracy;
- precision;
- recall/sensitivity for the malignant class;
- specificity;
- F1-score;
- ROC-AUC where both classes and usable scores are present;
- confusion matrix;
- number of examples per class;
- training duration;
- inference latency (report measurement method and sample count);
- parameter count and model artifact size.

Define the positive class clearly (recommended: malignant = positive) and compute specificity as TN / (TN + FP). Handle zero denominators explicitly. Do not silently omit failed metrics.

Save machine-readable metrics, for example `artifacts/metrics/summary.json`, and per-image predictions, for example `artifacts/metrics/test_predictions.csv`, containing:
- sample ID;
- true label;
- predicted label;
- score for the positive class;
- model ID;
- split/version metadata.

For confidence intervals or statistical comparisons, add them only if implemented correctly and described. Avoid presenting a single test-set result as a guarantee of generalization.

### Error analysis

Inspect false positives and false negatives. Create a consistent, documented annotation rubric for:
- border clarity: clear / partly hazy / highly indistinct;
- appearance uniformity: relatively homogeneous / mixed / strongly inhomogeneous;
- visible artifacts: hair / illumination / low contrast / other;
- whether the prediction was correct.

These annotations may be manual for a manageable subset. If possible, have labels reviewed consistently and document who annotated and how disagreements were handled. Compare error rates across categories, with sample counts. Do not claim that a characteristic causes an error merely because it co-occurs with one.

Use Grad-CAM as a qualitative aid. A heatmap is not proof of the model's reasoning and may be misleading.

---

## 8. Explainability / Grad-CAM

- Implement Grad-CAM for the supported CNN architectures, using a clearly specified target convolutional layer for each.
- Verify the target layer exists for each model; do not assume the same layer path works for all architectures.
- Preserve gradients for the target class even if ordinary inference uses `torch.inference_mode()`; Grad-CAM may need a separate gradient-enabled code path.
- Resize the heatmap to the displayed image dimensions and blend it with the original image using a documented colormap and opacity.
- Return a safe representation for the frontend, preferably a PNG encoded as base64 or a URL to a generated temporary image. For a local demo, base64 is simple; ensure response sizes remain reasonable.
- Use stable color scales and labels. Display original and overlay side by side.
- If Grad-CAM fails for a model, return the prediction with `explanation_available: false` and a user-safe message; do not fail the entire prediction unless explanation is mandatory.
- Do not describe Grad-CAM as a segmentation mask, medical explanation, or proof of causal feature importance.

---

## 9. FastAPI Backend Contract

### Required endpoints

#### `GET /health`

Return a simple health response, for example:

```json
{
  "status": "ok"
}
```

#### `GET /models`

Return available models and whether their trained artifacts are present. Do not report a model as available if the required weights cannot load.

Example shape:

```json
{
  "models": [
    {"id": "small_cnn", "name": "Small CNN", "ready": true},
    {"id": "efficientnet_b0", "name": "EfficientNet-B0", "ready": true},
    {"id": "mobilenet_v2", "name": "MobileNetV2", "ready": false}
  ]
}
```

#### `POST /predict`

Accept `multipart/form-data` with:
- `file`: required image file;
- `model_id`: required or defaulted to a documented model ID;
- `include_gradcam`: optional boolean.

Validate:
- allowed image formats (for example JPEG and PNG);
- actual decodability, not only the filename or MIME type;
- a reasonable configurable byte limit;
- valid model ID;
- safe image dimensions to avoid decompression/resource abuse.

Example successful response:

```json
{
  "model_id": "efficientnet_b0",
  "model_name": "EfficientNet-B0",
  "predicted_class": "benign",
  "class_scores": {
    "benign": 0.72,
    "malignant": 0.28
  },
  "score_semantics": "uncalibrated_model_score",
  "explanation_available": true,
  "gradcam_png_base64": "<base64-encoded PNG or null>",
  "disclaimer": "Research prototype only; not a medical diagnosis."
}
```

The values above are examples of response structure only, not real model results. The model score must not be labeled a clinical probability. If calibration is later implemented and validated, describe its method and limitations explicitly.

#### `GET /metrics`

Return saved evaluation metrics and metadata. Do not calculate test metrics on a request.

Optional later endpoint:
- `POST /compare`: run inference with multiple loaded models and return side-by-side outputs. Only add after the single-model `/predict` endpoint works reliably.

### Backend implementation rules

- Use Pydantic response models to enforce a stable API contract.
- Load models at application startup or lazily on first use and cache them. Never reload the weights on every request.
- Use an explicit model registry mapping stable IDs to constructors, weights, preprocessing and Grad-CAM layer configuration.
- Keep route handlers thin; put image decode, preprocessing, inference and explanation logic in service modules.
- Configure CORS only for the frontend development origin(s), such as the local Vite origin. Avoid wildcard origins when credentials are enabled.
- Return 400/413/415/422-style errors as appropriate with safe messages.
- Do not trust client-provided image filenames or MIME types.
- Do not log uploaded image bytes or sensitive information.
- Add a request-size limit and handle malformed image files.
- Do not let a missing checkpoint crash unrelated health routes. Report model readiness and return a clear prediction error if the selected model is unavailable.
- Document how to run the backend and what device it uses.

---

## 10. React + Tailwind Frontend

### Design goals

Build a modern, accessible research dashboard rather than a hospital or diagnostic interface. Use a clean visual hierarchy, restrained color palette, readable typography, responsive layout, and clear spacing. Avoid fake medical certifications, fabricated performance numbers, or alarming clinical language.

### Required pages / sections

1. **Dashboard**
   - Project title and short research description.
   - Model performance cards populated from `/metrics`.
   - Model comparison chart/table populated from saved evaluation results.
   - Clear note when metrics are not yet available.
2. **Analyze Image**
   - Drag-and-drop and click-to-upload image input.
   - Client-side preview and file validation for convenience (backend remains authoritative).
   - Model selector populated from `/models`.
   - Analyze button, loading indicator, error handling, and result state.
   - Predicted class and explicitly labeled model scores.
   - Original image and Grad-CAM overlay side by side when available.
   - Medical/research disclaimer.
3. **Research / Error Analysis**
   - Summary of evaluation protocol.
   - False-positive and false-negative examples if saved artifacts exist.
   - Border clarity, appearance uniformity, and artifact category summaries when annotations exist.
   - Empty state if analysis has not been generated yet.

### Component responsibilities

- `ImageUploader`: file selection, drag/drop, preview and client-side validation.
- `PredictionPanel`: class and score display with careful wording.
- `GradCamViewer`: original and heatmap display; show fallback when unavailable.
- `ModelComparison`: compare models using real `/metrics` values.
- `MetricsCards`: show sensitivity, specificity, F1, ROC-AUC, etc. with definitions/tooltips if practical.
- `LoadingState`: indicate request in progress.
- `ErrorMessage`: concise, actionable error text.
- `MedicalDisclaimer`: persistent, readable prototype limitation.
- `services/api.js`: centralized API base URL and typed/validated request handling where feasible.

### Frontend state requirements

- Keep upload, selected model, loading, API error, prediction and explanation state separate.
- Disable duplicate submissions while a request is running.
- Reset stale results when the user selects a new image or model, or clearly indicate which image/model produced the current result.
- Do not hardcode example model scores in production UI.
- Use accessible labels, keyboard navigation, meaningful alt text, and visible focus styles.
- Provide empty, loading, success, and failure states.
- Keep API base URL in an environment variable such as `VITE_API_BASE_URL`; include `.env.example`, never commit secrets.
- Use the browser only for presentation and input validation. The backend must validate all inputs again.

### Tailwind version

Choose a currently supported Tailwind setup compatible with the installed Vite/React project and follow that version's official setup. Do not blindly mix Tailwind v3 and v4 configuration instructions. Record the actual package versions in `package.json`.

---

## 11. API / Frontend Integration

1. Start FastAPI and confirm `GET /health` works.
2. Confirm `GET /models` reports correct readiness.
3. Test `POST /predict` with a known local sample using curl or the API docs.
4. Test malformed images, unsupported file formats, missing checkpoints and invalid model IDs.
5. Configure React's `VITE_API_BASE_URL` to point at the backend.
6. Upload the same test image through the UI and compare the output with the direct API response.
7. Verify the displayed class-score order matches the model's class mapping.
8. Verify Grad-CAM corresponds to the selected model and predicted/target class.
9. Test long-running requests and show loading state; do not claim inference is complete before a response arrives.
10. Confirm the metrics page shows saved test metrics, not fabricated demo values.

Use the FastAPI-generated OpenAPI docs (`/docs`) during development.

---

## 12. Testing and Quality Gates

### ML tests

- Dataset manifest has valid paths and labels.
- Split groups do not overlap by patient/lesion ID.
- Train-only augmentation is not applied to validation/test.
- Preprocessing returns expected tensor shape and dtype.
- Model forward pass returns expected class logits.
- Positive-class score is mapped to the correct label.
- Evaluation metrics handle missing classes and zero denominators.
- Checkpoint metadata matches the inference preprocessing.

### API tests

- Health endpoint returns success.
- Unknown model IDs return a safe client error.
- Invalid or corrupt images are rejected.
- Unsupported image formats are rejected.
- Oversized uploads are rejected.
- Missing model weights produce a controlled error.
- Valid image returns a schema-conforming response.
- Grad-CAM failure does not crash prediction if fallback is intended.
- Metrics endpoint reads saved results and does not start training.

### Frontend checks

- Frontend builds successfully.
- Upload preview works for supported images.
- Loading and error states are visible.
- New uploads do not show stale predictions.
- API errors are rendered safely.
- Dashboard uses API metrics, not hard-coded scores.
- Responsive layout works on desktop and smaller screens.
- Accessibility basics: labels, keyboard focus, alt text, contrast.

### Before marking a phase complete

Run available formatters, linters, unit tests, backend smoke tests, and frontend build. If a tool is not configured, do not pretend it ran; report it as not run.

---

## 13. Device and Local Development

The primary known development GPU is an NVIDIA RTX 4050 with 6 GB VRAM. Start with 224 × 224 images and batch size 8; increase only after confirming memory headroom. Use mixed precision only after testing. If out-of-memory errors occur, reduce batch size first.

On a Windows/NVIDIA setup:
- use a compatible PyTorch build with CUDA support;
- verify `torch.cuda.is_available()` before training;
- select CUDA only if available.

On Apple Silicon:
- CUDA is not supported natively;
- use PyTorch MPS where available;
- test operation compatibility and fall back to CPU if necessary.

Recommended local processes:

```text
Terminal 1: backend API
Terminal 2: frontend Vite dev server
Terminal 3: ML training/evaluation commands as needed
```

Keep training separate from the API process. Do not train on every application startup.

---

## 14. Implementation Roadmap for the Agent

### Phase 0 — Repository and environment inspection

Tasks:
- Inspect current files and Git status.
- Identify Python, Node, npm and GPU/device versions.
- Determine whether a frontend/backend scaffold already exists.
- Check available dataset and existing model artifacts.
- Create a concise implementation checklist.
- Do not overwrite useful work.

Acceptance criteria:
- Environment and repository state documented.
- Missing prerequisites identified.
- No unrequested destructive changes.

### Phase 1 — Dataset manifest and splits

Tasks:
- Implement dataset manifest generation and validation.
- Implement grouped train/validation/test split when identifiers allow.
- Save split manifests and class counts.
- Add tests for missing files, invalid labels and leakage.

Acceptance criteria:
- Reproducible split files exist.
- Leakage checks pass or limitation is documented.
- No training has begun on unverified labels.

### Phase 2 — Preprocessing and model definitions

Tasks:
- Implement shared dataset loading and model constructors.
- Add model-specific normalization for pretrained models.
- Add device selection.
- Test forward passes using synthetic tensors before full training.

Acceptance criteria:
- Each model produces two logits per image.
- Tensor shapes and class mapping are documented.
- Model construction tests pass.

### Phase 3 — Training and checkpointing

Tasks:
- Implement configurable training loop with validation, early stopping and best-checkpoint saving.
- Log epoch loss and validation metrics.
- Save configuration and environment metadata.
- Begin with a small smoke run, then full training.

Acceptance criteria:
- Training can resume or restart predictably as designed.
- Checkpoints are written and can be loaded.
- No test metrics are used for model selection.

### Phase 4 — Evaluation and failure analysis

Tasks:
- Evaluate each saved checkpoint on the same untouched test set.
- Save metrics JSON, confusion matrix figures and per-image predictions.
- Add a documented annotation workflow for border clarity, appearance uniformity and artifacts.
- Generate a concise research summary based only on actual results.

Acceptance criteria:
- All reported values trace to saved outputs.
- Test set and class mapping are fixed.
- No performance claims are invented.

### Phase 5 — Grad-CAM

Tasks:
- Implement architecture-specific Grad-CAM.
- Validate output size, finite values and overlay alignment.
- Save sample overlays for correct and incorrect predictions.
- Document model-specific target layers and limitations.

Acceptance criteria:
- At least one explanation can be generated for each supported architecture, or unsupported cases are explicitly reported.
- Prediction still works if optional explanation generation fails.

### Phase 6 — FastAPI

Tasks:
- Implement health, models, prediction and metrics endpoints.
- Load available models once and cache them.
- Validate images and request parameters.
- Add CORS for the local frontend origin.
- Add endpoint tests and manual API smoke tests.

Acceptance criteria:
- OpenAPI docs load.
- `/predict` works for a valid sample.
- Invalid requests fail safely.
- Missing artifacts are handled clearly.

### Phase 7 — React + Tailwind UI

Tasks:
- Set up React/Vite and a compatible Tailwind version.
- Build dashboard, analysis and research sections.
- Add reusable components and centralized API client.
- Implement upload preview, model selector, loading/error/success states.
- Display only real backend metrics and results.

Acceptance criteria:
- Frontend builds.
- Upload-to-prediction flow works end to end.
- No placeholder metrics are presented as real results.
- UI is responsive and accessible.

### Phase 8 — Integration, documentation and demo

Tasks:
- Run full end-to-end tests.
- Verify API and frontend predictions match.
- Add README with setup, dataset instructions, commands, architecture, API contract, limitations and results.
- Add screenshots only from the actual running application.
- Check `.gitignore` for data, secrets, environments, caches and large artifacts.

Acceptance criteria:
- A new developer can follow README to run the project, subject to obtaining the dataset and model artifacts.
- Test results and limitations are honestly documented.
- No secrets or restricted data are committed.

---

## 15. Suggested Commands (Adapt to the Actual Environment)

Do not run all commands blindly; inspect installed versions and choose the compatible setup.

### Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Expected local API address: `http://127.0.0.1:8000`  
Interactive API documentation: `http://127.0.0.1:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Use the local URL printed by Vite. Configure `VITE_API_BASE_URL` to the FastAPI base URL.

### ML

Training/evaluation commands must be implemented and documented by the agent after inspecting the actual code and config. Prefer commands with explicit config paths and output directories. Do not invent a CLI command before implementing it.

---

## 16. Research Integrity and Reporting

- The reference paper reported very high performance on its experimental setup. Those reported values are not targets, guarantees, or evidence that this implementation has reproduced the paper.
- This project is a focused redesign, not an exact replication of every method in the paper.
- Report the exact dataset version, image count, split strategy, label mapping, random seed, preprocessing, model weights, hyperparameters and metrics.
- Clearly distinguish the paper's published findings from the project's own measured results.
- Do not compare scores across papers as if the datasets and protocols were identical.
- If a model underperforms, report it honestly and investigate rather than tuning on the test set.
- Treat hazy borders and inhomogeneous appearance as error-analysis categories to test, not confirmed causes.
- Explain that Grad-CAM is a qualitative visualization, not a clinical rationale.
- Include sensitivity and specificity because accuracy alone can hide important error patterns.
- Do not present uncalibrated output scores as reliable clinical probabilities.
- Include a visible disclaimer in the UI and README.

Suggested disclaimer:

> **Research prototype only.** SkinSight is an educational machine-learning project, not a medical device. Its outputs may be wrong and are not a diagnosis or a measure of clinical risk. Do not use it to make healthcare decisions; consult a qualified healthcare professional about skin concerns.

---

## 17. Final Agent Handoff Checklist

Before declaring the project complete, confirm each item:

- [ ] Existing repository inspected and useful code preserved.
- [ ] Dataset source, license/terms and binary label mapping documented.
- [ ] Data manifest validated.
- [ ] Patient/lesion leakage checked where IDs are available.
- [ ] Train/validation/test splits are fixed and reproducible.
- [ ] Shared preprocessing implemented and tested.
- [ ] Small CNN trained and evaluated.
- [ ] EfficientNet-B0 trained and evaluated.
- [ ] MobileNetV2 trained and evaluated.
- [ ] Metrics and per-image test predictions saved.
- [ ] Error-analysis rubric documented and sample counts shown.
- [ ] Grad-CAM implemented or unsupported cases documented.
- [ ] FastAPI endpoints implemented and tested.
- [ ] React + Tailwind UI implemented and built.
- [ ] End-to-end image upload and prediction works.
- [ ] Model scores are described accurately.
- [ ] Loading, empty, error and success states are present.
- [ ] Medical/research disclaimer appears in UI and README.
- [ ] No fabricated results, secrets or restricted dataset files committed.
- [ ] README contains reproducible setup instructions.
- [ ] All executed tests and any unrun checks are reported truthfully.

**Final instruction to the coding agent:** Prioritize correctness, reproducibility, and a working end-to-end demo. Finish the simplest valid implementation before adding optional features. Never trade research integrity for impressive-looking numbers or UI.
