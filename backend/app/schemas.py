from __future__ import annotations

from pydantic import BaseModel

DISCLAIMER = (
    "Research prototype only. SkinSight is an educational machine-learning project, not a medical device. "
    "Its outputs may be wrong and are not a diagnosis or a measure of clinical risk. Consult a qualified "
    "healthcare professional about skin concerns."
)


class Health(BaseModel):
    status: str
    device: str


class ModelInfo(BaseModel):
    id: str
    name: str
    ready: bool
    checkpoint: str | None = None
    val_auroc: float | None = None


class ModelList(BaseModel):
    models: list[ModelInfo]


class Prediction(BaseModel):
    model_id: str
    model_name: str
    checkpoint: str
    predicted_class: str
    class_scores: dict[str, float]
    decision_threshold: float
    decision_rule: str = "malignant if malignant score >= threshold chosen on validation (Youden J)"
    score_semantics: str = "uncalibrated_model_score"
    explanation_available: bool
    explanation_message: str | None = None
    gradcam_png_base64: str | None = None
    disclaimer: str = DISCLAIMER


class CompareResult(BaseModel):
    results: list[Prediction]
