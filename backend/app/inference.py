"""Upload validation, prediction and optional Grad-CAM, built on the shared ml/ modules."""
from __future__ import annotations

import io
import logging

import numpy as np
import torch
from PIL import Image

from ml.explainability import gradcam_map, overlay, to_png_base64
from ml.models import CLASS_NAMES
from ml.preprocessing import display_image, preprocess_for_inference

from .registry import ModelEntry
from .schemas import Prediction

log = logging.getLogger(__name__)
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_SIDE = 8000


class InvalidImage(ValueError):
    pass


def validate_image(data: bytes) -> None:
    """Check real decodability and format; never trust filename or client MIME type."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            im.verify()
    except Exception as exc:
        raise InvalidImage("The file could not be read as an image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImage("Only JPEG and PNG images are supported.")
    if max(w, h) > MAX_SIDE:
        raise InvalidImage(f"Image is too large (max {MAX_SIDE}px per side).")


def predict(entry: ModelEntry, data: bytes, include_gradcam: bool, device: torch.device) -> Prediction:
    ckpt = entry.ckpt
    cc = bool(ckpt.get("color_constancy", False))
    x = preprocess_for_inference(data, cc).to(device)
    with torch.inference_mode():
        probs = torch.softmax(entry.model(x).float(), 1)[0].cpu().numpy()
    thr = float(ckpt["val_threshold"])
    pred = int(probs[1] >= thr)

    b64, msg = None, None
    if include_gradcam:
        try:
            cam = gradcam_map(entry.model, ckpt["model_id"], x.clone(), pred)
            b64 = to_png_base64(overlay(display_image(data, ckpt["image_size"]), cam))
        except Exception:
            log.exception("Grad-CAM failed for %s", entry.id)
            msg = "Explanation could not be generated for this image; the prediction is unaffected."
    else:
        msg = "Explanation not requested."

    return Prediction(
        model_id=entry.id, model_name=entry.name, checkpoint=entry.run or "",
        predicted_class=CLASS_NAMES[pred],
        class_scores={c: round(float(p), 4) for c, p in zip(CLASS_NAMES, np.asarray(probs))},
        decision_threshold=round(thr, 4),
        explanation_available=b64 is not None, explanation_message=msg, gradcam_png_base64=b64,
    )
