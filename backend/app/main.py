"""SkinSight local API.

Run from the repository root:
    uvicorn backend.app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ml.config import data_path, load_config
from ml.train import get_device

from .inference import InvalidImage, predict, validate_image
from .registry import load_registry
from .schemas import Health, ModelInfo, ModelList, Prediction

MAX_UPLOAD_BYTES = int(float(os.environ.get("SKINSIGHT_MAX_UPLOAD_MB", "10")) * 2**20)
CORS_ORIGINS = os.environ.get("SKINSIGHT_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    STATE["artifacts"] = data_path(cfg, "artifacts")
    STATE["device"] = get_device()
    STATE["registry"] = load_registry(STATE["artifacts"], STATE["device"])
    yield
    STATE.clear()


app = FastAPI(title="SkinSight API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", device=str(STATE["device"]))


@app.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(models=[
        ModelInfo(id=e.id, name=e.name, ready=e.ready, checkpoint=e.run, val_auroc=e.ckpt.get("val_auroc"))
        for e in STATE["registry"].values()
    ])


@app.post("/predict", response_model=Prediction)
async def predict_route(file: UploadFile = File(...), model_id: str = Form("efficientnet_b0"),
                        include_gradcam: bool = Form(True)) -> Prediction:
    entry = STATE["registry"].get(model_id)
    if entry is None:
        raise HTTPException(422, f"Unknown model_id '{model_id}'.")
    if not entry.ready:
        raise HTTPException(503, f"Model '{model_id}' has no trained checkpoint available.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        validate_image(data)
    except InvalidImage as exc:
        raise HTTPException(415, str(exc)) from None
    return predict(entry, data, include_gradcam, STATE["device"])


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


FIGURE_DIRS = ("metrics", "novelty")
FIGURE_NAME = re.compile(r"^[\w\-]+\.png$")


@app.get("/metrics")
def metrics() -> dict:
    """Saved evaluation outputs only; never runs evaluation."""
    art: Path = STATE["artifacts"]
    summary = _read_json(art / "metrics" / "summary.json")
    figures = sorted(p.name for d in FIGURE_DIRS for p in (art / d).glob("*.png")) if summary else []
    return {
        "available": summary is not None,
        "summary": summary,
        "stratified_test": _read_json(art / "novelty" / "stratified_test.json"),
        "figures": figures,
    }


@app.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    if not FIGURE_NAME.match(name):
        raise HTTPException(404, "Not found.")
    for d in FIGURE_DIRS:
        path = STATE["artifacts"] / d / name
        if path.is_file():
            return FileResponse(path, media_type="image/png")
    raise HTTPException(404, "Not found.")
