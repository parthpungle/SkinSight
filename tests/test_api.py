"""API tests against a temporary artifacts dir holding an untrained small-CNN checkpoint."""
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ml.models import build_model


@pytest.fixture()
def client(tmp_path, monkeypatch):
    (tmp_path / "artifacts" / "models").mkdir(parents=True)
    model = build_model("small_cnn", pretrained=False)
    torch.save({"state_dict": model.state_dict(), "model_id": "small_cnn", "class_names": ["benign", "malignant"],
                "image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                "color_constancy": False, "seed": 0, "epoch": 1, "val_auroc": 0.5, "val_threshold": 0.5},
               tmp_path / "artifacts" / "models" / "small_cnn_s0.pth")
    monkeypatch.setenv("SKINSIGHT_DATA", str(tmp_path))
    from backend.app.main import app
    with TestClient(app) as c:
        yield c


def _image_bytes(fmt: str = "JPEG", size=(120, 90)) -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def _post(client, data: bytes, model_id="small_cnn", gradcam=True, name="x.jpg"):
    return client.post("/predict", files={"file": (name, data, "image/jpeg")},
                       data={"model_id": model_id, "include_gradcam": str(gradcam).lower()})


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_models_readiness(client):
    models = {m["id"]: m["ready"] for m in client.get("/models").json()["models"]}
    assert models == {"small_cnn": True, "efficientnet_b0": False, "mobilenet_v2": False}


def test_valid_prediction_schema(client):
    r = _post(client, _image_bytes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["class_scores"]) == {"benign", "malignant"}
    assert abs(sum(body["class_scores"].values()) - 1) < 1e-3
    assert body["predicted_class"] == ("malignant" if body["class_scores"]["malignant"] >= 0.5 else "benign")
    assert body["score_semantics"] == "uncalibrated_model_score"
    assert body["explanation_available"] and body["gradcam_png_base64"]


def test_png_accepted_and_gradcam_optional(client):
    body = _post(client, _image_bytes("PNG"), gradcam=False, name="x.png").json()
    assert body["explanation_available"] is False and body["gradcam_png_base64"] is None


def test_unknown_model(client):
    assert _post(client, _image_bytes(), model_id="resnet999").status_code == 422


def test_missing_checkpoint(client):
    assert _post(client, _image_bytes(), model_id="efficientnet_b0").status_code == 503


def test_corrupt_image(client):
    assert _post(client, b"not an image at all").status_code == 415


def test_unsupported_format(client):
    assert _post(client, _image_bytes("GIF"), name="x.gif").status_code == 415


def test_oversized_upload(client, monkeypatch):
    import backend.app.main as main
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    assert _post(client, _image_bytes(size=(400, 400))).status_code == 413


def test_gradcam_failure_keeps_prediction(client, monkeypatch):
    import backend.app.inference as inf

    def boom(*a, **k):
        raise RuntimeError("cam failed")
    monkeypatch.setattr(inf, "gradcam_map", boom)
    body = _post(client, _image_bytes()).json()
    assert body["predicted_class"] in ("benign", "malignant")
    assert body["explanation_available"] is False and "could not be generated" in body["explanation_message"]


def test_metrics_reads_saved_files_only(client):
    body = client.get("/metrics").json()
    assert body["available"] is False and body["summary"] is None


def test_figure_path_traversal_rejected(client):
    assert client.get("/figures/..%2F..%2Fsecret.png").status_code == 404
