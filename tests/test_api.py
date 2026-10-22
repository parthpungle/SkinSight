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


# ---- /compare, /samples, /novelty -------------------------------------------------------------

def test_compare_returns_one_result_per_ready_model(client):
    r = client.post("/compare", files={"file": ("x.jpg", _image_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert [x["model_id"] for x in results] == ["small_cnn"]
    assert results[0]["explanation_available"] is False


def test_compare_rejects_corrupt_and_oversized(client, monkeypatch):
    assert client.post("/compare", files={"file": ("x.jpg", b"junk", "image/jpeg")}).status_code == 415
    import backend.app.main as main
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    big = _image_bytes(size=(400, 400))
    assert client.post("/compare", files={"file": ("x.jpg", big, "image/jpeg")}).status_code == 413


def test_samples_and_novelty_unavailable_without_artifacts(client):
    assert client.get("/samples").json()["available"] is False
    assert client.get("/samples/ISIC_0000001").status_code == 404
    assert client.get("/novelty").json() == {"available": False}


@pytest.fixture()
def data_client(tmp_path, monkeypatch):
    import json

    import pandas as pd
    art = tmp_path / "artifacts"
    for d in ("models", "metrics", "novelty"):
        (art / d).mkdir(parents=True)
    (tmp_path / "splits").mkdir()
    (tmp_path / "raw" / "images").mkdir(parents=True)
    model = build_model("small_cnn", pretrained=False)
    torch.save({"state_dict": model.state_dict(), "model_id": "small_cnn", "class_names": ["benign", "malignant"],
                "image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                "color_constancy": False, "seed": 0, "epoch": 1, "val_auroc": 0.5, "val_threshold": 0.5},
               art / "models" / "small_cnn_s0.pth")
    ids = [f"ISIC_000000{i}" for i in range(8)]
    split = ["train"] * 3 + ["test"] * 5
    labels = [0, 1, 0, 1, 0, 0, 1, 0]
    for i in ids:
        (tmp_path / "raw" / "images" / f"{i}.jpg").write_bytes(_image_bytes())
    pd.DataFrame({"image_id": ids, "lesion_id": [f"L{k}" for k in range(8)], "dx": ["nv"] * 7 + ["mel"],
                  "label": labels, "split": split,
                  "image_path": [f"raw/images/{i}.jpg" for i in ids]}).to_csv(tmp_path / "splits" / "manifest_split.csv",
                                                                             index=False)
    pd.DataFrame({"image_id": ids, "split": split, "border_haziness": [0.5, 0.7, 0.9, 0.4, 0.6, 0.8, 1.0, 0.75],
                  "lab_sd": [10, 12, 14, 9, 11, 13, 15, 12.5], "contrast_L": 10.0,
                  "hair_frac": 0.0}).to_csv(art / "novelty" / "lesion_features.csv", index=False)
    (art / "metrics" / "summary.json").write_text(json.dumps({"served_checkpoints": {"efficientnet_b0": "efficientnet_b0_s0"}}))
    test_ids = ids[3:]
    pd.DataFrame({"image_id": test_ids, "run": "efficientnet_b0_s0", "score_malignant": [0.9, 0.2, 0.1, 0.3, 0.8],
                  "pred_label": [1, 0, 0, 0, 1], "threshold": 0.5}).to_csv(art / "metrics" / "test_predictions.csv",
                                                                         index=False)
    (art / "novelty" / "stratified_test.json").write_text(json.dumps({"split": "test", "scores": {}}))
    pd.DataFrame({"image_id": test_ids, "true_label": [1, 0, 0, 1, 0], "pred_label": [1, 0, 0, 0, 1],
                  "cam_in_mask": [0.6, 0.5, 0.4, 0.9, 0.1], "mask_area_frac": [0.3, 0.25, 0.2, 0.3, 0.2]}).to_csv(
        art / "novelty" / "cam_in_mask_efficientnet_b0_s0_test.csv", index=False)
    monkeypatch.setenv("SKINSIGHT_DATA", str(tmp_path))
    from backend.app.main import app
    with TestClient(app) as c:
        yield c


def test_samples_filter_and_paginate(data_client):
    body = data_client.get("/samples").json()
    assert body["available"] and body["total"] == 5 and body["reference_run"] == "efficientnet_b0_s0"
    assert {x["image_id"] for x in body["items"]} == {f"ISIC_000000{i}" for i in range(3, 8)}
    assert data_client.get("/samples?label=1").json()["total"] == 2
    assert data_client.get("/samples?dx=mel").json()["total"] == 1
    page = data_client.get("/samples?limit=2&offset=4").json()
    assert page["total"] == 5 and len(page["items"]) == 1
    # disagreements: ISIC_0000006 (true 1, pred 0) and ISIC_0000007 (true 0, pred 1)
    assert {x["image_id"] for x in data_client.get("/samples?disagree=true").json()["items"]} == {
        "ISIC_0000006", "ISIC_0000007"}
    assert data_client.get("/samples?haziness_tertile=bogus").status_code == 422


def test_sample_detail_and_image(data_client):
    row = data_client.get("/samples/ISIC_0000003").json()
    assert row["label"] == 1 and row["ref_score"] == 0.9 and row["haziness_tertile"] in ("low", "mid", "high")
    img = data_client.get("/samples/ISIC_0000003/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_sample_image_rejects_bad_ids(data_client):
    for bad in ("ISIC_0000000", "ISIC_123", "..%2F..%2Fsplits%2Fmanifest_split.csv", "ISIC_9999999"):
        assert data_client.get(f"/samples/{bad}/image").status_code == 404, bad  # train id, malformed, traversal, unknown


def test_novelty_enrichment_from_cam_csv(data_client):
    body = data_client.get("/novelty").json()
    e = body["enrichment"]["test"]["efficientnet_b0_s0"]
    assert e["TP"] == {"median": 2.0, "n": 1}            # 0.6 / 0.3
    assert e["FP"] == {"median": 0.5, "n": 1}            # 0.1 / 0.2
    assert e["FN"]["n"] == 1 and e["TN"]["n"] == 2
    assert body["verdicts"] == []
