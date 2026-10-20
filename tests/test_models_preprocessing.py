import io

import numpy as np
import pytest
import torch
from PIL import Image

from ml.models import MODEL_IDS, build_model, gradcam_target_layer
from ml.preprocessing import eval_transform, load_rgb, preprocess_for_inference, shades_of_gray, train_transform


def _jpeg_bytes(w: int = 600, h: int = 450) -> bytes:
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_eval_transform_shape_and_dtype():
    x = eval_transform()(load_rgb(_jpeg_bytes()))
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32


def test_eval_transform_is_deterministic():
    img = load_rgb(_jpeg_bytes())
    assert torch.equal(eval_transform()(img), eval_transform()(img))


def test_train_and_eval_differ_only_by_augmentation():
    t = train_transform()
    assert any(type(s).__name__.startswith("Random") for s in t.transforms)
    assert not any(type(s).__name__.startswith("Random") for s in eval_transform().transforms)


def test_inference_path_matches_eval_path(tmp_path):
    """API (bytes) and evaluation (file path) must yield identical tensors."""
    data = _jpeg_bytes()
    p = tmp_path / "x.jpg"
    p.write_bytes(data)
    from_file = eval_transform()(load_rgb(p)).unsqueeze(0)
    assert torch.equal(preprocess_for_inference(data), from_file)


def test_grayscale_and_rgba_inputs_become_rgb():
    for mode in ("L", "RGBA"):
        buf = io.BytesIO()
        Image.new(mode, (50, 40)).save(buf, format="PNG")
        assert load_rgb(buf.getvalue()).mode == "RGB"


def test_shades_of_gray_neutralises_cast():
    arr = np.full((20, 20, 3), (200, 100, 50), dtype=np.uint8)
    out = np.asarray(shades_of_gray(Image.fromarray(arr)), dtype=float)
    assert out[..., 0].mean() == pytest.approx(out[..., 1].mean(), abs=2)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_forward_two_logits_and_gradcam_layer(model_id):
    model = build_model(model_id, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)
    layer = gradcam_target_layer(model, model_id)
    assert any(layer is m for m in model.modules())
