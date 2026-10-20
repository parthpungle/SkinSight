"""Single source of truth for image preprocessing (training, evaluation and API).

Images are resized as a whole to image_size x image_size: no center crop, so lesion
borders are never cut off (important for the border-haziness analysis).
All models use ImageNet mean/std; the from-scratch CNN uses the same fixed statistics,
so nothing is fitted on the data.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Guard against decompression bombs in uploaded files.
Image.MAX_IMAGE_PIXELS = 50_000_000


def load_rgb(source: str | Path | bytes) -> Image.Image:
    """Decode an image file path or raw bytes into an RGB PIL image."""
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img.load()
    return img.convert("RGB")


def shades_of_gray(img: Image.Image, p: int = 6) -> Image.Image:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi 2004), used in the ISIC literature.

    Rescales each channel so the Minkowski p-norm illuminant estimate becomes neutral grey.
    """
    arr = np.asarray(img, dtype=np.float32)
    illum = np.power(np.mean(arr ** p, axis=(0, 1)), 1.0 / p)
    illum = illum / (np.linalg.norm(illum) + 1e-6)
    arr = arr / (illum * np.sqrt(3) + 1e-6)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class ColorConstancy:
    def __init__(self, p: int = 6):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        return shades_of_gray(img, self.p)


def _base(image_size: int, color_constancy: bool) -> list:
    # Colour constancy runs after the resize: it is a per-channel scale, so estimating the illuminant
    # on the 224x224 image gives essentially the same correction at ~7x lower CPU cost.
    steps: list = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if color_constancy:
        steps.append(ColorConstancy())
    return steps


def _to_tensor() -> list:
    return [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def eval_transform(image_size: int = IMAGE_SIZE, color_constancy: bool = False) -> T.Compose:
    """Deterministic transform for validation, test and API inference."""
    return T.Compose(_base(image_size, color_constancy) + _to_tensor())


def train_transform(image_size: int = IMAGE_SIZE, color_constancy: bool = False) -> T.Compose:
    """Eval transform plus modest geometric augmentation (training split only)."""
    aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15)]
    return T.Compose(_base(image_size, color_constancy) + aug + _to_tensor())


def preprocess_for_inference(source: str | Path | bytes, color_constancy: bool = False) -> torch.Tensor:
    """Decode + eval transform, returning a (1, 3, H, W) batch."""
    return eval_transform(color_constancy=color_constancy)(load_rgb(source)).unsqueeze(0)


def display_image(source: str | Path | bytes, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """The resized RGB image (float32 in [0, 1]) the model sees, for Grad-CAM overlays."""
    img = load_rgb(source).resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0
