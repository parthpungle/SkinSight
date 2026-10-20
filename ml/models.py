"""Model registry: constructors, Grad-CAM target layers, and parameter groups.

Every model outputs 2 logits: index 0 = benign, index 1 = malignant.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

CLASS_NAMES = ("benign", "malignant")
MODEL_IDS = ("small_cnn", "efficientnet_b0", "mobilenet_v2")
MODEL_NAMES = {
    "small_cnn": "Small CNN (paper-style)",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v2": "MobileNetV2",
}


class PaperCNN(nn.Module):
    """Small CNN following Magdy et al. (2023): 5 conv layers with 32/32/64/128/128 filters + BN.

    Deviation: global average pooling before the single FC layer (the paper flattens), which keeps
    the parameter count small and the network input-size agnostic.
    """

    def __init__(self, num_classes: int = 2, widths: tuple[int, ...] = (32, 32, 64, 128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for w in widths:
            layers += [nn.Conv2d(in_ch, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            in_ch = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_ch, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


def build_model(model_id: str, pretrained: bool = True) -> nn.Module:
    if model_id == "small_cnn":
        return PaperCNN()
    if model_id == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    if model_id == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    raise ValueError(f"Unknown model_id: {model_id}")


def gradcam_target_layer(model: nn.Module, model_id: str) -> nn.Module:
    """Last convolutional block of each architecture."""
    if model_id == "small_cnn":
        # Last conv block output before pooling: the ReLU of block 5 (index -2 skips the MaxPool).
        return model.features[-2]
    if model_id in ("efficientnet_b0", "mobilenet_v2"):
        return model.features[-1]
    raise ValueError(f"Unknown model_id: {model_id}")


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.classifier.parameters())


def backbone_parameters(model: nn.Module) -> list[nn.Parameter]:
    head = {id(p) for p in head_parameters(model)}
    return [p for p in model.parameters() if id(p) not in head]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
