"""PyTorch network definitions shared by inference wrappers and training scripts."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger("models.arch")

FEATURE_DIM = 1280  # EfficientNet-B0 pooled feature size


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """EfficientNet-B0 with a binary REAL/FAKE head."""
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_xception(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """XceptionNet (FaceForensics++ baseline architecture) for facial manipulation artifacts."""
    import timm

    return timm.create_model("legacy_xception", pretrained=pretrained, num_classes=num_classes)


class CNNLSTM(nn.Module):
    """Spatial CNN encoder per frame + bidirectional LSTM over the frame sequence."""

    def __init__(self, pretrained_backbone: bool = False, hidden: int = 256, freeze_backbone: bool = True):
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = efficientnet_b0(weights=weights)
        self.encoder = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten())
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(FEATURE_DIM, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * 2, 2))

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) -> features (B, T, FEATURE_DIM)"""
        b, t = frames.shape[:2]
        feats = self.encoder(frames.flatten(0, 1))
        return feats.view(b, t, -1)

    def classify_features(self, feats: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(feats)
        return self.head(out.mean(dim=1))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.encode_frames(frames))


def load_state_dict_flex(model: nn.Module, path: Path) -> None:
    """Load a checkpoint tolerant of common wrappers/prefixes (DataParallel, FF++ `last_linear`)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        ckpt = ckpt.state_dict()
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    cleaned = {}
    for k, v in ckpt.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        k = k.replace("last_linear.", "fc.")
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        log.warning("%s: %d missing keys (e.g. %s)", path.name, len(missing), missing[:3])
    if unexpected:
        log.warning("%s: %d unexpected keys (e.g. %s)", path.name, len(unexpected), unexpected[:3])
