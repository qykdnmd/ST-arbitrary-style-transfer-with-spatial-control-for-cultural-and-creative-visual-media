import torch
import torch.nn as nn
from torchvision import models

# VGG19 feature indices (torchvision) for relu1_1 .. relu5_1
_RELU_SLICES = {
    "relu1_1": 2,
    "relu2_1": 7,
    "relu3_1": 12,
    "relu4_1": 21,
    "relu5_1": 30,
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGFeatures(nn.Module):
    """Frozen VGG19 feature extractor returning relu1_1..relu5_1 activations."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx in _RELU_SLICES.values():
            self.slices.append(vgg[prev:idx])
            prev = idx
        self.layer_names = list(_RELU_SLICES.keys())
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (img - self.mean) / self.std
        feats = {}
        for name, block in zip(self.layer_names, self.slices):
            x = block(x)
            feats[name] = x
        return feats


def mean_std(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise spatial mean/std of a (B,C,H,W) feature map."""
    mean = feat.mean(dim=(2, 3))
    std = feat.std(dim=(2, 3)) + 1e-6
    return mean, std


def mean_variance_norm(feat: torch.Tensor) -> torch.Tensor:
    mean, std = mean_std(feat)
    return (feat - mean[:, :, None, None]) / std[:, :, None, None]
