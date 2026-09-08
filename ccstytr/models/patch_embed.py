import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Conv2d patch embedding: 3 -> dim, kernel = stride = patch_size (StyTr2 official)."""

    def __init__(self, dim: int = 512, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, N, dim) with N = (H/p)*(W/p)."""
        feat = self.proj(x)
        return feat.flatten(2).transpose(1, 2)

    def grid_size(self, h: int, w: int) -> tuple[int, int]:
        return h // self.patch_size, w // self.patch_size


class CAPE(nn.Module):
    """Content-aware positional encoding (StyTr2): adaptive-pool content features
    to a fixed grid (18x18 in the official implementation), 1x1 conv, then
    upsample back to the token grid."""

    def __init__(self, dim: int = 512, pool_size: int = 18):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """tokens: (B,N,dim) -> positional encoding (B,N,dim)."""
        b, n, d = tokens.shape
        h, w = grid_hw
        feat = tokens.transpose(1, 2).reshape(b, d, h, w)
        pos = self.conv(self.pool(feat))
        pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        return pos.flatten(2).transpose(1, 2)
