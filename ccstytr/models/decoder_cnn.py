import torch
import torch.nn as nn


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_ch, out_ch, kernel_size=3),
        nn.ReLU(inplace=True),
    )


class CNNDecoder(nn.Module):
    """3-stage upsampling CNN decoder (StyTr2-style): 512 -> 256 -> 128 -> 64 -> 3,
    2x upsampling per stage; patch=8 token grid is restored to full resolution."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.blocks = nn.Sequential(
            _up_block(dim, 256),
            _up_block(256, 128),
            _up_block(128, 64),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3),
        )

    def forward(self, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """tokens (B,N,dim) on grid (h,w) -> image (B,3,h*8,w*8)."""
        b, n, d = tokens.shape
        h, w = grid_hw
        feat = tokens.transpose(1, 2).reshape(b, d, h, w)
        return self.blocks(feat)
