import torch
import torch.nn as nn


class StyleDecouple(nn.Module):
    """Decouple style encoder output into a color-statistics branch (AdaIN-style
    per-channel mean/std) and a texture/brushstroke token branch (for
    cross-attention)."""

    def __init__(self, dim: int):
        super().__init__()
        self.texture_proj = nn.Linear(dim, dim)
        self.color_proj = nn.Linear(dim, dim * 2)

    def forward(
        self, style_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """style_tokens: (B,Ns,dim) -> (texture_tokens (B,Ns,dim),
        color_mean (B,1,dim), color_std (B,1,dim))."""
        texture = self.texture_proj(style_tokens)
        pooled = style_tokens.mean(dim=1)  # (B,dim)
        stats = self.color_proj(pooled)  # (B,2*dim)
        mean, std = stats.chunk(2, dim=-1)
        return texture, mean.unsqueeze(1), nn.functional.softplus(std).unsqueeze(1)


def adain_tokens(
    content: torch.Tensor, style_mean: torch.Tensor, style_std: torch.Tensor
) -> torch.Tensor:
    """AdaIN over token sequences: normalize content channel statistics across
    tokens and re-target them to the style statistics."""
    c_mean = content.mean(dim=1, keepdim=True)
    c_std = content.std(dim=1, keepdim=True) + 1e-6
    return style_std * (content - c_mean) / c_std + style_mean
