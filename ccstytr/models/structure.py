import torch
import torch.nn as nn
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.t()


class SobelEdge(nn.Module):
    """Differentiable Sobel edge extractor, output normalized to [0,1].

    Serves both as the structure-saliency source for SMCA and as the frozen
    edge extractor E for L_struct. A learned HED can be swapped in later via
    the same interface.
    """

    def __init__(self):
        super().__init__()
        kernel = torch.stack([_SOBEL_X, _SOBEL_Y]).unsqueeze(1)  # (2,1,3,3)
        self.register_buffer("kernel", kernel)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [0,1] -> edge map (B,1,H,W) in [0,1]."""
        gray = img.mean(dim=1, keepdim=True)
        grad = F.conv2d(gray, self.kernel, padding=1)
        # eps inside sqrt keeps the gradient finite on flat regions (d/dx sqrt(x) -> inf at 0)
        mag = (grad.pow(2).sum(dim=1, keepdim=True) + 1e-12).sqrt()
        max_val = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return mag / max_val


def pool_map_to_tokens(spatial_map: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Pool a (B,1,H,W) map to token-level values (B,N) on the patch grid."""
    pooled = F.adaptive_avg_pool2d(spatial_map, grid_hw)
    return pooled.flatten(1)
