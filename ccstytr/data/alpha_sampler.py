import torch
import torch.nn.functional as F


def _constant(b: int, h: int, w: int, device) -> torch.Tensor:
    alpha = torch.empty(b, 1, 1, 1, device=device).uniform_(0.3, 1.0)
    return alpha.expand(b, 1, h, w).contiguous()


def _blocks(b: int, h: int, w: int, device) -> torch.Tensor:
    """Random rectangle / ellipse regions with different inside/outside alpha."""
    outside = torch.empty(b, 1, 1, 1, device=device).uniform_(0.0, 1.0)
    inside = torch.empty(b, 1, 1, 1, device=device).uniform_(0.0, 1.0)
    alpha = outside.expand(b, 1, h, w).clone()
    ys = torch.arange(h, device=device).view(1, h, 1).float()
    xs = torch.arange(w, device=device).view(1, 1, w).float()
    for i in range(b):
        cy, cx = torch.rand(2, device=device) * torch.tensor([h, w], device=device).float()
        ry = (0.1 + 0.3 * torch.rand((), device=device)) * h
        rx = (0.1 + 0.3 * torch.rand((), device=device)) * w
        if torch.rand(()) < 0.5:  # rectangle
            mask = ((ys - cy).abs() < ry) & ((xs - cx).abs() < rx)
        else:  # ellipse
            mask = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 < 1.0
        alpha[i, 0][mask[0]] = inside[i, 0, 0, 0]
    return alpha


def _smooth_field(b: int, h: int, w: int, device) -> torch.Tensor:
    """Low-frequency random field: coarse noise upsampled + blurred, min-max normalized."""
    coarse = torch.rand(b, 1, 8, 8, device=device)
    field = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
    kernel = torch.ones(1, 1, 7, 7, device=device) / 49.0
    field = F.conv2d(field, kernel, padding=3)
    fmin = field.amin(dim=(2, 3), keepdim=True)
    fmax = field.amax(dim=(2, 3), keepdim=True)
    return (field - fmin) / (fmax - fmin + 1e-6)


def sample_alpha_maps(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    p_constant: float = 0.4,
    p_blocks: float = 0.3,
    mode: str = "spatial",
) -> torch.Tensor:
    """Training-time alpha-map sampling (plan section 1.1-C):
    constant (p=0.4) / random blocks (p=0.3) / smooth random field (p=0.3).

    mode: "spatial" (default mixture) | "scalar" (always constant, ablation
    group 5) | "no_smooth" (constant+blocks renormalized, ablation group 5b).
    """
    if mode == "scalar":
        return _constant(batch_size, height, width, device)
    if mode == "no_smooth":
        total = p_constant + p_blocks
        p_constant, p_blocks = p_constant / total, p_blocks / total
    elif mode != "spatial":
        raise ValueError(f"unknown alpha sampling mode: {mode}")
    r = torch.rand(()).item()
    if r < p_constant:
        return _constant(batch_size, height, width, device)
    if r < p_constant + p_blocks:
        return _blocks(batch_size, height, width, device)
    return _smooth_field(batch_size, height, width, device)


def auto_alpha_map(
    saliency_map: torch.Tensor, low: float = 0.2, high: float = 0.9
) -> torch.Tensor:
    """Automatic inference mode: derive an alpha-map from a structure-saliency
    map (B,1,H,W) — low intensity on text/logo regions, high elsewhere."""
    return high - (high - low) * saliency_map.clamp(0.0, 1.0)
