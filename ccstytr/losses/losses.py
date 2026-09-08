from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccstytr.losses.vgg import VGGFeatures, mean_std, mean_variance_norm
from ccstytr.models.structure import SobelEdge, pool_map_to_tokens

_CONTENT_LAYERS = ("relu4_1", "relu5_1")
_STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")


@dataclass
class LossWeights:
    """Weights follow the StyTr2 official train.py where applicable."""

    content: float = 7.0
    style: float = 10.0
    identity1: float = 70.0  # image-level identity MSE
    identity2: float = 1.0  # feature-level identity MSE
    struct: float = 5.0
    alpha: float = 1.0


class LossComputer(nn.Module):
    def __init__(self, weights: LossWeights | None = None, vgg_pretrained: bool = True):
        super().__init__()
        self.w = weights or LossWeights()
        self.vgg = VGGFeatures(pretrained=vgg_pretrained)
        self.edge = SobelEdge()
        self.mse = nn.MSELoss()

    def content_loss(self, out_feats: dict, content_feats: dict) -> torch.Tensor:
        return sum(
            self.mse(
                mean_variance_norm(out_feats[layer]), mean_variance_norm(content_feats[layer])
            )
            for layer in _CONTENT_LAYERS
        )

    def style_loss(self, out_feats: dict, style_feats: dict) -> torch.Tensor:
        loss = out_feats["relu1_1"].new_zeros(())
        for layer in _STYLE_LAYERS:
            om, os = mean_std(out_feats[layer])
            sm, ss = mean_std(style_feats[layer])
            loss = loss + self.mse(om, sm) + self.mse(os, ss)
        return loss

    def struct_loss(self, output: torch.Tensor, content_img: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.edge(output), self.edge(content_img))

    def alpha_consistency_loss(
        self,
        out_f: torch.Tensor,
        c_f: torch.Tensor,
        s_f: torch.Tensor,
        alpha_map: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Constrain region-wise style statistics to interpolate monotonically
        with alpha: ||stat(out)_i - (a_i*stat_s + (1-a_i)*stat_c)||_1 over
        per-token patch statistics of relu3_1 features."""
        h, w = grid_hw
        out_stat = F.adaptive_avg_pool2d(out_f, (h, w)).flatten(2)  # (B,C,N)
        c_stat = F.adaptive_avg_pool2d(c_f, (h, w)).flatten(2)
        s_stat = s_f.mean(dim=(2, 3)).unsqueeze(-1).expand_as(c_stat)
        alpha = pool_map_to_tokens(alpha_map, (h, w)).unsqueeze(1)  # (B,1,N)
        target = alpha * s_stat + (1.0 - alpha) * c_stat
        return F.l1_loss(out_stat, target)

    def forward(
        self,
        model: nn.Module,
        output: torch.Tensor,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        alpha_map: torch.Tensor | None,
        grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        out_feats = self.vgg(output)
        with torch.no_grad():
            content_feats = self.vgg(content_img)
            style_feats = self.vgg(style_img)

        l_c = self.content_loss(out_feats, content_feats)
        l_s = self.style_loss(out_feats, style_feats)

        # Identity: same-image input (c=c, s=s) should reconstruct the input.
        id_cc = model(content_img, content_img)
        id_ss = model(style_img, style_img)
        l_id1 = self.mse(id_cc, content_img) + self.mse(id_ss, style_img)
        id_cc_feats = self.vgg(id_cc)
        id_ss_feats = self.vgg(id_ss)
        l_id2 = sum(
            self.mse(id_cc_feats[layer], content_feats[layer])
            + self.mse(id_ss_feats[layer], style_feats[layer])
            for layer in _CONTENT_LAYERS
        )

        l_struct = self.struct_loss(output, content_img)
        if alpha_map is None:
            # Ablation group 6: no alpha training, loss reported as zero.
            l_alpha = out_feats["relu3_1"].new_zeros(())
        else:
            l_alpha = self.alpha_consistency_loss(
                out_feats["relu3_1"],
                content_feats["relu3_1"],
                style_feats["relu3_1"],
                alpha_map,
                grid_hw,
            )

        total = (
            self.w.content * l_c
            + self.w.style * l_s
            + self.w.identity1 * l_id1
            + self.w.identity2 * l_id2
            + self.w.struct * l_struct
            + self.w.alpha * l_alpha
        )
        return {
            "total": total,
            "content": l_c,
            "style": l_s,
            "identity1": l_id1,
            "identity2": l_id2,
            "struct": l_struct,
            "alpha": l_alpha,
        }
