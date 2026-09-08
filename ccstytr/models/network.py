import torch
import torch.nn as nn

from ccstytr.models.decoder_cnn import CNNDecoder
from ccstytr.models.patch_embed import CAPE, PatchEmbed
from ccstytr.models.structure import SobelEdge, pool_map_to_tokens
from ccstytr.models.style_heads import StyleDecouple, adain_tokens
from ccstytr.models.transformer import SMCADecoder, TransformerEncoder


class CCStyTr(nn.Module):
    """Cultural-Creative Style Transformer.

    StyTr2 backbone (3-layer content/style encoders + 3-layer fusion decoder,
    dim=512, 8 heads, FFN 2048, patch=8, CAPE) extended with:
      - SMCA: structure-modulated cross-attention in every fusion decoder layer;
      - spatial alpha-map: token-level style-intensity interpolation over both
        the texture (cross-attention) branch and the color (AdaIN) branch.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        patch_size: int = 8,
        use_smca: bool = True,
        smca_mode: str | None = None,
        use_decouple: bool = True,
    ):
        super().__init__()
        if smca_mode is None:
            smca_mode = "smca" if use_smca else "off"
        if smca_mode not in ("smca", "gate", "off"):
            raise ValueError(f"unknown smca_mode: {smca_mode}")
        self.smca_mode = smca_mode
        self.use_decouple = use_decouple
        self.content_embed = PatchEmbed(dim, patch_size)
        self.style_embed = PatchEmbed(dim, patch_size)
        self.cape = CAPE(dim)
        self.content_encoder = TransformerEncoder(dim, num_heads, ffn_dim, num_encoder_layers)
        self.style_encoder = TransformerEncoder(dim, num_heads, ffn_dim, num_encoder_layers)
        self.style_decouple = StyleDecouple(dim) if use_decouple else None
        decoder_mode = smca_mode if smca_mode in ("smca", "gate") else "smca"
        self.fusion_decoder = SMCADecoder(
            dim, num_heads, ffn_dim, num_decoder_layers, mode=decoder_mode
        )
        self.cnn_decoder = CNNDecoder(dim)
        self.edge = SobelEdge()
        for p in self.edge.parameters():
            p.requires_grad_(False)

    def structure_saliency(self, content_img: torch.Tensor) -> torch.Tensor:
        """Token-level structure saliency s in [0,1], shape (B,N)."""
        grid = self.content_embed.grid_size(*content_img.shape[-2:])
        edge_map = self.edge(content_img)
        return pool_map_to_tokens(edge_map, grid)

    def forward(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        alpha_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """content_img/style_img: (B,3,H,W) in [0,1];
        alpha_map: (B,1,H,W) in [0,1] or None (defaults to all-ones)."""
        grid = self.content_embed.grid_size(*content_img.shape[-2:])

        content_tokens = self.content_embed(content_img)
        content_tokens = content_tokens + self.cape(content_tokens, grid)
        content_tokens = self.content_encoder(content_tokens)

        style_tokens = self.style_encoder(self.style_embed(style_img))
        if self.use_decouple:
            texture_tokens, color_mean, color_std = self.style_decouple(style_tokens)
        else:
            # Ablation (plan 4.4 group 6): single style token stream; color
            # statistics come from token moments instead of learned projections.
            texture_tokens = style_tokens
            color_mean = style_tokens.mean(dim=1, keepdim=True)
            color_std = style_tokens.std(dim=1, keepdim=True) + 1e-6

        saliency = self.structure_saliency(content_img) if self.smca_mode != "off" else None
        fused = self.fusion_decoder(content_tokens, texture_tokens, saliency)

        if alpha_map is None:
            alpha = torch.ones(
                content_tokens.shape[0], content_tokens.shape[1], device=content_img.device
            )
        else:
            alpha = pool_map_to_tokens(alpha_map, grid)
        alpha = alpha.unsqueeze(-1)  # (B,N,1)

        # Texture branch: interpolate fused vs. plain content tokens.
        tokens = alpha * fused + (1.0 - alpha) * content_tokens
        # Color branch: per-token interpolation toward style color statistics.
        colored = adain_tokens(tokens, color_mean, color_std)
        tokens = alpha * colored + (1.0 - alpha) * tokens

        return self.cnn_decoder(tokens, grid)
