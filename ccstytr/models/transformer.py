import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional additive logit bias (for SMCA)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        fallback_value: torch.Tensor | None = None,
        fallback_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, nq, d = query.shape
        nk = key.shape[1]
        q = self.q_proj(query).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        if attn_bias is not None:
            attn_bias = attn_bias.to(q.dtype).expand(*attn_bias.shape[:-1], nk).contiguous()
        if fallback_value is not None:
            if fallback_bias is None:
                raise ValueError("fallback_bias is required with fallback_value")
            fallback_v = (
                self.v_proj(fallback_value)
                .view(b, nq, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = torch.cat([k, torch.zeros_like(fallback_v)], dim=2)
            v = torch.cat([v, fallback_v], dim=2)

            style_bias = attn_bias
            if style_bias is None:
                style_bias = q.new_zeros((b, 1, nq, nk))
            diagonal = torch.eye(nq, dtype=torch.bool, device=q.device)[None, None]
            fallback_mask = q.new_full((b, 1, nq, nq), -torch.inf)
            fallback_mask = torch.where(
                diagonal,
                fallback_bias.to(q.dtype)[:, None, :, None],
                fallback_mask,
            )
            attn_bias = torch.cat([style_bias, fallback_mask], dim=-1).contiguous()
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, nq, d)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(dim, num_heads, ffn_dim) for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SMCADecoderLayer(nn.Module):
    """Fusion decoder layer with Structure-Modulated Cross-Attention.

    Each query attends to all style values plus its own content fallback.
    Structure saliency lowers all style logits by |gamma_l| * s_i while the
    content fallback receives logit(s_i), so salient queries route attention
    mass from style toward content. gamma_l is a learnable per-layer scalar.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        gamma_init: float = 0.1,
        mode: str = "smca",
    ):
        super().__init__()
        if mode not in ("smca", "gate"):
            raise ValueError(f"unknown SMCA layer mode: {mode}")
        self.mode = mode
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        style_tokens: torch.Tensor,
        saliency: torch.Tensor | None,
    ) -> torch.Tensor:
        """x: (B,Nc,dim) content queries; style_tokens: (B,Ns,dim);
        saliency: (B,Nc) token-level structure saliency in [0,1] or None."""
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x)))
        if saliency is not None and self.mode == "gate":
            # Post-hoc gating ablation (plan 4.4 group 4): plain cross-attention,
            # then scale the attention contribution per token by (1 - |gamma|*s).
            # Same parameter count as SMCA, but saliency acts after attention
            # instead of competing inside the softmax.
            attn_out = self.cross_attn(self.norm2(x), style_tokens, style_tokens)
            gate = (1.0 - self.gamma.abs() * saliency).clamp(0.0, 1.0)
            x = x + self.dropout(gate.unsqueeze(-1).to(attn_out.dtype) * attn_out)
            x = x + self.dropout(self.ffn(self.norm3(x)))
            return x
        attn_bias = None
        fallback_bias = None
        if saliency is not None:
            attn_bias = -self.gamma.abs() * saliency[:, None, :, None]
            saliency_float = saliency.float()
            clamped_saliency = saliency_float.clamp(1e-4, 1.0 - 1e-4)
            fallback_bias = torch.where(
                saliency_float > 0,
                torch.logit(clamped_saliency),
                torch.full_like(saliency_float, -torch.inf),
            )
        attn_out = self.cross_attn(
            self.norm2(x),
            style_tokens,
            style_tokens,
            attn_bias=attn_bias,
            fallback_value=x if saliency is not None else None,
            fallback_bias=fallback_bias,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class SMCADecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int = 3,
        mode: str = "smca",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            SMCADecoderLayer(dim, num_heads, ffn_dim, mode=mode) for _ in range(num_layers)
        )

    def forward(
        self,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        saliency: torch.Tensor | None,
    ) -> torch.Tensor:
        x = content_tokens
        for layer in self.layers:
            x = layer(x, style_tokens, saliency)
        return x
