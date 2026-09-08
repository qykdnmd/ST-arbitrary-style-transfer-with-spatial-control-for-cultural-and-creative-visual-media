import torch

from ccstytr.models.patch_embed import PatchEmbed
from ccstytr.models.structure import SobelEdge, pool_map_to_tokens
from ccstytr.models.transformer import SMCADecoder, SMCADecoderLayer, TransformerEncoder

DIM, HEADS, FFN, LAYERS = 64, 4, 128, 2


def test_sobel_edge_range():
    edge = SobelEdge()
    out = edge(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_pool_map_to_tokens():
    tokens = pool_map_to_tokens(torch.rand(2, 1, 64, 64), (8, 8))
    assert tokens.shape == (2, 64)
    assert tokens.min() >= 0.0 and tokens.max() <= 1.0


def test_smca_saliency_changes_output():
    embed = PatchEmbed(dim=DIM, patch_size=8)
    enc = TransformerEncoder(DIM, HEADS, FFN, LAYERS)
    dec = SMCADecoder(DIM, HEADS, FFN, LAYERS).eval()
    c = enc(embed(torch.rand(1, 3, 64, 64)))
    s = enc(embed(torch.rand(1, 3, 64, 64)))
    with torch.no_grad():
        base = dec(c, s, saliency=None)
        biased = dec(c, s, saliency=torch.ones(1, 64))
    assert base.shape == biased.shape == (1, 64, DIM)
    assert not torch.allclose(base, biased)


def test_smca_zero_saliency_matches_unmodulated_attention():
    layer = SMCADecoderLayer(DIM, HEADS, FFN).eval()
    content = torch.rand(1, 8, DIM)
    style = torch.rand(1, 8, DIM)
    with torch.no_grad():
        unmodulated = layer(content, style, saliency=None)
        zero_saliency = layer(content, style, saliency=torch.zeros(1, 8))
    assert torch.allclose(unmodulated, zero_saliency, atol=1e-5)


def test_smca_trains_gamma():
    layer = SMCADecoderLayer(DIM, HEADS, FFN)
    content = torch.rand(1, 8, DIM)
    style = torch.rand(1, 8, DIM)
    output = layer(content, style, saliency=torch.full((1, 8), 0.5))
    output.square().mean().backward()
    assert layer.gamma.grad is not None
    assert torch.isfinite(layer.gamma.grad)
    assert layer.gamma.grad.abs() > 0


def test_smca_bfloat16_saliency_stays_finite():
    layer = SMCADecoderLayer(DIM, HEADS, FFN)
    content = torch.rand(1, 8, DIM)
    style = torch.rand(1, 8, DIM)
    saliency = torch.ones(1, 8, dtype=torch.bfloat16)
    output = layer(content, style, saliency)
    assert torch.isfinite(output).all()
