import torch

from ccstytr.data.alpha_sampler import auto_alpha_map, sample_alpha_maps
from ccstytr.models.network import CCStyTr

TINY = dict(dim=64, num_heads=4, ffn_dim=128, num_encoder_layers=2, num_decoder_layers=2)


def test_forward_shapes():
    model = CCStyTr(**TINY)
    out = model(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 3, 64, 64)


def test_forward_with_alpha_map():
    model = CCStyTr(**TINY)
    alpha = sample_alpha_maps(1, 64, 64)
    assert alpha.shape == (1, 1, 64, 64)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    out = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), alpha)
    assert out.shape == (1, 3, 64, 64)


def test_alpha_zero_close_to_identity_branch():
    """alpha=0 must bypass both the texture and the color branch."""
    model = CCStyTr(**TINY).eval()
    c = torch.rand(1, 3, 64, 64)
    zeros = torch.zeros(1, 1, 64, 64)
    with torch.no_grad():
        o1 = model(c, torch.rand(1, 3, 64, 64), zeros)
        o2 = model(c, torch.rand(1, 3, 64, 64), zeros)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_smca_disabled_variant():
    model = CCStyTr(**TINY, use_smca=False)
    out = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    assert out.shape == (1, 3, 64, 64)


def test_structure_saliency_range():
    model = CCStyTr(**TINY)
    sal = model.structure_saliency(torch.rand(2, 3, 64, 64))
    assert sal.shape == (2, 64)  # 8x8 grid with patch=8
    assert sal.min() >= 0.0 and sal.max() <= 1.0


def test_auto_alpha_map():
    sal = torch.rand(1, 1, 64, 64)
    amap = auto_alpha_map(sal, low=0.2, high=0.9)
    assert amap.min() >= 0.2 - 1e-6 and amap.max() <= 0.9 + 1e-6
