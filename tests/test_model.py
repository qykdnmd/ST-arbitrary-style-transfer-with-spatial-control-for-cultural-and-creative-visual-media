import torch

from ccstytr.models.decoder_cnn import CNNDecoder
from ccstytr.models.patch_embed import PatchEmbed
from ccstytr.models.transformer import SMCADecoder, TransformerEncoder

DIM, HEADS, FFN, LAYERS = 64, 4, 128, 2


def test_patch_embed_shapes():
    embed = PatchEmbed(dim=DIM, patch_size=8)
    x = torch.rand(2, 3, 64, 64)
    tokens = embed(x)
    assert tokens.shape == (2, 64, DIM)
    assert embed.grid_size(64, 64) == (8, 8)


def test_encoder_decoder_roundtrip():
    embed = PatchEmbed(dim=DIM, patch_size=8)
    enc = TransformerEncoder(DIM, HEADS, FFN, LAYERS)
    dec = SMCADecoder(DIM, HEADS, FFN, LAYERS)
    cnn = CNNDecoder(dim=DIM)
    c = embed(torch.rand(2, 3, 64, 64))
    s = embed(torch.rand(2, 3, 64, 64))
    fused = dec(enc(c), enc(s), saliency=None)
    out = cnn(fused, (8, 8))
    assert out.shape == (2, 3, 64, 64)


def test_ablation_variants_forward():
    """Ablation switches (plan 4.4): gate-mode SMCA, no-decouple, no-SMCA.
    All must preserve the network I/O contract."""
    from ccstytr.models.network import CCStyTr

    tiny = dict(dim=DIM, num_heads=HEADS, ffn_dim=FFN, num_encoder_layers=1,
                num_decoder_layers=1, patch_size=8)
    content = torch.rand(1, 3, 64, 64)
    style = torch.rand(1, 3, 64, 64)
    alpha = torch.rand(1, 1, 64, 64)
    for kwargs in ({"smca_mode": "gate"}, {"use_decouple": False},
                   {"smca_mode": "off"}, {"use_smca": False}):
        model = CCStyTr(**tiny, **kwargs)
        out = model(content, style, alpha)
        assert out.shape == (1, 3, 64, 64)
        assert torch.isfinite(out).all()


def test_alpha_sampler_modes():
    from ccstytr.data.alpha_sampler import sample_alpha_maps

    for mode in ("spatial", "scalar", "no_smooth"):
        for _ in range(20):
            m = sample_alpha_maps(2, 32, 32, mode=mode)
            assert m.shape == (2, 1, 32, 32)
            assert 0.0 <= m.min() and m.max() <= 1.0
            if mode == "scalar":
                # constant across space per sample (batch values may differ)
                assert (m.std(dim=(1, 2, 3)) == 0.0).all()
    try:
        sample_alpha_maps(1, 8, 8, mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode must raise")
