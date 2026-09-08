import torch

from ccstytr.losses import LossComputer, LossWeights
from ccstytr.models.network import CCStyTr

TINY = dict(dim=64, num_heads=4, ffn_dim=128, num_encoder_layers=2, num_decoder_layers=2)


def test_loss_weights_defaults():
    w = LossWeights()
    assert (w.content, w.style, w.identity1, w.identity2, w.struct, w.alpha) == (
        7.0,
        10.0,
        70.0,
        1.0,
        5.0,
        1.0,
    )


def test_losses_backward():
    model = CCStyTr(**TINY)
    criterion = LossComputer(LossWeights(), vgg_pretrained=False)
    c = torch.rand(1, 3, 64, 64)
    s = torch.rand(1, 3, 64, 64)
    alpha = torch.full((1, 1, 64, 64), 0.7)
    out = model(c, s, alpha)
    losses = criterion(model, out, c, s, alpha, model.content_embed.grid_size(64, 64))
    for key in ("total", "content", "style", "identity1", "identity2", "struct", "alpha"):
        assert key in losses
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
