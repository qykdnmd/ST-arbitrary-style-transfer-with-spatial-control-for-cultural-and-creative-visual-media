import torch
from PIL import Image

from ccstytr.checkpoints import checkpoint_position, latest_checkpoint, prune_checkpoints
from ccstytr.data.datasets import FlatFolderDataset, train_transform
from ccstytr.scheduler import stytr2_lr


def test_lr_schedule():
    assert stytr2_lr(0) == 5e-4 * 0.1
    assert stytr2_lr(10000) == 2e-4
    assert stytr2_lr(160000) < 2e-4


def test_checkpoint_position_migrates_legacy_micro_iterations():
    step, micro_iteration, legacy = checkpoint_position({"iteration": 160000}, grad_accum=8)
    assert (step, micro_iteration, legacy) == (20000, 160000, True)


def test_checkpoint_position_restores_optimizer_steps():
    state = {"optimizer_step": 30000, "micro_iteration": 240000}
    step, micro_iteration, legacy = checkpoint_position(state, grad_accum=8)
    assert (step, micro_iteration, legacy) == (30000, 240000, False)


def test_latest_checkpoint_prefers_new_format(tmp_path):
    legacy = tmp_path / "ccstytr_iter_160000.pth"
    current = tmp_path / "ccstytr_step_030000.pth"
    legacy.touch()
    current.touch()
    assert latest_checkpoint(tmp_path) == current


def test_prune_checkpoints_sorts_numeric_steps(tmp_path):
    for step in (990000, 1000000, 1010000):
        (tmp_path / f"ccstytr_step_{step:06d}.pth").touch()
    prune_checkpoints(tmp_path, keep=2)
    remaining = sorted(path.name for path in tmp_path.glob("ccstytr_step_*.pth"))
    assert remaining == ["ccstytr_step_1000000.pth", "ccstytr_step_1010000.pth"]


def test_flat_folder_dataset(tmp_path):
    for name in ("a.jpg", "b.png"):
        Image.new("RGB", (300, 300), (128, 64, 32)).save(tmp_path / name)
    ds = FlatFolderDataset(str(tmp_path), train_transform(64, 32))
    assert len(ds) == 2
    img = ds[0]
    assert isinstance(img, torch.Tensor) and img.shape == (3, 32, 32)
