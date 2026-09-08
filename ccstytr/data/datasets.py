from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def train_transform(load_size: int = 512, crop_size: int = 256) -> transforms.Compose:
    """StyTr2 official transform: resize to 512, random-crop 256."""
    return transforms.Compose(
        [
            transforms.Resize(size=(load_size, load_size)),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ]
    )


def test_transform(size: int = 512) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(size=(size, size)), transforms.ToTensor()])


class FlatFolderDataset(data.Dataset):
    """Recursively loads all images under a root directory (COCO / WikiArt style)."""

    def __init__(self, root: str, transform: transforms.Compose):
        super().__init__()
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images found under {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = Image.open(self.paths[index]).convert("RGB")
        return self.transform(img)


class InfiniteSampler(data.Sampler):
    """Endless random sampler for iteration-based training."""

    def __init__(self, dataset_len: int, seed: int = 0):
        self.dataset_len = dataset_len
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            yield from rng.permutation(self.dataset_len).tolist()
