"""Train a WikiArt movement classifier for the Deception rate metric.

Data: data/train/wikiart/<Movement>/*.jpg (27 movements, 81444 images).
The 20 general-eval style images are EXCLUDED from train/val (their filenames
are listed in data/eval/general/manifest.csv, split==style).

Model: ResNet-50 (torchvision, ImageNet init), AMP, stratified 95/5 split.
Output: checkpoints/wikiart_movement_r50.pth  (+ training log printed)
"""

import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK console

import os
# download.pytorch.org is mangled by the local proxy (hash mismatch twice);
# pretrained weights come from HF hub via hf-mirror instead.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
WIKIART = ROOT / "data/train/wikiart"
EXCLUDE_MANIFEST = ROOT / "data/eval/general/manifest.csv"
CKPT = ROOT / "checkpoints/wikiart_movement_r50.pth"


def load_eval_style_filenames() -> set:
    names = set()
    with open(EXCLUDE_MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "style":
                # manifest file column is "<Movement>__<filename>"
                names.add(r["file"].split("__", 1)[1])
    return names


class WikiArt(Dataset):
    def __init__(self, items, tf):
        self.items = items  # list of (path, label)
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y


def build_items(exclude: set):
    classes = sorted(p.name for p in WIKIART.iterdir() if p.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    items = []
    skipped = 0
    for c in classes:
        for p in (WIKIART / c).iterdir():
            if p.name in exclude:
                skipped += 1
                continue
            items.append((p, cls_to_idx[c]))
    return classes, items, skipped


def stratified_split(items, val_frac=0.05, seed=2026):
    rng = random.Random(seed)
    by_label = {}
    for it in items:
        by_label.setdefault(it[1], []).append(it)
    train, val = [], []
    for y, lst in by_label.items():
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * val_frac))
        val += lst[:n_val]
        train += lst[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    per_cls_c, per_cls_t = Counter(), Counter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast("cuda"):
                pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
            for t, p in zip(y.tolist(), pred.tolist()):
                per_cls_t[t] += 1
                per_cls_c[t] += (t == p)
    return correct / total, per_cls_c, per_cls_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = "cuda"
    exclude = load_eval_style_filenames()
    classes, items, skipped = build_items(exclude)
    print(f"{len(items)} images, {len(classes)} movements, excluded {skipped} eval styles")
    train_items, val_items = stratified_split(items)
    print(f"train {len(train_items)} / val {len(val_items)}")

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # pin_memory=False: pin thread hit 'CUDA error: resource already mapped'
    # on this box (Windows + persistent workers); non-pinned is fine at 224px
    train_ld = DataLoader(WikiArt(train_items, tf_train), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers,
                          persistent_workers=True, drop_last=True)
    val_ld = DataLoader(WikiArt(val_items, tf_val), batch_size=args.batch,
                        shuffle=False, num_workers=2)

    # timm ResNet-50 (a1_in1k recipe); timm ResNet still exposes `.fc`
    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=len(classes))
    model.to(device)

    # class-weighted CE (27 classes, 98..13060 images per class)
    cnt = Counter(y for _, y in train_items)
    w = torch.tensor([len(train_items) / (len(classes) * cnt[i]) for i in range(len(classes))],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")],
         "lr": args.backbone_lr},
        {"params": model.fc.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_batches = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
            n_batches += 1
        sched.step()
        acc, pc_c, pc_t = accuracy(model, val_ld, device)
        print(f"epoch {ep}/{args.epochs}  loss {loss_sum/n_batches:.4f}  "
              f"val_top1 {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "val_top1": acc}, CKPT)
    print(f"best val_top1 {best:.4f} -> {CKPT}")

    # per-class report for the best checkpoint
    sd = torch.load(CKPT, weights_only=False)
    model.load_state_dict(sd["state_dict"])
    acc, pc_c, pc_t = accuracy(model, val_ld, device)
    print(f"reloaded best: val_top1 {acc:.4f}")
    for i, c in enumerate(classes):
        if pc_t[i]:
            print(f"  {c:<28} {pc_c[i]/pc_t[i]:.3f} (n={pc_t[i]})")


if __name__ == "__main__":
    main()
