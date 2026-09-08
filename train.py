"""CC-StyTr training entry point.

Usage:
    python train.py --config configs/base.yaml
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.data as data
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from ccstytr.checkpoints import (
    LATEST_NAME,
    checkpoint_position,
    prune_checkpoints,
    resume_candidates,
)
from ccstytr.data import FlatFolderDataset, InfiniteSampler, sample_alpha_maps, train_transform
from ccstytr.losses import LossComputer, LossWeights
from ccstytr.models.network import CCStyTr
from ccstytr.scheduler import stytr2_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint path to resume from, or 'auto' to pick the latest in ckpt_dir",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def build_loader(root: str, cfg: dict) -> iter:
    dataset = FlatFolderDataset(root, train_transform(cfg["load_size"], cfg["crop_size"]))
    num_workers = cfg["num_workers"]
    loader = data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=InfiniteSampler(len(dataset)),
        num_workers=num_workers,
        # Pinned buffers are non-pageable; on this machine the small pagefile
        # makes them a commit-charge liability (host std::bad_alloc crashes).
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return iter(loader)


def main() -> None:
    if sys.platform != "win32":  # expandable_segments is not supported on Windows
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CCStyTr(**cfg["model"]).to(device)
    criterion = LossComputer(LossWeights(**cfg["loss"])).to(device)
    tcfg = cfg["train"]
    if "max_steps" not in tcfg:
        raise KeyError(
            "train.max_steps is required and counts optimizer updates; "
            "replace the legacy train.max_iter setting"
        )
    max_steps = tcfg["max_steps"]
    grad_accum = tcfg["grad_accum"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    optimizer_step = 0
    micro_iteration = 0
    resume_state = None
    resume_path = args.resume
    if resume_path == "auto":
        # Try candidates newest-first; a checkpoint interrupted mid-write is
        # skipped rather than aborting the resume.
        for candidate in resume_candidates(Path(tcfg["ckpt_dir"])):
            try:
                torch.load(candidate, map_location="cpu", weights_only=True)
            except Exception as exc:  # noqa: BLE001 - corrupt file, try older one
                log(f"skipping unreadable checkpoint {candidate}: {exc}")
                continue
            resume_path = str(candidate)
            break
        if resume_path == "auto":
            resume_path = None
            log("no checkpoint found for --resume auto, training from scratch")
    if resume_path:
        state = torch.load(resume_path, map_location=device)
        if state.get("optimizer_step", 1) == 0 and state.get("micro_iteration", 1) == 0:
            # Step-0 checkpoint left behind by a crash before the first update;
            # carrying it forward would needlessly re-save 400MB on interrupt.
            log(f"ignoring step-0 checkpoint {resume_path}, training from scratch")
            resume_path = None
            state = None
    if resume_path:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_step, micro_iteration, legacy = checkpoint_position(state, grad_accum)
        resume_state = state
        if legacy:
            log(
                f"migrated legacy checkpoint {resume_path}: "
                f"{micro_iteration} micro-iterations -> {optimizer_step} optimizer steps"
            )
        else:
            log(f"resumed from {resume_path} at optimizer step {optimizer_step}")
        # Restore RNG so post-resume sampling (alpha maps, dropout) continues
        # the pre-interruption stream; dataloader worker order is not restored.
        if "rng_torch" in state:
            torch.set_rng_state(state["rng_torch"].cpu())
        if device.type == "cuda" and "rng_cuda" in state:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        if "rng_python" in state:
            random.setstate(state["rng_python"])

    loader_cfg = {**cfg["data"], "batch_size": tcfg["batch_size"]}
    content_iter = build_loader(cfg["data"]["content_dir"], loader_cfg)
    style_iter = build_loader(cfg["data"]["style_dir"], loader_cfg)

    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    use_amp = tcfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    if resume_state and "scaler" in resume_state:
        scaler.load_state_dict(resume_state["scaler"])

    ckpt_dir = Path(tcfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tcfg["log_dir"])

    keep_ckpts = tcfg.get("keep_ckpts", 5)
    log_interval = tcfg.get("log_interval", 100)
    grad_clip = tcfg.get("grad_clip", 1.0)
    quick_save_interval = tcfg.get("quick_save_interval", 500)
    alpha_mode = tcfg.get("alpha_mode", "spatial")

    def save_checkpoint(path: Path) -> None:
        checkpoint = {
            "format_version": 2,
            # Architecture flags travel with the weights so inference scripts
            # reconstruct the right variant (ablation checkpoints differ).
            "model_cfg": cfg["model"],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "optimizer_step": optimizer_step,
            "micro_iteration": micro_iteration,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "rng_python": random.getstate(),
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)

    if optimizer_step >= max_steps:
        log(f"already at step {optimizer_step} >= max_steps {max_steps}, nothing to do")
        return

    model.train()
    progress = tqdm(total=max_steps, initial=optimizer_step, disable=None)
    interrupted = False
    try:
      while optimizer_step < max_steps:
        lr = stytr2_lr(optimizer_step, tcfg["lr"], tcfg["warmup_iters"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sums: dict[str, float] = {}
        valid_update = True
        for _ in range(grad_accum):
            content = next(content_iter).to(device, non_blocking=True)
            style = next(style_iter).to(device, non_blocking=True)
            micro_iteration += 1
            b, _, h, w = content.shape
            if alpha_mode == "none":
                alpha_map = None
            else:
                alpha_map = sample_alpha_maps(b, h, w, device=device, mode=alpha_mode)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(content, style, alpha_map)
                grid = model.content_embed.grid_size(h, w)
                losses = criterion(model, output, content, style, alpha_map, grid)
                loss = losses["total"] / grad_accum

            if not torch.isfinite(losses["total"]):
                log(
                    f"step {optimizer_step}: non-finite loss at "
                    f"micro-iteration {micro_iteration}, retrying update"
                )
                valid_update = False
                break

            scaler.scale(loss).backward()
            for name, value in losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + value.item()

        if not valid_update:
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            log(f"step {optimizer_step}: non-finite gradients, retrying update")
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.step(optimizer)
        scaler.update()
        optimizer_step += 1
        progress.update(1)
        averaged_losses = {name: value / grad_accum for name, value in loss_sums.items()}

        if optimizer_step % log_interval == 0:
            for name, value in averaged_losses.items():
                writer.add_scalar(f"loss/{name}", value, optimizer_step)
            writer.add_scalar("lr", lr, optimizer_step)
            writer.add_scalar("micro_iteration", micro_iteration, optimizer_step)
            log(
                f"step {optimizer_step:6d}/{max_steps} | micro {micro_iteration:7d} | "
                f"lr {lr:.2e} | "
                + " | ".join(f"{name} {value:.4f}" for name, value in averaged_losses.items())
            )

        if optimizer_step % tcfg["sample_interval"] == 0:
            with torch.no_grad():
                grid_imgs = torch.cat([content[:4], style[:4], output[:4].clamp(0, 1)])
            save_image(grid_imgs, ckpt_dir / f"sample_step_{optimizer_step:06d}.png", nrow=4)

        if optimizer_step % tcfg["save_interval"] == 0:
            save_checkpoint(ckpt_dir / f"ccstytr_step_{optimizer_step:06d}.pth")
            prune_checkpoints(ckpt_dir, keep_ckpts)

        if quick_save_interval > 0 and optimizer_step % quick_save_interval == 0:
            # Frequently refreshed resume point: a hard kill loses at most
            # quick_save_interval optimizer steps.
            save_checkpoint(ckpt_dir / LATEST_NAME)

    except KeyboardInterrupt:
        interrupted = True
        log(f"interrupted at step {optimizer_step}, saving resume checkpoint")
    finally:
        if interrupted or optimizer_step < max_steps:
            save_checkpoint(ckpt_dir / LATEST_NAME)
            log(f"resume checkpoint written to {ckpt_dir / LATEST_NAME}")
        progress.close()
        writer.close()


if __name__ == "__main__":
    main()
