from pathlib import Path

LATEST_NAME = "latest.pth"


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    candidates = resume_candidates(ckpt_dir)
    return candidates[0] if candidates else None


def resume_candidates(ckpt_dir: Path) -> list[Path]:
    """Resume candidates, newest first: the frequently-written latest.pth,
    then milestone step checkpoints (descending), then legacy iter files."""
    candidates: list[Path] = []
    latest = ckpt_dir / LATEST_NAME
    if latest.exists():
        candidates.append(latest)
    candidates.extend(
        sorted(
            ckpt_dir.glob("ccstytr_step_*.pth"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            reverse=True,
        )
    )
    candidates.extend(
        sorted(
            ckpt_dir.glob("ccstytr_iter_*.pth"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            reverse=True,
        )
    )
    return candidates


def prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    ckpts = sorted(
        ckpt_dir.glob("ccstytr_step_*.pth"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    for old in ckpts[:-keep]:
        old.unlink()


def checkpoint_position(state: dict, grad_accum: int) -> tuple[int, int, bool]:
    if "optimizer_step" in state:
        optimizer_step = int(state["optimizer_step"])
        micro_iteration = int(state.get("micro_iteration", optimizer_step * grad_accum))
        return optimizer_step, micro_iteration, False

    micro_iteration = int(state.get("iteration", 0))
    return micro_iteration // grad_accum, micro_iteration, True
