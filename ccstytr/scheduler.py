def stytr2_lr(iteration: int, base_lr: float = 5e-4, warmup_iters: int = 10000) -> float:
    """StyTr2 official learning-rate schedule:
    warmup (first 10k steps): lr * 0.1 * (1 + 3e-4 * it)
    decay afterwards:         2e-4 / (1 + 1e-5 * (it - 1e4))
    """
    if iteration < warmup_iters:
        return base_lr * 0.1 * (1.0 + 3e-4 * iteration)
    return 2e-4 / (1.0 + 1e-5 * (iteration - warmup_iters))
