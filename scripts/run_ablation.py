"""Sequential ablation runner (experiment protocol, P7).

Runs each variant via `train.py --resume auto`, so the whole campaign is
resumable at any granularity:
  - Ctrl+C / killing the runner: train.py writes latest.pth first (or the
    periodic quick-save covers a hard kill), losing at most
    `quick_save_interval` (500) optimizer steps;
  - re-running this script skips completed variants and resumes the rest.

Usage:
  python scripts/run_ablation.py              # run all pending variants
  python scripts/run_ablation.py --status     # progress table only
  python scripts/run_ablation.py --only saliency_routed_spatial_alpha,residual_gating_spatial_alpha
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_ablation_configs import ORDER, VARIANTS  # noqa: E402

LOG_DIR = ROOT / "logs/ablation"
ENABLED_FILE = ROOT / "configs/ablation/enabled.txt"


def enabled_variants() -> set[str] | None:
    """Subset gate: if enabled.txt exists, only listed variants run.

    Re-read on each iteration to honor the requested configuration subset.
    """
    if not ENABLED_FILE.exists():
        return None
    names = {
        ln.strip()
        for ln in ENABLED_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    unknown = names - set(ORDER)
    if unknown:
        raise ValueError(f"enabled.txt lists unknown variants: {sorted(unknown)}")
    return names


def variant_status(name: str) -> tuple[int, int]:
    """(latest milestone step, max_steps) from disk, without loading weights."""
    cfg = yaml.safe_load(open(ROOT / f"configs/ablation/{name}.yaml", encoding="utf-8"))
    max_steps = int(cfg["train"]["max_steps"])
    ckpt_dir = ROOT / cfg["train"]["ckpt_dir"]
    steps = [
        int(p.stem.rsplit("_", 1)[-1]) for p in ckpt_dir.glob("ccstytr_step_*.pth")
    ] if ckpt_dir.exists() else []
    latest = ckpt_dir / "latest.pth"
    step = max(steps) if steps else 0
    if latest.exists() and not steps:
        step = -1  # in-progress, below first milestone
    return step, max_steps


def print_status(names: list[str]) -> None:
    print(f"{'variant':<22} {'progress':>14}  state")
    for name in names:
        step, max_steps = variant_status(name)
        label = "done" if step >= max_steps else (
            f"{step}/{max_steps}" if step >= 0 else "<10k (latest.pth only)"
        )
        state = "DONE" if step >= max_steps else ("running/pending" if step else "pending")
        print(f"{name:<38} {label:>14}  {state}  {VARIANTS[name]}")


def run_variant(name: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    err_path = LOG_DIR / f"{name}.err"
    cmd = [
        sys.executable, "train.py",
        "--config", f"configs/ablation/{name}.yaml",
        "--resume", "auto",
    ]
    print(f"[run_ablation] start {name} -> {log_path}", flush=True)
    with open(log_path, "ab") as log_f, open(err_path, "ab") as err_f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=err_f)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            print(f"[run_ablation] interrupted, terminating {name}", flush=True)
            proc.terminate()
            proc.wait(timeout=60)
            return 130


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    parser.add_argument("--only", default=None, help="comma-separated variant names")
    args = parser.parse_args()

    names = ORDER
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in VARIANTS]
        if unknown:
            raise SystemExit(f"unknown variants: {unknown}")
        names = wanted

    if args.status:
        print_status(names)
        return

    for name in names:
        enabled = enabled_variants()
        if enabled is not None and name not in enabled:
            print(f"[run_ablation] skip {name} (not in enabled.txt)", flush=True)
            continue
        step, max_steps = variant_status(name)
        if step >= max_steps:
            print(f"[run_ablation] skip {name} (done)", flush=True)
            continue
        code = run_variant(name)
        for attempt in range(3):
            if code in (0, 130):
                break
            # Transient host bad_alloc / pagefile pressure (WinError 1455)
            # usually clears after a cool-down; resume picks up from latest.pth.
            import time
            print(
                f"[run_ablation] {name} exited {code}; "
                f"retry {attempt + 1}/3 in 120s",
                flush=True,
            )
            time.sleep(120)
            code = run_variant(name)
        if code != 0:
            print(
                f"[run_ablation] {name} exited with code {code}; "
                "fix the issue and re-run to resume",
                flush=True,
            )
            sys.exit(code if code != 130 else 0)
        print(f"[run_ablation] finished {name}", flush=True)
    print("[run_ablation] all selected variants complete", flush=True)


if __name__ == "__main__":
    main()
