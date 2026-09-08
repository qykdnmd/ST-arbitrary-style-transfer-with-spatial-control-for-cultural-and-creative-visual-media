"""Ablation evaluation orchestrator (experiment protocol).

Thin driver over the existing, already-validated evaluation stack:
  1. infer   -> scripts/infer_batch.py per variant into results/ablation/<variant>/
                (featured protocol: 120 poster_curated x 30 style_flat)
  2. metrics -> scripts/eval_main.py on results/ablation (five metrics, 95% CI,
                Wilcoxon signed-rank vs --ref, default saliency_routed_spatial_alpha)
  3. ocr     -> scripts/ocr_consistency.py (RapidOCR char-level similarity)
  4. table   -> render results/ablation/table_ablation.md

Every stage is resumable: infer_batch skips existing outputs; metrics/ocr/table
recompute from whatever images are present. Run with --steps to select stages.

Usage:
  python scripts/eval_ablation.py --only saliency_routed_spatial_alpha,residual_gating_spatial_alpha --steps infer
  python scripts/eval_ablation.py --steps metrics,ocr,table
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from model_names import MODELS

ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "results/ablation"
CONTENT_DIR = ROOT / "data/eval/poster_curated"
STYLE_DIR = ROOT / "data/eval/featured/style_flat"
EXPECTED_PAIRS = 3600  # 120 posters x 30 styles

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def milestone_checkpoint(variant: str) -> Path:
    cfg = yaml.safe_load(open(ROOT / f"configs/ablation/{variant}.yaml", encoding="utf-8"))
    ckpt_dir = ROOT / cfg["train"]["ckpt_dir"]
    steps = sorted(ckpt_dir.glob("ccstytr_step_*.pth"))
    if steps:
        return steps[-1]
    latest = ckpt_dir / "latest.pth"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"no checkpoint for {variant} in {ckpt_dir}")


def count_outputs(variant: str) -> int:
    out = ABLATION_DIR / variant
    if not out.exists():
        return 0
    return sum(1 for p in out.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run(cmd: list[str], log_name: str) -> None:
    log_dir = ROOT / "logs/ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    print("[eval_ablation]", " ".join(cmd), flush=True)
    with open(log_dir / log_name, "ab") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"stage failed ({code}): {' '.join(cmd)}; see logs/ablation/{log_name}")


def stage_infer(variants: list[str]) -> None:
    for v in variants:
        n = count_outputs(v)
        if n >= EXPECTED_PAIRS:
            print(f"[eval_ablation] skip infer {v} ({n} outputs present)", flush=True)
            continue
        ckpt = milestone_checkpoint(v)
        run(
            [
                sys.executable, "scripts/infer_batch.py",
                "--checkpoint", str(ckpt),
                "--content_dir", str(CONTENT_DIR),
                "--style_dir", str(STYLE_DIR),
                "--output_dir", str(ABLATION_DIR / v),
            ],
            f"eval_{v}_infer.log",
        )


def stage_metrics(ref: str) -> None:
    run(
        [
            sys.executable, "scripts/eval_main.py",
            "--results_dir", str(ABLATION_DIR),
            "--content_dir", str(CONTENT_DIR),
            "--style_dir", str(STYLE_DIR),
            "--ref", ref,
        ],
        "eval_metrics.log",
    )


def stage_ocr(variants: list[str]) -> None:
    run(
        [
            sys.executable, "scripts/ocr_consistency.py",
            "--content_dir", str(CONTENT_DIR),
            "--style_dir", str(STYLE_DIR),
            "--out_dir", str(ABLATION_DIR),
            "--results", str(ABLATION_DIR / "ocr.csv"),
            "--methods", ",".join(variants),
            "--workers", "6",
        ],
        "eval_ocr.log",
    )


def stage_table(ref: str, variants: list[str]) -> None:
    """Render table_ablation.md from summary.csv (+ ocr.csv if present)."""
    summary_csv = ABLATION_DIR / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError("run metrics stage first")
    with open(summary_csv, encoding="utf-8") as f:
        rows = {r["method"]: r for r in csv.DictReader(f)}

    ocr_mean: dict[str, tuple[float, int]] = {}
    ocr_csv = ABLATION_DIR / "ocr.csv"
    if ocr_csv.exists():
        vals: dict[str, list[float]] = {}
        with open(ocr_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("ocr_sim"):
                    vals.setdefault(r["method"], []).append(float(r["ocr_sim"]))
        ocr_mean = {m: (float(np.mean(v)), len(v)) for m, v in vals.items()}

    order = [ref] + [v for v in variants if v != ref]
    lines = [
        "# Ablation on featured set (120 posters x 30 museum styles, 80k-step equal budget)",
        "",
        f"Wilcoxon signed-rank p-values vs {MODELS[ref]} (paired per content x style cell).",
        "",
        "| Variant | SSIM↑ | LPIPS↓ | E-SSIM↑ | L_c↓ | L_s↓ | OCR-sim↑ |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in order:
        if m not in rows:
            lines.append(f"| {m} | — | — | — | — | — | — |")
            continue
        r = rows[m]
        cells = []
        for k in ["ssim", "lpips", "e_ssim", "l_c", "l_s"]:
            mean, ci = float(r[f"{k}_mean"]), float(r[f"{k}_ci95"])
            p = r.get(f"wilcoxon_p_{k}", "")
            star = ""
            if p and m != ref:
                star = "***" if float(p) < 1e-3 else ("**" if float(p) < 1e-2 else ("*" if float(p) < 0.05 else ""))
            cells.append(f"{mean:.4f} ± {ci:.4f}{star}")
        if m in ocr_mean:
            mean, n = ocr_mean[m]
            cells.append(f"{mean:.4f} (n={n})")
        else:
            cells.append("—")
        lines.append(f"| {MODELS[m]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Significance vs {MODELS[ref]}: *** p<1e-3, ** p<1e-2, * p<0.05.",
        "Residual gating modulates the cross-attention residual; saliency-routed attention "
        "modulates attention competition. Scalar alpha is spatially uniform.",
    ]
    out = ABLATION_DIR / "table_ablation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved:", out, flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated variants (default: enabled.txt)")
    ap.add_argument("--steps", default="infer,metrics,ocr,table")
    ap.add_argument("--ref", default="saliency_routed_spatial_alpha", help="Wilcoxon reference variant")
    args = ap.parse_args()
    if args.ref not in MODELS:ap.error("Unknown reference configuration")

    if args.only:
        variants = [n.strip() for n in args.only.split(",") if n.strip()]
    else:
        enabled = ROOT / "configs/ablation/enabled.txt"
        variants = [
            ln.strip() for ln in enabled.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    if not variants or any(v not in MODELS for v in variants):ap.error("Unknown configuration")
    if "infer" in steps:
        stage_infer(variants)
    if "metrics" in steps:
        stage_metrics(args.ref)
    if "ocr" in steps:
        stage_ocr(variants)
    if "table" in steps:
        stage_table(args.ref, variants)


if __name__ == "__main__":
    main()
