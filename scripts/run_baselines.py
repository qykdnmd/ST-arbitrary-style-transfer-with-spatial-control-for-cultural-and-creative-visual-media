"""Unified baseline runner (experiment protocol unified protocol).

All methods receive the SAME preprocessed 512x512 square inputs, so content
geometry is identical across methods; each method's own transform is then an
identity op at this size (verified per repo). Outputs are collected into:

    <out_dir>/<method>/<content_stem>__<style_stem>.png

Methods: ccstytr (CC-StyTr Main), aesfa, stytr2, adain, sanet, cast.

Usage:
    python scripts/run_baselines.py --content_dir data/eval/content --style_dir data/eval/style \
        --out_dir results/main --methods ccstytr,stytr2,adain,sanet,cast
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

DEFAULT_CHECKPOINT = "checkpoints/ccstytr_step_160000.pth"


def prep_inputs(content_dir: Path, style_dir: Path, size: int, work: Path):
    """Square-resize all inputs once; returns (prep_content_dir, prep_style_dir)."""
    c_out, s_out = work / "content", work / "style"
    for src_dir, dst_dir in ((content_dir, c_out), (style_dir, s_out)):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            img.save(dst_dir / f"{p.stem}.png")
    return c_out, s_out


def collect(src_dir: Path, pattern: str, dst_dir: Path, rename=None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.glob(pattern)):
        name = rename(p) if rename else p.name
        shutil.copy(p, dst_dir / name)
        n += 1
    return n


def run_ccstytr(c_dir: Path, s_dir: Path, out: Path, size: int, checkpoint: str):
    # batch driver: loads the model once, writes <c>__<s>.png, skips existing
    subprocess.run(
        [PY, "scripts/infer_batch.py", "--checkpoint", checkpoint,
         "--content_dir", str(c_dir), "--style_dir", str(s_dir),
         "--output_dir", str(out), "--size", str(size)],
        cwd=ROOT, check=True,
    )


def run_aesfa(c_dir: Path, s_dir: Path, out: Path, size: int):
    subprocess.run(
        [PY, "scripts/infer_paired.py", "--method", "aesfa",
         "--pairing", "cartesian", "--content_dir", str(c_dir),
         "--style_dir", str(s_dir), "--output_dir", str(out),
         "--size", str(size), "--aesfa_root", "external/AesFA",
         "--checkpoint", "external/AesFA/ckpt/main/main.pth"],
        cwd=ROOT, check=True,
    )


def run_stytr2(c_dir: Path, s_dir: Path, out: Path, tmp: Path):
    o = tmp / "stytr2"
    subprocess.run(
        [PY, "test.py", "--content_dir", str(c_dir), "--style_dir", str(s_dir), "--output", str(o)],
        cwd=ROOT / "external/StyTR-2", check=True,
    )
    # official names: <c>_stylized_<s>.jpg (save_ext is hard-coded .jpg in test.py)
    collect(o, "*.jpg", out,
            rename=lambda p: p.name.replace("_stylized_", "__"))


def run_adain(c_dir: Path, s_dir: Path, out: Path, tmp: Path):
    o = tmp / "adain"
    # single invocation: test.py already loops the full content x style product
    subprocess.run(
        [PY, "test.py", "--content_dir", str(c_dir), "--style_dir", str(s_dir),
         "--content_size", "0", "--style_size", "0", "--output", str(o)],
        cwd=ROOT / "external/pytorch-AdaIN", check=True,
    )
    collect(o, "*.jpg", out, rename=lambda p: p.name.replace("_stylized_", "__"))


def run_sanet(c_dir: Path, s_dir: Path, out: Path, tmp: Path):
    o = tmp / "sanet"
    exp = "experiments"
    # Eval.py [local batch] dir mode: full cartesian product, weights loaded once
    subprocess.run(
        [PY, "Eval.py", "--content_dir", str(c_dir), "--style_dir", str(s_dir),
         "--vgg", f"{exp}/vgg_normalised.pth",
         "--decoder", f"{exp}/decoder_iter_500000.pth",
         "--transform", f"{exp}/transformer_iter_500000.pth",
         "--output", str(o)],
        cwd=ROOT / "external/SANET", check=True,
    )
    collect(o, "*.jpg", out, rename=lambda p: p.name.replace("_stylized_", "__"))


def run_cast(c_dir: Path, s_dir: Path, out: Path, tmp: Path):
    """CAST uses an unaligned loader (B index = A index % B_size), so a single
    run over all styles would NOT give the cartesian product. Run once per
    style with a singleton testB; collect after each run (same --name reuses
    the results dir)."""
    cast_root = ROOT / "external/CAST_pytorch"
    run_root = cast_root / "results/CAST_model/test_latest"
    imgs = run_root / "images"
    n_contents = len(list(c_dir.iterdir()))
    for s in sorted(s_dir.iterdir()):
        existing = len(list(out.glob(f"*__{s.stem}.png"))) if out.exists() else 0
        if existing >= n_contents:  # resume: style already fully collected
            print(f"cast skip {s.stem} ({existing} existing)", flush=True)
            continue
        if run_root.exists():  # drop stale outputs from earlier/died runs
            shutil.rmtree(run_root)
        data = tmp / f"cast_{s.stem}"
        (data / "testA").mkdir(parents=True)
        (data / "testB").mkdir(parents=True)
        for c in c_dir.iterdir():
            shutil.copy(c, data / "testA" / c.name)
        shutil.copy(s, data / "testB" / s.name)
        subprocess.run(
            [PY, "test.py", "--dataroot", str(data), "--name", "CAST_model",
             "--model", "cast", "--CAST_mode", "CAST", "--phase", "test",
             "--preprocess", "resize", "--load_size", "512", "--crop_size", "512",
             "--num_test", "100000"],
            cwd=cast_root, check=True,
        )
        collect(imgs, "*_fake_B.png", out,
                rename=lambda p, sn=s.stem: p.name.replace("_fake_B", f"__{sn}"))


METHODS = {
    "ccstytr": run_ccstytr,
    "aesfa": run_aesfa,
    "stytr2": run_stytr2,
    "adain": run_adain,
    "sanet": run_sanet,
    "cast": run_cast,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument(
        "--cc_checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="explicit CC-StyTr checkpoint; recorded protocol must identify the selected model",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        c_dir, s_dir = prep_inputs(Path(args.content_dir), Path(args.style_dir), args.size, work)
        for m in args.methods.split(","):
            m = m.strip()
            print(f"=== {m} ===", flush=True)
            run_dir = work / f"tmp_{m}"
            run_dir.mkdir(exist_ok=True)
            fn = METHODS[m]
            if m == "ccstytr":
                fn(c_dir, s_dir, out_root / m, args.size, args.cc_checkpoint)
            elif m == "aesfa":
                fn(c_dir, s_dir, out_root / m, args.size)
            else:
                fn(c_dir, s_dir, out_root / m, run_dir)
    print("done ->", out_root)


if __name__ == "__main__":
    main()
