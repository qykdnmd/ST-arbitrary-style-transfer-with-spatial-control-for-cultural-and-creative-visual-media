"""20-round content-leakage (structure robustness) analysis, StyTr2 2022
paradigm (experiment protocol): repeatedly re-stylize the output back as the
new content with the same style; record LPIPS(round_n output, original
content) drift per round. SMCA should slow the structural decay.

Saves every round to results/leakage/<method>/<pair>/round_XX.png and writes
results/leakage/leakage.csv (method, pair, round, lpips).

Default probe set: 4 contents x 2 styles from the general eval set, rounds
1..20. All methods see the same 512x512 inputs (outputs are already 512x512,
so each method's own transform is an identity on later rounds).

Usage:
    python scripts/leakage.py --methods ccstytr,stytr2,adain,sanet,cast
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = sys.executable
CHECKPOINT = "checkpoints/ccstytr_step_160000.pth"

ROUNDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)


# ---------- per-method single-shot stylizers (in_path, style_path, out_path) --

def make_ccstytr(checkpoint=CHECKPOINT):
    import torch
    from PIL import Image
    from torchvision.utils import save_image
    from ccstytr.data.datasets import test_transform
    from ccstytr.models.network import CCStyTr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ROOT / checkpoint, map_location=device)
    model_cfg = state.get("model_cfg") if isinstance(state, dict) else None
    model = CCStyTr(**(model_cfg or {})).to(device).eval()
    model.load_state_dict(state["model"] if "model" in state else state)
    tf = test_transform(512)

    def run(inp: Path, style: Path, out: Path):
        c = tf(Image.open(inp).convert("RGB")).unsqueeze(0).to(device)
        s = tf(Image.open(style).convert("RGB")).unsqueeze(0).to(device)
        a = torch.full((1, 1, 512, 512), 1.0, device=device)
        with torch.no_grad():
            o = model(c, s, a).clamp(0, 1)
        save_image(o, out)

    return run


def make_subprocess(cmd_fn, cwd, out_producer):
    """cmd_fn(inp, style, workdir)->cmd list; out_producer(workdir)->produced file."""
    def run(inp: Path, style: Path, out: Path):
        inp, style = Path(inp).resolve(), Path(style).resolve()  # cwd != ROOT
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            subprocess.run(cmd_fn(inp, style, work), cwd=cwd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.copy(out_producer(work), out)
    return run


def stytr2_cmd(inp, style, work):
    return [PY, "test.py", "--content", str(inp), "--style", str(style),
            "--output", str(work)]


def adain_cmd(inp, style, work):
    return [PY, "test.py", "--content", str(inp), "--style", str(style),
            "--content_size", "0", "--style_size", "0", "--output", str(work)]


def sanet_cmd(inp, style, work):
    return [PY, "Eval.py", "--content", str(inp), "--style", str(style),
            "--vgg", "experiments/vgg_normalised.pth",
            "--decoder", "experiments/decoder_iter_500000.pth",
            "--transform", "experiments/transformer_iter_500000.pth",
            "--output", str(work)]


def cast_cmd(inp, style, work):
    data = work / "data"
    (data / "testA").mkdir(parents=True)
    (data / "testB").mkdir(parents=True)
    shutil.copy(inp, data / "testA" / "probe.png")
    shutil.copy(style, data / "testB" / "style.png")
    return [PY, "test.py", "--dataroot", str(data), "--name", "CAST_model",
            "--model", "cast", "--CAST_mode", "CAST", "--phase", "test",
            "--preprocess", "resize", "--load_size", "512", "--crop_size", "512",
            "--num_test", "1"]


def stylized_in(work: Path) -> Path:
    return next(work.glob("*_stylized_*"))


def cast_out(work: Path) -> Path:
    return (ROOT / "external/CAST_pytorch/results/CAST_model/test_latest/images"
            / "probe_fake_B.png")


METHODS = {
    "ccstytr": make_ccstytr,
    "stytr2": lambda: make_subprocess(stytr2_cmd, ROOT / "external/StyTR-2", stylized_in),
    "adain": lambda: make_subprocess(adain_cmd, ROOT / "external/pytorch-AdaIN", stylized_in),
    "sanet": lambda: make_subprocess(sanet_cmd, ROOT / "external/SANET", stylized_in),
    "cast": lambda: make_subprocess(cast_cmd, ROOT / "external/CAST_pytorch", cast_out),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", default="data/eval/general/content")
    ap.add_argument("--style_dir", default="data/eval/general/style")
    ap.add_argument("--out_dir", default="results/leakage")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n_contents", type=int, default=4)
    ap.add_argument("--n_styles", type=int, default=2)
    ap.add_argument("--checkpoint", default=CHECKPOINT,
                    help="ccstytr checkpoint (e.g. an ablation variant's 80k milestone)")
    ap.add_argument("--cc_name", default="ccstytr",
                    help="output-dir/registry name for the ccstytr entry")
    args = ap.parse_args()

    # Use the requested checkpoint and output name for this evaluation.
    METHODS[args.cc_name] = lambda: make_ccstytr(args.checkpoint)

    out_root = Path(args.out_dir)
    styles = sorted(Path(args.style_dir).iterdir())[: args.n_styles]
    # unified protocol: every round sees the SAME 512x512 LANCZOS-preprocessed
    # content (raw COCO sizes crash SANET on odd dims and bypass AdaIN resize)
    prep = out_root / "_probe_inputs"
    prep.mkdir(parents=True, exist_ok=True)
    contents = []
    for c in sorted(Path(args.content_dir).iterdir())[: args.n_contents]:
        dst = prep / f"{c.stem}.png"
        if not dst.exists():
            Image.open(c).convert("RGB").resize((512, 512), Image.LANCZOS).save(dst)
        contents.append(dst)
    csv_path = out_root / "leakage.csv"
    if csv_path.exists():  # fresh run semantics: resume fills images, rewrite csv
        csv_path.unlink()
    out_root.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["method", "pair", "round", "lpips"])

    # metric pass uses the project Evaluator (LPIPS only is enough, but the
    # full tuple is nearly free once VGG is loaded)
    from scripts.evaluate import Evaluator
    ev = Evaluator(size=512)

    rows = []
    for m in args.methods.split(","):
        m = m.strip()
        run = METHODS[m]()  # weights load once per method
        print(f"=== {m} ===", flush=True)
        for c in contents:
            for s in styles:
                pair = f"{c.stem}__{s.stem}"
                pdir = out_root / m / pair
                pdir.mkdir(parents=True, exist_ok=True)
                cur = c
                for n in ROUNDS:
                    dst = pdir / f"round_{n:02d}.png"
                    if not dst.exists():
                        tmp = pdir / "_tmp.png"
                        run(cur, s, tmp)
                        tmp.replace(dst)
                    lp = float(ev.lpips(
                        ev.load(str(dst)) * 2 - 1,
                        ev.load(str(c)) * 2 - 1).mean())
                    rows.append((m, pair, n, round(lp, 4)))
                    cur = dst
                print(f"{m} {pair} done", flush=True)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)
                rows.clear()
    print("done ->", out_root / "leakage.csv")


if __name__ == "__main__":
    main()
