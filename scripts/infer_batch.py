"""Batch CC-StyTr inference: full content x style cartesian product with the
model loaded once (used by scripts/run_baselines.py for the main experiment).

Outputs: <output_dir>/<content_stem>__<style_stem>.png
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ccstytr.data.datasets import test_transform  # noqa: E402
from ccstytr.models.network import CCStyTr  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    # Checkpoint architecture flags determine the network configuration.
    # Files without architecture metadata use the network constructor defaults.
    model_cfg = state.get("model_cfg") if isinstance(state, dict) else None
    model = CCStyTr(**(model_cfg or {})).to(device).eval()
    model.load_state_dict(state["model"] if "model" in state else state)

    tf = test_transform(args.size)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contents = sorted(p for p in Path(args.content_dir).iterdir()
                      if p.suffix.lower() in IMAGE_EXTS)
    styles = sorted(p for p in Path(args.style_dir).iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    with torch.no_grad():
        for i, c in enumerate(contents):
            content = tf(Image.open(c).convert("RGB")).unsqueeze(0).to(device)
            for s in styles:
                dst = out_dir / f"{c.stem}__{s.stem}.png"
                if dst.exists():
                    continue
                style = tf(Image.open(s).convert("RGB")).unsqueeze(0).to(device)
                alpha = torch.full((1, 1, args.size, args.size), args.alpha, device=device)
                output = model(content, style, alpha).clamp(0, 1)
                save_image(output, dst)
            print(f"[{i + 1}/{len(contents)}] {c.name}", flush=True)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
