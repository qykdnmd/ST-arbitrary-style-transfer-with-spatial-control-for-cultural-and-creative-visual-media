"""CC-StyTr evaluation metrics (experiment protocol).

Metrics per (output, content, style) triple, all at a fixed size (512x512
by default, per the unified evaluation protocol):
  - SSIM(output, content)          structure preservation
  - LPIPS(output, content)         perceptual content distance (VGG backbone)
  - E-SSIM                         SSIM between Sobel edge maps of output/content
  - L_c / L_s                      project VGG losses (same definition as training)

Input JSON: [{"output": path, "content": path, "style": path}, ...]
Usage:
    python scripts/evaluate.py --pairs sanity_check/pairs.json --csv sanity_check/metrics.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ccstytr.data.datasets import test_transform  # noqa: E402
from ccstytr.losses.losses import LossComputer  # noqa: E402
from ccstytr.models.structure import SobelEdge  # noqa: E402

METRIC_KEYS = ("ssim", "lpips", "e_ssim", "l_c", "l_s")


class Evaluator:
    def __init__(self, size: int = 512):
        self.size = size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tf = test_transform(size)
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device).eval()
        self.losses = LossComputer().to(self.device).eval()
        self.edge = SobelEdge().to(self.device)

    def load(self, path: str) -> torch.Tensor:
        return self.tf(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)

    @staticmethod
    def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
        """(B,3,H,W) or (B,1,H,W) tensors in [0,1] -> scalar SSIM."""
        x = a[0].permute(1, 2, 0).cpu().numpy()
        y = b[0].permute(1, 2, 0).cpu().numpy()
        if x.shape[-1] == 1:
            x, y = x[..., 0], y[..., 0]
            return float(structural_similarity(x, y, data_range=1.0))
        return float(structural_similarity(x, y, channel_axis=-1, data_range=1.0))

    @torch.no_grad()
    def __call__(self, output: str, content: str, style: str) -> dict[str, float]:
        out = self.load(output)
        cnt = self.load(content)
        sty = self.load(style)

        ssim = self._ssim(out, cnt)
        lp = float(self.lpips(out * 2 - 1, cnt * 2 - 1).mean())
        e_ssim = self._ssim(self.edge(out), self.edge(cnt))

        out_feats = self.losses.vgg(out)
        c_feats = self.losses.vgg(cnt)
        s_feats = self.losses.vgg(sty)
        l_c = float(self.losses.content_loss(out_feats, c_feats))
        l_s = float(self.losses.style_loss(out_feats, s_feats))

        return {"ssim": ssim, "lpips": lp, "e_ssim": e_ssim, "l_c": l_c, "l_s": l_s}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="JSON list of output/content/style triples")
    parser.add_argument("--csv", default=None, help="write per-triple rows + summary to CSV")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    ev = Evaluator(size=args.size)

    rows = []
    for p in pairs:
        metrics = ev(p["output"], p["content"], p["style"])
        row = {"output": p["output"], **{k: round(v, 4) for k, v in metrics.items()}}
        rows.append(row)
        print(row)

    summary = {
        "output": "MEAN",
        **{k: round(float(np.mean([r[k] for r in rows])), 4) for k in METRIC_KEYS},
    }
    print(summary)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["output", *METRIC_KEYS])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(summary)
        print(f"saved: {args.csv}")


if __name__ == "__main__":
    main()
