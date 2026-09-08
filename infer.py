"""CC-StyTr inference.

Three alpha-map modes (plan section 1.1-C):
  a. global slider:   --alpha 0.8
  b. user-edited map: --alpha-map path/to/gray_map.png
  c. automatic:       --auto-alpha (low intensity on structure/text regions)

Usage:
    python infer.py --checkpoint ckpt.pth --content c.jpg --style s.jpg --output out.png
"""

import argparse

import torch
from PIL import Image
from torchvision.utils import save_image

from ccstytr.data.alpha_sampler import auto_alpha_map
from ccstytr.data.datasets import test_transform
from ccstytr.models.network import CCStyTr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0, help="global style intensity")
    parser.add_argument("--alpha-map", default=None, help="grayscale alpha-map image path")
    parser.add_argument(
        "--auto-alpha", action="store_true", help="derive alpha-map from structure saliency"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.checkpoint, map_location=device)
    model_cfg = state.get("model_cfg") if isinstance(state, dict) else None
    model = CCStyTr(**(model_cfg or {})).to(device).eval()
    model.load_state_dict(state["model"] if "model" in state else state)

    tf = test_transform(args.size)
    content = tf(Image.open(args.content).convert("RGB")).unsqueeze(0).to(device)
    style = tf(Image.open(args.style).convert("RGB")).unsqueeze(0).to(device)

    if args.alpha_map is not None:
        gray = Image.open(args.alpha_map).convert("L").resize((args.size, args.size))
        alpha_map = (
            torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8)
            .float()
            .view(1, 1, args.size, args.size)
            .to(device)
            / 255.0
        )
    elif args.auto_alpha:
        with torch.no_grad():
            alpha_map = auto_alpha_map(model.edge(content))
    else:
        alpha_map = torch.full((1, 1, args.size, args.size), args.alpha, device=device)

    with torch.no_grad():
        output = model(content, style, alpha_map).clamp(0, 1)
    save_image(output, args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
