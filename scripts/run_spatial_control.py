"""Generate the locked 4,200-output spatial-alpha controllability experiment.

The model is loaded once. For every one of 60 posters and 10 styles, this
script renders five uniform alpha levels and two equal-mean spatial maps:
``protected`` (low alpha on annotated text/logo boxes) and ``inverted``.
Existing valid files are skipped, so an interrupted run is safely resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccstytr.data.datasets import test_transform  # noqa: E402
from ccstytr.models.network import CCStyTr  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
GLOBAL_LEVELS = {
    "alpha_000": 0.0,
    "alpha_025": 0.25,
    "alpha_050": 0.5,
    "alpha_075": 0.75,
    "alpha_100": 1.0,
}
SPATIAL_MAPS = {
    "protected": "alpha_protected",
    "inverted": "alpha_inverted",
}


def images(directory: Path) -> dict[str, Path]:
    found = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    if not found:
        raise FileNotFoundError(f"no images found in {directory}")
    return found


def load_model(checkpoint: Path, device: torch.device) -> CCStyTr:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model_cfg = state.get("model_cfg") if isinstance(state, dict) else None
    model = CCStyTr(**(model_cfg or {})).to(device).eval()
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="data/eval/spatial_control")
    parser.add_argument("--output_dir", default="results/spatial_control")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    protocol = json.loads((eval_dir / "protocol.json").read_text(encoding="utf-8"))
    contents = images(eval_dir / "content")
    styles = images(eval_dir / "style")
    if len(contents) != protocol["n_posters"] or len(styles) != protocol["n_styles"]:
        raise RuntimeError("input counts do not match the locked spatial-control protocol")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)
    image_transform = test_transform(args.size)
    map_transform = transforms.Compose(
        [transforms.Resize((args.size, args.size)), transforms.ToTensor()]
    )
    output_root = Path(args.output_dir)
    for condition in (*GLOBAL_LEVELS, *SPATIAL_MAPS):
        (output_root / condition).mkdir(parents=True, exist_ok=True)

    pairs = [(content_id, style_id) for content_id in sorted(contents) for style_id in sorted(styles)]
    done = 0
    with torch.inference_mode():
        for content_id, style_id in pairs:
            content = image_transform(Image.open(contents[content_id]).convert("RGB"))
            style = image_transform(Image.open(styles[style_id]).convert("RGB"))
            content = content.unsqueeze(0).to(device)
            style = style.unsqueeze(0).to(device)
            alpha_maps = {
                condition: torch.full((1, 1, args.size, args.size), level, device=device)
                for condition, level in GLOBAL_LEVELS.items()
            }
            for condition, map_dir in SPATIAL_MAPS.items():
                map_path = eval_dir / map_dir / f"{content_id}.png"
                alpha_maps[condition] = map_transform(Image.open(map_path).convert("L")).unsqueeze(0).to(device)

            for condition, alpha_map in alpha_maps.items():
                destination = output_root / condition / f"{content_id}__{style_id}.png"
                if not destination.exists():
                    save_image(model(content, style, alpha_map).clamp(0, 1), destination)
                done += 1
                if done % 100 == 0 or done == protocol["outputs"]:
                    print(f"spatial-control: {done}/{protocol['outputs']}", flush=True)

    actual = sum(
        1
        for condition in (*GLOBAL_LEVELS, *SPATIAL_MAPS)
        for path in (output_root / condition).iterdir()
        if path.suffix.lower() == ".png"
    )
    if actual != protocol["outputs"]:
        raise RuntimeError(f"expected {protocol['outputs']} outputs, found {actual}")
    print(f"done -> {output_root}")


if __name__ == "__main__":
    main()
