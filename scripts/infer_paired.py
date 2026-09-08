"""Run paired or Cartesian inference with the six benchmark methods.

In paired mode both input directories must contain the same image stems and
outputs keep those stems, allowing the official ArtFID implementation to align
them lexicographically. Cartesian mode writes ``content__style.png`` names for
the manuscript's grid benchmarks. The model is loaded once and existing
outputs are skipped for safe resumption.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def indexed_images(directory: Path) -> dict[str, Path]:
    images = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    if not images:
        raise FileNotFoundError(f"no images found in {directory}")
    return images


def load_ccstytr(checkpoint: Path, size: int, device: torch.device) -> Callable:
    sys.path.insert(0, str(ROOT))
    from ccstytr.data.datasets import test_transform
    from ccstytr.models.network import CCStyTr

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model_cfg = state.get("model_cfg") if isinstance(state, dict) else None
    model = CCStyTr(**(model_cfg or {})).to(device).eval()
    model.load_state_dict(state["model"] if "model" in state else state)
    transform = test_transform(size)

    def stylize(content_path: Path, style_path: Path) -> torch.Tensor:
        content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        style = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        alpha = torch.ones((1, 1, size, size), device=device)
        return model(content, style, alpha).clamp(0, 1)

    return stylize


def load_aesfa(root: Path, checkpoint: Path, size: int, device: torch.device) -> Callable:
    if not root.is_dir():
        raise FileNotFoundError(
            f"missing AesFA checkout at {root}; clone https://github.com/Sooyyoungg/AesFA"
        )
    sys.path.insert(0, str(root))
    from Config import Config
    from model import AesFA_test

    config = Config()
    config.test_content_size = size
    config.test_style_size = size
    model = AesFA_test(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.netE.load_state_dict(state["netE"])
    model.netS.load_state_dict(state["netS"])
    model.netG.load_state_dict(state["netG"])
    del state
    model = model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(size=size),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    def stylize(content_path: Path, style_path: Path) -> torch.Tensor:
        content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        style = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        output, _ = model(content, style, False)
        return (output * std + mean).clamp(0, 1)

    return stylize


def load_stytr2(root: Path, size: int, device: torch.device) -> Callable:
    if not root.is_dir():
        raise FileNotFoundError(f"missing StyTr2 checkout at {root}")
    sys.path.insert(0, str(root))
    import models.StyTR as stytr
    import models.transformer as transformer

    arguments = SimpleNamespace(position_embedding="sine", hidden_dim=512)
    vgg = stytr.vgg
    decoder = stytr.decoder
    trans = transformer.Transformer()
    embedding = stytr.PatchEmbed()
    weights = root / "experiments"
    vgg.load_state_dict(
        torch.load(weights / "vgg_normalised.pth", map_location="cpu", weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(weights / "decoder_iter_160000.pth", map_location="cpu", weights_only=True)
    )
    trans.load_state_dict(
        torch.load(weights / "transformer_iter_160000.pth", map_location="cpu", weights_only=True)
    )
    embedding.load_state_dict(
        torch.load(weights / "embedding_iter_160000.pth", map_location="cpu", weights_only=True)
    )
    vgg = torch.nn.Sequential(*list(vgg.children())[:44])
    model = stytr.StyTrans(vgg, decoder, embedding, trans, arguments).to(device).eval()
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def stylize(content_path: Path, style_path: Path) -> torch.Tensor:
        content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        style = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        output = model(content, style)
        return (output[0] if isinstance(output, (tuple, list)) else output).clamp(0, 1)

    return stylize


def load_adain(root: Path, size: int, device: torch.device) -> Callable:
    if not root.is_dir():
        raise FileNotFoundError(f"missing AdaIN checkout at {root}")
    sys.path.insert(0, str(root))
    import net
    from function import adaptive_instance_normalization

    decoder = net.decoder
    vgg = net.vgg
    decoder.load_state_dict(
        torch.load(root / "models/decoder.pth", map_location="cpu", weights_only=True)
    )
    vgg.load_state_dict(
        torch.load(root / "models/vgg_normalised.pth", map_location="cpu", weights_only=True)
    )
    vgg = torch.nn.Sequential(*list(vgg.children())[:31]).to(device).eval()
    decoder = decoder.to(device).eval()
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def stylize(content_path: Path, style_path: Path) -> torch.Tensor:
        content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        style = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        feature = adaptive_instance_normalization(vgg(content), vgg(style))
        return decoder(feature).clamp(0, 1)

    return stylize


def load_sanet(root: Path, size: int, device: torch.device) -> Callable:
    """Load exact upstream definitions without executing Eval.py's CLI body."""
    if not root.is_dir():
        raise FileNotFoundError(f"missing SANet checkout at {root}")
    source_path = root / "Eval.py"
    source = source_path.read_text(encoding="utf-8")
    marker = "parser = argparse.ArgumentParser()"
    if marker not in source:
        raise RuntimeError(f"unexpected SANet Eval.py at {source_path}")
    namespace = {"__name__": "sanet_definitions", "__file__": str(source_path)}
    exec(compile(source.split(marker, maxsplit=1)[0], str(source_path), "exec"), namespace)
    decoder = namespace["decoder"]
    vgg = namespace["vgg"]
    transform_module = namespace["Transform"](in_planes=512)
    weights = root / "experiments"
    decoder.load_state_dict(
        torch.load(weights / "decoder_iter_500000.pth", map_location="cpu", weights_only=True)
    )
    transform_module.load_state_dict(
        torch.load(
            weights / "transformer_iter_500000.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    vgg.load_state_dict(
        torch.load(weights / "vgg_normalised.pth", map_location="cpu", weights_only=True)
    )
    enc_1 = torch.nn.Sequential(*list(vgg.children())[:4]).to(device).eval()
    enc_2 = torch.nn.Sequential(*list(vgg.children())[4:11]).to(device).eval()
    enc_3 = torch.nn.Sequential(*list(vgg.children())[11:18]).to(device).eval()
    enc_4 = torch.nn.Sequential(*list(vgg.children())[18:31]).to(device).eval()
    enc_5 = torch.nn.Sequential(*list(vgg.children())[31:44]).to(device).eval()
    decoder = decoder.to(device).eval()
    transform_module = transform_module.to(device).eval()
    image_transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )

    def encode(image: torch.Tensor):
        relu4 = enc_4(enc_3(enc_2(enc_1(image))))
        return relu4, enc_5(relu4)

    def stylize(content_path: Path, style_path: Path) -> torch.Tensor:
        content = image_transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        style = image_transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        content4, content5 = encode(content)
        style4, style5 = encode(style)
        return decoder(transform_module(content4, style4, content5, style5)).clamp(0, 1)

    return stylize


def load_cast(root: Path, size: int, device: torch.device) -> Callable:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing CAST checkout at {root}")
    sys.path.insert(0, str(root))
    from data.base_dataset import get_transform
    from models import create_model
    from options.test_options import TestOptions

    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        sys.argv = [
            "paired_cast",
            "--dataroot",
            str(root),
            "--name",
            "CAST_model",
            "--model",
            "cast",
            "--CAST_mode",
            "CAST",
            "--phase",
            "test",
            "--preprocess",
            "resize",
            "--load_size",
            str(size),
            "--crop_size",
            str(size),
            "--num_threads",
            "0",
            "--batch_size",
            "1",
            "--serial_batches",
            "--no_flip",
        ]
        options = TestOptions().parse()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    options.checkpoints_dir = str(root / "checkpoints")
    original_torch_load = torch.load

    def safe_torch_load(*load_args, **load_kwargs):
        load_kwargs["weights_only"] = True
        return original_torch_load(*load_args, **load_kwargs)

    try:
        os.chdir(root)
        torch.load = safe_torch_load
        model = create_model(options)
        model.setup(options)
        model.eval()
    finally:
        torch.load = original_torch_load
        os.chdir(previous_cwd)
    transform = get_transform(options, grayscale=False)

    def stylize(content_path: Path, style_path: Path) -> torch.Tensor:
        content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0)
        style = transform(Image.open(style_path).convert("RGB")).unsqueeze(0)
        model.set_input(
            {
                "A": content,
                "B": style,
                "A_paths": [str(content_path)],
                "B_paths": [str(style_path)],
            }
        )
        model.test()
        output = model.get_current_visuals()["fake_B"].to(device)
        return ((output + 1.0) / 2.0).clamp(0, 1)

    return stylize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=("ccstytr", "aesfa", "stytr2", "adain", "sanet", "cast"),
    )
    parser.add_argument("--pairing", choices=("paired", "cartesian"), default="paired")
    parser.add_argument("--content_dir", required=True)
    parser.add_argument("--style_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--checkpoint")
    parser.add_argument("--aesfa_root", default="external/AesFA")
    parser.add_argument("--stytr2_root", default="external/StyTR-2")
    parser.add_argument("--adain_root", default="external/pytorch-AdaIN")
    parser.add_argument("--sanet_root", default="external/SANET")
    parser.add_argument("--cast_root", default="external/CAST_pytorch")
    args = parser.parse_args()

    contents = indexed_images(Path(args.content_dir))
    styles = indexed_images(Path(args.style_dir))
    if args.pairing == "paired" and set(contents) != set(styles):
        missing_style = sorted(set(contents) - set(styles))[:5]
        missing_content = sorted(set(styles) - set(contents))[:5]
        raise ValueError(
            "paired input stems differ: "
            f"missing_style={missing_style}, missing_content={missing_content}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.method == "ccstytr":
        if not args.checkpoint:
            parser.error("--checkpoint is required for ccstytr")
        stylize = load_ccstytr(Path(args.checkpoint), args.size, device)
    elif args.method == "aesfa":
        if not args.checkpoint:
            parser.error("--checkpoint is required for aesfa")
        stylize = load_aesfa(
            Path(args.aesfa_root), Path(args.checkpoint), args.size, device
        )
    elif args.method == "stytr2":
        stylize = load_stytr2(Path(args.stytr2_root), args.size, device)
    elif args.method == "adain":
        stylize = load_adain(Path(args.adain_root), args.size, device)
    elif args.method == "sanet":
        stylize = load_sanet(Path(args.sanet_root), args.size, device)
    else:
        stylize = load_cast(Path(args.cast_root), args.size, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.pairing == "paired":
        pairs = [(pair_id, pair_id, pair_id) for pair_id in sorted(contents)]
    else:
        pairs = [
            (f"{content_id}__{style_id}", content_id, style_id)
            for content_id in sorted(contents)
            for style_id in sorted(styles)
        ]
    with torch.inference_mode():
        for index, (output_id, content_id, style_id) in enumerate(pairs, start=1):
            destination = output_dir / f"{output_id}.png"
            if not destination.exists():
                output = stylize(contents[content_id], styles[style_id])
                save_image(output, destination)
            if index % 100 == 0 or index == len(pairs):
                print(f"{args.method}: {index}/{len(pairs)}", flush=True)
    print(f"done -> {output_dir}")


if __name__ == "__main__":
    main()
