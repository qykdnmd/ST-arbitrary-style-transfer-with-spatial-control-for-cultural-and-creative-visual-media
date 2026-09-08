"""Evaluate the locked 5,000-pair benchmark with ArtFID-infinity and CSD.

This script intentionally uses the official upstream implementations from
``external/art-fid`` and ``external/CSD``. It validates one-to-one filename
alignment before any metric is computed, records the exact repository and
checkpoint identities, and writes both per-pair CSD similarities and summary
statistics. At N=5,000, ArtFID-infinity is the official estimator at its
minimum supported sample size; the manuscript must state that limitation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
METHODS = ("ccstytr", "aesfa", "stytr2", "adain", "sanet", "cast")
BOOTSTRAP_SEED = 20260902
N_BOOTSTRAP = 10_000


def indexed(directory: Path) -> dict[str, Path]:
    files = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    if not files:
        raise FileNotFoundError(f"no images in {directory}")
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(directory: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def seed_artfid_checkpoint(local_path: Path) -> Path | None:
    """Mirror a persistent art_inception.pth into the upstream download cache.

    ``art_fid`` downloads art_inception.pth from huggingface.co into
    ``tempfile.gettempdir()/art_fid`` on first use. That cache is volatile
    (the OS reclaims temp files) and the host is not reachable from this
    machine, so a verified local copy is mirrored in before any metric runs.
    Returns the local path when it was used, else None.
    """
    if not local_path.exists():
        return None
    cache = Path(tempfile.gettempdir()) / "art_fid" / local_path.name
    if not cache.exists() or cache.stat().st_size != local_path.stat().st_size:
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, cache)
    return local_path


def supported_call(function, *args, **kwargs):
    """Call an upstream function with only keywords supported by its version."""
    signature = inspect.signature(function)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **accepted)


class PairedImages(Dataset):
    def __init__(self, outputs: dict[str, Path], styles: dict[str, Path], transform):
        self.ids = sorted(outputs)
        self.outputs = outputs
        self.styles = styles
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        pair_id = self.ids[index]
        with Image.open(self.outputs[pair_id]) as image:
            output = self.transform(image.convert("RGB"))
        with Image.open(self.styles[pair_id]) as image:
            style = self.transform(image.convert("RGB"))
        return pair_id, output, style


def load_csd(csd_root: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(csd_root))
    from CSD.loss_utils import transforms_branch0
    from CSD.model import CSD_CLIP
    from CSD.utils import convert_state_dict

    model = CSD_CLIP("vit_large", "default")
    # The official checkpoint stores only tensors plus argparse/NumPy scalar
    # metadata. Keep PyTorch's restricted unpickler enabled and allowlist only
    # the three statically reported metadata globals and the concrete float64
    # dtype class required to reconstruct that scalar.
    safe_globals = [
        argparse.Namespace,
        np.dtype,
        np.dtypes.Float64DType,
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("CSD checkpoint lacks the official model_state_dict field")
    state = convert_state_dict(checkpoint["model_state_dict"])
    message = model.load_state_dict(state, strict=False)
    print(f"loaded CSD checkpoint: {message}", flush=True)
    return model.to(device).eval(), transforms_branch0


def style_embedding(model, batch: torch.Tensor) -> torch.Tensor:
    outputs = model(batch)
    embedding = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
    return torch.nn.functional.normalize(embedding.float(), dim=-1)


def bootstrap_mean_ci(values: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 100):
        stop = min(repeats, start + 100)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="data/eval/artfid5000")
    parser.add_argument("--results_dir", default="results/artfid5000")
    parser.add_argument("--artfid_root", default="external/art-fid")
    parser.add_argument("--artfid_ckpt", default="external/art-fid/ckpt/art_inception.pth")
    parser.add_argument("--csd_root", default="external/CSD")
    parser.add_argument("--csd_checkpoint", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    results_dir = Path(args.results_dir)
    protocol = json.loads((eval_dir / "protocol.json").read_text(encoding="utf-8"))
    pair_ids = set(indexed(eval_dir / "content"))
    styles = indexed(eval_dir / "style")
    if pair_ids != set(styles) or len(pair_ids) != protocol["n_pairs"]:
        raise RuntimeError("content/style files do not match the locked protocol")

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    method_outputs = {}
    for method in methods:
        outputs = indexed(results_dir / method)
        if set(outputs) != pair_ids:
            missing = sorted(pair_ids - set(outputs))[:5]
            extra = sorted(set(outputs) - pair_ids)[:5]
            raise RuntimeError(f"{method} filename mismatch: missing={missing}, extra={extra}")
        method_outputs[method] = outputs

    artfid_root = Path(args.artfid_root)
    sys.path.insert(0, str(artfid_root))
    artfid_ckpt = seed_artfid_checkpoint(Path(args.artfid_ckpt))
    if artfid_ckpt is None:
        print("warning: no persistent art_inception.pth; relying on upstream download")
    from art_fid import art_fid

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    csd_checkpoint = Path(args.csd_checkpoint)
    csd_model, csd_transform = load_csd(Path(args.csd_root), csd_checkpoint, device)
    summary_rows = []
    csd_rows = []
    for method_index, method in enumerate(methods):
        output_dir = results_dir / method
        print(f"=== {method}: ArtFID-infinity ===", flush=True)
        fid_infinity = float(
            supported_call(
                art_fid.compute_fid_infinity,
                str(output_dir),
                str(eval_dir / "style"),
                batch_size=args.batch_size,
                device=str(device),
                num_workers=args.workers,
                num_points=15,
            )
        )
        content_distance = float(
            supported_call(
                art_fid.compute_content_distance,
                str(output_dir),
                str(eval_dir / "content"),
                batch_size=args.batch_size,
                metric="lpips",
                device=str(device),
                num_workers=args.workers,
            )
        )
        artfid_infinity = (1.0 + content_distance) * (1.0 + fid_infinity)

        loader = DataLoader(
            PairedImages(method_outputs[method], styles, csd_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        similarities = []
        with torch.inference_mode():
            for ids, output_batch, style_batch in loader:
                output_embedding = style_embedding(csd_model, output_batch.to(device))
                style_embedding_batch = style_embedding(csd_model, style_batch.to(device))
                batch_similarity = (output_embedding * style_embedding_batch).sum(dim=-1)
                for pair_id, value in zip(ids, batch_similarity.cpu().tolist(), strict=True):
                    similarities.append(float(value))
                    csd_rows.append({"method": method, "pair_id": pair_id, "csd_similarity": value})
        similarity_array = np.asarray(similarities, dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(
            similarity_array, BOOTSTRAP_SEED + method_index, N_BOOTSTRAP
        )
        summary_rows.append(
            {
                "method": method,
                "n": len(similarities),
                "fid_infinity": fid_infinity,
                "lpips_content": content_distance,
                "artfid_infinity": artfid_infinity,
                "csd_similarity_mean": float(similarity_array.mean()),
                "csd_ci95_low": ci_low,
                "csd_ci95_high": ci_high,
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "csd_per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csd_rows[0].keys())
        writer.writeheader()
        writer.writerows(csd_rows)
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    provenance = {
        "protocol": protocol,
        "art_fid_revision": git_revision(artfid_root),
        "art_inception_checkpoint": str(artfid_ckpt) if artfid_ckpt else "upstream download",
        "art_inception_sha256": sha256(artfid_ckpt) if artfid_ckpt else "unknown",
        "csd_revision": git_revision(Path(args.csd_root)),
        "csd_checkpoint": str(csd_checkpoint),
        "csd_checkpoint_sha256": sha256(csd_checkpoint),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": N_BOOTSTRAP,
        "note": "N=5000 is the minimum supported ArtFID-infinity sample size.",
        "csd_weight_note": (
            "The official CSD repository warns that released weights may not exactly "
            "reproduce the paper's numbers; this run reports the checkpoint hash."
        ),
    }
    (results_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved metrics -> {results_dir}")


if __name__ == "__main__":
    main()
