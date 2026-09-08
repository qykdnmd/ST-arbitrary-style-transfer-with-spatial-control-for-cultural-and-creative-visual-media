"""Run all six methods on the locked 5,000 one-to-one evaluation pairs.

Each method is launched in a fresh process to avoid collisions among upstream
repositories that use generic module names such as ``models`` and ``net``.
Every worker loads its model once and resumes by skipping existing outputs.
The script refuses partial/misaligned result sets and records exact revisions
and checkpoint hashes for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ccstytr", "aesfa", "stytr2", "adain", "sanet", "cast")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def stems(directory: Path) -> set[str]:
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="data/eval/artfid5000")
    parser.add_argument("--output_dir", default="results/artfid5000")
    parser.add_argument("--cc_checkpoint", required=True)
    parser.add_argument("--aesfa_checkpoint", default="external/AesFA/ckpt/main/main.pth")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    protocol = json.loads((eval_dir / "protocol.json").read_text(encoding="utf-8"))
    expected = stems(eval_dir / "content")
    if expected != stems(eval_dir / "style") or len(expected) != protocol["n_pairs"]:
        raise RuntimeError("locked input directories are incomplete or misaligned")

    selected = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    for method in selected:
        command = [
            sys.executable,
            "scripts/infer_paired.py",
            "--method",
            method,
            "--pairing",
            "paired",
            "--content_dir",
            str(eval_dir / "content"),
            "--style_dir",
            str(eval_dir / "style"),
            "--output_dir",
            str(output_dir / method),
            "--size",
            str(args.size),
        ]
        if method == "ccstytr":
            command.extend(["--checkpoint", args.cc_checkpoint])
        elif method == "aesfa":
            command.extend(["--checkpoint", args.aesfa_checkpoint])
        print(f"=== {method} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        actual = stems(output_dir / method)
        if actual != expected:
            raise RuntimeError(
                f"{method} output mismatch: missing={sorted(expected - actual)[:5]}, "
                f"extra={sorted(actual - expected)[:5]}"
            )

    checkpoint_paths = {
        "ccstytr": Path(args.cc_checkpoint),
        "aesfa": Path(args.aesfa_checkpoint),
    }
    external_roots = {
        "aesfa": ROOT / "external/AesFA",
        "stytr2": ROOT / "external/StyTR-2",
        "adain": ROOT / "external/pytorch-AdaIN",
        "sanet": ROOT / "external/SANET",
        "cast": ROOT / "external/CAST_pytorch",
    }
    provenance = {
        "protocol": protocol,
        "methods": selected,
        "repository_revisions": {
            method: git_revision(directory) for method, directory in external_roots.items()
        },
        "checkpoint_sha256": {
            method: sha256(path) for method, path in checkpoint_paths.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inference_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"complete -> {output_dir}")


if __name__ == "__main__":
    main()
