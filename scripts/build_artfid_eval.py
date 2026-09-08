"""Build the locked 5,000-pair ArtFID-infinity/CSD evaluation set.

The protocol uses one-to-one pairing rather than a Cartesian product:

* 5,000 distinct COCO validation content images;
* 5,000 distinct WikiArt style images sampled with a fixed seed;
* identical ``pair_id.png`` names in the content and style directories.

Matching names let the official ArtFID implementation pair content, style and
stylized outputs without duplicating a small style set.  Images are square-
resized once with Lanczos interpolation, matching the manuscript's existing
512 x 512 evaluation protocol.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SEED = 20260902
DEFAULT_PAIRS = 5000


def image_files(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"no images found in {directory}")
    return files


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def save_square(source: Path, destination: Path, size: int) -> None:
    if destination.exists():
        return
    with Image.open(source) as image:
        image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(
            destination,
            format="PNG",
            compress_level=3,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", required=True, help="COCO val image directory")
    parser.add_argument("--style_dir", required=True, help="WikiArt root (recursive)")
    parser.add_argument("--out_dir", default="data/eval/artfid5000")
    parser.add_argument("--n_pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    args = parser.parse_args()

    if args.n_pairs < 5000:
        raise ValueError("ArtFID-infinity requires at least 5,000 pairs")

    root = Path(__file__).resolve().parents[1]
    content_sources = image_files(Path(args.content_dir), recursive=False)
    style_sources = image_files(Path(args.style_dir), recursive=True)
    if len(content_sources) < args.n_pairs:
        raise ValueError(
            f"need {args.n_pairs} distinct content images, found {len(content_sources)}"
        )
    if len(style_sources) < args.n_pairs:
        raise ValueError(
            f"need {args.n_pairs} distinct style images, found {len(style_sources)}"
        )

    content_rng = random.Random(args.seed)
    style_rng = random.Random(args.seed + 1)
    contents = content_rng.sample(content_sources, args.n_pairs)
    styles = style_rng.sample(style_sources, args.n_pairs)

    out = Path(args.out_dir)
    content_out = out / "content"
    style_out = out / "style"
    content_out.mkdir(parents=True, exist_ok=True)
    style_out.mkdir(parents=True, exist_ok=True)

    manifest = out / "manifest.csv"
    rows = []
    for index, (content, style) in enumerate(zip(contents, styles, strict=True)):
        pair_id = f"{index:05d}"
        rows.append(
            {
                "pair_id": pair_id,
                "content_source": relative_or_absolute(content, root),
                "style_source": relative_or_absolute(style, root),
                "style_group": style.parent.name,
            }
        )

    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if existing != rows:
            raise RuntimeError(
                f"{manifest} describes a different sample; use a new output directory"
            )
    else:
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    tasks = []
    for index, (content, style) in enumerate(zip(contents, styles, strict=True)):
        pair_id = f"{index:05d}.png"
        tasks.extend(
            [
                (content, content_out / pair_id, args.size),
                (style, style_out / pair_id, args.size),
            ]
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(save_square, *task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            future.result()
            if completed % 500 == 0 or completed == len(futures):
                print(
                    f"materialized {completed // 2}/{args.n_pairs} pairs "
                    f"({completed}/{len(futures)} images)",
                    flush=True,
                )

    protocol = {
        "name": "artfid5000",
        "n_pairs": args.n_pairs,
        "n_unique_content": len({row["content_source"] for row in rows}),
        "n_unique_style": len({row["style_source"] for row in rows}),
        "seed": args.seed,
        "size": args.size,
        "resize": "PIL.Image.Resampling.LANCZOS square resize",
        "png_compress_level": 3,
        "pairing": "one-to-one",
    }
    with (out / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"locked protocol -> {out}")


if __name__ == "__main__":
    main()
