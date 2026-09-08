"""Build the locked 60-poster x 10-style spatial-alpha evaluation set.

Posters are selected without looking at model outputs from the annotated
PKU-PosterLayout ralf-style split. Every selected poster contains both text
(class 0) and logo (class 1) boxes and is excluded from the existing featured
benchmark. Official boxes define the protected-region mask.

For each poster, the protected and inverted alpha maps both have mean 0.5.
Consequently the uniform alpha=0.5 output is an exact equal-mean control.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 20260902
N_POSTERS = 60
N_STYLES = 10
SIZE = 512
LOW_ALPHA = 0.1


def annotation_records(raw: object) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [record for record in raw if isinstance(record, dict)]
    if isinstance(raw, dict):
        lengths = [len(value) for value in raw.values() if isinstance(value, list)]
        if not lengths:
            return []
        return [
            {key: value[index] for key, value in raw.items() if isinstance(value, list)}
            for index in range(min(lengths))
        ]
    raise TypeError(f"unsupported annotation representation: {type(raw)!r}")


def dhash(image: Image.Image) -> np.ndarray:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    array = np.asarray(gray, dtype=np.int16)
    return (array[:, :-1] > array[:, 1:]).reshape(-1)


def scaled_box(box: list[int], width: int, height: int, size: int) -> tuple[int, ...]:
    if len(box) != 4:
        raise ValueError(f"expected [x1,y1,x2,y2], got {box!r}")
    x1, y1, x2, y2 = box
    coordinates = (
        round(x1 * size / width),
        round(y1 * size / height),
        round(x2 * size / width),
        round(y2 * size / height),
    )
    sx1, sy1, sx2, sy2 = coordinates
    sx1, sx2 = sorted((max(0, min(size - 1, sx1)), max(0, min(size, sx2))))
    sy1, sy2 = sorted((max(0, min(size - 1, sy1)), max(0, min(size, sy2))))
    return sx1, sy1, max(sx1 + 1, sx2), max(sy1 + 1, sy2)


def read_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {Path(row["file"]).name for row in csv.DictReader(handle)}


def collect_candidates(parquets: list[Path], exclusions: set[str], size: int) -> list[dict]:
    candidates = []
    seen_names: set[str] = set()
    hashes: list[np.ndarray] = []
    for parquet in parquets:
        table = pq.read_table(parquet, columns=["original_poster", "annotations"])
        for row_index in range(table.num_rows):
            poster = table.column("original_poster")[row_index].as_py()
            if not poster or not poster.get("bytes"):
                continue
            name = Path(poster.get("path") or f"{parquet.stem}_{row_index:05d}.png").name
            if name in exclusions or name in seen_names:
                continue
            records = annotation_records(table.column("annotations")[row_index].as_py())
            text_boxes = [record["box_elem"] for record in records if record.get("cls_elem") == 0]
            logo_boxes = [record["box_elem"] for record in records if record.get("cls_elem") == 1]
            if not text_boxes or not logo_boxes:
                continue

            image = Image.open(io.BytesIO(poster["bytes"])).convert("RGB")
            image_hash = dhash(image)
            if any(np.count_nonzero(image_hash != previous) <= 4 for previous in hashes):
                continue
            mask = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask)
            all_boxes = text_boxes + logo_boxes
            scaled = [scaled_box(box, image.width, image.height, size) for box in all_boxes]
            for box in scaled:
                draw.rectangle(box, fill=255)
            coverage = float(np.asarray(mask, dtype=np.float32).mean() / 255.0)
            if not 0.03 <= coverage <= 0.45:
                continue

            hashes.append(image_hash)
            seen_names.add(name)
            candidates.append(
                {
                    "name": name,
                    "image": image,
                    "mask": mask,
                    "coverage": coverage,
                    "n_text": len(text_boxes),
                    "n_logo": len(logo_boxes),
                    "text_boxes": text_boxes,
                    "logo_boxes": logo_boxes,
                    "source": str(parquet),
                    "source_row": row_index,
                }
            )
    return candidates


def stratified_sample(candidates: list[dict], count: int, seed: int) -> list[dict]:
    if len(candidates) < count:
        raise ValueError(f"need {count} eligible posters, found {len(candidates)}")
    ordered = sorted(candidates, key=lambda row: (row["coverage"], row["name"]))
    # Contiguous coverage tertiles preserve the intended low/medium/high
    # protected-area strata. Interleaving (ordered[index::3]) would make every
    # tier span the full coverage range and therefore would not be stratified.
    tiers = [list(tier) for tier in np.array_split(np.asarray(ordered, dtype=object), 3)]
    rng = random.Random(seed)
    selected = []
    for tier in tiers:
        selected.extend(rng.sample(tier, count // 3))
    if len(selected) < count:
        remainder = [row for row in ordered if row not in selected]
        selected.extend(rng.sample(remainder, count - len(selected)))
    return sorted(selected, key=lambda row: row["name"])


def select_styles(style_root: Path, count: int, seed: int) -> list[Path]:
    groups = {}
    for path in sorted(style_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            groups.setdefault(path.parent.name, []).append(path)
    if len(groups) < count:
        raise ValueError(f"need {count} WikiArt groups, found {len(groups)}")
    rng = random.Random(seed)
    chosen_groups = rng.sample(sorted(groups), count)
    return [rng.choice(groups[group]) for group in chosen_groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--style_dir", required=True)
    parser.add_argument("--out_dir", default="data/eval/spatial_control")
    parser.add_argument("--exclude_manifest", default="data/eval/poster_curated/manifest.csv")
    parser.add_argument("--n_posters", type=int, default=N_POSTERS)
    parser.add_argument("--n_styles", type=int, default=N_STYLES)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out = Path(args.out_dir)
    content_dir = out / "content"
    style_dir = out / "style"
    protected_dir = out / "alpha_protected"
    inverted_dir = out / "alpha_inverted"
    for directory in (content_dir, style_dir, protected_dir, inverted_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exclusions = read_exclusions(Path(args.exclude_manifest))
    candidates = collect_candidates([Path(path) for path in args.parquet], exclusions, args.size)
    posters = stratified_sample(candidates, args.n_posters, args.seed)
    styles = select_styles(Path(args.style_dir), args.n_styles, args.seed + 1)

    poster_rows = []
    for index, row in enumerate(posters):
        poster_id = f"p{index:03d}"
        row["image"].resize((args.size, args.size), Image.Resampling.LANCZOS).save(
            content_dir / f"{poster_id}.png"
        )
        protected_region = np.asarray(row["mask"], dtype=np.float32) / 255.0
        coverage = float(protected_region.mean())
        outside_alpha = (0.5 - coverage * LOW_ALPHA) / (1.0 - coverage)
        protected = np.where(protected_region > 0.5, LOW_ALPHA, outside_alpha)
        inverted = 1.0 - protected
        Image.fromarray(np.round(protected * 255).astype(np.uint8), mode="L").save(
            protected_dir / f"{poster_id}.png"
        )
        Image.fromarray(np.round(inverted * 255).astype(np.uint8), mode="L").save(
            inverted_dir / f"{poster_id}.png"
        )
        poster_rows.append(
            {
                "poster_id": poster_id,
                "source_file": row["name"],
                "source_parquet": row["source"],
                "source_row": row["source_row"],
                "n_text": row["n_text"],
                "n_logo": row["n_logo"],
                "protected_fraction": f"{coverage:.8f}",
                "protected_alpha": f"{LOW_ALPHA:.8f}",
                "outside_alpha": f"{outside_alpha:.8f}",
                "mean_alpha": f"{protected.mean():.8f}",
                "text_boxes": json.dumps(row["text_boxes"], separators=(",", ":")),
                "logo_boxes": json.dumps(row["logo_boxes"], separators=(",", ":")),
            }
        )

    with (out / "posters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=poster_rows[0].keys())
        writer.writeheader()
        writer.writerows(poster_rows)

    style_rows = []
    for index, source in enumerate(styles):
        style_id = f"s{index:02d}"
        with Image.open(source) as image:
            image.convert("RGB").resize(
                (args.size, args.size), Image.Resampling.LANCZOS
            ).save(style_dir / f"{style_id}.png")
        style_rows.append(
            {"style_id": style_id, "source_file": str(source), "style_group": source.parent.name}
        )
    with (out / "styles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=style_rows[0].keys())
        writer.writeheader()
        writer.writerows(style_rows)

    protocol = {
        "seed": args.seed,
        "n_posters": args.n_posters,
        "n_styles": args.n_styles,
        "global_alpha_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
        "spatial_conditions": ["protected", "inverted"],
        "equal_mean_control": 0.5,
        "outputs": args.n_posters * args.n_styles * 7,
        "class_mapping": {"0": "text", "1": "logo", "2": "underlay"},
    }
    with (out / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"locked spatial-control protocol -> {out}")


if __name__ == "__main__":
    main()
