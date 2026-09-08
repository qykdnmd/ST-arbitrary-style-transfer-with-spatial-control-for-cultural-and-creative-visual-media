"""Attach official PKU text/logo boxes to the frozen featured-set membership.

The existing ``poster_curated/manifest.csv`` defines the already evaluated
120-poster set and is never overwritten.  This script finds those exact files
in the original PKU ralf-style parquet shards, applies the official class
mapping (0=text, 1=logo, 2=underlay), and writes a separate enriched manifest
for qualitative crop selection and spatial-control analysis.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def load_membership(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "file" not in rows[0]:
        raise RuntimeError(f"invalid featured manifest: {path}")
    names = {Path(row["file"]).name for row in rows}
    if len(names) != len(rows):
        raise RuntimeError("featured manifest contains duplicate filenames")
    return rows, names


def scan_parquets(paths: list[Path], wanted: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for parquet_path in paths:
        table = pq.read_table(parquet_path, columns=["original_poster", "annotations"])
        for row_index in range(table.num_rows):
            poster = table.column("original_poster")[row_index].as_py()
            if not poster:
                continue
            name = Path(poster.get("path") or f"poster_{row_index:04d}.png").name
            if name not in wanted or name in found:
                continue
            annotations = table.column("annotations")[row_index].as_py() or {}
            classes = annotations.get("cls_elem") or []
            boxes = annotations.get("box_elem") or []
            if len(classes) != len(boxes):
                raise RuntimeError(f"class/box mismatch for {name}")
            size = (None, None)
            if poster.get("bytes"):
                with Image.open(io.BytesIO(poster["bytes"])) as image:
                    size = image.size
            found[name] = {
                "width": size[0],
                "height": size[1],
                "n_elements": len(classes),
                "n_text_elements": sum(int(cls == 0) for cls in classes),
                "n_logo_elements": sum(int(cls == 1) for cls in classes),
                "text_boxes": [box for cls, box in zip(classes, boxes) if cls == 0],
                "logo_boxes": [box for cls, box in zip(classes, boxes) if cls == 1],
                "source_parquet": str(parquet_path),
                "source_row": row_index,
            }
    missing = sorted(wanted - found.keys())
    if missing:
        raise RuntimeError(f"{len(missing)} featured posters not found in PKU shards: {missing[:5]}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--featured_manifest", default="data/eval/poster_curated/manifest.csv")
    parser.add_argument(
        "--parquets",
        nargs="+",
        default=["sanity_check/pkul_ralf_test.parquet", "sanity_check/pkul_ralf_test2.parquet"],
    )
    parser.add_argument(
        "--output", default="data/eval/poster_curated/manifest_annotations.csv"
    )
    args = parser.parse_args()

    membership, wanted = load_membership(Path(args.featured_manifest))
    annotations = scan_parquets([Path(path) for path in args.parquets], wanted)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "width",
        "height",
        "n_elements",
        "n_text_elements",
        "n_logo_elements",
        "text_boxes",
        "logo_boxes",
        "source_parquet",
        "source_row",
        "frozen_membership",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for original in membership:
            name = Path(original["file"]).name
            row = {"file": name, **annotations[name], "frozen_membership": "true"}
            row["text_boxes"] = json.dumps(row["text_boxes"], separators=(",", ":"))
            row["logo_boxes"] = json.dumps(row["logo_boxes"], separators=(",", ":"))
            writer.writerow(row)
    both = sum(
        value["n_text_elements"] > 0 and value["n_logo_elements"] > 0
        for value in annotations.values()
    )
    print(f"wrote {len(membership)} frozen members to {output}; {both} contain text and logo")


if __name__ == "__main__":
    main()
