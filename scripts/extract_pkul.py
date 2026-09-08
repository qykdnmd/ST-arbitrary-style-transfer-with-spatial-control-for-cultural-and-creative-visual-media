"""Extract original poster images from the PKU-PosterLayout test parquet
(creative-graphic-design/PKU-PosterLayout, CC-BY-SA-4.0) into data/eval/poster/.

Also writes a manifest with per-poster text/logo element counts. The official
Hugging Face representation is zero-indexed: 0=text, 1=logo, 2=underlay.
"""

import csv
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# NOTE: the data/ config parquets have original_poster/annotations nulled out;
# the ralf-style/ config carries the full images. Use the ralf-style test split.
SRCS = [
    "sanity_check/pkul_ralf_test.parquet",
    "sanity_check/pkul_ralf_test2.parquet",
]
OUT = Path("data/eval/poster")
MANIFEST = OUT / "manifest.csv"

OUT.mkdir(parents=True, exist_ok=True)

rows = []
skipped = 0
for src in SRCS:
    t = pq.read_table(src, columns=["original_poster", "annotations"])
    for i in range(t.num_rows):
        poster = t.column("original_poster")[i].as_py()
        if not poster or not poster.get("bytes"):
            skipped += 1
            continue
        ann = t.column("annotations")[i].as_py()
        img_bytes = poster["bytes"]
        name = poster["path"] or f"poster_{i:04d}.png"
        name = Path(name).name  # strip any directory component
        img = Image.open(io.BytesIO(img_bytes))
        dst = OUT / name
        img.save(dst)
        classes = ann.get("cls_elem") or []
        boxes = ann.get("box_elem") or []
        n_text = sum(1 for c in classes if c == 0)
        n_logo = sum(1 for c in classes if c == 1)
        text_boxes = [box for cls, box in zip(classes, boxes) if cls == 0]
        logo_boxes = [box for cls, box in zip(classes, boxes) if cls == 1]
        rows.append(
            (
                name,
                img.size[0],
                img.size[1],
                len(classes),
                n_text,
                n_logo,
                json.dumps(text_boxes, separators=(",", ":")),
                json.dumps(logo_boxes, separators=(",", ":")),
                src,
                i,
            )
        )

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
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
        ]
    )
    w.writerows(rows)

print(f"extracted {len(rows)} posters -> {OUT} (skipped {skipped} empty rows)")
print(f"with text elements: {sum(1 for row in rows if row[4] > 0)}")
print(f"with logo elements: {sum(1 for row in rows if row[5] > 0)}")
