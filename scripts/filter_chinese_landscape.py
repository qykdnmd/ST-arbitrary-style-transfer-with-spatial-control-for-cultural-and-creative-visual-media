"""Filter the Cleveland CC0 Chinese-painting pool down to the landscape
(shan-shui) subset required by experiment protocol (100-150 images).

Keyword scoring on the English titles in the Cleveland manifest; exclusion
terms drop figure/flower-and-bird/calligraphy works that merely mention a
landscape word in passing. Deterministic: manifest order is kept, no sampling
beyond the keyword pass/fail (cap 150).

Outputs data/eval/styles/chinese_landscape/ + manifest.csv (with the matched
keyword for manual audit).
"""

import csv
import shutil
from pathlib import Path

SRC = Path("data/eval/styles/chinese_painting")
OUT = Path("data/eval/styles/chinese_landscape")
CAP = 150

INCLUDE = (
    "landscape", "mountain", "river", "valley", "stream", "waterfall", "peak",
    "hills", "lake", "ravine", "gorge", "cliff", "shan shui", "waterside",
    "riverbank", "seascape", "village", "traveler", "traveller", "hermit",
    "pavilion", "retreat",
)
EXCLUDE = (
    "portrait", "figure", "lady", "court", "beggar", "street", "flower",
    "bird", "fish", "calligraphy", "poem", "song", "letter", "album of calligraphy",
    "buddha", "bodhisattva", "luohan", "arhat", "dragon", "horse", "buffalo",
    "bamboo", "plum", "orchid", "chrysanthemum", "peony", "lotus", "crane",
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out = []
    with open(SRC / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = (row["title"] or "").lower()
            if not title:
                continue
            hit = next((k for k in INCLUDE if k in title), None)
            if hit is None:
                continue
            if any(k in title for k in EXCLUDE):
                continue
            if not (SRC / row["file"]).exists():
                continue
            rows_out.append((row["file"], row["title"], row["creation_date"], hit))
            if len(rows_out) >= CAP:
                break

    for fname, *_ in rows_out:
        shutil.copy(SRC / fname, OUT / fname)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "title", "creation_date", "matched_keyword"])
        w.writerows(rows_out)
    print(f"landscape subset: {len(rows_out)} -> {OUT}")


if __name__ == "__main__":
    main()
