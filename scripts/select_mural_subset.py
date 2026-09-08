"""Materialize the visually-audited MuralDH subset (experiment protocol).

The audit was done on sanity_check/mural_audit/sheet_*.png (top 240 of
manifest_ranked.csv by colorfulness). ACCEPT_RANKS below lists the candidates
judged intact: vivid pigments, no large flaked/whitish/blackened damage, no
modern intrusions (color-chart cards, labels, black scale bars).

Copies the accepted murals from data/sources/mural512/ into
data/eval/styles/mural/ and writes manifest_selected.csv there.
"""

import csv
import shutil
from pathlib import Path

MANIFEST = Path("data/eval/styles/mural/manifest_ranked.csv")
RAW = Path("data/sources/mural512")
OUT = Path("data/eval/styles/mural")

ACCEPT_RANKS = [
    # sheet_00 (ranks 0-47)
    2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 23, 24, 26, 27, 29,
    30, 33, 34, 35, 36, 39, 41, 42, 43, 44, 45, 46, 47,
    # sheet_01 (ranks 48-95)
    48, 49, 50, 51, 53, 54, 55, 57, 58, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 81, 82, 83, 84, 86, 87, 89, 91, 92, 93, 94, 95,
    # sheet_02 (ranks 96-143)
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 110, 111, 112, 113,
    114, 115, 117, 118, 119, 120, 121, 123, 125, 126, 127, 128, 129, 131, 132,
    133, 134, 138, 139, 141, 143,
    # sheet_03 (ranks 144-191)
    144, 145, 146, 147, 148, 149, 150, 151, 153, 155, 156, 157, 159, 160, 161,
    164, 165, 166, 167, 168, 169, 170, 172, 175, 177, 178, 179, 182, 183, 184,
    186, 188, 189, 190,
    # sheet_04 (ranks 192-239)
    192, 194, 195, 196, 198, 199, 200, 201, 202, 204, 205, 206, 207, 208, 209,
    211, 212, 213, 214, 215, 216, 218, 220, 221, 222, 223, 225, 226, 229, 230,
    231, 232, 233, 234, 235, 236, 239,
]


def main() -> None:
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    selected = [(r, rows[r]) for r in ACCEPT_RANKS]
    copied = 0
    for rank, row in selected:
        src = RAW / row["file"]
        dst = OUT / row["file"]
        if not dst.exists():
            shutil.copy(src, dst)
            copied += 1
    with open(OUT / "manifest_selected.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "file", "colorfulness", "mean_saturation"])
        for rank, row in selected:
            w.writerow([rank, row["file"], row["colorfulness"], row["mean_saturation"]])
    print(f"selected {len(selected)} (copied {copied} new) -> {OUT}")


if __name__ == "__main__":
    main()
