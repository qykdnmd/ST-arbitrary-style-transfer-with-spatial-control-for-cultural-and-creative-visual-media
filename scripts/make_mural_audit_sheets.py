"""Build labeled contact sheets of the top-N MuralDH candidates for the visual
audit (experiment protocol: pick 150-200 intact murals from the ranked manifest).

Each sheet is a grid of 128px thumbnails with the candidate's rank index and
file name printed under it, so a subset can be selected by listing ranks.
"""

import csv
from pathlib import Path

from PIL import Image, ImageDraw

MANIFEST = Path("data/eval/styles/mural/manifest_ranked.csv")
RAW = Path("data/sources/mural512")
OUT = Path("sanity_check/mural_audit")
TOP_N = 240
COLS, ROWS = 6, 8
THUMB = 128
LABEL = 16


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))[:TOP_N]
    per_sheet = COLS * ROWS
    n_sheets = (len(rows) + per_sheet - 1) // per_sheet
    for si in range(n_sheets):
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * (THUMB + LABEL)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(rows[si * per_sheet:(si + 1) * per_sheet]):
            rank = si * per_sheet + j
            img = Image.open(RAW / r["file"]).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
            x, y = (j % COLS) * THUMB, (j // COLS) * (THUMB + LABEL)
            sheet.paste(img, (x, y))
            draw.text((x + 2, y + THUMB + 1), f"#{rank} {r['file'][:18]}", fill="black")
        dst = OUT / f"sheet_{si:02d}.png"
        sheet.save(dst)
        print("wrote", dst)
    print(f"{len(rows)} candidates over {n_sheets} sheets -> {OUT}")


if __name__ == "__main__":
    main()
