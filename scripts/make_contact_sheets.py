"""Generic labeled contact sheets for a directory of images (visual audit).

Usage:
    python scripts/make_contact_sheets.py --src data/eval/styles/chinese_landscape \
        --out sanity_check/audit_landscape
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--thumb", type=int, default=128)
    ap.add_argument("--glob", default="*")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.glob(args.glob) if p.suffix.lower() in IMAGE_EXTS)
    per = args.cols * args.rows
    label = 16
    n_sheets = (len(files) + per - 1) // per
    for si in range(n_sheets):
        sheet = Image.new("RGB", (args.cols * args.thumb, args.rows * (args.thumb + label)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, p in enumerate(files[si * per:(si + 1) * per]):
            idx = si * per + j
            img = Image.open(p).convert("RGB").resize((args.thumb, args.thumb), Image.LANCZOS)
            x, y = (j % args.cols) * args.thumb, (j // args.cols) * (args.thumb + label)
            sheet.paste(img, (x, y))
            draw.text((x + 2, y + args.thumb + 1), f"#{idx} {p.stem[:16]}", fill="black")
        dst = out / f"sheet_{si:02d}.png"
        sheet.save(dst)
        print("wrote", dst)
    print(f"{len(files)} files over {n_sheets} sheets -> {out}")


if __name__ == "__main__":
    main()
