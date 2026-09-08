"""ArtFID per method (experiment protocol): ArtFID = (1 + LPIPS(out, content)) *
(1 + FID(out, style set)), official implementation in external/art-fid.

Protocol adaptations for the 40x20 grid:
- FID reference = the 20 preprocessed 512x512 style images (persistent
  style_512 dir regenerated here if missing; small-N FID is high variance,
  noted in the paper as a limitation).
- compute_content_distance needs 1:1 sorted pairing, so each method gets a
  temp mirror dir where the 512x512 content image is hard-linked under the
  exact output file name (<c>__<s>.<ext>); sorted orders then align.

Usage:
    python scripts/eval_artfid.py --results_dir results/main \
        --content_dir data/eval/general/content --style_dir data/eval/general/style
"""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/art-fid"))

from art_fid import art_fid  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def ensure_prepped(src: Path, size: int) -> Path:
    """Persistent 512x512 square copy of a source dir (same as run_baselines)."""
    dst = src.parent / f"{src.name}_{size}"
    if dst.exists() and len(list(dst.iterdir())) > 0:
        return dst
    from PIL import Image
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        img.save(dst / f"{p.stem}.png")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=25)
    args = ap.parse_args()

    content512 = ensure_prepped(Path(args.content_dir), args.size)
    style512 = ensure_prepped(Path(args.style_dir), args.size)
    c_by_stem = {p.stem: p for p in content512.iterdir()}

    rows = []
    for mdir in sorted(Path(args.results_dir).iterdir()):
        if not mdir.is_dir():
            continue
        outs = [p for p in sorted(mdir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if not outs:
            continue
        with tempfile.TemporaryDirectory() as td:
            mirror = Path(td)
            for out in outs:
                c_stem = out.stem.partition("__")[0]
                src = c_by_stem.get(c_stem)
                if src is None:
                    continue
                dst = mirror / f"{out.stem}.png"
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copy(src, dst)
            print(f"=== {mdir.name}: {len(outs)} outputs ===", flush=True)
            fid = art_fid.compute_fid(str(mdir), str(style512),
                                      args.batch_size, "cuda", 2)
            dist = art_fid.compute_content_distance(
                str(mdir), str(mirror), args.batch_size, "lpips", "cuda", 2)
            value = (1 + float(dist)) * (1 + float(fid))
        rows.append((mdir.name, len(outs), round(float(fid), 4),
                     round(float(dist), 4), round(value, 4)))
        print(rows[-1], flush=True)

    csv_path = Path(args.results_dir) / "artfid.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n", "fid", "lpips_mean", "artfid"])
        w.writerows(rows)
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
