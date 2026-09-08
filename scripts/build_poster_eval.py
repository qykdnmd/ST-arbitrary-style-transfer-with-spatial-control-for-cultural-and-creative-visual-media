"""Curate the featured content test set from the extracted PKU-PosterLayout
posters (experiment protocol filtering protocol, adapted: PKU is the primary
source; Crello rendering is deprioritized per PROGRESS notes).

Steps:
1. keep posters with >= 1 logo element. This deliberately reproduces the
   already-published benchmark membership: the original extractor labelled
   PKU class 1 as text, while the official mapping defines class 1 as logo;
2. dHash de-duplication (hamming <= 4 on 64-bit dHash);
3. stratified sample 120 across annotated-logo-density tiers (seed 2026);
4. record sizes so the paper can state the min-edge>=512 / upscale policy
   (run_baselines.py square-resizes everything to 512 anyway).

Outputs data/eval/poster_curated/ + manifest.csv.
"""

import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("data/eval/poster")
OUT = Path("data/eval/poster_curated")
N_TARGET = 120
SEED = 2026
DUP_THRESH = 4


def dhash(path: Path) -> np.ndarray:
    g = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, :-1] > a[:, 1:]).flatten()  # 64 bits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SRC / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["n_logo_elements"]) > 0]
    print(f"posters with annotated logos: {len(rows)}")

    hashes = []
    kept = []
    for r in rows:
        h = dhash(SRC / r["file"])
        if any(np.count_nonzero(h != oh) <= DUP_THRESH for oh in hashes):
            continue
        hashes.append(h)
        kept.append(r)
    print(f"after dedup: {len(kept)}")

    # Logo-density tiers define the sample composition. The count distribution is
    # heavily skewed towards 1, so terciles collapse); shortfalls are filled
    # from the remaining pools in order of size.
    def tier(r):
        n = int(r["n_logo_elements"])
        return 0 if n == 1 else (1 if n <= 3 else 2)

    rng = random.Random(SEED)
    pools = []
    for t in range(3):
        pool = [r for r in kept if tier(r) == t]
        rng.shuffle(pool)
        pools.append(pool)
    picked = []
    for t, pool in enumerate(pools):
        picked.extend(pool[: N_TARGET // 3])
    short = N_TARGET - len(picked)
    if short > 0:
        rest = [r for pool in pools for r in pool if r not in picked]
        rng.shuffle(rest)  # uniform fill, no tier bias
        picked.extend(rest[:short])
    sizes = [len(p) for p in pools]
    print(f"picked: {len(picked)} (tier pools {sizes})")

    out_rows = []
    for r in picked:
        shutil.copy(SRC / r["file"], OUT / r["file"])
        short = min(int(r["width"]), int(r["height"]))
        out_rows.append(
            (
                r["file"],
                r["width"],
                r["height"],
                r["n_elements"],
                r["n_text_elements"],
                r["n_logo_elements"],
                r.get("text_boxes", "[]"),
                r.get("logo_boxes", "[]"),
                tier(r),
                short >= 512,
            )
        )
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
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
                "logo_tier",
                "short_edge_ge_512",
            ]
        )
        w.writerows(out_rows)
    ge = sum(1 for r in out_rows if r[-1])
    print(f"-> {OUT} ({ge}/{len(out_rows)} with short edge >= 512)")


if __name__ == "__main__":
    main()
