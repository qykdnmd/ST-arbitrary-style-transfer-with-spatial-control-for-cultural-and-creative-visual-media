"""Extract MuralDH/Mural512 from data/sources/MuralDH.zip and rank images by a
damage proxy (experiment protocol: keep 150-200 intact, non-severely-damaged
murals as the Dunhuang style subset).

Damage proxy: deteriorated mural regions are desaturated / whitish / flaked, so
intact murals score higher on colorfulness (Hasler-Suesstrunk) and saturation.
We extract all 512x512 murals to data/sources/mural512/ and write
data/eval/styles/mural/manifest_ranked.csv sorted by score desc; the final
subset is picked from the top after a visual audit.
"""

import csv
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ZIP = Path("data/sources/MuralDH.zip")
RAW = Path("data/sources/mural512")
OUT = Path("data/eval/styles/mural")
PREFIX = "MuralDH/Mural512/"


def colorfulness(arr: np.ndarray) -> float:
    """Hasler-Suesstrunk colorfulness metric."""
    r, g, b = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    names = [n for n in z.namelist() if n.startswith(PREFIX) and not n.endswith("/")]
    print(f"mural512 entries: {len(names)}")

    rows = []
    for i, n in enumerate(names):
        dst = RAW / Path(n).name
        if not dst.exists():
            dst.write_bytes(z.read(n))
        if (i + 1) % 500 == 0:
            print(f"extracted {i + 1}/{len(names)}", flush=True)
        with Image.open(dst) as img:
            arr = np.asarray(img.convert("RGB"))
            hsv = np.asarray(img.convert("HSV"))
        rows.append((Path(n).name, arr.shape[1], arr.shape[0],
                     round(colorfulness(arr), 2),
                     round(float(hsv[..., 1].mean()), 2)))

    rows.sort(key=lambda r: -r[3])
    with open(OUT / "manifest_ranked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "width", "height", "colorfulness", "mean_saturation"])
        w.writerows(rows)
    print(f"extracted -> {RAW}; ranked manifest -> {OUT / 'manifest_ranked.csv'}")
    print("top5:", [r[0] for r in rows[:5]])
    print("bottom5:", [r[0] for r in rows[-5:]])


if __name__ == "__main__":
    main()
