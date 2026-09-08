"""Build the general-purpose eval set (experiment protocol):
data/eval/general/content/  40 images sampled from COCO val2017 (never seen in
                            training by any method: all baselines train on
                            COCO train2014/train2017, disjoint from val2017)
data/eval/general/style/    20 WikiArt images, one per art movement (top-20
                            movements by local image count)

Selection is deterministic (seed 2026) and recorded in manifest.csv.

COCO val2017 file list comes from the public S3 bucket listing of
images.cocodataset.org (bucket listing enabled); images are fetched
individually so the 1 GB zip is not needed.
"""

import csv
import random
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SEED = 2026
N_CONTENT = 40
N_STYLE = 20
WIKIART = Path("data/train/wikiart")
OUT = Path("data/eval/general")
# NOTE: plain http — the local proxy MITMs TLS and breaks cert validation on https.
S3 = "http://images.cocodataset.org/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
UA = {"User-Agent": "Mozilla/5.0"}


def list_val2017_keys() -> list[str]:
    keys, token = [], None
    while True:
        url = f"{S3}?list-type=2&prefix=val2017/&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            root = ET.fromstring(r.read())
        keys.extend(k.text for k in root.iter(f"{NS}Key"))
        nxt = root.find(f"{NS}NextContinuationToken")
        if nxt is None:
            break
        token = nxt.text
    return sorted(keys)


def main() -> None:
    rng = random.Random(SEED)
    (OUT / "content").mkdir(parents=True, exist_ok=True)
    (OUT / "style").mkdir(parents=True, exist_ok=True)

    keys = list_val2017_keys()
    print(f"val2017 total: {len(keys)}")
    chosen_c = rng.sample(keys, N_CONTENT)

    rows = []
    for key in chosen_c:
        name = Path(key).name
        dst = OUT / "content" / name
        if not dst.exists():
            req = urllib.request.Request(S3 + key, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
        rows.append(("content", name, "", S3 + key))
    print(f"content done: {len(chosen_c)}")

    movements = sorted(
        (d for d in WIKIART.iterdir() if d.is_dir()),
        key=lambda d: -len(list(d.iterdir())),
    )[:N_STYLE]
    for mv in movements:
        pics = sorted(p for p in mv.iterdir() if p.suffix.lower() in {".jpg", ".png"})
        pick = rng.choice(pics)
        dst = OUT / "style" / f"{mv.name}__{pick.name}"
        shutil.copy(pick, dst)
        rows.append(("style", dst.name, mv.name, str(pick)))
    print(f"style done: {len(movements)}")

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "movement", "source"])
        w.writerows(rows)
    print("manifest ->", OUT / "manifest.csv")


if __name__ == "__main__":
    main()
