"""Build the Western-museum-painting style subset from The Met Open Access (CC0).

Pipeline (experiment protocol: 100-200 oil paintings/prints covering strongly
differentiated movements such as the Dutch Golden Age and Impressionism):

1. Stream-parse data/sources/MetObjects.csv (317 MB, ~490k rows); keep rows
   that are public domain AND in a painting/drawing department AND look like
   paintings/prints by medium.
2. Stratified sample across century buckets (seed 2026) for movement diversity.
3. Query collectionapi.metmuseum.org per object for primaryImageSmall and
   download the image (512-px class is enough; run_baselines re-sizes anyway).

Outputs data/eval/styles/western_painting/ + manifest.csv.
"""

import csv
import json
import random
import time
import urllib.request
from pathlib import Path

CSV = Path("data/sources/MetObjects.csv")
OUT = Path("data/eval/styles/western_painting")
API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
UA = {"User-Agent": "Mozilla/5.0"}
SEED = 2026
TARGET_PER_CENTURY = 40  # 17C/18C/19C + early-20C -> ~160 candidates

DEPTS = {"European Paintings", "American Paintings", "Drawings and Prints"}
MEDIUM_KEYS = ("oil on", "tempera on", "watercolor", "gouache", "pastel", "engraving",
               "etching", "lithograph", "woodcut")
CENTURIES = [(1600, 1700), (1700, 1800), (1800, 1900), (1900, 1930)]


def century_of(begin: str) -> int | None:
    try:
        y = int(begin)
    except ValueError:
        return None
    for i, (lo, hi) in enumerate(CENTURIES):
        if lo <= y < hi:
            return i
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    buckets: list[list[dict]] = [[] for _ in CENTURIES]
    with open(CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["Is Public Domain"] != "True":
                continue
            if row["Department"] not in DEPTS:
                continue
            med = row["Medium"].lower()
            if not any(k in med for k in MEDIUM_KEYS):
                continue
            c = century_of(row["Object Begin Date"])
            if c is None:
                continue
            buckets[c].append({
                "id": row["Object ID"], "title": row["Title"],
                "artist": row["Artist Display Name"], "date": row["Object Date"],
                "dept": row["Department"], "medium": row["Medium"],
            })
    for i, b in enumerate(buckets):
        print(f"century {CENTURIES[i]}: {len(b)} candidates")

    picks = []
    for i, b in enumerate(buckets):
        rng.shuffle(b)
        picks.extend(b[:TARGET_PER_CENTURY])
    print(f"total picks: {len(picks)}")

    rows = []
    for j, p in enumerate(picks):
        fname = f"met_{p['id']}.jpg"
        dst = OUT / fname
        url = ""
        try:
            req = urllib.request.Request(API.format(p["id"]), headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                meta = json.load(r)
            url = meta.get("primaryImageSmall") or meta.get("primaryImage") or ""
            if url and (not dst.exists() or dst.stat().st_size == 0):
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                    f.write(r.read())
                time.sleep(0.15)
        except Exception as e:  # noqa: BLE001
            print("FAIL", p["id"], e)
            continue
        if url:
            rows.append((fname, p["id"], p["title"], p["artist"], p["date"],
                         p["dept"], p["medium"], url))
        if (j + 1) % 20 == 0:
            print(f"{j + 1}/{len(picks)} done, {len(rows)} with image", flush=True)

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "object_id", "title", "artist", "date", "dept",
                    "medium", "image_url"])
        w.writerows(rows)
    print(f"downloaded {len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
