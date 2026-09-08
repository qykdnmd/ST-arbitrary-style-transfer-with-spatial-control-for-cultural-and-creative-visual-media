"""Download CC0 Chinese paintings from the Cleveland Museum of Art Open Access API
(no key required, CC0) as the Chinese-landscape style source.

Pulls all culture=China & type=Painting & has_image records (web-size images),
saves them under data/eval/styles/chinese_painting/ plus a manifest for the
landscape sub-filtering step (title keywords).
"""

import csv
import json
import time
import urllib.request
from pathlib import Path

API = ("https://openaccess-api.clevelandart.org/api/artworks"
       "?cc0=1&has_image=1&culture=China&type=Painting&limit=100&skip={skip}")
OUT = Path("data/eval/styles/chinese_painting")
UA = {"User-Agent": "Mozilla/5.0"}

OUT.mkdir(parents=True, exist_ok=True)

records = []
skip = 0
while True:
    req = urllib.request.Request(API.format(skip=skip), headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    batch = d["data"]
    if not batch:
        break
    records.extend(batch)
    skip += len(batch)
    print(f"fetched {skip}/{d['info']['total']}", flush=True)
    if skip >= d["info"]["total"]:
        break
    time.sleep(0.5)

rows = []
for a in records:
    url = (a.get("images") or {}).get("web", {}).get("url")
    if not url:
        continue
    fname = f"{a['id']}.jpg"
    dst = OUT / fname
    if not dst.exists() or dst.stat().st_size == 0:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
        except Exception as e:  # noqa: BLE001
            print("FAIL", a["id"], e)
            continue
        time.sleep(0.2)
    rows.append((fname, a["title"] or "", a.get("creation_date") or "", url))

with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "title", "creation_date", "source_url"])
    w.writerows(rows)

print(f"downloaded {len(rows)} images -> {OUT}")
