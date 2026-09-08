"""Download MuralDH.zip (7.7 GB, CC0) from Dryad with resume support.

Dryad file id 3347581, dataset doi:10.5061/dryad.bnzs7h4jd.
"""

import os
import time
import urllib.request

URL = "https://datadryad.org/api/v2/files/3347581/download"
DST = "data/sources/MuralDH.zip"
CHUNK = 1 << 20

os.makedirs(os.path.dirname(DST), exist_ok=True)

pos = os.path.getsize(DST) if os.path.exists(DST) else 0
headers = {"User-Agent": "Mozilla/5.0"}
if pos:
    headers["Range"] = f"bytes={pos}-"
    print(f"resuming at {pos / 1e9:.2f} GB")

for attempt in range(1, 6):
    try:
        req = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r, open(DST, "ab") as out:
            t0 = time.time()
            done = pos
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if time.time() - t0 > 30:
                    print(f"{done / 1e9:.2f} GB", flush=True)
                    t0 = time.time()
        break
    except Exception as e:  # noqa: BLE001
        print(f"attempt {attempt} failed at {os.path.getsize(DST) / 1e9:.2f} GB: {e}", flush=True)
        pos = os.path.getsize(DST)
        headers["Range"] = f"bytes={pos}-"
        time.sleep(10)

print("final size:", os.path.getsize(DST), "expected: 7738652652")
