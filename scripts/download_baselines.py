"""Download baseline checkpoints via hf-mirror (huggingface.co is unreachable).

StyTr2 official weights mirror: datnguyentien204/Sty_TR2_38 (experiments/).
vgg_normalised.pth is shared by StyTr2 (loss net) and AdaIN.
"""

import os
import urllib.request

BASE = "https://hf-mirror.com/datnguyentien204/Sty_TR2_38/resolve/main/experiments"
DEST = "external/StyTR-2/experiments"
FILES = [
    "transformer_iter_160000.pth",
    "decoder_iter_160000.pth",
    "embedding_iter_160000.pth",
    "vgg_normalised.pth",
]

os.makedirs(DEST, exist_ok=True)
for f in FILES:
    dst = os.path.join(DEST, f)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("skip (exists)", f)
        continue
    print("downloading", f, flush=True)
    urllib.request.urlretrieve(BASE + "/" + f, dst)
    print("  ->", round(os.path.getsize(dst) / 1e6, 1), "MB")
print("done")
