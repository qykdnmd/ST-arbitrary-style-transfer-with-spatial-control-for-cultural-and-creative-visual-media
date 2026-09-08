"""Download SANet + CAST checkpoints via the GitHub contents API
(raw.githubusercontent.com is unreachable; api.github.com with
Accept: application/vnd.github.raw serves file bytes directly).

Sources (TVCG-2024 evaluation repo, original official checkpoints in git):
  AST/SANet/transformer_iter_500000.pth, decoder_iter_500000.pth  (dypark86 official)
  AST/UCAST/checkpoints/CAST_model/latest_net_{AE,Dec_A,Dec_B}.pth (CAST_mode: CAST)
vgg_normalised.pth is shared with StyTr2/AdaIN and copied locally.
"""

import os
import shutil
import time
import urllib.request

REPO = "ZhouZJ-DL/A-Comprehensive-Evaluation-of-Arbitrary-Image-Style-Transfer-Methods"
API = f"https://api.github.com/repos/{REPO}/contents"

JOBS = [
    ("AST/SANet/transformer_iter_500000.pth", "external/SANET/experiments/transformer_iter_500000.pth"),
    ("AST/SANet/decoder_iter_500000.pth", "external/SANET/experiments/decoder_iter_500000.pth"),
    ("AST/UCAST/checkpoints/CAST_model/latest_net_AE.pth", "external/CAST_pytorch/checkpoints/CAST_model/latest_net_AE.pth"),
    ("AST/UCAST/checkpoints/CAST_model/latest_net_Dec_A.pth", "external/CAST_pytorch/checkpoints/CAST_model/latest_net_Dec_A.pth"),
    ("AST/UCAST/checkpoints/CAST_model/latest_net_Dec_B.pth", "external/CAST_pytorch/checkpoints/CAST_model/latest_net_Dec_B.pth"),
]

# vgg_normalised.pth: identical file already fetched for StyTr2
VGG_SRC = "external/StyTR-2/experiments/vgg_normalised.pth"
VGG_COPIES = [
    "external/pytorch-AdaIN/models/vgg_normalised.pth",
    "external/SANET/experiments/vgg_normalised.pth",
    "external/CAST_pytorch/models/vgg_normalised.pth",
]

# AdaIN decoder.pth was fetched earlier from the naoto0804 GitHub release
ADAIN_MOVE = ("external/AdaIN/models/decoder.pth", "external/pytorch-AdaIN/models/decoder.pth")


def download(url: str, dst: str, retries: int = 3) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github.raw"}
            )
            with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            print("  ->", round(os.path.getsize(dst) / 1e6, 1), "MB")
            return
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"failed to download {url}")


for remote, dst in JOBS:
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print("skip (exists)", dst)
        continue
    print("downloading", remote, flush=True)
    download(f"{API}/{remote}", dst)

src, dst = ADAIN_MOVE
if os.path.exists(src) and not os.path.exists(dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print("moved", src, "->", dst)

for dst in VGG_COPIES:
    if not os.path.exists(dst) and os.path.exists(VGG_SRC):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(VGG_SRC, dst)
        print("copied vgg ->", dst)

print("done")
