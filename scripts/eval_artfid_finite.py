"""Recompute finite-sample ArtFID from the locked existing 5,000-pair outputs.

Never calls the infinity estimator or overwrites historical evidence. Uses the
official ArtFID feature extractor, preprocessing, LPIPS and Frechet routine.
Style features are extracted once and reused without changing the estimator.
An independent symmetric-PSD formulation checks each Frechet distance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

METHODS = ("ccstytr", "aesfa", "stytr2", "adain", "sanet", "cast")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def independent_fid(mu1, cov1, mu2, cov2):
    """Bures trace via a symmetric PSD square root, not sqrtm(cov1 @ cov2)."""
    vals, vecs = scipy.linalg.eigh((cov1 + cov1.T) / 2)
    if vals.min() < -1e-8:
        raise RuntimeError("Covariance is not positive semidefinite")
    root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
    middle = root @ cov2 @ root
    vals2 = scipy.linalg.eigvalsh((middle + middle.T) / 2)
    if vals2.min() < -1e-8:
        raise RuntimeError("Sandwich covariance is not positive semidefinite")
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(cov1) + np.trace(cov2)
                 - 2 * np.sqrt(np.maximum(vals2, 0)).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inventory-from", type=Path,
                        help="Reuse hashes from an interrupted preflight only when path, size and mtime match")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this scheduled experiment")
    torch.manual_seed(20260902)
    np.random.seed(20260902)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = root / "data/eval/artfid5000"
    results = root / "results/artfid5000"
    upstream = root / "external/art-fid"
    sys.path.insert(0, str(upstream))
    from art_fid import art_fid
    # Upstream '*.png' and '*.PNG' each enumerate all PNG files on Windows.
    # Replace enumeration only; keep feature extraction and metrics unchanged.
    def unique_image_paths(directory, sort=False):
        files = [str(p) for p in Path(directory).iterdir()
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        return sorted(files) if sort else files
    art_fid.get_image_paths = unique_image_paths
    import csv
    with (inputs / "manifest.csv").open(encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f))
    ids = [r["pair_id"] for r in manifest]
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise RuntimeError("Expected exactly 5,000 unique locked pairs")
    protocol = json.loads((inputs / "protocol.json").read_text(encoding="utf-8"))
    if protocol["n_pairs"] != 5000:
        raise RuntimeError("Protocol mismatch")
    directories = {"content": inputs / "content", "style": inputs / "style"}
    directories.update({method: results / method for method in METHODS})
    paths = {}
    inventory = []
    previous = {}
    if args.inventory_from:
        previous = {item["path"]: item for item in json.loads(args.inventory_from.read_text(encoding="utf-8"))}
    for name, directory in directories.items():
        files = art_fid.get_image_paths(str(directory), sort=True)
        stems = [Path(p).stem for p in files]
        if len(files) != 5000 or len(set(stems)) != 5000 or set(stems) != set(ids):
            raise RuntimeError(f"Incomplete or duplicated inputs: {name}")
        paths[name] = files
        print(f"Hashing {name}: {len(files)} images", flush=True)
        for p in files:
            stat = Path(p).stat()
            relative = str(Path(p).relative_to(root))
            old = previous.get(relative)
            unchanged = old and (old["size"], old["mtime_ns"]) == (stat.st_size, stat.st_mtime_ns)
            inventory.append(dict(group=name, pair_id=Path(p).stem,
                                  path=relative, size=stat.st_size,
                                  mtime_ns=stat.st_mtime_ns, sha256=old["sha256"] if unchanged else digest(p)))
    save(out / "image_inventory.json", inventory)
    checkpoint = upstream / "ckpt/art_inception.pth"
    historical_dir = results / "historical_infinity"
    historical = json.loads((historical_dir / "provenance.json").read_text(encoding="utf-8"))
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != historical["art_inception_sha256"]:
        raise RuntimeError("Feature checkpoint differs from historical evaluation")
    provenance = dict(
        status="running", started_utc=started, estimator="finite_sample_artfid",
        formula="(1 + lpips_content) * (1 + fid_finite)", n=5000,
        protocol=protocol, manifest_sha256=digest(inputs / "manifest.csv"),
        image_inventory_sha256=digest(out / "image_inventory.json"),
        preflight_inventory_source=str(args.inventory_from) if args.inventory_from else None,
        preflight_inventory_source_sha256=digest(args.inventory_from) if args.inventory_from else None,
        historical_provenance_sha256=digest(historical_dir / "provenance.json"),
        historical_summary_sha256=digest(historical_dir / "summary.csv"),
        script_sha256=digest(__file__), feature_checkpoint_sha256=checkpoint_hash,
        upstream_revision="vendored source; see upstream_source_sha256 and archived run revision",
        upstream_source_sha256={str(p.relative_to(upstream)): digest(p) for p in upstream.rglob("*.py")},
        python=sys.version, platform=platform.platform(), gpu=torch.cuda.get_device_name(0),
        packages={p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "scipy", "lpips"]},
        batch_size=args.batch_size, workers=args.workers, seed=20260902,
        feature_dtype="float32 inference; float64 mean/covariance", tf32=False,
        image_enumeration="Single directory pass; case-insensitive extension filter; unique stems checked against manifest. Avoids upstream Windows png/PNG duplicate enumeration.",
        verification="Independent symmetric-PSD Frechet formula; atol=1e-5, rtol=1e-6",
        note="Fresh finite-sample recomputation. No inverse-sample-size regression. Historical infinity-labeled files remain unchanged. CSD is not recomputed.")
    save(out / "provenance.json", provenance)
    model = art_fid.inception.Inception3().cuda()
    loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=False)
    # The official checkpoint omits unused classification heads. Feature-mode
    # forward returns before these heads, and eval disables auxiliary logits.
    allowed = {prefix + suffix for prefix in ["fc1", "fc2", "AuxLogits1.fc", "AuxLogits2.fc"]
               for suffix in [".weight", ".bias"]}
    if set(loaded.missing_keys) - allowed or loaded.unexpected_keys:
        raise RuntimeError(f"Feature checkpoint mismatch: {loaded}")
    provenance["unused_classifier_keys_missing"] = loaded.missing_keys
    save(out / "provenance.json", provenance)
    model.eval()

    def features(name):
        print(f"Extracting ArtFID features: {name}", flush=True)
        values = art_fid.get_activations(paths[name], model, args.batch_size, "cuda", args.workers)
        if values.shape != (5000, 2048) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid activations: {name}")
        np.save(out / (name + "_features.npy"), values)
        return values.mean(axis=0), np.cov(values, rowvar=False)

    style_mu, style_cov = features("style")
    rows = []
    for method in METHODS:
        mu, cov = features(method)
        finite = float(art_fid.compute_frechet_distance(mu, cov, style_mu, style_cov))
        verified = independent_fid(mu, cov, style_mu, style_cov)
        if not np.isfinite(finite) or finite < 0 or not np.isclose(finite, verified, atol=1e-5, rtol=1e-6):
            raise RuntimeError(f"FID verification failed: {method}: {finite}, {verified}")
        print(f"Recomputing official LPIPS: {method}", flush=True)
        content = float(art_fid.compute_content_distance(
            str(directories[method]), str(directories["content"]),
            batch_size=args.batch_size, content_metric="lpips", device="cuda", num_workers=args.workers))
        if not np.isfinite(content) or content < 0:
            raise RuntimeError("Invalid LPIPS")
        row = dict(method=method, n=5000, fid_finite=finite, lpips_content=content,
                   artfid_finite=(1 + content) * (1 + finite),
                   fid_independent=verified, fid_abs_check_error=abs(finite - verified))
        rows.append(row)
        save(out / "summary.json", rows)
        print(json.dumps(row), flush=True)
    for item in inventory:
        stat = (root / item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise RuntimeError("An input file changed during evaluation")
    provenance.update(status="complete", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      summary_sha256=digest(out / "summary.json"),
                      feature_sha256={p.name: digest(p) for p in out.glob("*_features.npy")})
    save(out / "provenance.json", provenance)
    print("FINITE ARTFID COMPLETE: " + str(out), flush=True)


if __name__ == "__main__":
    main()
