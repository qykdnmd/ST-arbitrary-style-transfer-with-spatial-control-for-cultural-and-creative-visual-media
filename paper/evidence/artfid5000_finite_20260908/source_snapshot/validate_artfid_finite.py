"""Validate a completed finite ArtFID evidence bundle before manuscript use."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--lpips-weight", type=Path, action="append", default=[])
    a = p.parse_args()
    provenance = json.loads((a.run / "provenance.json").read_text())
    assert provenance["status"] == "complete", "Run is not complete"
    assert provenance["estimator"] == "finite_sample_artfid"
    assert sha(a.run / "summary.json") == provenance["summary_sha256"]
    assert sha(a.run / "image_inventory.json") == provenance["image_inventory_sha256"]
    assert sha(a.historical / "summary.csv") == provenance["historical_summary_sha256"]
    assert sha(a.historical / "provenance.json") == provenance["historical_provenance_sha256"]
    rows = json.loads((a.run / "summary.json").read_text())
    methods = {"ccstytr", "aesfa", "stytr2", "adain", "sanet", "cast"}
    assert len(rows) == 6 and {r["method"] for r in rows} == methods
    inventory = json.loads((a.run / "image_inventory.json").read_text())
    assert len(inventory) == 40000
    ids = {}
    for item in inventory:
        ids.setdefault(item["group"], []).append(item["pair_id"])
    assert set(ids) == methods | {"style", "content"}
    for values in ids.values():
        assert len(values) == len(set(values)) == 5000
        assert set(values) == set(ids["content"])
    with (a.historical / "summary.csv").open(newline="", encoding="utf-8") as f:
        old = {r["method"]: r for r in csv.DictReader(f)}
    with (a.historical / "csd_per_pair.csv").open(newline="", encoding="utf-8") as f:
        csd = list(csv.DictReader(f))
    assert len(csd) == 30000
    report = []
    for r in rows:
        assert r["n"] == 5000
        for k in ["fid_finite", "lpips_content", "artfid_finite"]:
            assert math.isfinite(r[k]) and r[k] >= 0
        assert math.isclose(r["artfid_finite"], (1+r["lpips_content"])*(1+r["fid_finite"]), abs_tol=1e-12)
        assert math.isclose(r["fid_finite"], r["fid_independent"], abs_tol=1e-5, rel_tol=1e-6)
        values = [v for v in csd if v["method"] == r["method"]]
        assert len(values) == 5000
        assert len({v["pair_id"] for v in values}) == 5000
        assert {v["pair_id"] for v in values} == set(ids["content"])
        mean = math.fsum(float(v["csd_similarity"]) for v in values)/5000
        assert math.isclose(mean, float(old[r["method"]]["csd_similarity_mean"]), abs_tol=1e-12)
        report.append(dict(r, csd_similarity_mean=mean,
                           lpips_change_from_historical=r["lpips_content"]-float(old[r["method"]]["lpips_content"])))
    result = dict(status="passed", checked_methods=6, checked_input_records=40000,
                  checked_csd_pairs=30000,
                  csd_source_sha256=sha(a.historical / "csd_per_pair.csv"),
                  run_provenance_sha256=sha(a.run / "provenance.json"),
                  lpips_weight_sha256={str(v): sha(v) for v in a.lpips_weight},
                  results=report,
                  note="Old infinity-labeled scores are preserved, not validated as infinity estimates. CSD is reused and its per-pair mean is checked; ArtFID and LPIPS are freshly recomputed.")
    a.report.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
