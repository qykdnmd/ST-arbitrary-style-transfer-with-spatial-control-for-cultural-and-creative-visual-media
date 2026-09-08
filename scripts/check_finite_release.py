"""Read-only checks joining the current table to immutable evaluation evidence."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check(root=ROOT):
    root = Path(root)
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    def require(ok, message):
        if not ok: raise ValueError(message)
    def read_rows(p):
        with p.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        require(len(rows) == 6 and len({r["method"] for r in rows}) == 6, "Expected six unique methods")
        return {r["method"]: r for r in rows}
    run = root / "paper/evidence/artfid5000_finite_20260908"
    p = json.loads((run / "provenance.json").read_text())
    redactions = json.loads((root / "provenance/public_redactions.json").read_text())
    for item in redactions["files"]:
        require(sha(root / item["path"]) == item["public_sha256"], "Public derivative hash: " + item["path"])
    public_report = json.loads((run / "verification.json").read_text())
    require(public_report["public_derivative"]["public_run_provenance_sha256"] == sha(run / "provenance.json"), "Public provenance link")
    require(public_report["run_provenance_sha256"] == p["public_derivative"]["original_sha256"], "Original provenance link")
    current = root / "results/artfid5000/summary.csv"
    require(current.read_bytes() == (root / "paper/evidence/artfid5000_summary.csv").read_bytes(), "Summary mirrors differ")
    require(p["status"] == "complete" and p["estimator"] == "finite_sample_artfid", "Incomplete or wrong estimator")
    require(sha(run / "summary.json") == p["summary_sha256"], "Finite summary hash")
    require(sha(run / "image_inventory.json") == p["image_inventory_sha256"], "Inventory hash")
    history = root / "results/artfid5000/historical_infinity"
    require(sha(history / "summary.csv") == p["historical_summary_sha256"], "Historical summary changed")
    require(sha(history / "provenance.json") == p["historical_provenance_sha256"], "Historical provenance changed")
    require(sha(run / "source_snapshot/eval_artfid_finite.py") == p["script_sha256"], "Original evaluator snapshot hash")
    for name, expected in p["upstream_source_sha256"].items():
        source = run / "source_snapshot/upstream" / Path(name.replace("\\", "/"))
        require(sha(source) == expected, "Upstream snapshot hash: " + name)
    lineage = json.loads((root / "results/artfid5000/current_provenance.json").read_text())
    for rel in ["results/artfid5000/provenance.json", "paper/evidence/artfid5000_provenance.json"]:
        require(json.loads((root / rel).read_text()) == lineage, "Current provenance mirrors differ")
    require(sha(current) == lineage["combined_summary_sha256"], "Combined summary hash")
    require(sha(run / "summary.json") == lineage["finite_summary_sha256"], "Combined finite source hash")
    require(sha(history / "summary.csv") == lineage["historical_summary_sha256"], "Combined CSD summary source hash")
    require(sha(root / "results/artfid5000/csd_per_pair.csv") == lineage["csd_per_pair_sha256"], "CSD source hash")
    rows = read_rows(current)
    old = read_rows(history / "summary.csv")
    finite = json.loads((run / "summary.json").read_text())
    require(set(rows) == {r["method"] for r in finite} == set(old), "Method mismatch")
    for source in finite:
        row = rows[source["method"]]
        require(int(row["n"]) == source["n"] == 5000, "Sample count")
        for key in ["fid_finite", "lpips_content", "artfid_finite"]:
            require(float(row[key]) == source[key], "Finite value differs: " + key)
        require(math.isclose(float(row["artfid_finite"]), (1 + float(row["lpips_content"])) * (1 + float(row["fid_finite"])), abs_tol=1e-12), "ArtFID formula")
        for key in ["csd_similarity_mean", "csd_ci95_low", "csd_ci95_high"]:
            require(row[key] == old[source["method"]][key], "CSD value changed: " + key)
    return {"status": "passed", "methods": len(rows), "n_per_method": 5000, "scope": "Current table, source hashes, source snapshots and unchanged CSD; no GPU recomputation"}

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
