"""Regenerate release checksums after intentional edits; not a validation step."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"provenance/artifact_manifest.json", "provenance/release_files.sha256",
            "provenance/verification.json", "provenance/finite_recheck.json"}
IGNORED = {".git", "__pycache__", ".pytest_cache", ".venv", ".local-agent-runtime"}
def main():
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in IGNORED for part in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDED or path.suffix in {".pyc", ".pth", ".pt", ".part"}:
            continue
        if rel.startswith(("results/recomputed_statistics/", "results/artfid5000_finite_NEW_RUN/", "data/train/", "runs/", "logs/")):
            continue
        files.append({"path": rel, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = {"scope": "Code, configuration, documentation and study data; checkpoint hashes are separate",
            "exclusions": sorted(EXCLUDED), "files": files}
    (ROOT / "provenance/artifact_manifest.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (ROOT / "provenance/release_files.sha256").write_text("".join(r["sha256"] + "  " + r["path"] + "\n" for r in files), encoding="utf-8")
    print(json.dumps({"files": len(files), "bytes": sum(r["bytes"] for r in files)}))
if __name__ == "__main__":
    main()
