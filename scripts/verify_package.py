"""Verify the distributed files and, optionally, all five author checkpoints.

Standard-library only. Checksums establish identity, not scientific validity.
The code/data result and weight result are deliberately separate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or ':' in relative or '\\' in relative:
        raise ValueError('Unsafe manifest path: ' + relative)
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Manifest path escapes root: ' + relative)
    return path


def verify(root, require_weights=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    code = []
    seen = set()
    for row in manifest['files']:
        if row['path'] in seen:
            raise ValueError('Duplicate manifest path: ' + row['path'])
        seen.add(row['path'])
        path = safe_path(root, row['path'])
        exists = path.is_file()
        size_ok = exists and path.stat().st_size == row['bytes']
        hash_ok = size_ok and sha256_file(path) == row['sha256']
        code.append({'path': row['path'], 'exists': exists, 'size_matches': size_ok, 'sha256_matches': hash_ok})
    weights = []
    for row in json.loads((root / 'provenance/checkpoints.json').read_text(encoding='utf-8')):
        path = safe_path(root, row['path'])
        exists = path.is_file()
        size_ok = exists and path.stat().st_size == row['bytes']
        hash_ok = size_ok and sha256_file(path) == row['sha256']
        weights.append({'model_id': row['model_id'], 'path': row['path'], 'exists': exists,
                        'size_matches': size_ok, 'sha256_matches': hash_ok})
    code_ok = bool(code) and all(r['sha256_matches'] for r in code)
    weights_ok = len(weights) == 5 and all(r['sha256_matches'] for r in weights)
    return {'checked_utc': datetime.now(timezone.utc).isoformat(),
            'code_data_integrity_pass': code_ok, 'listed_file_count': len(code),
            'code_data_failures': [r for r in code if not r['sha256_matches']],
            'weights_required': require_weights, 'all_five_weights_verified': weights_ok,
            'checkpoints': weights, 'pass': code_ok and (weights_ok or not require_weights),
            'scope': 'File sizes and SHA256 only; no model loading, inference, training or feature recomputation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--require-weights', action='store_true', help='Fail unless all five complete checkpoint hashes match')
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    report = verify(args.root, args.require_weights)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
