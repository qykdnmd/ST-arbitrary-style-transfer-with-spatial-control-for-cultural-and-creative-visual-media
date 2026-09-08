"""Verify separately stored checkpoints and optionally copy to runtime paths."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    base=args.source.resolve()
    rows=json.loads((ROOT/'provenance/checkpoints.json').read_text())
    planned=[]
    for row in rows:
        rel=Path(row['path'])
        source=(base/rel.relative_to('checkpoints')).resolve()
        target=(ROOT/rel).resolve()
        if not source.is_relative_to(base) or not target.is_relative_to(ROOT/'checkpoints'):
            raise ValueError('Checkpoint path escapes its permitted directory')
        if not source.is_file() or sha(source)!=row['sha256']:
            raise ValueError('Missing or mismatched source: '+str(source))
        if target.exists() and sha(target)!=row['sha256']:
            raise FileExistsError('Refusing overwrite of different checkpoint: '+str(target))
        planned.append((source,target,row['sha256']))
    for source,target,expected in planned:
        if not args.verify_only and not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive creation refuses even a file created after preflight.
            with source.open('rb') as src,target.open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            if sha(target)!=expected:raise RuntimeError('Copy verification failed: '+str(target))
    print(('Verified ' if args.verify_only else 'Restored/verified ')+str(len(planned))+' checkpoints.')

if __name__=='__main__':main()
