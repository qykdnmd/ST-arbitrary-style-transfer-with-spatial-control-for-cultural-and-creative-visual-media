"""Dependency-free verification of migrated artifacts; does not rerun models."""
from __future__ import annotations
import ast
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from model_names import MODELS, PROPOSED

def digest(path):
    checksum=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):checksum.update(block)
    return checksum.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir',type=Path,default=ROOT.parent/'CC-StyTr_model_weights')
    parser.add_argument('--skip-weights',action='store_true')
    args=parser.parse_args()
    errors=[]
    report={'checked_at':datetime.now(timezone.utc).isoformat(),'scope':'artifact integrity, row counts, means, syntax; not GPU reproduction'}
    meta=json.loads((ROOT/'provenance/artifact_manifest.json').read_text())
    for row in meta['files']:
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path)!=row['sha256']:
            errors.append('artifact hash: '+row['path'])
    report['artifact_files_verified']=len(meta['files'])
    provenance=json.loads((ROOT/'paper/evidence/spatial_control_provenance.json').read_text())
    report['spatial_result_hashes']={}
    for name,expected in provenance['result_files'].items():
        ok=digest(ROOT/'results/spatial_control'/name)==expected
        report['spatial_result_hashes'][name]=ok
        if not ok:errors.append('spatial result hash: '+name)
    cp=ROOT/'provenance/checkpoints.json'
    checkpoints=json.loads(cp.read_text()) if cp.exists() else []
    report['weights_check_skipped']=args.skip_weights
    for row in checkpoints:
        path=ROOT/row['path']
        if not path.exists():path=args.weights_dir/Path(row['path']).relative_to('checkpoints')
        if not args.skip_weights and (not path.is_file() or digest(path)!=row['sha256']):errors.append('checkpoint hash or missing weights: '+row['path'])
        if row['model_id']==PROPOSED and row['sha256']!=provenance['checkpoint']['sha256']:errors.append('proposed checkpoint differs from protocol')
    if {r['model_id'] for r in checkpoints}!=set(MODELS):errors.append('checkpoint model registry')
    report['checkpoints_verified']=[] if args.skip_weights else checkpoints
    report['expected_final_checkpoints']=5
    report['checkpoint_set_verified']=not args.skip_weights and len(checkpoints)==5 and not errors
    report['csv']={}
    inputs=['paper/evidence/metrics_per_pair_full.csv','results/aesfa_extension/metrics_per_pair_full.csv','results/featured/metrics_per_pair.csv','results/featured/ocr.csv','results/ablation/metrics_per_pair.csv','results/ablation/ocr.csv','results/artfid5000/csd_per_pair.csv','results/spatial_control/global_per_pair.csv','results/spatial_control/monotonicity_per_pair.csv','results/spatial_control/spatial_locality_per_pair.csv','data/eval/artfid5000/manifest.csv']
    all_rows={}
    for rel in inputs:
        with (ROOT/rel).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        all_rows[rel]=rows
        methods=Counter(r.get('method',r.get('condition','all')) for r in rows)
        means={}
        for method in methods:
            group=[r for r in rows if r.get('method',r.get('condition','all'))==method]
            means[method]={k:math.fsum(float(r[k]) for r in group)/len(group) for k in ['ssim','lpips','e_ssim','l_c','l_s','ocr_sim'] if k in group[0]}
        report['csv'][rel]={'rows':len(rows),'columns':list(rows[0]),'groups':dict(methods),'means':means}
    expected={'paper/evidence/metrics_per_pair_full.csv':4000,'results/featured/metrics_per_pair.csv':14400,'results/ablation/metrics_per_pair.csv':18000,'results/spatial_control/global_per_pair.csv':3000,'data/eval/artfid5000/manifest.csv':5000}
    for rel,n in expected.items():
        if len(all_rows[rel])!=n:errors.append('row count: '+rel)
    main_rows=all_rows['paper/evidence/metrics_per_pair_full.csv']
    if len({(r['method'],r['cell']) for r in main_rows})!=4000:errors.append('duplicate primary method/cell')
    report['main_means_match_manuscript']={}
    for k,v in {'ssim':0.5704,'lpips':0.4817,'e_ssim':0.5640,'l_c':1.5508,'l_s':0.9800}.items():
        actual=report['csv']['paper/evidence/metrics_per_pair_full.csv']['means']['ccstytr'][k]
        ok=abs(actual-v)<=0.00005001
        report['main_means_match_manuscript'][k]=ok
        if not ok:errors.append('main mean: '+k)
    report['alpha_maps']=sum(1 for p in (ROOT/'data/eval/spatial_control').glob('alpha_*/*.png'))
    if report['alpha_maps']!=120:errors.append('alpha map count')
    syntax=[]
    for folder in ['ccstytr','scripts','tests']:
        syntax.extend((ROOT/folder).rglob('*.py'))
    syntax += [ROOT/'train.py',ROOT/'infer.py']
    for path in syntax:
        try:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        except SyntaxError as exc:errors.append(str(exc))
    report['python_files_syntax_checked']=len(syntax)
    report['known_limitations']=['No GPU inference or training performed by this check','Featured/ablation measurements rounded to four decimals','Third-party weights and source/generated images not bundled','No permanent archive DOI yet; external resource access links remain incomplete']
    from check_finite_release import check
    try:
        report['finite_artfid'] = check(ROOT)
    except (ValueError, OSError, KeyError) as exc:
        errors.append('finite ArtFID: ' + str(exc))
    report['errors']=errors
    out=ROOT/'provenance/verification.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'errors':errors,'artifact_files_verified':report['artifact_files_verified'],'checkpoint_set_verified':report['checkpoint_set_verified'],'alpha_maps':report['alpha_maps'],'main_means_match_manuscript':report['main_means_match_manuscript'],'python_files_syntax_checked':len(syntax)},indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
