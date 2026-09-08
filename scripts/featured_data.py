"""Read the featured comparison using its explicit configuration sources."""
import csv
from pathlib import Path
from model_names import PROPOSED

ROOT=Path(__file__).resolve().parents[1]
BASELINES={'stytr2','adain','sanet','cast'}

def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))

def pair_key(row):
    if 'output' in row:return Path(row['output']).stem
    return row['content']+'__'+row['style']

def featured_rows(filename):
    if filename not in {'metrics_per_pair.csv','ocr.csv'}:raise ValueError(filename)
    baseline=read_rows(ROOT/'results/featured'/filename)
    if {r['method'] for r in baseline}!=BASELINES:raise ValueError('Expected four featured baselines')
    proposed=[{**r,'method':'ccstytr'} for r in read_rows(ROOT/'results/ablation'/filename) if r['method']==PROPOSED]
    if not proposed:raise ValueError('Missing proposed-model records')
    rows=baseline+proposed
    keys={}
    for row in rows:
        key=pair_key(row);method=row['method']
        keys.setdefault(method,set())
        if key in keys[method]:raise ValueError('Duplicate method/pair: '+method+'/'+key)
        keys[method].add(key)
    if any(value!=keys['ccstytr'] for value in keys.values()):raise ValueError('Unmatched featured pairs')
    return rows
