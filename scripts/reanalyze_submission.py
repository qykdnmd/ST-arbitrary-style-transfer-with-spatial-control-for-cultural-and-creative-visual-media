"""Read-only reproduction of current six-method tables and poster intervals.

Run inside the accompanying code/data package. Original CSVs are never edited.
NumPy is required; SciPy/pandas additionally reproduce the 25 primary tests.
The report does not imply that GPU inference, retraining or mixed models ran.
"""
from __future__ import annotations
import argparse
import ast
import collections
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

METRICS = ('ssim', 'lpips', 'e_ssim', 'l_c', 'l_s')
METHODS = ('ccstytr', 'stytr2', 'adain', 'sanet', 'cast', 'aesfa')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def identity(row):
    return row.get('cell') or (row['content'] + '__' + row['style'] if 'content' in row else Path(row['output']).stem)


def close(a, b, atol=1e-11):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=atol)


def bootstrap(matrix):
    n, m = matrix.shape
    rng = np.random.default_rng(20261014)
    contents = rng.multinomial(n, np.full(n, 1 / n), size=10000)
    styles = rng.multinomial(m, np.full(m, 1 / m), size=10000)
    samples = np.einsum('bi,ij,bj->b', contents, matrix, styles, optimize=True) / (n * m)
    return np.percentile(samples, [2.5, 97.5])


def analyze(root):
    ev = root / 'paper/evidence'
    poster = ev / 'poster_aesfa_20261014'
    checks = {}
    combined = rows(ev / 'metrics_per_pair_full.csv') + rows(ev / 'aesfa_extension_metrics_per_pair_full.csv')
    index = {}
    for row in combined:
        key = (row['method'], identity(row))
        if key in index:
            assert all(float(index[key][m]) == float(row[m]) for m in METRICS)
        index[key] = row
    ids = {cell for method, cell in index if method == 'ccstytr'}
    checks['general_six_matched_800_pair_grids'] = len(index) == 4800 and len(ids) == 800 and all({cell for method, cell in index if method == m} == ids for m in METHODS)
    means = {r['method']: r for r in rows(ev / 'general_benchmark_six_method_means.csv')}
    recalculated = {}
    for method in METHODS:
        group = [index[method, key] for key in sorted(ids)]
        recalculated[method] = {m: math.fsum(float(r[m]) for r in group) / len(group) for m in METRICS}
    checks['general_means_reproduced'] = all(close(v, means[method][m + '_mean']) for method, values in recalculated.items() for m, v in values.items())
    tests = rows(ev / 'general_benchmark_25_tests.csv')
    checks['primary_family_is_25'] = len(tests) == len({(r['baseline'], r['metric']) for r in tests}) == 25 and all(int(r['family_size']) == 25 for r in tests)
    checks['bonferroni_adjustments_reproduced'] = all(close(min(1, 25 * float(r['raw_p'])), r['bonferroni_adjusted_p'], atol=0) for r in tests)
    checks['all_25_adjusted_p_below_005'] = all(float(r['bonferroni_adjusted_p']) < .05 for r in tests)
    wilcoxon_report = {'executed': False}
    try:
        import pandas as pd
        from scipy.stats import wilcoxon
        frames = {n: pd.read_csv(ev / n) for n in ('metrics_per_pair_full.csv', 'aesfa_extension_metrics_per_pair_full.csv')}
        computed = []
        for r in tests:
            frame = frames['aesfa_extension_metrics_per_pair_full.csv' if r['baseline'] == 'aesfa' else 'metrics_per_pair_full.csv']
            ref = frame[frame.method == 'ccstytr'].set_index('cell')[r['metric']].sort_index()
            base = frame[frame.method == r['baseline']].set_index('cell')[r['metric']].sort_index()
            diff = ref - base if r['metric'] in ('ssim', 'e_ssim') else base - ref
            result = wilcoxon(diff.to_numpy(), zero_method='wilcox', correction=False, alternative='two-sided', method='asymptotic')
            computed.append({'baseline': r['baseline'], 'metric': r['metric'], 'statistic_matches': close(result.statistic, r['w_statistic']),
                             'raw_p_matches': math.isclose(float(result.pvalue), float(r['raw_p']), rel_tol=1e-8),
                             'adjusted_p': min(1, 25 * float(result.pvalue))})
        wilcoxon_report = {'executed': True, 'tests': computed}
        checks['all_25_wilcoxon_tests_reproduced'] = all(r['statistic_matches'] and r['raw_p_matches'] and r['adjusted_p'] < .05 for r in computed)
    except ImportError:
        wilcoxon_report['reason'] = 'SciPy/pandas unavailable; no p-value recomputation is claimed.'

    image_rows = rows(poster / 'reference_metrics_per_pair.csv') + rows(poster / 'metrics_per_pair_full.csv')
    image_map = {(r['method'], identity(r)): r for r in image_rows}
    contents = sorted({r['content'] for r in image_rows})
    styles = sorted({r['style'] for r in image_rows})
    cells = {c + '__' + s for c in contents for s in styles}
    checks['poster_six_complete_grids'] = len(contents) == 120 and len(styles) == 30 and len(image_map) == len(image_rows) == 21600 and all({k for m, k in image_map if m == method} == cells for method in METHODS)
    ocr_rows = rows(poster / 'reference_ocr.csv') + rows(poster / 'ocr.csv')
    ocr_map = {(r['method'], identity(r)): r for r in ocr_rows}
    eligible = sorted({r['content'] for r in ocr_rows})
    ocr_cells = {c + '__' + s for c in eligible for s in styles}
    checks['ocr_six_matched_3570_pair_grids'] = len(eligible) == 119 and len(ocr_map) == len(ocr_rows) == 21420 and all({k for m, k in ocr_map if m == method} == ocr_cells for method in METHODS)
    archived_means = {r['method']: r for r in rows(poster / 'six_method_summary.csv')}
    poster_means = {}
    for method in METHODS:
        poster_means[method] = {m: math.fsum(round(float(image_map[method, k][m]), 4) if method == 'aesfa' else float(image_map[method, k][m]) for k in sorted(cells)) / 3600 for m in METRICS}
        poster_means[method]['ocr_sim'] = math.fsum(float(ocr_map[method, k]['ocr_sim']) for k in sorted(ocr_cells)) / 3570
    checks['poster_means_reproduced'] = all(close(v, archived_means[method][m + '_mean']) for method, values in poster_means.items() for m, v in values.items())
    intervals = []
    for row in rows(poster / 'aesfa_paired_crossed_bootstrap.csv'):
        metric = row['metric']
        if metric == 'ocr_sim':
            levels = eligible
            matrix = np.array([[float(ocr_map['ccstytr', c + '__' + s][metric]) - float(ocr_map['aesfa', c + '__' + s][metric]) for s in styles] for c in levels])
        else:
            matrix = np.array([[float(image_map['ccstytr', c + '__' + s][metric]) - round(float(image_map['aesfa', c + '__' + s][metric]), 4) for s in styles] for c in contents])
            if metric in ('lpips', 'l_c', 'l_s'):
                matrix = -matrix
        low, high = bootstrap(matrix)
        intervals.append({'metric': metric, 'low': float(low), 'high': float(high), 'mean': float(matrix.mean()),
                          'matches_archive': close(low, row['crossed_bootstrap_ci95_low']) and close(high, row['crossed_bootstrap_ci95_high']) and close(matrix.mean(), row['mean_paired_difference'])})
    checks['six_10000_replicate_poster_intervals_reproduced'] = len(intervals) == 6 and all(r['matches_archive'] for r in intervals)

    art = rows(root / 'results/artfid5000/summary.csv')
    checks['six_finite_artfid_scores_reconstructed'] = len(art) == 6 and all(close((1 + float(r['lpips_content'])) * (1 + float(r['fid_finite'])), r['artfid_finite']) for r in art)
    csd = rows(root / 'results/artfid5000/csd_per_pair.csv')
    checks['six_5000_pair_csd_grids'] = len(csd) == 30000 and all(len({r['pair_id'] for r in csd if r['method'] == m}) == 5000 for m in METHODS)
    checks['csd_means_reproduced'] = all(close(math.fsum(float(r['csd_similarity']) for r in csd if r['method'] == m) / 5000, next(r for r in art if r['method'] == m)['csd_similarity_mean']) for m in METHODS)
    spatial = json.loads((ev / 'spatial_control_provenance.json').read_text())
    checks['spatial_result_hashes_match_archived_protocol'] = all(hashlib.sha256((root / 'results/spatial_control' / n).read_bytes()).hexdigest() == h for n, h in spatial['result_files'].items())
    syntax_files = [p for folder in ('ccstytr', 'scripts', 'tests') for p in (root / folder).rglob('*.py')]
    syntax_errors = []
    for path in syntax_files:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.relative_to(root).as_posix())
        except SyntaxError as error:
            syntax_errors.append(str(error))
    checks['python_syntax'] = not syntax_errors
    return {'pass': all(checks.values()), 'checks': checks, 'general_means': recalculated, 'poster_means': poster_means,
            'poster_intervals': intervals, 'wilcoxon': wilcoxon_report, 'python_syntax_files': len(syntax_files),
            'syntax_errors': syntax_errors, 'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy') if importlib.util.find_spec(p) is not None},
            'not_executed': ['GPU inference', 'model training', 'PyTorch/pytest model unit tests', 'statsmodels mixed-effects refits', 'FID features recomputation from absent images']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k in ('pass', 'checks', 'python_syntax_files', 'not_executed')}, indent=2))
    raise SystemExit(0 if report['pass'] else 1)
