"""Render the current six-method general means and unified 25-test family."""
import csv
from pathlib import Path
import argparse
ROOT = Path(__file__).resolve().parents[1]
NAMES = {'ccstytr': 'CC-StyTr (Main)', 'stytr2': 'StyTr²', 'adain': 'AdaIN', 'sanet': 'SANet', 'cast': 'CAST', 'aesfa': 'AesFA'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    def rows(name):
        with (ROOT / 'paper/evidence' / name).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    means = {r['method']: r for r in rows('general_benchmark_six_method_means.csv')}
    lines = ['# General benchmark: 40 contents × 20 styles', '', '| Method | SSIM ↑ | LPIPS ↓ | E-SSIM ↑ | Content loss ↓ | Style loss ↓ |', '|---|---|---|---|---|---|']
    for m, name in NAMES.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f"{float(means[m][k + '_mean']):.4f}" for k in ('ssim','lpips','e_ssim','l_c','l_s')) + ' |')
    lines += ['', '## Primary paired comparisons', '', 'One Bonferroni family of 25 tests; positive differences favor CC-StyTr.', '', '| Baseline | Metric | Adjusted p | Rank-biserial effect |', '|---|---|---|---|']
    for r in rows('general_benchmark_25_tests.csv'):
        lines.append(f"| {NAMES[r['baseline']]} | {r['metric']} | {float(r['bonferroni_adjusted_p']):.3e} | {float(r['rank_biserial_r']):.4f} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
