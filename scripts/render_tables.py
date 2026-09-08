"""Render primary means and the documented Bonferroni-corrected comparisons."""
import argparse
import csv
from pathlib import Path
from model_names import BASELINES

ROOT=Path(__file__).resolve().parents[1]

def rows(name):
    with (ROOT/'paper/evidence'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/main/table_main.md')
    args=parser.parse_args()
    means={r['method']:r for r in rows('evaluation_means_full.csv')}
    lines=['# General benchmark: 40 contents × 20 styles','',
           '| Method | SSIM ↑ | LPIPS ↓ | E-SSIM ↑ | Content loss ↓ | Style loss ↓ |',
           '|---|---|---|---|---|---|']
    for method in ['ccstytr','stytr2','adain','sanet','cast']:
        values=[float(means[method][k+'_mean']) for k in ['ssim','lpips','e_ssim','l_c','l_s']]
        lines.append('| '+BASELINES[method]+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['','## Paired comparisons','',
            'Two-sided Wilcoxon tests, Bonferroni correction across 20 comparisons. Positive rank-biserial effects favor CC-StyTr.','',
            '| Baseline | Metric | Adjusted p | Rank-biserial effect |','|---|---|---|---|']
    for r in rows('wilcoxon_pairwise.csv'):
        lines.append(f"| {BASELINES[r['baseline']]} | {r['metric']} | {float(r['bonferroni_adjusted_p']):.3e} | {float(r['rank_biserial_r']):.4f} |")
    lines+=['','Dependence-aware intervals and auxiliary recognition tests are supplied separately in paper/evidence/.',
            'ArtFID/CSD uses the separate 5000-pair protocol; see results/artfid5000/summary.csv.','']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines),encoding='utf-8')
    print('Saved',args.output)

if __name__=='__main__':main()
