"""Render current six-method poster/OCR means from the frozen summary."""
from pathlib import Path
import csv
import argparse
from render_tables import NAMES
ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with (ROOT / 'paper/evidence/poster_aesfa_20261014/six_method_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = {r['method']: r for r in csv.DictReader(f)}
    lines = ['# Poster benchmark: 120 contents × 30 styles', '', 'OCR: 119 eligible posters × 30 styles per method. Means use four-decimal pairwise values. AesFA preprocessing differs geometrically; see the protocol.', '', '| Method | SSIM ↑ | LPIPS ↓ | E-SSIM ↑ | Content loss ↓ | Style loss ↓ | OCR-sim ↑ |', '|---|---|---|---|---|---|---|']
    for method, name in NAMES.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f"{float(rows[method][m + '_mean']):.4f}" for m in ('ssim','lpips','e_ssim','l_c','l_s','ocr_sim')) + ' |')
    lines += ['', 'AesFA paired crossed-bootstrap intervals: paper/evidence/poster_aesfa_20261014/aesfa_paired_crossed_bootstrap.csv. The style-loss interval includes zero; intervals are marginal, not simultaneous.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')

if __name__ == '__main__':
    main()
