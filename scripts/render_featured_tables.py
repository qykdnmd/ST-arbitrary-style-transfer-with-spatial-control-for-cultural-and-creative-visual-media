"""Render descriptive featured-set means from the documented pair records."""
import argparse
import math
from pathlib import Path
from featured_data import ROOT,featured_rows
from model_names import BASELINES

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/featured/table_featured.md')
    args=parser.parse_args()
    metrics=featured_rows('metrics_per_pair.csv');ocr=featured_rows('ocr.csv')
    lines=['# Featured comparison','',
           'Means from 120 posters × 30 styles per method; OCR uses matched text-bearing pairs.',
           'Four baseline sources: results/featured/. Proposed-model source: results/ablation/.',
           'Measurements are rounded to four decimals. No additional inferential tests are reported here.','',
           '| Method | SSIM ↑ | LPIPS ↓ | E-SSIM ↑ | Content loss ↓ | Style loss ↓ | OCR-sim ↑ |',
           '|---|---|---|---|---|---|---|']
    for method in ['ccstytr','stytr2','adain','sanet','cast']:
        group=[r for r in metrics if r['method']==method]
        text=[r for r in ocr if r['method']==method]
        values=[math.fsum(float(r[k]) for r in group)/len(group) for k in ['ssim','lpips','e_ssim','l_c','l_s']]
        values.append(math.fsum(float(r['ocr_sim']) for r in text)/len(text))
        lines.append('| '+BASELINES[method]+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved',args.output)

if __name__=='__main__':main()
