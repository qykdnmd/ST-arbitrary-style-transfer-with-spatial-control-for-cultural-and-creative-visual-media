"""Recheck primary paired tests without requiring GPU or statsmodels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from statistical_analysis import load_and_validate, wilcoxon_statistics

ROOT=Path(__file__).resolve().parents[1]

def main():
    frame=load_and_validate(ROOT/'paper/evidence/metrics_per_pair_full.csv')
    actual,_=wilcoxon_statistics(frame)
    expected=pd.read_csv(ROOT/'paper/evidence/wilcoxon_pairwise.csv')
    joined=actual.merge(expected,on=['baseline','metric'],suffixes=('_new','_old'))
    checks={key:bool(np.allclose(joined[key+'_new'],joined[key+'_old'],rtol=1e-9,atol=0)) for key in ['w_statistic','raw_p','bonferroni_adjusted_p','rank_biserial_r']}
    result={'observations':len(frame),'comparisons':len(joined),'checks':checks,'all_bonferroni_significant':bool((actual.bonferroni_adjusted_p<.05).all()),'numpy_version':np.__version__,'pandas_version':pd.__version__,'scipy_version':scipy.__version__,'scope':'Paired Wilcoxon and rank-biserial effects only; mixed effects and bootstrap not rerun'}
    (ROOT/'provenance/statistical_recheck.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if len(joined)!=20 or not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
