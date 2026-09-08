"""CPU-only checks for the manuscript-facing release layout."""
import csv
import json
import math
from pathlib import Path
import re
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from model_names import MODELS,PROPOSED
from build_ablation_configs import build
from featured_data import featured_rows

class ReleaseLayoutTests(unittest.TestCase):
    def test_configuration_registry(self):
        names={p.stem for p in (ROOT/'configs/ablation').glob('*.yaml')}
        self.assertEqual(names,set(MODELS))
        enabled={s.strip() for s in (ROOT/'configs/ablation/enabled.txt').read_text().splitlines() if s.strip() and not s.startswith('#')}
        self.assertEqual(enabled,names)
        for name in names:
            cfg=build(name)
            self.assertEqual(cfg['train']['max_steps'],80000)
            self.assertEqual(cfg['train']['ckpt_dir'],'checkpoints/ablation/'+name)
            self.assertEqual(cfg['train']['batch_size']*cfg['train']['grad_accum'],16)

    def test_checkpoint_registry(self):
        rows=json.loads((ROOT/'provenance/checkpoints.json').read_text())
        self.assertEqual({r['model_id'] for r in rows},set(MODELS))
        for row in rows:self.assertEqual(Path(row['path']).parent.name,row['model_id'])
        main=next(r for r in rows if r['model_id']==PROPOSED)
        protocol=json.loads((ROOT/'paper/evidence/spatial_control_provenance.json').read_text())
        self.assertEqual(main['sha256'],protocol['checkpoint']['sha256'])

    def test_featured_pairs_and_means(self):
        rows=featured_rows('metrics_per_pair.csv');ocr=featured_rows('ocr.csv')
        self.assertEqual(len(rows),18000);self.assertEqual(len(ocr),17850)
        main=[r for r in rows if r['method']=='ccstytr']
        for key,value in {'ssim':.5394,'lpips':.5218,'e_ssim':.4795,'l_c':1.6570,'l_s':1.7904}.items():
            mean=math.fsum(float(r[key]) for r in main)/len(main)
            self.assertLessEqual(abs(mean-value),.00005001)
        scores=[float(r['ocr_sim']) for r in ocr if r['method']=='ccstytr']
        self.assertLessEqual(abs(math.fsum(scores)/len(scores)-.5906),.00005001)

    def test_ablation_data_identifiers(self):
        for name in ['metrics_per_pair.csv','summary.csv','ocr.csv']:
            with (ROOT/'results/ablation'/name).open(encoding='utf-8',newline='') as f:
                self.assertEqual({r['method'] for r in csv.DictReader(f)},set(MODELS))

    def test_functional_names_in_public_text(self):
        pattern=re.compile(r'\bv[0-9]+[a-z]?_[a-z]',re.I)
        for path in ROOT.rglob('*'):
            if not path.is_file() or 'external' in path.relative_to(ROOT).parts or '__pycache__' in path.parts:continue
            self.assertIsNone(pattern.search(path.relative_to(ROOT).as_posix()),str(path))
            if path.suffix in {'.py','.json','.csv','.md','.yaml','.toml','.txt','.ps1','.sh'}:
                self.assertIsNone(pattern.search(path.read_text(encoding='utf-8-sig')),str(path))

if __name__=='__main__':unittest.main()
