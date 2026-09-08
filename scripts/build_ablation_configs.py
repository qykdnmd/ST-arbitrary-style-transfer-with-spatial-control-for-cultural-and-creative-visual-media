"""Read or export the five packaged, equal-budget experiment configurations."""
import argparse
import copy
from pathlib import Path
import yaml
from model_names import MODELS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'configs/ablation'
VARIANTS=MODELS
ORDER=list(MODELS)

def build(name):
    if name not in MODELS:raise ValueError('Unknown configuration: '+name)
    with (OUT/(name+'.yaml')).open(encoding='utf-8') as handle:
        config=yaml.safe_load(handle)
    if config['train']['max_steps']!=80000:raise ValueError('Expected 80000 optimizer steps')
    return copy.deepcopy(config)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    for name in ORDER:
        cfg=build(name)
        if args.output_dir:
            args.output_dir.mkdir(parents=True,exist_ok=True)
            target=args.output_dir/(name+'.yaml')
            if target.exists():raise FileExistsError(target)
            target.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        print(name+': '+MODELS[name])

if __name__=='__main__':main()
