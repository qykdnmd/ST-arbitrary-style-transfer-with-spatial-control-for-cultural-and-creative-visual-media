# Current manuscript evidence map

| Analysis | Current evidence | Scope |
|---|---|---|
| General means | `paper/evidence/general_benchmark_six_method_means.csv` | Six methods, 800 matched pairs each |
| General tests | `paper/evidence/general_benchmark_25_tests.csv` | One primary Bonferroni family of 25 tests |
| General source rows | `metrics_per_pair_full.csv` and `aesfa_extension_metrics_per_pair_full.csv` in `paper/evidence/` | Deduplicate the identical CC-StyTr reference rows; 4,800 method-pair records |
| Poster means/OCR | `paper/evidence/poster_aesfa_20261014/` | Six methods; 3,600 image pairs and 3,570 OCR pairs each |
| Ablation | `results/ablation/` | Five functional configurations; same archived measurements |
| ArtFID/CSD | `results/artfid5000/summary.csv` | Six methods × 5,000; ordinary finite-sample ArtFID |
| Spatial control | `results/spatial_control/` | 60 contents × 10 styles × 7 conditions |
| Historical sensitivity | `mixed_effects.csv`, `crossed_bootstrap.csv` in `paper/evidence/` | Four original baselines, 20 models; not refitted for AesFA |

Older `wilcoxon_pairwise.csv`, `evaluation_means_full.csv`,
`aesfa_extension_pairwise.csv`, and any old rendered five-method tables are
historical source evidence, not the current primary test family. The two
renderers under `scripts/historical/` are retained only as historical source;
use the top-level `scripts/render_tables.py` and `render_featured_tables.py`.
Historical infinity-labeled ArtFID files are not the reported finite estimates.
Do not rerun old collection/migration scripts without adapting their documented
filesystem and resource prerequisites.

Display names: general benchmark `CC-StyTr (Main)`; ablation
`Residual saliency gating + spatial α (proposed model)`; prose `the proposed model`.
Checkpoint directories use functional configuration identifiers, not internal
experiment versions. Checkpoint hashes are in `provenance/checkpoints.json`.
