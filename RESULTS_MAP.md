# Models and experimental materials

| Configuration identifier | Manuscript display name |
|---|---|
| residual_gating_spatial_alpha | Residual saliency gating + spatial α (proposed model) / CC-StyTr (Main) |
| saliency_routed_spatial_alpha | Saliency-routed attention + spatial α |
| no_structure_guidance | No structure guidance |
| saliency_routed_scalar_alpha | Saliency-routed attention + scalar α |
| residual_gating_scalar_alpha | Residual saliency gating + scalar α |

Configurations: `configs/ablation/<configuration_id>.yaml`. Runtime checkpoints: `checkpoints/ablation/<configuration_id>/ccstytr_step_080000.pth`. All five configurations use 80,000 optimizer steps. Binary checkpoint hashes are recorded in `provenance/checkpoints.json`.

| Experiment | Data location and scope |
|---|---|
| General benchmark, 40×20 | `paper/evidence/metrics_per_pair_full.csv`: 4,000 rows across five methods. The same directory provides the 20 primary paired comparisons and dependence-aware analyses. |
| Featured comparison, 120×30 | Four baselines are in `results/featured/`. Proposed-model records are identified by residual_gating_spatial_alpha in `results/ablation/`. The reader, `scripts/featured_data.py`, verifies matching pair identifiers. |
| Five-configuration ablation | `results/ablation/metrics_per_pair.csv`, `ocr.csv`, and `summary.csv`: 18,000 image-metric rows; OCR denominators depend on eligible text-bearing samples. |
| AesFA extension | `results/aesfa_extension/` and `paper/evidence/aesfa_extension_*`: a separate five-test family. |
| 5,000 pairs, six methods | `data/eval/artfid5000/manifest.csv` and `results/artfid5000/`: an independent ArtFID/CSD protocol. |
| Spatial control, 60×10×7 | `data/eval/spatial_control/` and `results/spatial_control/`: global, local, monotonicity, and OCR records. |
| Repeated stylization | `paper/evidence/repeated_stylization_proposed_*`: the proposed model's 20-round trajectory only; these records do not support cross-method comparisons. |
| Qualitative comparison | `paper/figures/` and the qualitative sample and text/logo crop manifests in the evidence directory. |

The general benchmark uses `ccstytr` as the proposed model's method key. Ablations use functional configuration identifiers. Data sources, comparison conditions, metric directions, and sample counts must not be conflated.

Current 5,000-pair table: `results/artfid5000/summary.csv` (mirrored at `paper/evidence/artfid5000_summary.csv`). Finite-run evidence: `paper/evidence/artfid5000_finite_20260908/`. Combined-table lineage: `results/artfid5000/current_provenance.json`. Superseded infinity-labeled results: `results/artfid5000/historical_infinity/`. The current provenance files link to the immutable finite-run evidence and the historical CSD source.
