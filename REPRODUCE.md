# Reproduction guide

Run commands from the project root. Use a dedicated Python environment:

```sh
python -m pip install -r requirements.txt
python -m pip install -r requirements-statistics.txt
python -m pip install -e .
```

Package versions from the evaluation environment are recorded in `provenance/environment.json`. This is an environment record, not a cross-platform lockfile. Install PyTorch/CUDA, OCR, and third-party dependencies appropriate for your platform.

## Checkpoints and inference

```sh
python scripts/restore_checkpoints.py --source ../CC-StyTr_model_weights
python infer.py --checkpoint checkpoints/ablation/residual_gating_spatial_alpha/ccstytr_step_080000.pth --content CONTENT_IMAGE --style STYLE_IMAGE --output output.png --alpha 1.0
```

Use `--alpha-map ALPHA_IMAGE` for regional control and `--alpha` for uniform control. Restoring checkpoints requires approximately 2.06 GB. The restoration script verifies each file and refuses to overwrite different content. For inference alone, you may point directly to a checkpoint in the separate weights directory.

## Training and configurations

Set `data.content_dir` and `data.style_dir` in the selected configuration to your local datasets. The recorded training paths identify the input locations and are not portable download addresses.

```sh
python train.py --config configs/ablation/residual_gating_spatial_alpha.yaml
python scripts/build_ablation_configs.py
python scripts/run_ablation.py --status
```

The configuration-checking entry point reads the five supplied configurations. Specify `--output-dir NEW_DIRECTORY` to export copies without overwriting existing files. Training budgets, loss weights, architecture parameters, and α settings are defined by each configuration.

## Artifact and statistical verification

```sh
python scripts/verify_release.py
python scripts/recheck_migrated_statistics.py
python scripts/render_tables.py
python scripts/render_featured_tables.py
```

Without checkpoints, use `verify_release.py --skip-weights`; the report explicitly marks the weight check as skipped. Verification covers file hashes, model identifiers, pair counts, metric means, and syntax. It does not retrain models or perform GPU inference. The statistical recheck recomputes the 20 Wilcoxon comparisons and effect sizes.

To rebuild the full statistical analysis:

```sh
python scripts/statistical_analysis.py --input paper/evidence/metrics_per_pair_full.csv --output-dir results/recomputed_statistics
```

This entry point also fits mixed-effects models and performs crossed bootstrap analysis, requiring the complete statistical dependencies. Do not overwrite the input evidence. AesFA, ArtFID/CSD, and spatial control have separate scripts and protocols; consult their `--help` output and provenance records. Recomputing metrics from images requires the source images and third-party weights.

Featured-set tables use `featured_data.py` to combine matched records for the four baselines and the proposed model. The reader does not impute missing data or add significance claims. Tests in the ablation table use saliency-routed attention with spatial α as their reference, whereas percentage changes in the ablation figure use StyTr². These references are distinct.

## Finite-sample ArtFID

The current table uses the 2026-09-08 finite run. Do not run the legacy infinity entry point. First validate the supplied evidence without images or GPU:

```sh
python scripts/validate_artfid_finite.py paper/evidence/artfid5000_finite_20260908 --historical results/artfid5000/historical_infinity --report provenance/finite_recheck.json
```

This checks archived records, not the absent source images or feature arrays. The original training-host verification report is retained unchanged.

For GPU recomputation, obtain the locked content/style images, restore all six generated-output directories under `results/artfid5000/`, and place the recorded official checkpoint at `external/art-fid/ckpt/art_inception.pth`. Install the ArtFID dependencies (including scikit-learn and LPIPS) in a CUDA environment matching the archived run as closely as possible. Then run:

```sh
python scripts/eval_artfid_finite.py --output results/artfid5000_finite_NEW_RUN --batch-size 16 --workers 4
python scripts/validate_artfid_finite.py results/artfid5000_finite_NEW_RUN --historical results/artfid5000/historical_infinity --report results/artfid5000_finite_NEW_RUN/verification.json
```

The new output directory must not exist. The portable wrapper uses the historical subdirectory and records vendored source hashes without requiring an upstream Git checkout. The exact training-run wrapper remains in the immutable source snapshot. GPU execution was not repeated during this release synchronization. CSD values and bootstrap intervals remain from their archived evaluation.

Run `python scripts/check_finite_release.py` to check the combined current table against its source hashes and exact finite-run source snapshots. `verify_release.py` includes the same check. The relocated legacy script is a historical reference, not a supported executable entry point.

After intentional release changes, run python scripts/refresh_release_manifest.py, then python scripts/verify_release.py (or --skip-weights). Refreshing hashes only records file state; it does not validate numerical results. Regenerated verification reports are excluded from the checksum manifest to avoid self-reference.

Public logs and machine metadata are privacy-redacted derivatives. Numerical results and exact evaluation source snapshots are unchanged. See [PUBLIC_REDACTION.md](PUBLIC_REDACTION.md) for original/public hash lineage.
