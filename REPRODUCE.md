# Reproduction guide

Run all commands from the package root. `README.md` gives installation, source-data
layout, complete training/inference examples and expected outputs.

1. Run `python scripts/verify_package.py --report outputs/package_integrity.json`.
2. Install NumPy, pandas and SciPy (or `requirements-statistics.txt`) and run
   `python scripts/reanalyze_submission.py --root . --report outputs/numerical_audit.json`.
   Require both `pass: true` and `wilcoxon.executed: true` for full numerical audit.
3. Render `outputs/general_tables.md` and `outputs/poster_tables.md` using the two
   current renderers listed in the README; compare with `RESULTS_MAP.md`.
4. Install the companion author weights and run the strict check in `WEIGHTS.md`.
5. Obtain permitted source images and pinned third-party dependencies as needed.
   Follow the specific protocol's manifest, preprocessing and weight identity.
6. Use `infer.py` for one pair or `scripts/infer_paired.py` for a selected grid.
   Model training uses `train.py` and the five archived functional YAML files.

The current general comparison has six methods and one 25-comparison primary
Bonferroni family. Twenty historical mixed-effects models were not extended to
AesFA. Poster AesFA bootstrap intervals are a separate marginal sensitivity
analysis; the style-loss interval includes zero. Poster data informed model
selection and are not independent held-out validation. CSD and finite ArtFID need
their original upstream implementations and weights for feature-level reproduction.

Source/measurement integrity, CPU unit tests, checkpoint integrity, synthetic
inference, original-image inference and retraining are different verification
levels. Only checks actually executed are reported in `VALIDATION.md` and
`provenance/package_validation.json`. Numerical reanalysis alone is not an exact
end-to-end rerun. No original image collections or full generated-output grids are
included. Historical collection scripts may require omitted upstream resources
and original staging layouts; they are retained as provenance, not turnkey commands.
