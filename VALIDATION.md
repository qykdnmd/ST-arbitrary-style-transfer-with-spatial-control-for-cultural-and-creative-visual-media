# Validation of this submission package

Prepared 2026-10-25. Checks were run using Python 3.13.9; exact
package versions are in `provenance/package_validation.json`.

- CPU unit tests: 36 passed, no failures/errors/skips in the selected suite.
  The separate historical `tests/test_finite_release.py` was explicitly excluded:
  it requires an omitted upstream ArtFID snapshot and its omitted historical
  redaction manifest. It is not counted as passed or made to pass by removing checks.
- Current numerical audit: all checks passed, including actual recomputation of
  all 25 Wilcoxon tests and Bonferroni adjustments, six 10,000-replicate poster
  intervals, current six-method means, CSD means, finite-ArtFID composition and
  spatial-result file hashes. See `provenance/numerical_reanalysis_20261025.json`.
- Training, single-pair inference, paired inference and metric CLI help: passed.
  Both current table renderers ran successfully.
- Python syntax: 89 source files parsed successfully.
- Original scientific implementation/configuration/measurement files: unchanged
  relative to the author-prepared source archive. Changed original files are
  documentation or dependency lists and are enumerated in the validation record.
- Checkpoint integrity: 0/5 complete local files
  verified; companion prepared: false.

No GPU inference, author-model inference on the original image grids, model
retraining, fresh Inception/CSD feature extraction or mixed-effects refitting
was performed. CPU tests use synthetic tensors, not experimental image outputs.
Successful reanalysis is not a claim of complete end-to-end reproduction.

The SciPy lower bound is 1.15 because the audit uses `method="asymptotic"` in
`scipy.stats.wilcoxon`. Official API documentation:
https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.stats.wilcoxon.html
