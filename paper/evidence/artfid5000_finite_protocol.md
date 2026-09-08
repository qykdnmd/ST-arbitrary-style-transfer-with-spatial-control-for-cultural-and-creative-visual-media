# Finite-sample ArtFID recomputation

## Scope

Re-evaluate all six methods on the existing, locked 5,000 one-to-one
content/style pairs, without retraining or regenerating outputs. The methods
are CC-StyTr, AesFA, StyTr2, AdaIN, SANet, and CAST. The feature checkpoint must
match the SHA-256 recorded in the original evaluation provenance.

The estimator is ordinary finite-sample ArtFID:

`ArtFID = (1 + mean LPIPS(output, content)) * (1 + FID(output, style))`.

Here FID uses the official ArtFID art-trained Inception feature space, not
an interchangeable ImageNet-FID implementation. Both image distributions
contain exactly 5,000 distinct manifest entries. There is no inverse-sample-
size regression, infinity extrapolation, or unbiasedness claim.

## Historical implementation issue

The upstream image enumerator separately globs `*.png` and `*.PNG`.
On the Windows training host, each pattern matches the same 5,000 PNGs,
producing 10,000 records for only 5,000 distinct files. This was confirmed
on the actual evaluation directories on 2026-09-08.

Consequently, the earlier diagnosis that every regression subsample was
necessarily 5,000 was incomplete: the actual upstream enumeration can make
the nominal subsample sizes vary from 5,000 to 10,000 using duplicated files.
This is not an extrapolation based on 10,000 independent images. Historical
results must not be relabeled as verified finite-sample results.

The new wrapper replaces only image enumeration with a single directory
pass, validates unique stems against the locked manifest, and retains the
official feature extractor, preprocessing, LPIPS implementation, and
Frechet-distance routine. Historical summary, provenance, and logs are
preserved unchanged.

## Verification and records

- Hash all 40,000 files: content, style, and six method-output sets.
- Recompute features from images, using float32 network inference with TF32
  disabled and float64 sample mean/covariance calculations.
- Extract reference-style features once and reuse their unchanged statistics.
- Check each FID with an independent symmetric positive-semidefinite
  formulation of the covariance trace term (`atol=1e-5`, `rtol=1e-6`).
- Recompute the official AlexNet-based LPIPS on exactly 5,000 aligned pairs
  per method; do not substitute the primary benchmark's VGG-based LPIPS.
- Preserve feature arrays, file hashes, source hashes, package versions,
  checkpoint identity, and new full-precision results in a new run directory.
- Check input sizes and modification times again at completion.
- Retain CSD from its separately archived per-pair evaluation. Its existing
  single-pass directory indexing is not affected by the PNG glob duplication.
- Update manuscript values and conclusions only after all six methods pass.

## Reproduction

From the training repository, use a CUDA-enabled environment with the
official ArtFID dependencies installed:

```text
python scripts/eval_artfid_finite.py --output results/artfid5000_finite_NEW_RUN --batch-size 16 --workers 4
```

The output directory must not already exist. `summary.json` is incremental;
only a `provenance.json` with `status: complete` denotes a finished run.
The prior `scripts/eval_artfid_csd.py` is retained as historical code, not the
recommended finite-sample evaluation entry point.

## Completed run (2026-09-08)

The training-host run `results/artfid5000_finite_20260908_verified` completed
successfully. Its compact evidence bundle is archived locally in
`paper/evidence/artfid5000_finite_20260908/`, including the full-precision
summary, provenance, file inventory, verification report, log, and exact
evaluation/upstream source snapshots. Feature arrays remain on the training
host; their SHA-256 values are recorded in the provenance.

All six methods passed the independent FID check; the largest absolute
difference was 3.6949e-13. The audit checked 40,000 input records and the
30,000 archived CSD pairs. CSD means agree with the historical summary.
Fresh LPIPS values differ from historical values by at most 6.4970e-6 and
retain the same four-decimal display. ArtFID scores were actually recomputed,
not obtained by relabeling the historical infinity estimates.

The manuscript now uses the verified finite-sample results. The original
historical summary/provenance files remain unchanged. The bilingual
`main_zh.md` was not synchronized during this TeX-focused update.
