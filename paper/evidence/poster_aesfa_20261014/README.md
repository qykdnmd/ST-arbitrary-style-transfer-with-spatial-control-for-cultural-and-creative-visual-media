# AesFA poster benchmark evaluation evidence

## Dataset and protocol

The evaluation retains the existing 120 PKU PosterLayout content images and
30 cultural heritage reference styles: 10 MuralDH, 10 Cleveland Museum of Art,
and 10 Metropolitan Museum of Art images. Each method uses the same complete
Cartesian grid of 3,600 content-style pairs. Source images must be obtained
under their original access and licensing conditions.

AesFA uses the archived official checkpoint, without retraining, with an
inference seed of 20261014 and 512-square outputs. Its official short-edge
resize and center crop differ from the proposed model's direct square resize.
Thus comparisons against original-poster references include preprocessing
geometry and possible removal of text, not architecture alone.

The five existing methods' measurements are archived references. A fixed
regression check reproduced five image metrics and OCR for 30 distinct pairs
per method (150 existing outputs). The proposed-model references correspond
to the preselected residual-saliency-gating/spatial-alpha checkpoint. The
four baseline records are unchanged. Source hashes and mappings are retained
in the provenance record; raw archives are not overwritten.

## Files

- `metrics_per_pair_full.csv`: 3,600 new AesFA image-metric records, retaining
  full precision.
- `ocr.csv`: 3,570 new AesFA OCR records, retaining both full and four-decimal
  scores and the number of reference tokens.
- `reference_metrics_per_pair.csv` and `reference_ocr.csv`: five-method
  archived references with stable pair IDs and no private filesystem paths.
- `six_method_summary.csv`: six-method means. Image/OCR comparative means use
  four-decimal per-pair measurements, matching the existing poster tables.
- `aesfa_full_precision_means.csv`: unrounded new image-metric means.
- `aesfa_paired_crossed_bootstrap.csv`: direction-adjusted paired differences
  and marginal 95% confidence intervals.
- `outputs_manifest.json` and `run_manifest.json`: output hashes, checkpoint
  and code hashes, seed, software and deterministic settings.
- `regression_measurements.csv` and `regression_verification.json`: reproduced
  sample measurements and the fixed validation gate.
- `analysis_verification.json`, `independent_verification.json`,
  `reference_provenance.json`, and `archive_verification.json`: completeness,
  local independent calculations, reference mapping and archive hashes.

## OCR and statistical interpretation

RapidOCR 1.2.3 uses the archived PP-OCRv3 detector/recognizer and mobile-v2
orientation classifier. Retain tokens with confidence at least 0.5 and at
least two characters after whitespace removal and lowercasing. Each original
token is matched to the most similar output token using normalized character
edit similarity; reference-token scores are averaged. Missing output text
scores zero. Poster `2508` has no retained reference tokens, so its 30 pairs
are excluded uniformly. All methods therefore have 3,570 OCR observations
from 119 posters and 30 styles; these are dependent crossed observations,
not 3,570 independent contents.

Positive paired differences favor CC-StyTr: CC-StyTr minus AesFA for SSIM,
E-SSIM and OCR; AesFA minus CC-StyTr for LPIPS and both VGG losses. Independently
resample content and style levels with replacement for 10,000 crossed-bootstrap
replicates (seed 20261014). OCR uses 119 eligible content levels. Intervals
are marginal, not simultaneous, and do not extend the general benchmark's
25-test family. The style-loss interval includes zero. The historical poster
benchmark informed configuration selection and remains selection-linked.

All 3,600 new output hashes were reverified remotely before collection. Local
verification independently reaggregated means and reconstructed all 60,000
bootstrap samples with repeated indices, agreeing with the archived endpoints
within 1e-12. Original images and private raw logs are not included here.

## Code and dependencies

Relevant project scripts are `run_poster_aesfa_revision.py`,
`analyze_poster_aesfa_revision.py`, `collect_poster_aesfa_evidence.py`,
`verify_poster_aesfa_collected.py`, and `archive_poster_aesfa_verified.py`.
The collection/verification scripts target the local migration staging layout;
adapt only filesystem locations when reusing them, preserving the protocol.
Exact package versions, model and evaluation-weight checks are recorded in
the experiment evidence and source migration manifest. Third-party terms
continue to apply; this evidence does not grant rights to source images,
third-party code or checkpoints.
