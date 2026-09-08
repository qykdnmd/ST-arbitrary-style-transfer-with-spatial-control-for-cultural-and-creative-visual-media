# Current 5,000-pair results

`summary.csv` is the current manuscript table: finite-sample FID/ArtFID and freshly recomputed AlexNet LPIPS, combined by method identifier with the unchanged archived CSD means and 95% bootstrap intervals. Each method has 5,000 unique pairs. No infinity extrapolation is used.

Authoritative finite-run evidence: `../../paper/evidence/artfid5000_finite_20260908/`. Its immutable `summary.json` contains the new FID, LPIPS, and ArtFID values; `verification.json` records the training-host checks. `current_provenance.json` links the combined CSV to those sources.

`provenance.json` in this directory mirrors the current combined-table provenance. The historical ArtFID/CSD provenance is preserved under `historical_infinity/`. `inference_provenance.json` still describes the unchanged generated outputs. `csd_per_pair.csv` is unchanged and remains the CSD source.

`historical_infinity/` retains the original results and provenance. Those infinity-labeled scores are superseded and are not validated infinity estimates. The Windows PNG/PNG-case enumeration issue and the finite estimator are documented in `../../paper/evidence/artfid5000_finite_protocol.md`.

Public logs and machine metadata are privacy-redacted derivatives. Numerical results and exact evaluation source snapshots are unchanged. See [PUBLIC_REDACTION.md](../../PUBLIC_REDACTION.md) for original/public hash lineage.
