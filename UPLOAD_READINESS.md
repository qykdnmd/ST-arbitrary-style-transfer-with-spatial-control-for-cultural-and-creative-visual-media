# Repository and submission preparation — 2026-10-25

The current code/measurement snapshot is documented in `README.md`,
`REPRODUCE.md`, `MANIFEST.json` and `RELEASE_STATUS.md`. The earlier September
readiness record is preserved in `provenance/archive/upload_readiness_20260908.md`.

## Verified technical preparation

- The compact snapshot contains implementations, five functional configurations,
  training/inference entry points, manifests, masks, per-pair measurements,
  statistical reanalysis, dependencies and dataset/preprocessing documentation.
- Archived validation records document 36 selected CPU tests and full numerical
  reanalysis, including all 25 primary Wilcoxon comparisons. The separate
  historical `tests/test_finite_release.py` is outside the compact-package test
  route because it requires an omitted historical upstream snapshot; it is not
  counted as passed. The full repository retains older snapshots separately.
- The WikiArt provider release and exact/name-variant sample correspondence are
  documented in `WIKIART_PROVENANCE.md`; the historical acquisition service
  remains unrecorded.
- The v0.1.0 DOI has been verified as https://doi.org/10.5281/zenodo.22653939.
  It identifies the earlier archive, not all later evidence in the current branch.

## Remaining recorded items

A new versioned release/DOI is not published by this branch update. Complete
author checkpoint binaries remain separate and recovery is not complete in the
current packaged status. Refer to `RESOURCE_ACCESS.md` and `WEIGHTS.md`.
Previously recorded provider permissions and image/code use conditions remain
as stated in `PERMISSION_STATUS.md` and `RIGHTS_AND_PERMISSIONS.md`; this update
does not add a project-wide license or revise the license of the old Zenodo record.
No PeerJ upload or resubmission is implied by a GitHub commit.
