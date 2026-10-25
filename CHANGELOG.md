# Repository synchronization — 2026-10-25

- Synchronized the validated PeerJ AI Application code/measurement snapshot to `main`.
- Added the six-method general/poster evidence, AesFA OCR results and unified
  25-comparison analysis, with their exact archived configurations and provenance.
- Expanded installation, reproduction, data-source, AI-use and resource-access
  documentation. Added verified ArtGAN/WikiArt metadata correspondence.
- Confirmed the earlier v0.1.0 DOI as 10.5281/zenodo.22653939. No tag was moved,
  no past archive was replaced, and no new versioned release or DOI is claimed.
- Preserved implementation/configuration/measurement bytes from the validated
  submission package; this synchronization does not imply a new GPU experiment.

The following record describes preparation of the compact package before this
repository synchronization.

# Submission package changes — 2026-10-25

- Prepared an independent PeerJ Computer Science AI Application code/data package.
- Replaced the old journal-specific README with dataset, implementation,
  installation, usage, preprocessing, expected-output and citation information.
- Added package integrity checking, companion-weight installation instructions,
  current validation records and PeerJ attachment descriptions.
- Updated AI-use documentation and WikiArt acquisition-provenance limitations.
- Corrected development requirements to include statistics and metric imports used
  during test collection (the old development list omitted them).
- Raised the SciPy lower bound to 1.15 because the supplied audit uses the
  `wilcoxon(method="asymptotic")` API, introduced under that name in 1.15.
- Resumed recovery of the five original hash-identified author checkpoints;
  actual completion is recorded in `provenance/checkpoint_recovery_status.json`.
- Kept scientific algorithms, configurations, pairwise measurements and archived
  experimental provenance unchanged. No retraining or new scientific measurement
  is implied by assembling this package.

The earlier source archive SHA256 is recorded in `provenance/package_source.json`.
No new external upload, public release, repository license or DOI was created.
