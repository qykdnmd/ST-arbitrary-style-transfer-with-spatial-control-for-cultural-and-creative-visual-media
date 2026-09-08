# Public metadata redaction

Local usernames and absolute machine paths are not authorized for public release. The affected public files are labeled derivatives; original files are preserved in an author-controlled location outside this release. No private location or username is included in this document or the public mapping.

`provenance/public_redactions.json` records release-relative filenames, original SHA-256 values, public SHA-256 values, and the category of change. `[PROJECT_ROOT]` and `[USER_HOME]` are anonymized roots, not download addresses. Configuration paths are project-relative and diagram scripts discover their project root at runtime.

The original verification report's `run_provenance_sha256` still identifies the private original. Its `public_derivative.public_run_provenance_sha256` identifies the redacted public provenance. Fresh local validation reports refer to the public provenance. This distinction must not be presented as a new GPU experiment.

No FID, ArtFID, LPIPS, CSD, confidence interval, sample identifier, image hash, checkpoint hash, source snapshot, or scientific diagnostic was changed by this redaction. Source snapshots and numerical results remain byte-for-byte intact. The publicly distributed log, provenance and verification metadata are not byte-identical to the original private files.

Before any future release, scan new logs and reports again. Do not place original private logs or local backup archives inside this repository. Automated pattern scans cannot guarantee removal of every possible identifying detail.
