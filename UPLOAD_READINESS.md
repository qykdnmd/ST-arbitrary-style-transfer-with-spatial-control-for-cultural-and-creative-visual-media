# Upload readiness assessment (2026-09-08)

## Conclusion

The finite-sample ArtFID results, current tables, provenance links and documented evaluation entry point have been synchronized. GitHub repository: https://github.com/qykdnmd/ST-arbitrary-style-transfer-with-spatial-control-for-cultural-and-creative-visual-media. This upload does not establish a permanently archived release or DOI. File size is not a GitHub upload blocker. Public redistribution and submission-archive readiness remain conditional on the items below. This assessment covers the uploaded collection; the upload does not resolve the licensing and external-access conditions below.

## Required decisions before a public release

1. **Project license:** no top-level project LICENSE is present. The author must choose appropriate terms for original code and data after checking ownership and dependencies. No license was assigned automatically.
2. **Third-party redistribution:** the local StyTR-2 and ArtFID trees have no standalone LICENSE. ArtFID's README links to MIT, while its setup.py declares Apache License 2.0. Resolve the applicable upstream terms and required notices, or distribute acquisition instructions pinned to verified revisions instead of bundling unclear code. This also applies to the ArtFID source snapshot. AesFA, CAST, CSD, AdaIN and SANet include license files, but their presence alone is not a complete license audit.
3. **Image permissions:** included qualitative figures, example images and derived visual assets require source-specific permission checks. Source availability does not establish redistribution rights.
4. **Access for reproduction:** author-trained checkpoints, source images, complete generated outputs and feature arrays are not bundled. Provide accessible archives or documented review-access arrangements and precise links. Hashes do not provide access. Add a versioned public release and permanent archive identifier when available; do not claim an existing DOI.
5. **Machine metadata (addressed):** local usernames and absolute machine paths have been removed from the public files. Originals are held outside the release; public derivatives and their hash lineage are documented in PUBLIC_REDACTION.md. Future generated logs still require review.

## Technical scope

Current FID/ArtFID is finite-sample and based on 5,000 unique outputs per method. Both current CSV copies use the new full-precision scores and unchanged CSD means/intervals. Historical infinity-labeled results remain separate and superseded. The old combined evaluator is disabled at the public entry point. The new wrapper supports this vendored layout, while the exact training-run source is preserved unchanged.

File/row/mean checks, source hashes, the combined-table check and CPU release tests validate the migration. They do not establish a clean-room GPU reproduction or certify all third-party rights. The finite validator rechecks 40,000 archived inventory records and 30,000 archived CSD pairs; it does not read the absent image files during this local audit.

A pattern-based scan found no private-key headers or common GitHub, Hugging Face, or AWS credential formats, and no typical credential filenames. This is not a guarantee that no secrets exist. No Git history is present to scan.

## Official references

- [GitHub large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github): warnings above 50 MiB and ordinary Git files blocked above 100 MiB.
- [GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository): public visibility is not a substitute for a license.

The README and Data Availability statement should be updated again after actual publication, with real repository/archive and checkpoint links.

## Verification performed during synchronization

- Artifact verification passed: 516 tracked-by-manifest files before the final documentation update; five external model checkpoint hashes verified; 120 spatial alpha masks; 82 Python sources parsed successfully.
- Six CPU release regression tests passed, including the finite table and its immutable source snapshots.
- All 20 primary Wilcoxon/effect-size comparisons matched the archived values; all retained Bonferroni significance.
- The full pytest suite could not run because pytest is not installed in the audit interpreter. No GPU rerun or clean-room installation was performed.
- Largest bundled file: the image inventory, 10,229,984 bytes (approximately 9.76 MiB). The collection is approximately 58.4 MiB, excluding ignored caches. No individual bundled file reaches the GitHub 50 MiB warning threshold.
