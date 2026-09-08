# CC-StyTr

Structure-aware Transformer-based style transfer with spatial control for cultural-creative visual media.

This collection provides the implementation, configurations, evaluation scripts, and machine-readable evidence accompanying the manuscript. **CC-StyTr (Main)** uses residual saliency gating and spatial α, identified by `residual_gating_spatial_alpha`. The manuscript refers to it as **the proposed model**.

## Reproduction resources

- [Models and experiment-to-file mapping](RESULTS_MAP.md)
- [Installation, inference, training, and verification](REPRODUCE.md)
- [Checkpoints and access instructions](checkpoints/README.md)
- [Data dictionary](DATA_DICTIONARY.md)
- [Third-party implementations, models, and data](THIRD_PARTY.md)
- [Data availability](DATA_AVAILABILITY.md)

The collection includes five equal-budget configurations, tests, per-pair measurements, statistical evidence, manuscript figures, sampling manifests, and 120 spatial α masks. Checkpoints are stored separately. Source image collections and complete generated-image collections are not bundled with the code.

Repository: https://github.com/qykdnmd/ST-arbitrary-style-transfer-with-spatial-control-for-cultural-and-creative-visual-media. A permanent archived release and DOI have not yet been assigned. The environment record is in `provenance/environment.json`, and file hashes are in `provenance/artifact_manifest.json`. Experiment-specific protocols and implementation versions are recorded in the corresponding evidence/provenance files. Evaluation datasets and statistical test families must be interpreted separately.

## Current ArtFID evidence

The 5,000-pair table uses verified ordinary finite-sample ArtFID, not ArtFID-infinity. See [current results and provenance](results/artfid5000/README.md). Historical files are retained for traceability and are not current manuscript results.

Before public publication, resolve the outstanding items in [Upload readiness](UPLOAD_READINESS.md). The successful numerical checks do not establish permission to redistribute every bundled asset.

Public logs and machine metadata are privacy-redacted derivatives. Numerical results and exact evaluation source snapshots are unchanged. See [PUBLIC_REDACTION.md](PUBLIC_REDACTION.md) for original/public hash lineage.
