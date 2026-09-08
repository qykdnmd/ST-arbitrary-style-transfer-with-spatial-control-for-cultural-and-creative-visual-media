# Data Availability

This collection provides the CC-StyTr implementation, five model configurations, training/inference and evaluation scripts, per-pair experimental results, statistical evidence, evaluation sample manifests, and spatial α masks. Experiment-to-file mappings are described in RESULTS_MAP, fields and precision in DATA_DICTIONARY, and file checksums in `provenance/artifact_manifest.json`.

Full-precision per-pair metrics are provided for the general benchmark and the AesFA extension. Featured-set and ablation per-pair measurements are stored to four decimal places. The entire collection should therefore not be described as uniformly full precision.

Author-trained checkpoints are stored separately; access and restoration instructions are provided in `checkpoints/README.md`. COCO, WikiArt, PKU PosterLayout, MuralDH, and museum collection images must be obtained under the respective providers' access and licensing terms. THIRD_PARTY identifies the source and manifest entry points. Large source-image collections, complete generated outputs, and third-party weights are not bundled with the code.

Code and the bundled evidence are distributed through https://github.com/qykdnmd/ST-arbitrary-style-transfer-with-spatial-control-for-cultural-and-creative-visual-media. A permanent archived release and DOI have not yet been assigned. Before submission, provide accessible code/data archives, checkpoint links, and specific access or peer-review arrangements for restricted materials. Local filesystem paths must not be presented as public download addresses.

The current release also includes the verified finite-sample ArtFID evidence, the 40,000-record image hash inventory, and evaluation source snapshots. Seven feature arrays and the actual evaluation images remain external; their recorded hashes are identifiers, not download links. Historical infinity-labeled results are retained separately and are superseded by the finite table.

Public logs and machine metadata are privacy-redacted derivatives. Numerical results and exact evaluation source snapshots are unchanged. See [PUBLIC_REDACTION.md](PUBLIC_REDACTION.md) for original/public hash lineage.

## Research use and permissions

This repository is provided solely for academic research, evaluation, and reproduction of the study. No project-wide open-source license is granted. See [RESEARCH_USE.md](RESEARCH_USE.md). Third-party materials retain their own terms; a research purpose does not resolve missing or conflicting permissions.
