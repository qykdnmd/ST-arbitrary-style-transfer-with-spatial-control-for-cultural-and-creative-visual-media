# Supplemental-file descriptions for PeerJ manuscript 148632

## Code and measurement archive

Filename: `supplementary_code_and_data.zip`

Suggested title: CC-StyTr research code and measurement data

Suggested legend: Research code for CC-StyTr, including model implementation,
training and inference scripts, five ablation configurations, dataset selection
and pairing manifests, spatial-control masks, archived pairwise measurements,
statistical reanalysis scripts, a README with installation and usage instructions,
resource provenance and validation records. The archive includes the six-method
general/poster analyses and unified 25-comparison statistical family. Full source
image collections, baseline resources and author checkpoint binaries are separate.

The editorial email requests upload as a Supplemental File with “code” in its
descriptive legend and “dataset” as the file type. Use the actual portal labels.
This metadata is prepared locally; no journal upload is claimed.

## Author checkpoint companion

Filename when completely prepared: `author_model_weights.zip`

Suggested title: CC-StyTr author-trained model checkpoints

Suggested legend: Five final author-trained checkpoints at 80,000 optimizer
updates for the proposed residual saliency gating plus spatial-alpha model and
four functional ablations. Directory names, complete-file sizes and SHA256 hashes
are supplied to identify the matching configurations. Extract alongside the code
archive and run its strict checkpoint verification before inference.

Only offer this companion when `provenance/checkpoint_recovery_status.json`
reports `companion_archive_prepared: true` and the actual ZIP is present. No DOI,
public weight URL or completed upload is implied. Portal size acceptance has not
been verified; do not substitute limits from another journal.
