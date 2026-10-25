# Resource access and reproduction scope

## Accompanying archive

Supplementary File S1 is `supplementary_code_and_data.zip`. It contains implementation,
functional configurations, pairing/selection manifests, control masks, pairwise
measurements, statistical evidence and reanalysis scripts. The new AesFA poster/OCR
archive and unified 25-comparison general analysis are included. The earlier DOI
https://doi.org/10.5281/zenodo.22653939 identifies an earlier snapshot, not File S1.
File S1 is prepared locally for attachment; no public upload or new DOI is claimed.

## Author-trained model weights

All five final author-trained checkpoints were located on the original training
host, and their complete remote-file sizes and SHA256 hashes match the archived
records. The proposed model is `residual_gating_spatial_alpha`; see
`provenance/remote_checkpoint_verification.json` and `provenance/checkpoints.json`.
Private connection details and historical internal directory names are omitted.

The author authorized resumed transfer on 2026-10-25. The original-host full
file sizes and SHA256 identities were rechecked and all five matched before
transfer. The current local result and companion-archive status are recorded in
`provenance/checkpoint_recovery_status.json`, which supersedes the old paused
status for this package. See `WEIGHTS.md` for exact installation and verification.
Partial downloads are excluded. No public URL, DOI deposit or completed PeerJ
upload is claimed by this local preparation. Code and weights are separate
attachments, and reviewer access requires supplying the actual companion.

## Third-party weights

AesFA and CSD binaries located in the author workspace were hashed without loading
or unpickling them; both matched their archived hashes. They are not redistributed
in File S1. Obtain third-party weights from the providers and verify hashes against
the experiment-specific provenance before evaluating. A provider landing page is
not proof that the downloaded bytes match the experiment.

- AesFA: https://github.com/Sooyyoungg/AesFA
- StyTr2: https://github.com/diyiiyiii/StyTR-2 (four provider links in its README)
- AdaIN/VGG: https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0
- SANet: https://github.com/GlebSBrykin/SANET
- CAST: https://github.com/zyxElsa/CAST_pytorch
- CSD: https://github.com/learn2phoenix/CSD and https://huggingface.co/tomg-group-umd/CSD-ViT-L
- ArtFID/art-trained Inception: https://github.com/matthias-wright/art-fid

Official CSD documentation warns of possible differences between released weights
and results in its paper. This study identifies the evaluated checkpoint by hash;
it does not certify reproduction of the upstream paper's numerical results.

## Source images and generated outputs

Full source/generated image collections and feature caches are not included.
Use `DATA_SOURCES.md`, original providers and the archived manifests. The 30 styles
selected for the poster experiment were matched exactly to the frozen reference
measurements; see `provenance/resource_access_audit.json`. Do not confuse larger
candidate-pool manifests with the final evaluation selection.

The Met and Cleveland metadata were checked for all 20 selected museum objects;
the checked public-domain/CC0 flags and source URLs are recorded separately from
the image files. This does not certify that any replacement download is byte-identical
to the historical experimental input. MuralDH, PosterLayout, COCO and WikiArt need
their applicable acquisition/use terms and, where required, image-specific rights.

## What can be reproduced now

The supplied measurement archive supports current table aggregation, 25 primary
Wilcoxon tests, poster crossed-bootstrap intervals, CSD means and finite-ArtFID
score composition. It does not alone support an exact end-to-end inference rerun,
fresh Inception feature extraction or complete independent retraining. Do not
describe the entire experiment as fully reproduced by a numerical reanalysis.
