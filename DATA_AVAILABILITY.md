# Data and code availability — current repository snapshot

The current repository snapshot and prepared `supplementary_code_and_data.zip` contain the author model
implementation, training/inference entry points, functional configurations,
selection/pairing manifests, masks, pairwise measurements, statistical evidence,
checkpoint identities and reanalysis scripts. It includes the current AesFA
poster/OCR evidence and unified 25-comparison general analysis. It is prepared
for attachment to PeerJ Computer Science manuscript 148632; preparation is not
a journal upload or a new versioned DOI publication. `SUBMISSION_METADATA.md` supplies a descriptive legend
including the term “code”, as requested by the technical editor.

Author-trained binaries are handled in a separate `author_model_weights.zip`
companion, only when full local size/SHA256 verification has succeeded.
`provenance/checkpoint_recovery_status.json` states the actual recovery status.
See `WEIGHTS.md` and `RESOURCE_ACCESS.md`. Hash records or partial downloads do
not establish usable reviewer access.

Earlier public repository:
https://github.com/qykdnmd/ST-arbitrary-style-transfer-with-spatial-control-for-cultural-and-creative-visual-media

Earlier snapshot DOI: https://doi.org/10.5281/zenodo.22653939

That DOI is not a DOI for this revision or the new evidence. No new public
release, DOI or on-request access promise is created by this package.

Full third-party source image collections, original generated-output grids,
feature caches, third-party source trees and baseline/metric weight binaries
are not redistributed. Obtain them from the providers in `DATA_SOURCES.md`
and `THIRD_PARTY.md` under applicable terms. The corresponding ArtGAN refined-WikiArt release and provider-listed download
links are identified in `WIKIART_PROVENANCE.md`; the historical download service
remains unverified.

The supplied archive supports numerical reanalysis; an image-level rerun also
needs the checkpoint companion, original permitted inputs and protocol-specific
upstream resources. No complete retraining or fresh feature extraction is claimed.
Existing use conditions and unresolved permissions remain documented in
`RESEARCH_USE.md`, `PERMISSION_STATUS.md` and `RIGHTS_AND_PERMISSIONS.md`; no new
project-wide or weight license is granted.
