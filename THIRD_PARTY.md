# Third-party implementations, weights, and data

`external/` contains source code, configurations, and documentation for the relevant implementations, but not all binary resources or model weights. Upstream LICENSE/NOTICE files are retained where available. This collection does not grant a blanket license for third-party materials. Check each applicable license before redistribution; an absent license file does not imply unrestricted use.

Implementation revisions and the environment record are in `provenance/environment.json`. Versions and checkpoint hashes for the 5,000-pair experiment are recorded in `paper/evidence/artfid5000_inference_provenance.json` and `paper/evidence/artfid5000_provenance.json`. Each implementation's README provides installation and model-access instructions. ArtFID and CSD should use the implementation and checkpoint versions specified in the experiment records.

Obtain AesFA, StyTr², AdaIN, SANet, CAST, CSD, ArtInception, VGG, and other third-party weights from their respective providers. The five author-trained checkpoints are stored separately; see `checkpoints/README.md`. Accessible download links and applicable terms still need to be supplied for the submission archive.

## Data entry points

- COCO: `data/eval/general/manifest.csv` records selected filenames and source URLs. The 5,000-pair experiment uses `data/eval/artfid5000/manifest.csv`.
- WikiArt: pairing manifests record relative style-image paths and groups. Obtain the corresponding images under the provider's terms.
- Posters: see the `data/eval/poster*/` manifests. The spatial-control `posters.csv` records parquet files, row indices, and text/logo coordinates.
- Cultural-heritage styles: `data/eval/styles/` retains source fields and sample-selection records. Image permissions require individual verification.
- Spatial control: `data/eval/spatial_control/` provides the protocol, sample manifests, and protected/inverted α masks.

Some manifests contain relative paths rather than persistent download addresses, and not every source image has a SHA256 record. These manifests alone are therefore not a complete accessible data archive. Before submission, establish access or peer-review arrangements for source images, generated results, and restricted materials.

For the current finite ArtFID experiment, use the package versions, art-trained Inception checkpoint hash, and exact source hashes in `paper/evidence/artfid5000_finite_20260908/provenance.json`. The matching source snapshot is included there. The current `artfid5000_provenance.json` links the combined table to its finite and CSD sources. CSD uses the unchanged provenance under `results/artfid5000/historical_infinity/`.

## Research-use scope and unresolved permissions

The [research-use statement](RESEARCH_USE.md) does not relicense any upstream material or restrict rights already granted by an upstream license. Existing third-party LICENSE/NOTICE files remain unchanged.

The bundled StyTR-2 tree has no standalone LICENSE file. ArtFID has no standalone LICENSE file in the bundled tree; its README links to MIT while setup.py declares Apache License 2.0. These observations do not establish the applicable permission. Verify the relevant revision and rights-holder authorization before relying on redistribution or reuse rights. The statement of academic purpose does not resolve these issues, including for source snapshots.
