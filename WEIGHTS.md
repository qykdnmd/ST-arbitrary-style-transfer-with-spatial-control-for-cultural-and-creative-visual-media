# Author-trained checkpoint companion

The proposed model and four ablations use five final checkpoints, each saved
after 80,000 optimizer updates. Their exact sizes and SHA256 identities are in
`provenance/checkpoints.json`. They total 2,060,825,719 bytes before compression.
Current recovery/completeness status is recorded in
`provenance/checkpoint_recovery_status.json`; remote verification alone does
not establish a complete local checkpoint or reviewer access.

The code/measurement ZIP is kept separate from checkpoint binaries. When the
verified companion `author_model_weights.zip` is supplied, extract it into the
same root as this README, preserving the `checkpoints/ablation/<model_id>/`
directories. Do not flatten those folders. On systems with the Python zipfile CLI:

```bash
python -m zipfile -e author_model_weights.zip .
python scripts/verify_package.py --require-weights --report outputs/full_integrity.json
```

Expected only after all five weights are installed: `code_data_integrity_pass`
and `all_five_weights_verified` are both true, `pass` is true, and the exit code
is 0. A `.part` file, a matching filename or a remote-only hash is insufficient.
The strict check must fail if any full checkpoint is absent or changed.

The proposed checkpoint path is
`checkpoints/ablation/residual_gating_spatial_alpha/ccstytr_step_080000.pth`.
Its SHA256 is
`2601b1074dfbec80df7042a00720b67ffe2926de67bbe424949ea906c32efc97`.
Use that file for the inference commands in `README.md`. Checkpoints carry the
architecture in `model_cfg`; other configurations must not be silently loaded
as the proposed model. No checkpoint is loaded during the integrity check.

Third-party baseline, perceptual-loss and metric weights are not bundled.
Retrieve the pinned provider resources described in `THIRD_PARTY.md` and
`RESOURCE_ACCESS.md`; a provider URL does not guarantee byte-identical inputs.

Local recovery and preparation do not publish the weights, grant a new license,
establish a repository DOI or certify acceptance of a large file by the journal's
upload portal. No weight-specific license has been selected in the project.
The companion is not claimed to have been uploaded to PeerJ or a repository.
