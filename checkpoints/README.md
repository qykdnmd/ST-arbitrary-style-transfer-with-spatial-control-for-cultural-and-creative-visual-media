# Model checkpoints

Proposed model: **CC-StyTr (Main)**, configuration identifier `residual_gating_spatial_alpha`.

The five checkpoints use the following runtime directories. Each file is named `ccstytr_step_080000.pth`:

```text
checkpoints/ablation/
├── residual_gating_spatial_alpha/
├── saliency_routed_spatial_alpha/
├── no_structure_guidance/
├── saliency_routed_scalar_alpha/
└── residual_gating_scalar_alpha/
```

On this machine, the weights are stored separately in `../CC-StyTr_model_weights/ablation/`, using the same directory names. Public checkpoint download links have not yet been provided. A local path is not a public access address.

```sh
python scripts/restore_checkpoints.py --source ../CC-StyTr_model_weights --verify-only
python scripts/restore_checkpoints.py --source ../CC-StyTr_model_weights
```

The first command only verifies the files. The second verifies and copies them into the runtime directories. Their total size is approximately 2.06 GB. Model metadata and SHA256 hashes are recorded in `provenance/checkpoints.json`. Load only trusted checkpoints whose hashes match the recorded values.
