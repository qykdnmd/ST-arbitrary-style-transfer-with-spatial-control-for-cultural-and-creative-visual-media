# CC-StyTr: structure-preserving arbitrary style transfer with spatial control

Code and measurement snapshot updated on 2026-10-25 for PeerJ Computer Science
manuscript 148632 (AI Application), “Structure-preserving transformer-based
arbitrary style transfer with spatial control for cultural and creative visual
media”. The `main` branch contains this working snapshot. The published
`v0.1.0` tag and https://doi.org/10.5281/zenodo.22653939 identify the earlier
September 2026 archive. A new versioned release and DOI have not yet been published.

See [RELEASE_STATUS.md](RELEASE_STATUS.md) for the distinction between the current
snapshot, the earlier archive, the compact submission package and retained historical files.

## Description and methodology

CC-StyTr extends a transformer encoder/decoder with structure-aware attention and
spatial style-intensity control. The proposed configuration is
`residual_gating_spatial_alpha`: saliency gating acts on the cross-attention output
before residual addition. Token-wise alpha controls the texture and color branches.
Five functional configurations are supplied for the proposed model and ablations.
They use 80,000 optimizer updates and effective batch size 16. No model algorithm,
archived numerical measurement or training configuration was changed for packaging.

## Dataset information

COCO train2014 supplies training content; WikiArt supplies training styles. General
evaluation uses 40 COCO val2017 contents × 20 WikiArt styles (800 pairs per method).
Poster evaluation uses 120 PosterLayout contents × 30 styles; its OCR subset uses
119 contents (3,570 pairs per method). Independent finite-ArtFID/CSD evaluation uses
5,000 pairs per method. Spatial-control evaluation uses 60 contents × 10 styles ×
7 conditions. These sets must not be joined by filename without checking provenance.

`data/` includes selection/pairing manifests and author-generated control masks;
it does not contain the complete original image datasets. `paper/evidence/` and
`results/` contain frozen measurements and provenance. Read `DATA_SOURCES.md` for
original URLs and preprocessing, `DATA_DICTIONARY.md` for columns, and `RESULTS_MAP.md`
for current versus historical evidence. The collection source is https://www.wikiart.org/. Public ArtGAN refined-WikiArt
metadata were cross-checked against the evaluation sample manifests; see
`WIKIART_PROVENANCE.md` for the provider release, download links and documented
filename variants. The historical download service remains unrecorded.

Training preprocessing is RGB conversion → square 512×512 resize → random 256×256
crop → tensor in [0,1], without an added flip or channel normalization. Evaluation
preprocessing differs by method; AesFA uses shorter-edge resize and center crop
with its normalization. See `DATA_SOURCES.md` before interpreting cross-method OCR.

## Code information

| Path | Purpose |
|---|---|
| `ccstytr/` | Model, attention, saliency, alpha-map sampling, losses and checkpoint helpers |
| `train.py`, `infer.py` | Training and single-pair inference entry points |
| `configs/ablation/` | Five functional model/training configurations |
| `scripts/infer_paired.py` | Paired or Cartesian inference, including baseline wrappers |
| `scripts/reanalyze_submission.py` | Current six-method measurement reanalysis |
| `scripts/render_tables.py`, `scripts/render_featured_tables.py` | Current table renderers |
| `scripts/verify_package.py` | SHA256 integrity checks; optional strict weight checks |
| `tests/` | CPU model, pipeline, statistics and historical evidence checks |
| `provenance/` | Environment, source and checkpoint identities; package verification records |

## Requirements and installation

Python 3.10 or newer is required. Use a separate virtual environment and run commands
from the directory containing this README. For training/inference install a mutually
compatible PyTorch/torchvision build for your hardware, then:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

The main model needs PyTorch, torchvision, NumPy, Pillow, PyYAML, tqdm and TensorBoard.
Training initializes torchvision VGG19 ImageNet weights for perceptual losses; cache
or download these from the provider. They are not needed by `infer.py`. Evaluation
may require further pretrained resources and optional packages:

```bash
python -m pip install -r requirements-statistics.txt   # numerical analysis
python -m pip install -r requirements-evaluation.txt   # image metrics and OCR
python -m pip install -r requirements-dev.txt          # model unit tests
```

For archived-measurement reanalysis alone, the minimal dependencies are NumPy,
pandas and SciPy. statsmodels in the statistics requirements supports historical
analysis, but the current audit does not refit mixed-effects models.
`provenance/environment.json` records the experiment environment; it is not a
portable lock file. `provenance/package_validation.json` records the environment
and checks actually used to validate this submission package.

## Usage instructions

### 1. Verify the unpacked code/data archive

This command uses only the Python standard library:

```bash
python scripts/verify_package.py --report outputs/package_integrity.json
```

It checks every file listed in `MANIFEST.json`. Weight presence is reported
separately; without `--require-weights`, a successful code check does not imply that
weights are installed. See `WEIGHTS.md` for extraction and strict verification.

### 2. Recompute statistics from supplied measurements

```bash
python scripts/reanalyze_submission.py --root . --report outputs/numerical_audit.json
python scripts/render_tables.py --output outputs/general_tables.md
python scripts/render_featured_tables.py --output outputs/poster_tables.md
```

Expected: all numerical checks true, `wilcoxon.executed: true`, all 25 paired tests
matched, six 10,000-replicate poster intervals matched, and audit exit code 0.
Missing SciPy/pandas causes the legacy audit to skip Wilcoxon recomputation even
if other checks pass: inspect this field rather than treating exit code 0 alone
as complete statistical reproduction. This is aggregation/reanalysis of frozen
measurements, not a new image-level experiment. The poster style-loss interval
includes zero; the historical mixed-model family has 20, not 25, models.

### 3. Inference using an author-trained checkpoint

Obtain the matching author checkpoint companion and verify its full SHA256 first.
Create an `outputs` directory and supply your own permitted content/style images.
Replace the example input paths with real images:

```bash
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
python infer.py --checkpoint checkpoints/ablation/residual_gating_spatial_alpha/ccstytr_step_080000.pth --content inputs/content.jpg --style inputs/style.jpg --output outputs/styled.png --size 512 --alpha 1.0
python infer.py --checkpoint checkpoints/ablation/residual_gating_spatial_alpha/ccstytr_step_080000.pth --content inputs/content.jpg --style inputs/style.jpg --output outputs/styled_controlled.png --size 512 --alpha-map inputs/alpha.png
```

The alpha-map is grayscale: black selects lower style intensity and white higher
intensity. `--auto-alpha` derives a map from structure saliency; `--alpha 0.5` uses
a global constant. Choose one control mode per invocation. The single-pair entry
point selects CUDA when available and CPU otherwise, reconstructs the model from
the checkpoint's `model_cfg`, and saves a square RGB PNG. It does not reproduce a
table by itself. Tensor values are clamped to [0,1] before saving.

For the general Cartesian protocol, place the exact selected images described in
the manifests in the content/style directories and run:

```bash
python scripts/infer_paired.py --method ccstytr --pairing cartesian --content_dir data/eval/general/content --style_dir data/eval/general/style --output_dir outputs/general/ccstytr --checkpoint checkpoints/ablation/residual_gating_spatial_alpha/ccstytr_step_080000.pth --size 512
```

This produces 800 outputs only when the supplied folders contain exactly the 40
and 20 selected images. This archive supplies manifests, not those complete folders.
Baseline runs require separately obtained pinned implementations and weights in
`THIRD_PARTY.md`; none is silently substituted by the proposed-model checkpoint.

### 4. Train from source

Acquire the original data under its terms. Place COCO images beneath
`data/train/coco/train2014` and WikiArt images beneath `data/train/wikiart` (nested
movement folders are supported). Adjust the two paths in a copy of the selected
YAML if needed, while retaining the archived architecture and training parameters:

```bash
python train.py --config configs/ablation/residual_gating_spatial_alpha.yaml
python train.py --config configs/ablation/residual_gating_spatial_alpha.yaml --resume auto
```

Checkpoints and logs are written to the `ckpt_dir` and `log_dir` specified in the
YAML. `max_steps` counts optimizer updates, not micro-iterations. Resume restores
available optimizer/RNG state; worker ordering is not restored, and a new run is
not guaranteed to be bitwise identical to the original training. Other four
configurations use the correspondingly named YAML files. No new training was run
when preparing this package.

### 5. Run CPU unit tests

```bash
python -m pytest tests --ignore=tests/test_finite_release.py -q
```

The excluded historical test checks an upstream ArtFID source snapshot that is
not redistributed here; it requires that exact snapshot, not just the numerical
CSV files. It has not been weakened or silently counted as passed. The current
numerical audit independently checks the finite-ArtFID formula and archived CSD
means, but not fresh Inception features. See `VALIDATION.md` for actual results.

## Citation, AI assistance, license and contributions

Cite the accompanying manuscript and the original methods/data providers. The
earlier public snapshot is https://doi.org/10.5281/zenodo.22653939 and the project
repository is https://github.com/qykdnmd/ST-arbitrary-style-transfer-with-spatial-control-for-cultural-and-creative-visual-media.
That DOI does not identify this revision or its added AesFA poster/OCR evidence
and unified 25-test analysis. No new DOI is claimed. AI assistance is disclosed
in `AI_USE.md`; the tools are not inference components of CC-StyTr.

No new project-wide or weight-specific license is granted by this packaging step.
Existing author-controlled use conditions are in `RESEARCH_USE.md`; third-party
conditions and unresolved historical permission records remain in `THIRD_PARTY.md`,
`RIGHTS_AND_PERMISSIONS.md` and `PERMISSION_STATUS.md`. Please raise implementation
issues or contribution proposals through the repository maintainer before adding
data or third-party resources. Do not include credentials, private logs or source
images without applicable permission. Read `RESOURCE_ACCESS.md` for reproduction
boundaries and `SUBMISSION_METADATA.md` for attachment descriptions.
