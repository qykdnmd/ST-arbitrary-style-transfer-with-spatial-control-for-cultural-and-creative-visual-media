# Dataset sources and preprocessing

## Original third-party sources

| Dataset / collection | Original source | Role |
|---|---|---|
| MS COCO | https://cocodataset.org/ | train2014 content training; val2017 content evaluation |
| COCO train2014 archive | http://images.cocodataset.org/zips/train2014.zip | Official training images |
| COCO val2017 archive | http://images.cocodataset.org/zips/val2017.zip | Official evaluation images |
| WikiArt | https://www.wikiart.org/ | Style training and evaluation references |
| PKU PosterLayout | https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023 | Poster images and official annotations, subject to its access procedure |
| MuralDH | https://doi.org/10.5061/dryad.bnzs7h4jd | Dunhuang mural styles |
| Cleveland Museum of Art | https://www.clevelandart.org/open-access | Eligible Open Access / CC0 images, identified in manifests |
| The Metropolitan Museum of Art | https://www.metmuseum.org/about-the-met/policies-and-documents/open-access | Eligible Open Access / CC0 images, identified in manifests |

The project's historical download helpers also list https://www.kaggle.com/c/painter-by-numbers (train.zip) as a WikiArt acquisition entry point. Those helpers print instructions; they are not execution logs proving that this archive supplied the actual training images. No such acquisition claim is made here. Use the archived sample manifests and obey each source's access and redistribution conditions.

## Implemented transformations

- Training: Pillow RGB conversion; torchvision square Resize(512,512); RandomCrop(256); ToTensor(), producing values in [0,1]. No horizontal flip or additional channel normalization appears in this training transform. See ccstytr/data/datasets.py.
- Proposed-model evaluation: RGB conversion, square resizing to 512, and tensor conversion. The StyTr2, AdaIN, and SANet wrappers also use a square resize.
- AesFA: shorter-side resize to 512, center crop to 512x512, and its official ImageNet mean/std normalization. Saved outputs are returned to the image-value range. Crop coordinates in the qualitative poster figure account for this geometry.
- CAST: upstream test-data preprocessing invoked by scripts/infer_paired.py. Consult that wrapper and the vendored transform for the exact operations.
- Separate 5,000-pair reference set: RGB PNG images square-resized to 512x512 with Pillow LANCZOS; one-to-one pairs and seed 20260902 are recorded in data/eval/artfid5000/protocol.json.
- General-set selection: seed 2026 in scripts/build_general_eval.py, with 40 content and 20 style images and all 800 combinations per method.

Equal input/output dimensions do not establish identical preprocessing across all methods. The reported metrics, original sample selection, model outputs, and protocol values have not been recomputed during the editorial revision.


## Identified WikiArt release

The archived evaluation manifests have been cross-checked against the refined
WikiArt release at https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset.
Its maintainers provide image-download links, including
https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view.
See WIKIART_PROVENANCE.md for exact and filename-variant correspondence, pinned
metadata hashes, and the distinction from the unrecorded historical download service.
