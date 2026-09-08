#!/usr/bin/env bash
# Dataset download helpers (plan section 2). All datasets are existing open
# datasets; verify licenses before redistribution (see docs/ plan section 2.4).
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "== 1. MS-COCO train2014 (content training set, ~13GB) =="
echo "  wget http://images.cocodataset.org/zips/train2014.zip -P $DATA_DIR/coco"
echo "  unzip $DATA_DIR/coco/train2014.zip -d $DATA_DIR/coco"

echo "== 2. WikiArt (style training set, ~26GB) =="
echo "  Kaggle: https://www.kaggle.com/c/painter-by-numbers (train.zip) -> $DATA_DIR/wikiart"

echo "== 3. MuralDH Dunhuang murals (CC0, Dryad doi:10.5061/dryad.bnzs7h4jd) =="
echo "  https://datadryad.org/dataset/doi:10.5061/dryad.bnzs7h4jd -> $DATA_DIR/muraldh"

echo "== 4. Chinese landscape paintings (alicex2020) =="
echo "  git clone https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset $DATA_DIR/chinese-landscape"

echo "== 5. The Met Open Access (CC0) =="
echo "  Metadata CSV: https://github.com/metmuseum/openaccess"
echo "  Use scripts/fetch_met_paintings.py to pull the Paintings subset via the API."

echo "== 6. Crello design templates (CDLA-Permissive-2.0) =="
echo "  HuggingFace: cyberagent/crello (poster/postcard/banner categories)"

echo "== 7. PKU-PosterLayout (CC-BY-SA-4.0) =="
echo "  HuggingFace: creative-graphic-design/PKU-PosterLayout"
