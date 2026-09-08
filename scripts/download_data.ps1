# Dataset download helpers for Windows (plan section 2). All datasets are
# existing open datasets; verify licenses before redistribution (see docs/ plan
# section 2.4).
param([string]$DataDir = "data")

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

Write-Output "== 1. MS-COCO train2014 (content training set, ~13GB) =="
Write-Output "  Invoke-WebRequest http://images.cocodataset.org/zips/train2014.zip -OutFile $DataDir\coco\train2014.zip"
Write-Output "  Expand-Archive $DataDir\coco\train2014.zip -DestinationPath $DataDir\coco"

Write-Output "== 2. WikiArt (style training set, ~26GB) =="
Write-Output "  Kaggle: https://www.kaggle.com/c/painter-by-numbers (train.zip) -> $DataDir\wikiart"

Write-Output "== 3. MuralDH Dunhuang murals (CC0, Dryad doi:10.5061/dryad.bnzs7h4jd) =="
Write-Output "  https://datadryad.org/dataset/doi:10.5061/dryad.bnzs7h4jd -> $DataDir\muraldh"

Write-Output "== 4. Chinese landscape paintings (alicex2020) =="
Write-Output "  git clone https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset $DataDir\chinese-landscape"

Write-Output "== 5. The Met Open Access (CC0) =="
Write-Output "  Metadata CSV: https://github.com/metmuseum/openaccess"
Write-Output "  Use scripts/fetch_met_paintings.py to pull the Paintings subset via the API."

Write-Output "== 6. Crello design templates (CDLA-Permissive-2.0) =="
Write-Output "  HuggingFace: cyberagent/crello (poster/postcard/banner categories)"

Write-Output "== 7. PKU-PosterLayout (CC-BY-SA-4.0) =="
Write-Output "  HuggingFace: creative-graphic-design/PKU-PosterLayout"
