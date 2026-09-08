param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location -LiteralPath $RepoRoot

$assetRoot = Join-Path $RepoRoot 'data\submission_assets'
New-Item -ItemType Directory -Force -Path $assetRoot | Out-Null

# COCO val2017: the locked ArtFID/CSD protocol requires all 5,000 distinct
# validation images. Keep the archive so download provenance is inspectable.
$cocoRoot = Join-Path $RepoRoot 'data\coco'
$cocoVal = Join-Path $cocoRoot 'val2017'
$cocoZip = Join-Path $assetRoot 'val2017.zip'
$cocoUrl = 'https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip'
$expectedCocoBytes = 815585330
$expectedCocoMd5 = '442b8da7639aecaf257c1dceb8ba8c80'
if (-not (Test-Path -LiteralPath $cocoVal) -or
    (Get-ChildItem -LiteralPath $cocoVal -File | Measure-Object).Count -ne 5000) {
    New-Item -ItemType Directory -Force -Path $cocoRoot | Out-Null
    & curl.exe -L --fail --retry 10 --retry-delay 5 --continue-at - `
        $cocoUrl -o $cocoZip
    if ($LASTEXITCODE -ne 0) { throw "COCO download failed: exit $LASTEXITCODE" }
    $cocoInfo = Get-Item -LiteralPath $cocoZip
    if ($cocoInfo.Length -ne $expectedCocoBytes) {
        throw "COCO archive size mismatch: expected $expectedCocoBytes, found $($cocoInfo.Length)"
    }
    $cocoMd5 = (Get-FileHash -LiteralPath $cocoZip -Algorithm MD5).Hash.ToLowerInvariant()
    if ($cocoMd5 -ne $expectedCocoMd5) {
        throw "COCO archive MD5 mismatch: expected $expectedCocoMd5, found $cocoMd5"
    }
    Expand-Archive -LiteralPath $cocoZip -DestinationPath $cocoRoot -Force
}
$cocoCount = (Get-ChildItem -LiteralPath $cocoVal -File | Measure-Object).Count
if ($cocoCount -ne 5000) { throw "expected 5000 COCO val images, found $cocoCount" }

# Official AesFA checkpoint linked by the upstream AAAI 2024 repository.
$aesfaCheckpoint = Join-Path $RepoRoot 'external\AesFA\ckpt\main\main.pth'
if (-not (Test-Path -LiteralPath $aesfaCheckpoint)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $aesfaCheckpoint) | Out-Null
    python -m pip install --disable-pip-version-check gdown
    if ($LASTEXITCODE -ne 0) { throw "gdown installation failed: exit $LASTEXITCODE" }
    python -m gdown 1Y3OutPAsmPmJcnZs07ZVbDFf6nn3RzxR -O $aesfaCheckpoint
    if ($LASTEXITCODE -ne 0) { throw "AesFA checkpoint download failed: exit $LASTEXITCODE" }
}

# Official CSD ViT-L checkpoint release. The evaluation records its SHA-256;
# the upstream repository notes that released weights can differ numerically
# from the paper, so the manuscript reports this limitation explicitly.
$csdCheckpoint = Join-Path $RepoRoot 'external\CSD\checkpoints\pytorch_model.bin'
if (-not (Test-Path -LiteralPath $csdCheckpoint)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $csdCheckpoint) | Out-Null
    & curl.exe -L --fail --retry 10 --retry-delay 5 --continue-at - `
        'https://huggingface.co/tomg-group-umd/CSD-ViT-L/resolve/main/pytorch_model.bin?download=true' `
        -o $csdCheckpoint
    if ($LASTEXITCODE -ne 0) { throw "CSD checkpoint download failed: exit $LASTEXITCODE" }
}

$record = [ordered]@{
    completed_at = (Get-Date).ToString('o')
    coco_val2017_count = $cocoCount
    coco_archive_url = $cocoUrl
    coco_archive_bytes = (Get-Item -LiteralPath $cocoZip).Length
    coco_archive_md5 = (Get-FileHash -LiteralPath $cocoZip -Algorithm MD5).Hash.ToLowerInvariant()
    aesfa_checkpoint_bytes = (Get-Item -LiteralPath $aesfaCheckpoint).Length
    csd_checkpoint_bytes = (Get-Item -LiteralPath $csdCheckpoint).Length
    aesfa_revision = (git -C external\AesFA rev-parse HEAD).Trim()
    csd_revision = (git -C external\CSD rev-parse HEAD).Trim()
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $assetRoot 'assets_ready.json') -Encoding utf8
$record | Format-List
