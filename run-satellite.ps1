# Satellite edition: every frame of Bad Apple!! as a real piece of Earth's land and water,
# shown with the satellite photograph of the same place.
#
# The published video was built in stages, each one keeping a frame's previous match
# unless the new search scored better on the original 10 m mask. This script runs those
# stages in the same order; each writes its own output/<stage> directory and resumes.
#
#   ./run-satellite.ps1                   # everything, in order (days of downloads and GPU time)
#   ./run-satellite.ps1 -Stage precision  # one stage; earlier stages' outputs must exist
#
#   fetch      mask (7.3 GB), Natural Earth, source video and fonts
#   baseline   output/coastline                  first global match + Sentinel-2 photos
#   wide       output/coastline-wide-rust        6-768 km views, Rust/Rayon, lossless 10 m cache
#   repair     output/coastline-wide-rust-fixed  re-fetch white-out / snow photos
#   landsat    output/coastline-wide-rust-landsat  Landsat 8/9 for frames Sentinel-2 could not cover
#   fullres    output/coastline-fullres-cuda-v2  CUDA search at the source's full 1444x1080
#   precision  output/coastline-precision-v4     contour-aware CUDA search, mirroring, new photos
#   edition    output/coastline-precision-v4-2560x1920-30p-native-av1   the published video
#   landwater  output/landwater-osm              land & water + OpenStreetMap edition
#   site       output/landwater-site/web         data for the place-map website
param(
    [ValidateSet('all','fetch','baseline','wide','repair','landsat','fullres','precision','edition','landwater','site')][string]$Stage = 'all',
    [ValidateRange(1,61)][int]$ImageWorkers = 16,
    [string]$Site = 'output/landwater-site/web'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup.ps1 first.' }
$env:OMP_NUM_THREADS = '2'; $env:OPENBLAS_NUM_THREADS = '2'; $env:RAYON_NUM_THREADS = '16'; $env:GDAL_NUM_THREADS = '1'
function Test-Stage([string]$Name) { return $Stage -eq 'all' -or $Stage -eq $Name }
function Invoke-Python {
    param([string[]]$Arguments)
    & $python -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $($Arguments -join ' '). Completed work stays cached." }
}
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Exe $($Arguments -join ' ')" }
}
function Build-Rust([string]$Crate) { Invoke-Native 'cargo' @('build','--release','--locked','--manifest-path',"rust/$Crate/Cargo.toml") }
function Build-Cuda([string]$Name) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File "cuda/$Name/build.ps1"
    if ($LASTEXITCODE -ne 0) { throw "CUDA build failed: $Name" }
}

if (Test-Stage 'fetch') {
    if (-not (Test-Path -LiteralPath 'assets/badapple.mp4')) { Invoke-Python @('-m','badapple_media','fetch') }
    else { Invoke-Python @('-m','badapple_media','fetch','--no-video') }
    Invoke-Python @('-m','badapple_coastline','fetch','--workers','4')
}
if (Test-Stage 'baseline') {
    # 0.01 degree search atlas, 32,000 coastal centres x 24/48/96 km x 30 degree steps.
    Invoke-Python @('-m','badapple_coastline','atlas')
    Invoke-Python @('-m','badapple_coastline','build','--centers','32000')
    Invoke-Python @('-m','badapple_coastline','match','--output','output/coastline','--top-k','6','--metric','iou','--backend','python')
    Invoke-Python @('-m','badapple_coastline','imagery','--output','output/coastline','--workers',"$ImageWorkers",'--max-scenes','16')
    Invoke-Python @('-m','badapple_coastline','render','--output','output/coastline','--encoder','h264_nvenc')
}
if (Test-Stage 'wide') {
    Build-Rust 'coastline-matcher'; Build-Rust 'coastline-native'
    Invoke-Python @('-m','badapple_coastline','atlas','--atlas','cache/coastline-atlas-fine','--resolution','0.0025')
    Invoke-Python @('-m','badapple_coastline','native-cache','--atlas','cache/coastline-atlas-fine')
    Invoke-Python @('-m','badapple_coastline','build','--atlas','cache/coastline-atlas-fine','--dictionary','cache/coastline-dictionary-wide',
        '--centers','32000','--widths','6','12','24','48','96','192','384','768')
    Invoke-Python @('-m','badapple_coastline','improve','--baseline','output/coastline','--dictionary','cache/coastline-dictionary-wide',
        '--output','output/coastline-wide-rust','--backend','rust')
    Invoke-Python @('-m','badapple_coastline','imagery','--output','output/coastline-wide-rust','--workers',"$ImageWorkers",'--max-scenes','96')
    Invoke-Python @('-m','badapple_coastline','render','--output','output/coastline-wide-rust','--encoder','h264_nvenc')
}
if (Test-Stage 'repair') {
    Invoke-Python @('-m','badapple_coastline.repair','--workers',"$ImageWorkers",'--max-scenes','12')
    Invoke-Python @('-m','badapple_coastline','render','--output','output/coastline-wide-rust-fixed','--encoder','h264_nvenc')
}
if (Test-Stage 'landsat') {
    Invoke-Python @('-m','badapple_coastline.landsat','--workers','8','--max-scenes','24')
    # Frames still without a full photograph: later years, then the saved alternative views.
    Invoke-Python @('-m','badapple_coastline.landsat','--workers','1','--max-scenes','64','--interval','2022-01-01T00:00:00Z/2025-12-31T23:59:59Z')
    Invoke-Python @('-m','badapple_coastline.landsat','--workers','1','--alternatives-only')
    Invoke-Python @('-m','badapple_coastline','render','--output','output/coastline-wide-rust-landsat','--encoder','h264_nvenc')
}
if (Test-Stage 'fullres') {
    Build-Cuda 'coastline'
    Invoke-Native 'cargo' @('test','--release','--locked','--manifest-path','rust/coastline-native-index/Cargo.toml')
    Build-Rust 'coastline-native-index'
    if (-not (Test-Path 'cache/coastline-native-dictionary/centers.json')) {
        Invoke-Native './rust/coastline-native-index/target/release/coastline-native-index.exe' @('cache/coastline-native-rle/manifest.json','cache/coastline-native-dictionary','64000','375')
    }
    Invoke-Python @('scripts/prepare_fullres_coastline_frames.py')
    Invoke-Python @('-m','badapple_coastline.cuda_dictionary')
    Invoke-Python @('-m','badapple_coastline.cuda_search')
    Invoke-Python @('-m','badapple_coastline.native_imagery','--workers',"$ImageWorkers")
    Invoke-Python @('-m','badapple_coastline.retry_native_primary','--workers',"$ImageWorkers")
    Invoke-Python @('-m','badapple_coastline.native_finalize')
}
if (Test-Stage 'precision') {
    Build-Cuda 'coastline'
    Invoke-Python @('-m','badapple_coastline.precision_pipeline','--workers',"$ImageWorkers",'--include-satellite-photos')
}
if (Test-Stage 'edition') {
    Build-Cuda 'video_flash'; Build-Rust 'coastline-native'
    Invoke-Python @('-m','badapple_coastline.native_2560')
}
if (Test-Stage 'landwater') {
    # OpenStreetMap panels are drawn by MapLibre in headless Chromium (npx playwright install chromium).
    Build-Cuda 'video_flash'; Build-Rust 'coastline-native'
    Invoke-Python @('-m','badapple_coastline.landwater_osm','prepare')
    Invoke-Python @('-m','badapple_coastline.landwater_osm','all','--workers','14')
}
if (Test-Stage 'site') {
    Invoke-Python @('-m','badapple_coastline.landwater_web','--site',$Site)
}
