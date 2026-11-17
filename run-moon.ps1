# Moon edition: every frame of Bad Apple!! as the closest real lunar surface.
#
#   ./run-moon.ps1                  # data -> frames -> verify -> search -> refine -> crops -> encode
#   ./run-moon.ps1 -Stage search    # any stage can be re-run or resumed on its own
#
# Needs: Windows, an NVIDIA GPU with the CUDA Toolkit (cuBLAS), Visual Studio C++ tools,
# Rust, FFmpeg on PATH and the .venv from setup.ps1. Downloads ~100 GB (Kaguya TC
# overviews + the 6 GB WAC mosaic); the search is about a day on an RTX 4090.
param(
    [ValidateSet('all','data','frames','verify','search','refine','crops','encode')][string]$Stage = 'all',
    [string]$InputVideo = 'assets/badapple.mp4',
    [ValidateSet('av1_nvenc','libx264','h264_nvenc')][string]$Encoder = 'av1_nvenc',
    [int]$Rotations = 16,
    [int]$Workers = 24,
    [int]$IngestShards = 3,
    [double]$ShardGB = 14
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup.ps1 first.' }
$search = Join-Path $PSScriptRoot 'rust/moon-search/target/release/moon-search.exe'
function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Moon stage failed: $($Arguments[0..2] -join ' '). Finished work stays cached." }
}
function Test-Stage([string[]]$Names) { return $Stage -eq 'all' -or $Names -contains $Stage }

if (Test-Stage 'data') {
    if (-not (Test-Path -LiteralPath $InputVideo)) { Invoke-Step $python @('-u','-m','badapple_media','fetch') }
    else { Invoke-Step $python @('-u','-m','badapple_media','fetch','--no-video') }
    if (-not (Test-Path 'assets/moon/raw/names/MOON_nomenclature_center_pts.dbf')) { Invoke-Step $python @('-u','-m','badapple_moon','fetch-names') }
    Invoke-Step $python @('-u','-m','badapple_moon','fetch-wac')
    # Single-image (mono) series: one <=95 m/px overview per image, read by HTTP range requests.
    if (-not (Test-Path 'cache/moon/tc_catalog.jsonl')) { Invoke-Step $python @('-u','-m','badapple_moon','catalog') }
    Invoke-Step $python @('-u','-m','badapple_moon','ingest-tc','--workers','64')
    Invoke-Step $python @('-u','-m','badapple_moon','qa-images')
    Invoke-Step $python @('-u','-m','badapple_moon','ingest-global')
    # Stereo series (TC1W/TC2W), split into shards ingested in parallel. The Spectral Profiler
    # support series ('sp') is not used: its strong onboard compression shows as blocks in the crops.
    $catalog = 'cache/moon/tc_catalog_stereo.jsonl'
    if (-not (Test-Path $catalog)) { Invoke-Step $python @('-u','-m','badapple_moon','catalog','--series','stereo') }
    $jobs = @()
    for ($k = 0; $k -lt $IngestShards; $k++) {
        $job = Start-Process -FilePath $python -NoNewWindow -PassThru -ArgumentList @('-u','-m','badapple_moon','ingest-tc',
            '--catalog',$catalog,'--part',"$k",'--parts',"$IngestShards",'--output',"cache/moon/tiles-stereo-$k",'--workers','40')
        $null = $job.Handle  # keeps the exit code readable after the process ends
        $jobs += $job
    }
    $jobs | Wait-Process
    foreach ($job in $jobs) { if ($job.ExitCode -ne 0) { throw "Stereo ingest shard failed (exit $($job.ExitCode)). Finished work stays cached." } }
    for ($k = 0; $k -lt $IngestShards; $k++) { Invoke-Step $python @('-u','-m','badapple_moon','qa-images','--tiles',"cache/moon/tiles-stereo-$k") }
}
$extra = (Get-ChildItem -Directory -Path 'cache/moon' -Filter 'tiles-stereo-*' -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'tc_index.jsonl') } | Sort-Object Name | ForEach-Object { "cache/moon/$($_.Name)" }) -join ','
# Windows PowerShell 5.1 drops empty arguments, so pass --extra-tiles only when there are stereo tiles.
$tiles = @('--shard-gb',"$ShardGB")
if ($extra) { $tiles = @('--extra-tiles',$extra) + $tiles }
if (Test-Stage @('frames','verify','search','refine')) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File cuda/moon/build.ps1
    if ($LASTEXITCODE -ne 0) { throw 'CUDA build failed' }
    cargo build --release --locked --manifest-path rust/moon-search/Cargo.toml
    if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
}
if (Test-Stage 'frames') { Invoke-Step $search @('frames','--input',$InputVideo) }
if (Test-Stage 'verify') { Invoke-Step $search @('verify','--rotations',"$Rotations") }
if (Test-Stage 'search') { Invoke-Step $search (@('search','--rotations',"$Rotations") + $tiles) }
if (Test-Stage 'refine') { Invoke-Step $search (@('refine','--out','output/moon/matches.json') + $tiles) }
if (Test-Stage 'crops') { Invoke-Step $python @('-u','-m','badapple_moon','crops','--workers',"$Workers") }
if (Test-Stage 'encode') { Invoke-Step $python @('-u','-m','badapple_moon','encode','--video',$InputVideo,'--encoder',$Encoder) }
