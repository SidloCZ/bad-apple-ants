# Create .venv and install the three packages (badapple_media, badapple_coastline, badapple_moon).
#   .\setup.ps1                 # uses uv if installed, else the given/default Python 3.11+
#   .\setup.ps1 -Python C:\Python312\python.exe
param([string]$Python = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venvPython = '.\.venv\Scripts\python.exe'
if (-not $Python -and (Get-Command uv -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path -LiteralPath $venvPython)) { uv venv --python 3.12 .venv; if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' } }
    uv pip install --python $venvPython -e '.[test]'
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install dependencies.' }
} else {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (-not $Python) { $Python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' } }
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv; pass -Python with a Python 3.11+ executable.' }
    }
    & $venvPython -m pip install -e '.[test]'
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install dependencies.' }
}
foreach ($tool in 'ffmpeg','ffprobe','cargo') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Write-Warning "$tool is not on PATH; it is needed by the pipelines." }
}
Write-Output 'Ready. Next: .\.venv\Scripts\python.exe -m badapple_media fetch   (source video + fonts), then see README.md.'
