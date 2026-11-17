# Shared nvcc build for the CUDA DLLs (Windows, MSVC + CUDA Toolkit).
#
#   CUDA_PATH  CUDA Toolkit root (default: newest under "NVIDIA GPU Computing Toolkit\CUDA")
#   CUDA_ARCH  GPU architecture, e.g. sm_89 (RTX 40), sm_86 (RTX 30), sm_120 (RTX 50). Default sm_89.
#
# The DLL name carries the source hash and architecture, so an edited source or a
# different GPU never loads a stale build. build/current.json tells the Python and
# Rust callers which DLL to load.

function Find-CudaRoot {
    if ($env:CUDA_PATH -and (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) { return $env:CUDA_PATH }
    $base = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    $found = Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'bin\nvcc.exe') } |
        Sort-Object { [version]($_.Name.TrimStart('v')) } | Select-Object -Last 1
    if (-not $found) { throw 'CUDA Toolkit not found: install it or set CUDA_PATH' }
    return $found.FullName
}

function Find-VcVars {
    $vsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vsWhere)) { throw 'Visual Studio (C++ build tools) not found' }
    $vsRoot = & $vsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath |
        Select-Object -First 1
    if (-not $vsRoot) { throw 'Visual Studio with the C++ x64 build tools not found' }
    $vcvars = Join-Path $vsRoot.Trim() 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) { throw "vcvars64.bat not found under $vsRoot" }
    return $vcvars
}

function Build-CudaLibrary {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Name,
        [string]$Flags = '',
        [string]$Libraries = '',
        [hashtable]$Extra = @{}
    )
    $ErrorActionPreference = 'Stop'
    $arch = if ($env:CUDA_ARCH) { $env:CUDA_ARCH } else { 'sm_89' }
    $cudaRoot = Find-CudaRoot
    $setup = Find-VcVars
    $compiler = Join-Path $cudaRoot 'bin\nvcc.exe'
    $sourcePath = Join-Path $Directory $Source
    $buildRoot = Join-Path $Directory 'build'
    New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
    $hash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLower()
    $library = Join-Path $buildRoot "${Name}_$($hash.Substring(0,16))_$arch.dll"
    if (-not (Test-Path -LiteralPath $library)) {
        $compile = 'call "{0}" >nul && "{1}" -std=c++17 -O3 {2} -arch={3} --shared -Xcompiler=/MD -Xcompiler=/EHsc -Xcompiler=/utf-8 -DWIN32_LEAN_AND_MEAN -DNOMINMAX "{4}" -o "{5}" {6}' -f $setup, $compiler, $Flags, $arch, $sourcePath, $library, $Libraries
        Push-Location $buildRoot
        try { & cmd.exe /d /s /c $compile; if ($LASTEXITCODE -ne 0) { throw "CUDA compilation failed: $Source" } }
        finally { Pop-Location }
    }
    $record = @{ library = $library; source_sha256 = $hash; cuda_root = $cudaRoot; architecture = $arch } + $Extra
    $record | ConvertTo-Json | Set-Content (Join-Path $buildRoot 'current.json') -Encoding utf8
    Get-Content (Join-Path $buildRoot 'current.json')
}
