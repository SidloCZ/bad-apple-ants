# Moon pose scoring (pyramid sampling, cuBLAS GemmEx, top-k merge). Loaded by rust/moon-search.
. (Join-Path $PSScriptRoot '..\common.ps1')
Build-CudaLibrary -Directory $PSScriptRoot -Source 'moon.cu' -Name 'moon_search' -Libraries '-lcublas'
