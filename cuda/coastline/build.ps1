# Coastline search kernels (sampling, features, FFT refinement). --fmad=false keeps
# the geometry bit-identical to the CPU reference used by the validation scripts.
. (Join-Path $PSScriptRoot '..\common.ps1')
Build-CudaLibrary -Directory $PSScriptRoot -Source 'coastline.cu' -Name 'coastline_cuda' -Flags '--fmad=false' -Libraries '-lcublas -lcufft' -Extra @{ geometry_fma = $false }
