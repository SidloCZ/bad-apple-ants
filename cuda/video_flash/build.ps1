# Photo-panel luminance limiting for the satellite edition (flash reduction).
. (Join-Path $PSScriptRoot '..\common.ps1')
Build-CudaLibrary -Directory $PSScriptRoot -Source 'video_flash.cu' -Name 'video_flash' -Flags '--fmad=false'
