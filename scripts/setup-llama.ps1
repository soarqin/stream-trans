[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [ValidateRange(1, 64)][int]$Jobs = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot '.tools/llama-cuda128' }
$runtime = [IO.Path]::GetFullPath($OutputDirectory)
& (Join-Path $PSScriptRoot 'build-llama.ps1') -OutputDirectory $runtime -Jobs $Jobs
$cudaRuntime = & (Join-Path $PSScriptRoot 'setup-cuda-runtime.ps1') -RuntimeDirectory $runtime
foreach ($name in @('llama-server.exe','llama.dll','ggml.dll','ggml-base.dll','ggml-cuda.dll','licenses/LICENSE-llama.cpp','.streamtrans-runtime.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $runtime $name) -PathType Leaf)) { throw "llama.cpp 构建产物缺失：$name" }
}
if (-not @(Get-ChildItem -LiteralPath $runtime -File -Filter 'ggml-cpu*.dll').Count) { throw 'llama.cpp 构建产物缺少 CPU 后端。' }
$manifestPath = Join-Path $runtime '.streamtrans-runtime.json'
$metadata = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($metadata.version -ne 'b11960' -or $metadata.commit -ne '8642541b255bfb5dc39cd351a178be09013a55d6' -or $metadata.cuda_build -ne '12.8') {
    throw 'llama.cpp 产物不是固定 b11960 / CUDA 12.8 构建，请检查 build-llama.ps1 输出。'
}
$metadata | Add-Member -NotePropertyName cuda_runtime -NotePropertyValue $cudaRuntime -Force
$metadata | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host "llama.cpp b11960 / CUDA 12.8 构建与运行库已准备：$runtime"
