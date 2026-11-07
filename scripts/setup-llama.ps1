[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [ValidateRange(1, 64)][int]$Jobs = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot '.tools/llama-embedded' }
$runtime = [IO.Path]::GetFullPath($OutputDirectory)
& (Join-Path $PSScriptRoot 'build-llama.ps1') -OutputDirectory $runtime -Jobs $Jobs
$cudaRuntime = & (Join-Path $PSScriptRoot 'setup-cuda-runtime.ps1') -RuntimeDirectory $runtime
foreach ($name in @('llama-server-impl.dll','llama-common.dll','llama.dll','mtmd.dll','ggml.dll','ggml-base.dll','ggml-cuda.dll','licenses/LICENSE-llama.cpp','.streamtrans-runtime.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $runtime $name) -PathType Leaf)) { throw "llama.cpp 构建产物缺失：$name" }
}
if (-not @(Get-ChildItem -LiteralPath $runtime -File -Filter 'ggml-cpu*.dll').Count) { throw 'llama.cpp 构建产物缺少 CPU 后端。' }
$manifestPath = Join-Path $runtime '.streamtrans-runtime.json'
$metadata = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($metadata.version -ne 'b11960' -or $metadata.commit -ne '8642541b255bfb5dc39cd351a178be09013a55d6' -or $metadata.cuda_build -ne '12.8') {
    throw 'llama.cpp 产物不是固定 b11960 / CUDA 12.8 构建，请检查 build-llama.ps1 输出。'
}
$exports = @('streamtrans_server_abi_version','streamtrans_server_create','streamtrans_server_run','streamtrans_server_request_stop','streamtrans_server_status','streamtrans_server_destroy')
if (-not $metadata.PSObject.Properties['embedded_server'] -or $metadata.embedded_server -isnot [pscustomobject]) {
    throw 'llama.cpp 缺少进程内服务构建记录，请检查 build-llama.ps1 输出。'
}
$embedded = $metadata.embedded_server
if (-not $embedded.PSObject.Properties['abi_version'] -or $embedded.abi_version -ne 1 -or
    -not $embedded.PSObject.Properties['library'] -or $embedded.library -cne 'llama-server-impl.dll' -or
    -not $embedded.PSObject.Properties['exports'] -or @($embedded.exports).Count -ne $exports.Count -or
    @(Compare-Object $exports @($embedded.exports) -CaseSensitive).Count -or
    (Test-Path -LiteralPath (Join-Path $runtime 'llama-server.exe'))) {
    throw 'llama.cpp 进程内服务 ABI、动态库或导出入口不匹配，或仍包含服务 EXE，请检查 build-llama.ps1 输出。'
}
$metadata | Add-Member -NotePropertyName cuda_runtime -NotePropertyValue $cudaRuntime -Force
$metadata | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host "llama.cpp b11960 / CUDA 12.8 进程内服务动态库与运行库已准备：$runtime"
