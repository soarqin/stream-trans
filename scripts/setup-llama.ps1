[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw '此脚本需要 Windows x64 PowerShell。' }
if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) { throw '找不到 tar.exe，请使用 Windows 自带的 tar 或将其加入 PATH。' }
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$downloads = Join-Path $projectRoot '.tools/downloads'
$runtime = Join-Path $projectRoot '.tools/llama'
$licenses = Join-Path $runtime 'licenses'
New-Item -ItemType Directory -Path $downloads,$runtime,$licenses -Force | Out-Null
$releaseUrl = 'https://github.com/ggml-org/llama.cpp/releases/download/b11960/'
$assets = @(
    @{ name = 'llama-b11960-bin-win-cuda-12.4-x64.zip'; sha256 = '8d9d819261de10cc2f03921b8b79c779c5c458e21edb60c170ee30a7d462d684' },
    @{ name = 'cudart-llama-bin-win-cuda-12.4-x64.zip'; sha256 = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6' }
)
function Get-VerifiedAsset([string]$Name, [string]$Url, [string]$Sha256) {
    $archive = Join-Path $downloads $Name
    if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $Sha256) {
        Write-Host "下载并校验 $Name"
        $partial = "$archive.partial"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial
        if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ine $Sha256) { throw "下载文件 SHA256 校验失败：$partial。请检查网络后重试。" }
        Move-Item -LiteralPath $partial -Destination $archive -Force
    }
    return $archive
}
foreach ($asset in $assets) {
    $archive = Get-VerifiedAsset $asset.name ($releaseUrl + $asset.name) $asset.sha256
    & tar.exe -xf $archive -C $runtime
    if ($LASTEXITCODE -ne 0) { throw "llama.cpp 运行库解压失败：$archive" }
}
foreach ($name in @('llama-server.exe','llama-server-impl.dll','llama-common.dll','llama.dll','ggml.dll','ggml-base.dll','ggml-cuda.dll','cublas64_12.dll','cublasLt64_12.dll','cudart64_12.dll','libomp.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $runtime $name))) { throw "llama.cpp 运行文件缺失：$name" }
}
$mit = Get-VerifiedAsset 'llama-b11960-LICENSE' 'https://raw.githubusercontent.com/ggml-org/llama.cpp/8642541b255bfb5dc39cd351a178be09013a55d6/LICENSE' '94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d'
$cudaLicense = Get-VerifiedAsset 'cuda-12.4-EULA.html' 'https://docs.nvidia.com/cuda/archive/12.4.0/eula/index.html' '182734e0cee7c6217b911ef88fc92c4a1f487d1003c24dd9dab1597a399fb7c5'
Copy-Item -LiteralPath $mit -Destination (Join-Path $licenses 'LICENSE-llama.cpp') -Force
Copy-Item -LiteralPath (Join-Path $runtime 'LICENSE-LLVM-OpenMP') -Destination $licenses -Force
Copy-Item -LiteralPath $cudaLicense -Destination $licenses -Force
@{
    version = 'b11960'; commit = '8642541b255bfb5dc39cd351a178be09013a55d6'; cuda = '12.4'
    assets = @($assets | ForEach-Object { @{ name = $_.name; url = $releaseUrl + $_.name; sha256 = $_.sha256 } })
    license_sources = @('https://raw.githubusercontent.com/ggml-org/llama.cpp/8642541b255bfb5dc39cd351a178be09013a55d6/LICENSE','https://docs.nvidia.com/cuda/archive/12.4.0/eula/index.html')
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runtime '.streamtrans-runtime.json') -Encoding utf8
Write-Host "llama.cpp b11960 / CUDA 12.4 运行库已准备：$runtime"
