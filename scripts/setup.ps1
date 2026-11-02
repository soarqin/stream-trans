[CmdletBinding()]
param(
    [ValidateRange(0, 255)][int]$Gpu = 0,
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$downloads = Join-Path $projectRoot '.tools/downloads'
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw '此脚本需要 Windows x64 PowerShell。' }
foreach ($command in @('tar.exe', 'nvidia-smi')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "找不到 $command，请安装后重新运行。" }
}
& nvidia-smi "--id=$Gpu" --query-gpu=name --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw "无法访问 GPU $Gpu，请检查 NVIDIA 驱动和 GPU 编号。" }
New-Item -ItemType Directory -Path $downloads -Force | Out-Null

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

$asrAsset = 'transcribe-native-0.4.1-windows-x86_64-cuda.tar.gz'
$asrUrl = "https://github.com/handy-computer/transcribe.cpp/releases/download/v0.4.1/$asrAsset"
$asrSha = 'ba43fb21a3fe0ab2630e0614fbd742e039c389c68ad10fe23aca903f986adc5b'
$archive = Get-VerifiedAsset $asrAsset $asrUrl $asrSha
$asrRoot = Join-Path $projectRoot '.tools/transcribe'
New-Item -ItemType Directory -Path $asrRoot -Force | Out-Null
& tar.exe -xzf $archive -C $asrRoot
if ($LASTEXITCODE -ne 0) { throw 'ASR 原生库解压失败，请检查下载缓存后重试。' }
$asrDirectory = Join-Path $asrRoot 'transcribe-native-windows-x86_64-cuda'
if (-not (Test-Path -LiteralPath (Join-Path $asrDirectory 'transcribe.dll'))) { throw 'ASR 压缩包中缺少 transcribe.dll。' }

# ASR 预编译库使用 CUDA 12.8；只准备运行 DLL，不安装 CUDA 编译工具。
# URL、版本和 SHA256 来自 NVIDIA 的 redistrib_12.8.1.json。
$cudaBase = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$cudaPackages = @(
    @{
        Name = 'cuda_cudart'; Archive = 'cuda_cudart-windows-x86_64-12.8.90-archive.zip'
        Sha256 = '4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b'
        Dlls = @{ 'cudart64_12.dll' = 'c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344' }
    },
    @{
        Name = 'libcublas'; Archive = 'libcublas-windows-x86_64-12.8.4.1-archive.zip'
        Sha256 = '57a470112cec7e112c95253dde8b3c7184d795dbd92b0bde77a4cb7f8c94c8aa'
        Dlls = @{
            'cublas64_12.dll' = '9513540e4ec4c51ee9e7304138c2cc255c29a8c181f9e80c38efa25738becd99'
            'cublasLt64_12.dll' = 'b199d1ff892a81b7fd3d57ba1781549609b41500b36008fef326038393ad46c7'
        }
    }
)
$cudaLicenses = Join-Path $asrDirectory 'licenses/cuda-asr'
New-Item -ItemType Directory -Path $cudaLicenses -Force | Out-Null
foreach ($package in $cudaPackages) {
    $stage = Join-Path $downloads ($package.Name + '-unpacked')
    $packageRoot = Join-Path $stage ([IO.Path]::GetFileNameWithoutExtension($package.Archive))
    $existingBin = Join-Path $projectRoot '.tools/cuda/bin'
    $canReuse = Test-Path -LiteralPath (Join-Path $packageRoot 'LICENSE')
    foreach ($dll in $package.Dlls.Keys) {
        $candidate = Join-Path $existingBin $dll
        if (-not (Test-Path -LiteralPath $candidate) -or (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash -ine $package.Dlls[$dll]) { $canReuse = $false }
    }
    if (-not $canReuse) {
        $url = $cudaBase + $package.Name + '/windows-x86_64/' + $package.Archive
        $cudaArchive = Get-VerifiedAsset $package.Archive $url $package.Sha256
        New-Item -ItemType Directory -Path $stage -Force | Out-Null
        & tar.exe -xf $cudaArchive -C $stage
        if ($LASTEXITCODE -ne 0) { throw "CUDA 运行组件解压失败：$($package.Archive)" }
        $existingBin = Join-Path $packageRoot 'bin'
    }
    foreach ($dll in $package.Dlls.Keys) {
        $source = Join-Path $existingBin $dll
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ine $package.Dlls[$dll]) { throw "CUDA 运行 DLL 校验失败：$source" }
        Copy-Item -LiteralPath $source -Destination $asrDirectory -Force
    }
    Copy-Item -LiteralPath (Join-Path $packageRoot 'LICENSE') -Destination (Join-Path $cudaLicenses ($package.Name + '-LICENSE')) -Force
}
@{
    version = '0.4.1'; url = $asrUrl; sha256 = $asrSha
    cuda_manifest = $cudaBase + 'redistrib_12.8.1.json'
    cuda_packages = @($cudaPackages | ForEach-Object { @{ name = $_.Name; url = $cudaBase + $_.Name + '/windows-x86_64/' + $_.Archive; sha256 = $_.Sha256 } })
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $asrDirectory '.streamtrans-runtime.json') -Encoding utf8

& (Join-Path $PSScriptRoot 'setup-llama.ps1')
$msvcLicenses = Join-Path $projectRoot '.tools/licenses/msvc'
New-Item -ItemType Directory -Path $msvcLicenses -Force | Out-Null
$msvcLicense = Get-VerifiedAsset 'Visual-C-Runtime-2015-2022-License.docx' 'https://visualstudio.microsoft.com/wp-content/uploads/2021/09/Visual-C-Runtime-2015-2022-License-1.docx' 'f1e3d56ceb2ad68aae0711b910375009e651ac5530fa0760f0dea6e81e54fae1'
Copy-Item -LiteralPath $msvcLicense -Destination $msvcLicenses -Force
if (-not $SkipBuild) { & (Join-Path $PSScriptRoot 'build.ps1') }
Write-Host '运行库准备完成，不需要 Python 或 CUDA Toolkit。FFmpeg 请自行准备；运行 .\target\release\streamtrans.exe，或执行 .\scripts\dist.ps1 制作分发目录。首次启动会从 ModelScope 下载模型。'
