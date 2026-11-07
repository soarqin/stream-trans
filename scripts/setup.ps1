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

$cudaRuntime = & (Join-Path $PSScriptRoot 'setup-cuda-runtime.ps1') -RuntimeDirectory $asrDirectory -LicenseDirectory (Join-Path $asrDirectory 'licenses/cuda-asr')
@{
    version = '0.4.1'; url = $asrUrl; sha256 = $asrSha
    cuda_runtime = $cudaRuntime
    cuda_manifest = $cudaRuntime.manifest
    cuda_packages = $cudaRuntime.packages
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $asrDirectory '.streamtrans-runtime.json') -Encoding utf8

& (Join-Path $PSScriptRoot 'setup-ffmpeg.ps1')
& (Join-Path $PSScriptRoot 'setup-llama.ps1')
$msvcLicenses = Join-Path $projectRoot '.tools/licenses/msvc'
New-Item -ItemType Directory -Path $msvcLicenses -Force | Out-Null
$msvcLicense = Get-VerifiedAsset 'Visual-C-Runtime-2015-2022-License.docx' 'https://visualstudio.microsoft.com/wp-content/uploads/2021/09/Visual-C-Runtime-2015-2022-License-1.docx' 'f1e3d56ceb2ad68aae0711b910375009e651ac5530fa0760f0dea6e81e54fae1'
Copy-Item -LiteralPath $msvcLicense -Destination $msvcLicenses -Force
if (-not $SkipBuild) { & (Join-Path $PSScriptRoot 'build.ps1') }
Write-Host '运行库准备完成；llama.cpp 使用项目内 CUDA 12.8 工具链构建，FFmpeg 通过音频动态库调用，运行机器无需 Python 或 CUDA Toolkit。运行 .\target\release\streamtrans.exe，或执行 .\scripts\dist.ps1 制作分发目录。首次启动默认从 Hugging Face 下载模型；使用 --model-source modelscope 可选择 ModelScope。'
