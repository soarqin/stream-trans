param([int]$Gpu = 0)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectRoot = Split-Path $PSScriptRoot -Parent
$cudaRoot = Join-Path $projectRoot '.tools/cuda'
$downloadRoot = Join-Path $projectRoot '.tools/downloads'
$manifestUrl = 'https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json'
$markerPath = Join-Path $cudaRoot '.streamtrans-cuda.json'
New-Item -ItemType Directory -Path $cudaRoot,$downloadRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $markerPath)) {
    $redistrib = Invoke-RestMethod -Uri $manifestUrl -UseBasicParsing
    foreach ($package in @('cuda_nvcc','cuda_cudart','cuda_cccl','cuda_nvrtc','libcurand','libcublas')) {
        $info = $redistrib.$package.'windows-x86_64'
        if (-not $info) { throw "NVIDIA 清单缺少 Windows CUDA 软件包：$package" }
        $archive = Join-Path $downloadRoot (Split-Path $info.relative_path -Leaf)
        if (-not (Test-Path -LiteralPath $archive)) {
            Write-Host "下载 $package（$($info.size) 字节）"
            Invoke-WebRequest -UseBasicParsing -Uri ("https://developer.download.nvidia.com/compute/cuda/redist/" + $info.relative_path) -OutFile $archive
        }
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $info.sha256) {
            throw "CUDA 文件校验失败，请删除此缓存文件后重试：$archive"
        }
        $stage = Join-Path $downloadRoot ($package + '-unpacked')
        $stageMarker = Join-Path $stage '.streamtrans-extracted'
        if (-not (Test-Path -LiteralPath $stageMarker)) {
            Expand-Archive -LiteralPath $archive -DestinationPath $stage -Force
            Set-Content -LiteralPath $stageMarker -Value $info.sha256 -Encoding ascii
        }
        $packageRoot = Get-ChildItem -LiteralPath $stage -Directory | Select-Object -First 1
        if (-not $packageRoot) { throw "CUDA 压缩包中找不到安装目录：$archive" }
        foreach ($directory in @('bin','include','lib','nvvm')) {
            $source = Join-Path $packageRoot.FullName $directory
            if (Test-Path -LiteralPath $source) {
                Copy-Item -LiteralPath $source -Destination $cudaRoot -Force -Recurse
            }
        }
        Write-Host "已准备 $package"
    }
    @{ manifest = $manifestUrl; version = '12.8.1' } | ConvertTo-Json | Set-Content -LiteralPath $markerPath -Encoding utf8
}

foreach ($required in @('bin/nvcc.exe','bin/cublas64_12.dll','bin/cublasLt64_12.dll','bin/curand64_10.dll','bin/nvrtc64_120_0.dll','lib/x64/cuda.lib','include/cuda.h')) {
    if (-not (Test-Path -LiteralPath (Join-Path $cudaRoot $required))) {
        throw "项目 CUDA 文件缺失：$required。请删除 $markerPath 后重新运行安装脚本。"
    }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw '未找到 Visual Studio C++ Build Tools，请先安装“使用 C++ 的桌面开发”组件。' }
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsRoot) { throw '未找到 Visual Studio C++ 编译器，请安装“使用 C++ 的桌面开发”组件。' }
$vcvars = Join-Path $vsRoot 'VC/Auxiliary/Build/vcvars64.bat'
$env:PATH = (Split-Path $vswhere -Parent) + ';' + $env:PATH
$envFile = Join-Path $downloadRoot 'streamtrans-vcvars.cmd'
Set-Content -LiteralPath $envFile -Value @("@call `"$vcvars`" >nul", '@set') -Encoding ascii
$envLines = & cmd.exe /d /c $envFile
if ($LASTEXITCODE -ne 0) { throw 'Visual Studio C++ 构建环境初始化失败。' }
foreach ($line in $envLines) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}
$env:CUDA_PATH = $cudaRoot
$env:CUDA_ROOT = $cudaRoot
$env:PATH = (Join-Path $cudaRoot 'bin') + ';' + $env:PATH
$computeCap = & nvidia-smi "--id=$Gpu" --query-gpu=compute_cap --format=csv,noheader
if ($LASTEXITCODE -ne 0 -or -not $computeCap) { throw "无法读取 GPU $Gpu 的 CUDA 计算能力，请检查 NVIDIA 驱动。" }
$env:CUDA_COMPUTE_CAP = ($computeCap | Select-Object -First 1).Trim().Replace('.', '')
Write-Host "项目 CUDA 12.8.1 已就绪：$cudaRoot；GPU 计算能力 $env:CUDA_COMPUTE_CAP"
