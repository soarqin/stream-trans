[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RuntimeDirectory,
    [string]$LicenseDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw '此脚本需要 Windows x64 PowerShell。' }
if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) { throw '找不到 tar.exe，请使用 Windows 自带的 tar 或将其加入 PATH。' }
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$downloads = Join-Path $projectRoot '.tools/downloads'
$runtime = [IO.Path]::GetFullPath($RuntimeDirectory)
if (-not $LicenseDirectory) { $LicenseDirectory = Join-Path $runtime 'licenses/cuda' }
$licenses = [IO.Path]::GetFullPath($LicenseDirectory)
New-Item -ItemType Directory -Path $downloads,$runtime,$licenses -Force | Out-Null
$base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$licenseHash = 'e2c71babfd18a8e69542dd7e9ca018f9caa438094001a58e6bc4d8c999bf0d07'
$packages = @(
    @{
        Name = 'cuda_cudart'; Version = '12.8.90'; Archive = 'cuda_cudart-windows-x86_64-12.8.90-archive.zip'
        Sha256 = '4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b'
        Dlls = @{ 'cudart64_12.dll' = 'c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344' }
    },
    @{
        Name = 'libcublas'; Version = '12.8.4.1'; Archive = 'libcublas-windows-x86_64-12.8.4.1-archive.zip'
        Sha256 = '57a470112cec7e112c95253dde8b3c7184d795dbd92b0bde77a4cb7f8c94c8aa'
        Dlls = @{
            'cublas64_12.dll' = '9513540e4ec4c51ee9e7304138c2cc255c29a8c181f9e80c38efa25738becd99'
            'cublasLt64_12.dll' = 'b199d1ff892a81b7fd3d57ba1781549609b41500b36008fef326038393ad46c7'
        }
    }
)

function Test-VerifiedFile([string]$Path, [string]$Sha256) {
    return (Test-Path -LiteralPath $Path -PathType Leaf) -and (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ieq $Sha256
}

function Copy-VerifiedFile([string]$Source, [string]$Destination, [string]$Sha256) {
    if (-not (Test-VerifiedFile $Source $Sha256)) { throw "CUDA 运行文件校验失败：$Source" }
    if (-not (Test-VerifiedFile $Destination $Sha256)) { Copy-Item -LiteralPath $Source -Destination $Destination -Force }
}

foreach ($package in $packages) {
    $stage = Join-Path $downloads ($package.Name + '-unpacked')
    $packageRoot = Join-Path $stage ([IO.Path]::GetFileNameWithoutExtension($package.Archive))
    $bins = @(
        $runtime,
        (Join-Path $projectRoot '.tools/transcribe/transcribe-native-windows-x86_64-cuda'),
        (Join-Path $projectRoot '.tools/llama'),
        (Join-Path $packageRoot 'bin'),
        (Join-Path $projectRoot '.tools/cuda/bin')
    )
    $licenseCandidates = @(
        (Join-Path $licenses ($package.Name + '-LICENSE')),
        (Join-Path $projectRoot ('.tools/transcribe/transcribe-native-windows-x86_64-cuda/licenses/cuda-asr/' + $package.Name + '-LICENSE')),
        (Join-Path $projectRoot ('.tools/llama/licenses/cuda/' + $package.Name + '-LICENSE')),
        (Join-Path $packageRoot 'LICENSE')
    )
    $bin = $null
    foreach ($candidate in $bins) {
        $valid = $true
        foreach ($dll in $package.Dlls.Keys) { if (-not (Test-VerifiedFile (Join-Path $candidate $dll) $package.Dlls[$dll])) { $valid = $false; break } }
        if ($valid) { $bin = $candidate; break }
    }
    $license = $null
    foreach ($candidate in $licenseCandidates) { if (Test-VerifiedFile $candidate $licenseHash) { $license = $candidate; break } }
    if (-not $bin -or -not $license) {
        $archive = Join-Path $downloads $package.Archive
        if (-not (Test-VerifiedFile $archive $package.Sha256)) {
            Write-Host "下载并校验 $($package.Archive)"
            $partial = "$archive.partial"
            Invoke-WebRequest -UseBasicParsing -Uri ($base + $package.Name + '/windows-x86_64/' + $package.Archive) -OutFile $partial
            if (-not (Test-VerifiedFile $partial $package.Sha256)) { throw "CUDA 压缩包 SHA256 校验失败：$partial。请检查网络后重试。" }
            Move-Item -LiteralPath $partial -Destination $archive -Force
        }
        New-Item -ItemType Directory -Path $stage -Force | Out-Null
        & tar.exe -xf $archive -C $stage
        if ($LASTEXITCODE -ne 0) { throw "CUDA 运行组件解压失败：$archive" }
        $bin = Join-Path $packageRoot 'bin'
        $license = Join-Path $packageRoot 'LICENSE'
    }
    foreach ($dll in $package.Dlls.Keys) { Copy-VerifiedFile (Join-Path $bin $dll) (Join-Path $runtime $dll) $package.Dlls[$dll] }
    Copy-VerifiedFile $license (Join-Path $licenses ($package.Name + '-LICENSE')) $licenseHash
}

[pscustomobject]@{
    version = '12.8.1'; cudart = '12.8.90'; cublas = '12.8.4.1'; manifest = $base + 'redistrib_12.8.1.json'
    packages = @($packages | ForEach-Object { @{ name = $_.Name; version = $_.Version; url = $base + $_.Name + '/windows-x86_64/' + $_.Archive; sha256 = $_.Sha256 } })
    files = @($packages | ForEach-Object { foreach ($name in $_.Dlls.Keys) { @{ name = $name; sha256 = $_.Dlls[$name] } } })
    license_sha256 = $licenseHash
}
