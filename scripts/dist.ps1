[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [switch]$SkipBuild,
    [string]$Ffmpeg,
    [string]$FfmpegLicenseDirectory,
    [string]$FfmpegSourceUrl,
    [string]$MsvcRedistDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot 'dist/StreamTrans' }
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) {
    $existing = Get-Item -LiteralPath $output -Force
    if (-not $existing.PSIsContainer -or ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -or @(Get-ChildItem -LiteralPath $output -Force).Count -gt 0) {
        throw "输出目录必须不存在或为空，脚本不会覆盖非空目录：$output。请指定新的 -OutputDirectory。"
    }
}

$asr = Join-Path $projectRoot '.tools/transcribe/transcribe-native-windows-x86_64-cuda'
$llama = Join-Path $projectRoot '.tools/llama'
$msvcLicenses = Join-Path $projectRoot '.tools/licenses/msvc'
foreach ($required in @(
    (Join-Path $asr 'transcribe.dll'), (Join-Path $asr 'ggml-cuda.dll'),
    (Join-Path $asr 'cudart64_12.dll'), (Join-Path $asr 'cublas64_12.dll'), (Join-Path $asr 'cublasLt64_12.dll'),
    (Join-Path $asr 'licenses/LICENSE'), (Join-Path $asr 'licenses/cuda-asr/cuda_cudart-LICENSE'),
    (Join-Path $asr 'licenses/cuda-asr/libcublas-LICENSE'), (Join-Path $asr '.streamtrans-runtime.json'),
    (Join-Path $llama 'llama-server.exe'), (Join-Path $llama 'llama-server-impl.dll'),
    (Join-Path $llama 'mtmd.dll'), (Join-Path $llama 'cublas64_12.dll'), (Join-Path $llama 'cublasLt64_12.dll'),
    (Join-Path $llama 'cudart64_12.dll'), (Join-Path $llama 'licenses/LICENSE-llama.cpp'),
    (Join-Path $llama 'licenses/cuda-12.4-EULA.html'), (Join-Path $llama '.streamtrans-runtime.json'),
    (Join-Path $msvcLicenses 'Visual-C-Runtime-2015-2022-License.docx')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "缺少分发依赖：$required。请先执行 .\scripts\setup.ps1。" }
}

# 优先使用已准备的可分发静态包；不会下载或打包模型、测试音频。
$knownFfmpegRoot = Join-Path $projectRoot '.tools/ffmpeg/ffmpeg-9.0.2-essentials_build'
if (-not $Ffmpeg) {
    $knownExecutable = Join-Path $knownFfmpegRoot 'bin/ffmpeg.exe'
    if (Test-Path -LiteralPath $knownExecutable) { $Ffmpeg = $knownExecutable }
    else {
        $command = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
        if ($command) { $Ffmpeg = $command.Source }
    }
}
if (-not $Ffmpeg -or -not (Test-Path -LiteralPath $Ffmpeg -PathType Leaf)) { throw '找不到 FFmpeg，请通过 -Ffmpeg 指定可分发的 Windows x64 ffmpeg.exe。' }
$Ffmpeg = (Resolve-Path -LiteralPath $Ffmpeg).Path
$ffmpegVersion = (& $Ffmpeg -version | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "FFmpeg 版本查询失败：$Ffmpeg。请指定可以独立运行的静态构建。" }
if ($ffmpegVersion -match '--enable-nonfree') { throw "此 FFmpeg 包含 nonfree 组件，不可再分发：$Ffmpeg。请通过 -Ffmpeg 指定可分发构建及其许可证。" }
if ($ffmpegVersion -match '--enable-shared') { throw "此 FFmpeg 依赖共享 DLL：$Ffmpeg。请通过 -Ffmpeg 指定可独立运行的静态构建。" }
if (-not $FfmpegLicenseDirectory) {
    $FfmpegLicenseDirectory = Split-Path $Ffmpeg -Parent
    if ((Split-Path $FfmpegLicenseDirectory -Leaf) -ieq 'bin') { $FfmpegLicenseDirectory = Split-Path $FfmpegLicenseDirectory -Parent }
}
if (-not (Test-Path -LiteralPath $FfmpegLicenseDirectory -PathType Container)) { throw '找不到 FFmpeg 许可证目录，请指定 -FfmpegLicenseDirectory。' }
$ffmpegNotices = @(Get-ChildItem -LiteralPath $FfmpegLicenseDirectory -File | Where-Object { $_.Name -match '^(LICENSE|COPYING|NOTICE|README|SOURCE)' })
if (-not @($ffmpegNotices | Where-Object { $_.Name -match '^(LICENSE|COPYING)' }).Count) { throw 'FFmpeg 目录中缺少许可证正文，请通过 -FfmpegLicenseDirectory 指定包含 LICENSE 或 COPYING 的目录。' }
if (-not $FfmpegSourceUrl) {
    foreach ($notice in $ffmpegNotices) {
        $sourceMatch = [regex]::Match((Get-Content -LiteralPath $notice.FullName -Raw), '(?im)^Source Code:\s*(https://\S+)')
        if ($sourceMatch.Success) { $FfmpegSourceUrl = $sourceMatch.Groups[1].Value; break }
    }
}
if (-not $FfmpegSourceUrl -or $FfmpegSourceUrl -notmatch '^https://') { throw '缺少 FFmpeg 对应源码地址，请通过 -FfmpegSourceUrl 指定 HTTPS 源码地址。' }

if (-not $MsvcRedistDirectory) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw '找不到 Visual Studio Redist，请指定 -MsvcRedistDirectory 为 x64 Microsoft.VC143.CRT 目录。' }
    $vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $vsRoot) { throw '无法定位 Visual Studio C++ Redist，请指定 -MsvcRedistDirectory。' }
    $versions = @(Get-ChildItem -LiteralPath (Join-Path $vsRoot 'VC/Redist/MSVC') -Directory | Where-Object { $_.Name -match '^\d+(\.\d+)+$' } | Sort-Object { [version]$_.Name } -Descending)
    foreach ($version in $versions) {
        $crt = Get-ChildItem -LiteralPath (Join-Path $version.FullName 'x64') -Directory -Filter 'Microsoft.VC*.CRT' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($crt) { $MsvcRedistDirectory = $crt.FullName; break }
    }
}
foreach ($dll in @('vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll')) {
    if (-not $MsvcRedistDirectory -or -not (Test-Path -LiteralPath (Join-Path $MsvcRedistDirectory $dll) -PathType Leaf)) { throw "MSVC app-local 运行库缺少 $dll，请指定完整的 x64 -MsvcRedistDirectory。" }
}

if (-not $SkipBuild) { & (Join-Path $PSScriptRoot 'build.ps1') }
$executable = Join-Path $projectRoot 'target/release/streamtrans.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw '找不到 target/release/streamtrans.exe，请先执行 .\scripts\build.ps1。' }
$runtimeAsr = Join-Path $output 'runtime/asr'
$runtimeLlama = Join-Path $output 'runtime/llama'
$runtimeFfmpeg = Join-Path $output 'runtime/ffmpeg'
$licenseRoot = Join-Path $output 'licenses'
New-Item -ItemType Directory -Path $output,$runtimeAsr,$runtimeLlama,$runtimeFfmpeg -Force | Out-Null
foreach ($name in @('asr','llama','cuda-asr','cuda-llama','msvc','ffmpeg')) { New-Item -ItemType Directory -Path (Join-Path $licenseRoot $name) -Force | Out-Null }
Copy-Item -LiteralPath $executable -Destination $output
Get-ChildItem -LiteralPath $asr -File -Filter '*.dll' | Copy-Item -Destination $runtimeAsr
Get-ChildItem -LiteralPath $llama -File -Filter '*.dll' | Copy-Item -Destination $runtimeLlama
Copy-Item -LiteralPath (Join-Path $llama 'llama-server.exe') -Destination $runtimeLlama
Copy-Item -LiteralPath $Ffmpeg -Destination (Join-Path $runtimeFfmpeg 'ffmpeg.exe')
foreach ($destination in @($output,$runtimeAsr,$runtimeLlama)) { Get-ChildItem -LiteralPath $MsvcRedistDirectory -File -Filter '*.dll' | Copy-Item -Destination $destination }
Get-ChildItem -LiteralPath (Join-Path $asr 'licenses') -File | Copy-Item -Destination (Join-Path $licenseRoot 'asr')
Get-ChildItem -LiteralPath (Join-Path $asr 'licenses/cuda-asr') -File | Copy-Item -Destination (Join-Path $licenseRoot 'cuda-asr')
Get-ChildItem -LiteralPath (Join-Path $llama 'licenses') -File | Where-Object { $_.Name -ne 'cuda-12.4-EULA.html' } | Copy-Item -Destination (Join-Path $licenseRoot 'llama')
Copy-Item -LiteralPath (Join-Path $llama 'licenses/cuda-12.4-EULA.html') -Destination (Join-Path $licenseRoot 'cuda-llama')
Get-ChildItem -LiteralPath $msvcLicenses -File | Copy-Item -Destination (Join-Path $licenseRoot 'msvc')
$ffmpegNotices | Copy-Item -Destination (Join-Path $licenseRoot 'ffmpeg')
foreach ($name in @('README.md','LICENSE','LICENSE-MIT','Cargo.lock')) {
    $source = Join-Path $projectRoot $name
    if (Test-Path -LiteralPath $source -PathType Leaf) { Copy-Item -LiteralPath $source -Destination $output }
}
if (Test-Path -LiteralPath (Join-Path $projectRoot 'docs')) { Copy-Item -LiteralPath (Join-Path $projectRoot 'docs') -Destination $output -Recurse }
foreach ($name in @('glossary.tsv', 'translation-cases.jsonl', 'translation-glossary.tsv')) {
    $example = Join-Path $projectRoot ('examples/' + $name)
    if (Test-Path -LiteralPath $example -PathType Leaf) {
        $exampleDirectory = Join-Path $output 'examples'
        New-Item -ItemType Directory -Path $exampleDirectory -Force | Out-Null
        Copy-Item -LiteralPath $example -Destination $exampleDirectory
    }
}
$streamClient = Join-Path $projectRoot 'scripts/test-stream.ps1'
if (Test-Path -LiteralPath $streamClient -PathType Leaf) {
    $clientDirectory = Join-Path $output 'scripts'
    New-Item -ItemType Directory -Path $clientDirectory -Force | Out-Null
    Copy-Item -LiteralPath $streamClient -Destination $clientDirectory
}

@'
@echo off
setlocal
cd /d "%~dp0"
"%~dp0streamtrans.exe" %*
exit /b %errorlevel%
'@ | Set-Content -LiteralPath (Join-Path $output 'start.cmd') -Encoding ascii
$provenance = [ordered]@{
    asr = (Get-Content -LiteralPath (Join-Path $asr '.streamtrans-runtime.json') -Raw | ConvertFrom-Json)
    llama = (Get-Content -LiteralPath (Join-Path $llama '.streamtrans-runtime.json') -Raw | ConvertFrom-Json)
    ffmpeg = @{ version = $ffmpegVersion; source_url = $FfmpegSourceUrl; executable_sha256 = (Get-FileHash -LiteralPath $Ffmpeg -Algorithm SHA256).Hash.ToLowerInvariant() }
    msvc = @{ version = (Get-Item -LiteralPath (Join-Path $MsvcRedistDirectory 'vcruntime140.dll')).VersionInfo.FileVersion; license_url = 'https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/' }
}
if ($Ffmpeg -ieq (Join-Path $knownFfmpegRoot 'bin/ffmpeg.exe')) {
    $provenance.ffmpeg.binary_url = 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip'
    $provenance.ffmpeg.archive_sha256 = '60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba'
}
$provenance | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $output 'THIRD_PARTY.json') -Encoding utf8
$files = @(Get-ChildItem -LiteralPath $output -File -Recurse | Sort-Object FullName | ForEach-Object {
    @{ path = $_.FullName.Substring($output.TrimEnd('\').Length + 1).Replace('\','/'); size = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})
@{ format = 1; created_utc = [DateTime]::UtcNow.ToString('o'); models_included = $false; files = $files } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host "分发目录已生成：$output（不包含模型）；运行 start.cmd 或 streamtrans.exe 启动服务。"
