[CmdletBinding()]
param([string]$OutputDirectory)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw '此脚本需要 Windows x64 PowerShell。' }
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot '.tools/ffmpeg-shared' }
$runtime = [IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\','/')
$downloads = Join-Path $projectRoot '.tools/downloads'
$asset = 'ffmpeg-n9.0.2-39-g7adfbe40dc-win64-lgpl-shared-9.0.zip'
$archiveHash = 'e907b01fa4783913151237e37df2a10fa97d99a1598a021bf59d4ae62ffed4c0'
$archiveUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-31-12-59/' + $asset
$releaseUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2026-10-31-12-59'
$sourceUrl = 'https://github.com/FFmpeg/FFmpeg/commit/7adfbe40dcde598f93a6963d2f7b22cd75202ccc'
$licenseHash = 'da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768'
$ffmpegDllHashes = @{
    'avcodec-63.dll' = 'c20d1a5b282a83b5bc586b65b23cf4837d83c2caa3b4dd4863ede2dd0f1858b5'
    'avformat-63.dll' = '7c13073a52f7ad76cfc1c7e76d782e89d88c1155cf2500c3320d034b507605f6'
    'avutil-61.dll' = 'cb50cdee16a5031090330997b51c016fe7788487150cd07fa20ddbda4d108d09'
    'swresample-7.dll' = '12b907ddfff765e10c3b7da1a7985513dc9282c08953259f337f952f7b8a9f2a'
}
$bridgeName = 'streamtrans-ffmpeg.dll'
$owner = Join-Path $runtime '.streamtrans-ffmpeg-owner'
$stage = Join-Path $downloads 'ffmpeg-shared-unpacked'
foreach ($path in @($runtime,$stage,$owner)) {
    $ancestor = $path
    while ($ancestor) {
        if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "路径包含链接或重解析点，请选择普通目录：$ancestor"
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
}
$runtimePrefix = $runtime + '\'
$stagePrefix = $stage + '\'
if ($runtimePrefix.StartsWith($stagePrefix, [StringComparison]::OrdinalIgnoreCase) -or $stagePrefix.StartsWith($runtimePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'FFmpeg 输出目录不能与下载解压缓存重叠，请指定单独的 -OutputDirectory。'
}
$owned = (Test-Path -LiteralPath $owner -PathType Leaf) -and (Get-Content -LiteralPath $owner -Raw).Trim() -eq $archiveHash
if (Test-Path -LiteralPath $runtime) {
    if (-not (Test-Path -LiteralPath $runtime -PathType Container) -or (@(Get-ChildItem -LiteralPath $runtime -Force).Count -and -not $owned)) {
        throw "FFmpeg 输出目录不为空且不属于本脚本：$runtime。请指定新的 -OutputDirectory。"
    }
}
foreach ($tool in @('curl.exe','tar.exe')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "缺少工具：$tool。请安装后重试。" }
}
New-Item -ItemType Directory -Path $downloads,$runtime,$stage -Force | Out-Null
Set-Content -LiteralPath $owner -Value $archiveHash -Encoding ascii
$archive = Join-Path $downloads $asset
if (-not (Test-Path -LiteralPath $archive -PathType Leaf) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $archiveHash) {
    Write-Host "下载并校验 $asset"
    $partial = "$archive.partial"
    & curl.exe --fail --location --retry 3 --output $partial $archiveUrl
    if ($LASTEXITCODE -ne 0) { throw 'FFmpeg shared 下载失败，请检查网络后重试。' }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ine $archiveHash) { throw 'FFmpeg shared SHA256 校验失败，请重新下载。' }
    Move-Item -LiteralPath $partial -Destination $archive -Force
}
& tar.exe -xf $archive -C $stage
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg shared 解压失败，请检查下载缓存和磁盘空间后重试。' }
$packageRoot = Join-Path $stage ([IO.Path]::GetFileNameWithoutExtension($asset))
$bin = Join-Path $runtime 'bin'
New-Item -ItemType Directory -Path $bin -Force | Out-Null
foreach ($name in $ffmpegDllHashes.Keys) {
    $source = Join-Path $packageRoot ('bin/' + $name)
    if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ine $ffmpegDllHashes[$name]) {
        throw "FFmpeg shared 组件校验失败：$source。请检查下载缓存后重试。"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $bin $name) -Force
}
foreach ($name in @('include','lib')) {
    $directory = Join-Path $runtime $name
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $packageRoot $name) -Force | Copy-Item -Destination $directory -Recurse -Force
}
$license = Join-Path $packageRoot 'LICENSE.txt'
if ((Get-FileHash -LiteralPath $license -Algorithm SHA256).Hash -ine $licenseHash) { throw 'FFmpeg LGPLv3 许可证校验失败，请检查下载缓存后重试。' }
Copy-Item -LiteralPath $license -Destination (Join-Path $runtime 'LICENSE.txt') -Force
@("FFmpeg: n9.0.2-39-g7adfbe40dc (Windows x64, LGPLv3 shared)", "Source Code: $sourceUrl", "Binary Release: $releaseUrl", "Binary Archive: $archiveUrl", "Archive SHA256: $archiveHash", 'This is the last build of 2026-10; upstream keeps month-end builds for two years.', 'Only avformat, avcodec, avutil and swresample are redistributed; programs and video-only libraries are omitted.') | Set-Content -LiteralPath (Join-Path $runtime 'SOURCE.txt') -Encoding utf8
& (Join-Path $PSScriptRoot 'build-ffmpeg.ps1') -FfmpegDirectory $runtime -OutputDirectory $bin
if (-not (Test-Path -LiteralPath (Join-Path $bin $bridgeName) -PathType Leaf)) { throw 'FFmpeg 音频桥编译后未生成 streamtrans-ffmpeg.dll，请检查构建输出。' }

# The four pinned libraries have only Windows/UCRT imports besides each other.
$dumpbinCommand = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
$dumpbin = if ($dumpbinCommand) { $dumpbinCommand.Source } else { $null }
if (-not $dumpbin) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
    $toolset = Get-ChildItem -LiteralPath (Join-Path $vs 'VC/Tools/MSVC') -Directory | Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    $dumpbin = Join-Path $toolset.FullName 'bin/Hostx64/x64/dumpbin.exe'
}
if (-not (Test-Path -LiteralPath $dumpbin -PathType Leaf)) { throw '找不到 dumpbin.exe，无法检查 FFmpeg 运行库依赖；请检查 Visual Studio C++ 工具链。' }
$runtimeDlls = @($ffmpegDllHashes.Keys) + @($bridgeName)
$sharedMsvc = @('vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll')
$imports = @()
foreach ($name in $runtimeDlls) {
    $dependencies = @()
    $lines = & $dumpbin /nologo /dependents (Join-Path $bin $name)
    if ($LASTEXITCODE -ne 0) { throw "无法读取 FFmpeg PE 依赖：$name" }
    foreach ($line in $lines) {
        if ($line -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') {
            $dependency = $Matches[1]
            if ($dependency -notin $runtimeDlls -and $dependency -notin $sharedMsvc -and
                $dependency -notmatch '^(api-ms-win-|ext-ms-win-)' -and
                -not (Test-Path -LiteralPath (Join-Path ([Environment]::SystemDirectory) $dependency) -PathType Leaf)) {
                throw "FFmpeg 存在未包含的原生依赖：$name -> $dependency。请检查构建配置。"
            }
            $dependencies += $dependency
        }
    }
    $imports += [ordered]@{ name = $name; imports = @($dependencies | Sort-Object -Unique) }
}
foreach ($file in Get-ChildItem -LiteralPath $bin -File) {
    if ($file.Extension -in @('.dll','.exe') -and $file.Name -notin $runtimeDlls -and $file.Name -notin $sharedMsvc) {
        throw "FFmpeg bin 目录存在无关原生文件，请指定干净的输出目录：$($file.FullName)"
    }
}
$files = @($runtimeDlls | ForEach-Object { 'bin/' + $_ }) + @('LICENSE.txt','SOURCE.txt')
$fileRecords = @($files | ForEach-Object {
    $file = Get-Item -LiteralPath (Join-Path $runtime $_)
    [ordered]@{ name = $_; size = $file.Length; sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})
[ordered]@{
    format = 1; version = 'n9.0.2-39-g7adfbe40dc'; license = 'LGPLv3'; backend = 'shared-library'
    source_url = $sourceUrl; release_url = $releaseUrl; archive_url = $archiveUrl; archive_sha256 = $archiveHash
    api_major_versions = @{ avformat = 63; avcodec = 63; avutil = 61; swresample = 7 }
    upstream_retention = 'Last build of 2026-10; month-end builds retained for two years'
    runtime_dlls = $runtimeDlls; dependencies = $imports; files = $fileRecords
    shared_msvc = @($imports | ForEach-Object { $_.imports } | Where-Object { $_ -in $sharedMsvc } | Sort-Object -Unique)
    bridge = @{
        name = $bridgeName; abi_version = 1; source = 'native/streamtrans_ffmpeg.c'; license = 'MIT'; license_file = 'LICENSE'
        sha256 = (Get-FileHash -LiteralPath (Join-Path $bin $bridgeName) -Algorithm SHA256).Hash.ToLowerInvariant()
        source_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot 'native/streamtrans_ffmpeg.c') -Algorithm SHA256).Hash.ToLowerInvariant()
        build_script_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'build-ffmpeg.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $runtime '.streamtrans-ffmpeg.json') -Encoding utf8
Write-Host "FFmpeg 9 LGPLv3 音频动态库已准备：$runtime（不使用 FFmpeg EXE）"
