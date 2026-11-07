[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [switch]$SkipBuild,
    [string]$Ffmpeg,
    [string]$FfmpegLicenseDirectory,
    [string]$FfmpegSourceUrl,
    [string]$MsvcRedistDirectory,
    [string]$LlamaDirectory,
    [string]$FfmpegDirectory,
    [string]$AsrDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Ffmpeg -or $FfmpegLicenseDirectory -or $FfmpegSourceUrl) {
    throw 'FFmpeg 已改为动态库调用；旧 -Ffmpeg/-FfmpegLicenseDirectory/-FfmpegSourceUrl 参数已停用。请运行 scripts/setup-ffmpeg.ps1，或使用 -FfmpegDirectory 指定其输出根目录。'
}
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot 'dist/StreamTrans' }
$output = [IO.Path]::GetFullPath($OutputDirectory)
function Assert-EmptyOutput {
    if (Test-Path -LiteralPath $output) {
        $existing = Get-Item -LiteralPath $output -Force
        if (-not $existing.PSIsContainer -or ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -or @(Get-ChildItem -LiteralPath $output -Force).Count -gt 0) {
            throw "输出目录必须不存在或为空，脚本不会覆盖非空目录：$output。请指定新的 -OutputDirectory。"
        }
    }
}
Assert-EmptyOutput

function Copy-Exclusive([string]$Source, [string]$Destination) {
    # Every shared runtime file has one authoritative build and one destination.
    [IO.File]::Copy($Source, $Destination, $false)
}

function Assert-RecordedArtifact([string]$Directory, $Files, [string]$Name, [string]$Component = 'llama.cpp', [string]$SetupScript = 'scripts/setup-llama.ps1') {
    $records = @($Files | Where-Object { $_.PSObject.Properties['name'] -and $_.name -ieq $Name })
    if ($records.Count -ne 1 -or -not $records[0].PSObject.Properties['sha256'] -or
        $records[0].sha256 -notmatch '^[a-fA-F0-9]{64}$') {
        throw "$Component 构建记录缺少有效的文件校验值：$Name。请重新运行 $SetupScript。"
    }
    $path = Join-Path $Directory $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $records[0].sha256) {
        throw "$Component 构建产物与记录不匹配：$path。请重新运行 $SetupScript。"
    }
}

function Assert-EmbeddedServer($Metadata) {
    $exports = @('streamtrans_server_abi_version','streamtrans_server_create','streamtrans_server_run','streamtrans_server_request_stop','streamtrans_server_status','streamtrans_server_destroy')
    if (-not $Metadata.PSObject.Properties['embedded_server'] -or $Metadata.embedded_server -isnot [pscustomobject]) {
        throw 'llama.cpp 缺少进程内服务构建记录，请重新运行 scripts/setup-llama.ps1。'
    }
    $embedded = $Metadata.embedded_server
    if (-not $embedded.PSObject.Properties['abi_version'] -or $embedded.abi_version -ne 1 -or
        -not $embedded.PSObject.Properties['library'] -or $embedded.library -cne 'llama-server-impl.dll' -or
        -not $embedded.PSObject.Properties['exports'] -or @($embedded.exports).Count -ne $exports.Count -or
        @(Compare-Object $exports @($embedded.exports) -CaseSensitive).Count) {
        throw 'llama.cpp 进程内服务 ABI、动态库或导出入口不匹配，请重新运行 scripts/setup-llama.ps1。'
    }
}

function Get-SharedGgmlFiles($AsrMetadata, $LlamaMetadata) {
    $sets = @()
    foreach ($metadata in @($AsrMetadata, $LlamaMetadata)) {
        if (-not $metadata.PSObject.Properties['ggml'] -or -not $metadata.ggml -or
            -not $metadata.ggml.PSObject.Properties['commit'] -or $metadata.ggml.commit -ne '8642541b255bfb5dc39cd351a178be09013a55d6' -or
            -not $metadata.ggml.PSObject.Properties['compatibility_id'] -or
            $metadata.ggml.compatibility_id -isnot [string] -or [string]::IsNullOrWhiteSpace($metadata.ggml.compatibility_id) -or
            -not $metadata.ggml.PSObject.Properties['files']) {
            throw '缺少统一 ggml 的源码、兼容补丁或文件校验记录，请重新运行 scripts/setup.ps1。'
        }
        $files = @{}
        foreach ($record in $metadata.ggml.files) {
            if (-not $record.PSObject.Properties['name'] -or -not $record.PSObject.Properties['sha256'] -or
                $record.name -isnot [string] -or $record.sha256 -isnot [string] -or
                $record.name -notmatch '^ggml(?:-base|-cuda|-cpu(?:-[a-z0-9_-]+)?)?\.dll$' -or
                $record.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or $files.ContainsKey($record.name)) {
                throw '统一 ggml 文件记录无效、重复或包含其他后端，请重新运行 scripts/setup.ps1。'
            }
            $files[$record.name] = $record.sha256
        }
        foreach ($name in @('ggml.dll','ggml-base.dll','ggml-cuda.dll')) {
            if (-not $files.ContainsKey($name)) { throw "统一 ggml 缺少必要文件记录：$name。请重新运行 scripts/setup.ps1。" }
        }
        if (-not @($files.Keys | Where-Object { $_ -match '^ggml-cpu(?:-[a-z0-9_-]+)?\.dll$' }).Count) {
            throw '统一 ggml 缺少 CPU 后端记录，请重新运行 scripts/setup.ps1。'
        }
        $sets += ,$files
    }
    if ($AsrMetadata.ggml.compatibility_id -ne $LlamaMetadata.ggml.compatibility_id -or $sets[0].Count -ne $sets[1].Count) {
        throw 'ASR 与 llama.cpp 的 ggml 兼容补丁或文件集合不一致，请重新运行 scripts/setup.ps1。'
    }
    foreach ($name in $sets[0].Keys) {
        if (-not $sets[1].ContainsKey($name) -or $sets[0][$name] -ine $sets[1][$name]) {
            throw "ASR 与 llama.cpp 的共享 ggml 文件 SHA256 不一致：$name。请重新运行 scripts/setup.ps1。"
        }
    }
    return @($sets[0].Keys | Sort-Object)
}

if (-not $AsrDirectory) { $AsrDirectory = Join-Path $projectRoot '.tools/transcribe-shared' }
$asr = [IO.Path]::GetFullPath($AsrDirectory)
if (-not $LlamaDirectory) { $LlamaDirectory = Join-Path $projectRoot '.tools/llama-embedded' }
$llama = [IO.Path]::GetFullPath($LlamaDirectory)
$msvcLicenses = Join-Path $projectRoot '.tools/licenses/msvc'
$asrDlls = @('transcribe.dll')
$llamaCoreDlls = @('llama-server-impl.dll','llama-common.dll','llama.dll','mtmd.dll')
$ggmlCoreDlls = @('ggml.dll','ggml-base.dll','ggml-cuda.dll')
$llamaOptionalDlls = @('vcomp140.dll','libomp.dll')
$sharedCuda = @{
    'cudart64_12.dll' = 'c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344'
    'cublas64_12.dll' = '9513540e4ec4c51ee9e7304138c2cc255c29a8c181f9e80c38efa25738becd99'
    'cublasLt64_12.dll' = 'b199d1ff892a81b7fd3d57ba1781549609b41500b36008fef326038393ad46c7'
}
foreach ($required in @(
    (Join-Path $asr 'licenses/LICENSE'), (Join-Path $asr '.streamtrans-runtime.json'),
    (Join-Path $llama 'licenses/LICENSE-llama.cpp'),
    (Join-Path $llama 'licenses/cuda/cuda_cudart-LICENSE'), (Join-Path $llama 'licenses/cuda/libcublas-LICENSE'),
    (Join-Path $llama '.streamtrans-runtime.json'),
    (Join-Path $msvcLicenses 'Visual-C-Runtime-2015-2022-License.docx')
) + @($asrDlls | ForEach-Object { Join-Path $asr $_ }) + @(($llamaCoreDlls + $ggmlCoreDlls) | ForEach-Object { Join-Path $llama $_ })) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "缺少分发依赖：$required。请先执行 .\scripts\setup.ps1。" }
}
$asrProvenance = Get-Content -LiteralPath (Join-Path $asr '.streamtrans-runtime.json') -Raw | ConvertFrom-Json
$llamaProvenance = Get-Content -LiteralPath (Join-Path $llama '.streamtrans-runtime.json') -Raw | ConvertFrom-Json
Assert-EmbeddedServer $llamaProvenance
$llamaCudaBuild = if ($llamaProvenance.PSObject.Properties['cuda_build']) { $llamaProvenance.cuda_build } else { $null }
$llamaRuntimeMatches = $llamaProvenance.PSObject.Properties['cuda_runtime'] -and
    $llamaProvenance.cuda_runtime.PSObject.Properties['version'] -and $llamaProvenance.cuda_runtime.version -eq '12.8.1'
if ($asrProvenance.version -ne '0.4.1' -or $llamaProvenance.version -ne 'b11960' -or $llamaProvenance.commit -ne '8642541b255bfb5dc39cd351a178be09013a55d6' -or $llamaCudaBuild -ne '12.8' -or -not $llamaRuntimeMatches) {
    throw '原生程序版本不匹配：需要自构建 transcribe.cpp 0.4.1 与固定 b11960 / CUDA 12.8 共享 ggml 构建；请运行 scripts/setup.ps1。'
}
$sharedGgmlDlls = @(Get-SharedGgmlFiles $asrProvenance $llamaProvenance)
$actualSharedGgml = @(Get-ChildItem -LiteralPath $llama -File -Filter '*.dll' | Where-Object {
    $_.Name -match '^ggml(?:-base|-cuda|-cpu(?:-[a-z0-9_-]+)?)?\.dll$'
} | Select-Object -ExpandProperty Name)
if (@(Compare-Object $sharedGgmlDlls $actualSharedGgml).Count) {
    throw 'llama.cpp 实际 ggml 文件集合与统一构建记录不一致，请重新运行 scripts/setup.ps1。'
}
$sharedMsvc = @('vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll')
$llamaDlls = @(Get-ChildItem -LiteralPath $llama -File -Filter '*.dll' | Where-Object {
    $_.Name -in $llamaCoreDlls -or $_.Name -in $llamaOptionalDlls
} | Select-Object -ExpandProperty Name)
$llamaRuntimeFiles = @($llamaDlls + $sharedGgmlDlls)
if ($llamaDlls -contains 'libomp.dll' -and -not (Test-Path -LiteralPath (Join-Path $llama 'licenses/LICENSE-LLVM-OpenMP') -PathType Leaf)) {
    throw 'llama.cpp 使用 libomp.dll，但缺少 LLVM OpenMP 许可证。'
}
if (-not $llamaProvenance.PSObject.Properties['files'] -or -not $asrProvenance.PSObject.Properties['files']) {
    throw 'ASR 或 llama.cpp 缺少构建文件校验记录，请重新运行 scripts/setup.ps1。'
}
$asrLicenseFiles = @(Get-ChildItem -LiteralPath (Join-Path $asr 'licenses') -File)
foreach ($name in $asrDlls + @($asrLicenseFiles | ForEach-Object { 'licenses/' + $_.Name })) {
    Assert-RecordedArtifact $asr $asrProvenance.files $name 'transcribe.cpp' 'scripts/build-asr.ps1'
}
$llamaLicenseFiles = @(Get-ChildItem -LiteralPath (Join-Path $llama 'licenses') -File)
foreach ($name in $llamaRuntimeFiles + @($llamaLicenseFiles | ForEach-Object { 'licenses/' + $_.Name })) {
    Assert-RecordedArtifact $llama $llamaProvenance.files $name
}
foreach ($name in $sharedGgmlDlls) { Assert-RecordedArtifact $llama $llamaProvenance.ggml.files $name }
foreach ($name in $sharedCuda.Keys) {
    $path = Join-Path $llama $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $sharedCuda[$name]) {
        throw "共享 CUDA 12.8 运行库缺失或校验失败：$path。请运行 scripts/setup.ps1。"
    }
}

if (-not $FfmpegDirectory) { $FfmpegDirectory = Join-Path $projectRoot '.tools/ffmpeg-shared' }
$ffmpeg = [IO.Path]::GetFullPath($FfmpegDirectory)
$ffmpegDllHashes = @{
    'avcodec-63.dll' = 'c20d1a5b282a83b5bc586b65b23cf4837d83c2caa3b4dd4863ede2dd0f1858b5'
    'avformat-63.dll' = '7c13073a52f7ad76cfc1c7e76d782e89d88c1155cf2500c3320d034b507605f6'
    'avutil-61.dll' = 'cb50cdee16a5031090330997b51c016fe7788487150cd07fa20ddbda4d108d09'
    'swresample-7.dll' = '12b907ddfff765e10c3b7da1a7985513dc9282c08953259f337f952f7b8a9f2a'
}
$ffmpegDlls = @($ffmpegDllHashes.Keys) + @('streamtrans-ffmpeg.dll')
$ffmpegLicenseHash = 'da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768'
$ffmpegManifest = Join-Path $ffmpeg '.streamtrans-ffmpeg.json'
if (-not (Test-Path -LiteralPath $ffmpegManifest -PathType Leaf)) { throw '缺少 FFmpeg 动态库构建记录，请先执行 scripts/setup-ffmpeg.ps1。' }
$ffmpegProvenance = Get-Content -LiteralPath $ffmpegManifest -Raw | ConvertFrom-Json
if ($ffmpegProvenance.version -ne 'n9.0.2-39-g7adfbe40dc' -or $ffmpegProvenance.license -ne 'LGPLv3' -or
    $ffmpegProvenance.archive_sha256 -ne 'e907b01fa4783913151237e37df2a10fa97d99a1598a021bf59d4ae62ffed4c0' -or
    $ffmpegProvenance.backend -ne 'shared-library' -or $ffmpegProvenance.bridge.abi_version -ne 1) {
    throw 'FFmpeg 构建记录版本不匹配，请重新执行 scripts/setup-ffmpeg.ps1。'
}
foreach ($name in $ffmpegDllHashes.Keys) {
    $path = Join-Path $ffmpeg ('bin/' + $name)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $ffmpegDllHashes[$name]) {
        throw "FFmpeg shared 固定组件校验失败：$path。请重新执行 scripts/setup-ffmpeg.ps1。"
    }
}
if ((Get-FileHash -LiteralPath (Join-Path $ffmpeg 'LICENSE.txt') -Algorithm SHA256).Hash -ine $ffmpegLicenseHash) {
    throw 'FFmpeg LGPLv3 许可证校验失败，请重新执行 scripts/setup-ffmpeg.ps1。'
}
foreach ($name in @($ffmpegDlls | ForEach-Object { 'bin/' + $_ }) + @('LICENSE.txt','SOURCE.txt')) {
    Assert-RecordedArtifact $ffmpeg $ffmpegProvenance.files $name 'FFmpeg' 'scripts/setup-ffmpeg.ps1'
}

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
foreach ($dll in $sharedMsvc) {
    if (-not $MsvcRedistDirectory -or -not (Test-Path -LiteralPath (Join-Path $MsvcRedistDirectory $dll) -PathType Leaf)) { throw "MSVC app-local 运行库缺少 $dll，请指定完整的 x64 -MsvcRedistDirectory。" }
}

if (-not $SkipBuild) { & (Join-Path $PSScriptRoot 'build.ps1') }
$executable = Join-Path $projectRoot 'target/release/streamtrans.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw '找不到 target/release/streamtrans.exe，请先执行 .\scripts\build.ps1。' }
$runtimeShared = Join-Path $output 'runtime'
$licenseRoot = Join-Path $output 'licenses'
Assert-EmptyOutput
New-Item -ItemType Directory -Path $output,$runtimeShared -Force | Out-Null
foreach ($name in @('asr','llama','cuda','msvc','ffmpeg')) { New-Item -ItemType Directory -Path (Join-Path $licenseRoot $name) -Force | Out-Null }
Copy-Exclusive $executable (Join-Path $output 'streamtrans.exe')
foreach ($name in $asrDlls) { Copy-Exclusive (Join-Path $asr $name) (Join-Path $runtimeShared $name) }
foreach ($name in $llamaRuntimeFiles) { Copy-Exclusive (Join-Path $llama $name) (Join-Path $runtimeShared $name) }
foreach ($name in $sharedCuda.Keys) { Copy-Exclusive (Join-Path $llama $name) (Join-Path $runtimeShared $name) }
foreach ($name in $ffmpegDlls) { Copy-Exclusive (Join-Path $ffmpeg ('bin/' + $name)) (Join-Path $runtimeShared $name) }
foreach ($name in $sharedMsvc) { Copy-Exclusive (Join-Path $MsvcRedistDirectory $name) (Join-Path $runtimeShared $name) }
Copy-Exclusive (Join-Path $MsvcRedistDirectory 'vcruntime140.dll') (Join-Path $output 'vcruntime140.dll')
$asrLicenseFiles | Copy-Item -Destination (Join-Path $licenseRoot 'asr')
foreach ($name in @('cuda_cudart-LICENSE','libcublas-LICENSE')) { Copy-Exclusive (Join-Path $llama ('licenses/cuda/' + $name)) (Join-Path $licenseRoot ('cuda/' + $name)) }
$llamaLicenseFiles | Copy-Item -Destination (Join-Path $licenseRoot 'llama')
Get-ChildItem -LiteralPath $msvcLicenses -File | Copy-Item -Destination (Join-Path $licenseRoot 'msvc')
foreach ($name in @('LICENSE.txt','SOURCE.txt')) { Copy-Exclusive (Join-Path $ffmpeg $name) (Join-Path $licenseRoot ('ffmpeg/' + $name)) }
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
    asr = $asrProvenance
    llama = $llamaProvenance
    shared_ggml = $llamaProvenance.ggml
    shared_cuda = @{
        version = '12.8.1'; cudart = '12.8.90'; cublas = '12.8.4.1'; directory = 'runtime'
        manifest = $llamaProvenance.cuda_runtime.manifest; packages = $llamaProvenance.cuda_runtime.packages
        files = @($sharedCuda.Keys | Sort-Object | ForEach-Object { @{ path = 'runtime/' + $_; sha256 = $sharedCuda[$_] } })
    }
    ffmpeg = $ffmpegProvenance
    msvc = @{ version = (Get-Item -LiteralPath (Join-Path $MsvcRedistDirectory 'vcruntime140.dll')).VersionInfo.FileVersion; license_url = 'https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/' }
}
$provenance | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $output 'THIRD_PARTY.json') -Encoding utf8
$files = @(Get-ChildItem -LiteralPath $output -File -Recurse | Sort-Object FullName | ForEach-Object {
    @{ path = $_.FullName.Substring($output.TrimEnd('\').Length + 1).Replace('\','/'); size = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})
@{ format = 1; created_utc = [DateTime]::UtcNow.ToString('o'); models_included = $false; files = $files } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host "分发目录已生成：$output（不包含模型）；运行 start.cmd 或 streamtrans.exe 启动服务。"
