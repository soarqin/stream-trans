[CmdletBinding()]
param(
    [string]$LlamaDirectory,
    [string]$OutputDirectory,
    [ValidateRange(1, 64)][int]$Jobs = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw '此构建需要 Windows x64 PowerShell。' }
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildRoot = Join-Path $projectRoot '.tools/transcribe-build-shared'
if (-not $LlamaDirectory) { $LlamaDirectory = Join-Path $projectRoot '.tools/llama-embedded' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot '.tools/transcribe-shared' }
$llama = [IO.Path]::GetFullPath($LlamaDirectory).TrimEnd('\','/')
$output = [IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\','/')
$source = Join-Path $buildRoot 'source'
$build = Join-Path $buildRoot 'build'
$downloads = Join-Path $projectRoot '.tools/downloads'
$commit = '4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b'
$llamaCommit = '8642541b255bfb5dc39cd351a178be09013a55d6'
$archiveHash = '77bfafa35bd7eee447ec37b832e99c342c62b7db0c6cb18c36e763563b586b1f'
$archiveUrl = 'https://codeload.github.com/handy-computer/transcribe.cpp/tar.gz/' + $commit
$archive = Join-Path $downloads ('transcribe-' + $commit + '.tar.gz')
$owner = Join-Path $output '.streamtrans-build-owner'
$manifestPath = Join-Path $output '.streamtrans-runtime.json'
$cmakeAdapter = Join-Path $projectRoot 'native/transcribe-shared-ggml.cmake'

foreach ($path in @($output,$buildRoot,$llama,$owner)) {
    $ancestor = $path
    while ($ancestor) {
        if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "路径包含链接或重解析点，请选择普通目录：$ancestor"
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
}
$outputPrefix = $output + '\'
foreach ($protected in @($buildRoot,$llama)) {
    $prefix = $protected.TrimEnd('\','/') + '\'
    if ($outputPrefix.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase) -or $prefix.StartsWith($outputPrefix,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'ASR 输出目录不能与 llama.cpp 或构建缓存重叠，请指定独立的 -OutputDirectory。'
    }
}
$owned = (Test-Path -LiteralPath $owner -PathType Leaf) -and ([string](Get-Content -LiteralPath $owner -Raw)).Trim() -eq $commit
if ((Test-Path -LiteralPath $output) -and @(Get-ChildItem -LiteralPath $output -Force).Count -and -not $owned) {
    throw "为避免覆盖已有文件，不能写入非本脚本创建的目录：$output"
}

$llamaManifest = Join-Path $llama '.streamtrans-runtime.json'
if (-not (Test-Path -LiteralPath $llamaManifest -PathType Leaf)) { throw '缺少共享 ggml 构建记录，请先运行 scripts/setup-llama.ps1。' }
$llamaMetadata = Get-Content -LiteralPath $llamaManifest -Raw | ConvertFrom-Json
if ($llamaMetadata.version -ne 'b11960' -or $llamaMetadata.commit -ne $llamaCommit -or
    -not $llamaMetadata.PSObject.Properties['ggml'] -or $llamaMetadata.ggml.commit -ne $llamaCommit -or
    -not $llamaMetadata.ggml.compatibility_id -or -not $llamaMetadata.ggml.files -or
    -not $llamaMetadata.ggml.PSObject.Properties['cpu_openmp'] -or $llamaMetadata.ggml.cpu_openmp -ne $false) {
    throw 'llama.cpp 缺少固定版本的共享 ggml 兼容记录，请重新运行 scripts/setup-llama.ps1。'
}
foreach ($file in $llamaMetadata.ggml.files) {
    $path = Join-Path $llama $file.name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $file.sha256) {
        throw "共享 ggml 构建产物校验失败：$path。请重新运行 scripts/setup-llama.ps1。"
    }
}
$ggmlSource = Join-Path $projectRoot '.tools/llama-build-cuda128/source/ggml'
$ggmlBuild = Join-Path $projectRoot '.tools/llama-build-cuda128/build/ggml/src'
foreach ($path in @($cmakeAdapter,(Join-Path $ggmlSource 'include/ggml.h'),(Join-Path $ggmlBuild 'ggml.lib'),(Join-Path $ggmlBuild 'ggml-base.lib'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "缺少共享 ggml 构建输入：$path。请先运行 scripts/setup-llama.ps1。" }
}
# The import libraries must describe the same DLLs whose identity is recorded above.
$llamaBuildBin = Join-Path $projectRoot '.tools/llama-build-cuda128/build/bin'
foreach ($name in @('ggml.dll','ggml-base.dll')) {
    if ((Get-FileHash -LiteralPath (Join-Path $llamaBuildBin $name) -Algorithm SHA256).Hash -ine (Get-FileHash -LiteralPath (Join-Path $llama $name) -Algorithm SHA256).Hash) {
        throw 'ggml 构建缓存与运行库不一致，请重新运行 scripts/setup-llama.ps1。'
    }
}
foreach ($tool in @('cmake.exe','curl.exe','tar.exe')) { if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "缺少构建工具：$tool" } }
$cmake = (Get-Command cmake.exe).Source
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw '请安装包含 x64 C++ 工作负载的 Visual Studio Build Tools。' }
$vs = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
if (-not $vs) { throw '找不到 Visual Studio x64 C++ 工作负载。' }
$ninjaCommand = Get-Command ninja.exe -ErrorAction SilentlyContinue
$ninja = if ($ninjaCommand) { $ninjaCommand.Source } else { Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe' }
if (-not (Test-Path -LiteralPath $ninja)) { throw 'PATH 和 Visual Studio 中均找不到 Ninja。' }
New-Item -ItemType Directory -Path $downloads,$source,$build,$output -Force | Out-Null
Set-Content -LiteralPath $owner -Value $commit -Encoding ascii
if (-not (Test-Path -LiteralPath $archive -PathType Leaf) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $archiveHash) {
    $partial = "$archive.partial"
    Write-Host '正在下载并校验 transcribe.cpp 0.4.1 源码'
    & curl.exe --fail --location --retry 3 --output $partial $archiveUrl
    if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ine $archiveHash) { throw 'transcribe.cpp 源码下载或校验失败，请检查网络后重试。' }
    Move-Item -LiteralPath $partial -Destination $archive -Force
}
$stamp = Join-Path $buildRoot '.source-archive-sha256'
if (-not (Test-Path -LiteralPath $stamp) -or ([string](Get-Content -LiteralPath $stamp -Raw)).Trim() -ne $archiveHash) {
    & tar.exe -xf $archive -C $source --strip-components=1
    if ($LASTEXITCODE -ne 0) { throw 'transcribe.cpp 源码解压失败，请检查磁盘空间后重试。' }
    Set-Content -LiteralPath $stamp -Value $archiveHash -Encoding ascii
}

function Set-ChangedText([string]$Path,[string]$Text) {
    if ([IO.File]::ReadAllText($Path) -cne $Text) { [IO.File]::WriteAllText($Path,$Text,[Text.UTF8Encoding]::new($false)) }
}
$rootCmake = Join-Path $source 'CMakeLists.txt'
$text = [IO.File]::ReadAllText($rootCmake)
$include = 'include("' + $cmakeAdapter.Replace('\','/') + '")'
if ($text -match '(?m)^add_subdirectory\(ggml\)\s*$') {
    $text = [regex]::Replace($text,'(?m)^add_subdirectory\(ggml\)\s*$',$include)
} elseif (-not $text.Contains($include)) { throw 'transcribe.cpp CMake 结构不匹配，请检查固定源码缓存。' }
$text = $text.Replace('set(TRANSCRIBE_BUILD_COMMIT "unknown")',('set(TRANSCRIBE_BUILD_COMMIT "' + $commit + '")'))
Set-ChangedText $rootCmake $text
$transcribeCpp = Join-Path $source 'src/transcribe.cpp'
$text = [IO.File]::ReadAllText($transcribeCpp)
$declaration = 'extern "C" __declspec(dllimport) void ggml_backend_set_reg_filter(bool (*filter)(const char *));'
if (-not $text.Contains($declaration)) {
    $needle = '#include "ggml.h"  // ggml_log_set: route ggml diagnostics into our sink'
    if (-not $text.Contains($needle)) { throw 'transcribe.cpp ggml 接口位置不匹配，请检查固定源码缓存。' }
    $text = $text.Replace($needle,$needle + "`n" + $declaration)
    Set-ChangedText $transcribeCpp $text
}

$environmentScript = Join-Path $buildRoot 'build-environment.cmd'
Set-Content -LiteralPath $environmentScript -Encoding ascii -Value @('@echo off',('set "PATH=' + (Split-Path $vswhere -Parent) + ';%PATH%"'),('call "' + (Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat') + '" >nul'),'if errorlevel 1 exit /b 1','set')
$environment = & $env:ComSpec /d /c $environmentScript
if ($LASTEXITCODE -ne 0) { throw 'Visual Studio 构建环境初始化失败，请检查 C++ 工具链。' }
$savedEnvironment = @{}
try {
    foreach ($line in $environment) { if ($line -match '^([^=]+)=(.*)$') {
        $savedEnvironment[$Matches[1]] = [Environment]::GetEnvironmentVariable($Matches[1],'Process')
        [Environment]::SetEnvironmentVariable($Matches[1],$Matches[2],'Process')
    } }
    if (-not $savedEnvironment.ContainsKey('VSLANG')) { $savedEnvironment['VSLANG'] = [Environment]::GetEnvironmentVariable('VSLANG','Process') }
    $env:VSLANG = '1033'
    $configuration = @('-S',$source,'-B',$build,'-G','Ninja',"-DCMAKE_MAKE_PROGRAM=$ninja",
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_FLAGS=/utf-8','-DCMAKE_CXX_FLAGS=/utf-8',
        '-DTRANSCRIBE_BUILD_SHARED=ON','-DTRANSCRIBE_GGML_BACKEND_DL=ON','-DTRANSCRIBE_CUDA=ON',
        '-DTRANSCRIBE_BUILD_TESTS=OFF','-DTRANSCRIBE_BUILD_EXAMPLES=OFF','-DTRANSCRIBE_BUILD_TOOLS=OFF',
        '-DTRANSCRIBE_INSTALL=OFF','-DTRANSCRIBE_USE_SYSTEM_BLAS=OFF','-DTRANSCRIBE_USE_OPENMP=OFF',
        "-DSTREAMTRANS_GGML_SOURCE=$ggmlSource","-DSTREAMTRANS_GGML_BUILD=$ggmlBuild")
    function Invoke-Cmake([string[]]$Arguments,[string]$Log,[string]$Failure) {
        $ErrorActionPreference = 'Continue'
        & $cmake @Arguments 2>&1 | Tee-Object -FilePath $Log | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { throw $Failure }
    }
    Invoke-Cmake $configuration (Join-Path $buildRoot 'configure.log') 'ASR CMake 配置失败，请检查 .tools/transcribe-build-shared/configure.log。'
    Invoke-Cmake @('--build',$build,'--target','transcribe','--parallel',"$Jobs") (Join-Path $buildRoot 'build.log') 'ASR 编译失败，请检查 .tools/transcribe-build-shared/build.log。'
    Copy-Item -LiteralPath (Join-Path $build 'bin/transcribe.dll') -Destination $output -Force
    # Development output is standalone; dist copies these shared dependencies only once.
    foreach ($file in Get-ChildItem -LiteralPath $llama -File -Filter '*.dll' | Where-Object { $_.Name -match '^(ggml(?:-.*)?|cudart64_12|cublas64_12|cublasLt64_12|msvcp140|vcruntime140(?:_1)?|vcomp140)\.dll$' }) {
        Copy-Item -LiteralPath $file.FullName -Destination $output -Force
    }
    $licenses = Join-Path $output 'licenses'
    New-Item -ItemType Directory -Path $licenses -Force | Out-Null
    foreach ($license in @(
        @{source=(Join-Path $source 'LICENSE');name='LICENSE'},
        @{source=(Join-Path $source 'src/third_party/miniz/LICENSE');name='LICENSE-miniz'},
        @{source=(Join-Path $source 'src/third_party/silero_vad/LICENSE');name='LICENSE-silero_vad'},
        @{source=(Join-Path $llama 'licenses/LICENSE-llama.cpp');name='LICENSE-ggml'}
    )) { Copy-Item -LiteralPath $license.source -Destination (Join-Path $licenses $license.name) -Force }
    $files = @((Get-Item -LiteralPath (Join-Path $output 'transcribe.dll'))) + @(Get-ChildItem -LiteralPath $licenses -File)
    $records = @($files | Sort-Object FullName | ForEach-Object { @{name=$_.FullName.Substring($output.Length+1).Replace('\','/');size=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()} })
    $inputs = @($rootCmake,$transcribeCpp,$cmakeAdapter,(Join-Path $PSScriptRoot 'build-asr.ps1')) + @('ggml.h','ggml-backend.h','ggml-alloc.h','gguf.h' | ForEach-Object { Join-Path $ggmlSource ('include/'+$_) }) + @('ggml.lib','ggml-base.lib' | ForEach-Object { Join-Path $ggmlBuild $_ })
    $inputHashes = @($inputs | ForEach-Object { @{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()} })
    @{
        version='0.4.1';source=@{commit=$commit;url=$archiveUrl;archive_sha256=$archiveHash;patches=$inputHashes}
        ggml=$llamaMetadata.ggml;cuda_build=$llamaMetadata.cuda_build;cuda_runtime=$llamaMetadata.cuda_runtime
        cmake_arguments=$configuration;compiler='MSVC';msvc_toolset=$env:VCToolsVersion
        files=$records;development_dependency_source=$llama
    } | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Write-Host "ASR 0.4.1 已使用 llama.cpp 的同一套 ggml 构建：$output"
} finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name,$savedEnvironment[$name],'Process') }
}
