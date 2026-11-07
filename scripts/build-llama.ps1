[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [ValidateRange(1, 64)][int]$Jobs = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw '此构建需要 Windows x64 PowerShell。' }
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildRoot = Join-Path $projectRoot '.tools/llama-build-cuda128'
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot '.tools/llama-shared' }
$output = [IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\','/')
$downloads = Join-Path $buildRoot 'downloads'
$source = Join-Path $buildRoot 'source'
$cuda = Join-Path $buildRoot 'cuda'
$build = Join-Path $buildRoot 'build'
$commit = '8642541b255bfb5dc39cd351a178be09013a55d6'
$architectures = '50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real'
$owner = Join-Path $output '.streamtrans-build-owner'
$manifestPath = Join-Path $output '.streamtrans-runtime.json'
$outputPrefix = $output.TrimEnd('\','/') + '\'
$buildPrefix = $buildRoot.TrimEnd('\','/') + '\'
if ($outputPrefix.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase) -or $buildPrefix.StartsWith($outputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw '输出目录不能与源码、CUDA 或构建缓存目录重叠，请使用单独的目录。'
}
foreach ($path in @($output,$buildRoot,$owner)) {
    $ancestor = $path
    while ($ancestor) {
        if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "路径包含链接或重解析点，请选择普通目录：$ancestor"
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
}
$owned = (Test-Path -LiteralPath $owner -PathType Leaf) -and ([string](Get-Content -LiteralPath $owner -Raw)).Trim() -eq $commit
if ((Test-Path -LiteralPath $output) -and @(Get-ChildItem -LiteralPath $output -Force).Count -and -not $owned) {
    throw "为避免覆盖已有文件，不能写入非本脚本创建的目录：$output"
}
New-Item -ItemType Directory -Path $downloads,$source,$cuda,$build,$output -Force | Out-Null
Set-Content -LiteralPath $owner -Value $commit -Encoding ascii

$packages = @(
    @{ name = 'llama-source'; version = 'b11960'; archive = "llama-$commit.tar.gz"; url = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$commit"; sha256 = '1688a167a8edd9499cdd0c19b8bd044f0fa184b466981977bbf2222d884953a4' },
    @{ name = 'cuda_nvcc'; version = '12.8.93'; archive = 'cuda_nvcc-windows-x86_64-12.8.93-archive.zip'; sha256 = '9fdc70b4271ed9aad4d64cd7076a7d96ec36512d074b9995fe638de669197391' },
    @{ name = 'cuda_cudart'; version = '12.8.90'; archive = 'cuda_cudart-windows-x86_64-12.8.90-archive.zip'; sha256 = '4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b' },
    @{ name = 'cuda_cccl'; version = '12.8.90'; archive = 'cuda_cccl-windows-x86_64-12.8.90-archive.zip'; sha256 = 'bd8548fa1ae82f92910bebc3079e14bd58c5a92aa64596d46bd610a478cb39d7' },
    @{ name = 'libcublas'; version = '12.8.4.1'; archive = 'libcublas-windows-x86_64-12.8.4.1-archive.zip'; sha256 = '57a470112cec7e112c95253dde8b3c7184d795dbd92b0bde77a4cb7f8c94c8aa' }
)
foreach ($package in $packages) {
    if (-not $package.ContainsKey('url')) { $package.url = "https://developer.download.nvidia.com/compute/cuda/redist/$($package.name)/windows-x86_64/$($package.archive)" }
}

function Test-Hash([string]$Path, [string]$Hash) {
    return (Test-Path -LiteralPath $Path -PathType Leaf) -and (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ieq $Hash
}
function Invoke-Cmake([string[]]$Arguments, [string]$Log, [string]$Failure) {
    # Windows PowerShell 5 treats native stderr warnings as errors; use the exit code.
    $ErrorActionPreference = 'Continue'
    & $cmake @Arguments 2>&1 | Tee-Object -FilePath $Log | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw $Failure }
}
foreach ($tool in @('cmake.exe','curl.exe','tar.exe','git.exe')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "缺少构建工具：$tool" }
}
$cmake = (Get-Command cmake.exe).Source
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw '请安装包含 x64 C++ 工作负载的 Visual Studio Build Tools。' }
$vs = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
if (-not $vs) { throw '找不到 Visual Studio x64 C++ 工作负载。' }
$ninjaCommand = Get-Command ninja.exe -ErrorAction SilentlyContinue
$ninja = if ($ninjaCommand) { $ninjaCommand.Source } else { Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe' }
if (-not (Test-Path -LiteralPath $ninja)) { throw 'PATH 和 Visual Studio 中均找不到 Ninja。' }

foreach ($package in $packages) {
    $archive = Join-Path $downloads $package.archive
    if (-not (Test-Hash $archive $package.sha256)) {
        $partial = "$archive.partial"
        Write-Host "正在下载 $($package.archive)"
        & curl.exe --fail --location --retry 3 --output $partial $package.url
        if ($LASTEXITCODE -ne 0) { throw "下载失败：$($package.url)；请检查网络后重试。" }
        if (-not (Test-Hash $partial $package.sha256)) { throw "SHA256 校验失败：$partial；请重新下载。" }
        Move-Item -LiteralPath $partial -Destination $archive -Force
    }
    $destination = if ($package.name -eq 'llama-source') { $source } else { $cuda }
    $stamp = Join-Path $destination ('.' + $package.name + '-sha256')
    if (-not (Test-Path -LiteralPath $stamp) -or ([string](Get-Content -LiteralPath $stamp -Raw)).Trim() -ne $package.sha256) {
        Write-Host "正在解压 $($package.archive)"
        & tar.exe -xf $archive -C $destination --strip-components=1
        if ($LASTEXITCODE -ne 0) { throw "解压失败：$archive；请检查磁盘空间后重试。" }
        Set-Content -LiteralPath $stamp -Value $package.sha256 -Encoding ascii
    }
}

# Preserve transcribe's backend allow-list and CPU starvation fix without changing
# public ggml headers, so existing CUDA compilation units remain reusable.
$ggmlPatches = @(
    @{
        name = 'ggml-backend-reg-filter.patch'; source = 'ggml/src/ggml-backend-reg.cpp'
        sha256 = '017f906befe1193450f109ef68a9177f055f86f76949064f5b981ff0e1ca4d18'
        original_sha256 = '1436b46bd1e8355345d454baa4b40069be73e9cb09a3a188d32426ea1e1037fe'
        result_sha256 = 'a33e9273282c98dd43c7dbd65122bbc12c2dc80664665b01f0c5142372f64b07'
    },
    @{
        name = 'ggml-cpu-threadpool.patch'; source = 'ggml/src/ggml-cpu/ggml-cpu.c'
        sha256 = '94e10e2053943db0ba9cc2446e935775442456cf9feb70ec196f8899e00d3638'
        original_sha256 = '97a3d16efc5bbb0417b78b9a689b32162b4e16392b044888645ecdcfec23cca1'
        result_sha256 = 'e2b005072bc831a1cbc698236354553bb7422cf1a8a051716b96c6a1f0b2720c'
    }
)
$patchCache = Join-Path $buildRoot 'patches'
New-Item -ItemType Directory -Path $patchCache -Force | Out-Null
$patchRecords = @()
foreach ($patch in $ggmlPatches) {
    $patchPath = Join-Path $projectRoot ('native/' + $patch.name)
    if (-not (Test-Path -LiteralPath $patchPath -PathType Leaf)) { throw "缺少 ggml 兼容补丁：$patchPath" }
    $patchText = [IO.File]::ReadAllText($patchPath).Replace("`r`n", "`n")
    $normalized = Join-Path $patchCache $patch.name
    [IO.File]::WriteAllText($normalized, $patchText, [Text.UTF8Encoding]::new($false))
    if (-not (Test-Hash $normalized $patch.sha256)) { throw "ggml 兼容补丁校验失败：$patchPath" }
    $target = Join-Path $source $patch.source
    $targetText = [IO.File]::ReadAllText($target)
    $normalizedTarget = $targetText.Replace("`r`n", "`n")
    $targetHasher = [Security.Cryptography.SHA256]::Create()
    try { $targetHash = ([BitConverter]::ToString($targetHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($normalizedTarget)))).Replace('-','').ToLowerInvariant() } finally { $targetHasher.Dispose() }
    if ($targetHash -notin @($patch.original_sha256, $patch.result_sha256)) { throw "ggml 源码与固定版本不匹配：$target；请使用未修改的下载源码重试。" }
    if ($normalizedTarget -cne $targetText) { [IO.File]::WriteAllText($target, $normalizedTarget, [Text.UTF8Encoding]::new($false)) }
    if (-not (Test-Hash $target $patch.result_sha256)) {
        if (-not (Test-Hash $target $patch.original_sha256)) { throw "ggml 源码与固定版本不匹配：$target；请使用未修改的下载源码重试。" }
        & git -c core.autocrlf=false -C $projectRoot apply --check --directory=.tools/llama-build-cuda128/source $normalized
        if ($LASTEXITCODE -ne 0) { throw "ggml 兼容补丁不能应用：$patchPath" }
        & git -c core.autocrlf=false -C $projectRoot apply --directory=.tools/llama-build-cuda128/source $normalized
        if ($LASTEXITCODE -ne 0 -or -not (Test-Hash $target $patch.result_sha256)) { throw "ggml 兼容补丁应用或结果校验失败：$patchPath" }
    }
    $patchRecords += @{ name = 'native/' + $patch.name; sha256 = $patch.sha256; source = $patch.source; result_sha256 = $patch.result_sha256 }
}
$compatibilityText = $commit + "`nGGML_OPENMP=OFF`n" + (($patchRecords | ForEach-Object { $_.name + "`t" + $_.sha256 }) -join "`n")
$compatibilityHasher = [Security.Cryptography.SHA256]::Create()
try { $compatibilityId = 'sha256:' + ([BitConverter]::ToString($compatibilityHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($compatibilityText)))).Replace('-','').ToLowerInvariant() } finally { $compatibilityHasher.Dispose() }

# Initialize only this process; do not install CUDA or change machine/user PATH.
$vcvars = Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$envScript = Join-Path $buildRoot 'build-environment.cmd'
Set-Content -LiteralPath $envScript -Encoding ascii -Value @('@echo off', ('call "' + $vcvars + '" >nul'), 'if errorlevel 1 exit /b 1', 'set')
$environment = & $env:ComSpec /d /c $envScript
if ($LASTEXITCODE -ne 0) { throw 'Visual Studio 构建环境初始化失败，请检查 C++ 工具链。' }
$savedEnvironment = @{}
try {
    foreach ($line in $environment) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $Matches[1]
            $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $Matches[2], 'Process')
        }
    }
    foreach ($name in @('CUDA_PATH','CUDACXX','VSLANG')) {
        if (-not $savedEnvironment.ContainsKey($name)) { $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    }
    $env:CUDA_PATH = $cuda
    $env:CUDACXX = Join-Path $cuda 'bin/nvcc.exe'
    $env:VSLANG = '1033'
    $env:PATH = (Join-Path $cuda 'bin') + ';' + $env:PATH
    $configuration = @(
        '-S', $source, '-B', $build, '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja",
        '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_CUDA_COMPILER=$env:CUDACXX", "-DCUDAToolkit_ROOT=$cuda",
        "-DCMAKE_CUDA_ARCHITECTURES=$architectures", '-DBUILD_SHARED_LIBS=ON',
        # The archive has no .git; prevent ggml from reporting the parent project's commit.
        '-DGIT_EXE:FILEPATH=OFF',
        '-DLLAMA_BUILD_NUMBER=11960', "-DLLAMA_BUILD_COMMIT=$commit", '-DLLAMA_BUILD_TESTS=OFF',
        '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON',
        '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF', '-DLLAMA_OPENSSL=OFF',
        '-DGGML_NATIVE=OFF', '-DGGML_BACKEND_DL=ON', '-DGGML_CPU_ALL_VARIANTS=ON',
        '-DGGML_CUDA=ON', '-DGGML_CUDA_COMPRESSION_MODE=size', '-DGGML_OPENMP=OFF', '-DGGML_OPENMP_FETCH=OFF'
    )
    Write-Host "正在配置 llama.cpp b11960 / CUDA 12.8 ($architectures)"
    Invoke-Cmake $configuration (Join-Path $buildRoot 'configure.log') 'CMake 配置失败，请查看 .tools/llama-build-cuda128/configure.log 并修复工具链后重试。'
    Invoke-Cmake @('--build',$build,'--target','llama-server','--parallel',"$Jobs") (Join-Path $buildRoot 'build.log') 'llama-server 编译失败，请查看 .tools/llama-build-cuda128/build.log 并修复后重试。'

    $bin = Join-Path $build 'bin'
    $serverDlls = @('llama-server-impl.dll','llama-common.dll','llama.dll','mtmd.dll','ggml.dll','ggml-base.dll','ggml-cuda.dll')
    $serverDlls += @(Get-ChildItem -LiteralPath $bin -File -Filter 'ggml-cpu*.dll' | ForEach-Object Name)
    $cudaDlls = @('cudart64_12.dll','cublas64_12.dll','cublasLt64_12.dll')
    foreach ($native in Get-ChildItem -LiteralPath $output -File | Where-Object { $_.Extension -in @('.dll','.exe') }) {
        if ($native.Name -ne 'llama-server.exe' -and $native.Name -notin $serverDlls -and $native.Name -notin $cudaDlls -and $native.Name -notmatch '^(msvcp|vcruntime|vcomp|concrt)[0-9_].*\.dll$') {
            throw "输出目录中存在无关程序或运行库，请改用干净的输出目录：$($native.FullName)"
        }
    }
    Copy-Item -LiteralPath (Join-Path $bin 'llama-server.exe') -Destination $output -Force
    foreach ($dll in $serverDlls) { Copy-Item -LiteralPath (Join-Path $bin $dll) -Destination $output -Force }
    $licenses = Join-Path $output 'licenses'
    New-Item -ItemType Directory -Path $licenses -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $source 'LICENSE') -Destination (Join-Path $licenses 'LICENSE-llama.cpp') -Force
    $notices = @{
        'LICENSE-jsonhpp' = 'licenses/LICENSE-jsonhpp'
        'LICENSE-cpp-httplib' = 'vendor/cpp-httplib/LICENSE'
        'LICENSE-xxhash' = 'vendor/hash/xxhash/LICENSE'
        'LICENSE-sha256' = 'vendor/hash/sha256/LICENSE'
        'LICENSE-rotate-bits' = 'vendor/hash/rotate-bits/LICENSE.md'
    }
    foreach ($name in $notices.Keys) { Copy-Item -LiteralPath (Join-Path $source $notices[$name]) -Destination (Join-Path $licenses $name) -Force }
    # Stage only the CRT imports used by the shared, non-OpenMP CPU backend.
    $cudaRuntime = & (Join-Path $PSScriptRoot 'setup-cuda-runtime.ps1') -RuntimeDirectory $output
    $redist = Join-Path $env:VCToolsRedistDir 'x64'
    $dumpbin = Join-Path $env:VCToolsInstallDir 'bin/Hostx64/x64/dumpbin.exe'
    $runtimeCandidates = @{}
    foreach ($directory in @('Microsoft.VC143.CRT','Microsoft.VC143.OpenMP')) {
        $runtimeDirectory = Join-Path $redist $directory
        if (-not (Test-Path -LiteralPath $runtimeDirectory)) {
            if ($directory -eq 'Microsoft.VC143.CRT') { throw "缺少 MSVC 可分发运行库：$runtimeDirectory" }
            continue
        }
        foreach ($dll in Get-ChildItem -LiteralPath $runtimeDirectory -File -Filter '*.dll') { $runtimeCandidates[$dll.Name] = $dll.FullName }
    }
    $pending = New-Object 'System.Collections.Generic.Queue[string]'
    Get-ChildItem -LiteralPath $output -File | Where-Object { $_.Extension -in @('.dll','.exe') -and -not $runtimeCandidates.ContainsKey($_.Name) } | ForEach-Object { $pending.Enqueue($_.FullName) }
    $checked = @{}
    while ($pending.Count) {
        $binary = $pending.Dequeue()
        if ($checked.ContainsKey($binary)) { continue }
        $checked[$binary] = $true
        $imports = & $dumpbin /nologo /dependents $binary
        if ($LASTEXITCODE -ne 0) { throw "无法读取 PE 依赖：$binary" }
        foreach ($line in $imports) {
            if ($line -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$' -and $runtimeCandidates.ContainsKey($Matches[1])) {
                $destination = Join-Path $output $Matches[1]
                Copy-Item -LiteralPath $runtimeCandidates[$Matches[1]] -Destination $destination -Force
                $pending.Enqueue($destination)
            }
        }
    }
    foreach ($name in $runtimeCandidates.Keys) {
        $candidate = Join-Path $output $name
        if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and -not $checked.ContainsKey($candidate)) { Remove-Item -LiteralPath $candidate }
    }
    foreach ($required in @('llama-server.exe','llama-server-impl.dll','llama.dll','ggml.dll','ggml-base.dll','ggml-cuda.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $output $required) -PathType Leaf)) { throw "缺少构建产物：$required" }
    }
    if (-not @(Get-ChildItem -LiteralPath $output -File -Filter 'ggml-cpu*.dll').Count) { throw '构建产物缺少 CPU 后端。' }
    $buildInfo = Get-Content -LiteralPath (Join-Path $build 'common/build-info.cpp') -Raw
    if ($buildInfo -notmatch 'LLAMA_BUILD_NUMBER = 11960;' -or -not $buildInfo.Contains('LLAMA_COMMIT = "' + $commit + '"')) {
        throw '生成的构建版本不匹配固定 b11960 commit，请检查 CMake 配置。'
    }
    $compiler = [regex]::Match($buildInfo, 'LLAMA_COMPILER = "([^"]+)"').Groups[1].Value
    $sourceFiles = @(Get-ChildItem -LiteralPath $source -File -Recurse -Force | Where-Object { $_.Name -ne '.llama-source-sha256' } | Sort-Object FullName | ForEach-Object {
        $_.FullName.Substring($source.Length + 1).Replace('\','/') + "`t" + (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $sourceTreeHash = ([BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes(($sourceFiles -join "`n"))))).Replace('-','').ToLowerInvariant() } finally { $hasher.Dispose() }
    $files = @(Get-ChildItem -LiteralPath $output -File -Recurse | Where-Object { $_.FullName -ne $manifestPath -and $_.Name -ne '.streamtrans-build-owner' } | ForEach-Object {
        [ordered]@{ name = $_.FullName.Substring($output.Length + 1).Replace('\','/'); size = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    $ggmlFiles = @($files | Where-Object { $_.name -match '^ggml(?:-[a-z0-9_-]+)?\.dll$' } | Sort-Object name)
    [ordered]@{
        version = 'b11960'; commit = $commit; cuda_build = '12.8'; cuda_runtime = $cudaRuntime
        compiler = $compiler; msvc_toolset = $env:VCToolsVersion; windows_sdk = $env:WindowsSDKVersion
        cmake = (& $cmake --version | Select-Object -First 1); ninja = (& $ninja --version); nvcc = (& $env:CUDACXX --version) -join "`n"
        source_tree_sha256 = $sourceTreeHash; nvcc_sha256 = (Get-FileHash -LiteralPath $env:CUDACXX -Algorithm SHA256).Hash.ToLowerInvariant()
        source_directory = $source; build_directory = $build
        ggml = @{ commit = $commit; compatibility_id = $compatibilityId; cpu_openmp = $false; patches = $patchRecords; files = $ggmlFiles }
        cmake_arguments = $configuration
        cuda_architectures = $architectures.Split(';'); cuda_compression = 'size'; cpu_all_variants = $true; cpu_openmp = $false
        ui = $false; openssl = $false; target = 'llama-server'; packages = $packages; files = $files
    } | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Write-Host "已构建 llama.cpp b11960 / CUDA 12.8：$output"
} finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process') }
}
