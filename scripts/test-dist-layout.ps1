[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ScriptAst([string]$Name) {
    $tokens = $null
    $errors = $null
    $path = Join-Path $PSScriptRoot $Name
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw "PowerShell 语法检查失败：$Name" }
    return $ast
}

function Import-Function($Ast, [string]$Name) {
    $node = $Ast.Find({ param($item) $item -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq $Name }, $true)
    if (-not $node) { throw "自检缺少函数：$Name" }
    return $node.Extent.Text
}

function Get-Assignment($Ast, [string]$Name) {
    $node = $Ast.Find({ param($item)
        $item -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $item.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
        $item.Left.VariablePath.UserPath -eq $Name
    }, $true)
    if (-not $node) { throw "自检缺少配置：$Name" }
    return $node.Extent.Text
}

$dist = Get-ScriptAst 'dist.ps1'
$cuda = Get-ScriptAst 'setup-cuda-runtime.ps1'
$ffmpegSetup = Get-ScriptAst 'setup-ffmpeg.ps1'
Get-ScriptAst 'setup.ps1' | Out-Null
Get-ScriptAst 'setup-llama.ps1' | Out-Null
Get-ScriptAst 'build-llama.ps1' | Out-Null
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'build-ffmpeg.ps1')) { Get-ScriptAst 'build-ffmpeg.ps1' | Out-Null }
foreach ($name in @('Assert-EmptyOutput','Copy-Exclusive','Assert-RecordedArtifact')) { Invoke-Expression (Import-Function $dist $name) }
foreach ($name in @('Test-VerifiedFile','Copy-VerifiedFile')) { Invoke-Expression (Import-Function $cuda $name) }
Invoke-Expression (Get-Assignment $dist 'asrDlls')
Invoke-Expression (Get-Assignment $dist 'llamaCoreDlls')
Invoke-Expression (Get-Assignment $dist 'llamaOptionalDlls')
Invoke-Expression (Get-Assignment $dist 'sharedCuda')
Invoke-Expression (Get-Assignment $dist 'sharedMsvc')
Invoke-Expression (Get-Assignment $dist 'ffmpegDllHashes')
Invoke-Expression (Get-Assignment $dist 'ffmpegDlls')
$distFfmpegHashes = $ffmpegDllHashes
Invoke-Expression (Get-Assignment $ffmpegSetup 'ffmpegDllHashes')
if ($distFfmpegHashes.Count -ne 4 -or $ffmpegDllHashes.Count -ne 4 -or $ffmpegDlls.Count -ne 5 -or $ffmpegDlls -notcontains 'streamtrans-ffmpeg.dll') { throw 'FFmpeg 动态库集合不完整' }
foreach ($name in $ffmpegDllHashes.Keys) { if ($ffmpegDllHashes[$name] -ne $distFfmpegHashes[$name]) { throw "安装与分发 FFmpeg 校验值不一致：$name" } }
Invoke-Expression (Get-Assignment $dist 'ffmpegLicenseHash')
Invoke-Expression (Get-Assignment $ffmpegSetup 'licenseHash')
if ($ffmpegLicenseHash -ne $licenseHash) { throw '安装与分发 FFmpeg 许可证校验值不一致' }
if (@($ffmpegDlls | Where-Object { $_ -match '\.exe$|avfilter|swscale|avdevice' }).Count) { throw 'FFmpeg 发行集合包含无关组件' }
Invoke-Expression (Get-Assignment $cuda 'packages')
$packageDlls = @{}
foreach ($package in $packages) { foreach ($name in $package.Dlls.Keys) { $packageDlls[$name] = $package.Dlls[$name] } }
if ($sharedCuda.Count -ne 3 -or $packageDlls.Count -ne 3) { throw 'CUDA 运行文件集合不完整' }
foreach ($name in $sharedCuda.Keys) { if ($sharedCuda[$name] -ne $packageDlls[$name]) { throw "安装与分发 CUDA 校验值不一致：$name" } }
if ($asrDlls.Count -ne 5 -or $asrDlls -contains 'ggml-vulkan.dll') { throw 'ASR 运行文件集合不正确' }
foreach ($metadata in @(
    [pscustomobject]@{ cuda_runtime = [pscustomobject]@{ version = '12.8.1' } },
    [pscustomobject]@{ cuda_runtime = [pscustomobject]@{ version = '12.4' } },
    [pscustomobject]@{ cuda_runtime = '12.8.1' },
    [pscustomobject]@{ cuda = '12.4' }
)) {
    $llamaProvenance = $metadata
    Invoke-Expression (Get-Assignment $dist 'llamaRuntimeMatches')
    $expected = $metadata.PSObject.Properties['cuda_runtime'] -and
        $metadata.cuda_runtime -is [pscustomobject] -and $metadata.cuda_runtime.version -eq '12.8.1'
    if ([bool]$llamaRuntimeMatches -ne [bool]$expected) { throw 'CUDA 运行库来源检查不正确' }
}

$temporary = Join-Path ([IO.Path]::GetTempPath()) ('streamtrans-dist-layout-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($temporary) | Out-Null
$source = Join-Path $temporary 'source.txt'
$target = Join-Path $temporary 'target.txt'
$nativeDirectory = Join-Path $temporary 'native'
$nativeNames = @('ggml-cpu.dll','ggml-cpu-zen4.dll','vcomp140.dll','llama-bench-impl.dll','llama-embedding-impl.dll','unrelated.dll','ggml-rpc.dll','ggml-vulkan.dll') + $llamaCoreDlls + @($sharedCuda.Keys) + $sharedMsvc
try {
    $output = $temporary
    Assert-EmptyOutput
    [IO.File]::WriteAllText($source, 'first')
    $rejected = $false
    try { Assert-EmptyOutput } catch { $rejected = $true }
    if (-not $rejected) { throw '非空分发目录没有被拒绝' }
    Copy-Exclusive $source $target
    [IO.File]::WriteAllText($source, 'second')
    $rejected = $false
    try { Copy-Exclusive $source $target } catch { $rejected = $true }
    if (-not $rejected -or [IO.File]::ReadAllText($target) -ne 'first') { throw '同名分发文件被覆盖' }
    $expected = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    Assert-RecordedArtifact $temporary @([pscustomobject]@{ name = 'source.txt'; sha256 = $expected }) 'source.txt'
    foreach ($records in @(
        @([pscustomobject]@{ name = 'source.txt'; sha256 = ('0' * 64) }),
        @([pscustomobject]@{ name = 'source.txt'; sha256 = $expected }, [pscustomobject]@{ name = 'source.txt'; sha256 = $expected }),
        @([pscustomobject]@{ name = 'other.txt'; sha256 = $expected })
    )) {
        $rejected = $false
        try { Assert-RecordedArtifact $temporary $records 'source.txt' } catch { $rejected = $true }
        if (-not $rejected) { throw '构建文件校验未拒绝损坏、重复或缺失的记录' }
    }
    if (Test-VerifiedFile $target $expected) { throw '损坏的缓存通过了校验' }
    Copy-VerifiedFile $source $target $expected
    if (-not (Test-VerifiedFile $target $expected)) { throw '校验后复制失败' }
    $rejected = $false
    try { Copy-VerifiedFile $source $target ('0' * 64) } catch { $rejected = $true }
    if (-not $rejected -or [IO.File]::ReadAllText($target) -ne 'second') { throw '错误校验值没有被安全拒绝' }
    [IO.Directory]::CreateDirectory($nativeDirectory) | Out-Null
    foreach ($name in $nativeNames) { [IO.File]::WriteAllText((Join-Path $nativeDirectory $name), 'fixture') }
    $llama = $nativeDirectory
    Invoke-Expression (Get-Assignment $dist 'llamaDlls')
    foreach ($name in @('ggml-cpu.dll','ggml-cpu-zen4.dll','vcomp140.dll') + $llamaCoreDlls) {
        if ($llamaDlls -notcontains $name) { throw "必要原生依赖被过滤：$name" }
    }
    if (@($llamaDlls | Where-Object { $_ -in $sharedCuda.Keys -or $_ -in $sharedMsvc -or $_ -match 'bench|embedding|unrelated|rpc|vulkan' }).Count) { throw '共享运行库或无关组件重复进入后端目录' }
} finally {
    # Only this self-check's two files and now-empty private directory are removed.
    [IO.File]::Delete($source)
    [IO.File]::Delete($target)
    foreach ($name in $nativeNames) { [IO.File]::Delete((Join-Path $nativeDirectory $name)) }
    if ([IO.Directory]::Exists($nativeDirectory)) { [IO.Directory]::Delete($nativeDirectory) }
    [IO.Directory]::Delete($temporary)
}
Write-Output '分发布局自检通过：CUDA/FFmpeg 校验一致，拒绝目录/文件覆盖，保留必要后端并去重共享运行库。'
