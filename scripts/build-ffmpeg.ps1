[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$FfmpegDirectory,
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw '此构建需要 Windows x64 PowerShell。' }
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$ffmpegRoot = [IO.Path]::GetFullPath($FfmpegDirectory)
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $ffmpegRoot 'bin' }
$output = [IO.Path]::GetFullPath($OutputDirectory)
$buildRoot = Join-Path $projectRoot '.tools/ffmpeg-bridge-build'
$source = Join-Path $projectRoot 'native/streamtrans_ffmpeg.c'
$include = Join-Path $ffmpegRoot 'include'
$lib = Join-Path $ffmpegRoot 'lib'
foreach ($path in @($source, (Join-Path $include 'libavformat/avformat.h')) + @(
    'avformat.lib','avcodec.lib','avutil.lib','swresample.lib' | ForEach-Object { Join-Path $lib $_ }
)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "缺少 FFmpeg 桥构建文件：$path" }
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw '请安装包含 x64 C++ 工作负载的 Visual Studio Build Tools。' }
$vs = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
if (-not $vs) { throw '找不到 Visual Studio x64 C++ 工作负载。' }
New-Item -ItemType Directory -Path $buildRoot,$output -Force | Out-Null
$vcvars = Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$environmentScript = Join-Path $buildRoot 'build-environment.cmd'
Set-Content -LiteralPath $environmentScript -Encoding ascii -Value @(
    '@echo off', ('set "PATH=' + (Split-Path $vswhere -Parent) + ';%PATH%"'),
    ('call "' + $vcvars + '" >nul'), 'if errorlevel 1 exit /b 1', 'set'
)
$environment = & $env:ComSpec /d /c $environmentScript
if ($LASTEXITCODE -ne 0) { throw 'Visual Studio 构建环境初始化失败。' }
$savedEnvironment = @{}
try {
    foreach ($line in $environment) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $Matches[1]
            $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $Matches[2], 'Process')
        }
    }
    $dll = Join-Path $output 'streamtrans-ffmpeg.dll'
    $arguments = @(
        '/nologo','/TC','/std:c11','/utf-8','/O2','/MD','/W4','/WX','/LD',
        '/D_CRT_SECURE_NO_WARNINGS','/external:W0', "/external:I$include", $source,
        ('/Fo' + (Join-Path $buildRoot 'streamtrans_ffmpeg.obj')),
        '/link', ('/OUT:' + $dll), ('/IMPLIB:' + (Join-Path $buildRoot 'streamtrans-ffmpeg.lib')),
        ('/LIBPATH:' + $lib), 'avformat.lib','avcodec.lib','avutil.lib','swresample.lib'
    )
    $ErrorActionPreference = 'Continue'
    & cl.exe @arguments 2>&1 | Tee-Object -FilePath (Join-Path $buildRoot 'build.log') | ForEach-Object { Write-Host $_ }
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($exitCode -ne 0) { throw 'FFmpeg 桥编译失败，请查看 .tools/ffmpeg-bridge-build/build.log。' }
    if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) { throw 'FFmpeg 桥编译未生成 streamtrans-ffmpeg.dll。' }
    Write-Host "FFmpeg 9 音频桥已构建：$dll"
} finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process') }
}
