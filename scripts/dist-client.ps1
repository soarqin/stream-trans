[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [switch]$SkipBuild,
    [string]$MsvcRedistDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot 'dist/StreamTransClient' }
$output = [IO.Path]::GetFullPath($OutputDirectory)

function Assert-EmptyOutput {
    if (Test-Path -LiteralPath $output) {
        $existing = Get-Item -LiteralPath $output -Force
        if (-not $existing.PSIsContainer -or ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -or @(Get-ChildItem -LiteralPath $output -Force).Count -gt 0) {
            throw "输出目录必须不存在或为空，脚本不会覆盖非空目录或目录链接：$output。请指定新的 -OutputDirectory。"
        }
    }
}

Assert-EmptyOutput
$assets = Join-Path $projectRoot 'client/assets'
$sources = @(
    @{ Source = (Join-Path $assets 'output.css'); Target = 'output.css' },
    @{ Source = (Join-Path $projectRoot 'docs/client.md'); Target = 'README.md' },
    @{ Source = (Join-Path $projectRoot 'LICENSE'); Target = 'licenses/LICENSE-StreamTrans' },
    @{ Source = (Join-Path $assets 'font-OFL.txt'); Target = 'licenses/font-OFL.txt' }
)
foreach ($file in $sources) {
    if (-not (Test-Path -LiteralPath $file.Source -PathType Leaf)) {
        throw "缺少客户端分发文件：$($file.Source)。请恢复该文件后重新打包。"
    }
}
$fontSource = Join-Path $assets 'font-source.json'
if (Test-Path -LiteralPath $fontSource -PathType Leaf) {
    $sources += @{ Source = $fontSource; Target = 'licenses/font-source.json' }
}

if (-not $SkipBuild) { & (Join-Path $PSScriptRoot 'build-client.ps1') }
$executable = Join-Path $projectRoot 'target/release/streamtrans-client.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw '找不到 target/release/streamtrans-client.exe，请先执行 .\scripts\build-client.ps1，或移除 -SkipBuild 后重试。'
}
$sources = @(@{ Source = $executable; Target = 'streamtrans-client.exe' }) + $sources

# Windows x64 MSVC 发布构建实际导入 VCRUNTIME140.dll；只带所需运行库。
if (-not $MsvcRedistDirectory) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        $vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $vsRoot) {
            $redistRoot = Join-Path $vsRoot 'VC/Redist/MSVC'
            $versions = @(Get-ChildItem -LiteralPath $redistRoot -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^\d+(\.\d+)+$' } | Sort-Object { [version]$_.Name } -Descending)
            foreach ($version in $versions) {
                $crt = Get-ChildItem -LiteralPath (Join-Path $version.FullName 'x64') -Directory -Filter 'Microsoft.VC*.CRT' -ErrorAction SilentlyContinue | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'vcruntime140.dll') -PathType Leaf } | Select-Object -First 1
                if ($crt) { $MsvcRedistDirectory = $crt.FullName; break }
            }
        }
    }
}
if (-not $MsvcRedistDirectory -or -not (Test-Path -LiteralPath (Join-Path $MsvcRedistDirectory 'vcruntime140.dll') -PathType Leaf)) {
    throw '找不到 x64 vcruntime140.dll，请通过 -MsvcRedistDirectory 指定 Visual Studio 的 x64 Microsoft.VC*.CRT 目录后重试。'
}
$sources += @{ Source = (Join-Path $MsvcRedistDirectory 'vcruntime140.dll'); Target = 'vcruntime140.dll' }

# 许可证可单独准备，不要求安装服务端运行库。
$licenseName = 'Visual-C-Runtime-2015-2022-License.docx'
$licenseHash = 'f1e3d56ceb2ad68aae0711b910375009e651ac5530fa0760f0dea6e81e54fae1'
$licenseDirectory = Join-Path $projectRoot '.tools/licenses/msvc'
$runtimeLicense = Join-Path $licenseDirectory $licenseName
if (-not (Test-Path -LiteralPath $runtimeLicense -PathType Leaf)) {
    [IO.Directory]::CreateDirectory($licenseDirectory) | Out-Null
    $temporaryLicense = Join-Path $licenseDirectory ($licenseName + '.' + [guid]::NewGuid().ToString('N') + '.download')
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri 'https://visualstudio.microsoft.com/wp-content/uploads/2021/09/Visual-C-Runtime-2015-2022-License-1.docx' -OutFile $temporaryLicense
        if ((Get-FileHash -LiteralPath $temporaryLicense -Algorithm SHA256).Hash -ine $licenseHash) { throw 'Microsoft Runtime 许可证校验失败，请检查网络后重试。' }
        [IO.File]::Move($temporaryLicense, $runtimeLicense)
    } finally {
        if (Test-Path -LiteralPath $temporaryLicense -PathType Leaf) { [IO.File]::Delete($temporaryLicense) }
    }
}
if ((Get-FileHash -LiteralPath $runtimeLicense -Algorithm SHA256).Hash -ine $licenseHash) {
    throw "Microsoft Runtime 许可证缓存校验失败：$runtimeLicense。请移走该缓存文件后重新运行打包脚本。"
}
$sources += @{ Source = $runtimeLicense; Target = ('licenses/' + $licenseName) }

# 构建期间目标目录可能变化，因此创建前再次检查。不清理已有文件。
Assert-EmptyOutput
[IO.Directory]::CreateDirectory($output) | Out-Null
[IO.Directory]::CreateDirectory((Join-Path $output 'licenses')) | Out-Null
foreach ($file in $sources) {
    # overwrite=false：即使复制期间出现同名文件，也不会覆盖。
    [IO.File]::Copy($file.Source, (Join-Path $output $file.Target), $false)
}
Write-Host "客户端已打包：$output"
Write-Host '启动 streamtrans-client.exe 后，填写服务端 API 地址并选择录音设备。此包不包含模型、GPU 运行库或服务端。'
