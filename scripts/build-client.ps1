[CmdletBinding()]
param(
    [ValidateSet('build', 'test', 'clippy')][string]$Action = 'build'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw '找不到 cargo，请先安装 Rust stable x64 MSVC 工具链和 Visual Studio C++ Build Tools。'
}

# 仅构建桌面客户端；不运行服务端安装脚本，不下载模型或 GPU 运行库。
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    switch ($Action) {
        'build' { & cargo build --release -p streamtrans-client --locked }
        'test' { & cargo test -p streamtrans-client --locked }
        'clippy' { & cargo clippy -p streamtrans-client --all-targets --locked -- -D warnings }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "客户端 cargo $Action 失败，请修复上方错误后重试。"
    }
} finally {
    Pop-Location
}
