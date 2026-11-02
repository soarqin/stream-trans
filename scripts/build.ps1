[CmdletBinding()]
param(
    [ValidateSet('build', 'test', 'clippy')][string]$Action = 'build',
    [ValidateRange(0, 255)][int]$Gpu = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw '找不到 cargo，请先安装 Rust stable x64 MSVC 工具链。' }
# 保留 -Gpu 参数以兼容旧命令；Rust 构建不需要 GPU、Python 或 CUDA Toolkit。
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    switch ($Action) {
        'build' { & cargo build --release --bin streamtrans --locked }
        'test' { & cargo test --locked }
        'clippy' { & cargo clippy --all-targets --locked -- -D warnings }
    }
    if ($LASTEXITCODE -ne 0) { throw "cargo $Action 失败，请修复上方错误后重试。" }
} finally {
    Pop-Location
}
