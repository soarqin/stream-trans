[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$RequestJson = '.tools/benchmark/llama-ja-k40-request.json',
    [string]$OutputDirectory = ('.tools/benchmark/llama-aligned-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [ValidateRange(1, 100)][int]$WarmRuns = 3,
    [ValidateRange(0, 1000000)][int]$ExpectedPromptTokens = 99,
    [ValidateRange(1, 3600)][int]$StartupTimeoutSeconds = 300,
    [ValidateRange(1, 3600)][int]$RequestTimeoutSeconds = 600
)

# Benchmark only. Uses existing model/server files; never installs or downloads dependencies.
$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne 'Win32NT') { throw '此脚本用于 Windows PowerShell。' }
$serverPath = (Resolve-Path -LiteralPath $Server).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$requestPath = (Resolve-Path -LiteralPath $RequestJson).Path
$request = Get-Content -LiteralPath $requestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not ($request.prompt -is [string]) -or [string]::IsNullOrWhiteSpace($request.prompt)) {
    throw '请求 JSON 缺少非空 prompt。请提供原始 /completion 请求。'
}
$defaults = [ordered]@{
    n_predict = 2048; seed = 42; temperature = 1.0; top_k = 40; top_p = 0.95; min_p = 0.0
    samplers = @('temperature', 'top_k', 'top_p')
    repeat_penalty = 1.0; presence_penalty = 0.0; frequency_penalty = 0.0
}
foreach ($name in $defaults.Keys) {
    if (-not $request.PSObject.Properties[$name]) { $request | Add-Member -NotePropertyName $name -NotePropertyValue $defaults[$name] }
}
# Always disable prompt reuse and streaming so each request measures a full prefill and decode.
foreach ($pair in @(@('cache_prompt', $false), @('stream', $false), @('return_tokens', $true))) {
    $request | Add-Member -NotePropertyName $pair[0] -NotePropertyValue $pair[1] -Force
}
$body = $request | ConvertTo-Json -Depth 30 -Compress
$utf8 = New-Object System.Text.UTF8Encoding($false)
if ((Test-Path -LiteralPath $OutputDirectory) -and @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count) {
    throw '输出目录已有文件。请用 -OutputDirectory 指定新的空目录，避免混合不同轮次结果。'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$outputPath = (Resolve-Path -LiteralPath $OutputDirectory).Path
function Save-Json([string]$Name, $Value) {
    [IO.File]::WriteAllText((Join-Path $outputPath $Name), ($Value | ConvertTo-Json -Depth 40), $utf8)
}
function Median($Values) {
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return $sorted[$middle] }
    return ($sorted[$middle - 1] + $sorted[$middle]) / 2
}
function Post-Json([string]$Route, [string]$Json) {
    Invoke-WebRequest -UseBasicParsing -Uri ($endpoint + $Route) -Method Post -ContentType 'application/json; charset=utf-8' -Body $utf8.GetBytes($Json) -TimeoutSec $RequestTimeoutSeconds
}

$listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = $listener.LocalEndpoint.Port
$listener.Stop()
$endpoint = 'http://127.0.0.1:' + $port
$serverArgs = @(
    '--model', ('"' + $modelPath + '"'), '--alias', 'MiniCPM5-2B', '--host', '127.0.0.1', '--port', $port,
    '--device', 'CUDA0', '--n-gpu-layers', '99', '--split-mode', 'none', '--fit', 'off',
    '--ctx-size', '4096', '--batch-size', '256', '--ubatch-size', '128', '--parallel', '1', '--threads', '8',
    '--cache-type-k', 'f32', '--cache-type-v', 'f32', '--flash-attn', 'off',
    '--no-warmup', '--no-cache-prompt', '--cache-ram', '0', '--verbosity', '4'
)
$process = $null
try {
    $loading = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $serverPath -ArgumentList $serverArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputPath 'server.stdout.log') -RedirectStandardError (Join-Path $outputPath 'server.stderr.log')
    $ready = $false
    while ($loading.Elapsed.TotalSeconds -lt $StartupTimeoutSeconds) {
        if ($process.HasExited) { throw 'llama-server 在就绪前退出。请查看输出目录中的 server 日志。' }
        try {
            $health = Invoke-WebRequest -UseBasicParsing -Uri ($endpoint + '/health') -TimeoutSec 2
            if ($health.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 100
    }
    if (-not $ready -or $process.HasExited) { throw 'llama-server 就绪检查超时。请查看 server 日志并检查 CUDA 和显存。' }
    $serverReadyMs = $loading.Elapsed.TotalMilliseconds
    $logs = (Get-Content (Join-Path $outputPath 'server.stdout.log') -Raw) + (Get-Content (Join-Path $outputPath 'server.stderr.log') -Raw)
    $gpuLayers = [regex]::Matches($logs, 'offloaded (\d+)/(\d+) layers to GPU')
    if (-not @($gpuLayers | Where-Object { [int]$_.Groups[1].Value -gt 0 -and $_.Groups[1].Value -eq $_.Groups[2].Value }).Count) {
        throw '日志未确认全部模型层已加载到 GPU。请检查原生库、驱动和 server 日志。'
    }
    Save-Json 'request.effective.json' $request
    Save-Json 'environment.json' ([ordered]@{
        server = $serverPath; model = $modelPath; request_source = $requestPath; server_arguments = $serverArgs
        process_id = $process.Id; endpoint = $endpoint; server_ready_ms = $serverReadyMs
        expected_prompt_tokens = $ExpectedPromptTokens; warm_runs = $WarmRuns
        cold_definition = 'first /completion after fresh server load; --no-warmup; OS file cache is not flushed'
        tokenize_precheck = 'CPU tokenizer check before the first completion; no generation warmup'
        kv_type = 'f32'; flash_attention = 'off'; context_size = 4096; batch_size = 256; ubatch_size = 128
    })
    $tokenizeBody = @{ content = $request.prompt; add_special = $false; parse_special = $true } | ConvertTo-Json -Compress
    $tokenized = (Post-Json '/tokenize' $tokenizeBody).Content | ConvertFrom-Json
    if ($ExpectedPromptTokens -gt 0 -and @($tokenized.tokens).Count -ne $ExpectedPromptTokens) {
        throw "prompt token 数为 $(@($tokenized.tokens).Count)，预期 $ExpectedPromptTokens。请核对请求和模型；其他输入可用 -ExpectedPromptTokens 显式指定。"
    }
    Save-Json 'prompt.tokens.json' $tokenized
    $runs = @()
    for ($run = 0; $run -le $WarmRuns; $run++) {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $http = Post-Json '/completion' $body
        $wallMs = $clock.Elapsed.TotalMilliseconds
        $response = $http.Content | ConvertFrom-Json
        [IO.File]::WriteAllText((Join-Path $outputPath ("response-$run.json")), $http.Content, $utf8)
        if ($response.timings.cache_n -ne 0) { throw '服务复用了 prompt 缓存，当前结果不能作为完整 prefill 基准。' }
        if ($response.truncated -or $response.stop_type -ne 'eos') { throw '生成未通过 EOS 正常结束。请检查完整响应与 token 预算。' }
        $phase = if ($run -eq 0) { 'cold' } else { 'warm' }
        $row = [ordered]@{ run = $run; phase = $phase; wall_ms = $wallMs; tokens_predicted = $response.tokens_predicted; tokens_evaluated = $response.tokens_evaluated; response_tokens_count = @($response.tokens).Count; timings = $response.timings; stop_type = $response.stop_type }
        $runs += [pscustomobject]$row
        $line = $row | ConvertTo-Json -Depth 10 -Compress
        [IO.File]::AppendAllText((Join-Path $outputPath 'runs.jsonl'), ($line + [Environment]::NewLine), $utf8)
        Write-Output $line
    }
    $warm = @($runs | Where-Object phase -eq 'warm')
    $summary = [ordered]@{ event = 'warm_summary'; warm_runs = $WarmRuns; median_wall_ms = (Median $warm.wall_ms); median_prefill_ms = (Median $warm.timings.prompt_ms); median_decode_ms = (Median $warm.timings.predicted_ms); median_decode_tokens_per_second = (Median $warm.timings.predicted_per_second) }
    Save-Json 'summary.json' $summary
    Write-Output ($summary | ConvertTo-Json -Compress)
} finally {
    if ($process -and -not $process.HasExited) { $process.Kill(); $process.WaitForExit() }
}
