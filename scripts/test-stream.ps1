<#
.SYNOPSIS
持续发送音频并并行接收流式识别、翻译事件，不需要 Python。
.EXAMPLE
.\scripts\test-stream.ps1 -Wav .\audio.wav -AssertBeforeFinish -OutputPath .\stream.jsonl
.EXAMPLE
.\scripts\test-stream.ps1 -Wav .\audio.mp3 -Ffmpeg C:\Tools\ffmpeg\bin\ffmpeg.exe -SendAsFastAsPossible
.EXAMPLE
.\scripts\test-stream.ps1 -Wav .\audio.wav -RequestKey your-key
.NOTES
默认按真实时长发送 100 ms 音频块。快速发送可能触发服务端 audio_buffer_full，适合验证 finish。
默认直接读取 16 kHz 单声道 PCM16 WAV；-Ffmpeg 仅用于开发测试前转换其他格式，不是新服务端发行包的运行依赖。
请求 key 默认从 STREAMTRANS_REQUEST_KEY 读取，以 Bearer 握手头发送，不放入 URL 或输出记录。
每行输出 elapsedMs、sentAudioMs、finishSent 和完整 event。finishSent 从开始发送 finish 时置真，
因此 AssertBeforeFinish 要求首个 transcript 在此之前被接收；翻译模式还要求首个 translation 提前返回。
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Wav,
    [uri]$Uri = 'ws://127.0.0.1:3000/v1/audio/translations/stream?source_language=auto&target_language=en',
    [AllowEmptyString()][string]$RequestKey = $env:STREAMTRANS_REQUEST_KEY,
    [string]$Ffmpeg,
    [ValidateRange(1, 1000)][int]$ChunkMs = 100,
    [bool]$Realtime = $true,
    [switch]$SendAsFastAsPossible,
    [switch]$AssertBeforeFinish,
    [ValidateRange(1, 3600)][int]$TimeoutSeconds = 300,
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$hasRequestKey = $PSBoundParameters.ContainsKey('RequestKey') -or $null -ne $env:STREAMTRANS_REQUEST_KEY
if ($hasRequestKey -and $RequestKey -notmatch '\A[\x21-\x7E]{1,256}\z') { throw '请求 key 必须为 1–256 个可见 ASCII 字符，不能包含空白。' }
if ($Uri.Scheme -notin @('ws','wss')) { throw '-Uri 必须使用 ws:// 或 wss://。' }
if (-not (Test-Path -LiteralPath $Wav -PathType Leaf)) { throw "找不到输入音频：$Wav。请通过 -Wav 指定文件。" }

# .NET 异步发送在后台完成，主线程始终接收；不从后台线程调用 PowerShell 脚本块。
$clientSource = @'
using System;
using System.Diagnostics;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public sealed class StreamTransWsProbe {
    private int sentBytes;
    private int finishSent;
    public int SentBytes { get { return Volatile.Read(ref sentBytes); } }
    public bool FinishSent { get { return Volatile.Read(ref finishSent) != 0; } }

    public async Task SendAudioAsync(ClientWebSocket socket, byte[] pcm, int chunkBytes,
                                    bool realtime, CancellationToken cancellation) {
        var clock = Stopwatch.StartNew();
        for (int offset = 0; offset < pcm.Length;) {
            int count = Math.Min(chunkBytes, pcm.Length - offset);
            await socket.SendAsync(new ArraySegment<byte>(pcm, offset, count),
                WebSocketMessageType.Binary, true, cancellation).ConfigureAwait(false);
            offset += count;
            Interlocked.Exchange(ref sentBytes, offset);
            if (realtime) {
                long delay = offset / 32L - clock.ElapsedMilliseconds;
                if (delay > 0) await Task.Delay((int)delay, cancellation).ConfigureAwait(false);
            }
        }
        byte[] finish = Encoding.UTF8.GetBytes("{\"type\":\"finish\"}");
        Interlocked.Exchange(ref finishSent, 1);
        await socket.SendAsync(new ArraySegment<byte>(finish), WebSocketMessageType.Text,
            true, cancellation).ConfigureAwait(false);
    }

    private static string FourCc(BinaryReader reader) {
        byte[] bytes = reader.ReadBytes(4);
        if (bytes.Length != 4) throw new InvalidDataException("WAV 文件意外结束。");
        return Encoding.ASCII.GetString(bytes);
    }

    public static byte[] ReadPcmWav(string path) {
        using (var reader = new BinaryReader(File.OpenRead(path)))
        using (var pcm = new MemoryStream()) {
            if (reader.BaseStream.Length < 12 || FourCc(reader) != "RIFF")
                throw new InvalidDataException("音频不是 RIFF WAV，请用 -Ffmpeg 转换。");
            long end = reader.ReadUInt32() + 8L;
            if (FourCc(reader) != "WAVE" || end > reader.BaseStream.Length || end < 12)
                throw new InvalidDataException("WAV 的 RIFF 长度或 WAVE 标识无效。");
            bool formatSeen = false;
            byte[] buffer = new byte[65536];
            while (reader.BaseStream.Position + 8 <= end) {
                string kind = FourCc(reader);
                uint size = reader.ReadUInt32();
                long start = reader.BaseStream.Position;
                long next = start + size + (size & 1);
                if (next > end) throw new InvalidDataException("WAV chunk 超过 RIFF 边界。");
                if (kind == "fmt ") {
                    if (size < 16) throw new InvalidDataException("WAV fmt chunk 太短。");
                    ushort format = reader.ReadUInt16();
                    ushort channels = reader.ReadUInt16();
                    uint rate = reader.ReadUInt32();
                    uint bytesPerSecond = reader.ReadUInt32();
                    ushort blockAlign = reader.ReadUInt16();
                    ushort bits = reader.ReadUInt16();
                    if (format == 0xfffe && size >= 40) {
                        ushort extensionSize = reader.ReadUInt16();
                        ushort validBits = reader.ReadUInt16();
                        reader.ReadUInt32(); // channel mask
                        Guid subtype = new Guid(reader.ReadBytes(16));
                        if (extensionSize >= 22 && validBits == 16 &&
                            subtype == new Guid("00000001-0000-0010-8000-00aa00389b71")) format = 1;
                    }
                    if (format != 1 || channels != 1 || rate != 16000 || bits != 16 ||
                        blockAlign != 2 || bytesPerSecond != 32000)
                        throw new InvalidDataException("需要 16 kHz、单声道、PCM16LE WAV，请用 -Ffmpeg 转换。");
                    formatSeen = true;
                } else if (kind == "data") {
                    if ((size & 1) != 0 || pcm.Length + size > Int32.MaxValue)
                        throw new InvalidDataException("WAV PCM 数据长度无效或过大。");
                    long remaining = size;
                    while (remaining > 0) {
                        int read = reader.Read(buffer, 0, (int)Math.Min(buffer.Length, remaining));
                        if (read == 0) throw new InvalidDataException("WAV PCM 数据意外结束。");
                        pcm.Write(buffer, 0, read);
                        remaining -= read;
                    }
                }
                reader.BaseStream.Position = next;
            }
            if (!formatSeen || pcm.Length == 0)
                throw new InvalidDataException("WAV 缺少有效的 fmt 或非空 data chunk。");
            return pcm.ToArray();
        }
    }
}
'@
if (-not ('StreamTransWsProbe' -as [type])) { Add-Type -TypeDefinition $clientSource }

$temporaryWav = $null
$socket = $null
$cancellation = $null
$senderTask = $null
$writer = $null
$message = $null
$done = $false
try {
    $inputPath = (Resolve-Path -LiteralPath $Wav).Path
    if ($Ffmpeg) {
        $ffmpegCommand = Get-Command $Ffmpeg -CommandType Application -ErrorAction Stop
        $temporaryWav = Join-Path ([IO.Path]::GetTempPath()) ('streamtrans-' + [Guid]::NewGuid().ToString('N') + '.wav')
        & $ffmpegCommand.Source -nostdin -hide_banner -loglevel error -y -i $inputPath -vn -ar 16000 -ac 1 -c:a pcm_s16le -f wav $temporaryWav
        if ($LASTEXITCODE -ne 0) { throw 'FFmpeg 音频转换失败，请检查输入文件及 -Ffmpeg 路径。' }
        $inputPath = $temporaryWav
    }
    $pcm = [StreamTransWsProbe]::ReadPcmWav($inputPath)
    if ($OutputPath) {
        $outputFile = [IO.Path]::GetFullPath($OutputPath)
        $directory = Split-Path $outputFile -Parent
        if (-not (Test-Path -LiteralPath $directory)) { New-Item -ItemType Directory -Path $directory -Force | Out-Null }
        $writer = New-Object IO.StreamWriter($outputFile, $false, (New-Object Text.UTF8Encoding($false)))
        $writer.AutoFlush = $true
    }
    $probe = New-Object StreamTransWsProbe
    $socket = New-Object Net.WebSockets.ClientWebSocket
    if ($hasRequestKey) { $socket.Options.SetRequestHeader('Authorization', 'Bearer ' + $RequestKey) }
    $cancellation = New-Object Threading.CancellationTokenSource
    $cancellation.CancelAfter($TimeoutSeconds * 1000)
    $clock = [Diagnostics.Stopwatch]::StartNew()
    [void]$socket.ConnectAsync($Uri, $cancellation.Token).GetAwaiter().GetResult()
    $buffer = New-Object byte[] 8192
    $receiveBuffer = New-Object 'System.ArraySegment[byte]' (,$buffer)
    $message = New-Object IO.MemoryStream
    $receiveTask = $socket.ReceiveAsync($receiveBuffer, $cancellation.Token)
    $ready = $false
    $mode = $null
    $firstTranscriptBeforeFinish = $null
    $firstTranslationBeforeFinish = $null

    while (-not $done) {
        if ($cancellation.IsCancellationRequested) { throw "流式测试超过 $TimeoutSeconds 秒，请检查服务状态或增大 -TimeoutSeconds。" }
        if ($senderTask -and ($senderTask.IsFaulted -or $senderTask.IsCanceled)) { [void]$senderTask.GetAwaiter().GetResult() }
        if (-not $receiveTask.IsCompleted) { Start-Sleep -Milliseconds 10; continue }
        $result = $receiveTask.GetAwaiter().GetResult()
        if ($result.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close) { throw '服务在 done 事件之前关闭连接，请检查服务日志。' }
        if ($result.MessageType -ne [Net.WebSockets.WebSocketMessageType]::Text) { throw '服务返回了非文本消息，无法解析事件。' }
        if ($message.Length + $result.Count -gt 1048576) { throw '服务事件超过 1 MiB，请检查响应内容。' }
        $message.Write($buffer, 0, $result.Count)
        if ($result.EndOfMessage) {
            $raw = [Text.Encoding]::UTF8.GetString($message.ToArray())
            $message.SetLength(0)
            $event = $raw | ConvertFrom-Json
            if (-not $event.PSObject.Properties['type']) { throw '服务事件缺少 type 字段，请检查接口协议。' }
            if ($hasRequestKey -and $event.type -eq 'error') {
                foreach ($field in @('message', 'error')) {
                    $property = $event.PSObject.Properties[$field]
                    if ($property -and $property.Value -is [string]) { $property.Value = $property.Value.Replace($RequestKey, '[redacted]') }
                }
                $raw = $event | ConvertTo-Json -Depth 20 -Compress
            }
            $finishSent = $probe.FinishSent
            $record = [ordered]@{ elapsedMs = $clock.ElapsedMilliseconds; sentAudioMs = $probe.SentBytes / 32.0; finishSent = $finishSent; event = $event }
            $line = $record | ConvertTo-Json -Depth 20 -Compress
            if ($writer) { $writer.WriteLine($line) }
            Write-Output $line
            switch ($event.type) {
                'ready' {
                    if ($ready) { throw '服务重复发送 ready，请检查接口协议。' }
                    $ready = $true
                    $mode = $event.mode
                    $senderTask = $probe.SendAudioAsync($socket, $pcm, $ChunkMs * 32, ($Realtime -and -not $SendAsFastAsPossible), $cancellation.Token)
                }
                'transcript' {
                    if ($null -eq $firstTranscriptBeforeFinish) { $firstTranscriptBeforeFinish = -not $finishSent }
                }
                'translation' {
                    if ($null -eq $firstTranslationBeforeFinish) { $firstTranslationBeforeFinish = -not $finishSent }
                }
                'done' {
                    if (-not $ready -or -not $finishSent) { throw '服务在发送 finish 之前返回 done，请检查接口协议。' }
                    [void]$senderTask.GetAwaiter().GetResult()
                    $done = $true
                }
                'error' { throw "服务返回 error：$raw。请检查服务日志。" }
                default { throw "未知服务事件类型：$($event.type)。请检查接口协议。" }
            }
            if (-not $ready) { throw '首个服务事件必须是 ready。' }
        }
        if (-not $done) { $receiveTask = $socket.ReceiveAsync($receiveBuffer, $cancellation.Token) }
    }
    if ($AssertBeforeFinish -and ($firstTranscriptBeforeFinish -ne $true -or ($mode -eq 'translate' -and $firstTranslationBeforeFinish -ne $true))) {
        throw '提前返回验证失败：首个 transcript 必须在发送 finish 之前到达；翻译模式还需首个 translation。请使用包含句间停顿且足够长的音频，并保持实时发送。'
    }
} catch {
    $failure = $_.Exception.GetBaseException().Message
    if ($hasRequestKey) { $failure = $failure.Replace($RequestKey, '[redacted]') }
    throw "流式测试失败：$failure"
} finally {
    if ($cancellation) { $cancellation.Cancel() }
    if ($socket) {
        if ($done -and $socket.State -eq [Net.WebSockets.WebSocketState]::Open) {
            $closeTimeout = New-Object Threading.CancellationTokenSource
            try {
                $closeTimeout.CancelAfter(2000)
                [void]$socket.CloseOutputAsync([Net.WebSockets.WebSocketCloseStatus]::NormalClosure, 'done', $closeTimeout.Token).GetAwaiter().GetResult()
            } catch { $socket.Abort() } finally { $closeTimeout.Dispose() }
        } else { $socket.Abort() }
        $socket.Dispose()
    }
    if ($senderTask) { try { [void]$senderTask.GetAwaiter().GetResult() } catch {} }
    if ($message) { $message.Dispose() }
    if ($cancellation) { $cancellation.Dispose() }
    if ($writer) { $writer.Dispose() }
    if ($temporaryWav -and (Test-Path -LiteralPath $temporaryWav)) { Remove-Item -LiteralPath $temporaryWav -Force }
}
