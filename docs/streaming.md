# 实时音频识别与翻译

客户端连续发送录音采样即可，无需运行 VAD 或自行断句。服务端将音频归并为 100 ms 数据块，由独立的 CPU FSMN-VAD 持续检测语音边界；闭合的语音段进入待推理队列，由 GPU 依次识别和翻译。仅识别模式返回原文；翻译模式随后翻译该段原文。段级结果可在录音结束前返回。

当前服务端使用 Fun-ASR-Nano-2512 BF16 识别、Hy-MT2-1.8B Q6_K 翻译，二者由随包原生 CUDA 运行时执行；无需 Python。客户端音频格式、消息协议和逐段返回顺序不随翻译模型更换。

需要直接使用麦克风时，可使用 [Rust 桌面客户端](client.md)。客户端负责采集和重采样，也提供本机网页输出及可编辑 CSS。

## 建立连接

| 模式 | WebSocket 地址 | 结果 |
| --- | --- | --- |
| 仅识别 | `ws://127.0.0.1:3000/v1/audio/transcriptions/stream?source_language=auto` | `transcript`，不调用翻译推理 |
| 识别并翻译 | `ws://127.0.0.1:3000/v1/audio/translations/stream?source_language=auto&target_language=ja` | 每段先 `transcript`，再 `translation` |

| 查询参数 | 必填 | 说明 |
| --- | --- | --- |
| `target_language` | 翻译接口必填 | 与文件上传接口相同的目标语言代码或名称，例如 `ja`、`en`、`French`；最多 64 个字符；仅识别接口不接受此参数 |
| `source_language` | 否 | `auto`（默认）、`zh`、`en`、`ja`；作为 ASR 识别提示 |

语言名称按 URL 查询参数编码。设置了请求 key 的服务要求在握手中携带单个 `Authorization: Bearer <key>` 请求头；缺失、错误或重复认证头在升级前返回 HTTP `401 unauthorized`，并带有 `WWW-Authenticate: Bearer`。认证通过后，参数错误返回 HTTP `400`；其他上传或实时会话占用处理许可时返回 HTTP `429 busy`。两个 WebSocket 接口与两个文件上传接口共用一个处理许可，同一时刻只处理一个会话。连接建立后，服务端首先发送 `ready`。

服务端通过 `--request-key your-key` 或 `STREAMTRANS_REQUEST_KEY` 设置 key，默认仍监听 `127.0.0.1:3000`。设置 key 后可显式用 `--bind 0.0.0.0:3000` 或 `--bind [::]:3000` 供其他机器连接；未设置 key 时只允许回环监听。key 必须为 1–256 个可见 ASCII 字符，不能包含空白或为空；本机调用同样需要认证，不支持 URL 查询参数传 key。

认证头应在 WebSocket 握手时设置，而不是连接建立后发送消息。例如使用 .NET `ClientWebSocket` 时，在 `ConnectAsync` 前调用 `Options.SetRequestHeader('Authorization', 'Bearer ' + key)`。原生浏览器的 `WebSocket` API 无法直接设置此请求头，需要使用本项目 Rust 客户端或其他支持自定义握手头的客户端。跨不可信网络时使用 HTTPS/WSS 反向代理；请求 key 不提供传输加密。

仅识别跳过 LLM 推理，但当前服务启动时仍预加载 ASR 和 LLM，不减少模型常驻显存。客户端选择仅识别时应更换为 `transcriptions/stream` 路径，而不是向翻译接口传递空目标语言。

服务端若通过 `--glossary` 加载了[完整术语表](glossary.md)，每段翻译会共享固定术语前缀并复用 KV 缓存。客户端不需增加参数或反复上传词表；术语表不会修改 `transcript` 原文。

## 连续发送 PCM

Binary 消息承载 **16 kHz、单声道、PCM16LE** 原始采样，每个采样为 2 字节有符号小端整数。客户端先完成麦克风采集、单声道转换和必要的重采样，再连续发送；不需要检测静音或裁切语音段。WAV 输入需提取采样数据，MP3、WebM 等格式需先解码。

建议每个消息包含 20–200 ms 音频，即 640–6400 字节。每个消息必须非空、字节数为偶数，最大 32000 字节（1 秒）；WebSocket 帧和重组后的消息都受大小限制。服务端按采样顺序归并为 100 ms 数据块，较小的消息不会额外占用处理队列条目。

录音停止后，发送一条 Text 消息：

```json
{"type":"finish"}
```

`finish` 会提交最后不足 100 ms 的采样，并冲刷 VAD 中尚未结束的语音段。随后继续接收事件，直到 `done`；不要在发送 `finish` 后立即关闭连接。`finish` 只能发送一次，之后不能继续发送音频。文本控制消息最多 64 字节，仅接受这个命令。

## 接收逐段事件

服务端事件均为 JSON Text 消息。`segment_id` 从 1 开始递增；翻译模式下，同一段的 `transcript` 先于 `translation`。仅识别模式不发送 `translation` 事件。时间单位为毫秒，以音频流首个采样为起点，表示左闭右开的绝对语音区间，不是网络到达时间。

`ready` 声明音频格式和限制，以下数值以默认配置为例：

```json
{
  "type": "ready",
  "request_id": "05ec9760-ffb7-4cd2-9c19-1effa1f3b4c0",
  "mode": "translate",
  "source_language": "auto",
  "target_language": "日本語",
  "format": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "max_chunk_bytes": 32000,
  "max_audio_seconds": 0,
  "inactivity_timeout_seconds": 30,
  "max_buffered_audio_ms": 12800,
  "audio_buffer_stage": "network_to_vad",
  "max_pending_speech_ms": 120000,
  "max_pending_segments": 64
}
```

`ready.mode` 为 `translate` 或 `transcribe`；仅识别时，`target_language` 为 `null`。其他音频格式和限制字段在两种模式中相同。

`ready.max_audio_seconds` 表示实时会话的总音频时长上限，`0` 表示不限总时长，由服务参数 `--max-stream-seconds` 设置。它与文件上传接口默认 60 秒的限制独立。

`audio_buffer_stage` 为 `network_to_vad`，`max_buffered_audio_ms` 表示网络接收到 VAD 之间的原始音频缓冲。`max_pending_speech_ms` 和 `max_pending_segments` 分别限制等待识别、翻译的语音采样时长与段数，包含在途段。限制针对已闭合的语音段，其中包含段内停顿；未成段的纯静音不进入待推理队列。两级队列的用途和限制见下文。

语音段闭合后，先返回该段识别结果：

```json
{
  "type": "transcript",
  "segment_id": 1,
  "start_ms": 0,
  "end_ms": 4460,
  "text": "你好，这是一个本地语音识别和翻译测试。",
  "is_final": true,
  "end_reason": "silence",
  "elapsed_ms": {"asr": 523, "queue_wait": 0}
}
```

`end_reason` 为 `silence`（VAD 确认停顿）、`max_duration`（达到 15 秒分段上限）或 `final`（由 `finish` 冲刷）。VAD 检测语音活动边界，不保证语音段一定是完整语句。

`elapsed_ms.queue_wait` 为语音段进入待推理队列后、开始 ASR 前的等待时间；`elapsed_ms.asr` 仅计 ASR 调用耗时。等待时间持续增长表示处理正在落后于输入，不表示客户端需要等待识别结果后再发送音频。

翻译模式下，翻译完成后返回相同 `segment_id` 和时间区间：

```json
{
  "type": "translation",
  "segment_id": 1,
  "start_ms": 0,
  "end_ms": 4460,
  "text": "你好，这是一个本地语音识别和翻译测试。",
  "translation": "こんにちは、これは地元の音声認識と翻訳テストです。",
  "target_language": "日本語",
  "is_final": true,
  "elapsed_ms": {"translation": 2167}
}
```

全部音频处理完成后返回 `done`，随后关闭连接：

```json
{"type":"done","duration_ms":18243,"segments_count":4,"mode":"translate"}
```

`duration_ms` 是接收到的音频时长，包含静音；`segments_count` 是已处理语音段数量。仅识别模式的 `done.mode` 为 `transcribe`。纯静音可直接返回 `ready` 和 `done`，段数为 0，不调用 ASR 或 LLM。只有收到 `done` 才表示整条音频已处理完成，连接关闭本身不能作为成功标志。

## 缓冲、超时与错误

| 限制 | 行为 |
| --- | --- |
| 总音频时长 | `--max-stream-seconds` 默认 `0`，不限总时长；可设为 1–86400 秒，按采样数量计算；文件上传的 60 秒默认限制不适用于实时会话 |
| 单个语音段 | 最长 15 秒，连续讲话会自动切段 |
| 原始入站音频缓冲 | 128 个 100 ms 数据块，最多排队 12.8 秒音频；吸收网络突发和接收、VAD 处理间的短暂调度延迟，与客户端消息大小无关 |
| 待推理队列 | 已闭合语音段的 PCM 总计最多 120 秒、最多 64 段；两个上限同时生效，包含正在执行 ASR 或翻译的语音段 |
| 无音频超时 | 连续 30 秒没有收到有效 PCM 时结束连接；Ping/Pong 不延长该期限 |
| 输出缓冲 | 最多排队 8 个事件；发送事件或关闭连接最多等待 5 秒 |
| 模型推理 | 使用 `--inference-timeout-seconds`，默认 300 秒 |

网络接收、CPU VAD 和 GPU 推理分别推进。ASR 或 LLM 翻译期间，VAD 仍可消费后续采样、确认段边界并提交语音段；GPU 工作仍按段顺序执行，不并发运行两个语音段。

原始入站缓冲满时返回 `audio_buffer_full`；待推理队列达到任一上限时返回 `processing_backlog_full`。两者都会结束会话，服务端不会丢弃音频后仍报告成功。客户端应按录音速率发送，避免把长文件一次性突发上传。实时录音也可能因 CPU 调度异常或处理速度长期低于输入速度而积压，不能将所有缓冲错误都归因于讲话太快。

不限总时长不等于缓存整条录音。服务端保留有界的原始入站缓冲、VAD 所需的约 17 秒 PCM 窗口，以及有界的待推理语音段；已处理的采样会释放。如果 GPU 长期处理不过来，扩大队列只能推迟报错，并会增加结果延迟。

Hy-MT2 按翻译模板直接输出译文，不提供思考模式开关。服务端默认采样为 `temperature=0.7`、`top_p=0.6`、`top_k=20`、重复惩罚 `1.05`、`seed=42`，默认上下文 4096 token、输出预算 2048 token。具体耗时与完整性应按语言和输入检查，当前模型复测见 [Hy-MT2 测试记录](hy-mt2-20261103.md)。

升级后的错误事件采用以下结构，发出后连接结束：

```json
{"type":"error","code":"processing_backlog_full","message":"识别/翻译待处理语音积压已达上限（最多 120 秒或 64 段，含在途段），会话已停止；未完成音频不能视为识别完成，请查看服务日志后重试"}
```

| 错误码 | 处理方式 |
| --- | --- |
| `invalid_request` | 修正控制消息，避免重复 `finish` 或在 `finish` 后继续发送音频 |
| `invalid_audio` | 检查 PCM 字节序、非空和偶数字节要求 |
| `audio_chunk_too_large` | 将消息缩小至 32000 字节以内 |
| `audio_too_long` | 缩短会话，或调整服务端 `--max-stream-seconds` |
| `audio_buffer_full` | 检查音频采样时长与发送墙钟时间是否匹配，排查突发上传、网络积压后集中发送或 CPU VAD 未及时消费；修正后重新连接 |
| `processing_backlog_full` | ASR 或翻译未能及时处理语音段，已达到待推理队列的时长或段数上限；检查服务端每段耗时和 GPU 负载，等待当前 GPU 调用结束后重新连接 |
| `audio_timeout` | 在有音频数据时重新连接 |
| `stream_failed` | 查看完整错误原因；LLM 超时或通信失败后需重启服务 |

排查缓冲问题时，先查看错误对应的服务日志 `stage`。`network_to_vad` 记录 `received_audio_ms`、`elapsed_ms` 和 `input_queued_frames`，用于核对采样时长与实际经过时间；`vad` 记录 VAD 已消费的采样和待推理积压；`pending_speech` 记录达到队列上限时的 `pending_speech_ms`、`pending_segments`。ASR 日志和事件中的 `queue_wait_ms`、`elapsed_ms.queue_wait` 可帮助确认处理等待是否持续增长。仅凭一句话的时长或词数不能确定缓冲错误的原因。

超过 WebSocket 帧或消息上限等传输错误可能直接关闭连接，不能保证再收到 JSON 错误事件。ASR 或翻译失败不会撤回已经返回的 `transcript`；客户端可保留这些原文。

主动断开属于取消，不会隐式冲刷尾段。服务端停止提交尚未开始的后续工作；已经开始的 GPU 调用结束前，仍持有处理许可。因此刚断开后重新连接或上传文件，可能暂时收到 `429`。

## 使用测试客户端

[scripts/test-stream.ps1](../scripts/test-stream.ps1) 使用 .NET WebSocket 并行发送音频、接收事件，不依赖 Python。脚本读取文件并按实时速率回放，模拟持续录音客户端；文件需为 16 kHz、单声道、PCM16LE WAV。

服务端设置了 key 时，脚本可通过 `-RequestKey` 指定，或从 `STREAMTRANS_REQUEST_KEY` 读取。脚本将 key 写入握手请求头，不写入 URL 或 JSONL 结果。例如连接另一台机器的服务：

```powershell
$env:STREAMTRANS_REQUEST_KEY = 'your-key'
powershell -ExecutionPolicy Bypass -File scripts/test-stream.ps1 `
  -Wav .\sample.wav `
  -Uri 'ws://192.168.1.10:3000/v1/audio/translations/stream?source_language=auto&target_language=ja'

# 或显式指定：在上述命令中添加 -RequestKey your-key
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test-stream.ps1 `
  -Wav .\sample.wav `
  -Uri 'ws://127.0.0.1:3000/v1/audio/translations/stream?source_language=auto&target_language=ja' `
  -ChunkMs 20 -AssertBeforeFinish -OutputPath .\stream.jsonl
```

`-AssertBeforeFinish` 要求首个原文和译文都在开始发送 `finish` 前到达，适合包含句间停顿且长度足够的音频。短音频或讲话一直持续到录音结束时，结果在 `finish` 后返回也属于正常行为。

仅识别时更换连接地址，不传 `target_language`，也不要加要求译文的 `-AssertBeforeFinish`：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test-stream.ps1 `
  -Wav .\sample.wav `
  -Uri 'ws://127.0.0.1:3000/v1/audio/transcriptions/stream?source_language=auto' `
  -ChunkMs 20 -OutputPath .\transcription.jsonl
```

其他格式可使用 `-Ffmpeg` 先转换，例如：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test-stream.ps1 `
  -Wav .\sample.mp3 `
  -Ffmpeg .\dist\StreamTrans\runtime\ffmpeg\ffmpeg.exe `
  -ChunkMs 100 -OutputPath .\stream.jsonl
```

脚本默认按实时速率发送。`-SendAsFastAsPossible` 可验证短文件快速提交及 `finish`，较长音频快速发送可能触发缓冲上限。JSONL 每行包含 `elapsedMs`、`sentAudioMs`、`finishSent` 和服务端 `event`，可据此核实结果是否在录音结束前返回。

## 验证记录

2026-11-07，请求 key 回归通过全工作区 120 项测试，默认跳过 5 项需要本地模型的测试；Clippy `-D warnings`、格式检查及服务端和客户端发布构建通过。认证回归覆盖 HTTP 路由校验、未授权请求不读取音频或占用推理许可、真实 TCP 上两种 WebSocket 模式的拒绝与成功握手，以及客户端收到 `401` 时不发送音频、不泄漏响应中的 key。原生程序还验证了环境变量 key、无 key 的非回环配置拒绝、非法 key 不回显，以及帮助信息隐藏环境变量值。本轮未进行跨机器真实录音或 GPU 推理回归。

以下具体回放、耗时和译文记录来自 2026-10-31 至 2026-11-01 的 MiniCPM5-2B 版本，保留为历史接口回归，不代表当前 Hy-MT2 的性能或翻译质量。新模型结果单独记录于 [Hy-MT2 复测](hy-mt2-20261103.md)。

### 2026-11-01 持续输入回归（历史）

Windows、RTX 3050（6 GB）上，使用独立 VAD 流水线和默认空思考翻译配置，将 60.037 秒密集短句按 1× 实时速率、100 ms 数据块发送。收到 22 个 `transcript`、22 个 `translation` 和 `done`，没有错误；全部 `elapsed_ms.queue_wait` 为 0 ms。`done.duration_ms` 为 60037，与输入准确时长一致，包含最后不足 100 ms 的 37 ms PCM 尾块；接收到 `done` 的墙钟时间为 60.589 秒。

同一环境还完成了以下回放：

- 35 秒连续语音按 1×、100 ms 数据块发送，仅识别。返回 3 段原文，区间为 0–15 秒、15–30 秒、30–34.98 秒，`end_reason` 依次为 `max_duration`、`max_duration`、`final`。无译文、无错误，排队等待均为 0 ms；`done.duration_ms` 为 35000，墙钟时间为 35.641 秒。
- 2.23 秒快速多词语音加 2 秒静音，按 1×、20 ms 数据块发送并翻译为日语。返回 1 段原文和译文，分别在 3.527 秒、3.805 秒收到，均早于发送 `finish`；翻译调用耗时为 275 ms。`done.duration_ms` 为 4230，墙钟时间为 4.350 秒。
- 原有文件上传翻译接口 `POST /v1/audio/translations` 在新默认配置下返回 `200`。

这些测试覆盖密集分段、15 秒强制分段、持续输入和尾块提交。它们同时使用新的队列调度和空思考翻译配置，不能将结果单独归因于某一项改动，也不能据此保证所有语种或输入都不会积压。原始记录位于开发工作区 `.tools/benchmark/`，分别为 `stream-after-dense60s-20261101.jsonl`、`stream-after-continuous35s-20261101.jsonl`、`stream-after-fast2s-ja-20261101.jsonl` 和 `http-after-en-20261101.json`；输入采样数、剪辑参数和 SHA-256 位于 `.tools/benchmark/continuous-audio/receipt.json`。

### 修复前的密集短句回放（历史）

旧版将 VAD、ASR 和翻译串行放在同一消费循环中。60.037 秒密集短句按 1× 实时速率发送时，在约 37.420 秒收到 `audio_buffer_full`；该样本中每段 ASR 约 110 ms、翻译约 3.81 秒，处理落后持续累积。这个复现用于验证旧版调度问题，不代表新版的处理能力。原始记录为 `.tools/benchmark/stream-before-dense60s-20261101.jsonl`。

另一个样本将首句加速为 2.23 秒，再附加 2 秒静音，旧版可以正常完成英语翻译。它没有复现「短句词多就立即缓冲满」；实际问题仍需结合接收采样时长、墙钟时间和错误码判断。该样本与新版日语翻译的目标语言不同，不作耗时对比。原始记录为 `.tools/benchmark/stream-before-fast2s-20261101.jsonl`。

### 早期短音频与生命周期回归（历史）

以下记录来自独立 VAD 队列和默认空思考前缀调整前的版本。Windows、RTX 3050（6 GB）上，使用发行二进制、项目外工作目录和仅含 `System32` 的 `PATH`，两个模型均运行在 GPU：

- 18.243 秒中文按 20 ms 连续发送，客户端不运行 VAD。共收到 4 段原文、4 段日语翻译和 `done`。首段原文在 5.739 秒、译文在 7.900 秒收到，当时分别只发送了约 5.640 秒、7.800 秒音频，`finishSent` 均为 `false`。第二、第三段也在 `finish` 前返回，第四段由 `finish` 冲刷，`end_reason` 为 `final`。
- 9.121 秒中文快速发送并翻译为英语，收到 2 段原文、2 段译文和 `done`；最后一段在 `finish` 后冲刷，总时长为 9121 ms。
- 2 秒纯静音只返回 `ready` 和 `done`，`duration_ms` 为 2000，`segments_count` 为 0。
- 实时连接占用处理许可期间，文件上传返回 `429`；空闲连接断开后许可释放，无需重启服务。现有文件上传接口随后完成真实请求并返回 `200`。
- 将推理超时设为 1 秒后，先返回已识别的 `transcript`，再返回包含完整原因和重启提示的 `error`。返回前托管 LLM 子进程已退出，`GET /health` 返回 `503`；原始记录为 `streaming-timeout.jsonl` 和 `streaming-timeout-receipt.json`。

这些记录验证实时输入、段级返回和生命周期，不用于和全文一次性翻译比较吞吐。原始材料位于开发工作区 `.tools/benchmark/streaming-ja-20ms.jsonl`、`streaming-en-fast.jsonl` 及对应兼容性、处理许可记录。当前接口提供段级最终结果，不提供 ASR 逐 token 中间假设或跨段模型 KV 增量。
