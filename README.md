# StreamTrans

本地 Rust 服务：音频 → FSMN-VAD 断句 → Fun-ASR-Nano-2512 识别 → 可选 Hy-MT2-1.8B 翻译。支持文件上传和 WebSocket 连续音频输入，返回原文、可选译文及语音段时间范围。另提供 Rust 桌面录音客户端，内含网页输出和可编辑 CSS；客户端无需运行 VAD，服务端负责归并音频和断句。

项目不依赖 Python 或 PyTorch 运行环境。ASR 使用 Rust 调用 transcribe.cpp 原生 CUDA 库，LLM 使用随程序分发的 llama.cpp CUDA 后端，FSMN-VAD 使用 Rust Candle 在 CPU 上运行。Rust 服务自动管理本机 `llama-server` 子进程，无须另外部署模型服务。FFmpeg 用于音频格式转换。

| 用途 | 模型 | 执行方式 |
| --- | --- | --- |
| 断句 | FSMN-VAD | Rust Candle，CPU FP32 |
| 语音识别 | Fun-ASR-Nano-2512 | transcribe.cpp CUDA；二维矩阵权重 BF16，音频激活 FP32 |
| 翻译 | Hy-MT2-1.8B Q6_K | llama.cpp CUDA；保留原始 GGUF 量化权重 |

**最终翻译模型选型（2026-11-03）：Hy-MT2-1.8B GGUF Q6_K。** 经过 12 个量化版本的中日韩英翻译对比，确认保留该模型作为生产默认。固定版本、选型依据和已知限制见 [模型说明](docs/models.md#翻译模型最终选型)。

ASR 和 LLM 必须运行在 GPU 上，不自动回退 CPU。Windows 分发包包含 ASR 原生库、llama.cpp、CUDA 运行库和 FFmpeg；运行机器仍需兼容的 NVIDIA 驱动。项目不内置模型权重，首次启动默认从 **Hugging Face** 下载三个固定版本，也可通过 `--model-source modelscope` 选择 **ModelScope**。模型由 Rust 下载、校验并转换 ASR 缓存，后续复用并支持离线启动。来源、精度与限制见 [模型说明](docs/models.md)，打包与部署见 [分发说明](docs/distribution.md)。

Windows 分发目录将一套共享的 CUDA 12.8 运行库和 FFmpeg 解码 DLL 放在 `runtime` 根目录，ASR 与 llama.cpp 各自的 ggml 保留在独立子目录。llama.cpp 固定为 b11960 源码，使用 CUDA Toolkit 12.8.1 本地编译，与 ASR 使用相同版本的 CUDA 运行库。文件上传在进程内解码，不启动 FFmpeg 子进程；实时录音直接传输 PCM，不经过 FFmpeg。

2026-11-07 的 Windows 实测发行目录缩小约 44.3%；9.12 秒 WAV 的独立规范化耗时中位数由 32.1 ms 降至 6.7 ms。构建、预热方法、音频完整性、模块来源和限制见 [共享运行库验证](docs/shared-runtime-20261107.md)。

## Windows 安装与启动

源码构建需要 Windows x64、Rust stable 和 Visual Studio 2022 C++ Build Tools（「使用 C++ 的桌面开发」组件）。构建 llama.cpp 前需安装 CMake 并将其加入 `PATH`；Ninja 可从 `PATH` 获取，也可复用 Visual Studio 随带的组件。构建脚本将固定的 CUDA 12.8.1 编译组件下载到项目内 `.tools/llama-build-cuda128/cuda`，无须全局安装 CUDA Toolkit，也不使用 Python。运行机器只需兼容随包 CUDA 运行库的 NVIDIA 驱动；GPU 需支持原生 BF16，显存需同时容纳 ASR、LLM 及推理缓存。

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\target\release\streamtrans.exe
```

安装脚本下载并校验固定版本的 ASR 原生库、llama.cpp 源码、CUDA 组件和 FFmpeg 9.0 LGPL 共享库，构建 llama.cpp 和小型音频桥接 DLL，再编译 Rust 发布版本。FFmpeg 头文件和导入库只用于构建，不进入分发包；不使用 Python、pkg-config 或 bindgen。首次安装需要联网访问上游下载站点和 crates.io；模型在服务首次启动时下载。开发模式从项目根目录启动，并保留 `.tools` 中的运行库和 `models` 缓存。

Windows 安装通过 `scripts/setup-cuda-runtime.ps1` 为两个后端准备相同版本的 CUDA 运行 DLL，复用下载缓存。开发目录各自保留 DLL 副本，制作分发目录时仅携带一套共享副本。CUDA Toolkit 和 NVCC 只用于构建机器，不进入分发目录。

单独构建 llama.cpp 可运行 `scripts/setup-llama.ps1 -Jobs 4`，默认输出到 `.tools/llama-cuda128`；`-OutputDirectory` 可覆盖此目录。

重新构建或运行检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build.ps1
powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Action test
powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Action clippy
cargo fmt --check
```

指定 GPU 或 FFmpeg 路径：

```powershell
.\target\release\streamtrans.exe --gpu 0 --ffmpeg-library .tools/ffmpeg-shared/bin/streamtrans-ffmpeg.dll
```

`--llama-server` 和 `--asr-library` 可覆盖默认原生程序路径。发行模式自动查找可执行文件旁的 `runtime` 目录；部署时复制完整分发目录，无须在运行机器安装 Rust、C++ 编译器、CUDA Toolkit 或 Python：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dist.ps1
.\dist\StreamTrans\streamtrans.exe
```

FFmpeg DLL 默认由 `scripts/setup-ffmpeg.ps1` 准备；可通过 `scripts/dist.ps1 -FfmpegDirectory ...` 指定已经构建并带有来源清单的目录。分发包保留对应许可证和源码获取信息，完整参数见 [分发说明](docs/distribution.md)。Windows 的旧 `--ffmpeg` 和打包脚本的 `-Ffmpeg` EXE 参数改用 DLL 参数。

仅下载模型而不加载模型或启动服务：

```powershell
.\target\release\streamtrans.exe --download-only

# 显式选择下载源；同样适用于正常启动
.\target\release\streamtrans.exe --model-source huggingface --download-only
.\target\release\streamtrans.exe --model-source modelscope --download-only
```

`--model-source` 统一选择 ASR、VAD 和翻译模型的下载源，接受 `huggingface`（默认）或 `modelscope`；连接失败时不会自动改用另一来源。已有 ModelScope 缓存可使用 `--model-source modelscope` 继续复用。两个来源的仓库、固定 revision 和缓存清单分别管理，默认 Hugging Face 不会误用旧 ModelScope 清单。`--asr-revision`、`--vad-revision` 和 `--llm-revision` 可覆盖当前下载源的相应固定版本。

全部参数见 `.\target\release\streamtrans.exe --help`。开发模式默认缓存目录为当前工作目录下的 `models`；检测到发行布局时，默认缓存目录为可执行文件旁的 `models`，可用 `--cache-dir` 覆盖。模型清单按来源和模型规格独立生成，切换来源或模型版本时保留原有缓存。默认接口地址为 `127.0.0.1:3000`；设置请求 key 后，可显式绑定非回环地址供其他机器访问。

### 请求 key 与跨机器访问

服务端可通过 `--request-key` 或环境变量 `STREAMTRANS_REQUEST_KEY` 设置请求 key。未设置时，仅允许绑定回环地址；设置后，可用 `--bind 0.0.0.0:3000` 或 `--bind [::]:3000` 监听非回环地址。设置 key 不会自动修改默认监听地址。

```powershell
$env:STREAMTRANS_REQUEST_KEY = 'your-key'
.\target\release\streamtrans.exe --bind 0.0.0.0:3000

# 也可直接指定参数
.\target\release\streamtrans.exe --bind 0.0.0.0:3000 --request-key your-key
```

请求 key 必须为 1–256 个可见 ASCII 字符，不能包含空白或为空。设置后，所有 HTTP 请求（含 `GET /health`）及两条 WebSocket 握手都必须携带单个 `Authorization: Bearer <key>` 请求头，本机调用也不例外。不支持将 key 放入 URL 查询参数；缺失、错误或重复认证头返回 `401 unauthorized`，并带有 `WWW-Authenticate: Bearer`。

```powershell
curl.exe http://127.0.0.1:3000/health -H "Authorization: Bearer $env:STREAMTRANS_REQUEST_KEY"
curl.exe http://127.0.0.1:3000/v1/audio/translations -H "Authorization: Bearer $env:STREAMTRANS_REQUEST_KEY" -F "audio=@sample.wav" -F "target_language=en"
```

桌面客户端填写同一请求 key，或使用客户端的 `--request-key` / `STREAMTRANS_REQUEST_KEY`。请求 key 用于认证，不提供传输加密；跨不可信网络时通过 HTTPS/WSS 反向代理保护传输。

## Linux 安装与启动

Linux x86_64 流程尚未实机验证。先安装 Rust、C++ 编译器、兼容的 NVIDIA 驱动、FFmpeg、curl 和 tar，再运行：

```bash
bash scripts/setup.sh
LD_LIBRARY_PATH="$PWD/.tools/transcribe:$PWD/.tools/llama${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./target/release/streamtrans \
  --asr-library .tools/transcribe/libtranscribe.so \
  --llama-server .tools/llama/llama-server
```

脚本下载并校验 transcribe.cpp v0.4.1 Linux CUDA 库、llama.cpp b11960 Ubuntu x64 CUDA 12.8 程序及其 CUDA 运行库，然后构建 Rust 服务，不调用 Python 或 NVCC。llama.cpp 官方包面向 Ubuntu 24.04 构建，其他发行版需验证系统库兼容性。也可自行构建兼容的 CUDA 后端，通过 `--asr-library` 和 `--llama-server` 指定路径。

## 桌面录音客户端

启动服务端后，可使用独立的 Rust GUI 选择录音设备和目标语言，开始、停止录音，显示原文和译文，并复制或清空结果。目标语言选择「不翻译」时仅进行语音识别。客户端连续发送音频，由服务端断句。

客户端可单独构建和分发，无需 Python、Node.js、FFmpeg、CUDA 或模型，也不需要 NVIDIA GPU；模型运算仍在服务端执行。Windows 源码构建只需 Rust 和 Visual Studio C++ Build Tools：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-client.ps1
.\target\release\streamtrans-client.exe

# 打包到新的独立目录，不包含服务端、模型或 GPU 运行库
powershell -ExecutionPolicy Bypass -File scripts/dist-client.ps1
.\dist\StreamTransClient\streamtrans-client.exe
```

GUI 中填写 API 地址（默认 `http://127.0.0.1:3000`）；服务端设置了请求 key 时，填写同一「请求 key」。选择设备和语言后，点击「开始语音识别」或「开始自动翻译」。点击「停止录音」后等待末段处理完成。勾选「显示原文」「显示译文」会同步到客户端本机网页输出，默认地址为 `http://127.0.0.1:3001/`，可用于浏览器或 OBS 浏览器源。

网页每 500 ms 刷新状态，样式从可执行文件旁的 `output.css` 读取，保存修改后自动生效；已有 CSS 不会被模板覆盖。中文 GUI 字体嵌入可执行文件，字体许可证随客户端分发。构建、打包、网页路径和 CSS 说明见 [客户端说明](docs/client.md)。服务端可按上述请求 key 配置允许跨机器连接；客户端网页输出仍仅监听本机回环地址。

网页地址添加 `?mode=latest` 可只显示最后一个语音段，内容最后更新 5 秒后自动隐藏；例如 `http://127.0.0.1:3001/?mode=latest&timeout=8` 将保留时间设为 8 秒。此模式只影响网页显示，桌面历史保留。

## 音频接口

请求格式均为 `multipart/form-data`：

| 路径 | 模式 | 目标语言 |
| --- | --- | --- |
| `POST /v1/audio/transcriptions` | 仅识别 | 不接受 `target_language` |
| `POST /v1/audio/translations` | 识别并翻译 | 必须提供 `target_language` |

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `audio` | 文件 | 是 | WAV、MP3、FLAC、M4A 等 FFmpeg 可解码音频；最大 32 MiB |
| `target_language` | UTF-8 字符串 | 翻译接口必填 | 目标语言代码或名称，如 `en`、`zh-CN`、`ja`、`French`；最多 64 个字符；仅识别接口不接受此字段 |
| `source_language` | 字符串 | 否 | `auto`（默认）、`zh`、`en`、`ja`；这是识别提示，不是语言检测结果 |

音频先转换为 16 kHz 单声道 PCM WAV，再由 FSMN-VAD 给出语音段，单段时长上限为 15 秒。时间范围以转换后音频起点为基准，单位为毫秒。VAD 检测语音活动边界，不保证每段都是完整语句。ASR 逐段识别后合并原文，翻译模式再提交 LLM 翻译。默认最长 60 秒，可用 `--max-audio-seconds` 调整，上限为 600 秒。

Windows 使用 `curl.exe`，避免旧版 PowerShell 的 `curl` 别名：

```powershell
curl.exe http://127.0.0.1:3000/v1/audio/translations -F "audio=@sample.wav" -F "source_language=zh" -F "target_language=en"

# 仅识别，不调用翻译推理
curl.exe http://127.0.0.1:3000/v1/audio/transcriptions -F "audio=@sample.wav" -F "source_language=zh"
```

Linux：

```bash
curl http://127.0.0.1:3000/v1/audio/translations \
  -F 'audio=@sample.wav' \
  -F 'source_language=zh' \
  -F 'target_language=en'
```

成功响应示意，时间与分段以实际推理结果为准：

```json
{
  "request_id": "583b4bea-2e7c-4de1-a14c-9f0ac7c12fca",
  "mode": "translate",
  "text": "你好，世界。",
  "translation": "Hello, world.",
  "source_language": "zh",
  "target_language": "English",
  "duration_ms": 2500,
  "segments": [{ "start_ms": 200, "end_ms": 2200, "text": "你好，世界。" }],
  "elapsed_ms": { "asr": 300, "translation": 1000, "total": 1320 },
  "models": {
    "asr": "Fun-ASR-Nano-2512",
    "asr_precision": "BF16 matrix weights; FP32 audio activations",
    "vad": "FSMN-VAD",
    "vad_device": "cpu",
    "llm": "Hy-MT2-1.8B",
    "llm_backend": "llama.cpp",
    "llm_quantization": "Q6_K"
  }
}
```

仅识别响应保留同样的原文、分段、音频时长、模型和耗时字段，`mode` 为 `transcribe`，`translation` 和 `target_language` 均为 `null`，`elapsed_ms.translation` 为 `0`。当前服务端仍预加载 ASR 和 LLM，仅识别只跳过 LLM 推理，不释放其常驻显存。

`elapsed_ms.audio_decode` 单独报告文件解码、重采样及规范化写入耗时，已计入 `elapsed_ms.total`；不包含在 `asr` 或 `translation` 中。

静音或没有检测到语音时，返回 `200`，`text` 为空字符串，`segments` 为空数组，不调用 LLM。翻译模式的 `translation` 为空字符串，仅识别模式仍为 `null`。目标语言代码会规范化为语言名称；其他合法名称原样传给模型，译文质量取决于模型能力。

| HTTP 状态 | 错误码 | 处理方式 |
| --- | --- | --- |
| `400` | `invalid_request` | 补全字段、删除重复字段或修正参数 |
| `401` | `unauthorized` | 设置正确的单个 `Authorization: Bearer <key>` 请求头 |
| `413` | `audio_too_large` | 将文件缩小至 32 MiB 以内 |
| `422` | `invalid_audio`、`empty_audio`、`audio_too_long`、`audio_decode_timeout` | 更换有效音频、缩短音频或重新编码 |
| `429` | `busy` | 当前正在处理音频，稍后重试 |
| `502` | `asr_failed`、`translation_failed` | 查看日志、显存、超时设置和 token 预算 |
| `503`（`GET /health`） | `backend_unavailable` | 模型状态不可用，查看日志后重启服务 |

错误体为 `{"error":{"code":"...","message":"..."}}`。翻译失败时，响应额外包含 `recognition: {text, segments}`，保留已识别的原文。

Hy-MT2-1.8B Q6_K 基于官方翻译模板，追加逐句完整翻译、保留数字、单位、条件和否定关系的提醒，直接生成译文，不提供思考模式开关。此提示经过项目调整，并非逐字使用官方示例。默认采样为 `temperature=0.7`、`top_p=0.6`、`top_k=20`、重复惩罚 `1.05`、`seed=42`；默认 `--context-size 4096`、`--max-tokens 2048`。模型未正常结束时返回错误，不把达到长度上限的内容作为完整翻译。输入过长时，缩短音频或显式调整输出预算，为原文保留上下文空间；程序不自动截断输入或压缩输出预算，超限明确拒绝。完整性提醒和正常结束标记都不能保证事实全部保留，固定用例与速度对照见 [Hy-MT2 复测](docs/hy-mt2-20261103.md)。

2026-11-03 完成了显存上限 3.5 GB 的中日韩英模型对比，覆盖 12 个量化版本，保留原文、实际译文、逐项问题与冷热耗时。见 [完整对比报告](docs/translation-model-comparison-20261103.md) 和 [离线逐句对照页](docs/translation-model-comparison-20261103.html)。本轮最终确定保留 Hy-MT2-1.8B Q6_K；普通翻译的速度备选与完整术语场景分别评价。

### 加载完整术语表

服务端支持 UTF-8 TSV，三列依次为原词、目标语言、指定译法。分发目录中可运行：

```powershell
.\streamtrans.exe --glossary .\examples\glossary.tsv
```

完整表启动时加载一次，全部语言、全部条目按模型推荐的「原词翻译成指定译法」句式编码为同一条 user 消息的固定前缀，并使用完整中文语言标签；不是将原始 TSV 原样放入提示，也不按当前原文筛词。每次翻译明确当前目标语言，只编码变化后缀，复用 llama.cpp 的前缀 KV；客户端无需发送术语表。默认 `--glossary-max-tokens 1024` 检查规范化后的完整 TSV token 数，实际渲染前缀另查上下文容量，超限明确拒绝，不自动截断。上下文还需容纳原文和完整输出预算。

本机 llama-server API 仍接收完整拼接 token ID 序列，KV 缓存减少重复计算，不代表网络零重传或无限容量。未加载术语表时保持 `cache_prompt=false`。术语遵从不是硬约束：例如日语译文仍可能混入只为英语指定的 `on-device`，目标语言隔离仍有限制。格式、部署和实际缓存命中见 [术语表说明](docs/glossary.md)。

`--inference-timeout-seconds` 默认 300 秒。ASR 使用原生取消回调；LLM 通过托管子进程的本地接口执行翻译。LLM 请求超时或通信失败时，服务先停止并回收子进程，再返回错误，避免超时后继续占用 GPU；此时需要重启 StreamTrans。其他后端错误可通过 `GET /health` 检查状态。

`--startup-timeout-seconds` 限制启动等待时间，但 Rust 转换和原生模型加载不能被强制中断；发生超时或在加载期间退出时，进程仍会等待当前阻塞任务完成。

`GET /health` 检查原生模型状态。服务在所有模型加载完成并确认 ASR、LLM 使用 CUDA 后才开始监听。托管的 `llama-server` 仅监听 `127.0.0.1` 的临时端口，使用每次启动生成的 API 密钥，随服务退出清理。文件上传接口完成后统一返回 JSON，`segments` 包含各段原文，`translation` 对应合并后的全文。连续录音使用下述 WebSocket 接口；麦克风采集由客户端完成。

## 实时录音接口

| 模式 | WebSocket 地址 |
| --- | --- |
| 仅识别 | `ws://127.0.0.1:3000/v1/audio/transcriptions/stream?source_language=auto` |
| 识别并翻译 | `ws://127.0.0.1:3000/v1/audio/translations/stream?source_language=auto&target_language=ja` |

客户端连续发送 16 kHz、单声道、PCM16LE Binary 音频块即可，无需 VAD 或自行断句。建议每块 20–200 ms，单条消息最多 1 秒。服务端归并为 100 ms 数据块，保留 FSMN-VAD 状态，检测到段结束后返回 `transcript`；翻译模式再返回该段 `translation`。事件包含 `segment_id` 和绝对时间范围，`ready`、`done` 带有 `mode`。仅识别模式不接受目标语言，也不发送译文事件。

停止录音后发送 `{"type":"finish"}`，继续接收事件直到 `done`，尾段会在此时冲刷。会话与文件上传共用处理许可；CPU 收音和 VAD 独立于 GPU 推理，原始输入缓冲为 12.8 秒，已断句待处理语音最多 120 秒或 64 段（包含在途段），达到上限时明确返回错误。`--max-stream-seconds` 默认 `0`，支持持续录音而不限制总时长，也可显式设置上限；文件上传的默认 60 秒限制独立保留。服务端不积累整条录音，连续 30 秒没有音频数据仍会结束连接。

可用不依赖 Python 的 [PowerShell 测试客户端](scripts/test-stream.ps1) 按录音速率回放规范 WAV，`-Ffmpeg` 可先转换其他音频格式。协议、事件、错误和命令见 [实时接口说明](docs/streaming.md)。

2026-11-01 使用 MiniCPM5-2B 的历史回归完成了 60.037 秒密集短句的实时输入验证，得到 22 段原文和英文译文，无缓冲错误或逐句增长的推理等待；35 秒连续语音验证了 15 秒强制断句及尾段提交。这些记录用于验证队列调度，不能视为当前 Hy-MT2 的速度或翻译质量结果。旧版原地复现、翻译冷热对照和限制见 [历史延迟复测](docs/latency-20261101.md)。此前也已验证 20 ms 消息、短文件快速提交、纯静音和文件上传兼容性。

## 验证范围

当前 Hy-MT2-1.8B Q6_K 的固定语料完整性、术语遵从和速度对照见 [2026-11-03 复测](docs/hy-mt2-20261103.md)。下述具体 GPU 耗时、译文 token 数和录音回放记录来自此前 MiniCPM5-2B 版本，保留为历史，不代表新模型结果。

自动化回归使用 `cargo test --workspace --all-targets --locked`，并执行工作区 Clippy `-D warnings`；当前执行结果和用例计数见 [Hy-MT2 复测](docs/hy-mt2-20261103.md)。回归覆盖术语表完整加载与容量、缓存 token 计数、慢翻译期间持续 VAD、在途语音计入积压预算、断开等待当前 GPU 调用，以及推理失败或线程异常时唤醒收音任务。

服务端测试覆盖模型下载与缓存、格式转换、FSMN-VAD、LLM 输出解析和 HTTP/WebSocket 参数与错误处理；另已使用本机权重单独运行 FSMN-VAD 的两项真实模型测试。仅识别接口使用可控后端验证不调用翻译推理。

WebSocket 测试覆盖真实握手、PCM 归并、尾段提交、录音结束前返回、缓冲上限，以及断开后等待在途任务结束再释放处理许可；接口测试使用可控模型替身，真实 GPU 验证另行执行。

客户端已在 Windows、RTX 3050（6 GB）上回放 9.121 秒中文 WAV，使用真实服务完成仅识别和英语翻译：仅识别得到 2 段中文，译文为 `null`，LLM 推理调用数为 0；翻译得到 2 段中英结果并按相同段 ID 合并显示。新增文件识别接口返回 `200`、翻译耗时为 `0`，误传目标语言返回 `400`；原文件翻译接口仍返回 `200`。

原生 GUI 已启动检查，设备列表成功枚举网易虚拟麦克风；Edge 网页在 320 × 720 视口下无横向溢出，CSS 字号改为 31 px 及恢复均自动生效。**尚未实际打开麦克风或录制音频**，WAV 回放不替代真实设备采集验证。客户端细节见 [客户端说明](docs/client.md#验证范围)。

以下保留 2026-10-31 MiniCPM5-2B 完整思考配置的分发验证结果；该模型后续空思考配置的结果见 [历史延迟复测](docs/latency-20261101.md)。Windows 分发目录当时已在 RTX 3050（6 GB）上完成真实 GPU 验证：从项目外的临时目录启动，`PATH` 仅保留 `System32`，程序自动找到随包运行库；测试仅显式指定已有模型缓存和测试监听地址。复用缓存时，从模型校验到接口开始监听耗时 5.429 秒，不包含首次模型下载和 ASR 转换。

以下为同一服务进程中的两次单次请求，均返回 `200`。表格使用响应中的处理耗时，不作中位数统计，也不包含启动时间：

| 音频与目标语言 | 请求顺序 | VAD+ASR | 翻译 | 处理总耗时 |
| --- | --- | ---: | ---: | ---: |
| 18.243 秒中文，auto → 日语 | 首请求 | 979 ms | 3916 ms | 4939 ms |
| 9.121 秒中文，auto → 英语 | 随后请求 | 345 ms | 1056 ms | 1438 ms |

日语生成 267 个思考 token、1 个分隔符和 51 个译文 token；英语分别为 58、1 和 25，EOS 均另计 1 个。进程模块检查确认 ASR 与 LLM 分别使用各自分发目录中的 ggml 和 CUDA 运行库。正常 `Ctrl+C` 后，主进程和托管子进程均已退出。

另以 1 秒推理超时验证：接口返回 `502 translation_failed` 并保留已识别原文，llama.cpp 子进程已退出，健康检查返回 `503`。该行为符合超时后停止后端、需要重启服务的约定。完整分发验证记录见 [分发说明](docs/distribution.md#验证记录)。

完整思考配置下曾完成 Windows 独立 LLM GPU 测试。对同一条 99 个输入 token 的日语翻译，子进程启动、模型加载及内置预热合计 1.134 秒，单独计时；首请求为 4.009 秒，随后 3 次请求的中位数为 4.016 秒，解码吞吐为 80.405 token/s。配置为 FP16 KV、FlashAttention 自动选择；实际生成 267 个思考 token、1 个分隔符及 51 个译文 token，EOS 另计 1 个。该测试不包含 ASR 和音频接口流程。

此前 MiniCPM5-2B 的 Candle 与 llama.cpp 采样和性能对照保留在 [历史复测记录](docs/sampling-benchmark.md)。这些结果来自旧模型的后端实验，不代表当前 Hy-MT2 的 HTTP 性能。Linux 安装流程和数小时持续录音未实测；其他硬件的效果、显存需求与速度需要单独验证。
