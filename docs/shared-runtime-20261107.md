# CUDA 共享运行库与 FFmpeg DLL 验证（2026-11-07）

Windows 发行包已统一为 CUDA 12.8，并以进程内 FFmpeg DLL 替代文件上传时的 FFmpeg 子进程。模型仍为 Fun-ASR-Nano-2512 BF16、FSMN-VAD 和 Hy-MT2-1.8B Q6_K，未修改采样参数或模型权重。

## 发行包与构建

| 项目 | 旧包 | 新包 |
| --- | --- | --- |
| 目录 | `dist/StreamTrans` | `dist/StreamTrans-shared-cuda` |
| 文件总大小（非 ZIP） | 约 2.215 GiB | 约 1.234 GiB |
| CUDA | ASR 12.8、llama.cpp 12.4 各一套 | 共享一套 12.8 |
| llama.cpp CUDA DLL | 594,121,728 字节 | 173,513,216 字节 |
| FFmpeg | 静态 `ffmpeg.exe` | 四个 FFmpeg DLL 与一个音频桥 |
| 模型 | 不包含 | 不包含 |

总量减少约 **44.3%（约 1005 MiB）**。新包额外包含五个 FFmpeg DLL，合计 115,490,816 字节；相比旧 EXE 增加约 9.60 MiB。本次采用现成共享构建，音频专用 FFmpeg 编译不在本次范围内。最终逐文件大小和 SHA256 以包内 `manifest.json` 为准。

llama.cpp 固定 [b11960 源码提交](https://github.com/ggml-org/llama.cpp/tree/8642541b255bfb5dc39cd351a178be09013a55d6)，使用 NVCC 12.8.93、MSVC 19.44.35229、CMake 4.4.5、Ninja 1.12.1 构建；保留通用 CPU 后端与多 GPU 架构，启用 CUDA `size` 压缩。首次编译约 51 分钟，包含从 4 并发受控续编至 8 并发；这不是模型加载或推理时间。源码、工具组件和文件校验值记录在构建清单及 `THIRD_PARTY.json`。

共享 CUDA 取自与 ASR 一致的 [NVIDIA CUDA 12.8.1 组件清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json)：CUDART 12.8.90、cuBLAS 12.8.4.1。ASR 与 llama.cpp 的 ggml ABI 不兼容，继续分别保存在 `runtime/asr` 和 `runtime/llama`。

FFmpeg 固定为 [BtbN 2026-10-31 月末共享构建](https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2026-10-31-12-59)，版本 `n9.0.2-39-g7adfbe40dc`、LGPLv3，归档 SHA256 为 `e907b01fa4783913151237e37df2a10fa97d99a1598a021bf59d4ae62ffed4c0`。使用月末保留版而非短期日构建；同包头文件和导入库只用于编译音频桥，发行包只携带实际 DLL、许可和源码获取说明。

## 文件解码的独立速度对照

机器为 Ryzen 7 5700G、RTX 3050 6 GB，驱动 621.14。编译任务全部结束后，单独运行 CPU 音频对照工具，不加载 GPU 模型或 HTTP 服务。输入为工作区 `zh16.wav`：145,951 个采样，16 kHz、单声道 PCM16，约 9.122 秒。两端均完成解码、重采样及 PCM16 WAV 写入。

各阶段首轮和一次额外预热单独记录，再统计后续 10 轮；首轮不代表清空操作系统文件缓存。

| 阶段 | 首轮 | 额外预热 | 后续 10 轮中位数 |
| --- | ---: | ---: | ---: |
| CLI `-version` | 25.24 ms | 19.58 ms | 19.87 ms |
| CLI 完整规范化 | 36.66 ms | 33.81 ms | 32.09 ms |
| DLL 完整规范化 | 7.16 ms | 6.94 ms | 6.70 ms |

DLL 首次加载另计 **9.79 ms**，服务启动时加载并保留，后续文件请求复用。完整规范化耗时减少约 79.1%。CLI 与 DLL 来自不同的 FFmpeg 构建；上述差额是实际处理链路差额，不能全部归因为纯进程创建成本。`-version` 包含程序初始化和版本输出，也不是单独的 `CreateProcess` 测量。

可复测命令：

```powershell
cargo build --release --locked --example bench_audio
.\target\release\examples\bench_audio.exe `
  --input .tools/smoke/zh16.wav `
  --ffmpeg .tools/ffmpeg/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe `
  --ffmpeg-library .tools/ffmpeg-shared/bin/streamtrans-ffmpeg.dll `
  --warm-runs 10 --output .tools/benchmark/audio-formal.json
```

这里的 EXE 仅为开发测速对照，不是新 Windows 服务或发行包的运行依赖。正式数据为 `.tools/benchmark/audio-formal-20261107.json`；编译过程中生成的单轮格式报告只用于正确性检查，不作为速度结论。

## 音频完整性

8 种文件输入完成 CLI/DLL 逐采样对照，并通过新发行包的真实 GPU ASR 接口：中文文件名 PCM WAV、44.1 kHz 立体声 WAV、48 kHz 浮点立体声 WAV、MP3、FLAC、M4A/AAC、Ogg/Vorbis、Opus。

所有规范化结果均为 145,951 个采样；整段波形相对旧 CLI 最多相差 S16 的 1 个最小单位，最后 160 个采样完全一致。WAV 原始输入与 FLAC 整段误差为 0。8 个真实 ASR 请求均返回两段相同原文：

> 你好，这是一个本地语音识别和翻译测试。 今天下午三点开会。

实现显式使用与原 S16 输出一致的混音幅度限制，结束时分别冲刷解码器与重采样器，避免丢失尾部。单元测试还覆盖中文路径、DLL 缓存复用、立体声幅度、精确采样上限、超限、超时及错误输入。实际坏文件请求返回 `422 invalid_audio`；仅识别返回 `translation=null`、翻译耗时为 0。

## GPU、接口与退出检查

新包从项目外目录启动，服务进程 `PATH` 仅包含 `System32`，只显式指定 ModelScope 现有模型缓存、回环监听地址和测试请求 key。未重新下载模型或转换 ASR。缓存校验到监听约 5.57 秒，ASR 使用 CUDA0，LLM 的 33/33 层全量加载 GPU。

实际进程模块检查确认：

- ASR 与 llama-server 均从同一 `runtime` 根目录加载 `cudart64_12.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`，路径和 SHA256 一致。
- ASR 的 ggml 来自 `runtime/asr`，LLM 的 ggml 来自 `runtime/llama`。
- 主进程加载的五个 FFmpeg DLL 全部来自 `runtime`，没有 FFmpeg 子进程或 EXE 依赖。
- 未认证健康检查返回 `401`；正确 key 的文件和 WebSocket 请求成功。
- 连续录音翻译在 `finish` 前返回第一句原文与译文，两段最终结果正常完成。仅识别连续录音也提前返回原文并正常完成。
- 测试结束后正常 `Ctrl+C`，新旧测试服务及托管 llama-server 均已退出。

英语和日语文件翻译均返回完整内容，与旧包对照文本一致。整链路 GPU 测量存在明显阶段波动，未作 CUDA 版本速度优劣结论：例如新包初次英语组翻译为 400–575 ms，后续交错组为 404–423 ms；旧包不同组也从约 403–410 ms 变化至 577–618 ms，ASR 同时变化。请求前后观察到 P3 空闲低频与 P2 运行频率变化，但不足以证明全部波动的单一原因。原始首轮、后续请求及频率观测均保留，避免将某组较快结果当作普遍提升。

正式接口回执在 `.tools/benchmark/shared-cuda-20261107-*`；`new-receipt.json` 记录模块来源，`*-controlled.json` 记录交错请求与 GPU 状态，`file-formats.json` 记录 8 种格式，`ws-translate.jsonl` 与 `ws-transcribe-validated.jsonl` 记录连续录音事件，`shutdown.json` 记录退出状态。

本次仅在 Windows x64、RTX 3050 上实测，其他 GPU 未实测。Linux 保留原有 FFmpeg CLI 路径且未进行实机验证。进程内解码的 30 秒时限通过 FFmpeg 中断回调与循环检查实现，不具备原子进程强制终止任意原生函数的能力。
