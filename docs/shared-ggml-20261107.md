# 统一 ggml 构建与验证（2026-11-07）

ASR 与 llama.cpp 已共用同一套 ggml 核心、CUDA 后端和 CPU 后端。相较上一轮仅共享 CUDA 的发行包，本轮再减少约 **194 MiB（15.4%）**，目录约从 **1.234 GiB 降至 1.044 GiB**。这是文件总大小，未统计 ZIP 压缩率或显存节省；最终逐文件大小与 SHA256 以包内 `manifest.json` 为准。

新发行目录为 `dist/StreamTrans-shared-ggml`，上一轮 `dist/StreamTrans-shared-cuda` 保留作对照。模型、模型精度、ASR 转换缓存、LLM 采样参数与接口模式均未更换。

## 如何统一

原 ASR 预编译包采用 ggml 0.26，llama.cpp b11960 使用 0.28。虽然后端 API 版本号相同，算子枚举已有变化，例如新增 `GGML_OP_TOP_K_MASK` 后部分编号偏移；旧核心不能直接配新版 CUDA 插件。ASR 还依赖原 fork 的后端注册过滤接口，单独替换同名 DLL 不足以保证兼容。

本轮采用同源重编：

1. 继续固定 [llama.cpp b11960 提交](https://github.com/ggml-org/llama.cpp/tree/8642541b255bfb5dc39cd351a178be09013a55d6) 及其 ggml 0.28。
2. 移植后端注册过滤与 Windows CPU 线程池兼容补丁，保留 ASR 允许后端选择和线程调度行为。补丁只修改核心注册及 CPU 实现，不修改 ggml 公共头文件。
3. [transcribe.cpp 0.4.1 固定提交](https://github.com/handy-computer/transcribe.cpp/tree/4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b) 使用同一套 ggml 头文件和导入库重编。只调整构建接入和过滤接口声明，不修改模型图、权重或解码算法；保留公开 ABI 0.4.1。
4. 两端 CPU 后端统一关闭 OpenMP，使用原生线程池；保留 9 个 CPU 变体，运行时按本机能力选择。
5. 全部原生库平铺在 `runtime` 根目录，`ggml.dll`、`ggml-base.dll`、`ggml-cuda.dll` 和各 CPU 变体只复制一份。

本次 146 个 CUDA 对象未重新编译，文件大小与修改时间保持不变；CUDA DLL 的 SHA256 也未改变，复用了上一轮编译结果。仅重新编译 CPU/核心兼容部分并链接 ASR，无须重复上次完整的 CUDA 内核编译。

两端构建清单记录同一 ggml 源码提交、兼容补丁标识及完整 12 个 ggml DLL 的哈希；打包时交叉校验，任一核心、CPU 变体、文件集合或哈希不一致都会拒绝。构建脚本与补丁进入源码仓库，下载缓存、编译二进制及模型不进入版本控制。

兼容标识为：

```text
sha256:ff9af0c64c12b80f180792f7581982a45ac2a18b2e309087e9d7a9c848861177
```

构建顺序为 `setup-llama.ps1` → `build-asr.ps1` → `setup-ffmpeg.ps1` → Rust 服务。开发产物为 `.tools/llama-shared` 和 `.tools/transcribe-shared`；开发目录各自携带相同依赖副本便于独立测试，发行目录统一去重。安装和分发参数见 [分发说明](distribution.md)。

## 原生 ASR 对照

测试机器为 Ryzen 7 5700G、RTX 3050 6 GB，驱动 621.14。比较上一轮发行包的 ASR 与本轮重编 ASR，使用相同 BF16 模型、FSMN-VAD、音频、语言参数和 GPU 编号；测试时没有后台编译或并行 GPU 模型任务。

每组在新进程中分别记录模型加载和首轮推理，再额外预热 2 轮，随后测量 5 轮。表中首轮仅为首次推理，不含模型加载，也不表示清空了系统文件缓存。计时为各 VAD 分段的 ASR 耗时总和，不含 HTTP、FFmpeg 或翻译。

| 音频 | 旧版首轮 | 统一版首轮 | 旧版预热后中位数 | 统一版预热后中位数 |
| --- | ---: | ---: | ---: | ---: |
| 中文 9.122 秒、2 段 | 537.2 ms | 602.7 ms | 351.7 ms | 366.3 ms |
| 中文重复 18.244 秒、4 段 | 1805.7 ms | 1127.6 ms | 1460.8 ms | 925.8 ms |
| 英文 6.410 秒、2 段 | 555.1 ms | 576.0 ms | 406.8 ms | 412.2 ms |

三组音频在新旧版本全部测量轮次中的原文和 VAD 边界一致。中文原文为「你好，这是一个本地语音识别和翻译测试。今天下午三点开会。」；长音频完整返回两次重复内容。英文原文为：

> This is a local speech recognition test. The meeting starts at three o'clock this afternoon.

本轮主要验证兼容性与磁盘去重。短中文和英文样本接近，长中文这组更快；测试轮次有限、GPU 频率未锁定，且 CPU 后端实现也发生变化，不能将单组结果推广为所有音频的速度结论。加载时间、首轮、预热及逐轮数据均保存在 `.tools/ggml-unify-probe/bench/`，汇总为 `bench-summary.json`。

## 实际发行包检查

新包从项目外目录启动，`PATH` 仅保留 `System32`，只显式指定现有 ModelScope 模型缓存、回环地址和测试 key。ASR 继续使用 CUDA0，BF16 权重在 GPU 上；Hy-MT2 的 33/33 层全部加载到 GPU。缓存校验至监听约 6.11 秒，这是一次启动记录。

进程模块检查证明两端同时使用同一路径、同 SHA256 的 `ggml.dll`、`ggml-base.dll`、`ggml-cuda.dll` 和 `ggml-cpu-haswell.dll`，全部来自新包 `runtime`。三份 CUDA DLL 也继续共用；未加载 `VCOMP` 或 `libomp`，FFmpeg 继续通过进程内 DLL 解码。

实际 HTTP 与连续录音检查通过：

- 中文识别及英语、日语翻译完整返回，英文识别和中文翻译正常；长中文仅识别完整返回 4 段。
- 8 种输入（中文文件名 WAV、44.1 kHz 立体声 WAV、48 kHz 浮点 WAV、MP3、FLAC、M4A/AAC、Ogg/Vorbis、Opus）均返回相同中文原文，时长与上一轮一致。
- 仅识别模式的译文为 `null`、翻译耗时为 0；未认证健康请求返回 `401`。
- 连续录音日语翻译在 `finish` 前收到第一句原文及译文，两段最终结果正常完成。仅识别连续录音也在 `finish` 前返回第一句原文。
- 正常 `Ctrl+C` 后，主服务和托管 llama-server 均退出。

公开 ABI、PE 依赖、构建幂等性、共享来源不一致拒绝、唯一 DLL 副本及配置新旧路径优先级也已检查；Rust 测试、Clippy 和 PowerShell 5 布局自检通过。

原始回执位于 `.tools/ggml-unify-probe/`：`asr-build-receipt.json`、`llama-shared-build-receipt.json`、`runtime-receipt.json`、`http-results.json`、`file-formats.json`、`ws-translate.jsonl`、`ws-transcribe.jsonl` 和 `shutdown.json`。这些原始测试文件不进入发行包。

本轮只在 Windows x64、RTX 3050 上实测；其他 GPU、Linux 及日语语音识别未实测。日语结果来自中文语音的翻译输出。使用完整新发行目录，不要将旧版 ASR 或 ggml DLL 手工混入共享目录。
