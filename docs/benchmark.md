# 本机推理速度排查记录（修复前）

本文记录 MiniCPM5 的历史实验。2026-11-03 起默认翻译模型为 Hy-MT2-1.8B Q6_K，新的完整性与响应速度测试见 [Hy-MT2 验证](hy-mt2-20261103.md)。

文中指向 `.tools`、`src` 和基准源码的本地链接仅在完整开发工作区可用，这些材料不随运行包分发。

当前生产翻译已切回随发行包携带的 llama.cpp。本文及所链接的 Candle 复测保留为历史记录，命令对应当时实现；现行流程见 [README](../README.md)。

2026-10-31 在 Windows x64、AMD Ryzen 7 5700G、RTX 3050（6 GB）、NVIDIA 驱动 621.01 上复测。原生 Rust 程序使用 Release 构建，Candle 版本为 0.9.1，CUDA 运行库为 12.8.1。

本页保留采样修复前的排查数据，不代表当前默认配置。参数、采样链和随机数算法统一后的最终结果见 [最终复测](sampling-benchmark.md)：同一日文输入，Candle 三次热请求中位数为 **7.315 秒**，统一 FP32 KV、关闭 FlashAttention 的 llama.cpp 为 **4.863 秒**，差距约 **1.50 倍**。下文约 85 秒及 21–22 倍的记录仅用于追溯当时问题。

## 测量口径

- **模型加载**：从创建运行设备及加载权重开始计时，独立于推理耗时。
- **进程首轮（cold）**：模型加载完成后，在该进程中执行的第一次推理。没有清除操作系统文件缓存，不代表冷磁盘读取。
- **预热后（warm）**：同一进程、同一模型继续执行相同工作量，不重新加载模型。ASR 额外预热 2 次，再测量 5 次；LLM 将首轮后的 3 次执行单独统计。
- **GPU 调度**：各项基准依次运行，不并发执行其他模型测试；结束并退出前一项测试进程后，再运行下一项。桌面图形程序仍正常运行。
- **LLM 计时**：在阶段边界同步 GPU，分别记录 prefill 和 decode；没有额外添加逐层或逐 token 的同步。
- **统计**：Candle 与 ASR 表中的预热后值为中位数；旧 llama.cpp 对照工具输出的是多次执行的均值，表中明确标注。

两类 LLM 工作量不同，不能直接比较总耗时：

| 工作量 | 输入与输出 | 用途 |
| --- | --- | --- |
| 固定 token | 128 个输入 token，随后执行 128 次单 token decode；不采样，不遇 EOS 停止 | 观察模型计算吞吐 |
| 实际英文翻译 | 使用真实聊天模板、原文及思考模式，采样直到 EOS | 观察当前翻译请求的工作量和耗时 |

实际翻译固定 `seed=42`、`temperature=1.0`、`top_p=0.95`，每轮重置采样器并清理上一轮 KV 内容。固定 token 测试不执行采样，seed 不影响该工作量。固定 token 对照只比较指定上下文深度与 decode 次数下的吞吐，不作跨引擎数值一致性验证。

本文件同时记录独立组件基准和 ASR、LLM 同驻显存的完整 HTTP 测量；端到端耗时使用接口实测，不将组件中位数相加代替。

## FSMN-VAD 与 ASR

输入为 `.tools/smoke/zh16.wav`，16 kHz、单声道、16 位 PCM WAV，实际时长为 **9.1219375 秒**。FSMN-VAD 在 CPU 上执行，Fun-ASR-Nano-2512 在 CUDA 上执行。ASR 使用原生 BF16 矩阵权重，norm、bias、FSMN 卷积及音频激活保持相应 FP32 路径。

| 模型加载 | 耗时 |
| --- | ---: |
| FSMN-VAD | 4.677 ms |
| ASR 原生库、CUDA 设备及模型 | 1618.055 ms |
| 合计 | 1622.733 ms |

以下推理时间不包含模型加载：

| 阶段 | 进程首轮 | 预热后中位数 | 预热后最小–最大 |
| --- | ---: | ---: | ---: |
| FSMN-VAD | 27.192 ms | 22.579 ms | 21.087–27.145 ms |
| ASR，两段合计 | 448.790 ms | 298.449 ms | 295.434–298.555 ms |
| VAD + ASR | 475.982 ms | 320.858 ms | 318.013–323.491 ms |

| 语音段 | 进程首轮 ASR | 预热后中位数 | 识别文本 |
| --- | ---: | ---: | --- |
| 0–4460 ms | 327.314 ms | 184.057 ms | 你好，这是一个本地语音识别和翻译测试。 |
| 6220–9100 ms | 121.473 ms | 114.088 ms | 今天下午三点开会。 |

首轮、额外预热 2 次和正式测量 5 次的分段及识别文本均一致。

原始结果：[bench-asr-vad.json](../.tools/smoke/bench-asr-vad.json)。程序：[bench_asr.rs](../examples/bench_asr.rs)。

## MiniCPM5-2B 固定 token 工作量

所有对照使用同一份 `MiniCPM5-2B-Q4_K_M.gguf`。Candle 测试在独立进程加载模型，模型加载耗时为 **1138.194 ms**。

| Candle CUDA 阶段 | 进程首轮 | 预热后中位数，3 次 |
| --- | ---: | ---: |
| Prefill，128 个 token | 128.276 ms | 109.602 ms |
| Decode，128 个 token | 2850.366 ms | 2826.639 ms |
| Prefill + decode | 2978.642 ms | 2936.886 ms |
| Decode 吞吐 | 44.907 token/s | 45.283 token/s |

旧 llama.cpp b11960 仅用于本轮引擎对照，不是当前服务的运行依赖。旧引擎的 decode 对照生成 128 个 token，起始 KV 深度为 128；下表不含 prefill 时间，也不含真实翻译的采样和思考停止逻辑。

| 引擎与设备 | 预热后 decode 吞吐 | 统计方式 |
| --- | ---: | --- |
| Candle CUDA | 45.283 token/s | 3 次中位数 |
| 旧 llama.cpp CUDA | 81.077 token/s | 5 次均值 |
| 旧 llama.cpp CPU，8 线程 | 21.973 token/s | 3 次均值 |

在这个固定工作量下，Candle CUDA 的吞吐高于旧 llama.cpp CPU，低于旧 llama.cpp CUDA。这是引擎与设备组合的比较，不能将旧 llama.cpp 的 GPU 吞吐作为当前 Candle 服务的吞吐。

另做了仅用于基准的 CUDA event tracking 对照：关闭后，3 次 decode 吞吐的中位数为 **44.344 token/s**，没有观察到收益。生产路径保留默认 event tracking。

原始结果：

- [Candle 默认 event tracking](../.tools/benchmark/candle-gpu-fixed-events-on.jsonl)
- [Candle 关闭 event tracking 的对照](../.tools/benchmark/candle-gpu-fixed-events-off.jsonl)
- [旧 llama.cpp CUDA](../.tools/benchmark/llama-gpu-decode.json)
- [旧 llama.cpp CPU](../.tools/benchmark/llama-cpu-decode.json)

## MiniCPM5-2B 实际英文翻译

本项使用独立的新进程。输入原文与 API 合并后的文本完全一致，包括两句之间的一个半角空格：

```text
你好，这是一个本地语音识别和翻译测试。 今天下午三点开会。
```

目标语言为 `English`，`context_size=4096`，`max_tokens=2048`，固定 seed 与采样参数。模型加载耗时为 **1120.029 ms**，不计入下表。

| 执行 | 完整翻译耗时 |
| --- | ---: |
| 进程首轮 | 19.171 s |
| 预热后第 1 次 | 19.172 s |
| 预热后第 2 次 | 19.735 s |
| 预热后第 3 次 | 19.535 s |
| 预热后中位数 | **19.535 s** |

4 次执行的译文和 token 计数一致：

| 计数 | token 数 |
| --- | ---: |
| 输入聊天模板及原文 | 82 |
| 思考内容 | 600 |
| `</think>` 分隔符 | 1 |
| 最终译文 | 25 |
| 生成内容合计，不含 EOS | 626 |
| 终止 EOS，单独计数 | 1 |

预热后 decode 吞吐的中位数为 **32.171 token/s**，首 token 延迟的中位数为 **107.136 ms**。decode 速率按 625 个 token 计算：626 个生成内容 token 中，首个 token 已计入 prefill；EOS 单独计数，不计入速率分子。decode 计时仍包含最终 EOS 的计算。

实际翻译的工作量主要是思考内容，不能只用 25 个译文 token 估算耗时。真实输出的上下文长度也随生成增长，固定 128 次 decode 的吞吐不能直接套用到 626 个生成 token。

本样本冷、热翻译均约 19 秒，不支持将此前倍数差异归因于预热。旧的整次翻译观测缺少 token 计数；以下完整接口复测补充了英文及日语的实际生成工作量。

原始结果：[candle-gpu-translation-events-on.jsonl](../.tools/benchmark/candle-gpu-translation-events-on.jsonl)。程序：[bench_llm.rs](../examples/bench_llm.rs)。

## 完整 HTTP 链路与显存

新服务进程同时加载 ASR 和 LLM 后，依次执行同一份 9.122 秒音频的 4 次中文到英文请求：首轮单列，后续 3 轮取中位数。随后执行 18.243 秒重复音频的中文到日文请求；日文请求仅测 1 次，开始时模型已充分预热。所有请求均返回 HTTP 200。

启动时仅保留 Windows 系统 PATH，并用 `--ffmpeg` 指定已有的 FFmpeg，未调用 Python 或 llama-server。从模型缓存校验日志到开始监听约 5.17 秒，独立于下表的请求耗时。

| 请求 | VAD + ASR | 翻译 | 完整请求 |
| --- | ---: | ---: | ---: |
| 英文，进程首轮 | 573 ms | 19507 ms | 20175 ms |
| 英文，预热后第 1 次 | 331 ms | 19434 ms | 19862 ms |
| 英文，预热后第 2 次 | 329 ms | 19322 ms | 19747 ms |
| 英文，预热后第 3 次 | 332 ms | 19382 ms | 19804 ms |
| 英文，预热后中位数 | **331 ms** | **19382 ms** | **19804 ms** |
| 日文，已预热单次 | 654 ms | 85107 ms | 85866 ms |

| 翻译工作量 | 英文，每轮 | 日文，单次 |
| --- | ---: | ---: |
| 输入 token | 82 | 99 |
| 思考 token | 600 | 1949 |
| 思考分隔符 | 1 | 1 |
| 译文 token | 25 | 51 |
| 生成 token 合计，不含 EOS | 626 | 2001 |
| EOS | 1 | 1 |
| 解码速率 | 热态中位数 32.429 token/s | 23.531 token/s |
| 首 token 延迟 | 热态中位数 108.613 ms | 113.902 ms |

英文完整接口的解码吞吐与独立 LLM 的约 32.2 token/s 接近。日文生成序列更长，包含 1949 个思考 token，已预热后仍需要约 85 秒；不能将这部分时间记为 CUDA 预热。没有更改生产采样参数或关闭思考模式。

测试期间约每秒通过 Windows GPU 性能计数器记录进程及适配器内存：进程 Local/Dedicated 为 4350.7–4638.7 MiB，进程 Shared/NonLocal 始终为 80 MiB；适配器 Dedicated 为 5505.4–5795.9 MiB，包含桌面及其他进程。未观察到 Shared/NonLocal 持续增长，英文同驻模型时也未出现相对独立 LLM 的明显吞吐下降。该采样不代表其他输入或更长上下文的显存保证。

原始记录：[服务日志](../.tools/benchmark/http-server.log)、[显存采样](../.tools/benchmark/http-gpu-memory.csv)、[英文首轮](../.tools/benchmark/http-en-0.json)、[英文热态 1](../.tools/benchmark/http-en-1.json)、[英文热态 2](../.tools/benchmark/http-en-2.json)、[英文热态 3](../.tools/benchmark/http-en-3.json)、[日文热态](../.tools/benchmark/http-ja-warm.json)。

## llama.cpp 同输入日文自然生成对照

对照使用上节 18.243 秒重复音频的 ASR 原文，目标语言为 `日本語`，相同 `MiniCPM5-2B-Q4_K_M.gguf` 和 RTX 3050。这里只比较 LLM 翻译阶段，不包含 ASR、音频解码或模型加载。llama.cpp 使用 b11960；仅用于测试，不恢复为生产依赖。

两边使用完全相同的聊天提示词。额外通过 CPU-only probe 核对了生产 Candle 分词器：提示词字符串和全部 **99 个输入 token ID** 均与 llama.cpp 逐项一致。验证结果见 [tokenizer-parity-ja.json](../.tools/native-llm-probe/tokenizer-parity-ja.json)。

共同设置为 `seed=42`、`temperature=1.0`、`top_p=0.95`、`min_p=0`、上下文 4096、生成预算 2048，启用思考并自然生成到 EOS，不人为缩短思考。llama.cpp 先测 `top_k=0` 的纯 Top-P，与当前 Candle 配置对照；再测 `top_k=40` 的旧版默认设置。Candle 不使用 Top-K。

llama-server 用 `--no-warmup --no-cache-prompt --cache-ram 0 --parallel 1` 启动。每次 `/completion` 请求还设置 `cache_prompt:false`；所有响应均确认 `timings.cache_n=0`、`timings.prompt_n=99`、`stop_type="eos"`。没有通过重用前缀 KV 来缩短预填充。Top-K=0 组为新进程首轮加 3 次热请求；Top-K=40 组在同一进程继续执行，单列该组首次，再统计后续 3 次热请求。

llama.cpp 的「翻译阶段」使用服务端 `prompt_ms + predicted_ms`，不含 curl 启动和 HTTP 传输；另行保留请求墙钟时间。Candle 使用此前 `translate` 内部的 85106.935 ms，而非完整音频接口的 85866 ms。

| 引擎与采样 | 该组首次翻译阶段 | 热态翻译阶段 | 热态 HTTP 墙钟 | 热态样本数 |
| --- | ---: | ---: | ---: | ---: |
| Candle，纯 Top-P | 未单测日文首轮 | 85.107 s | 不适用，见完整音频接口 | 1 |
| llama.cpp，Top-K=0 | 4.049 s，新进程首轮 | **3.855 s** | 3.878 s | 3，中位数 |
| llama.cpp，Top-K=40 | 4.371 s，已运行上一组 | **4.070 s** | 4.091 s | 3，中位数 |

llama.cpp 每组内部的重复请求生成了相同的内容与 token 数，但两个引擎的输出不同：

| 计数 | Candle | llama.cpp，Top-K=0 | llama.cpp，Top-K=40 |
| --- | ---: | ---: | ---: |
| 输入 token | 99 | 99 | 99 |
| 思考 token | 1949 | 223 | 267 |
| 思考分隔符 | 1 | 1 | 1 |
| 译文 token | 51 | 56 | 51 |
| 生成内容合计，不含 EOS | 2001 | 280 | 319 |
| EOS，单独列出 | 1 | 1 | 1 |
| 后端报告的解码速率 | 23.531 token/s | 73.547 token/s | 79.271 token/s |

token 分段来自实际生成 ID：先去掉末尾 EOS（本轮为 `130073`），再按 `</think>` 的 ID `9` 分割，没有重新分词估算。llama.cpp 的速率分子包含终止 EOS 步，Candle 的速率分子排除 EOS；两边均排除 prefill 中的首个 token。此处保留各后端原生口径，该 1 token 差异不影响本样本的倍数结论。

在这条输入上，llama.cpp 并不需要约 85 秒。Candle 的自然生成延迟约慢 21–22 倍，观察到的差距同时包含更长的思考输出和更低的逐 token 吞吐。固定 token 的吞吐测试不能代替这组实际日文翻译对照。

这不是同等输出或同等翻译质量的比较。Top-K=0 的 llama.cpp 译文混入了英文，Top-K=40 的输出也有措辞和时间精度差异；原始响应保留了全文，未将这些结果作为质量验收。相同 seed 在不同引擎中也不保证相同随机序列，不能据此把思考长度差异归因于某一个采样参数。日文 Candle 仍仅测过一次热请求，因此这些数据说明本条样本的实际表现，不代表所有日文输入。

原始记录：[比较汇总](../.tools/benchmark/llama-ja-comparison.json)、[纯 Top-P 请求](../.tools/benchmark/llama-ja-k0-request.json)、[纯 Top-P 四轮计时](../.tools/benchmark/llama-ja-k0-summary.jsonl)、[Top-K=40 请求](../.tools/benchmark/llama-ja-k40-request.json)、[Top-K=40 四轮计时](../.tools/benchmark/llama-ja-k40-summary.jsonl)、[服务端日志](../.tools/benchmark/llama-ja-server.log)。

## 基准运行方法

以下命令现在使用修复后的生产采样器，不会重现本页修复前的生成序列。当前配置的日文复现命令见 [最终复测](sampling-benchmark.md#复现)。

在项目根目录的同一个 PowerShell 会话中运行。模型与 ASR 原生库须已通过安装和下载流程准备好；下列路径对应本次本机缓存，其他机器应替换成自己的实际路径。音频夹具和 `.tools` 原始结果属于本机文件，不随 Git 仓库提供。

1. 准备构建环境并构建 Release examples。

   ```powershell
   & .\scripts\setup-cuda.ps1
   cargo build --release --examples --locked
   ```

   `setup-cuda.ps1` 会在当前会话中将项目 CUDA `bin` 目录加入 `PATH`。在新终端直接运行 examples 时，需要重新准备这个环境；ASR 后端也需要找到 CUDA DLL。

2. 指定本机模型和原生库路径。

   ```powershell
   $llm = 'models/models/OpenBMB--MiniCPM5-2B-GGUF/snapshots/5cd297aab1231e38021260d1813963aff897c0da/MiniCPM5-2B-Q4_K_M.gguf'
   $asr = 'models/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/05201c46f1c38592b1567f857c0d56eab3d0d8ef/Fun-ASR-Nano-2512-native-bf16-v1.gguf'
   $vad = 'models/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/f9a8b8274674755d925277e27063869038d41515'
   $native = '.tools/transcribe/transcribe-native-windows-x86_64-cuda/transcribe.dll'
   ```

3. 单独运行 ASR/VAD。等待程序退出后，再运行下一项。

   ```powershell
   & .\target\release\examples\bench_asr.exe --asr-library $native --asr-model $asr --vad-dir $vad --audio .tools/smoke/zh16.wav --gpu 0 --warmup 2 --runs 5 --output .tools/smoke/bench-asr-vad.json
   ```

4. 在新进程运行固定 token LLM 基准。

   ```powershell
   & .\target\release\examples\bench_llm.exe --model $llm --device cuda --workload fixed --gpu 0 --cpu-threads 8 --prompt-tokens 128 --decode-tokens 128 --warm-runs 3 --context-size 4096 --max-tokens 2048 --seed 42
   ```

5. 等待固定 token 基准退出，再启动实际翻译基准。

   ```powershell
   & .\target\release\examples\bench_llm.exe --model $llm --device cuda --workload translation --gpu 0 --cpu-threads 8 --warm-runs 3 --context-size 4096 --max-tokens 2048 --seed 42 --temperature 1.0 --top-p 0.95 --target English --text '你好，这是一个本地语音识别和翻译测试。 今天下午三点开会。'
   ```

`bench_asr` 输出完整 JSON；`bench_llm` 逐行输出环境、模型加载、每轮执行、预热后统计和模型释放事件。不同硬件、驱动、GPU 后台负载和引擎版本的结果需要重新测量。
