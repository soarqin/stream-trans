# Candle 后端统一采样配置后的历史复测

本文记录 MiniCPM5 的历史实验。2026-11-03 起默认翻译模型为 Hy-MT2-1.8B Q6_K，新的完整性与响应速度测试见 [Hy-MT2 验证](hy-mt2-20261103.md)。

文中指向 `.tools`、`src` 和基准源码的本地链接仅在完整开发工作区可用，这些材料不随运行包分发。

本文保留切回 llama.cpp 前的性能排查结果。当前生产翻译已改用随发行包携带的 llama.cpp；下文 Candle 实现、测试数量和复现命令对应当时版本，现行安装与运行方式见 [README](../README.md)。

2026-10-31，Windows x64、Ryzen 7 5700G、RTX 3050（6 GB）、驱动 621.01。使用 Release 构建，Candle 0.9.1，参考 llama.cpp b11960。

生产采样器已修复。同一条中文原文译为日文，Candle 首轮为 **7.436 秒**，三次热请求中位数为 **7.315 秒**；统一配置后的 llama.cpp 为 **4.863 秒**，差距约 **1.50 倍**。这些是 LLM 翻译时间，不含模型加载或 ASR。修复前约 85 秒的记录不代表当前默认配置。

## 对齐的配置与实现

- 使用同一份 `MiniCPM5-2B-Q4_K_M.gguf`，提示词及全部 99 个输入 token ID 逐项一致。
- `seed=42`、`temperature=1.0`、`top_k=40`、`top_p=0.95`、`min_p=0.0`。
- 重复惩罚系数为 `1.0`，presence/frequency 惩罚均为 `0.0`。
- 上下文 4096、生成预算 2048，Think 模式保持启用，自然生成到 EOS，没有缩短思考预算。
- FP32 KV 缓存，不使用 FlashAttention，模型计算位于 CUDA0。各项测试依次运行。
- 采样顺序为温度缩放 → Top-K 排序 → FP32 softmax / Top-P 截断 → 对保留候选重算 expf → FP64 累积概率抽样。
- MT19937、参考 Windows 构建的均匀数映射、候选顺序和 CDF 遍历方式均已对齐。

原实现缺少默认 Top-K=40，使用的 Top-K/Top-P 处理与参考链存在差异，随机数及最终候选抽样顺序也不同。本次移除了生产路径中的 Candle `StdRng/WeightedIndex`，按固定参考版本的实际代码实现采样，没有增加外部运行依赖。

CPU 测试直接调用实际缓存的 `llama.dll` 完整采样链，生成不同 logits、温度和截断参数的受控向量。三组向量共 48 次输出 ID 逐项匹配；另验证 10000 次 MT 输出、同分候选顺序及单候选的随机数消耗。实现与测试见 [llm.rs](../src/llm.rs)，参考数据见 [完整链向量](../.tools/native-llm-probe/sampler-chain-reference.json)及 [RNG 向量](../.tools/native-llm-probe/rng-parity-msvc.json)。

## 冷、热分离的日文翻译

输入原文为：

```text
你好，这是一个本地语音识别和翻译测试。 今天下午三点开会。 你好，这是一个本地语音识别和翻译测试。 今天下午三点开会。
```

目标语言为 `日本語`。Candle 模型加载单独耗时 **1143.700 ms**，不计入下表。双方均单列进程首轮，再测相同输入的 3 次热请求；没有清除操作系统文件缓存。

llama.cpp 禁止自动预热和 prompt 缓存。所有响应均确认 `cache_n=0`、`prompt_n=99`、正常 EOS。Candle 使用 `translate_measured` 总时间；llama.cpp 使用服务端 `prompt_ms + predicted_ms`，不包含 HTTP 客户端和传输时间。

| 测量 | Candle | llama.cpp，FP32 KV / FA off |
| --- | ---: | ---: |
| 进程首轮 | 7.436 s | 4.892 s |
| 热请求 1 | 7.328 s | 4.854 s |
| 热请求 2 | 7.284 s | 4.863 s |
| 热请求 3 | 7.315 s | 4.871 s |
| 热请求中位数 | **7.315 s** | **4.863 s** |

每个引擎内部的四次请求分别产生相同内容，但跨引擎的输出不同：

| 计数 | Candle | llama.cpp |
| --- | ---: | ---: |
| 输入 token | 99 | 99 |
| 思考 token | 254 | 293 |
| 思考分隔符 | 1 | 1 |
| 译文 token | 53 | 57 |
| 生成内容，不含 EOS | 308 | 351 |
| EOS | 1 | 1 |
| 后端报告的热态 decode 中位数 | 42.573 token/s | 72.917 token/s |

吞吐保留后端自身的计数口径：双方排除 prefill 中的首个 token，Candle 的速率分子还排除终止 EOS，llama.cpp 的速率分子包含该 EOS 步。这一 token 的差异不影响本样本的延迟倍数结论。

Candle 首轮 prefill 为 124.115 ms，热态 prefill 中位数为 103.125 ms；热态 decode 中位数为 7211.142 ms。预热并不是此前约 85 秒的主要原因。修复采样后，这条输入的思考量从 1949 个 token 减少到 254 个，翻译耗时降至约 7.3 秒。

参数、采样链与随机数算法相同，不代表两个引擎的前向浮点结果逐位一致。8 步相同前缀的教师强制检查中，每一步最高概率 token 均一致，但共同 top20 候选的 logprob 平均绝对差约为 0.220。因此本表是同输入的真实翻译延迟对照，不是同输出或同等质量验证。[前向数值对照](../.tools/native-llm-probe/logits-parity-ja-eight-comparison.json)保留了具体差异。

原始数据：[Candle 四轮及统计](../.tools/benchmark/aligned-candle-ja-mt.jsonl)、[参考配置](../.tools/benchmark/aligned-llama-ja-k40/environment.json)、[参考四轮计时](../.tools/benchmark/aligned-llama-ja-k40/runs.jsonl)、[参考 token 分段](../.tools/benchmark/aligned-llama-ja-k40/token-summary.json)。

## 完整音频接口

最新生产程序启动时仅保留 Windows 系统 PATH，并显式指定 FFmpeg；没有调用 Python 或 llama-server。模型加载不包含在请求时间中。

| 音频与目标语言 | VAD + ASR | 翻译 | 完整接口 |
| --- | ---: | ---: | ---: |
| 18.243 秒重复中文 → 日文，新进程首请求 | 1034 ms | 7491 ms | **8630 ms** |
| 随后 9.122 秒中文 → 英文 | 320 ms | 1961 ms | **2374 ms** |

这两项各测一次，不能视为多轮中位数。两次均返回 HTTP 200、原文、完整译文与语音段。日文 token 数与独立基准一致；英文生成 52 个思考 token、1 个分隔符和 27 个译文 token。

原始记录：[日文接口](../.tools/benchmark/aligned-mt-http-ja.json)、[英文接口](../.tools/benchmark/aligned-mt-http-en.json)、[生产日志](../.tools/benchmark/aligned-mt-http-server.log)。`cargo test --locked` 通过 23 项、跳过 1 项本地模型测试；格式检查、Clippy 和 Release 构建通过。

## 复现

先按 [README](../README.md) 准备模型和原生环境。在同一个 PowerShell 会话中运行构建环境脚本，然后构建基准：

```powershell
& .\scripts\setup-cuda.ps1
cargo build --release --examples --locked
```

使用本机的模型路径和以上原文运行当前 Candle 基准：

```powershell
$llm = 'models/models/OpenBMB--MiniCPM5-2B-GGUF/snapshots/5cd297aab1231e38021260d1813963aff897c0da/MiniCPM5-2B-Q4_K_M.gguf'
$text = '你好，这是一个本地语音识别和翻译测试。 今天下午三点开会。 你好，这是一个本地语音识别和翻译测试。 今天下午三点开会。'
& .\target\release\examples\bench_llm.exe --model $llm --device cuda --workload translation --text $text --target 日本語 --top-k 40 --temperature 1 --top-p 0.95 --seed 42 --warm-runs 3
```

参考引擎仅用于测试，需要显式提供已有的 `llama-server.exe`。以下命令使用本机已保存的原始请求；在其他机器上需自行准备同样的请求和文件路径。脚本不安装或下载任何依赖。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bench-llama.ps1 -Server .tools/llama/llama-server.exe -Model $llm -RequestJson .tools/benchmark/llama-ja-k40-request.json -OutputDirectory .tools/benchmark/reference-new-run -WarmRuns 3
```

等待前一项进程退出后再运行下一项，避免显存和计算争用。模型、音频与 `.tools` 中的原始结果属于本机缓存，不随 Git 仓库提供；其他硬件和更长输入需要单独测量。
