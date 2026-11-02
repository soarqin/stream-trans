# 本地模型与原生 GPU 环境

StreamTrans 使用 Rust 提供 HTTP 接口、下载模型、转换权重和调度推理。ASR 通过 transcribe.cpp C API 在 CUDA 上运行；文本翻译通过随包分发的 llama.cpp 在 CUDA 上运行；FSMN-VAD 通过 Rust Candle 在 CPU 上运行。运行时无需 Python 或 PyTorch 环境。Rust 自动启动和清理本机 `llama-server` 子进程，无须另行部署模型服务。ASR 原生库、llama.cpp、CUDA 运行库和 FFmpeg 均属于运行依赖，Windows 分发包会一并携带。

## 固定模型版本

| 用途 | ModelScope 仓库 | 固定 revision | 精度与执行方式 |
| --- | --- | --- | --- |
| 语音识别 | `FunAudioLLM/Fun-ASR-Nano-2512` | `05201c46f1c38592b1567f857c0d56eab3d0d8ef` | CUDA；二维矩阵权重 BF16，音频激活 FP32 |
| 语音分段 | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | `f9a8b8274674755d925277e27063869038d41515` | Rust Candle，CPU FP32 |
| 文本翻译 | `Tencent-Hunyuan/Hy-MT2-1.8B-GGUF` | `e97e6af0ec0950385644d7c0a8e5c899071f722b` | llama.cpp CUDA，`Q6_K` |

模型由 Rust 从 ModelScope 下载。开发模式默认使用当前工作目录下的 `models`，发行模式默认使用可执行文件旁的 `models`，可用 `--cache-dir` 修改。原始下载文件的列表、大小和 SHA256 来自 ModelScope 仓库 API，下载完成后校验；后续通过本地清单复用完整缓存，可以离线启动。清单按模型仓库、revision 和文件规格生成独立指纹，切换到 Hy-MT2 时保留旧模型和旧清单，新模型独立下载及校验。模型权重和生成的转换缓存不提交到 Git，也不放入分发包。

模型页面：[Fun-ASR-Nano-2512](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)、[FSMN-VAD](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)、[Hy-MT2-1.8B-GGUF](https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-1.8B-GGUF)。文件请求使用 [ModelScope 官方 SDK 定义的 HTTP 接口](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/file_download.py)，项目不依赖该 Python SDK。

## 翻译模型最终选型

2026-11-03，经用户确认，项目最终选用 **Hy-MT2-1.8B GGUF Q6_K**，继续通过 llama.cpp CUDA 运行。生产默认下载上述固定 revision 的 `Hy-MT2-1.8B-Q6_K.gguf`，保留现有翻译模板、推荐采样和完整术语表前缀缓存。

选型依据是显存上限 3.5 GB 的 12 个量化版本实测，比较中日韩英翻译的事实保留、内容完整性、术语遵从与响应时间。其他候选未展示稳定的整体质量优势；Gemma4 QAT 虽然更快，仍有术语不遵从和整句遗漏。完整证据见 [模型对比报告](translation-model-comparison-20261103.md) 和 [离线逐句对照页](translation-model-comparison-20261103.html)。

此决定保留现有生产默认，不表示模型已消除错译。金额、否定关系及跨语言术语污染等已知限制仍需由调用方核对；正常 EOS 也不保证语义完整。对比的 greedy 参数用于评测，生产仍使用下述推荐采样值。

## ASR 权重转换与精度

首次启动时，Rust 读取 Fun-ASR-Nano 官方 PyTorch 检查点和分词器，转换成 ASR 模型目录下的 `Fun-ASR-Nano-2512-native-bf16-v1.gguf`。转换不执行 Python 或 pickle 代码，通过临时文件原子保存，成功后复用本地 GGUF。派生 GGUF 在加载时检查结构，未另行记录其内容哈希；原始下载文件仍保留，因此需要为转换后的副本预留磁盘空间。

转换采用上游 BF16 GGUF 的存储策略：编码器、适配器和解码器的二维矩阵权重使用 BF16；归一化参数、偏置和 FSMN 卷积权重使用 FP32。音频计算的激活保持 FP32，ASR 未改成 INT4。映射、元数据和存储策略对应固定版本的 [transcribe.cpp Fun-ASR-Nano 转换器](https://github.com/handy-computer/transcribe.cpp/blob/v0.4.1/scripts/convert-funasr_nano.py)。原生前端、CUDA 算子和 KV 缓存与 PyTorch 参考实现不同，不保证逐位一致。

ASR 原生库明确选择 CUDA 后端，服务检查实际模型后端，不自动回退 CPU。`--asr-library` 可指定兼容的原生库。Windows 默认路径为：

```text
.tools/transcribe/transcribe-native-windows-x86_64-cuda/transcribe.dll
```

发行模式自动查找可执行文件旁的 `runtime/asr/transcribe.dll`。原生库来自 [transcribe.cpp v0.4.1 官方发行版](https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.4.1)。安装脚本固定版本并检查 SHA256：

| 平台 | 官方资产 | SHA256 |
| --- | --- | --- |
| Windows x64 CUDA | `transcribe-native-0.4.1-windows-x86_64-cuda.tar.gz` | `ba43fb21a3fe0ab2630e0614fbd742e039c389c68ad10fe23aca903f986adc5b` |
| Linux x86_64 CUDA | `transcribe-native-0.4.1-linux-x86_64-cuda.tar.gz` | `c4bdf9c4798bac1868ecdcaa04f27eb7712fb43125268313ce14d508c70d247d` |

## Hy-MT2 量化推理与托管子进程

项目使用固定版本的 [llama.cpp b11960](https://github.com/ggml-org/llama.cpp/releases/tag/b11960)，直接加载 ModelScope 官方的 `Hy-MT2-1.8B-Q6_K.gguf`，保留原始量化权重。所有模型层加载到 CUDA；llama.cpp 负责分词、模型计算和采样，Rust 负责请求调度与结果检查，不需要 Python 推理环境。

该文件大小为 `1474785120` 字节，SHA256 为：

```text
d98fe604dec1f28f58f80d7d560f7177e584d3b8e5835862687660e5ff97cb40
```

来源：[固定 revision 的 ModelScope 文件清单](https://modelscope.cn/api/v1/models/Tencent-Hunyuan/Hy-MT2-1.8B-GGUF/repo/files?Revision=e97e6af0ec0950385644d7c0a8e5c899071f722b&Recursive=true)。分词器与聊天控制标记使用 GGUF 内的数据。提示基于当前模型的官方翻译模板，追加逐句完整翻译、保留数字、单位、条件和否定关系的提醒；不是逐字使用官方示例，也不使用旧 MiniCPM5 的空思考前缀。该提醒不能保证所有事实都会正确保留，仍需检查实际译文。

生产默认使用官方 1.8B 推荐采样值 `temperature=0.7`、`top_p=0.6`、`top_k=20`、重复惩罚 `1.05`；项目固定 `seed=42`。`min_p=0.0`，presence 与 frequency 惩罚为 `0.0`。模型直接生成译文，不提供思考模式开关；没有译文、未返回正常结束标记或生成耗尽预算时返回错误。官方模板与参数见 [Hy-MT2-1.8B-GGUF 模型卡](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF#inference-and-deployment)。

`llama-server` 由 Rust 启动，绑定 `127.0.0.1` 的临时端口，使用启动时生成的 API 密钥。服务加载模型并检查 CUDA 后端后，再开放 StreamTrans 音频接口。子进程退出后，健康检查将报告后端不可用；StreamTrans 退出时清理托管子进程。该接口仅供 Rust 主进程调用，不作为外部模型 API 分发。

`--llama-server` 可指定可执行文件。Windows 开发模式默认使用 `.tools/llama/llama-server.exe`，发行模式使用可执行文件旁的 `runtime/llama/llama-server.exe`。默认 `--context-size 4096`、`--max-tokens 2048`，输出预算用于译文。当前服务的参数范围不代表原模型完整的上下文能力。

生产子进程使用 FP16 KV 缓存，FlashAttention 由 llama.cpp 自动选择。未加载术语表时使用 `cache_prompt=false`；加载术语表时，启动只读入一次完整 canonical TSV，再将所有语言的全部条目按官方推荐的逐行术语句式编码进同一条 user 消息的固定前缀，使用完整中文语言标签。每次请求明确当前目标语言，后续复用完整前缀 token 和 KV，不按当前原文筛词。术语表 token 上限仍按 canonical TSV 计算，实际渲染前缀另查上下文容量，不自动压缩输出预算。部署与容量规则见 [术语表说明](glossary.md)。LLM 本地请求超时或通信失败时，Rust 停止并回收子进程，再返回错误；此时需要重启 StreamTrans。

指定译法仍由模型理解，不能保证语言隔离或强制遵从。例如日语请求仍可能带入仅为英语指定的 `on-device`；不能把提示格式调整当作这项限制已经解决。

新模型的完整性、术语遵从和速度记录见 [Hy-MT2 复测](hy-mt2-20261103.md)。此前 MiniCPM5 的 FP32 KV、FlashAttention 关闭及思考模式对照属于历史实验，其数字不代表当前模型。

## FSMN-VAD 分段

VAD 使用 ModelScope 原始 FSMN 权重和 CMVN，按固定模型配置由 Rust 实现音频特征、FSMN 网络和语音边界状态机，使用 CPU FP32。断句不依赖 Python，也不占用大模型的 GPU 显存。

VAD 检测语音活动边界，不保证每段都是完整语句。原生浮点计算与参考实现的边界可能有小幅差异，应根据实际音频评估；接口继续使用毫秒时间戳，单段上限为 15 秒。

## Windows 构建与运行

构建需要：

- Windows x64、Rust stable、Visual Studio 2022 C++ Build Tools，安装「使用 C++ 的桌面开发」组件。
- 构建不需要 CUDA Toolkit 或 NVCC。
- 运行需要支持原生 BF16 的 NVIDIA GPU，以及兼容随包 CUDA 运行库的 NVIDIA 驱动。
- 首次安装可访问原生程序上游下载站点和 crates.io；首次模型下载可访问 ModelScope。

安装命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Gpu 0
```

脚本下载并校验 ASR 原生库、llama.cpp 和 CUDA 运行库，然后执行 `cargo build --release`。FFmpeg 需放入 `PATH`，或通过 `--ffmpeg` 指定路径。Candle 仅用于 CPU VAD 和 ASR 权重转换，不再启用 Candle CUDA 或编译 CUDA 内核。原生推理后端使用预编译二进制，运行库放入项目内 `.tools`；系统不需要安装 CUDA Toolkit。

从项目根目录启动：

```powershell
.\target\release\streamtrans.exe --gpu 0
```

重新编译使用 `scripts/build.ps1`，可用 `-Action test` 或 `-Action clippy` 执行检查。制作完整运行目录使用 `scripts/dist.ps1`，具体布局、运行依赖和许可证见 [分发说明](distribution.md)。

## Linux 构建与运行

Linux x86_64 流程尚未实机验证。需要 NVIDIA 驱动、Rust、C++ 编译器、FFmpeg、curl 和 tar。`scripts/setup.sh` 校验并解压 ASR 原生库、llama.cpp b11960 Ubuntu CUDA 12.8 二进制及其 CUDA 运行库，然后构建 Rust 服务，不要求 NVCC 或 CUDA Toolkit：

```bash
bash scripts/setup.sh
LD_LIBRARY_PATH="$PWD/.tools/transcribe:$PWD/.tools/llama${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./target/release/streamtrans \
  --asr-library .tools/transcribe/libtranscribe.so \
  --llama-server .tools/llama/llama-server
```

llama.cpp 官方包面向 Ubuntu 24.04 构建，其他发行版需验证 glibc、C++ 运行库等系统依赖。固定下载资产与 SHA256 来自 [b11960 官方发行清单](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11960)：

| Linux x64 资产 | SHA256 |
| --- | --- |
| `llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz` | `64d57beb9abb637c177c6afb313760a0c0782224cac71e74869eed5b8f9a1ea4` |
| `cudart-llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz` | `7c4d5974bd39649df649649e5d71950688a70e37aaae00bf11afbb656ea2cc56` |

如果官方预编译库与系统不兼容，可按 [transcribe.cpp CUDA 构建说明](https://github.com/handy-computer/transcribe.cpp/tree/v0.4.1#build) 和 [llama.cpp 构建说明](https://github.com/ggml-org/llama.cpp/blob/b11960/docs/build.md) 构建兼容的 CUDA 后端，再通过 `--asr-library` 和 `--llama-server` 指定路径。自行编译 CUDA 后端需要对应的构建工具；这不属于 Rust 服务构建的前置条件。原生库及其依赖必须能被系统动态链接器找到。

## 许可与已知限制

模型许可由各模型仓库规定，使用前应阅读对应模型卡。Candle 使用 MIT/Apache-2.0 双许可，transcribe.cpp、llama.cpp 和 ggml 使用 MIT；分发时保留上游许可证。FFmpeg 的许可与构建选项有关，分发包需保留所用二进制对应的许可和源码获取说明。CUDA 组件遵循 NVIDIA 的分发许可。

GPU 按语音段顺序执行 ASR 与 LLM，两者共用所选 GPU；CPU 收音和 VAD 独立推进。显存占用随音频、上下文和运行库变化，不能仅凭模型文件大小推算。原生 ASR 与 VAD 不承诺和 PyTorch 实现逐位一致。Linux、长音频负载及其他硬件的性能仍需单独验证。
