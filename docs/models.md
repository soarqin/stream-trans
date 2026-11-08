# 本地模型与原生 GPU 环境

StreamTrans 使用 Rust 提供 HTTP 接口、下载模型、转换权重和调度推理。ASR 通过 transcribe.cpp C API 在 CUDA 上运行；文本翻译通过随包分发的 llama.cpp 在 CUDA 上运行；FSMN-VAD 通过 Rust Candle 在 CPU 上运行。运行时无需 Python 或 PyTorch 环境。Windows 在主进程内调用 `llama-server-impl.dll` 的项目 C ABI，Linux 由 Rust 自动启动和清理本机 `llama-server` 子进程，无须另行部署模型服务。ASR 原生库、llama.cpp、CUDA 运行库和 FFmpeg 均属于运行依赖，Windows 分发包会一并携带。

## 固定模型版本

服务端通过 `--model-source huggingface|modelscope` 统一选择三个模型的下载源，默认是 `huggingface`。未显式指定 `--asr-revision`、`--vad-revision` 或 `--llm-revision` 时，使用对应来源的下列固定版本；显式指定时，仅覆盖当前来源的相应模型版本。连接失败时返回错误，不自动切换来源。

### Hugging Face（默认）

| 用途 | Hugging Face 仓库 | 固定 revision | 精度与执行方式 |
| --- | --- | --- | --- |
| 语音识别 | `FunAudioLLM/Fun-ASR-Nano-2512` | `272c57b82523ada6fd87095e955f8e29100979ab` | CUDA；二维矩阵权重 BF16，音频激活 FP32 |
| 语音分段 | `funasr/fsmn-vad` | `df20e6b30c653645fa4ff125cacfcabd1020a669` | Rust Candle，CPU FP32 |
| 文本翻译 | `tencent/Hy-MT2-1.8B-GGUF` | `a0c709d9fac510f2c807aa3af52872340dc37a4a` | llama.cpp CUDA，`Q6_K` |

模型页面：[Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)、[FSMN-VAD](https://huggingface.co/funasr/fsmn-vad)、[Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)。

### ModelScope

| 用途 | ModelScope 仓库 | 固定 revision | 精度与执行方式 |
| --- | --- | --- | --- |
| 语音识别 | `FunAudioLLM/Fun-ASR-Nano-2512` | `05201c46f1c38592b1567f857c0d56eab3d0d8ef` | CUDA；二维矩阵权重 BF16，音频激活 FP32 |
| 语音分段 | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | `f9a8b8274674755d925277e27063869038d41515` | Rust Candle，CPU FP32 |
| 文本翻译 | `Tencent-Hunyuan/Hy-MT2-1.8B-GGUF` | `e97e6af0ec0950385644d7c0a8e5c899071f722b` | llama.cpp CUDA，`Q6_K` |

模型页面：[Fun-ASR-Nano-2512](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)、[FSMN-VAD](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)、[Hy-MT2-1.8B-GGUF](https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-1.8B-GGUF)。文件请求使用 [ModelScope 官方 SDK 定义的 HTTP 接口](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/file_download.py)，项目不依赖该 Python SDK。

### 下载、校验与缓存

模型下载由 Rust HTTP 客户端完成，不依赖 Hugging Face 或 ModelScope 的 Python SDK。Hugging Face 使用固定 revision 的仓库 API：LFS 文件读取 SHA256 与大小，非 LFS 文件读取内容后计算 SHA256；不会将 Git blob 的 SHA1 当作文件 SHA256。ModelScope 使用仓库 API 提供的文件列表、大小和 SHA256。两种来源都在下载时检查完整长度和 SHA256，完成后原子保存；中断的部分文件不会作为有效模型使用。

开发模式默认缓存根目录为当前工作目录下的 `models`，发行模式默认使用可执行文件旁的 `models`，可用 `--cache-dir` 修改。Hugging Face 文件位于 `<缓存根目录>/huggingface/models/<仓库>/snapshots/<revision>/`；ModelScope 沿用 `<缓存根目录>/models/<仓库>/snapshots/<revision>/`，其中仓库名的 `/` 转为 `--`。清单按来源、仓库、revision 和文件规格分别管理，不同来源不会误用彼此的清单。

已有的无来源字段旧清单视为 ModelScope，使用 `--model-source modelscope` 可继续校验并离线复用。离线机器须保留整个缓存目录及清单，选择与下载时相同的来源和版本。切换来源不会删除旧缓存；需要为另一个来源的文件和 ASR 转换副本预留磁盘空间。模型权重和生成的转换缓存不提交到 Git，也不放入分发包。

两个来源的 revision 不能互换；ASR 的 `model.pt` 打包文件并非逐字节相同，下载和转换均按所选来源独立校验与缓存。

2026-11-07 已通过默认来源实际下载全部 15 个文件，并在不可联网的代理配置下分别验证 Hugging Face 和旧 ModelScope 缓存复用。Hugging Face ASR 检查点经现有 Rust 转换器生成的 BF16 GGUF 为 `1667503232` 字节，SHA256 为 `9410f344a3f51d4fd9797d4fc7ef36ea66575878eea6828485f1df247e5f7e7c`，与既有 ModelScope 转换缓存完全一致。该本地转换回归可用 `cargo test --release --locked --bin streamtrans huggingface_checkpoint_matches_existing_native_bf16_weights -- --ignored` 复现，需要默认 Hugging Face 检查点已下载到项目的 `models` 缓存，不需要 GPU。本轮服务端及基准工具通过 98 项常规测试，另行通过上述转换测试；未重复执行 GPU 推理测试。

```powershell
# 默认来源，也可显式写 --model-source huggingface
.\target\release\streamtrans.exe --download-only

# 选择 ModelScope；去掉 --download-only 可正常启动服务
.\target\release\streamtrans.exe --model-source modelscope --download-only
```

## 翻译模型最终选型

2026-11-03，经用户确认，项目最终选用 **Hy-MT2-1.8B GGUF Q6_K**，继续通过 llama.cpp CUDA 运行。两个下载源均使用 `Hy-MT2-1.8B-Q6_K.gguf`，对应各自的固定 revision，保留现有翻译模板、推荐采样和完整术语表前缀缓存。

选型依据是显存上限 3.5 GB 的 12 个量化版本实测，比较中日韩英翻译的事实保留、内容完整性、术语遵从与响应时间。其他候选未展示稳定的整体质量优势；Gemma4 QAT 虽然更快，仍有术语不遵从和整句遗漏。完整证据见 [模型对比报告](translation-model-comparison-20261103.md) 和 [离线逐句对照页](translation-model-comparison-20261103.html)。

此决定保留现有生产默认，不表示模型已消除错译。金额、否定关系及跨语言术语污染等已知限制仍需由调用方核对；正常 EOS 也不保证语义完整。对比的 greedy 参数用于评测，生产仍使用下述推荐采样值。

## ASR 权重转换与精度

首次启动时，Rust 读取 Fun-ASR-Nano 官方 PyTorch 检查点和分词器，转换成 ASR 模型目录下的 `Fun-ASR-Nano-2512-native-bf16-v1.gguf`。转换不执行 Python 或 pickle 代码，通过临时文件原子保存，成功后复用本地 GGUF。派生 GGUF 在加载时检查结构，未另行记录其内容哈希；原始下载文件仍保留，因此需要为转换后的副本预留磁盘空间。

转换采用上游 BF16 GGUF 的存储策略：编码器、适配器和解码器的二维矩阵权重使用 BF16；归一化参数、偏置和 FSMN 卷积权重使用 FP32。音频计算的激活保持 FP32，ASR 未改成 INT4。映射、元数据和存储策略对应固定版本的 [transcribe.cpp Fun-ASR-Nano 转换器](https://github.com/handy-computer/transcribe.cpp/blob/v0.4.1/scripts/convert-funasr_nano.py)。原生前端、CUDA 算子和 KV 缓存与 PyTorch 参考实现不同，不保证逐位一致。

ASR 原生库明确选择 CUDA 后端，服务检查实际模型后端，不自动回退 CPU。`--asr-library` 可指定兼容的原生库。Windows 默认路径为：

```text
.tools/transcribe-shared/transcribe.dll
```

发行模式优先查找可执行文件旁的 `runtime/transcribe.dll`，仍可发现旧的 `runtime/asr/transcribe.dll` 入口。Windows 原生库由 [transcribe.cpp 0.4.1 固定源码](https://github.com/handy-computer/transcribe.cpp/tree/4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b) 重编，并链接下述统一 ggml；Linux 安装仍使用官方预编译库。安装脚本固定来源并检查 SHA256：

| 平台 | 来源文件 | SHA256 |
| --- | --- | --- |
| Windows x64 CUDA | 固定提交 `4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b` 的源码归档 | `77bfafa35bd7eee447ec37b832e99c342c62b7db0c6cb18c36e763563b586b1f` |
| Linux x86_64 CUDA | `transcribe-native-0.4.1-linux-x86_64-cuda.tar.gz` | `c4bdf9c4798bac1868ecdcaa04f27eb7712fb43125268313ce14d508c70d247d` |

## Hy-MT2 量化推理与本机服务

项目使用固定版本的 [llama.cpp b11960](https://github.com/ggml-org/llama.cpp/releases/tag/b11960)，直接加载所选来源官方仓库中的 `Hy-MT2-1.8B-Q6_K.gguf`，保留原始量化权重。所有模型层加载到 CUDA；llama.cpp 负责分词、模型计算和采样，Rust 负责请求调度与结果检查，不需要 Python 推理环境。

该文件大小为 `1474785120` 字节，SHA256 为：

```text
d98fe604dec1f28f58f80d7d560f7177e584d3b8e5835862687660e5ff97cb40
```

来源：[固定 revision 的 Hugging Face 文件清单](https://huggingface.co/api/models/tencent/Hy-MT2-1.8B-GGUF/revision/a0c709d9fac510f2c807aa3af52872340dc37a4a?blobs=true)、[固定 revision 的 ModelScope 文件清单](https://modelscope.cn/api/v1/models/Tencent-Hunyuan/Hy-MT2-1.8B-GGUF/repo/files?Revision=e97e6af0ec0950385644d7c0a8e5c899071f722b&Recursive=true)。两个来源的 Q6_K 文件大小与 SHA256 一致。分词器与聊天控制标记使用 GGUF 内的数据。提示基于当前模型的官方翻译模板，追加逐句完整翻译、保留数字、单位、条件和否定关系的提醒；不是逐字使用官方示例，也不使用旧 MiniCPM5 的空思考前缀。该提醒不能保证所有事实都会正确保留，仍需检查实际译文。

生产默认使用官方 1.8B 推荐采样值 `temperature=0.7`、`top_p=0.6`、`top_k=20`、重复惩罚 `1.05`；项目固定 `seed=42`。`min_p=0.0`，presence 与 frequency 惩罚为 `0.0`。模型直接生成译文，不提供思考模式开关；没有译文、未返回正常结束标记或生成耗尽预算时返回错误。官方模板与参数见 [Hy-MT2-1.8B-GGUF 模型卡](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF#inference-and-deployment)。

llama.cpp HTTP 服务由 Rust 管理，绑定 `127.0.0.1` 的临时端口，使用启动时生成的 API 密钥。Windows 通过固定 C ABI 在主进程的原生线程内运行服务，直接构造模型参数，不调用 CLI 入口或接管进程信号；Linux 保留 `llama-server` 子进程。服务加载模型并检查 CUDA 后端后，再开放 StreamTrans 音频接口。原生服务停止后，健康检查将报告后端不可用。该接口仅供 Rust 调度翻译，不作为外部模型 API 分发。

Windows 用 `--llama-library` 指定带项目 C ABI 的原生库。开发模式默认使用 `.tools/llama-embedded/llama-server-impl.dll`，发行模式优先使用可执行文件旁的 `runtime/llama-server-impl.dll`。官方原版或旧构建缺少嵌入入口，不能直接替换；运行安装脚本重新构建。Linux 用 `--llama-server` 指定可执行文件。默认 `--context-size 4096`、`--max-tokens 2048`，输出预算用于译文。当前服务的参数范围不代表原模型完整的上下文能力。

生产推理使用 FP16 KV 缓存，FlashAttention 由 llama.cpp 自动选择。未加载术语表时使用 `cache_prompt=false`；加载术语表时，启动只读入一次完整 canonical TSV，再将所有语言的全部条目按官方推荐的逐行术语句式编码进同一条 user 消息的固定前缀，使用完整中文语言标签。每次请求明确当前目标语言，后续复用完整前缀 token 和 KV，不按当前原文筛词。术语表 token 上限仍按 canonical TSV 计算，实际渲染前缀另查上下文容量，不自动压缩输出预算。部署与容量规则见 [术语表说明](glossary.md)。LLM 本地请求超时或通信失败时，Windows 请求取消并等待原生线程停止，Linux 停止并回收子进程，再返回错误；此时需要重启 StreamTrans。Windows 的取消在可检查的加载或推理阶段生效，不提供对任意 CUDA 内核的强制中断。

指定译法仍由模型理解，不能保证语言隔离或强制遵从。例如日语请求仍可能带入仅为英语指定的 `on-device`；不能把提示格式调整当作这项限制已经解决。

新模型的完整性、术语遵从和速度记录见 [Hy-MT2 复测](hy-mt2-20261103.md)。此前 MiniCPM5 的 FP32 KV、FlashAttention 关闭及思考模式对照属于历史实验，其数字不代表当前模型。

## Windows 统一 ggml 与 CUDA 运行库

Windows 分发以 llama.cpp b11960 固定提交 `8642541b255bfb5dc39cd351a178be09013a55d6` 随附的 ggml 0.28 为唯一源码基础，使用 CUDA Toolkit 12.8.1 构建。transcribe.cpp 0.4.1 固定提交 `4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b` 针对同一套 ggml 头文件和库重编，保留 transcribe 0.4.1 C ABI，以及 Fun-ASR BF16 权重和 FP32 音频激活。

统一 ggml 应用两项 CPU 侧兼容补丁：保留 transcribe 使用的后端注册过滤接口，以及其 CPU 线程池同步语义。CPU 后端关闭 OpenMP，使用原生线程池；CUDA 模型执行要求不变。构建记录同时保存固定提交、兼容补丁标识及各 ggml DLL 的 SHA256。仅标注相同 ggml 版本号不足以证明可以混用，打包时必须核对完整身份和文件集合。

`scripts/setup-cuda-runtime.ps1` 按固定版本和 SHA256 准备运行 DLL 和许可证，供两个后端共用。两个开发后端目录使用同一套版本和下载缓存；Windows 分发目录只保留 `runtime/cudart64_12.dll`、`runtime/cublas64_12.dll`、`runtime/cublasLt64_12.dll` 各一份。原始组件来源见 [NVIDIA CUDA 12.8.1 分发清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json)。llama.cpp 构建要求已安装的 CMake 位于 `PATH`，Ninja 可来自 `PATH` 或 Visual Studio 组件；构建脚本将固定的 CUDA 12.8.1 编译组件下载到项目内 `.tools/llama-build-cuda128/cuda`，无须全局安装 CUDA Toolkit。构建及运行均不使用 Python；运行机器不需要 CUDA Toolkit 或 NVCC。

原预编译 transcribe.cpp 0.4.1 使用 ggml 0.26，与 llama.cpp b11960 的 ggml 0.28 不能直接互换；单独复制 `ggml-cuda.dll` 会留下其他 ggml ABI 差异。当前方案先按同源头文件重编 ASR，再共享完整的 `ggml.dll`、`ggml-base.dll`、`ggml-cuda.dll` 和 9 个 CPU 变体。ASR、`llama-server-impl.dll` 及其依赖、ggml、CUDA、FFmpeg、MSVC DLL 全部平铺到 `runtime` 根目录，分发包不携带 `llama-server.exe`。升级时使用完整的新发行目录，不要手工混入旧版 ASR 或 ggml DLL。目录布局与来源记录见 [分发说明](distribution.md#目录布局)，此前共享 ggml 构建的验证见 [共享 ggml 验证](shared-ggml-20261107.md)，其中的双进程记录不代表当前进程内运行方式。

## FSMN-VAD 分段

VAD 使用所选来源的原始 FSMN 权重和 CMVN，按固定模型配置由 Rust 实现音频特征、FSMN 网络和语音边界状态机，使用 CPU FP32。断句不依赖 Python，也不占用大模型的 GPU 显存。

服务端在 VAD 和 ASR 之前按 20 ms 帧计算 RMS 音量，低于 `--min-volume-db` 门限（默认 -30 dBFS）的帧置为静音，等于或高于门限的帧保持原样；最后不足 20 ms 的帧按实际采样数计算。此门限适用于文件上传和实时流，dBFS 以归一化 PCM 的满幅为基准，不代表麦克风的声压级。过滤不删除采样，因此音频时长和时间戳保持不变；低于门限的人声同样会被过滤。

`--min-volume-db` 接受不大于 `0` 的有限数值。数值越接近 `0`，过滤越强；设置为更小的数值（例如 `--min-volume-db -40`）可保留更多安静声音。

VAD 检测语音活动边界，不保证每段都是完整语句。原生浮点计算与参考实现的边界可能有小幅差异，应根据实际音频评估；接口继续使用毫秒时间戳，单段上限为 15 秒。

## Windows 构建与运行

构建需要：

- Windows x64、Rust stable、Visual Studio 2022 C++ Build Tools，安装「使用 C++ 的桌面开发」组件。
- Git 加入 `PATH`，用于应用 ggml 兼容补丁。
- 构建 llama.cpp 需事先安装 CMake 并加入 `PATH`；Ninja 可来自 `PATH` 或 Visual Studio 组件。脚本将 CUDA 12.8.1 编译组件下载到项目目录，无须全局安装；单独重新编译 Rust 服务不使用 NVCC。
- 运行需要支持原生 BF16 的 NVIDIA GPU，以及兼容随包 CUDA 运行库的 NVIDIA 驱动。
- 首次安装可访问原生程序上游下载站点和 crates.io；首次模型下载可访问所选的 Hugging Face 或 ModelScope。

安装命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Gpu 0
```

脚本下载并校验固定版本的 llama.cpp、transcribe.cpp 源码、CUDA 12.8 组件和 FFmpeg 9.0 LGPL 共享库，依次构建带兼容补丁的 llama.cpp/ggml、链接同源 ggml 的 ASR、音频桥，最后执行 `cargo build --release`。Windows 文件上传直接通过 DLL 解码和重采样；`--ffmpeg-library` 可覆盖默认 `.tools/ffmpeg-shared/bin/streamtrans-ffmpeg.dll` 路径，四个配套 FFmpeg DLL 应在同一目录。桥接 DLL 使用同包头文件和导入库构建，Rust 通过现有 `libloading` 调用，不需要 Python、pkg-config 或 bindgen。Candle 仅用于 CPU VAD 和 ASR 权重转换，不启用 Candle CUDA；CUDA 内核由 llama.cpp 原生构建过程编译。开发用原生程序与运行库放入项目内 `.tools`；分发目录只携带运行依赖，不携带编译工具。

`scripts/setup-llama.ps1` 调用 `scripts/build-llama.ps1` 构建固定源码、ggml 兼容补丁及进程内服务入口，并校验共享 CUDA 运行库，默认输出到 `.tools/llama-embedded`。`scripts/build-asr.ps1` 读取该目录的运行库清单，复用同一次 llama 构建缓存中的 ggml 头文件和导入库，将 ASR 输出到 `.tools/transcribe-shared`；可用 `-LlamaDirectory` 选择配套 llama 构建。两个构建脚本均可用 `-OutputDirectory` 覆盖输出目录，`-Jobs` 控制并行构建数，默认 `4`。开发模式的 ASR 默认路径为 `.tools/transcribe-shared/transcribe.dll`，发行模式优先查找 `runtime/transcribe.dll`，仍可发现旧 `runtime/asr/transcribe.dll` 路径。Windows 使用自定义目录时，通过 `--llama-library` 与 `--asr-library` 显式指定入口。

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
