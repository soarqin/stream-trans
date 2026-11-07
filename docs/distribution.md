# Windows 分发与部署

分发目录包含 StreamTrans、ASR 原生库、llama.cpp、一套共享 CUDA 12.8 运行库、FFmpeg 共享库和第三方许可证，不包含模型权重、模型转换缓存、Python 环境或 CUDA 编译工具。运行机器需要 Windows x64、支持原生 BF16 的 NVIDIA GPU，以及兼容随包 CUDA 运行库的 NVIDIA 驱动。

最终翻译模型选定为 **Hy-MT2-1.8B GGUF Q6_K**（2026-11-03），首次启动默认从 Hugging Face 下载固定版本，也可通过 `--model-source modelscope` 使用 ModelScope。模型文件、各来源的 revision 与选型依据见 [模型说明](models.md#翻译模型最终选型)。

## 制作分发目录

构建机器需要 Rust stable、Visual Studio 2022 C++ Build Tools（「使用 C++ 的桌面开发」组件）和可访问上游下载站点的网络。构建 llama.cpp 前需安装 CMake 并将其加入 `PATH`；Ninja 可从 `PATH` 获取，也可复用 Visual Studio 随带的组件。构建脚本将固定的 CUDA 12.8.1 编译组件下载到项目内 `.tools/llama-build-cuda128/cuda`，无须全局安装 CUDA Toolkit。单独重新编译 Rust 服务不使用 NVCC；构建与运行均不需要 Python。

Windows 安装脚本通过 `scripts/setup-cuda-runtime.ps1` 为两个后端准备相同的 CUDART 12.8.90、cuBLAS 12.8.4.1 及许可文件，并共用下载缓存。ASR 保留 transcribe.cpp v0.4.1 的 CUDA 12.8 编译产物；llama.cpp b11960 从固定提交 `8642541b255bfb5dc39cd351a178be09013a55d6` 使用 CUDA Toolkit 12.8.1 构建。两个后端的编译和运行均采用 CUDA 12.8。CMake、Ninja、CUDA Toolkit 和 NVCC 只用于构建机器，不进入分发目录。

1. 在项目根目录准备原生程序并构建服务。

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
   ```

2. 安装脚本会调用 `scripts/setup-ffmpeg.ps1`，下载并校验固定版本的 BtbN FFmpeg 9.0 LGPL 共享构建，使用同包头文件和导入库编译 `native/streamtrans_ffmpeg.c`。小型桥接 DLL 将解封装、解码和重采样封装为固定 C ABI，Rust 使用现有 `libloading` 加载。头文件、导入库、编译器和 FFmpeg CLI 均不进入分发包。

3. 运行打包脚本。默认使用项目内已经准备好的 FFmpeg DLL 及来源清单；需要覆盖时使用下表中的参数。

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/dist.ps1
   ```

单独准备 FFmpeg，或指定已经构建的共享库目录：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-ffmpeg.ps1
powershell -ExecutionPolicy Bypass -File scripts/dist.ps1 -FfmpegDirectory .tools/ffmpeg-shared
```

共享库目录需保留脚本生成的 `.streamtrans-ffmpeg.json`、许可正文和源码获取说明。打包脚本核对构建来源及文件 SHA256，不接收任意 DLL 或原来的 EXE 包。旧 `-Ffmpeg`、`-FfmpegLicenseDirectory` 和 `-FfmpegSourceUrl` 参数会返回迁移提示。

脚本默认生成 `dist/StreamTrans`，输出完整目录，不自动生成 ZIP。输出目录必须不存在或为空，脚本不会覆盖非空目录。打包过程不下载模型，也不复制现有 `models` 缓存。脚本失败时，根据错误补充所需的原生程序、FFmpeg 或许可材料；已有部分输出时，使用新的 `-OutputDirectory` 重新打包，不要分发未完成的目录。

| 参数 | 用途 |
| --- | --- |
| `-OutputDirectory` | 修改输出目录，默认 `dist/StreamTrans` |
| `-SkipBuild` | 使用已经构建的 Rust 发布程序 |
| `-FfmpegDirectory` | 指定包含 `bin` 和构建来源清单的 FFmpeg 共享库目录，默认 `.tools/ffmpeg-shared` |
| `-MsvcRedistDirectory` | 指定需要随包携带的 MSVC 运行库目录 |
| `-LlamaDirectory` | 指定固定 b11960 / CUDA 12.8 构建目录，默认 `.tools/llama-cuda128` |

FFmpeg 许可证和源码地址必须对应实际打包的二进制。分发时应一并保留适用的许可、版权声明和源码获取说明；打包脚本不替代上游授权。

## 目录布局

```text
dist/StreamTrans/
├── streamtrans.exe
├── vcruntime140.dll       # Rust 主程序的启动依赖
├── start.cmd
├── scripts/test-stream.ps1  # 无 Python 的连续音频测试客户端
├── manifest.json
├── THIRD_PARTY.json
├── runtime/
│   ├── cudart64_12.dll    # 共享 CUDART 12.8.90
│   ├── cublas64_12.dll    # 共享 cuBLAS 12.8.4.1
│   ├── cublasLt64_12.dll
│   ├── streamtrans-ffmpeg.dll # 固定 ABI 的音频桥
│   ├── avformat-63.dll
│   ├── avcodec-63.dll
│   ├── avutil-61.dll
│   ├── swresample-7.dll
│   ├── msvcp140.dll
│   ├── vcruntime140.dll
│   ├── vcruntime140_1.dll
│   ├── asr/              # transcribe.dll 与独立的 ASR ggml 后端
│   └── llama/            # llama-server.exe 与独立的 llama ggml、服务端依赖
└── licenses/
    ├── asr/
    ├── llama/
    ├── cuda/
    ├── msvc/
    └── ffmpeg/
```

CUDA、FFmpeg 和共享 MSVC DLL 放在 `runtime` 根目录；主程序目录额外保留一份 `vcruntime140.dll`，使 Rust 程序能够先启动。两个后端子目录不再各自携带 CUDA 和 MSVC 副本。打包只保留实际运行依赖，不包含 ASR 的可选 Vulkan 后端及 llama.cpp 的独立 CLI、基准和 RPC 工具组件。

两套 ggml 使用各自的 ABI，ASR 还依赖其配套 ggml 的 `ggml_backend_set_reg_filter` 等入口点。统一 CUDA 版本后仍将后端保留在 `runtime/asr`、`runtime/llama`，不能将两个目录中的同名 ggml DLL 平铺覆盖。此布局用于新生成的分发目录；打包脚本不会清理或覆盖已有分发目录。

`manifest.json` 记录分发文件及其 SHA256；`THIRD_PARTY.json` 记录 llama.cpp 的固定源码提交、CUDA 编译版本和运行版本，并在 `shared_cuda` 中记录共享 CUDA 组件的版本、来源及校验值。模型版本与校验由运行时模型缓存清单管理。

## 在运行机器启动

1. 将整个 `StreamTrans` 目录复制到运行机器的可写位置。
2. 确认 NVIDIA 驱动已安装，GPU 满足 BF16 和显存要求。
3. 运行 `start.cmd`，或直接运行 `streamtrans.exe`。

程序自动查找可执行文件旁的 `runtime/asr/transcribe.dll`、`runtime/llama/llama-server.exe` 和 `runtime/streamtrans-ffmpeg.dll`。`--asr-library`、`--llama-server` 和 `--ffmpeg-library` 可显式覆盖这些路径。Windows 服务不再接受旧 `--ffmpeg` EXE 参数；升级时应复制完整新发行目录。

FFmpeg DLL 在启动时加载并保留至进程退出，启动检查和文件上传均不创建 FFmpeg 子进程。文件解码使用首个音轨，转换为单声道 16 kHz PCM，保留 32 MiB 上传、配置时长及 30 秒解码时限；逐帧检查容量与截止时间，结束时冲刷解码器和重采样器。`elapsed_ms.audio_decode` 单独报告文件解码和规范化耗时。原生解码使用中断回调和循环检查；该时限不是可强制杀死任意原生函数的进程隔离。实时录音路径直接处理客户端发来的 PCM，不经过 FFmpeg。

ASR 注册自己的后端目录及共享 `runtime` 目录，llama-server 的子进程搜索路径包含后端目录和共享目录。部署时需复制完整分发目录，无须依赖系统 `PATH` 中另行安装的 CUDA 或 FFmpeg。

发行目录的 `examples/glossary.tsv` 是术语表模板，可用 `start.cmd --glossary .\examples\glossary.tsv` 加载。业务术语表由部署者另外提供，打包脚本只复制示例，不收集 `--glossary` 指向的私有文件。相对路径按工作目录解析，跨目录启动时宜使用绝对路径；修改表后重启服务。完整容量和前缀缓存说明见 [术语表](glossary.md)。

首次启动默认从 Hugging Face 下载模型，随后用 Rust 转换 ASR 权重。需要使用 ModelScope 时，运行 `start.cmd --model-source modelscope` 或 `streamtrans.exe --model-source modelscope`。该选项统一用于 ASR、VAD 和翻译模型，不改变原生运行库或 FFmpeg 的来源，下载失败时也不会自动切换来源。

缓存根目录默认是可执行文件旁的 `models`，需要足够的磁盘空间和写入权限。不同来源的仓库版本及缓存清单分开管理；已有 ModelScope 缓存需选择 `--model-source modelscope` 才会复用。模型加载和 CUDA 检查完成后，音频接口才开始监听，默认地址为 `127.0.0.1:3000`。音频请求和响应格式见 [README](../README.md#音频接口)。

需要让其他机器直连时，设置请求 key 后显式绑定非回环地址。例如：

```powershell
$env:STREAMTRANS_REQUEST_KEY = 'your-key'
.\start.cmd --bind 0.0.0.0:3000

# 也可使用 --request-key your-key；IPv6 监听可写 --bind [::]:3000
```

未设置 key 时，只允许绑定回环地址。key 必须为 1–256 个可见 ASCII 字符，不能包含空白或为空；所有 HTTP 请求（包括 `GET /health`）和 WebSocket 握手都需携带单个 `Authorization: Bearer <key>`。缺失、错误或重复认证头返回 `401 unauthorized`，并带有 `WWW-Authenticate: Bearer`；不支持 URL 查询参数传 key。Rust 客户端可在 GUI 中填写，或通过 `--request-key` / `STREAMTRANS_REQUEST_KEY` 配置。

请求 key 不提供传输加密；跨不可信网络时使用 HTTPS/WSS 反向代理，并保留认证头。客户端网页输出的本机回环限制不受此服务端选项影响。

托管的 `llama-server` 使用单独的回环地址临时端口和每次启动生成的 API 密钥。调用方只访问 StreamTrans 音频接口，无须配置、启动或直接连接这个子进程。StreamTrans 退出时会清理托管子进程。

## 模型缓存与离线部署

分发目录不包含模型。可在联网机器上仅下载模型：

```powershell
.\streamtrans.exe --download-only

# 通过 ModelScope 准备离线缓存
.\streamtrans.exe --model-source modelscope --download-only
```

下载完成后，将整个 `models` 缓存目录复制到目标机器的程序目录旁，保留缓存清单和原有目录结构。离线启动时选择与下载时一致的 `--model-source`；使用版本覆盖参数时也需保持一致。运行时会校验缓存文件，完整有效的缓存可离线使用。首次运行仍可能需要生成 ASR 转换缓存；转换由 Rust 完成，不调用 Python。也可用 `--cache-dir` 指向已有的可写缓存目录。

## 验证分发目录

`streamtrans.exe --help` 可检查主程序是否能启动，但不会加载 GPU 模型。完整验证还需要首次下载、ASR 转换、两个 GPU 后端加载，以及真实音频请求。启动后通过 `GET /health` 检查后端状态，并按 README 上传一段有效音频。

共享 CUDA 布局还需核对实际进程模块：父进程与 llama-server 应从同一个 `runtime` 根目录加载三份 CUDA DLL，各自的 ggml 应来自对应后端子目录。静态导入符号齐全不能替代这项加载路径检查和真实 GPU 请求。

性能结果应记录运行机器、驱动、音频、目标语言、进程首请求与后续请求，并将模型下载和加载时间单独列出。后端切换前的 [Candle 对照数据](sampling-benchmark.md) 不能作为当前分发包的 HTTP 性能结果。

Linux 安装脚本下载官方预编译 CUDA 后端，但 Linux 分发包、Ubuntu 系统库兼容性和真实推理流程尚未实机验证；当前 `scripts/dist.ps1` 面向 Windows x64。

## 验证记录

2026-11-07 的共享 CUDA 与进程内 FFmpeg DLL 发行验证见 [共享运行库验证](shared-runtime-20261107.md)，包含约 44.3% 的目录缩减、音频逐采样对照、真实 GPU 请求及模块来源。下述段落是早期分目录运行库与 FFmpeg EXE 的历史记录。

本节保留切回 llama.cpp 时初版发行目录的验证结果。新增连续录音接口后的验证见 [实时接口说明](streaming.md#验证记录)；当前文件数量与校验值以 `manifest.json` 为准。

Windows x64 分发目录已在 RTX 3050（6 GB）上完成验证，目录包含 96 个文件，约 2.214 GiB，不包含模型。测试从项目外的临时目录启动，`PATH` 仅保留 `System32`；程序自动查找 `runtime`，仅显式指定已有模型缓存和测试监听地址。从缓存 SHA256 校验到接口监听耗时 5.429 秒，未重新下载模型或转换 ASR 权重。

18.243 秒中文音频的首个日语翻译请求返回 `200`，响应中的 VAD+ASR、翻译和处理总耗时分别为 979、3916 和 4939 ms。随后上传 9.121 秒中文音频并翻译为英语，返回 `200`，三项耗时分别为 345、1056 和 1438 ms。这是同一服务进程中的两次单次请求，不作中位数统计。

模块检查确认：父进程的 ggml、cudart 和 cuBLAS 来自 `runtime/asr`，llama.cpp 子进程的对应模块来自 `runtime/llama`。两个后端没有借用开发目录或系统 `PATH` 中的这些运行库。正常 `Ctrl+C` 后，主进程与托管子进程均退出。

重新启动并将推理超时设为 1 秒后，音频接口返回 `502 translation_failed`，响应保留 `recognition`。返回前 llama.cpp 子进程已停止，随后健康检查返回 `503`；测试完成后主服务也已停止。

原始验证材料保留在开发工作区的 `.tools/benchmark`，不放入无模型分发目录：

| 文件 | 内容 |
| --- | --- |
| `restored-dist-http-ja.json`、`restored-dist-http-en.json` | 真实音频接口响应 |
| `restored-dist-server.log` | 启动和推理日志 |
| `restored-dist-receipt.json` | 最小 `PATH`、模块来源和运行检查 |
| `restored-dist-shutdown.json` | 正常退出后的进程状态 |
| `restored-dist-timeout-receipt.json` | 超时响应、子进程回收和健康检查 |
