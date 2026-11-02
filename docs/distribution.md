# Windows 分发与部署

分发目录包含 StreamTrans、ASR 原生库、llama.cpp、CUDA 运行库、FFmpeg 和第三方许可证，不包含模型权重、模型转换缓存、Python 环境或 CUDA 编译工具。运行机器需要 Windows x64、支持原生 BF16 的 NVIDIA GPU，以及兼容随包 CUDA 运行库的 NVIDIA 驱动。

最终翻译模型选定为 **Hy-MT2-1.8B GGUF Q6_K**（2026-11-03），首次启动从 ModelScope 下载固定版本。模型文件、revision 与选型依据见 [模型说明](models.md#翻译模型最终选型)。

## 制作分发目录

构建机器需要 Rust stable、Visual Studio 2022 C++ Build Tools（「使用 C++ 的桌面开发」组件）和可访问上游下载站点的网络。Rust 构建不需要 CUDA Toolkit 或 NVCC。

1. 在项目根目录准备原生程序并构建服务。

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
   ```

2. 准备允许再分发的 FFmpeg Windows x64 静态构建，同时保留对应的许可证和源码获取信息。打包脚本只复制 `ffmpeg.exe`，拒绝带有 `--enable-shared` 的共享构建。FFmpeg 的许可取决于构建选项；打包脚本同时拒绝 `--enable-nonfree` 构建。不要直接复制来源和许可不明的本机 FFmpeg。

3. 运行打包脚本。默认从可发现的位置获取 FFmpeg 和许可信息；需要覆盖时使用下表中的参数。

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/dist.ps1
   ```

项目内已准备 Gyan FFmpeg 9.0.2 essentials 构建时，可显式指定程序及其上级许可目录：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dist.ps1 `
  -Ffmpeg .tools/ffmpeg/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe `
  -FfmpegLicenseDirectory .tools/ffmpeg/ffmpeg-9.0.2-essentials_build
```

该示例要求目录中同时保留构建来源记录；若未能自动发现源码地址，再通过 `-FfmpegSourceUrl` 指定与二进制对应的源码获取地址。

脚本默认生成 `dist/StreamTrans`，输出完整目录，不自动生成 ZIP。输出目录必须不存在或为空，脚本不会覆盖非空目录。打包过程不下载模型，也不复制现有 `models` 缓存。脚本失败时，根据错误补充所需的原生程序、FFmpeg 或许可材料；已有部分输出时，使用新的 `-OutputDirectory` 重新打包，不要分发未完成的目录。

| 参数 | 用途 |
| --- | --- |
| `-OutputDirectory` | 修改输出目录，默认 `dist/StreamTrans` |
| `-SkipBuild` | 使用已经构建的 Rust 发布程序 |
| `-Ffmpeg` | 指定允许再分发的静态 `ffmpeg.exe` |
| `-FfmpegLicenseDirectory` | 指定该 FFmpeg 构建的许可证目录 |
| `-FfmpegSourceUrl` | 指定该 FFmpeg 构建对应的源码获取地址 |
| `-MsvcRedistDirectory` | 指定需要随包携带的 MSVC 运行库目录 |

FFmpeg 许可证和源码地址必须对应实际打包的二进制。分发时应一并保留适用的许可、版权声明和源码获取说明；打包脚本不替代上游授权。

## 目录布局

```text
dist/StreamTrans/
├── streamtrans.exe
├── start.cmd
├── scripts/test-stream.ps1  # 无 Python 的连续音频测试客户端
├── manifest.json
├── THIRD_PARTY.json
├── runtime/
│   ├── asr/       # transcribe.dll、依赖库与所需 CUDA DLL
│   ├── llama/     # llama-server.exe、依赖库与所需 CUDA DLL
│   └── ffmpeg/    # 静态 ffmpeg.exe
└── licenses/
    ├── asr/
    ├── llama/
    ├── cuda-asr/
    ├── cuda-llama/
    ├── msvc/
    └── ffmpeg/
```

ASR 和 llama.cpp 的原生库分目录存放，避免不同版本的 ggml 或 CUDA DLL 混用。不要仅复制 `streamtrans.exe`，也不要将两个原生目录中的同名 DLL 合并到一起。`manifest.json` 记录分发文件及其 SHA256，`THIRD_PARTY.json` 记录原生程序来源和版本；模型版本与校验由运行时模型缓存清单管理。

## 在运行机器启动

1. 将整个 `StreamTrans` 目录复制到运行机器的可写位置。
2. 确认 NVIDIA 驱动已安装，GPU 满足 BF16 和显存要求。
3. 运行 `start.cmd`，或直接运行 `streamtrans.exe`。

程序自动查找可执行文件旁的 `runtime/asr/transcribe.dll`、`runtime/llama/llama-server.exe` 和 `runtime/ffmpeg/ffmpeg.exe`。`--asr-library`、`--llama-server` 和 `--ffmpeg` 可显式覆盖这些路径。

发行目录的 `examples/glossary.tsv` 是术语表模板，可用 `start.cmd --glossary .\examples\glossary.tsv` 加载。业务术语表由部署者另外提供，打包脚本只复制示例，不收集 `--glossary` 指向的私有文件。相对路径按工作目录解析，跨目录启动时宜使用绝对路径；修改表后重启服务。完整容量和前缀缓存说明见 [术语表](glossary.md)。

首次启动从 ModelScope 下载模型，默认缓存到可执行文件旁的 `models`，随后用 Rust 转换 ASR 权重。该目录需要足够的磁盘空间和写入权限。模型加载和 CUDA 检查完成后，音频接口才开始监听 `127.0.0.1:3000`。音频请求和响应格式见 [README](../README.md#音频接口)。

托管的 `llama-server` 使用单独的回环地址临时端口和每次启动生成的 API 密钥。调用方只访问 StreamTrans 音频接口，无须配置、启动或直接连接这个子进程。StreamTrans 退出时会清理托管子进程。

## 模型缓存与离线部署

分发目录不包含模型。可在联网机器上仅下载模型：

```powershell
.\streamtrans.exe --download-only
```

下载完成后，将整个 `models` 缓存目录复制到目标机器的程序目录旁，保留缓存清单和原有目录结构。运行时会校验缓存文件，完整有效的缓存可离线使用。首次运行仍可能需要生成 ASR 转换缓存；转换由 Rust 完成，不调用 Python。也可用 `--cache-dir` 指向已有的可写缓存目录。

## 验证分发目录

`streamtrans.exe --help` 可检查主程序是否能启动，但不会加载 GPU 模型。完整验证还需要首次下载、ASR 转换、两个 GPU 后端加载，以及真实音频请求。启动后通过 `GET /health` 检查后端状态，并按 README 上传一段有效音频。

性能结果应记录运行机器、驱动、音频、目标语言、进程首请求与后续请求，并将模型下载和加载时间单独列出。后端切换前的 [Candle 对照数据](sampling-benchmark.md) 不能作为当前分发包的 HTTP 性能结果。

Linux 安装脚本下载官方预编译 CUDA 后端，但 Linux 分发包、Ubuntu 系统库兼容性和真实推理流程尚未实机验证；当前 `scripts/dist.ps1` 面向 Windows x64。

## 验证记录

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
