# StreamTrans 桌面录音客户端

客户端采集麦克风音频，转换为 16 kHz、单声道 PCM，连续发送到 StreamTrans 服务端。服务端负责 FSMN-VAD 断句、Fun-ASR-Nano-2512 语音识别和可选 Hy-MT2-1.8B Q6_K 翻译。客户端无需安装 Python、Node.js、FFmpeg、CUDA 或模型，也不需要 NVIDIA GPU；运行模型的机器仍需满足服务端的 GPU 要求。

术语表由服务端通过 `--glossary` 加载，客户端不用上传或重复发送词表，直接使用该服务即可。配置方式见源码项目中的 `docs/glossary.md`；显示原文不受术语表影响。

Windows 客户端与模型服务可运行在同一台或不同的机器上。发行目录中的 `streamtrans-client.exe` 是桌面客户端，不会自行启动模型服务。

## 开始录音

1. 先启动 StreamTrans 服务，等待模型加载完成、接口开始监听。默认 API 地址为 `http://127.0.0.1:3000`。
2. 启动 `streamtrans-client.exe`，填写「API 地址」；服务端设置了请求 key 时，填写同一「请求 key」，再选择录音输入设备。
3. 选择目标语言。选择「不翻译」时仅识别原文；选择英语、日语等语言时，先显示原文，再显示同一段的译文。
4. 点击「开始语音识别」（仅识别模式）或「开始自动翻译」（翻译模式），连续讲话即可。客户端不检测静音或自行断句，服务端会在停顿或达到分段上限后返回结果。
5. 点击「停止录音」后，等待状态变为已完成。客户端会提交剩余采样和 `finish`，保留连接以接收末段结果；直接关闭程序不能替代正常停止。

服务端在独立的 CPU VAD 任务中持续断句，再将语音段交给 GPU 依次识别和翻译。前一段正在翻译时，客户端仍会继续采集和发送后续音频；已经断句不表示该段已经完成 GPU 处理。

「显示原文」和「显示译文」控制输出框及网页中的显示内容；隐藏译文不会停止服务端翻译。需要跳过翻译推理时，录音前将目标语言设为「不翻译」。仅识别模式始终显示原文。

输出框只读，可用「复制文本」复制当前显示内容，或用「清空结果」清除已显示结果。每次新会话会重新开始显示，客户端仅保留最近 500 段、合计最多 2 MiB 的文本，超出后移除最早的结果；它不提供完整录音归档。

仅识别模式使用 `/v1/audio/transcriptions/stream`，不会调用翻译推理；翻译模式使用 `/v1/audio/translations/stream`。当前服务端启动时仍预加载 ASR 和 LLM，切换为仅识别不会释放 LLM 显存。两种模式与文件上传共用同一个处理许可；收到 `429` 时，等待已有会话结束后再开始。

当前翻译使用 Hy-MT2 的翻译模板直接生成译文，不提供思考模式开关。模型下载、采样和上下文由服务端管理，切换模型后客户端无需安装新模型或修改音频协议。译文完整性、术语遵从和速度结果见源码项目中的 `docs/hy-mt2-20261103.md`。

实时会话默认不限总音频时长，与文件上传默认 60 秒的限制独立。服务端原始入站缓冲仍限制为 12.8 秒，闭合语音段的待推理队列限制为 120 秒 PCM 或 64 段，包含正在执行的语音段。两级队列都有上限；连续录音不表示处理速度不足时可以无限积压。连续 30 秒没有有效音频数据时，服务端会结束连接。

## 网页输出

客户端同时提供本机输出页，默认地址为 **http://127.0.0.1:3001/**。主界面显示实际地址，可点击打开浏览器，也可将其作为本机 OBS 浏览器源。

网页每 500 ms 获取一次当前状态，显示与桌面一致的原文和译文选择。网页刷新频率不代表模型的识别延迟；结果仍按服务端已结束的语音段到达。网页只负责展示，录音的开始、停止和清空在桌面端操作。

### 只显示最后一句

在网页地址中添加 `mode=latest`，只显示当前会话最新语音段的原文和译文，默认在最后一次内容更新的 5 秒后隐藏。可通过 `timeout` 覆盖保留秒数，例如：

```text
http://127.0.0.1:3001/?mode=latest
http://127.0.0.1:3001/?mode=latest&timeout=8
```

`timeout` 接受大于 0、至多 86400 的有限秒数，支持小数；省略或无效值使用 5 秒。参数仅影响当前网页，可分别用于不同的浏览器或 OBS 浏览器源；普通地址继续显示历史结果。

「最后一句」指服务端 VAD 切出的最后一段，不再按标点拆句。同一段的译文随后到达或文本被修正时重新计时；重复轮询、重复内容、状态和样式更新均不延长时间。过期后刷新页面也不会重新显示，较早语音段的迟到译文不会顶替最新段。仅识别和原文、译文显示选项继续生效；最新段没有可显示内容时保持空白，不回退旧段。

该模式隐藏常驻状态文字，错误信息仍显示。超时只清空网页中的语音段显示，不删除客户端历史或 `/api/state` 中的结果，桌面输出框仍显示历史。输出服务断开时，当前段也会按已设定的期限隐藏。

### 输出接口

| 路径 | 用途 |
| --- | --- |
| `/` | 输出页，包含状态、逐段原文、译文和错误信息 |
| `/api/state` | 只读 JSON 状态，包括 `status`、`mode`、显示选项、`rows` 和 `error` |
| `/style.css` | 每次请求读取自定义 CSS 文件；设置 `Cache-Control: no-store` |
| `/style-template.css` | 内置的默认 CSS 模板，用于参考或恢复 |

`rows` 中每段包含 `id`、`start_ms`、`end_ms`、`text`、可为空的 `translation`，以及 `updated_at_ms`。`start_ms`、`end_ms` 以本次录音首个采样为起点，单位为毫秒；`updated_at_ms` 是客户端最后收到该段文本内容变更时的 Unix 毫秒时间戳，用于网页过期判断，不代表说话结束时间。显示选项仅影响呈现，`/api/state` 仍包含已收到的原文和译文。输出页仅允许监听本机回环地址，未提供远程访问或录音控制接口。

## 修改网页样式

默认样式文件为客户端可执行文件旁的 `output.css`。文件不存在时，客户端创建默认模板；文件存在时保留原内容。建议将发行目录放在当前用户可写的位置。

使用 UTF-8 文本编辑器修改并保存 `output.css`，网页会自动重新读取，无需重启客户端。可修改字体、字号、行距、背景、原文和译文的样式。例如在文件末尾添加：

```css
body { font-size: 30px; line-height: 1.6; }
.original { opacity: 0.75; }
.translation { font-weight: 700; }
.timestamp { display: block; }
```

常用选择器为 `.output`、`.output-row`、`.original`、`.translation`、`.timestamp`、`.status` 和 `.error`。默认页面背景透明，适用于 OBS；需要底色时可设置 `body { background: #202124; color: white; }`。模板保留 `[hidden] { display: none !important; }`，修改时请保留此规则，以便显示选项正常生效。

CSS 文件暂时不可读或被删除时，网页回退到内置模板。若要恢复默认文件，可保存 `/style-template.css` 的内容到 `output.css`。样式仅影响网页，不改变桌面界面、识别结果或模型参数。

## 连接其他机器的服务

「API 地址」填写服务端入口，例如通过端口转发得到的 `http://127.0.0.1:13000`；不要填写网页输出端口 `3001` 或完整 WebSocket 路径。客户端根据目标语言自动选择识别或翻译接口。

服务端默认监听回环地址。跨机器直连时，先在服务端配置 `--request-key`（或 `STREAMTRANS_REQUEST_KEY`），并显式设置 `--bind 0.0.0.0:3000` 或 `--bind [::]:3000`；随后在客户端填写可达的服务端 API 地址和相同请求 key。未设置 key 的服务端只允许回环监听，也可经已有的安全端口转发连接。

客户端通过 WebSocket 握手的 `Authorization: Bearer <key>` 请求头发送 key，不把 key 放入 URL、网页输出或 `/api/state`。GUI 的请求 key 输入框以密码形式显示，录音期间不能修改；CLI 可用 `--request-key` 或 `STREAMTRANS_REQUEST_KEY`，同样适用于无界面的 WAV 回放。key 必须为 1–256 个可见 ASCII 字符，不能包含空白。服务端未设置 key 时，客户端可留空。

请求 key 用于认证，不提供传输加密。跨不可信网络时使用 HTTPS/WSS 反向代理；代理需支持 WebSocket、持续音频传输并保留 `Authorization` 请求头。客户端网页输出继续只监听本机回环地址，不接受远程录音控制。

```powershell
$env:STREAMTRANS_REQUEST_KEY = 'your-key'
.\streamtrans-client.exe --api http://192.168.1.10:3000

# 也可直接指定参数
.\streamtrans-client.exe --api http://192.168.1.10:3000 --request-key your-key
```

## 启动参数与文件回放

```powershell
.\streamtrans-client.exe --api http://127.0.0.1:13000 --target none `
  --web-bind 127.0.0.1:3002 --css C:\StreamTrans\output.css
```

| 参数 | 说明 |
| --- | --- |
| `--api` | 服务端基础地址，默认 `http://127.0.0.1:3000` |
| `--request-key` | 可选的服务端请求 key；也可从 `STREAMTRANS_REQUEST_KEY` 读取，发送为 Bearer 认证头 |
| `--target` | 初始目标语言，默认 `en`；`none` 为仅识别 |
| `--web-bind` | 客户端输出页监听地址，默认 `127.0.0.1:3001`，必须为本机回环地址 |
| `--css` | 自定义 CSS 文件路径；未指定时使用可执行文件旁的 `output.css` |
| `--list-devices` | 枚举录音输入设备，不连接模型服务 |
| `--headless` | 不打开桌面窗口，启动本机输出页 |
| `--wav` | 仅与 `--headless` 合用，通过同一网络管线按实时速率回放 WAV，无需麦克风 |

文件回放用于检查连接和逐段输出。输入必须已经是 **16 kHz、单声道、PCM16 WAV**；客户端不调用 FFmpeg。目标设为 `none` 可检查仅识别：

```powershell
.\streamtrans-client.exe --headless --wav C:\Audio\sample.wav --target none
```

回放会发送最后的 `finish` 并等待 `done`，完成后继续保留网页输出。它不生成 ASR 中间假设或逐 token 译文。仅运行 `--headless` 而不指定 WAV 时，只启动输出页，不打开麦克风。

## 从源码构建与打包

以下命令在源码项目根目录运行。构建需要 Rust stable x64 MSVC 工具链和 Visual Studio 2022 C++ Build Tools，不运行服务端的安装脚本，也不下载模型或 GPU 运行库：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-client.ps1
.\target\release\streamtrans-client.exe
```

测试、Clippy 与格式检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-client.ps1 -Action test
powershell -ExecutionPolicy Bypass -File scripts/build-client.ps1 -Action clippy
cargo fmt --all --check
```

创建独立客户端发行目录：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dist-client.ps1
.\dist\StreamTransClient\streamtrans-client.exe
```

默认输出为 `dist/StreamTransClient`，包含客户端、可编辑的 `output.css`、本说明、所需的 `vcruntime140.dll` 和许可证。中文界面嵌入官方 Noto Sans CJK SC Regular 静态 OTF 字体，许可证为 `licenses/font-OFL.txt`，来源与 SHA-256 见 `licenses/font-source.json`；网页使用浏览器可用的系统字体。发行包不包含服务端、模型权重、FFmpeg 或 GPU 运行库，运行机器不需要 Rust 或 C++ 编译器。

打包时自动查找 Visual Studio 的 x64 C++ Redist；找不到时，可用 `-MsvcRedistDirectory` 指定含 `vcruntime140.dll` 的 x64 `Microsoft.VC*.CRT` 目录。Microsoft Runtime 许可证优先复用已校验缓存，缺失时从微软下载并校验，不调用服务端安装脚本。

打包脚本只接受不存在或为空的输出目录，不会清理或覆盖已有发行包。重新打包可用 `-OutputDirectory .\dist\StreamTransClient-new` 指定新目录；已构建后可加 `-SkipBuild` 复用 `target/release/streamtrans-client.exe`。

## 验证范围

2026-11-07，最后一句自动隐藏模式已通过客户端 15 项测试、Clippy `-D warnings` 和格式检查，发布版本构建成功。实际浏览器使用受控状态数据完成 20 项检查，覆盖默认历史、最新段选择、超时隐藏、重复轮询及重新加载不复活、迟到旧段、最新段补译文续时、显示选项、仅识别、清空、超时参数回落、断连计时和 320 px 无横向溢出。本轮未进行真实麦克风录音或 GPU 模型链路回归。

以下具体回放耗时和译文记录来自 MiniCPM5-2B 服务端，保留为历史；它们验证客户端与连续输入管线，不代表当前 Hy-MT2 的翻译速度或质量。当前模型的测试记录见源码项目中的 `docs/hy-mt2-20261103.md`。

### 2026-11-01 服务端连续输入回归（历史）

Windows、RTX 3050（6 GB）上，协议测试脚本以 1× 实时速率、100 ms 数据块发送 60.037 秒密集短句，服务端使用独立 VAD 队列与默认空思考翻译配置。收到 22 段原文、22 段英语译文和 `done`，无错误，所有段的推理排队等待均为 0 ms；`done.duration_ms` 为 60037，包含最后 37 ms 短尾，墙钟时间为 60.589 秒。

另以 1× 实时速率回放 35 秒连续语音，仅识别模式返回 3 段原文，按 15 秒、15 秒及末段切分，无翻译、无错误；`done.duration_ms` 为 35000。2.23 秒快速语音加 2 秒静音，以 20 ms 数据块回放并翻译为日语，原文和译文都在 `finish` 前返回，`done.duration_ms` 为 4230。原有文件上传翻译接口在新默认配置下仍返回 `200`。

同一密集样本在旧版串行 VAD、ASR 和翻译流程中于约 37.420 秒触发 `audio_buffer_full`。这项对照验证了新版在该样本上的持续处理行为，不代表任意负载都不会达到队列上限，也没有验证真实麦克风采集。原始记录为开发工作区 `.tools/benchmark/stream-after-dense60s-20261101.jsonl`。

### 早期客户端回归（历史）

以下为独立 VAD 队列和默认空思考前缀调整前的客户端回归记录。Windows、RTX 3050（6 GB）上，已通过客户端 `--headless --wav` 按实时速率回放 9.121 秒中文 PCM16 WAV，使用真实 GPU 服务完成两种模式：

- 仅识别收到 2 段中文，输出状态中的 `translation` 为 `null`，LLM 推理调用数为 0。
- 英语翻译收到 2 段原文和译文，客户端按相同 `segment_id` 合并显示。
- 文件接口 `POST /v1/audio/transcriptions` 返回 `200`，`elapsed_ms.translation` 为 `0`；误传 `target_language` 返回 `400`。原有 `POST /v1/audio/translations` 仍返回 `200`。

原生 GUI 已启动并截图检查；设备枚举成功识别出网易虚拟麦克风。网页在 Edge 的 320 × 720 视口下没有横向溢出，CSS 改为 31 px 字号及恢复模板均自动生效。工作区测试通过 85 项、默认跳过 4 项本地模型测试，工作区 Clippy `-D warnings` 通过。

以上 WAV 回放未实际打开麦克风或录制音频。客户端回放验证了网络管线、服务端分段及逐段显示，尚不能替代真实录音设备的采集、权限和持续录音验证。Linux 客户端未实机验证。

## 问题处理

| 现象 | 下一步 |
| --- | --- |
| 没有录音设备或设备已变化 | 检查系统麦克风权限、连接状态，再刷新并重新选择设备 |
| 无法连接 API | 确认服务已完成模型加载，核对 API 地址和端口；`3001` 是客户端输出页，服务默认使用 `3000` |
| 服务返回 `401` | 核对客户端请求 key 与服务端一致；反向代理需保留单个 `Authorization` 请求头 |
| 返回 `429` | 等待已有录音或文件请求结束；主动断开后，正在执行的 GPU 调用可能仍需短暂收尾 |
| 服务返回 `audio_buffer_full` | 原始入站音频未及时被 VAD 消费；核对采样时长与发送时间，检查网络突发和服务端 CPU 调度，不要仅按一句话的词数判断原因 |
| 服务返回 `processing_backlog_full` | 等待识别或翻译的语音段达到 120 秒 PCM 或 64 段上限；查看服务端每段耗时和 GPU 负载，待当前任务结束后重新连接。可用「不翻译」区分识别和翻译开销 |
| 客户端报告音频发送队列已满 | 客户端采集速度超过当前网络发送能力；已停止录音，检查连接和服务状态后重试。这与服务端两种缓冲错误不同 |
| 停止后没有立即出结果 | 等待末段识别和翻译完成；结果按段返回，不是逐字中间结果 |
| 网页端口被占用 | 关闭占用端口的另一客户端，或通过 `--web-bind` 指定其他本机回环端口 |
| CSS 没有变化 | 确认修改的是当前 `--css` 指向的文件，使用 UTF-8 保存，检查 `/style.css` 返回的内容 |
| 服务报告模型超时或后端不可用 | 查看服务端错误，按提示重启服务，再重新开始录音；已收到的原文会保留 |

完整参数以 `streamtrans-client.exe --help` 为准。实时协议和服务端限制见源码项目中的 `docs/streaming.md`。
