use std::{
    collections::VecDeque,
    path::Path,
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[cfg(not(windows))]
use tokio::process::Command;
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, mpsc, watch},
    time::{Instant, timeout},
};

use crate::{
    asr::Asr,
    config::Config,
    glossary::Glossary,
    llm::Llm,
    stream::StreamInput,
    vad::{Vad, VadUpdate},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct Models {
    pub asr_dir: std::path::PathBuf,
    pub vad_dir: std::path::PathBuf,
    pub llm_file: std::path::PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recognition {
    pub text: String,
    pub segments: Vec<Segment>,
}

#[async_trait]
pub trait Backend: Send + Sync {
    async fn recognize(&self, path: &Path, language: &str) -> Result<Recognition>;
    async fn translate(&self, text: &str, target: &str) -> Result<String>;
    async fn healthy(&self) -> bool;
    async fn stream(
        &self,
        _input: mpsc::Receiver<StreamInput>,
        _output: mpsc::Sender<Value>,
        _source: &str,
        _target: Option<&str>,
    ) -> Result<()> {
        anyhow::bail!("当前后端不支持流式音频")
    }
}

struct Engine {
    asr: Asr,
    vad: Arc<Vad>,
    llm: Llm,
}

pub struct Runtime {
    engine: Arc<Mutex<Option<Engine>>>,
    alive: AtomicBool,
    inference_timeout: Duration,
}

struct CancelStartup(Arc<AtomicBool>);
impl Drop for CancelStartup {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

impl Runtime {
    pub async fn check_tools(config: &Config) -> Result<()> {
        if config.download_only {
            return Ok(());
        }
        #[cfg(windows)]
        crate::audio::Decoder::load(&config.ffmpeg_library)?;
        #[cfg(not(windows))]
        {
            let ffmpeg = timeout(
                Duration::from_secs(15),
                Command::new(&config.ffmpeg)
                    .arg("-version")
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .context("FFmpeg 检查超时")?
            .context("找不到 FFmpeg，请先安装或使用 --ffmpeg 指定路径")?;
            ensure!(ffmpeg.status.success(), "FFmpeg 启动失败");
        }
        ensure!(
            config.asr_library.is_file(),
            "找不到 ASR 原生库 {}，请先运行 scripts/setup.ps1",
            config.asr_library.display()
        );
        #[cfg(not(windows))]
        ensure!(
            config.llama_server.is_file(),
            "找不到 llama-server {}，请运行 scripts/setup.ps1 或使用 --llama-server 指定路径",
            config.llama_server.display()
        );
        #[cfg(windows)]
        ensure!(
            config.llama_library.is_file(),
            "找不到进程内翻译库 {}，请运行 scripts/setup.ps1 或使用 --llama-library 指定路径",
            config.llama_library.display()
        );
        Ok(())
    }

    pub async fn download(config: &Config) -> Result<Models> {
        crate::models::download(config).await
    }

    pub async fn start(config: &Config, models: Models) -> Result<Self> {
        let config = config.clone();
        let startup_timeout = Duration::from_secs(config.startup_timeout_seconds);
        let inference_timeout = Duration::from_secs(config.inference_timeout_seconds);
        let cancelled = Arc::new(AtomicBool::new(false));
        let _cancel_on_drop = CancelStartup(cancelled.clone());
        let engine = timeout(
            startup_timeout,
            tokio::task::spawn_blocking(move || -> Result<Engine> {
                let glossary = config.glossary.as_deref().map(Glossary::load).transpose()?;
                if let Some(glossary) = &glossary {
                    tracing::info!(
                        entries = glossary.len(),
                        "已读取完整术语表，准备固定提示前缀"
                    );
                }
                let converted = models.asr_dir.join("Fun-ASR-Nano-2512-native-bf16-v1.gguf");
                if !converted.is_file() {
                    tracing::info!("首次启动：使用 Rust 转换 Fun-ASR BF16 权重，后续复用转换缓存");
                    crate::convert_asr::convert(&models.asr_dir, &converted)?;
                }
                ensure!(!cancelled.load(Ordering::Acquire), "模型启动已取消");
                let vad = Arc::new(Vad::load(&models.vad_dir)?);
                ensure!(!cancelled.load(Ordering::Acquire), "模型启动已取消");
                let asr = Asr::load(&config.asr_library, &converted, config.gpu as usize)?;
                ensure!(!cancelled.load(Ordering::Acquire), "模型启动已取消");
                let mut llm = Llm::load(
                    #[cfg(not(windows))]
                    &config.llama_server,
                    #[cfg(windows)]
                    &config.llama_library,
                    &models.llm_file,
                    config.gpu as usize,
                    config.context_size,
                    config.max_tokens,
                    startup_timeout,
                    &cancelled,
                )?;
                ensure!(!cancelled.load(Ordering::Acquire), "模型启动已取消");
                if let Some(glossary) = glossary {
                    llm.set_glossary(&glossary, config.glossary_max_tokens)?;
                }
                llm.set_timeout(inference_timeout);
                Ok(Engine { asr, vad, llm })
            }),
        )
        .await
        .context("模型加载超时，请检查显存和 --startup-timeout-seconds")??;
        Ok(Self {
            engine: Arc::new(Mutex::new(Some(engine?))),
            alive: AtomicBool::new(true),
            inference_timeout,
        })
    }

    pub async fn stop(&self) {
        self.alive.store(false, Ordering::Release);
        let engine = self.engine.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Ok(mut guard) = engine.lock() {
                drop(guard.take());
            }
        })
        .await;
    }
}

pub(crate) const MAX_PENDING_SPEECH_MS: usize = 120_000;
pub(crate) const MAX_PENDING_SEGMENTS: usize = 64;
const RESULT_QUEUE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub(crate) struct SpeechBacklogFull;

impl std::fmt::Display for SpeechBacklogFull {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("识别/翻译待处理语音积压已达上限（最多 120 秒或 64 段，含在途段），会话已停止；未完成音频不能视为识别完成，请查看服务日志后重试")
    }
}
impl std::error::Error for SpeechBacklogFull {}

pub(crate) fn stream_error_event(cause: &anyhow::Error) -> Value {
    json!({"type":"error", "code":if cause.is::<SpeechBacklogFull>() {"processing_backlog_full"} else {"stream_failed"}, "message":format!("{cause:#}")})
}

#[derive(Clone, Copy)]
struct StreamLimits {
    samples: usize,
    segments: usize,
}
impl Default for StreamLimits {
    fn default() -> Self {
        Self {
            samples: MAX_PENDING_SPEECH_MS * 16,
            segments: MAX_PENDING_SEGMENTS,
        }
    }
}

#[derive(Clone)]
struct SpeechBudget {
    samples: Arc<Semaphore>,
    segments: Arc<Semaphore>,
    limits: StreamLimits,
}
impl SpeechBudget {
    fn new(limits: StreamLimits) -> Self {
        Self {
            samples: Arc::new(Semaphore::new(limits.samples)),
            segments: Arc::new(Semaphore::new(limits.segments)),
            limits,
        }
    }
    fn pending(&self) -> (usize, usize) {
        (
            self.limits.samples - self.samples.available_permits(),
            self.limits.segments - self.segments.available_permits(),
        )
    }
    fn reserve(&self, samples: usize) -> Result<(OwnedSemaphorePermit, OwnedSemaphorePermit)> {
        let count = self
            .segments
            .clone()
            .try_acquire_owned()
            .map_err(|_| SpeechBacklogFull)?;
        let audio = self
            .samples
            .clone()
            .try_acquire_many_owned(samples.try_into()?)
            .map_err(|_| SpeechBacklogFull)?;
        Ok((audio, count))
    }
}

struct SpeechWork {
    id: u64,
    segment: crate::vad::VadSegment,
    audio: Vec<f32>,
    queued: Instant,
    // Includes the active inference, not just items still waiting in the channel.
    _budget: (OwnedSemaphorePermit, OwnedSemaphorePermit),
}
enum InferenceWork {
    Speech(SpeechWork),
    Finish { duration_ms: u64, segments: u64 },
}

trait SpeechInference: Send {
    fn recognize(&mut self, audio: &[f32], source: &str) -> Result<String>;
    fn translate(&mut self, text: &str, target: &str) -> Result<String>;
}
struct NativeInference {
    engine: Arc<Mutex<Option<Engine>>>,
    timeout: Duration,
}
impl SpeechInference for NativeInference {
    fn recognize(&mut self, audio: &[f32], source: &str) -> Result<String> {
        self.engine
            .lock()
            .map_err(|_| anyhow::anyhow!("模型状态异常，请重启服务"))?
            .as_mut()
            .context("模型已停止")?
            .asr
            .transcribe(audio, source, self.timeout)
    }
    fn translate(&mut self, text: &str, target: &str) -> Result<String> {
        self.engine
            .lock()
            .map_err(|_| anyhow::anyhow!("模型状态异常，请重启服务"))?
            .as_mut()
            .context("模型已停止")?
            .llm
            .translate(text, target)
    }
}

fn stream_audio(
    engine: Arc<Mutex<Option<Engine>>>,
    input: mpsc::Receiver<StreamInput>,
    output: mpsc::Sender<Value>,
    source: &str,
    target: Option<&str>,
    inference_timeout: Duration,
) -> Result<()> {
    // CPU model weights are shared, but no VAD operation acquires the GPU mutex.
    let vad = engine
        .lock()
        .map_err(|_| anyhow::anyhow!("模型状态异常，请重启服务"))?
        .as_ref()
        .context("模型已停止")?
        .vad
        .clone();
    let mut state = vad.start_stream();
    stream_pipeline(
        input,
        output,
        source,
        target,
        move |samples, final_chunk| vad.push_stream(&mut state, samples, final_chunk),
        NativeInference {
            engine,
            timeout: inference_timeout,
        },
        StreamLimits::default(),
    )
}

fn send_stream_event(output: &mpsc::Sender<Value>, event: Value) -> Result<bool> {
    tokio::runtime::Handle::current().block_on(async {
        match timeout(RESULT_QUEUE_TIMEOUT, output.send(event)).await {
            Ok(Ok(())) => Ok(true),
            Ok(Err(_)) => Ok(false),
            Err(_) => {
                tracing::warn!(
                    stage = "result_output",
                    capacity = output.max_capacity(),
                    queued = output.max_capacity() - output.capacity(),
                    "流式结果队列持续阻塞，停止会话"
                );
                anyhow::bail!("结果输出队列持续 5 秒未能排空，会话已停止；请检查客户端接收速度")
            }
        }
    })
}

fn report_stream_error(output: &mpsc::Sender<Value>, reported: &AtomicBool, cause: &anyhow::Error) {
    if !reported.swap(true, Ordering::AcqRel) {
        let _ = send_stream_event(output, stream_error_event(cause));
    }
}

fn stream_pipeline<V: FnMut(&[f32], bool) -> Result<VadUpdate>, I: SpeechInference>(
    mut input: mpsc::Receiver<StreamInput>,
    output: mpsc::Sender<Value>,
    source: &str,
    target: Option<&str>,
    mut vad: V,
    inference: I,
    limits: StreamLimits,
) -> Result<()> {
    let (jobs_tx, jobs_rx) = mpsc::channel(limits.segments);
    let (cancel, cancellation) = watch::channel(false);
    let reported = AtomicBool::new(false);
    let budget = SpeechBudget::new(limits);
    let runtime = tokio::runtime::Handle::current();
    // Scoped join is mandatory: a dropped socket must not let the caller return its
    // GPU admission permit while an uncancellable native inference is still running.
    thread::scope(|scope| {
        let worker_output = output.clone();
        let worker_cancel = cancel.clone();
        let worker_cancellation = cancellation.clone();
        let worker_runtime = runtime.clone();
        let worker_budget = budget.clone();
        let reported = &reported;
        let worker = scope.spawn(move || {
            let _runtime = worker_runtime.enter();
            let result = infer_speech(
                jobs_rx,
                &worker_output,
                source,
                target,
                inference,
                worker_cancellation,
                worker_budget,
            );
            if let Err(cause) = &result {
                let _ = worker_cancel.send(true);
                report_stream_error(&worker_output, reported, cause);
            }
            result
        });
        let collection = collect_speech(
            &mut input,
            &output,
            &jobs_tx,
            &mut vad,
            &budget,
            cancellation,
        );
        if !matches!(&collection, Ok(true)) {
            let _ = cancel.send(true);
        }
        if let Err(cause) = &collection {
            report_stream_error(&output, reported, cause);
        }
        drop(jobs_tx);
        // Keep input alive while joining so a pending terminal error cannot race an
        // ingress "backend closed" error. The network loop prioritizes output events.
        let inference_result = worker
            .join()
            .map_err(|_| anyhow::anyhow!("流式 GPU 工作线程异常退出"));
        let inference_result = inference_result.and_then(|result| result);
        if let Err(cause) = &inference_result {
            report_stream_error(&output, reported, cause);
        }
        collection?;
        inference_result
    })
}

fn collect_speech<V: FnMut(&[f32], bool) -> Result<VadUpdate>>(
    input: &mut mpsc::Receiver<StreamInput>,
    output: &mpsc::Sender<Value>,
    jobs: &mpsc::Sender<InferenceWork>,
    vad: &mut V,
    budget: &SpeechBudget,
    mut cancel: watch::Receiver<bool>,
) -> Result<bool> {
    let runtime = tokio::runtime::Handle::current();
    let started = Instant::now();
    let mut last_progress = Instant::now();
    let mut pcm = PcmBuffer::default();
    let mut segment_id = 0;
    loop {
        let next = runtime.block_on(async {
            tokio::select! {
                biased;
                _ = output.closed() => None,
                _ = jobs.closed() => None,
                _ = cancel.changed() => None,
                next = input.recv() => next,
            }
        });
        let Some(next) = next else {
            return Ok(false);
        };
        let (samples, is_final) = match next {
            StreamInput::Audio(samples) => (samples, false),
            StreamInput::Finish => (Vec::new(), true),
        };
        ensure!(samples.len() <= 16_000, "流式音频块不能超过 1 秒");
        pcm.append(&samples)?;
        let vad_started = Instant::now();
        let update = vad(&samples, is_final)?;
        let vad_ms = vad_started.elapsed().as_millis();
        let emitted = !update.segments.is_empty();
        for segment in update.segments {
            if output.is_closed() || *cancel.borrow() {
                return Ok(false);
            }
            let length = usize::try_from(
                segment
                    .end_sample
                    .checked_sub(segment.start_sample)
                    .context("FSMN-VAD 段时间无效")?,
            )?;
            let reservation = budget.reserve(length).inspect_err(|_| {
                let (samples, segments) = budget.pending();
                tracing::warn!(
                    stage = "pending_speech",
                    pending_speech_ms = samples / 16,
                    pending_segments = segments,
                    attempted_speech_ms = length / 16,
                    max_pending_speech_ms = budget.limits.samples / 16,
                    max_pending_segments = budget.limits.segments,
                    received_audio_ms = pcm.received / 16,
                    elapsed_ms = started.elapsed().as_millis(),
                    input_queued_frames = input.len(),
                    retained_pcm_ms = pcm.samples.len() / 16,
                    "已结束语音等待推理的积压达到上限"
                );
            })?;
            let audio = pcm.segment(segment.start_sample, segment.end_sample)?;
            segment_id += 1;
            let job = InferenceWork::Speech(SpeechWork {
                id: segment_id,
                segment,
                audio,
                queued: Instant::now(),
                _budget: reservation,
            });
            match jobs.try_send(job) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Closed(_)) => return Ok(false),
                Err(mpsc::error::TrySendError::Full(_)) => return Err(SpeechBacklogFull.into()),
            }
        }
        // Release the original waveform immediately after taking owned speech chunks;
        // it no longer waits for that speech's ASR or translation to finish.
        pcm.discard_before(update.retain_from_sample)?;
        ensure!(
            pcm.samples.len() <= 16_000 * 17,
            "流式 VAD 保留音频超出内部上限"
        );
        if emitted || is_final || last_progress.elapsed() >= Duration::from_secs(5) {
            let (pending_samples, pending_segments) = budget.pending();
            tracing::info!(
                stage = "vad",
                received_audio_ms = pcm.received / 16,
                elapsed_ms = started.elapsed().as_millis(),
                input_queued_frames = input.len(),
                retained_pcm_ms = pcm.samples.len() / 16,
                pending_speech_ms = pending_samples / 16,
                pending_segments,
                vad_ms,
                finalized = is_final,
                "流式收音与 VAD 进度"
            );
            last_progress = Instant::now();
        }
        if is_final {
            return runtime.block_on(async {
                tokio::select! {
                    biased;
                    _ = output.closed() => Ok(false),
                    _ = cancel.changed() => Ok(false),
                    result = jobs.send(InferenceWork::Finish { duration_ms:pcm.received/16, segments:segment_id }) => Ok(result.is_ok()),
                }
            });
        }
    }
}

fn infer_speech<I: SpeechInference>(
    mut jobs: mpsc::Receiver<InferenceWork>,
    output: &mpsc::Sender<Value>,
    source: &str,
    target: Option<&str>,
    mut inference: I,
    mut cancel: watch::Receiver<bool>,
    budget: SpeechBudget,
) -> Result<()> {
    let runtime = tokio::runtime::Handle::current();
    loop {
        if output.is_closed() || *cancel.borrow() {
            return Ok(());
        }
        let next = runtime.block_on(async {
            tokio::select! {
                biased;
                _ = output.closed() => None,
                _ = cancel.changed() => None,
                next = jobs.recv() => next,
            }
        });
        let Some(next) = next else {
            return Ok(());
        };
        let work = match next {
            InferenceWork::Finish {
                duration_ms,
                segments,
            } => {
                send_stream_event(
                    output,
                    json!({"type":"done", "duration_ms":duration_ms,
                    "segments_count":segments,"mode":if target.is_some() {"translate"} else {"transcribe"}}),
                )?;
                return Ok(());
            }
            InferenceWork::Speech(work) => work,
        };
        let id = work.id;
        let start_ms = work.segment.start_sample / 16;
        let end_ms = work.segment.end_sample / 16;
        let queue_wait_ms = work.queued.elapsed().as_millis();
        let started = Instant::now();
        let text = inference
            .recognize(&work.audio, source)
            .with_context(|| format!("第 {id} 段 ASR 识别失败"))?;
        let asr_ms = started.elapsed().as_millis();
        let (pending_samples, pending_segments) = budget.pending();
        tracing::info!(
            stage = "asr",
            segment_id = id,
            start_ms,
            end_ms,
            queue_wait_ms,
            asr_ms,
            pending_speech_ms = pending_samples / 16,
            pending_segments,
            "语音段识别完成"
        );
        if output.is_closed() || *cancel.borrow() {
            return Ok(());
        }
        if !send_stream_event(
            output,
            json!({"type":"transcript","segment_id":id,"start_ms":start_ms,
            "end_ms":end_ms,"text":text,"is_final":true,"end_reason":work.segment.reason,
            "elapsed_ms":{"asr":asr_ms,"queue_wait":queue_wait_ms}}),
        )? {
            return Ok(());
        }
        if output.is_closed() || *cancel.borrow() {
            return Ok(());
        }
        if let Some(target) = target {
            let started = Instant::now();
            let translation = if text.trim().is_empty() {
                String::new()
            } else {
                inference
                    .translate(&text, target)
                    .with_context(|| format!("第 {id} 段翻译失败，原文已通过 transcript 返回"))?
            };
            let translation_ms = started.elapsed().as_millis();
            tracing::info!(
                stage = "translation",
                segment_id = id,
                translation_ms,
                pending_speech_ms = budget.pending().0 / 16,
                pending_segments = budget.pending().1,
                "语音段翻译完成"
            );
            if output.is_closed() || *cancel.borrow() {
                return Ok(());
            }
            if !send_stream_event(
                output,
                json!({"type":"translation","segment_id":id,
                "start_ms":start_ms,"end_ms":end_ms,"text":text,"translation":translation,
                "target_language":target,"is_final":true,"elapsed_ms":{"translation":translation_ms}}),
            )? {
                return Ok(());
            }
        }
    }
}

#[derive(Default)]
struct PcmBuffer {
    start: u64,
    received: u64,
    samples: VecDeque<f32>,
}

impl PcmBuffer {
    fn append(&mut self, samples: &[f32]) -> Result<()> {
        self.received = self
            .received
            .checked_add(samples.len() as u64)
            .context("音频流采样计数溢出")?;
        self.samples.extend(samples);
        Ok(())
    }

    fn segment(&self, start: u64, end: u64) -> Result<Vec<f32>> {
        ensure!(
            start >= self.start && start < end && end <= self.received && end - start <= 240_000,
            "FSMN-VAD 返回了超出音频缓存或 15 秒上限的语音段"
        );
        Ok(self
            .samples
            .range((start - self.start) as usize..(end - self.start) as usize)
            .copied()
            .collect())
    }

    fn discard_before(&mut self, sample: u64) -> Result<()> {
        ensure!(
            (self.start..=self.received).contains(&sample),
            "FSMN-VAD 返回了无效的缓存释放位置"
        );
        self.samples.drain(..(sample - self.start) as usize);
        self.start = sample;
        Ok(())
    }
}

#[async_trait]
impl Backend for Runtime {
    async fn stream(
        &self,
        input: mpsc::Receiver<StreamInput>,
        output: mpsc::Sender<Value>,
        source: &str,
        target: Option<&str>,
    ) -> Result<()> {
        ensure!(self.alive.load(Ordering::Acquire), "模型已停止，请重启服务");
        let engine = self.engine.clone();
        let source = source.to_owned();
        let target = target.map(str::to_owned);
        let inference_timeout = self.inference_timeout;
        // The caller's task owns the GPU permit and awaits this worker even after disconnect.
        tokio::task::spawn_blocking(move || {
            stream_audio(
                engine,
                input,
                output,
                &source,
                target.as_deref(),
                inference_timeout,
            )
        })
        .await
        .context("流式音频任务异常退出")?
    }

    async fn recognize(&self, path: &Path, language: &str) -> Result<Recognition> {
        ensure!(self.alive.load(Ordering::Acquire), "模型已停止，请重启服务");
        let path = path.to_owned();
        let language = language.to_owned();
        let engine = self.engine.clone();
        let deadline = Instant::now() + self.inference_timeout;
        // 等待原生调用结束才归还 GPU 许可；超时通过模型内部回调协作取消。
        tokio::task::spawn_blocking(move || -> Result<Recognition> {
            let mut reader = hound::WavReader::open(&path).context("无法读取规范化 WAV")?;
            let spec = reader.spec();
            ensure!(
                spec.channels == 1
                    && spec.sample_rate == 16000
                    && spec.bits_per_sample == 16
                    && spec.sample_format == hound::SampleFormat::Int,
                "音频必须为 16 kHz 单声道 PCM16 WAV"
            );
            let samples = reader
                .samples::<i16>()
                .map(|s| s.map(|v| f32::from(v) / 32768.0))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let mut guard = engine
                .lock()
                .map_err(|_| anyhow::anyhow!("模型状态异常，请重启服务"))?;
            let engine = guard.as_mut().context("模型已停止")?;
            let ranges = engine.vad.detect(&samples)?;
            let mut segments = Vec::with_capacity(ranges.len());
            for (start_ms, end_ms) in ranges {
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .context("ASR 推理超时")?;
                let start = (start_ms as usize * 16).min(samples.len());
                let end = (end_ms as usize * 16).min(samples.len());
                ensure!(start < end, "FSMN-VAD 返回了无效时间段");
                let text = engine
                    .asr
                    .transcribe(&samples[start..end], &language, remaining)?;
                segments.push(Segment {
                    start_ms,
                    end_ms,
                    text,
                });
            }
            let text = segments
                .iter()
                .map(|s| s.text.as_str())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            Ok(Recognition { text, segments })
        })
        .await
        .context("ASR 原生任务异常退出")?
    }

    async fn translate(&self, text: &str, target: &str) -> Result<String> {
        ensure!(self.alive.load(Ordering::Acquire), "模型已停止，请重启服务");
        let engine = self.engine.clone();
        let text = text.to_owned();
        let target = target.to_owned();
        tokio::task::spawn_blocking(move || {
            let mut guard = engine
                .lock()
                .map_err(|_| anyhow::anyhow!("模型状态异常，请重启服务"))?;
            guard
                .as_mut()
                .context("模型已停止")?
                .llm
                .translate(&text, &target)
        })
        .await
        .context("LLM 原生任务异常退出")?
    }

    async fn healthy(&self) -> bool {
        if !self.alive.load(Ordering::Acquire) {
            return false;
        }
        match self.engine.try_lock() {
            Ok(mut guard) => guard.as_mut().is_some_and(|engine| engine.llm.healthy()),
            Err(TryLockError::WouldBlock) => true,
            Err(TryLockError::Poisoned(_)) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::{VadEndReason, VadSegment};
    use std::sync::{
        Condvar,
        atomic::{AtomicU64, AtomicUsize},
    };

    #[derive(Default)]
    struct Gate {
        released: Mutex<bool>,
        changed: Condvar,
    }
    impl Gate {
        fn wait(&self) {
            let mut released = self.released.lock().unwrap();
            while !*released {
                released = self.changed.wait(released).unwrap();
            }
        }
        fn release(&self) {
            *self.released.lock().unwrap() = true;
            self.changed.notify_all();
        }
    }
    struct ReleaseOnDrop(Arc<Gate>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    struct FakeInference {
        gate: Option<(Arc<Gate>, mpsc::UnboundedSender<()>)>,
        calls: Arc<AtomicUsize>,
        fail: bool,
        panic: bool,
    }
    impl SpeechInference for FakeInference {
        fn recognize(&mut self, _: &[f32], _: &str) -> Result<String> {
            let id = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            assert!(!self.panic, "synthetic inference panic");
            ensure!(!self.fail, "synthetic inference failure");
            Ok(format!("segment {id}"))
        }
        fn translate(&mut self, _: &str, _: &str) -> Result<String> {
            if let Some((gate, entered)) = self.gate.take() {
                let _ = entered.send(());
                gate.wait();
            }
            Ok("translation".into())
        }
    }

    fn fake_vad(
        ranges: Vec<(u64, u64)>,
        received: Arc<AtomicU64>,
    ) -> impl FnMut(&[f32], bool) -> Result<VadUpdate> + Send {
        let mut ranges: VecDeque<_> = ranges.into();
        let mut total = 0;
        move |samples, final_chunk| {
            total += samples.len() as u64;
            received.store(total, Ordering::SeqCst);
            let mut segments = Vec::new();
            while let Some(&(start, end)) = ranges.front() {
                if end > total && !final_chunk {
                    break;
                }
                ranges.pop_front();
                if start < total {
                    segments.push(VadSegment {
                        start_sample: start,
                        end_sample: end.min(total),
                        reason: if final_chunk {
                            VadEndReason::Final
                        } else {
                            VadEndReason::Silence
                        },
                    });
                }
            }
            Ok(VadUpdate {
                segments,
                speech_start_sample: None,
                speech_active: false,
                processed_samples: total,
                received_samples: total,
                retain_from_sample: if final_chunk {
                    total
                } else {
                    ranges
                        .front()
                        .map_or(total, |(start, _)| (*start).min(total))
                },
            })
        }
    }
    async fn audio(input: &mpsc::Sender<StreamInput>, count: usize) {
        for _ in 0..count {
            timeout(
                Duration::from_secs(2),
                input.send(StreamInput::Audio(vec![0.0; 1600])),
            )
            .await
            .unwrap()
            .unwrap();
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    async fn event(output: &mut mpsc::Receiver<Value>) -> Value {
        timeout(Duration::from_secs(2), output.recv())
            .await
            .unwrap()
            .unwrap()
    }

    #[tokio::test]
    async fn slow_translation_does_not_stop_short_speech_and_silence_ingestion() {
        let (input_tx, input_rx) = mpsc::channel(2);
        let (output_tx, mut output_rx) = mpsc::channel(8);
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let received = Arc::new(AtomicU64::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Gate::default());
        let _release = ReleaseOnDrop(gate.clone());
        let vad = fake_vad(vec![(0, 32_000), (48_000, 80_000)], received.clone());
        let inference = FakeInference {
            gate: Some((gate.clone(), entered_tx)),
            calls: calls.clone(),
            fail: false,
            panic: false,
        };
        let task = tokio::task::spawn_blocking(move || {
            stream_pipeline(
                input_rx,
                output_tx,
                "auto",
                Some("en"),
                vad,
                inference,
                StreamLimits {
                    samples: 64_000,
                    segments: 2,
                },
            )
        });
        audio(&input_tx, 20).await;
        timeout(Duration::from_secs(2), entered_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event(&mut output_rx).await["type"], "transcript");
        // Accelerated 100 ms input ticks: two-second phrases and twenty seconds
        // of silence exceed the original 128-frame queue while LLM is held.
        audio(&input_tx, 230).await;
        assert_eq!(received.load(Ordering::SeqCst), 400_000);
        assert!(!task.is_finished());
        input_tx.send(StreamInput::Finish).await.unwrap();
        gate.release();
        for (kind, id) in [("translation", 1), ("transcript", 2), ("translation", 2)] {
            let output = event(&mut output_rx).await;
            assert_eq!(output["type"], kind);
            assert_eq!(output["segment_id"], id);
        }
        let done = event(&mut output_rx).await;
        assert_eq!(done["type"], "done");
        assert_eq!(done["duration_ms"], 25_000);
        assert_eq!(done["segments_count"], 2);
        timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn speech_budget_includes_active_work_and_reports_before_native_work_returns() {
        let (input_tx, input_rx) = mpsc::channel(2);
        let (output_tx, mut output_rx) = mpsc::channel(8);
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let gate = Arc::new(Gate::default());
        let _release = ReleaseOnDrop(gate.clone());
        let vad = fake_vad(
            vec![(0, 32_000), (32_000, 64_000), (64_000, 96_000)],
            Arc::new(AtomicU64::new(0)),
        );
        let inference = FakeInference {
            gate: Some((gate.clone(), entered_tx)),
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
            panic: false,
        };
        let task = tokio::task::spawn_blocking(move || {
            stream_pipeline(
                input_rx,
                output_tx,
                "auto",
                Some("en"),
                vad,
                inference,
                StreamLimits {
                    samples: 64_000,
                    segments: 2,
                },
            )
        });
        audio(&input_tx, 20).await;
        timeout(Duration::from_secs(2), entered_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event(&mut output_rx).await["type"], "transcript");
        audio(&input_tx, 40).await;
        let error = event(&mut output_rx).await;
        assert_eq!(error["type"], "error");
        assert_eq!(error["code"], "processing_backlog_full");
        assert!(
            !task.is_finished(),
            "active inference must still retain admission"
        );
        gate.release();
        let cause = timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(cause.is::<SpeechBacklogFull>());
    }

    #[tokio::test]
    async fn disconnect_cancels_queued_segments_but_joins_the_active_call() {
        let (input_tx, input_rx) = mpsc::channel(2);
        let (output_tx, mut output_rx) = mpsc::channel(8);
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let gate = Arc::new(Gate::default());
        let _release = ReleaseOnDrop(gate.clone());
        let calls = Arc::new(AtomicUsize::new(0));
        let inference = FakeInference {
            gate: Some((gate.clone(), entered_tx)),
            calls: calls.clone(),
            fail: false,
            panic: false,
        };
        let vad = fake_vad(
            vec![(0, 32_000), (32_000, 64_000)],
            Arc::new(AtomicU64::new(0)),
        );
        let task = tokio::task::spawn_blocking(move || {
            stream_pipeline(
                input_rx,
                output_tx,
                "auto",
                Some("en"),
                vad,
                inference,
                StreamLimits::default(),
            )
        });
        audio(&input_tx, 20).await;
        timeout(Duration::from_secs(2), entered_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event(&mut output_rx).await["type"], "transcript");
        audio(&input_tx, 20).await;
        drop(output_rx);
        drop(input_tx);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!task.is_finished());
        gate.release();
        timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "queued ASR must not start after disconnect"
        );
    }

    #[tokio::test]
    async fn inference_error_or_panic_wakes_an_idle_vad_receiver() {
        for panic in [false, true] {
            let (input_tx, input_rx) = mpsc::channel(2);
            let (output_tx, mut output_rx) = mpsc::channel(8);
            let vad = fake_vad(vec![(0, 1600)], Arc::new(AtomicU64::new(0)));
            let inference = FakeInference {
                gate: None,
                calls: Arc::new(AtomicUsize::new(0)),
                fail: !panic,
                panic,
            };
            let task = tokio::task::spawn_blocking(move || {
                stream_pipeline(
                    input_rx,
                    output_tx,
                    "auto",
                    None,
                    vad,
                    inference,
                    StreamLimits::default(),
                )
            });
            input_tx
                .send(StreamInput::Audio(vec![0.0; 1600]))
                .await
                .unwrap();
            let error = event(&mut output_rx).await;
            assert_eq!(error["type"], "error");
            assert_eq!(error["code"], "stream_failed");
            assert!(
                timeout(Duration::from_secs(2), task)
                    .await
                    .unwrap()
                    .unwrap()
                    .is_err()
            );
            // Sender intentionally remains open: worker failure must wake VAD itself.
            drop(input_tx);
        }
    }

    #[test]
    fn stream_audio_buffer_preserves_absolute_offsets_after_trimming() -> anyhow::Result<()> {
        let mut pcm = PcmBuffer::default();
        pcm.append(&[0.0, 0.1, 0.2, 0.3])?;
        pcm.discard_before(2)?;
        pcm.append(&[0.4, 0.5])?;
        assert_eq!(pcm.segment(2, 6)?, [0.2, 0.3, 0.4, 0.5]);
        assert!(pcm.segment(1, 4).is_err());
        assert!(pcm.segment(3, 7).is_err());
        assert!(pcm.segment(4, 4).is_err());
        assert!(pcm.discard_before(1).is_err());
        assert!(pcm.discard_before(7).is_err());
        pcm.discard_before(6)?;
        assert!(pcm.samples.is_empty());
        assert_eq!(pcm.received, 6);
        Ok(())
    }
}
