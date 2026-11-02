use std::{sync::Arc, time::Duration};

use axum::{
    extract::{
        Query, State,
        rejection::QueryRejection,
        ws::{
            CloseFrame, Message, WebSocket, WebSocketUpgrade, rejection::WebSocketUpgradeRejection,
        },
    },
    http::StatusCode,
    response::Response,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{
    sync::{OwnedSemaphorePermit, mpsc},
    time::{Instant, sleep_until, timeout},
};

use crate::{
    api::{ApiError, AppState, bad_request, error, validate_target},
    runtime::{Backend, MAX_PENDING_SEGMENTS, MAX_PENDING_SPEECH_MS, stream_error_event},
};

const SAMPLE_RATE: usize = 16_000;
const MAX_CHUNK_BYTES: usize = SAMPLE_RATE * 2;
const MAX_CONTROL_BYTES: usize = 64;
const FRAME_SAMPLES: usize = SAMPLE_RATE / 10;
const INPUT_CAPACITY: usize = 128;
const OUTPUT_CAPACITY: usize = 8;
const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(30);
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub enum StreamInput {
    Audio(Vec<f32>),
    Finish,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StreamQuery {
    target_language: Option<String>,
    #[serde(default = "default_source")]
    source_language: String,
}

fn default_source() -> String {
    "auto".to_owned()
}

pub(super) async fn upgrade(
    State(state): State<AppState>,
    query: Result<Query<StreamQuery>, QueryRejection>,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Result<Response, ApiError> {
    upgrade_mode(state, query, upgrade, true)
}

pub(super) async fn transcribe_upgrade(
    State(state): State<AppState>,
    query: Result<Query<StreamQuery>, QueryRejection>,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Result<Response, ApiError> {
    upgrade_mode(state, query, upgrade, false)
}

fn upgrade_mode(
    state: AppState,
    query: Result<Query<StreamQuery>, QueryRejection>,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
    translate: bool,
) -> Result<Response, ApiError> {
    let Query(query) =
        query.map_err(|_| bad_request("流式接口查询参数无效，请指定 target_language"))?;
    let target = if translate {
        Some(validate_target(
            query
                .target_language
                .as_deref()
                .ok_or_else(|| bad_request("缺少 target_language 查询参数"))?,
        )?)
    } else {
        if query.target_language.is_some() {
            return Err(bad_request(
                "仅识别接口不接受 target_language，请使用 translations 接口进行翻译",
            ));
        }
        None
    };
    if !matches!(query.source_language.as_str(), "auto" | "zh" | "en" | "ja") {
        return Err(bad_request("source_language 仅支持 auto、zh、en、ja"));
    }
    let upgrade = upgrade.map_err(|_| bad_request("此接口需要有效的 WebSocket 升级请求"))?;
    let permit = state.gate.clone().try_acquire_owned().map_err(|_| {
        error(
            StatusCode::TOO_MANY_REQUESTS,
            "busy",
            "GPU 正在处理其他音频，请稍后重试",
        )
    })?;
    Ok(upgrade
        .max_frame_size(MAX_CHUNK_BYTES)
        .max_message_size(MAX_CHUNK_BYTES)
        .on_upgrade(move |socket| {
            session(
                socket,
                state.backend,
                permit,
                query.source_language,
                target,
                state.config.max_stream_seconds,
            )
        }))
}

async fn session(
    mut socket: WebSocket,
    backend: Arc<dyn Backend>,
    permit: OwnedSemaphorePermit,
    source: String,
    target: Option<String>,
    max_stream_seconds: u32,
) {
    let ready = json!({
        "type":"ready", "request_id":uuid::Uuid::new_v4().to_string(),
        "source_language":source, "target_language":target,
        "mode":if target.is_some() {"translate"} else {"transcribe"},
        "format":"pcm_s16le", "sample_rate":SAMPLE_RATE, "channels":1,
        "max_chunk_bytes":MAX_CHUNK_BYTES, "max_audio_seconds":max_stream_seconds,
        "inactivity_timeout_seconds":INACTIVITY_TIMEOUT.as_secs(),
        "max_buffered_audio_ms":INPUT_CAPACITY * 100,
        "audio_buffer_stage":"network_to_vad",
        "max_pending_speech_ms":MAX_PENDING_SPEECH_MS,
        "max_pending_segments":MAX_PENDING_SEGMENTS
    });
    if send_event(&mut socket, &ready).await.is_err() {
        return;
    }
    let (input_tx, input_rx) = mpsc::channel(INPUT_CAPACITY);
    let (output_tx, mut output_rx) = mpsc::channel(OUTPUT_CAPACITY);
    // This detached worker owns admission until native work actually returns. Dropping
    // a WebSocket or its JoinHandle must never free admission during an active GPU call.
    tokio::spawn(async move {
        let _permit = permit;
        if let Err(cause) = backend
            .stream(input_rx, output_tx.clone(), &source, target.as_deref())
            .await
        {
            let message = format!("{cause:#}");
            tracing::error!(error = %message, "流式模型推理失败");
            let _ = timeout(WRITE_TIMEOUT, output_tx.send(stream_error_event(&cause))).await;
        }
    });
    let mut input = InputState::new(max_stream_seconds);
    let session_started = Instant::now();
    let mut last_audio = Instant::now();
    let ending = loop {
        tokio::select! {
            // A terminal backend error already queued must not be replaced by a
            // simultaneous ingress Full/Closed error with less specific context.
            biased;
            output = output_rx.recv() => {
                let Some(output) = output else {
                    break Ending::Error(ProtocolError::new("stream_failed", "流式后端未返回完成事件", 1011));
                };
                let terminal = match output.get("type").and_then(Value::as_str) {
                    Some("done") => Some(1000),
                    Some("error") => Some(1011),
                    _ => None,
                };
                if send_event(&mut socket, &output).await.is_err() {
                    break Ending::Disconnected;
                }
                if let Some(code) = terminal {
                    break Ending::Close(code);
                }
            }
            message = socket.recv() => {
                let Some(Ok(message)) = message else { break Ending::Disconnected; };
                let audio_received = matches!(&message, Message::Binary(_));
                let next = match message {
                    Message::Binary(bytes) => input.audio(&bytes),
                    Message::Text(text) => input.finish(text.as_str()),
                    Message::Close(_) => break Ending::Disconnected,
                    Message::Ping(_) | Message::Pong(_) => continue,
                };
                let next = match next {
                    Ok(next) => next,
                    Err(cause) => break Ending::Error(cause),
                };
                if audio_received {
                    last_audio = Instant::now();
                }
                let mut queue_error = None;
                for next in next {
                    if let Err(cause) = input_tx.try_send(next) {
                        queue_error = Some(match cause {
                        mpsc::error::TrySendError::Full(_) => {
                            tracing::warn!(stage="network_to_vad", input_capacity=INPUT_CAPACITY,
                                input_queued_frames=INPUT_CAPACITY-input_tx.capacity(),
                                received_audio_ms=input.samples/16,
                                elapsed_ms=session_started.elapsed().as_millis(),
                                "网络输入到 VAD 的音频队列已满，终止会话而不静默丢弃音频");
                            ProtocolError::new(
                                "audio_buffer_full", "网络输入到 VAD 的音频接收队列已满，会话已停止；请查看服务日志中的接收速率及处理积压后重试", 1013,
                            )
                        },
                        mpsc::error::TrySendError::Closed(_) => ProtocolError::new(
                            "stream_failed", "流式后端已停止", 1011,
                        ),
                        });
                        break;
                    }
                }
                if let Some(cause) = queue_error {
                    break Ending::Error(cause);
                }
            }
            _ = sleep_until(last_audio + INACTIVITY_TIMEOUT), if !input.finished => {
                break Ending::Error(ProtocolError::new(
                    "audio_timeout", "连续 30 秒没有收到音频数据，请重新连接", 1008,
                ));
            }
        }
    };
    // Notify the backend before any potentially slow final send/close. It may finish
    // its current GPU operation, but must discard work that has not started.
    drop(input_tx);
    drop(output_rx);
    match ending {
        Ending::Error(cause) => {
            let _ = send_event(
                &mut socket,
                &json!({
                    "type":"error", "code":cause.code, "message":cause.message
                }),
            )
            .await;
            close_socket(&mut socket, cause.close_code).await;
        }
        Ending::Close(code) => close_socket(&mut socket, code).await,
        Ending::Disconnected => {}
    }
}

enum Ending {
    Error(ProtocolError),
    Close(u16),
    Disconnected,
}

async fn send_event(socket: &mut WebSocket, event: &Value) -> Result<(), ()> {
    timeout(
        WRITE_TIMEOUT,
        socket.send(Message::Text(event.to_string().into())),
    )
    .await
    .map_err(|_| ())?
    .map_err(|_| ())
}

async fn close_socket(socket: &mut WebSocket, code: u16) {
    let _ = timeout(
        WRITE_TIMEOUT,
        socket.send(Message::Close(Some(CloseFrame {
            code,
            reason: "".into(),
        }))),
    )
    .await;
}

struct InputState {
    samples: usize,
    maximum_samples: usize,
    finished: bool,
    pending: Vec<f32>,
}

impl InputState {
    fn new(max_stream_seconds: u32) -> Self {
        Self {
            samples: 0,
            maximum_samples: if max_stream_seconds == 0 {
                usize::MAX
            } else {
                max_stream_seconds as usize * SAMPLE_RATE
            },
            finished: false,
            pending: Vec::new(),
        }
    }

    fn audio(&mut self, bytes: &[u8]) -> Result<Vec<StreamInput>, ProtocolError> {
        if self.finished {
            return Err(ProtocolError::new(
                "invalid_request",
                "finish 后不能继续发送音频",
                1008,
            ));
        }
        if bytes.is_empty() || !bytes.len().is_multiple_of(2) {
            return Err(ProtocolError::new(
                "invalid_audio",
                "PCM16LE 音频块必须非空，且字节数为偶数",
                1008,
            ));
        }
        if bytes.len() > MAX_CHUNK_BYTES {
            return Err(ProtocolError::new(
                "audio_chunk_too_large",
                "单个音频块不能超过 32000 字节（1 秒）",
                1009,
            ));
        }
        let next = self
            .samples
            .checked_add(bytes.len() / 2)
            .ok_or_else(|| ProtocolError::new("audio_too_long", "音频超过总时长限制", 1008))?;
        if next > self.maximum_samples {
            return Err(ProtocolError::new(
                "audio_too_long",
                "音频超过 max_stream_seconds 总时长限制",
                1008,
            ));
        }
        self.samples = next;
        self.pending.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32768.0),
        );
        let complete = self.pending.len() / FRAME_SAMPLES * FRAME_SAMPLES;
        let frames = self.pending[..complete]
            .as_chunks::<FRAME_SAMPLES>()
            .0
            .iter()
            .map(|frame| StreamInput::Audio(frame.to_vec()))
            .collect();
        self.pending.drain(..complete);
        Ok(frames)
    }

    fn finish(&mut self, text: &str) -> Result<Vec<StreamInput>, ProtocolError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Control {
            #[serde(rename = "type")]
            kind: String,
        }
        if self.finished {
            return Err(ProtocolError::new(
                "invalid_request",
                "finish 只能发送一次",
                1008,
            ));
        }
        if text.len() > MAX_CONTROL_BYTES {
            return Err(ProtocolError::new(
                "invalid_request",
                "控制消息不能超过 64 字节",
                1008,
            ));
        }
        let command: Control = serde_json::from_str(text).map_err(|_| {
            ProtocolError::new(
                "invalid_request",
                "控制消息仅支持 {\"type\":\"finish\"}",
                1008,
            )
        })?;
        if command.kind != "finish" {
            return Err(ProtocolError::new(
                "invalid_request",
                "控制消息仅支持 finish",
                1008,
            ));
        }
        self.finished = true;
        let mut frames = Vec::with_capacity(2);
        if !self.pending.is_empty() {
            frames.push(StreamInput::Audio(std::mem::take(&mut self.pending)));
        }
        frames.push(StreamInput::Finish);
        Ok(frames)
    }
}

#[derive(Debug)]
struct ProtocolError {
    code: &'static str,
    message: &'static str,
    close_code: u16,
}

impl ProtocolError {
    fn new(code: &'static str, message: &'static str, close_code: u16) -> Self {
        Self {
            code,
            message,
            close_code,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use clap::Parser;
    use futures_util::{SinkExt, StreamExt};
    use tokio::{net::TcpStream, sync::Semaphore, task::JoinHandle};
    use tokio_tungstenite::{
        MaybeTlsStream, WebSocketStream, connect_async,
        tungstenite::{Error as WsError, Message as ClientMessage},
    };

    use crate::{config::Config, runtime::Recognition};

    #[test]
    fn pcm_is_merged_into_fixed_frames_and_finish_flushes_exact_tail() {
        let mut input = InputState::new(60);
        let mut frames = Vec::new();
        // Individual two-byte messages must not consume queue entries before 100 ms.
        for index in 0..FRAME_SAMPLES + 3 {
            frames.extend(input.audio(&(index as i16).to_le_bytes()).unwrap());
        }
        assert_eq!(frames.len(), 1);
        let StreamInput::Audio(frame) = &frames[0] else {
            panic!("expected audio")
        };
        assert_eq!(frame.len(), FRAME_SAMPLES);
        assert_eq!(frame[0], 0.0);
        assert_eq!(frame[1599], 1599.0 / 32768.0);
        let tail = input.finish(r#"{"type":"finish"}"#).unwrap();
        assert_eq!(tail.len(), 2);
        let StreamInput::Audio(frame) = &tail[0] else {
            panic!("expected tail")
        };
        assert_eq!(
            frame,
            &[1600.0 / 32768.0, 1601.0 / 32768.0, 1602.0 / 32768.0]
        );
        assert!(matches!(tail[1], StreamInput::Finish));
        assert!(input.audio(&[0, 0]).is_err());
        assert!(input.finish(r#"{"type":"finish"}"#).is_err());
    }

    #[test]
    fn pcm_and_control_limits_reject_malformed_or_excess_audio() {
        let mut input = InputState::new(1);
        assert!(input.audio(&[]).is_err());
        assert!(input.audio(&[0]).is_err());
        assert_eq!(
            input.audio(&vec![0; MAX_CHUNK_BYTES + 2]).unwrap_err().code,
            "audio_chunk_too_large"
        );
        assert_eq!(input.audio(&vec![0; MAX_CHUNK_BYTES]).unwrap().len(), 10);
        assert_eq!(input.audio(&[0, 0]).unwrap_err().code, "audio_too_long");
        for text in [
            "",
            "{}",
            r#"{"type":"unknown"}"#,
            r#"{"type":"finish","extra":1}"#,
            r#"{"type":"finish","type":"finish"}"#,
        ] {
            assert!(InputState::new(60).finish(text).is_err());
        }
        assert!(
            InputState::new(60)
                .finish(&format!("{}{{\"type\":\"finish\"}}", " ".repeat(65)))
                .is_err()
        );
        let mut input = InputState::new(1);
        input.audio(&[0, 128, 255, 127]).unwrap();
        let tail = input.finish(r#"{"type":"finish"}"#).unwrap();
        let StreamInput::Audio(frame) = &tail[0] else {
            panic!("expected tail")
        };
        assert_eq!(frame, &[-1.0, 32767.0 / 32768.0]);
    }

    #[test]
    fn unlimited_stream_accepts_more_than_file_duration_without_retaining_history() {
        let mut input = InputState::new(0);
        for _ in 0..61 {
            let frames = input.audio(&vec![0; MAX_CHUNK_BYTES]).unwrap();
            assert_eq!(frames.len(), 10);
            assert!(input.pending.is_empty());
        }
        assert_eq!(input.samples, 61 * SAMPLE_RATE);
        assert_eq!(input.maximum_samples, usize::MAX);
        assert!(matches!(
            input.finish(r#"{"type":"finish"}"#).unwrap()[0],
            StreamInput::Finish
        ));
        let mut bounded = InputState::new(1);
        bounded.audio(&vec![0; MAX_CHUNK_BYTES]).unwrap();
        assert_eq!(bounded.audio(&[0, 0]).unwrap_err().code, "audio_too_long");
    }

    #[derive(Default)]
    struct FakeBackend {
        entered: Option<mpsc::UnboundedSender<()>>,
        release: Option<Arc<Semaphore>>,
        fail: bool,
        asr_only: bool,
        translations: AtomicUsize,
    }

    #[async_trait]
    impl Backend for FakeBackend {
        async fn recognize(&self, _: &Path, _: &str) -> anyhow::Result<Recognition> {
            unreachable!("streaming must not call the file API")
        }
        async fn translate(&self, _: &str, _: &str) -> anyhow::Result<String> {
            unreachable!("streaming must not call the batch API")
        }
        async fn healthy(&self) -> bool {
            true
        }
        async fn stream(
            &self,
            mut input: mpsc::Receiver<StreamInput>,
            output: mpsc::Sender<Value>,
            _: &str,
            target: Option<&str>,
        ) -> anyhow::Result<()> {
            if self.asr_only {
                assert!(
                    target.is_none(),
                    "ASR-only route must never request translation"
                );
            }
            if self.fail {
                return Err(anyhow::anyhow!("LLM 推理超时，子进程已停止，请重启服务")
                    .context("第 2 段翻译失败，原文已通过 transcript 返回"));
            }
            let (mut samples, mut segments) = (0usize, 0usize);
            while let Some(message) = input.recv().await {
                match message {
                    StreamInput::Audio(audio) => {
                        let start_ms = samples as u64 * 1000 / SAMPLE_RATE as u64;
                        samples += audio.len();
                        if let Some(entered) = &self.entered {
                            let _ = entered.send(());
                        }
                        if let Some(release) = &self.release {
                            release.acquire().await?.forget();
                        }
                        if output.is_closed() {
                            return Ok(());
                        }
                        segments += 1;
                        let end_ms = samples as u64 * 1000 / SAMPLE_RATE as u64;
                        output.send(json!({"type":"transcript","segment_id":segments,"start_ms":start_ms,"end_ms":end_ms,"text":"原文","is_final":true})).await?;
                        if let Some(target) = target {
                            self.translations.fetch_add(1, Ordering::SeqCst);
                            output.send(json!({"type":"translation","segment_id":segments,"start_ms":start_ms,"end_ms":end_ms,"translation":"translation","target_language":target,"is_final":true})).await?;
                        }
                    }
                    StreamInput::Finish => {
                        output.send(json!({"type":"done","duration_ms":samples * 1000 / SAMPLE_RATE,"segments_count":segments,"mode":if target.is_some() {"translate"} else {"transcribe"}})).await?;
                        return Ok(());
                    }
                }
            }
            Ok(())
        }
    }

    struct TestServer {
        address: std::net::SocketAddr,
        task: JoinHandle<()>,
    }
    impl TestServer {
        async fn start(backend: Arc<dyn Backend>, config: Config) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let task = tokio::spawn(async move {
                axum::serve(listener, crate::api::router(backend, config))
                    .await
                    .unwrap();
            });
            Self { address, task }
        }
        fn url(&self, query: &str) -> String {
            format!("ws://{}/v1/audio/translations/stream?{query}", self.address)
        }
        fn transcription_url(&self, query: &str) -> String {
            format!(
                "ws://{}/v1/audio/transcriptions/stream?{query}",
                self.address
            )
        }
        async fn batch_status(&self) -> reqwest::StatusCode {
            reqwest::Client::new()
                .post(format!("http://{}/v1/audio/translations", self.address))
                .send()
                .await
                .unwrap()
                .status()
        }
    }
    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }
    type ClientSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

    async fn event(socket: &mut ClientSocket) -> Value {
        timeout(Duration::from_secs(3), async {
            loop {
                match socket.next().await.unwrap().unwrap() {
                    ClientMessage::Text(text) => {
                        return serde_json::from_str(text.as_str()).unwrap();
                    }
                    ClientMessage::Ping(_) | ClientMessage::Pong(_) => {}
                    other => panic!("expected JSON event, got {other:?}"),
                }
            }
        })
        .await
        .unwrap()
    }
    fn config() -> Config {
        Config::parse_from(["streamtrans"])
    }

    #[tokio::test]
    async fn transcription_stream_has_no_target_or_translation_and_flushes_the_tail() {
        let backend = Arc::new(FakeBackend {
            asr_only: true,
            ..Default::default()
        });
        let server = TestServer::start(backend.clone(), config()).await;
        for query in ["target_language=en", "source_language=fr"] {
            let WsError::Http(response) = connect_async(server.transcription_url(query))
                .await
                .unwrap_err()
            else {
                panic!("expected query rejection")
            };
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        let (mut socket, _) = connect_async(server.transcription_url("")).await.unwrap();
        let ready = event(&mut socket).await;
        assert_eq!(ready["type"], "ready");
        assert_eq!(ready["mode"], "transcribe");
        assert_eq!(ready["source_language"], "auto");
        assert!(ready["target_language"].is_null());
        assert_eq!(server.batch_status().await, StatusCode::TOO_MANY_REQUESTS);
        for _ in 0..5 {
            socket
                .send(ClientMessage::Binary(vec![0; 640].into()))
                .await
                .unwrap();
        }
        let transcript = event(&mut socket).await;
        assert_eq!(transcript["type"], "transcript");
        assert_eq!(transcript["end_ms"], 100);
        socket
            .send(ClientMessage::Binary(vec![0; 640].into()))
            .await
            .unwrap();
        socket
            .send(ClientMessage::Text(r#"{"type":"finish"}"#.into()))
            .await
            .unwrap();
        let tail = event(&mut socket).await;
        assert_eq!(tail["type"], "transcript");
        assert_eq!(tail["start_ms"], 100);
        assert_eq!(tail["end_ms"], 120);
        let done = event(&mut socket).await;
        assert_eq!(done["type"], "done");
        assert_eq!(done["mode"], "transcribe");
        assert_eq!(done["duration_ms"], 120);
        assert_eq!(done["segments_count"], 2);
        assert_eq!(backend.translations.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn handshake_validates_languages_and_shares_http_admission() {
        let server = TestServer::start(Arc::new(FakeBackend::default()), config()).await;
        for query in [
            "",
            "target_language=123",
            "target_language=en&source_language=fr",
        ] {
            let error = connect_async(server.url(query)).await.unwrap_err();
            let WsError::Http(response) = error else {
                panic!("expected HTTP rejection")
            };
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        let (mut socket, response) = connect_async(server.url("target_language=ja"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
        let ready = event(&mut socket).await;
        assert_eq!(ready["type"], "ready");
        assert_eq!(ready["target_language"], "日本語");
        assert_eq!(ready["source_language"], "auto");
        assert_eq!(ready["max_chunk_bytes"], MAX_CHUNK_BYTES);
        assert_eq!(ready["max_audio_seconds"], 0);
        assert_eq!(ready["max_buffered_audio_ms"], 12800);
        assert_eq!(ready["audio_buffer_stage"], "network_to_vad");
        assert_eq!(ready["max_pending_speech_ms"], 120_000);
        assert_eq!(ready["max_pending_segments"], 64);
        assert_eq!(server.batch_status().await, StatusCode::TOO_MANY_REQUESTS);
        let WsError::Http(response) = connect_async(server.url("target_language=en"))
            .await
            .unwrap_err()
        else {
            panic!("expected busy")
        };
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        socket.close(None).await.unwrap();
    }

    #[tokio::test]
    async fn emits_segment_before_finish_and_flushes_short_tail_before_done() {
        let server = TestServer::start(Arc::new(FakeBackend::default()), config()).await;
        let (mut socket, _) = connect_async(server.url("target_language=en"))
            .await
            .unwrap();
        assert_eq!(event(&mut socket).await["type"], "ready");
        for _ in 0..5 {
            socket
                .send(ClientMessage::Binary(vec![0; 640].into()))
                .await
                .unwrap();
        }
        let transcript = event(&mut socket).await;
        assert_eq!(transcript["type"], "transcript");
        assert_eq!(transcript["end_ms"], 100);
        assert_eq!(event(&mut socket).await["type"], "translation");
        socket
            .send(ClientMessage::Binary(vec![0; 640].into()))
            .await
            .unwrap();
        socket
            .send(ClientMessage::Text(r#"{"type":"finish"}"#.into()))
            .await
            .unwrap();
        let tail = event(&mut socket).await;
        assert_eq!(tail["start_ms"], 100);
        assert_eq!(tail["end_ms"], 120);
        assert_eq!(event(&mut socket).await["type"], "translation");
        let done = event(&mut socket).await;
        assert_eq!(done["type"], "done");
        assert_eq!(done["duration_ms"], 120);
        assert_eq!(done["segments_count"], 2);
    }

    #[tokio::test]
    async fn malformed_text_and_pcm_return_terminal_errors() {
        for message in [
            ClientMessage::Text(r#"{"type":"unknown"}"#.into()),
            ClientMessage::Binary(vec![0].into()),
            ClientMessage::Binary(Vec::new().into()),
        ] {
            let server = TestServer::start(Arc::new(FakeBackend::default()), config()).await;
            let (mut socket, _) = connect_async(server.url("target_language=en"))
                .await
                .unwrap();
            event(&mut socket).await;
            socket.send(message).await.unwrap();
            let error = event(&mut socket).await;
            assert_eq!(error["type"], "error");
            assert!(matches!(
                error["code"].as_str(),
                Some("invalid_request" | "invalid_audio")
            ));
            let close = timeout(Duration::from_secs(2), socket.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(matches!(close, ClientMessage::Close(_)));
        }
    }

    #[tokio::test]
    async fn backend_error_preserves_context_and_restart_instructions() {
        let server = TestServer::start(
            Arc::new(FakeBackend {
                fail: true,
                ..Default::default()
            }),
            config(),
        )
        .await;
        let (mut socket, _) = connect_async(server.url("target_language=en"))
            .await
            .unwrap();
        assert_eq!(event(&mut socket).await["type"], "ready");
        let error = event(&mut socket).await;
        assert_eq!(error["code"], "stream_failed");
        let message = error["message"].as_str().unwrap();
        assert!(message.contains("第 2 段翻译失败"));
        assert!(message.contains("LLM 推理超时"));
        assert!(message.contains("请重启服务"));
        assert!(matches!(
            socket.next().await.unwrap().unwrap(),
            ClientMessage::Close(_)
        ));
    }

    #[tokio::test]
    async fn disconnected_client_keeps_shared_admission_until_active_work_returns() {
        let (entered, mut started) = mpsc::unbounded_channel();
        let release = Arc::new(Semaphore::new(0));
        let server = TestServer::start(
            Arc::new(FakeBackend {
                entered: Some(entered),
                release: Some(release.clone()),
                ..Default::default()
            }),
            config(),
        )
        .await;
        let (mut socket, _) = connect_async(server.url("target_language=en"))
            .await
            .unwrap();
        event(&mut socket).await;
        socket
            .send(ClientMessage::Binary(vec![0; FRAME_SAMPLES * 2].into()))
            .await
            .unwrap();
        timeout(Duration::from_secs(2), started.recv())
            .await
            .unwrap()
            .unwrap();
        socket.close(None).await.unwrap();
        drop(socket);
        assert_eq!(server.batch_status().await, StatusCode::TOO_MANY_REQUESTS);
        release.add_permits(1);
        timeout(Duration::from_secs(2), async {
            while server.batch_status().await == StatusCode::TOO_MANY_REQUESTS {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(server.batch_status().await, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn full_canonical_audio_queue_errors_instead_of_dropping_samples() {
        let (entered, mut started) = mpsc::unbounded_channel();
        let release = Arc::new(Semaphore::new(0));
        let server = TestServer::start(
            Arc::new(FakeBackend {
                entered: Some(entered),
                release: Some(release.clone()),
                ..Default::default()
            }),
            config(),
        )
        .await;
        let (mut socket, _) = connect_async(server.url("target_language=en"))
            .await
            .unwrap();
        event(&mut socket).await;
        socket
            .send(ClientMessage::Binary(vec![0; FRAME_SAMPLES * 2].into()))
            .await
            .unwrap();
        timeout(Duration::from_secs(2), started.recv())
            .await
            .unwrap()
            .unwrap();
        for _ in 0..13 {
            socket
                .send(ClientMessage::Binary(vec![0; MAX_CHUNK_BYTES].into()))
                .await
                .unwrap();
        }
        let error = event(&mut socket).await;
        assert_eq!(error["type"], "error");
        assert_eq!(error["code"], "audio_buffer_full");
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .contains("网络输入到 VAD")
        );
        assert_eq!(server.batch_status().await, StatusCode::TOO_MANY_REQUESTS);
        release.add_permits(1);
    }
}
