use std::{path::Path, process::Stdio, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, State, multipart::MultipartRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};
use tokio::{
    process::Command,
    sync::Semaphore,
    time::{Instant, timeout},
};

use crate::{
    config::Config,
    runtime::{Backend, Recognition},
};

const MAX_UPLOAD: usize = 32 * 1024 * 1024;

#[derive(Clone)]
pub(super) struct AppState {
    pub(super) backend: Arc<dyn Backend>,
    pub(super) config: Config,
    pub(super) gate: Arc<Semaphore>,
}

pub fn router(backend: Arc<dyn Backend>, config: Config) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/audio/transcriptions", post(transcribe_audio))
        .route("/v1/audio/translations", post(translate_audio))
        .route(
            "/v1/audio/transcriptions/stream",
            get(crate::stream::transcribe_upgrade),
        )
        .route("/v1/audio/translations/stream", get(crate::stream::upgrade))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD + 8192))
        .with_state(AppState {
            backend,
            config,
            gate: Arc::new(Semaphore::new(1)),
        })
}

async fn health(State(state): State<AppState>) -> Response {
    if state.backend.healthy().await {
        Json(json!({"status":"ok", "device":"cuda", "models":model_names()})).into_response()
    } else {
        error(
            StatusCode::SERVICE_UNAVAILABLE,
            "backend_unavailable",
            "模型进程不可用，请查看日志并重启服务",
        )
        .into_response()
    }
}

async fn translate_audio(
    State(state): State<AppState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Value>, ApiError> {
    process_audio(state, multipart, true).await
}

async fn transcribe_audio(
    State(state): State<AppState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Value>, ApiError> {
    process_audio(state, multipart, false).await
}

async fn process_audio(
    state: AppState,
    multipart: Result<Multipart, MultipartRejection>,
    translate: bool,
) -> Result<Json<Value>, ApiError> {
    let started = Instant::now();
    let permit = state.gate.clone().try_acquire_owned().map_err(|_| {
        error(
            StatusCode::TOO_MANY_REQUESTS,
            "busy",
            "GPU 正在处理其他音频，请稍后重试",
        )
    })?;
    let mut multipart = multipart.map_err(|_| bad_request("请求必须使用 multipart/form-data"))?;
    let mut audio = None;
    let mut target = None;
    let mut source = None;
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "target_language" if !translate => {
                return Err(bad_request(
                    "仅识别接口不接受 target_language，请使用 translations 接口进行翻译",
                ));
            }
            "audio" => {
                if audio.is_some() {
                    return Err(bad_request("audio 字段重复"));
                }
                let bytes = field.bytes().await.map_err(multipart_error)?;
                if bytes.is_empty() {
                    return Err(bad_request("audio 不能为空"));
                }
                if bytes.len() > MAX_UPLOAD {
                    return Err(error(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "audio_too_large",
                        "音频超过 32 MiB，请缩小文件后重试",
                    ));
                }
                audio = Some(bytes);
            }
            "target_language" | "source_language" => {
                let slot = if name == "target_language" {
                    &mut target
                } else {
                    &mut source
                };
                if slot.is_some() {
                    return Err(bad_request(format!("{name} 字段重复")));
                }
                let bytes = field.bytes().await.map_err(multipart_error)?;
                if bytes.len() > 192 {
                    return Err(bad_request(format!("{name} 超过长度限制")));
                }
                *slot = Some(
                    String::from_utf8(bytes.to_vec())
                        .map_err(|_| bad_request(format!("{name} 必须为 UTF-8 文本")))?,
                );
            }
            _ => return Err(bad_request(format!("不支持的字段：{name}"))),
        }
    }
    let audio = audio.ok_or_else(|| bad_request("缺少 audio 字段"))?;
    let target = if translate {
        Some(validate_target(
            &target.ok_or_else(|| bad_request("缺少 target_language 字段"))?,
        )?)
    } else {
        None
    };
    let source = source.unwrap_or_else(|| "auto".into());
    if !matches!(source.as_str(), "auto" | "zh" | "en" | "ja") {
        return Err(bad_request("source_language 仅支持 auto、zh、en、ja"));
    }
    // 请求断开时，已开始的推理仍持有 permit 和临时文件，防止重叠访问同一个 GPU。
    tokio::spawn(async move {
        let _permit = permit;
        let temp = tempfile::tempdir().map_err(internal)?;
        let input = temp.path().join("input.audio");
        let wav = temp.path().join("audio.wav");
        tokio::fs::write(&input, audio).await.map_err(internal)?;
        let duration_ms = normalize_audio(&state.config, &input, &wav).await?;
        let asr_started = Instant::now();
        let recognition = state.backend.recognize(&wav, &source).await
            .map_err(|e| backend_error("asr_failed", e))?;
        let asr_ms = asr_started.elapsed().as_millis();
        let (translation, translation_ms) = if let Some(target) = target.as_deref() {
            let llm_started = Instant::now();
            let translation = if recognition.text.trim().is_empty() {
                String::new()
            } else {
                state.backend.translate(&recognition.text, target).await.map_err(|e| {
                    let mut error = backend_error("translation_failed", e);
                    error.recognition = Some(recognition.clone());
                    error
                })?
            };
            (Some(translation), llm_started.elapsed().as_millis())
        } else {
            (None, 0)
        };
        Ok(Json(json!({
            "request_id":uuid::Uuid::new_v4().to_string(),
            "mode":if translate {"translate"} else {"transcribe"},
            "text":recognition.text, "translation":translation, "segments":recognition.segments,
            "source_language":source, "target_language":target, "duration_ms":duration_ms,
            "elapsed_ms":{"asr":asr_ms,"translation":translation_ms,"total":started.elapsed().as_millis()},
            "models":model_names()
        })))
    }).await.map_err(internal)?
}

pub(super) fn validate_target(value: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if value.is_empty()
        || value.chars().count() > 64
        || !value
            .chars()
            .all(|c| c.is_alphabetic() || c == ' ' || c == '-')
    {
        return Err(bad_request(
            "target_language 应为语言代码或名称，最多 64 个字符，仅允许字母、汉字、空格与连字符",
        ));
    }
    Ok(crate::glossary::normalize_target(value))
}

async fn normalize_audio(config: &Config, input: &Path, wav: &Path) -> Result<u64, ApiError> {
    let result = timeout(
        Duration::from_secs(30),
        Command::new(&config.ffmpeg)
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-protocol_whitelist",
                "file,pipe",
            ])
            .arg("-i")
            .arg(input)
            .args([
                "-map",
                "0:a:0",
                "-vn",
                "-t",
                &(config.max_audio_seconds + 1).to_string(),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(wav)
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| {
        error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "audio_decode_timeout",
            "音频解码超时，请缩短或重新编码音频后重试",
        )
    })?
    .map_err(internal)?;
    if !result.status.success() {
        tracing::warn!(details = %String::from_utf8_lossy(&result.stderr), "音频解码失败");
        return Err(error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_audio",
            "无法解码音频，请提供有效的 WAV、MP3、FLAC、M4A 或其他 FFmpeg 支持的音频文件",
        ));
    }
    let reader = hound::WavReader::open(wav).map_err(internal)?;
    let samples = reader.duration();
    if samples == 0 {
        return Err(error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "empty_audio",
            "音频中没有可解码的采样数据",
        ));
    }
    if samples > config.max_audio_seconds * 16000 {
        return Err(error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "audio_too_long",
            format!(
                "音频超过 {} 秒，请分段上传或调整 --max-audio-seconds",
                config.max_audio_seconds
            ),
        ));
    }
    Ok(u64::from(samples) * 1000 / 16000)
}

fn model_names() -> Value {
    json!({
        "asr": "Fun-ASR-Nano-2512",
        "asr_precision": "BF16 matrix weights; FP32 audio activations",
        "vad": "FSMN-VAD",
        "vad_device": "cpu",
        "llm": "Hy-MT2-1.8B",
        "llm_quantization": "Q6_K",
        "llm_backend": "llama.cpp"
    })
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    recognition: Option<Recognition>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut body = json!({"error":{"code":self.code,"message":self.message}});
        if let Some(recognition) = self.recognition {
            body["recognition"] = json!(recognition);
        }
        (self.status, Json(body)).into_response()
    }
}

pub(super) fn error(
    status: StatusCode,
    code: &'static str,
    message: impl Into<String>,
) -> ApiError {
    ApiError {
        status,
        code,
        message: message.into(),
        recognition: None,
    }
}

pub(super) fn bad_request(message: impl Into<String>) -> ApiError {
    error(StatusCode::BAD_REQUEST, "invalid_request", message)
}

fn multipart_error(cause: axum::extract::multipart::MultipartError) -> ApiError {
    if cause.status() == StatusCode::PAYLOAD_TOO_LARGE {
        error(cause.status(), "audio_too_large", "请求超过 32 MiB 限制")
    } else {
        bad_request("multipart 数据格式有误，请重新上传")
    }
}

fn internal(cause: impl std::fmt::Display) -> ApiError {
    tracing::error!(error = %cause, "内部处理失败");
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        "内部处理失败，请查看服务日志",
    )
}

fn backend_error(code: &'static str, cause: anyhow::Error) -> ApiError {
    tracing::error!(error = %cause, "模型推理失败");
    error(StatusCode::BAD_GATEWAY, code, cause.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Segment;
    use async_trait::async_trait;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use clap::Parser;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;

    #[derive(Default)]
    struct FakeBackend {
        translation_fails: bool,
        recognitions: AtomicUsize,
        translations: AtomicUsize,
    }

    #[async_trait]
    impl Backend for FakeBackend {
        async fn recognize(&self, _: &Path, _: &str) -> anyhow::Result<Recognition> {
            self.recognitions.fetch_add(1, Ordering::SeqCst);
            Ok(Recognition {
                text: "你好。".into(),
                segments: vec![Segment {
                    start_ms: 0,
                    end_ms: 100,
                    text: "你好。".into(),
                }],
            })
        }
        async fn translate(&self, _: &str, _: &str) -> anyhow::Result<String> {
            self.translations.fetch_add(1, Ordering::SeqCst);
            if self.translation_fails {
                anyhow::bail!("translation test failure")
            }
            Ok("Hello.".into())
        }
        async fn healthy(&self) -> bool {
            true
        }
    }

    fn app(fails: bool) -> Router {
        router(
            Arc::new(FakeBackend {
                translation_fails: fails,
                ..Default::default()
            }),
            Config::parse_from(["streamtrans"]),
        )
    }

    fn multipart(audio: &[u8], fields: &[(&str, &str)]) -> Request<Body> {
        let mut body = Vec::new();
        for (key, value) in fields {
            body.extend_from_slice(format!("--test-boundary\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n").as_bytes());
        }
        body.extend_from_slice(b"--test-boundary\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n");
        body.extend_from_slice(audio);
        body.extend_from_slice(b"\r\n--test-boundary--\r\n");
        Request::post("/v1/audio/translations")
            .header(
                "content-type",
                "multipart/form-data; boundary=test-boundary",
            )
            .body(Body::from(body))
            .unwrap()
    }

    fn wav() -> Vec<u8> {
        let mut cursor = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for _ in 0..1600 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }

    fn transcription(audio: &[u8], fields: &[(&str, &str)]) -> Request<Body> {
        let mut request = multipart(audio, fields);
        *request.uri_mut() = "/v1/audio/transcriptions".parse().unwrap();
        request
    }

    #[tokio::test]
    async fn transcription_returns_nonempty_asr_without_calling_translation() {
        let backend = Arc::new(FakeBackend {
            translation_fails: true,
            ..Default::default()
        });
        let app = router(backend.clone(), Config::parse_from(["streamtrans"]));
        let response = app
            .clone()
            .oneshot(transcription(&wav(), &[]))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(body["mode"], "transcribe");
        assert_eq!(body["text"], "你好。");
        assert_eq!(body["segments"][0]["text"], "你好。");
        assert_eq!(body["source_language"], "auto");
        assert!(body["target_language"].is_null());
        assert!(body["translation"].is_null());
        assert_eq!(body["elapsed_ms"]["translation"], 0);
        assert_eq!(body["duration_ms"], 100);
        assert_eq!(backend.recognitions.load(Ordering::SeqCst), 1);
        assert_eq!(backend.translations.load(Ordering::SeqCst), 0);
        for fields in [
            vec![("target_language", "en")],
            vec![("source_language", "fr")],
        ] {
            let response = app
                .clone()
                .oneshot(transcription(b"audio", &fields))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        assert_eq!(backend.recognitions.load(Ordering::SeqCst), 1);
        assert_eq!(backend.translations.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn validates_required_fields_before_inference() {
        let response = app(false).oneshot(multipart(b"audio", &[])).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let response = app(false)
            .oneshot(multipart(
                b"audio",
                &[("target_language", "en"), ("source_language", "fr")],
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(validate_target("en\nignore instructions").is_err());
        assert_eq!(validate_target("zh-CN").unwrap(), "简体中文");
    }

    // 需要 FFmpeg；模型后端用可控替身，覆盖上传、解码、返回结构及部分失败保留原文。
    #[tokio::test]
    async fn upload_to_translation_and_partial_failure() {
        for fails in [false, true] {
            let response = app(fails)
                .oneshot(multipart(&wav(), &[("target_language", "en")]))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                if fails {
                    StatusCode::BAD_GATEWAY
                } else {
                    StatusCode::OK
                }
            );
            let body: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                    .unwrap();
            if fails {
                assert_eq!(body["error"]["code"], "translation_failed");
                assert_eq!(body["recognition"]["text"], "你好。");
            } else {
                assert_eq!(body["text"], "你好。");
                assert_eq!(body["translation"], "Hello.");
                assert_eq!(body["duration_ms"], 100);
                assert_eq!(body["segments"][0]["end_ms"], 100);
            }
        }
    }

    #[tokio::test]
    async fn cancelled_client_keeps_audio_and_gpu_permit_until_inference_finishes() {
        struct BlockingBackend {
            entered: tokio::sync::mpsc::UnboundedSender<std::path::PathBuf>,
            release: Semaphore,
        }

        #[async_trait]
        impl Backend for BlockingBackend {
            async fn recognize(&self, path: &Path, _: &str) -> anyhow::Result<Recognition> {
                self.entered.send(path.to_owned())?;
                self.release.acquire().await?.forget();
                assert!(std::fs::read(path)?.len() > 44);
                Ok(Recognition {
                    text: String::new(),
                    segments: Vec::new(),
                })
            }

            async fn translate(&self, _: &str, _: &str) -> anyhow::Result<String> {
                panic!("silence must not invoke the translator")
            }

            async fn healthy(&self) -> bool {
                true
            }
        }

        let (entered, mut requests) = tokio::sync::mpsc::unbounded_channel();
        let backend = Arc::new(BlockingBackend {
            entered,
            release: Semaphore::new(0),
        });
        let gate = Arc::new(Semaphore::new(1));
        let app = Router::new()
            .route("/v1/audio/translations", post(translate_audio))
            .route("/v1/audio/transcriptions", post(transcribe_audio))
            .with_state(AppState {
                backend: backend.clone(),
                config: Config::parse_from(["streamtrans"]),
                gate: gate.clone(),
            });
        let request = tokio::spawn(
            app.clone()
                .oneshot(multipart(&wav(), &[("target_language", "en")])),
        );
        let audio_path = timeout(Duration::from_secs(5), requests.recv())
            .await
            .unwrap()
            .unwrap();
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        assert!(audio_path.is_file());
        let busy = app
            .clone()
            .oneshot(multipart(&wav(), &[("target_language", "en")]))
            .await
            .unwrap();
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
        let busy = app
            .clone()
            .oneshot(transcription(&wav(), &[]))
            .await
            .unwrap();
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);

        backend.release.add_permits(2);
        let completed = timeout(Duration::from_secs(5), gate.acquire())
            .await
            .unwrap()
            .unwrap();
        assert!(!audio_path.exists());
        drop(completed);
        let response = timeout(
            Duration::from_secs(5),
            app.oneshot(multipart(&wav(), &[("target_language", "en")])),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(body["text"], "");
        assert_eq!(body["translation"], "");
        assert_eq!(body["segments"], json!([]));
    }

    #[tokio::test]
    async fn rejects_corrupt_and_overlong_audio() {
        let response = app(false)
            .oneshot(multipart(b"not audio", &[("target_language", "ja")]))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let config = Config {
            max_audio_seconds: 1,
            ..Config::parse_from(["streamtrans"])
        };
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("long.wav");
        let mut writer = hound::WavWriter::create(
            &input,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..32000 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
        assert_eq!(
            normalize_audio(&config, &input, &dir.path().join("out.wav"))
                .await
                .unwrap_err()
                .code,
            "audio_too_long"
        );
    }
}
