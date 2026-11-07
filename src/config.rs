use std::path::Path;
use std::{fmt, net::SocketAddr, path::PathBuf};

use anyhow::{Result, ensure};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    #[value(name = "huggingface")]
    HuggingFace,
    #[value(name = "modelscope")]
    ModelScope,
}

impl fmt::Display for ModelSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::HuggingFace => "Hugging Face",
            Self::ModelScope => "ModelScope",
        })
    }
}

#[derive(Clone, Parser)]
#[command(version, about = "本地 GPU 语音识别与翻译服务")]
pub struct Config {
    #[arg(long, default_value = "127.0.0.1:3000")]
    pub bind: SocketAddr,
    /// 可选请求 key；设置后所有请求须携带 Authorization: Bearer <key>。
    #[arg(long, env = "STREAMTRANS_REQUEST_KEY", hide_env_values = true)]
    pub request_key: Option<String>,
    #[arg(long, default_value_os_t = default_cache_dir())]
    pub cache_dir: PathBuf,
    #[arg(long, default_value_os_t = default_asr_library())]
    pub asr_library: PathBuf,
    #[arg(long, default_value_os_t = default_llama_server())]
    pub llama_server: PathBuf,
    #[cfg(not(windows))]
    #[arg(long, default_value_os_t = default_ffmpeg())]
    pub ffmpeg: PathBuf,
    /// FFmpeg 9.0 进程内解码桥；配套 DLL 与此文件放在同一目录。
    #[cfg(windows)]
    #[arg(long, default_value_os_t = default_ffmpeg_library())]
    pub ffmpeg_library: PathBuf,
    /// CUDA 设备编号。
    #[arg(long, default_value_t = 0)]
    pub gpu: u32,
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..=600))]
    pub max_audio_seconds: u32,
    /// 连续录音的总时长上限（秒）；0 表示不限，音频缓冲仍有界。
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=86_400))]
    pub max_stream_seconds: u32,
    #[arg(long, default_value_t = 4096, value_parser = clap::value_parser!(u32).range(2048..=4096))]
    pub context_size: u32,
    #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(1..=3072))]
    pub max_tokens: u32,
    /// UTF-8 TSV 术语表，三列为 source、target_language、translation；启动时加载一次。
    #[arg(long)]
    pub glossary: Option<PathBuf>,
    /// 完整术语表的模型 token 上限；不会自动筛选或截断条目。
    #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u32).range(1..=4096))]
    pub glossary_max_tokens: u32,
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    pub inference_timeout_seconds: u64,
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    pub startup_timeout_seconds: u64,
    /// 模型下载源，统一用于 ASR、VAD 和翻译模型。
    #[arg(long, value_enum, default_value = "huggingface")]
    pub model_source: ModelSource,
    /// 覆盖当前下载源的 ASR revision；省略时使用该源的固定版本。
    #[arg(long)]
    pub asr_revision: Option<String>,
    /// 覆盖当前下载源的翻译模型 revision；省略时使用该源的固定版本。
    #[arg(long)]
    pub llm_revision: Option<String>,
    /// 覆盖当前下载源的 VAD revision；省略时使用该源的固定版本。
    #[arg(long)]
    pub vad_revision: Option<String>,
    /// 仅下载模型，不加载 GPU 或启动 HTTP 服务。
    #[arg(long)]
    pub download_only: bool,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if let Some(key) = &self.request_key {
            ensure!(
                !key.is_empty() && key.len() <= 256 && key.bytes().all(|b| b.is_ascii_graphic()),
                "请求 key 必须为 1–256 个可见 ASCII 字符，不能包含空白"
            );
        }
        ensure!(
            self.max_tokens + 1024 < self.context_size,
            "context-size 必须大于 max-tokens + 1024，为输入文本保留空间"
        );
        ensure!(
            self.bind.ip().is_loopback() || self.request_key.is_some(),
            "未设置请求 key 时仅允许绑定回环地址；非回环监听须设置 --request-key 或 STREAMTRANS_REQUEST_KEY"
        );
        Ok(())
    }
}

fn default_asr_library() -> PathBuf {
    let name = if cfg!(windows) {
        "transcribe.dll"
    } else {
        "libtranscribe.so"
    };
    if let Some(packaged) = packaged_backend(&bundled_path("runtime"), name, "asr") {
        return packaged;
    }
    if cfg!(windows) {
        ".tools/transcribe-shared/transcribe.dll".into()
    } else {
        ".tools/transcribe/libtranscribe.so".into()
    }
}

fn bundled_path(relative: &str) -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join(relative)))
        .unwrap_or_else(|| relative.into())
}

fn default_llama_server() -> PathBuf {
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    if let Some(packaged) = packaged_backend(&bundled_path("runtime"), name, "llama") {
        packaged
    } else if cfg!(windows) {
        ".tools/llama-shared/llama-server.exe".into()
    } else {
        ".tools/llama/llama-server".into()
    }
}

fn packaged_backend(runtime: &Path, name: &str, legacy: &str) -> Option<PathBuf> {
    [runtime.join(name), runtime.join(legacy).join(name)]
        .into_iter()
        .find(|path| path.is_file())
}

#[cfg(not(windows))]
fn default_ffmpeg() -> PathBuf {
    packaged_ffmpeg(&bundled_path("runtime")).unwrap_or_else(|| "ffmpeg".into())
}

#[cfg(not(windows))]
fn packaged_ffmpeg(runtime: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    [runtime.join(name), runtime.join("ffmpeg").join(name)]
        .into_iter()
        .find(|path| path.is_file())
}

#[cfg(windows)]
fn default_ffmpeg_library() -> PathBuf {
    let packaged = bundled_path("runtime/streamtrans-ffmpeg.dll");
    if packaged.is_file() {
        packaged
    } else {
        ".tools/ffmpeg-shared/bin/streamtrans-ffmpeg.dll".into()
    }
}

fn default_cache_dir() -> PathBuf {
    if bundled_path("runtime").is_dir() {
        bundled_path("models")
    } else {
        "models".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packaged_backends_prefer_shared_directory_and_accept_legacy_layout() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let runtime = temporary.path();
        for (name, legacy) in [("transcribe.dll", "asr"), ("llama-server.exe", "llama")] {
            assert!(packaged_backend(runtime, name, legacy).is_none());
            std::fs::create_dir(runtime.join(legacy))?;
            let old = runtime.join(legacy).join(name);
            std::fs::write(&old, [])?;
            assert_eq!(packaged_backend(runtime, name, legacy), Some(old));
            let shared = runtime.join(name);
            std::fs::write(&shared, [])?;
            assert_eq!(packaged_backend(runtime, name, legacy), Some(shared));
        }
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn packaged_ffmpeg_prefers_shared_runtime_and_accepts_legacy_layout() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let runtime = temporary.path();
        let name = if cfg!(windows) {
            "ffmpeg.exe"
        } else {
            "ffmpeg"
        };
        assert!(packaged_ffmpeg(runtime).is_none());
        std::fs::create_dir(runtime.join("ffmpeg"))?;
        let legacy = runtime.join("ffmpeg").join(name);
        std::fs::write(&legacy, [])?;
        assert_eq!(packaged_ffmpeg(runtime), Some(legacy));
        let shared = runtime.join(name);
        std::fs::write(&shared, [])?;
        assert_eq!(packaged_ffmpeg(runtime), Some(shared));
        Ok(())
    }

    #[test]
    fn non_loopback_binding_requires_a_valid_request_key() {
        let explicit =
            Config::try_parse_from(["streamtrans", "--request-key", "test-request-key"]).unwrap();
        assert!(explicit.bind.ip().is_loopback());
        assert_eq!(explicit.request_key.as_deref(), Some("test-request-key"));
        assert!(explicit.validate().is_ok());

        let mut config = Config::try_parse_from(["streamtrans"]).unwrap();
        config.request_key = None;
        for bind in ["127.0.0.1:3000", "[::1]:3000"] {
            config.bind = bind.parse().unwrap();
            assert!(config.validate().is_ok());
        }
        for bind in ["0.0.0.0:3000", "192.168.1.10:3000", "[::]:3000"] {
            config.bind = bind.parse().unwrap();
            config.request_key = None;
            assert!(config.validate().is_err());
            config.request_key = Some("test-request-key".into());
            assert!(config.validate().is_ok());
        }
        for key in [
            "",
            " ",
            " leading",
            "trailing ",
            "new\nline",
            "tab\tkey",
            "中文",
        ] {
            config.request_key = Some(key.into());
            let error = config.validate().unwrap_err().to_string();
            assert!(error.contains("ASCII"));
        }
        config.request_key = Some("x".repeat(257));
        assert!(config.validate().is_err());
        config.request_key = Some("x".repeat(256));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn hy_mt_defaults_preserve_translation_output_budget() {
        let config = Config::try_parse_from(["streamtrans"]).unwrap();
        assert_eq!(config.context_size, 4096);
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.model_source, ModelSource::HuggingFace);
        assert!(config.llm_revision.is_none());
        assert!(Config::try_parse_from(["streamtrans", "--llm-thinking"]).is_err());
    }

    #[test]
    fn download_source_and_revision_overrides_are_explicit() {
        let config = Config::try_parse_from([
            "streamtrans",
            "--model-source",
            "modelscope",
            "--llm-revision",
            "custom-revision",
            "--download-only",
        ])
        .unwrap();
        assert_eq!(config.model_source, ModelSource::ModelScope);
        assert_eq!(config.llm_revision.as_deref(), Some("custom-revision"));
        assert!(config.download_only);
        assert!(Config::try_parse_from(["streamtrans", "--model-source", "unknown"]).is_err());
        assert_eq!(
            serde_json::to_string(&ModelSource::HuggingFace).unwrap(),
            "\"huggingface\""
        );
        assert_eq!(
            serde_json::to_string(&ModelSource::ModelScope).unwrap(),
            "\"modelscope\""
        );
    }
}
