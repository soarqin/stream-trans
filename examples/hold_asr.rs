//! Keep a verified native ASR model/session resident while a separate LLM is tested.
//! This example never records audio or runs VAD.

#[path = "../src/asr.rs"]
mod asr;

use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "识别一次已有 WAV，保持 BF16 ASR 常驻以验证 LLM 合载")]
struct Args {
    #[arg(long)]
    asr_library: PathBuf,
    #[arg(long)]
    asr_model: PathBuf,
    #[arg(long)]
    audio: PathBuf,
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(0..=86_400))]
    hold_seconds: u64,
    #[arg(long)]
    ready_file: Option<PathBuf>,
}

fn read_audio(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).context("无法读取已有 WAV 音频")?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate == 16_000
            && spec.channels == 1
            && spec.sample_format == hound::SampleFormat::Int
            && spec.bits_per_sample == 16,
        "音频必须为 16 kHz、单声道、16 位 PCM WAV"
    );
    ensure!(
        reader.duration() > 0 && reader.duration() <= 15 * 16_000,
        "请提供非空且不超过 15 秒的单段 WAV；该工具不使用 VAD"
    );
    reader
        .into_samples::<i16>()
        .map(|sample| sample.map(|value| value as f32 / 32768.0))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("读取 PCM16 音频采样失败")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let _ = tracing_subscriber::fmt().with_writer(io::stderr).try_init();
    let samples = read_audio(&args.audio)?;
    let started = Instant::now();
    let mut model = asr::Asr::load(&args.asr_library, &args.asr_model, args.gpu)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let text = model.transcribe(&samples, "auto", Duration::from_secs(120))?;
    let inference_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ready = json!({
        "event":"ready", "pid":std::process::id(), "gpu":args.gpu,
        "model":args.asr_model, "library":args.asr_library, "audio":args.audio,
        "audio_samples":samples.len(), "audio_duration_ms":samples.len() as f64 / 16.0,
        "load_ms":load_ms, "inference_ms":inference_ms, "text":text,
        "hold_seconds":args.hold_seconds, "asr_backend":"cuda", "asr_precision":"BF16"
    });
    let line = serde_json::to_string(&ready)?;
    if let Some(path) = &args.ready_file {
        std::fs::write(path, format!("{line}\n")).context("无法保存 ASR ready JSON")?;
    }
    println!("{line}");
    io::stdout().flush()?;
    let reason = tokio::select! {
        () = tokio::time::sleep(Duration::from_secs(args.hold_seconds)) => "hold_elapsed",
        signal = tokio::signal::ctrl_c() => {
            signal.context("无法监听 Ctrl+C，ASR 模型将释放")?;
            "ctrl_c"
        }
    };
    drop(model);
    println!(
        "{}",
        json!({"event":"released", "pid":std::process::id(), "reason":reason})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(rate: u32, samples: &[i16]) -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut writer = hound::WavWriter::create(
            file.path(),
            hound::WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in samples {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();
        file
    }

    #[test]
    fn pcm16_audio_is_normalized_and_bad_input_rejected_without_loading_asr() {
        let file = wav(16_000, &[i16::MIN, 0, i16::MAX]);
        assert_eq!(
            read_audio(file.path()).unwrap(),
            [-1.0, 0.0, 32767.0 / 32768.0]
        );
        assert!(read_audio(wav(48_000, &[0]).path()).is_err());
        assert!(read_audio(wav(16_000, &[]).path()).is_err());
        assert!(read_audio(wav(16_000, &vec![0; 15 * 16_000 + 1]).path()).is_err());
    }

    #[test]
    fn hold_defaults_and_optional_ready_file_parse_without_gpu() {
        let args = Args::try_parse_from([
            "hold_asr",
            "--asr-library",
            "native.dll",
            "--asr-model",
            "model.gguf",
            "--audio",
            "clip.wav",
            "--ready-file",
            "ready.json",
        ])
        .unwrap();
        assert_eq!(args.hold_seconds, 300);
        assert_eq!(args.gpu, 0);
        assert_eq!(args.ready_file.as_deref(), Some(Path::new("ready.json")));
    }
}
