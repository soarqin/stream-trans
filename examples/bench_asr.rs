//! Release benchmark: model loading, first inference, warmups, then measured repeats.
//! Requires the same native CUDA libraries and downloaded models as the service.

#[path = "../src/asr.rs"]
mod asr;
#[cfg(windows)]
#[path = "../src/native_library.rs"]
mod native_library;
#[path = "../src/vad.rs"]
#[allow(dead_code)]
mod vad;

use std::{
    fs::File,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Serialize;

#[derive(Parser, Serialize)]
#[command(about = "分开测量模型加载、首次推理和预热后的 FSMN-VAD / ASR 速度")]
struct Args {
    #[arg(long)]
    asr_library: PathBuf,
    #[arg(long)]
    asr_model: PathBuf,
    #[arg(long)]
    vad_dir: PathBuf,
    #[arg(long, default_value = ".tools/smoke/zh16.wav")]
    audio: PathBuf,
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    /// 首次推理之外的额外预热次数，不计入统计。
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=100))]
    warmup: u32,
    /// 正式测量次数，至少 5 次。
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(5..=1000))]
    runs: u32,
    #[arg(long, default_value = "zh")]
    language: String,
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u64).range(1..=3600))]
    timeout_seconds: u64,
    /// 保存 UTF-8 JSON；同时向标准输出写出完整报告。
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Serialize)]
struct Segment {
    start_ms: u64,
    end_ms: u64,
    asr_ms: f64,
    text: String,
}

#[derive(Serialize)]
struct Round {
    vad_ms: f64,
    asr_ms: f64,
    total_ms: f64,
    segments: Vec<Segment>,
}

#[derive(Serialize)]
struct Stats {
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl Stats {
    fn new(values: impl IntoIterator<Item = f64>) -> Self {
        let mut values: Vec<_> = values.into_iter().collect();
        assert!(!values.is_empty());
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        let median_ms = if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        };
        Self {
            median_ms,
            min_ms: values[0],
            max_ms: values[values.len() - 1],
        }
    }
}

#[derive(Serialize)]
struct SegmentStats {
    start_ms: u64,
    end_ms: u64,
    asr: Stats,
}

#[derive(Serialize)]
struct Report {
    settings: Args,
    audio_duration_ms: f64,
    cold_definition: &'static str,
    model_load_ms: ModelLoad,
    cold: Round,
    warmups: Vec<Round>,
    measured: Vec<Round>,
    measured_summary: Summary,
}

#[derive(Serialize)]
struct ModelLoad {
    vad: f64,
    asr: f64,
    total: f64,
}

#[derive(Serialize)]
struct Summary {
    vad: Stats,
    asr: Stats,
    total: Stats,
    segments: Vec<SegmentStats>,
    all_transcripts_match_cold: bool,
}

fn ms(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

fn infer(vad: &vad::Vad, asr: &mut asr::Asr, samples: &[f32], args: &Args) -> Result<Round> {
    let total = Instant::now();
    let started = Instant::now();
    let intervals = vad.detect(samples)?;
    let vad_ms = ms(started.elapsed());
    let started = Instant::now();
    let mut segments = Vec::with_capacity(intervals.len());
    for (start_ms, end_ms) in intervals {
        let start = usize::try_from(start_ms)?
            .checked_mul(16)
            .context("语音段起点过大")?;
        let end = usize::try_from(end_ms)?
            .checked_mul(16)
            .context("语音段终点过大")?
            .min(samples.len());
        ensure!(start < end, "FSMN-VAD 返回了空语音段或超出音频的时间范围");
        let started = Instant::now();
        let text = asr.transcribe(
            &samples[start..end],
            &args.language,
            Duration::from_secs(args.timeout_seconds),
        )?;
        segments.push(Segment {
            start_ms,
            end_ms,
            asr_ms: ms(started.elapsed()),
            text,
        });
    }
    let asr_ms = ms(started.elapsed());
    Ok(Round {
        vad_ms,
        asr_ms,
        total_ms: ms(total.elapsed()),
        segments,
    })
}

fn main() -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "请使用 cargo run --release --example bench_asr 执行基准测试"
    );
    let args = Args::parse();
    let reader = hound::WavReader::open(&args.audio).context("无法读取测试音频")?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate == 16000
            && spec.channels == 1
            && spec.sample_format == hound::SampleFormat::Int
            && spec.bits_per_sample == 16,
        "基准音频必须为 16 kHz、单声道、16 位 PCM WAV"
    );
    let samples = reader
        .into_samples::<i16>()
        .map(|sample| sample.map(|sample| sample as f32 / 32768.0))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(!samples.is_empty(), "测试音频不能为空");

    let loading = Instant::now();
    let started = Instant::now();
    let vad = vad::Vad::load(&args.vad_dir)?;
    let vad_load_ms = ms(started.elapsed());
    let started = Instant::now();
    let mut asr = asr::Asr::load(&args.asr_library, &args.asr_model, args.gpu)?;
    let model_load_ms = ModelLoad {
        vad: vad_load_ms,
        asr: ms(started.elapsed()),
        total: ms(loading.elapsed()),
    };

    let cold = infer(&vad, &mut asr, &samples, &args).context("首次推理失败")?;
    ensure!(
        !cold.segments.is_empty(),
        "测试音频未检测到语音，无法测量 ASR"
    );
    let mut warmups = Vec::new();
    for _ in 0..args.warmup {
        warmups.push(infer(&vad, &mut asr, &samples, &args).context("预热推理失败")?);
    }
    let mut measured = Vec::new();
    for _ in 0..args.runs {
        measured.push(infer(&vad, &mut asr, &samples, &args).context("正式测量失败")?);
    }
    let same_intervals = |round: &Round| {
        round
            .segments
            .iter()
            .map(|s| (s.start_ms, s.end_ms))
            .eq(cold.segments.iter().map(|s| (s.start_ms, s.end_ms)))
    };
    ensure!(
        warmups.iter().chain(&measured).all(same_intervals),
        "重复推理的 VAD 分段不一致，不能按段比较计时"
    );
    let summary = Summary {
        vad: Stats::new(measured.iter().map(|round| round.vad_ms)),
        asr: Stats::new(measured.iter().map(|round| round.asr_ms)),
        total: Stats::new(measured.iter().map(|round| round.total_ms)),
        segments: cold
            .segments
            .iter()
            .enumerate()
            .map(|(index, segment)| SegmentStats {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                asr: Stats::new(measured.iter().map(|round| round.segments[index].asr_ms)),
            })
            .collect(),
        all_transcripts_match_cold: warmups.iter().chain(&measured).all(|round| {
            round
                .segments
                .iter()
                .map(|segment| &segment.text)
                .eq(cold.segments.iter().map(|segment| &segment.text))
        }),
    };
    let report = Report {
        audio_duration_ms: samples.len() as f64 / 16.0,
        cold_definition: "同一新进程加载模型后的首次推理；未清除操作系统文件缓存。正式统计排除模型加载、首次推理和额外预热。",
        settings: args,
        model_load_ms,
        cold,
        warmups,
        measured,
        measured_summary: summary,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &report.settings.output {
        let mut file =
            File::create(path).with_context(|| format!("无法创建报告：{}", path.display()))?;
        writeln!(file, "{json}")?;
    }
    println!("{json}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_median_works_for_odd_and_even_samples() {
        let stats = Stats::new([9.0, 1.0, 4.0, 2.0, 3.0]);
        assert_eq!(
            (stats.median_ms, stats.min_ms, stats.max_ms),
            (3.0, 1.0, 9.0)
        );
        assert_eq!(Stats::new([9.0, 1.0, 4.0, 2.0]).median_ms, 3.0);
    }
}
