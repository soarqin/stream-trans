//! CPU-only audio comparison: DLL load, CLI startup, and complete PCM16 normalization.

#[cfg(windows)]
#[allow(dead_code)]
#[path = "../src/audio.rs"]
mod audio;

#[cfg(windows)]
mod benchmark {
    use std::{
        path::{Path, PathBuf},
        process::{Output, Stdio},
        time::{Duration, Instant},
    };

    use anyhow::{Context, Result, ensure};
    use clap::Parser;
    use serde::Serialize;
    use serde_json::json;
    use tokio::{process::Command, time::timeout};

    use super::audio::Decoder;

    #[derive(Parser, Serialize)]
    #[command(about = "独立比较 FFmpeg CLI 与 DLL 的音频解码；不加载 GPU 模型")]
    struct Args {
        #[arg(long)]
        input: PathBuf,
        /// 仅用于对照的旧版 ffmpeg.exe。
        #[arg(long)]
        ffmpeg: PathBuf,
        #[arg(long)]
        ffmpeg_library: PathBuf,
        /// 首轮与一次额外预热之外的计时轮数。
        #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..=1000))]
        warm_runs: u32,
        #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..=600))]
        max_seconds: u32,
        #[arg(long)]
        output: Option<PathBuf>,
    }

    #[derive(Serialize)]
    struct Timing {
        first_ms: f64,
        warmup_ms: f64,
        warm_ms: Vec<f64>,
        median_ms: f64,
        min_ms: f64,
        max_ms: f64,
    }

    impl Timing {
        fn new(first_ms: f64, warmup_ms: f64, warm_ms: Vec<f64>) -> Self {
            let mut sorted = warm_ms.clone();
            sorted.sort_by(f64::total_cmp);
            let middle = sorted.len() / 2;
            let median_ms = if sorted.len().is_multiple_of(2) {
                (sorted[middle - 1] + sorted[middle]) / 2.0
            } else {
                sorted[middle]
            };
            Self {
                first_ms,
                warmup_ms,
                min_ms: sorted[0],
                max_ms: sorted[sorted.len() - 1],
                median_ms,
                warm_ms,
            }
        }
    }

    fn command(args: &Args, wav: Option<&Path>) -> Command {
        let mut command = Command::new(&args.ffmpeg);
        if let Some(wav) = wav {
            command
                .args([
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                ])
                .arg(&args.input)
                .args(["-map", "0:a:0", "-vn", "-t"])
                .arg((args.max_seconds + 1).to_string())
                .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
                .arg(wav);
        } else {
            command.arg("-version");
        }
        command.stdin(Stdio::null()).kill_on_drop(true);
        command
    }

    async fn run_cli(mut command: Command) -> Result<Output> {
        let output = timeout(Duration::from_secs(30), command.output())
            .await
            .context("FFmpeg CLI 对照超时")?
            .context("无法启动 FFmpeg CLI 对照程序")?;
        ensure!(
            output.status.success(),
            "FFmpeg CLI 对照失败：{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(output)
    }

    async fn cli_version(args: &Args) -> Result<(f64, String)> {
        let started = Instant::now();
        let output = run_cli(command(args, None)).await?;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        Ok((
            elapsed,
            String::from_utf8_lossy(&output.stdout).into_owned(),
        ))
    }

    async fn cli_normalize(args: &Args, wav: &Path) -> Result<(f64, usize)> {
        let started = Instant::now();
        run_cli(command(args, Some(wav))).await?;
        let reader = hound::WavReader::open(wav).context("无法读取 CLI 规范化 WAV")?;
        let count = reader.duration() as usize;
        ensure!(count > 0, "CLI 规范化未返回音频采样");
        ensure!(
            count <= args.max_seconds as usize * 16_000,
            "CLI 音频超过配置时长"
        );
        Ok((started.elapsed().as_secs_f64() * 1000.0, count))
    }

    fn dll_normalize(decoder: &Decoder, args: &Args, wav: &Path) -> Result<(f64, usize)> {
        let started = Instant::now();
        let samples = decoder
            .decode(&args.input, args.max_seconds as usize * 16_000, 30_000)
            .map_err(|error| anyhow::anyhow!("FFmpeg DLL 对照失败：{error:?}"))?;
        ensure!(!samples.is_empty(), "DLL 规范化未返回音频采样");
        let mut writer = hound::WavWriter::create(
            wav,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        // Match src/api.rs, including float-to-PCM16 rounding and clipping.
        for sample in &samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * 32768.0).round() as i16)?;
        }
        writer.finalize()?;
        Ok((started.elapsed().as_secs_f64() * 1000.0, samples.len()))
    }

    fn pcm16(path: &Path) -> Result<Vec<i16>> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        ensure!(
            spec.channels == 1
                && spec.sample_rate == 16_000
                && spec.bits_per_sample == 16
                && spec.sample_format == hound::SampleFormat::Int,
            "规范化 WAV 格式不一致：{}",
            path.display()
        );
        Ok(reader.samples::<i16>().collect::<Result<_, _>>()?)
    }

    pub async fn run() -> Result<()> {
        let args = Args::parse();
        ensure!(
            args.input.is_file(),
            "输入文件不存在：{}",
            args.input.display()
        );
        let directory = tempfile::tempdir()?;
        let cli_wav = directory.path().join("cli.wav");
        let dll_wav = directory.path().join("dll.wav");

        let started = Instant::now();
        let decoder = Decoder::load(&args.ffmpeg_library)?;
        let dll_load_ms = started.elapsed().as_secs_f64() * 1000.0;

        let (version_first, version) = cli_version(&args).await?;
        let (version_warmup, _) = cli_version(&args).await?;
        let mut version_warm = Vec::new();
        for _ in 0..args.warm_runs {
            version_warm.push(cli_version(&args).await?.0);
        }

        let (cli_first, cli_count) = cli_normalize(&args, &cli_wav).await?;
        let (cli_warmup, count) = cli_normalize(&args, &cli_wav).await?;
        ensure!(count == cli_count, "CLI 预热采样数不一致");
        let mut cli_warm = Vec::new();
        for _ in 0..args.warm_runs {
            let (elapsed, count) = cli_normalize(&args, &cli_wav).await?;
            ensure!(count == cli_count, "CLI 多轮采样数不一致");
            cli_warm.push(elapsed);
        }

        let (dll_first, dll_count) = dll_normalize(&decoder, &args, &dll_wav)?;
        let (dll_warmup, count) = dll_normalize(&decoder, &args, &dll_wav)?;
        ensure!(count == dll_count, "DLL 预热采样数不一致");
        let mut dll_warm = Vec::new();
        for _ in 0..args.warm_runs {
            let (elapsed, count) = dll_normalize(&decoder, &args, &dll_wav)?;
            ensure!(count == dll_count, "DLL 多轮采样数不一致");
            dll_warm.push(elapsed);
        }

        let cli_samples = pcm16(&cli_wav)?;
        let dll_samples = pcm16(&dll_wav)?;
        let differences: Vec<_> = cli_samples
            .iter()
            .zip(&dll_samples)
            .map(|(&left, &right)| (i32::from(left) - i32::from(right)).abs())
            .collect();
        let max_error = differences.iter().copied().max().unwrap_or(0);
        let same_length = cli_samples.len() == dll_samples.len();
        let report = json!({
            "settings": args,
            "method": {
                "gpu_used": false,
                "warmup_runs_per_stage": 1,
                "summary_excludes_first_and_warmup": true,
                "first_run_note": "first_ms 是各阶段首次调用，不代表清空文件系统缓存；CLI 和 DLL 来自各自指定构建",
                "normalization": "mono 16000 Hz PCM16 WAV; DLL float clamp(-1,1), multiply 32768, round, saturating i16 cast",
                "timing_scope": "完整解码、重采样、写入 WAV 与采样数检查；不含 GPU 模型或 HTTP",
            },
            "cli_version": version,
            "dll_load_ms": dll_load_ms,
            "cli_version_timing": Timing::new(version_first, version_warmup, version_warm),
            "cli_normalize_timing": Timing::new(cli_first, cli_warmup, cli_warm),
            "dll_normalize_timing": Timing::new(dll_first, dll_warmup, dll_warm),
            "comparison": {
                "cli_samples": cli_samples.len(),
                "dll_samples": dll_samples.len(),
                "same_sample_count": same_length,
                "sample_count_delta": dll_samples.len() as i64 - cli_samples.len() as i64,
                "compared_samples": differences.len(),
                "max_abs_error_lsb": max_error,
                "samples_over_one_lsb": differences.iter().filter(|&&error| error > 1).count(),
                "last_160_common_samples_max_abs_error_lsb": differences.iter().rev().take(160).copied().max().unwrap_or(0),
                "within_one_lsb": same_length && max_error <= 1,
                "note": "仅比较共有采样；长度差异单独显示。超过 1 LSB 时需结合构建版本、长度与尾部检查，不能直接认定等价。",
            },
        });
        let output = serde_json::to_string_pretty(&report)?;
        if let Some(path) = &args.output {
            std::fs::write(path, &output)
                .with_context(|| format!("无法写入对照报告：{}", path.display()))?;
        }
        println!("{output}");
        Ok(())
    }
}

#[cfg(windows)]
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    benchmark::run().await
}

#[cfg(not(windows))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("此音频 DLL 对照工具仅支持 Windows x64；其他平台尚未提供对应桥接库")
}
