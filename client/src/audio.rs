//! Device capture stays on its owner thread. Audio callbacks never wait for the network.
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as control,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use cpal::{
    FromSample, Sample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use tokio::sync::mpsc;

const OUTPUT_RATE: u32 = 16_000;
const CHUNK_BYTES: usize = 3_200;
const RESAMPLE_FRAMES: usize = 1_024;
const STOP_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Clone, Debug, serde::Serialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let mut result = Vec::new();
    // A separate default selector is unambiguous even when two devices have the same name.
    if let Some(device) = host.default_input_device() {
        result.push(DeviceInfo {
            id: "default".into(),
            name: format!(
                "系统默认输入（{}）",
                device.name().context("无法读取默认输入设备名称")?
            ),
            is_default: true,
        });
    }
    for (index, device) in host
        .input_devices()
        .context("无法枚举录音设备")?
        .enumerate()
    {
        let name = device.name().context("无法读取录音设备名称")?;
        result.push(DeviceInfo {
            id: format!("{index}:{name}"),
            name,
            is_default: false,
        });
    }
    Ok(result)
}

fn find_device(id: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();
    if id == "default" {
        return host
            .default_input_device()
            .context("没有默认录音设备，请连接麦克风并刷新设备列表");
    }
    let (index, expected_name) = id
        .split_once(':')
        .context("录音设备 ID 无效，请刷新设备列表")?;
    let index: usize = index.parse().context("录音设备 ID 无效，请刷新设备列表")?;
    let device = host
        .input_devices()
        .context("无法枚举录音设备")?
        .nth(index)
        .context("录音设备已移除，请刷新设备列表")?;
    ensure!(
        device
            .name()
            .context("无法读取选中录音设备名称，请刷新设备列表")?
            == expected_name,
        "录音设备列表已变化，请刷新并重新选择设备"
    );
    Ok(device)
}

enum Command {
    Stop,
    Failed(String),
}

#[derive(Clone)]
struct Failure {
    stopped: Arc<AtomicBool>,
    reported: Arc<AtomicBool>,
    commands: control::Sender<Command>,
    errors: control::Sender<String>,
}

impl Failure {
    fn report(&self, message: String) {
        self.stopped.store(true, Ordering::Release);
        if !self.reported.swap(true, Ordering::AcqRel) {
            let _ = self.errors.send(message.clone());
            let _ = self.commands.send(Command::Failed(message));
        }
    }
}

/// Safe to move into GUI state: the CPAL stream itself remains on its owner thread.
pub struct Capture {
    commands: control::Sender<Command>,
    stopped: Arc<AtomicBool>,
    completed: control::Receiver<()>,
    worker: Option<JoinHandle<Result<()>>>,
}

impl Capture {
    pub fn start(
        device_id: &str,
        sender: mpsc::Sender<Vec<u8>>,
        errors: control::Sender<String>,
    ) -> Result<Self> {
        let (commands, receiver) = control::channel();
        let (ready_tx, ready_rx) = control::sync_channel(1);
        let (completed_tx, completed) = control::channel();
        let stopped = Arc::new(AtomicBool::new(false));
        let failure = Failure {
            stopped: stopped.clone(),
            reported: Arc::new(AtomicBool::new(false)),
            commands: commands.clone(),
            errors,
        };
        let id = device_id.to_owned();
        let worker = thread::Builder::new()
            .name("microphone".into())
            .spawn(move || {
                let result = capture_worker(&id, sender, &receiver, &ready_tx, &failure);
                let _ = completed_tx.send(());
                result
            })
            .context("无法创建录音线程")?;
        let mut capture = Self {
            commands,
            stopped,
            completed,
            worker: Some(worker),
        };
        match ready_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(capture),
            Ok(Err(message)) => {
                let _ = capture.stop_inner();
                bail!(message)
            }
            Err(error) => {
                let _ = capture.stop_inner();
                bail!("录音设备未能启动：{error}。请检查麦克风权限和设备状态")
            }
        }
    }

    /// Stops callbacks, releases the device and sends the final short PCM block.
    /// Queue saturation is an error; callers must not treat it as a complete recording.
    pub fn stop(mut self) -> Result<()> {
        self.stop_inner()
    }

    fn stop_inner(&mut self) -> Result<()> {
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        self.stopped.store(true, Ordering::Release);
        let _ = self.commands.send(Command::Stop);
        match self.completed.recv_timeout(STOP_TIMEOUT) {
            Ok(()) | Err(control::RecvTimeoutError::Disconnected) => {
                worker.join().map_err(|_| anyhow!("录音线程异常退出"))?
            }
            Err(control::RecvTimeoutError::Timeout) => {
                // Rust cannot interrupt a stuck driver call. Detach rather than hang the UI;
                // the stopped flag prevents any further callback data from being accepted.
                bail!("录音设备未在 3 秒内释放，请检查设备或重新启动客户端")
            }
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.stop_inner();
    }
}

fn capture_worker(
    id: &str,
    sender: mpsc::Sender<Vec<u8>>,
    commands: &control::Receiver<Command>,
    ready: &control::SyncSender<std::result::Result<(), String>>,
    failure: &Failure,
) -> Result<()> {
    let setup = (|| -> Result<_> {
        let device = find_device(id)?;
        let supported = device
            .default_input_config()
            .context("无法读取录音设备格式，请检查麦克风权限")?;
        let config = supported.config();
        let pipeline = Arc::new(Mutex::new(Pipeline::new(
            config.sample_rate.0,
            config.channels as usize,
            sender,
        )?));
        macro_rules! build {
            ($sample:ty) => {
                build_stream::<$sample>(&device, &config, pipeline.clone(), failure.clone())
            };
        }
        let stream = match supported.sample_format() {
            SampleFormat::I8 => build!(i8),
            SampleFormat::I16 => build!(i16),
            SampleFormat::I24 => build!(cpal::I24),
            SampleFormat::I32 => build!(i32),
            SampleFormat::I64 => build!(i64),
            SampleFormat::U8 => build!(u8),
            SampleFormat::U16 => build!(u16),
            SampleFormat::U32 => build!(u32),
            SampleFormat::U64 => build!(u64),
            SampleFormat::F32 => build!(f32),
            SampleFormat::F64 => build!(f64),
            format => bail!("录音设备使用了不支持的采样格式：{format}"),
        }?;
        ensure!(!failure.stopped.load(Ordering::Acquire), "录音启动已取消");
        stream
            .play()
            .context("无法开始录音，请检查麦克风权限或设备占用")?;
        Ok((stream, pipeline))
    })();
    let (stream, pipeline) = match setup {
        Ok(result) => result,
        Err(error) => {
            let _ = ready.send(Err(format!("{error:#}")));
            return Err(error);
        }
    };
    let _ = ready.send(Ok(()));
    let command = commands.recv().unwrap_or(Command::Stop);
    failure.stopped.store(true, Ordering::Release);
    let _ = stream.pause();
    drop(stream);
    let reason = match command {
        Command::Failed(message) => Some(message),
        Command::Stop => commands.try_iter().find_map(|command| match command {
            Command::Failed(message) => Some(message),
            Command::Stop => None,
        }),
    };
    if let Some(message) = reason {
        bail!(message);
    }
    ensure!(
        !failure.reported.load(Ordering::Acquire),
        "录音因音频错误停止，请查看错误事件"
    );
    let result = pipeline
        .lock()
        .map_err(|_| anyhow!("录音处理状态异常"))?
        .finish();
    if let Err(error) = &result {
        let _ = failure.errors.send(format!("{error:#}"));
    }
    result
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    pipeline: Arc<Mutex<Pipeline>>,
    failure: Failure,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let stream_error = failure.clone();
    device
        .build_input_stream(
            config,
            move |samples: &[T], _| {
                if failure.stopped.load(Ordering::Acquire) {
                    return;
                }
                let result = match pipeline.try_lock() {
                    Ok(mut pipeline) => pipeline.push(samples),
                    Err(_) => Err(anyhow!("录音处理状态不可用，已停止录音")),
                };
                if let Err(error) = result {
                    failure.report(format!("{error:#}"));
                }
            },
            move |error| {
                stream_error.report(format!("录音设备出错，已停止录音：{error}。请重新连接设备"))
            },
            None,
        )
        .context("无法打开录音设备，请检查权限、格式或设备占用")
}

struct Pipeline {
    input_rate: u32,
    channels: usize,
    channel_sum: f32,
    channel_count: usize,
    input_frames: u64,
    output_frames: u64,
    input: Vec<f32>,
    resampler: Option<SincFixedIn<f32>>,
    output: Vec<Vec<f32>>,
    skip_delay: usize,
    pending: Vec<u8>,
    sender: mpsc::Sender<Vec<u8>>,
}

impl Pipeline {
    fn new(input_rate: u32, channels: usize, sender: mpsc::Sender<Vec<u8>>) -> Result<Self> {
        ensure!(input_rate > 0 && channels > 0, "录音设备采样率或声道数无效");
        let resampler = if input_rate == OUTPUT_RATE {
            None
        } else {
            Some(
                SincFixedIn::<f32>::new(
                    OUTPUT_RATE as f64 / input_rate as f64,
                    1.0,
                    SincInterpolationParameters {
                        sinc_len: 256,
                        f_cutoff: 0.95,
                        interpolation: SincInterpolationType::Cubic,
                        oversampling_factor: 128,
                        window: WindowFunction::BlackmanHarris2,
                    },
                    RESAMPLE_FRAMES,
                    1,
                )
                .context("无法初始化音频重采样器")?,
            )
        };
        let skip_delay = resampler.as_ref().map_or(0, Resampler::output_delay);
        let output = resampler
            .as_ref()
            .map_or_else(Vec::new, |resampler| resampler.output_buffer_allocate(true));
        Ok(Self {
            input_rate,
            channels,
            channel_sum: 0.0,
            channel_count: 0,
            input_frames: 0,
            output_frames: 0,
            input: Vec::with_capacity(RESAMPLE_FRAMES),
            resampler,
            output,
            skip_delay,
            pending: Vec::with_capacity(CHUNK_BYTES),
            sender,
        })
    }

    fn push<T: Sample>(&mut self, samples: &[T]) -> Result<()>
    where
        f32: FromSample<T>,
    {
        for sample in samples {
            let sample = sample.to_sample::<f32>();
            self.channel_sum += if sample.is_finite() { sample } else { 0.0 };
            self.channel_count += 1;
            if self.channel_count != self.channels {
                continue;
            }
            let mono = self.channel_sum / self.channels as f32;
            self.channel_sum = 0.0;
            self.channel_count = 0;
            self.input_frames += 1;
            if self.resampler.is_some() {
                self.input.push(mono);
                if self.input.len() == RESAMPLE_FRAMES {
                    self.process_block(None)?;
                }
            } else {
                self.emit(mono)?;
            }
        }
        Ok(())
    }

    fn process_block(&mut self, limit: Option<u64>) -> Result<()> {
        let (_, count) = self
            .resampler
            .as_mut()
            .context("重采样器未初始化")?
            .process_into_buffer(&[self.input.as_slice()], &mut self.output, None)
            .context("连续音频重采样失败")?;
        for index in 0..count {
            if self.skip_delay > 0 {
                self.skip_delay -= 1;
                continue;
            }
            if limit.is_some_and(|limit| self.output_frames >= limit) {
                break;
            }
            self.emit(self.output[0][index])?;
        }
        self.input.clear();
        Ok(())
    }

    fn emit(&mut self, sample: f32) -> Result<()> {
        let sample = (sample.clamp(-1.0, 1.0) * 32_768.0).round() as i16;
        self.pending.extend_from_slice(&sample.to_le_bytes());
        self.output_frames += 1;
        if self.pending.len() == CHUNK_BYTES {
            self.flush_pending()?;
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let chunk = std::mem::replace(&mut self.pending, Vec::with_capacity(CHUNK_BYTES));
        self.sender.try_send(chunk).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => {
                anyhow!("音频发送队列已满，已停止录音。请检查网络或服务处理速度")
            }
            mpsc::error::TrySendError::Closed(_) => anyhow!("音频网络连接已关闭，已停止录音"),
        })
    }

    fn finish(&mut self) -> Result<()> {
        ensure!(
            self.channel_count == 0,
            "录音在不完整声道帧处结束，无法完整提交尾帧"
        );
        if self.resampler.is_some() {
            let target = (self.input_frames * u64::from(OUTPUT_RATE)
                + u64::from(self.input_rate) / 2)
                / u64::from(self.input_rate);
            // Drain filter lookahead with zeros, remove the initial filter delay and trim
            // padding back to the exact duration of the captured input.
            while self.output_frames < target {
                self.input.resize(RESAMPLE_FRAMES, 0.0);
                self.process_block(Some(target))?;
            }
        }
        self.flush_pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(rate: u32, channels: usize, samples: &[f32], split: usize) -> Vec<Vec<u8>> {
        let (sender, mut receiver) = mpsc::channel(1024);
        let mut pipeline = Pipeline::new(rate, channels, sender).unwrap();
        for chunk in samples.chunks(split) {
            pipeline.push(chunk).unwrap();
        }
        pipeline.finish().unwrap();
        let mut output = Vec::new();
        while let Ok(chunk) = receiver.try_recv() {
            output.push(chunk);
        }
        output
    }

    fn tone(rate: u32, frequency: f32) -> Vec<f32> {
        (0..rate)
            .map(|index| {
                (std::f32::consts::TAU * frequency * index as f32 / rate as f32).sin() * 0.8
            })
            .collect()
    }

    fn rms(chunks: Vec<Vec<u8>>) -> f64 {
        let bytes = chunks.concat();
        let samples: Vec<_> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f64 / 32768.0)
            .collect();
        let interior = &samples[512..samples.len() - 512];
        (interior.iter().map(|sample| sample * sample).sum::<f64>() / interior.len() as f64).sqrt()
    }

    #[test]
    fn continuous_resampling_is_independent_of_callback_boundaries() {
        for rate in [44_100, 48_000] {
            let mono = tone(rate, 1_000.0);
            let stereo: Vec<_> = mono
                .iter()
                .flat_map(|sample| [*sample, *sample * 0.5])
                .collect();
            let reference = convert(rate, 2, &stereo, stereo.len());
            for split in [1, 17, 1023, 4096] {
                assert_eq!(reference, convert(rate, 2, &stereo, split));
            }
            assert_eq!(reference.concat().len(), OUTPUT_RATE as usize * 2);
            assert!(reference.iter().all(|chunk| chunk.len() == CHUNK_BYTES));
        }
    }

    #[test]
    fn sinc_rejects_above_nyquist_aliases() {
        for rate in [44_100, 48_000] {
            let pass = rms(convert(rate, 1, &tone(rate, 1_000.0), 257));
            let reject = rms(convert(rate, 1, &tone(rate, 12_000.0), 257));
            assert!(pass > 0.5, "pass-band gain {pass}");
            assert!(reject / pass < 0.01, "alias rejection: {reject}/{pass}");
        }
    }

    #[test]
    fn downmix_clamp_and_short_tail_are_preserved() {
        let cancelled: Vec<_> = tone(16_000, 1000.0)
            .iter()
            .flat_map(|sample| [*sample, -*sample])
            .collect();
        assert!(
            convert(16_000, 2, &cancelled, 13)
                .concat()
                .iter()
                .all(|byte| *byte == 0)
        );
        let clamped = convert(16_000, 1, &[-2.0, -1.0, 0.0, 1.0, 2.0, f32::NAN], 1).concat();
        let expected: Vec<_> = [i16::MIN, i16::MIN, 0, i16::MAX, i16::MAX, 0]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        assert_eq!(clamped, expected);
        let tail = convert(44_100, 1, &vec![0.25; 1000], 17);
        assert_eq!(tail.concat().len(), 726);
    }

    #[test]
    fn backpressure_and_disconnected_receiver_are_errors() {
        let (sender, _receiver) = mpsc::channel(1);
        let mut pipeline = Pipeline::new(16_000, 1, sender).unwrap();
        assert!(
            pipeline
                .push(&vec![0.0_f32; 3200])
                .unwrap_err()
                .to_string()
                .contains("已满")
        );
        let (sender, _receiver) = mpsc::channel(1);
        let mut pipeline = Pipeline::new(16_000, 1, sender).unwrap();
        pipeline.push(&vec![0.0_f32; 1601]).unwrap();
        assert!(pipeline.finish().unwrap_err().to_string().contains("已满"));
        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        let mut pipeline = Pipeline::new(16_000, 1, sender).unwrap();
        pipeline.push(&[0.0_f32]).unwrap();
        assert!(
            pipeline
                .finish()
                .unwrap_err()
                .to_string()
                .contains("已关闭")
        );
    }
}
