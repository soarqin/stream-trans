use std::{collections::VecDeque, path::Path, sync::Arc};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear, linear_no_bias};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use serde::Serialize;

const SAMPLE_RATE: usize = 16_000;
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const MEL_BINS: usize = 80;
const FFT_SIZE: usize = 512;
const FSMN_HISTORY: usize = 19;
const PROJECT_DIM: usize = 128;
const STREAM_FRAMES: usize = 64;
const MAX_SEGMENT_FRAMES: u64 = 1500;

/// Per-session CPU state. No model weights or complete waveform history are copied.
pub struct VadStream {
    frontend: FrontendStream,
    caches: Vec<Option<Tensor>>,
    endpoint: StreamEndpoints,
    received_samples: u64,
    finalized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VadEndReason {
    Silence,
    MaxDuration,
    Final,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VadSegment {
    pub start_sample: u64,
    pub end_sample: u64,
    pub reason: VadEndReason,
}

#[derive(Debug, Clone, Serialize)]
pub struct VadUpdate {
    pub segments: Vec<VadSegment>,
    /// Includes the pending silence tail until an endpoint is finally confirmed.
    pub speech_start_sample: Option<u64>,
    /// Current smoothed speech decision; a pending segment may outlive this flag.
    pub speech_active: bool,
    /// Absolute 10 ms frame timestamp, excluding unprocessed right context.
    pub processed_samples: u64,
    pub received_samples: u64,
    /// Consume `segments` first, then discard PCM older than this absolute offset.
    pub retain_from_sample: u64,
}

struct FrontendStream {
    waveform: Vec<f32>,
    fbanks: VecDeque<[f32; MEL_BINS]>,
    fbank_begin: u64,
    total_frames: u64,
    next_feature: u64,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

#[derive(Default)]
struct StreamEndpoints {
    window: [bool; 20],
    position: usize,
    count: usize,
    in_speech: bool,
    quiet_frames: usize,
    start: Option<u64>,
    earliest: u64,
    next_frame: u64,
}

// Native implementation of the pinned FunASR FSMN architecture and endpoint settings.
// Reference: https://github.com/modelscope/FunASR/tree/main/funasr/models/fsmn_vad_streaming (Apache-2.0).
pub struct Vad {
    frontend: Frontend,
    input1: Linear,
    input2: Linear,
    blocks: Vec<Block>,
    output1: Linear,
    output2: Linear,
}

struct Block {
    project: Linear,
    kernel: Tensor,
    affine: Linear,
}

struct Frontend {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    mel: Vec<Vec<(usize, f32)>>,
    shift: Vec<f32>,
    scale: Vec<f32>,
}

impl Vad {
    pub fn load(directory: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_pth(directory.join("model.pt"), DType::F32, &device)
            .context("无法读取 FSMN-VAD 原始权重")?
            .pp("encoder");
        let mut blocks = Vec::with_capacity(4);
        for index in 0..4 {
            let block = vb.pp(format!("fsmn.{index}"));
            blocks.push(Block {
                project: linear_no_bias(250, 128, block.pp("linear.linear"))?,
                kernel: block
                    .get((128, 1, 20, 1), "fsmn_block.conv_left.weight")?
                    .squeeze(3)?
                    .contiguous()?,
                affine: linear(128, 250, block.pp("affine.linear"))?,
            });
        }
        Ok(Self {
            frontend: Frontend::load(directory)?,
            input1: linear(400, 140, vb.pp("in_linear1.linear"))?,
            input2: linear(140, 250, vb.pp("in_linear2.linear"))?,
            blocks,
            output1: linear(250, 140, vb.pp("out_linear1.linear"))?,
            output2: linear(140, 248, vb.pp("out_linear2.linear"))?,
        })
    }

    /// Input is mono, 16 kHz PCM normalized to [-1, 1]; intervals are milliseconds.
    pub fn detect(&self, samples: &[f32]) -> Result<Vec<(u64, u64)>> {
        ensure!(
            samples.iter().all(|x| x.is_finite() && x.abs() <= 1.0),
            "FSMN-VAD 输入必须为归一化的有限音频采样"
        );
        if samples.len() < FRAME_LENGTH {
            return Ok(Vec::new());
        }
        let features = self.frontend.features(samples)?;
        Ok(endpoints(&self.probabilities(&features)?))
    }

    fn probabilities(&self, features: &Tensor) -> Result<Vec<f32>> {
        let mut x = self
            .input2
            .forward(&self.input1.forward(features)?)?
            .relu()?;
        for block in &self.blocks {
            let projected = block.project.forward(&x)?;
            let memory = projected
                .t()?
                .unsqueeze(0)?
                .pad_with_zeros(2, 19, 0)?
                .conv1d(&block.kernel, 0, 1, 1, 128)?
                .squeeze(0)?
                .t()?;
            x = block.affine.forward(&(&projected + memory)?)?.relu()?;
        }
        let logits = self.output2.forward(&self.output1.forward(&x)?)?;
        let silence = candle_nn::ops::softmax(&logits, 1)?
            .narrow(1, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        ensure!(
            silence.iter().all(|x| x.is_finite()),
            "FSMN-VAD 输出包含非有限概率"
        );
        Ok(silence)
    }

    /// Starts an independent 16 kHz mono stream; this does not initialize a GPU.
    pub fn start_stream(&self) -> VadStream {
        VadStream {
            frontend: FrontendStream {
                waveform: Vec::with_capacity(STREAM_FRAMES * FRAME_SHIFT + FRAME_LENGTH),
                fbanks: VecDeque::new(),
                fbank_begin: 0,
                total_frames: 0,
                next_feature: 0,
                spectrum: vec![Complex32::default(); FFT_SIZE],
                scratch: vec![Complex32::default(); self.frontend.fft.get_inplace_scratch_len()],
            },
            caches: vec![None; self.blocks.len()],
            endpoint: StreamEndpoints::default(),
            received_samples: 0,
            finalized: false,
        }
    }

    /// Each real frame is computed once. `is_final` flushes the two-frame LFR
    /// lookahead with repeated edge features, matching the batch frontend.
    pub fn push_stream(
        &self,
        state: &mut VadStream,
        samples: &[f32],
        is_final: bool,
    ) -> Result<VadUpdate> {
        ensure!(!state.finalized, "FSMN-VAD 音频流已结束，请创建新的会话");
        ensure!(
            samples.iter().all(|x| x.is_finite() && x.abs() <= 1.0),
            "FSMN-VAD 输入必须为归一化的有限音频采样"
        );
        state.received_samples = state
            .received_samples
            .checked_add(samples.len() as u64)
            .context("FSMN-VAD 音频流采样计数溢出")?;
        let mut segments = Vec::new();
        // Bound work tensors even when a caller submits one very large PCM chunk.
        for chunk in samples.chunks(STREAM_FRAMES * FRAME_SHIFT) {
            state.frontend.append(&self.frontend, chunk);
            if let Some(features) = state.frontend.features(&self.frontend, false)? {
                self.infer_stream(state, &features, &mut segments)?;
            }
        }
        if is_final {
            if let Some(features) = state.frontend.features(&self.frontend, true)? {
                self.infer_stream(state, &features, &mut segments)?;
            }
            state.endpoint.finish(&mut segments);
            state.frontend.waveform.clear();
            state.frontend.fbanks.clear();
            state.finalized = true;
        }
        let endpoint = &state.endpoint;
        Ok(VadUpdate {
            segments,
            speech_start_sample: endpoint.start.map(|frame| frame * FRAME_SHIFT as u64),
            speech_active: endpoint.in_speech && !is_final,
            processed_samples: endpoint.next_frame * FRAME_SHIFT as u64,
            received_samples: state.received_samples,
            retain_from_sample: if is_final {
                state.received_samples
            } else {
                endpoint.start.unwrap_or(endpoint.earliest) * FRAME_SHIFT as u64
            },
        })
    }

    fn infer_stream(
        &self,
        state: &mut VadStream,
        features: &Tensor,
        segments: &mut Vec<VadSegment>,
    ) -> Result<()> {
        let silence = self.stream_probabilities(&mut state.caches, features)?;
        for probability in silence {
            state.endpoint.push(probability, segments);
        }
        Ok(())
    }

    fn stream_probabilities(
        &self,
        caches: &mut [Option<Tensor>],
        features: &Tensor,
    ) -> Result<Vec<f32>> {
        let mut x = self
            .input2
            .forward(&self.input1.forward(features)?)?
            .relu()?;
        for (block, cache) in self.blocks.iter().zip(caches) {
            let projected = block.project.forward(&x)?;
            let history = match cache {
                Some(previous) => Tensor::cat(&[previous, &projected], 0)?,
                None => projected.pad_with_zeros(0, FSMN_HISTORY, 0)?,
            };
            let memory = history
                .t()?
                .unsqueeze(0)?
                .conv1d(&block.kernel, 0, 1, 1, PROJECT_DIM)?
                .squeeze(0)?
                .t()?;
            // A narrow view can retain its complete source allocation. Materialize
            // only these 19 rows so each layer's persistent cache has fixed size.
            let rows = history.dim(0)?;
            let tail = history
                .narrow(0, rows - FSMN_HISTORY, FSMN_HISTORY)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            *cache = Some(Tensor::from_vec(
                tail,
                (FSMN_HISTORY, PROJECT_DIM),
                &Device::Cpu,
            )?);
            x = block.affine.forward(&(&projected + memory)?)?.relu()?;
        }
        let logits = self.output2.forward(&self.output1.forward(&x)?)?;
        let silence = candle_nn::ops::softmax(&logits, 1)?
            .narrow(1, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        ensure!(
            silence.iter().all(|p| p.is_finite()),
            "FSMN-VAD 输出包含非有限概率"
        );
        Ok(silence)
    }
}

impl Frontend {
    fn load(directory: &Path) -> Result<Self> {
        let cmvn =
            std::fs::read_to_string(directory.join("am.mvn")).context("无法读取 FSMN-VAD CMVN")?;
        let values = |marker: &str| -> Result<Vec<f32>> {
            let section = cmvn.split_once(marker).context("CMVN 缺少所需字段")?.1;
            let section = section.split_once('[').context("CMVN 缺少参数起始括号")?.1;
            let section = section.split_once(']').context("CMVN 缺少参数结束括号")?.0;
            let parsed: Vec<f32> = section
                .split_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?;
            ensure!(
                parsed.len() == 400 && parsed.iter().all(|x| x.is_finite()),
                "FSMN-VAD CMVN 必须含 400 个有限参数"
            );
            Ok(parsed)
        };
        let shift = values("<AddShift>")?;
        let scale = values("<Rescale>")?;
        let mel_freq = |hz: f32| 1127.0 * (1.0 + hz / 700.0).ln();
        let low = mel_freq(20.0);
        let delta = (mel_freq(8000.0) - low) / (MEL_BINS + 1) as f32;
        let mel = (0..MEL_BINS)
            .map(|band| {
                let left = low + band as f32 * delta;
                let center = low + (band + 1) as f32 * delta;
                let right = low + (band + 2) as f32 * delta;
                (0..FFT_SIZE / 2)
                    .filter_map(|bin| {
                        let m = mel_freq(bin as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32);
                        let weight =
                            ((m - left) / (center - left)).min((right - m) / (right - center));
                        (weight > 0.0).then_some((bin, weight))
                    })
                    .collect()
            })
            .collect();
        let window = (0..FRAME_LENGTH)
            .map(|i| {
                (0.54
                    - 0.46
                        * (2.0 * std::f64::consts::PI * i as f64 / (FRAME_LENGTH - 1) as f64).cos())
                    as f32
            })
            .collect();
        Ok(Self {
            fft: FftPlanner::new().plan_fft_forward(FFT_SIZE),
            window,
            mel,
            shift,
            scale,
        })
    }

    fn features(&self, samples: &[f32]) -> Result<Tensor> {
        let frames = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
        let mut fbank = Vec::with_capacity(frames * MEL_BINS);
        let mut spectrum = vec![Complex32::default(); FFT_SIZE];
        let mut scratch = vec![Complex32::default(); self.fft.get_inplace_scratch_len()];
        for start in (0..frames).map(|i| i * FRAME_SHIFT) {
            let frame = &samples[start..start + FRAME_LENGTH];
            fbank.extend(self.fbank_frame(frame, &mut spectrum, &mut scratch));
        }
        let mut features = Vec::with_capacity(frames * 400);
        for frame in 0..frames {
            for offset in 0..5usize {
                let source = (frame + offset).saturating_sub(2).min(frames - 1);
                for band in 0..MEL_BINS {
                    let column = offset * MEL_BINS + band;
                    features.push(
                        (fbank[source * MEL_BINS + band] + self.shift[column]) * self.scale[column],
                    );
                }
            }
        }
        Ok(Tensor::from_vec(features, (frames, 400), &Device::Cpu)?)
    }

    fn fbank_frame(
        &self,
        frame: &[f32],
        spectrum: &mut [Complex32],
        scratch: &mut [Complex32],
    ) -> [f32; MEL_BINS] {
        let mean = frame.iter().map(|x| x * 32768.0).sum::<f32>() / FRAME_LENGTH as f32;
        spectrum.fill(Complex32::default());
        let mut previous = frame[0] * 32768.0 - mean;
        for i in 0..FRAME_LENGTH {
            let value = frame[i] * 32768.0 - mean;
            spectrum[i].re = (value - 0.97 * previous) * self.window[i];
            previous = value;
        }
        self.fft.process_with_scratch(spectrum, scratch);
        std::array::from_fn(|band| {
            let energy = self.mel[band]
                .iter()
                .map(|&(bin, weight)| spectrum[bin].norm_sqr() * weight)
                .sum::<f32>();
            energy.max(f32::EPSILON).ln()
        })
    }
}

impl FrontendStream {
    fn append(&mut self, frontend: &Frontend, samples: &[f32]) {
        self.waveform.extend_from_slice(samples);
        if self.waveform.len() < FRAME_LENGTH {
            return;
        }
        let frames = 1 + (self.waveform.len() - FRAME_LENGTH) / FRAME_SHIFT;
        for index in 0..frames {
            let start = index * FRAME_SHIFT;
            self.fbanks.push_back(frontend.fbank_frame(
                &self.waveform[start..start + FRAME_LENGTH],
                &mut self.spectrum,
                &mut self.scratch,
            ));
        }
        self.total_frames += frames as u64;
        self.waveform.drain(..frames * FRAME_SHIFT);
    }

    fn features(&mut self, frontend: &Frontend, is_final: bool) -> Result<Option<Tensor>> {
        let end = if is_final {
            self.total_frames
        } else {
            self.total_frames.saturating_sub(2)
        };
        if self.next_feature == end {
            return Ok(None);
        }
        let frames = (end - self.next_feature) as usize;
        let mut features = Vec::with_capacity(frames * 400);
        for frame in self.next_feature..end {
            for offset in 0..5 {
                let source = (frame + offset)
                    .saturating_sub(2)
                    .min(self.total_frames - 1);
                let fbank = self
                    .fbanks
                    .get((source - self.fbank_begin) as usize)
                    .context("FSMN-VAD 增量特征缓存缺少上下文")?;
                for (band, &value) in fbank.iter().enumerate() {
                    let column = offset as usize * MEL_BINS + band;
                    features.push((value + frontend.shift[column]) * frontend.scale[column]);
                }
            }
        }
        self.next_feature = end;
        let retain = end.saturating_sub(2);
        while self.fbank_begin < retain {
            self.fbanks.pop_front();
            self.fbank_begin += 1;
        }
        Ok(Some(Tensor::from_vec(
            features,
            (frames, 400),
            &Device::Cpu,
        )?))
    }
}

impl StreamEndpoints {
    fn emit(&self, begin: u64, end: u64, reason: VadEndReason, segments: &mut Vec<VadSegment>) {
        let mut start = begin;
        while start < end {
            let next = (start + MAX_SEGMENT_FRAMES).min(end);
            segments.push(VadSegment {
                start_sample: start * FRAME_SHIFT as u64,
                end_sample: next * FRAME_SHIFT as u64,
                reason: if next < end {
                    VadEndReason::MaxDuration
                } else {
                    reason
                },
            });
            start = next;
        }
    }

    fn push(&mut self, probability: f32, segments: &mut Vec<VadSegment>) {
        let frame = self.next_frame;
        self.next_frame += 1;
        let speech = 1.0 - probability >= probability + 0.6;
        self.count -= usize::from(self.window[self.position]);
        self.count += usize::from(speech);
        self.window[self.position] = speech;
        self.position = (self.position + 1) % self.window.len();
        let previous = self.in_speech;
        self.in_speech = if previous {
            self.count > 15
        } else {
            self.count >= 15
        };
        if self.in_speech {
            self.quiet_frames = 0;
            if self.start.is_none() {
                self.start = Some(frame.saturating_sub(40).max(self.earliest));
            }
        } else if previous {
            self.quiet_frames = 0;
        } else {
            self.quiet_frames += 1;
            if let Some(begin) = self.start {
                if self.quiet_frames >= 65 {
                    let end = frame.saturating_sub(54) + 1;
                    self.emit(begin, end, VadEndReason::Silence, segments);
                    self.start = None;
                    self.earliest = end;
                    self.window.fill(false);
                    self.count = 0;
                    self.position = 0;
                    self.quiet_frames = 0;
                }
            } else {
                self.earliest = self.earliest.max(frame.saturating_sub(40));
            }
        }
        if let Some(begin) = self.start {
            // During a pending silence tail the eventual endpoint can look back
            // 54 frames. Emit only boundaries that can no longer move backwards.
            let stable_end = if self.in_speech {
                frame + 1
            } else {
                frame.saturating_sub(54) + 1
            };
            if stable_end >= begin + MAX_SEGMENT_FRAMES {
                let end = begin + MAX_SEGMENT_FRAMES;
                self.emit(begin, end, VadEndReason::MaxDuration, segments);
                self.start = Some(end);
                self.earliest = end;
            }
        }
    }

    fn finish(&mut self, segments: &mut Vec<VadSegment>) {
        if let Some(begin) = self.start.take() {
            self.emit(begin, self.next_frame, VadEndReason::Final, segments);
        }
        self.in_speech = false;
    }
}

fn endpoints(silence: &[f32]) -> Vec<(u64, u64)> {
    let mut window = [false; 20];
    let (mut position, mut count) = (0, 0usize);
    let (mut in_speech, mut quiet_frames) = (false, 0);
    let mut start = None;
    let mut earliest = 0;
    let mut segments = Vec::new();
    for (frame, &probability) in silence.iter().enumerate() {
        let speech = 1.0 - probability >= probability + 0.6;
        count -= usize::from(window[position]);
        count += usize::from(speech);
        window[position] = speech;
        position = (position + 1) % window.len();
        let previous = in_speech;
        in_speech = if previous { count > 15 } else { count >= 15 };
        if in_speech {
            quiet_frames = 0;
            if start.is_none() {
                start = Some(frame.saturating_sub(40).max(earliest));
            }
        } else if previous {
            quiet_frames = 0;
        } else {
            quiet_frames += 1;
            if let Some(begin) = start {
                if quiet_frames >= 65 {
                    let end = frame.saturating_sub(54) + 1;
                    segments.push((begin as u64 * 10, end as u64 * 10));
                    start = None;
                    earliest = end;
                    window.fill(false);
                    count = 0;
                    position = 0;
                    quiet_frames = 0;
                }
            } else {
                earliest = earliest.max(frame.saturating_sub(40));
            }
        }
    }
    if let Some(begin) = start {
        segments.push((begin as u64 * 10, silence.len() as u64 * 10));
    }
    segments
        .into_iter()
        .flat_map(|(begin, end)| {
            (begin..end)
                .step_by(15_000)
                .map(move |start| (start, (start + 15_000).min(end)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream_intervals(probabilities: &[f32]) -> Vec<(u64, u64)> {
        let mut state = StreamEndpoints::default();
        let mut segments = Vec::new();
        for &probability in probabilities {
            state.push(probability, &mut segments);
        }
        state.finish(&mut segments);
        segments
            .into_iter()
            .map(|segment| (segment.start_sample / 16, segment.end_sample / 16))
            .collect()
    }

    #[test]
    fn incremental_endpoints_match_batch_across_cut_and_silence_boundaries() {
        for length in [
            0, 10, 14, 15, 16, 19, 20, 100, 1480, 1490, 1499, 1500, 1501, 1510, 2999, 3000, 3001,
            4000,
        ] {
            for initial_silence in [0, 80] {
                for final_silence in [0, 20, 54, 64, 65, 100] {
                    let mut posterior = vec![1.0; initial_silence];
                    posterior.extend(vec![0.0; length]);
                    posterior.extend(vec![1.0; final_silence]);
                    assert_eq!(
                        stream_intervals(&posterior),
                        endpoints(&posterior),
                        "speech={length}, before={initial_silence}, after={final_silence}"
                    );
                }
            }
        }
        let posterior: Vec<_> = (0..25_000)
            .map(|frame| {
                if (frame * 17 / 37) % 191 < 113 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        assert_eq!(stream_intervals(&posterior), endpoints(&posterior));
    }

    fn test_frontend() -> Frontend {
        Frontend {
            fft: FftPlanner::new().plan_fft_forward(FFT_SIZE),
            window: vec![1.0; FRAME_LENGTH],
            mel: (0..MEL_BINS)
                .map(|band| vec![(band * 3 + 1, 0.75)])
                .collect(),
            shift: (0..400).map(|index| index as f32 * 0.001).collect(),
            scale: vec![0.5; 400],
        }
    }

    fn test_frontend_stream(frontend: &Frontend) -> FrontendStream {
        FrontendStream {
            waveform: Vec::new(),
            fbanks: VecDeque::new(),
            fbank_begin: 0,
            total_frames: 0,
            next_feature: 0,
            spectrum: vec![Complex32::default(); FFT_SIZE],
            scratch: vec![Complex32::default(); frontend.fft.get_inplace_scratch_len()],
        }
    }

    #[test]
    fn incremental_features_are_exact_for_arbitrary_pcm_splits_and_eos() -> Result<()> {
        let frontend = test_frontend();
        for length in [0, 1, 159, 399, 400, 401, 559, 560, 1000, 16_037] {
            let samples: Vec<_> = (0..length)
                .map(|i| (i as f32 * 0.027).sin() * 0.5)
                .collect();
            let expected = if length >= FRAME_LENGTH {
                frontend
                    .features(&samples)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            } else {
                Vec::new()
            };
            for chunk_size in [1, 97, 160, 511, 1600, 8191] {
                let mut state = test_frontend_stream(&frontend);
                let mut actual = Vec::new();
                for chunk in samples.chunks(chunk_size) {
                    state.append(&frontend, chunk);
                    if let Some(features) = state.features(&frontend, false)? {
                        actual.extend(features.flatten_all()?.to_vec1::<f32>()?);
                    }
                    assert!(state.waveform.len() < FRAME_LENGTH);
                    assert!(state.fbanks.len() <= 4);
                }
                if let Some(features) = state.features(&frontend, true)? {
                    actual.extend(features.flatten_all()?.to_vec1::<f32>()?);
                }
                assert_eq!(actual, expected, "length={length}, chunk={chunk_size}");
                assert!(state.features(&frontend, true)?.is_none());
            }
        }
        Ok(())
    }

    #[test]
    fn long_speech_is_emitted_before_eos_and_pending_silence_never_moves_back() {
        let mut state = StreamEndpoints::default();
        let mut segments = Vec::new();
        for _ in 0..1500 {
            state.push(0.0, &mut segments);
        }
        assert_eq!(segments.len(), 1);
        assert_eq!(
            segments[0],
            VadSegment {
                start_sample: 0,
                end_sample: 240_000,
                reason: VadEndReason::MaxDuration
            }
        );
        for _ in 0..100 {
            state.push(1.0, &mut segments);
        }
        state.finish(&mut segments);
        assert!(
            segments
                .windows(2)
                .all(|pair| pair[0].end_sample <= pair[1].start_sample)
        );
        assert!(
            segments
                .iter()
                .all(|segment| segment.end_sample > segment.start_sample
                    && segment.end_sample - segment.start_sample <= 240_000)
        );
    }

    #[test]
    fn endpoint_smoothing_extends_speech_and_keeps_silence_empty() {
        assert!(endpoints(&[1.0; 300]).is_empty());
        let mut posterior = vec![0.0; 100];
        posterior.extend([1.0; 200]);
        posterior.extend([0.0; 100]);
        assert_eq!(endpoints(&posterior), vec![(0, 1160), (2740, 4000)]);
    }

    #[test]
    fn endpoint_final_flush_and_long_segments_stay_bounded() {
        let segments = endpoints(&vec![0.0; 4000]);
        assert_eq!(segments, [(0, 15000), (15000, 30000), (30000, 40000)]);
        assert!(endpoints(&[0.0; 10]).is_empty());
    }

    #[test]
    #[ignore = "requires locally downloaded FSMN weights and smoke audio"]
    fn real_fsmn_matches_recorded_funasr_boundaries() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let model = root.join("models/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/f9a8b8274674755d925277e27063869038d41515");
        let mut reader = hound::WavReader::open(root.join(".tools/smoke/zh16.wav"))?;
        let samples = reader
            .samples::<i16>()
            .map(|x| x.map(|x| x as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let vad = Vad::load(&model)?;
        let segments = vad.detect(&samples)?;
        assert_eq!(segments.len(), 2, "{segments:?}");
        for ((start, end), (expected_start, expected_end)) in
            segments.iter().zip([(0, 4490), (6220, 9100)])
        {
            assert!(
                start.abs_diff(expected_start) <= 50 && end.abs_diff(expected_end) <= 50,
                "{segments:?}"
            );
        }
        assert!(vad.detect(&vec![0.0; SAMPLE_RATE * 2])?.is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "requires locally downloaded FSMN weights and smoke audio; CPU only"]
    fn real_incremental_fsmn_matches_batch_features_probabilities_and_segments() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let model = root.join("models/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/f9a8b8274674755d925277e27063869038d41515");
        let mut reader = hound::WavReader::open(root.join(".tools/smoke/zh16.wav"))?;
        let samples: Vec<_> = reader
            .samples::<i16>()
            .map(|x| x.map(|x| f32::from(x) / 32768.0))
            .collect::<std::result::Result<_, _>>()?;
        let vad = Vad::load(&model)?;
        let batch_features = vad.frontend.features(&samples)?;
        let batch_probabilities = vad.probabilities(&batch_features)?;
        let expected = vad.detect(&samples)?;
        for chunk_size in [97, 1600, 8191, samples.len()] {
            let mut stream = vad.start_stream();
            let mut segments = Vec::new();
            for chunk in samples.chunks(chunk_size) {
                let update = vad.push_stream(&mut stream, chunk, false)?;
                segments.extend(update.segments);
                assert!(stream.frontend.waveform.len() < FRAME_LENGTH);
                assert!(stream.frontend.fbanks.len() <= 4);
                assert!(
                    stream
                        .caches
                        .iter()
                        .flatten()
                        .all(|cache| cache.elem_count() == FSMN_HISTORY * PROJECT_DIM)
                );
                assert!(update.retain_from_sample <= update.received_samples);
            }
            let update = vad.push_stream(&mut stream, &[], true)?;
            segments.extend(update.segments);
            let actual: Vec<_> = segments
                .iter()
                .map(|segment| (segment.start_sample / 16, segment.end_sample / 16))
                .collect();
            assert_eq!(actual, expected, "chunk={chunk_size}");
            assert_eq!(
                update.processed_samples,
                batch_probabilities.len() as u64 * FRAME_SHIFT as u64
            );
            assert_eq!(update.received_samples, samples.len() as u64);
            assert!(update.speech_start_sample.is_none());
            assert!(vad.push_stream(&mut stream, &[], true).is_err());
        }
        let mut caches = vec![None; vad.blocks.len()];
        let mut incremental = Vec::new();
        let frame_count = batch_features.dim(0)?;
        for start in (0..frame_count).step_by(13) {
            let features = batch_features.narrow(0, start, 13.min(frame_count - start))?;
            incremental.extend(vad.stream_probabilities(&mut caches, &features)?);
        }
        let error = batch_probabilities
            .iter()
            .zip(&incremental)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("stream/batch max posterior difference: {error}; batch segments: {expected:?}");
        assert_eq!(incremental.len(), batch_probabilities.len());
        assert!(error < 0.0001, "posterior max error {error}");
        let mut silence = vad.start_stream();
        for _ in 0..100 {
            assert!(
                vad.push_stream(&mut silence, &[0.0; 320], false)?
                    .segments
                    .is_empty()
            );
        }
        assert!(
            vad.push_stream(&mut silence, &[], true)?
                .segments
                .is_empty()
        );
        Ok(())
    }
}
