//! Fixed FFmpeg bridge ABI; libraries stay loaded for the process lifetime.
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{Context, Result, ensure};
use libloading::Library;

type Decode = unsafe extern "C" fn(
    *const c_char,
    u64,
    u32,
    *mut *mut f32,
    *mut usize,
    *mut c_char,
    usize,
) -> i32;
type Free = unsafe extern "C" fn(*mut f32);

pub struct Decoder {
    decode: Decode,
    free: Free,
    _library: Library,
}

#[derive(Debug)]
pub enum DecodeError {
    Invalid(String),
    Timeout,
    TooLong,
    Internal(anyhow::Error),
}

impl Decoder {
    pub fn load(path: &Path) -> Result<Arc<Self>> {
        static LIBRARIES: OnceLock<Mutex<HashMap<PathBuf, Arc<Decoder>>>> = OnceLock::new();
        let path = std::fs::canonicalize(path).with_context(|| {
            format!(
                "找不到 FFmpeg 解码库 {}，请运行 scripts/setup-ffmpeg.ps1",
                path.display()
            )
        })?;
        let mut libraries = LIBRARIES
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| anyhow::anyhow!("FFmpeg 解码库状态异常，请重启服务"))?;
        if let Some(decoder) = libraries.get(&path) {
            return Ok(decoder.clone());
        }
        // Only the selected directory and Windows' default safe directories are searched.
        let library: Library = unsafe {
            libloading::os::windows::Library::load_with_flags(&path, 0x00000100 | 0x00001000)
        }
        .with_context(|| format!("无法加载 FFmpeg 解码库及其配套 DLL：{}", path.display()))?
        .into();
        let decoder = unsafe {
            let abi = library.get::<unsafe extern "C" fn() -> u32>(b"st_audio_abi_version\0")?;
            ensure!(
                abi() == 1,
                "FFmpeg 桥接库 ABI 不匹配，请重新运行 scripts/setup-ffmpeg.ps1"
            );
            let versions = library.get::<unsafe extern "C" fn(*mut u32)>(b"st_audio_versions\0")?;
            let mut actual = [0; 4];
            versions(actual.as_mut_ptr());
            ensure!(
                actual.map(|version| version >> 16) == [63, 63, 61, 7],
                "FFmpeg DLL 版本不匹配，需要固定 FFmpeg 9.0 的 avformat/avcodec/avutil/swresample"
            );
            Self {
                decode: *library.get(b"st_audio_decode\0")?,
                free: *library.get(b"st_audio_free\0")?,
                _library: library,
            }
        };
        let decoder = Arc::new(decoder);
        libraries.insert(path, decoder.clone());
        Ok(decoder)
    }

    pub fn decode(
        &self,
        input: &Path,
        max_samples: usize,
        timeout_ms: u32,
    ) -> Result<Vec<f32>, DecodeError> {
        let input =
            CString::new(input.to_str().ok_or_else(|| {
                DecodeError::Internal(anyhow::anyhow!("音频路径必须为有效 UTF-8"))
            })?)
            .map_err(|error| DecodeError::Internal(error.into()))?;
        let mut pointer = std::ptr::null_mut();
        let mut count = 0;
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            (self.decode)(
                input.as_ptr(),
                max_samples as u64,
                timeout_ms,
                &mut pointer,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        struct Buffer {
            pointer: *mut f32,
            free: Free,
        }
        impl Drop for Buffer {
            fn drop(&mut self) {
                if !self.pointer.is_null() {
                    unsafe { (self.free)(self.pointer) };
                }
            }
        }
        let buffer = Buffer {
            pointer,
            free: self.free,
        };
        error[error.len() - 1] = 0;
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        match status {
            0 => {
                if count > max_samples
                    || count > isize::MAX as usize / size_of::<f32>()
                    || (count > 0 && buffer.pointer.is_null())
                {
                    return Err(DecodeError::Internal(anyhow::anyhow!(
                        "FFmpeg 桥接库返回无效的采样缓冲"
                    )));
                }
                let samples = if count == 0 {
                    Vec::new()
                } else {
                    unsafe { std::slice::from_raw_parts(buffer.pointer, count) }.to_vec()
                };
                if samples.iter().any(|sample| !sample.is_finite()) {
                    return Err(DecodeError::Invalid("解码音频包含非有限采样值".into()));
                }
                Ok(samples)
            }
            1 => Err(DecodeError::Invalid(message)),
            2 => Err(DecodeError::Timeout),
            3 => Err(DecodeError::TooLong),
            _ => Err(DecodeError::Internal(anyhow::anyhow!(
                "FFmpeg 解码失败：{message}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_library_resamples_unicode_input_and_preserves_tail_and_bounds() -> Result<()> {
        let library = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".tools/ffmpeg-shared/bin/streamtrans-ffmpeg.dll");
        let decoder = Decoder::load(&library)?;
        assert!(Arc::ptr_eq(&decoder, &Decoder::load(&library)?));
        let directory = tempfile::tempdir()?;
        let input = directory.path().join("重采样尾部.wav");
        let mut writer = hound::WavWriter::create(
            &input,
            hound::WavSpec {
                channels: 2,
                sample_rate: 44_100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for index in 0..11_025 {
            let value = (6000.0 * (index as f32 * 0.1).sin()) as i16;
            writer.write_sample(value)?;
            writer.write_sample(value)?;
        }
        writer.finalize()?;
        let samples = decoder.decode(&input, 4000, 30_000).unwrap();
        assert_eq!(samples.len(), 4000);
        let peak = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0, f32::max);
        assert!(
            (0.17..0.20).contains(&peak),
            "duplicate stereo channels must retain their original amplitude: {peak}"
        );
        assert!(samples[3980..].iter().any(|sample| sample.abs() > 0.01));
        assert!(matches!(
            decoder.decode(&input, 3999, 30_000),
            Err(DecodeError::TooLong)
        ));
        assert!(matches!(
            decoder.decode(&input, 4000, 0),
            Err(DecodeError::Timeout)
        ));
        std::fs::write(&input, b"not an audio file")?;
        assert!(matches!(
            decoder.decode(&input, 4000, 30_000),
            Err(DecodeError::Invalid(_))
        ));
        Ok(())
    }
}
