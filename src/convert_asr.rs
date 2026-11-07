//! Native conversion of the official checkpoint, without executing Python or pickle code.
//!
//! Tensor names and metadata follow the MIT-licensed transcribe.cpp converter:
//! https://github.com/handy-computer/transcribe.cpp/blob/v0.4.1/scripts/convert-funasr_nano.py
//! Original matrix/embedding weights stay BF16. Biases, norms and FSMN
//! convolutions are expanded to FP32, matching the upstream BF16 converter.
//! The native audio encoder/adaptor still use FP32 activations.

use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{BufWriter, Seek, Write},
    path::Path,
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, pickle::PthTensors};
use serde_json::Value as Json;
use tempfile::NamedTempFile;

const ENCODER: &[(&str, &str)] = &[
    ("self_attn.linear_q_k_v.weight", "attn.qkv.weight"),
    ("self_attn.linear_q_k_v.bias", "attn.qkv.bias"),
    ("self_attn.linear_out.weight", "attn.out.weight"),
    ("self_attn.linear_out.bias", "attn.out.bias"),
    ("self_attn.fsmn_block.weight", "attn.fsmn.weight"),
    ("feed_forward.w_1.weight", "ffn.fc1.weight"),
    ("feed_forward.w_1.bias", "ffn.fc1.bias"),
    ("feed_forward.w_2.weight", "ffn.fc2.weight"),
    ("feed_forward.w_2.bias", "ffn.fc2.bias"),
    ("norm1.weight", "norm_attn.weight"),
    ("norm1.bias", "norm_attn.bias"),
    ("norm2.weight", "norm_ffn.weight"),
    ("norm2.bias", "norm_ffn.bias"),
];

const ADAPTOR: &[(&str, &str)] = &[
    ("self_attn.linear_q.weight", "attn.q.weight"),
    ("self_attn.linear_q.bias", "attn.q.bias"),
    ("self_attn.linear_k.weight", "attn.k.weight"),
    ("self_attn.linear_k.bias", "attn.k.bias"),
    ("self_attn.linear_v.weight", "attn.v.weight"),
    ("self_attn.linear_v.bias", "attn.v.bias"),
    ("self_attn.linear_out.weight", "attn.out.weight"),
    ("self_attn.linear_out.bias", "attn.out.bias"),
    ("feed_forward.w_1.weight", "ffn.fc1.weight"),
    ("feed_forward.w_1.bias", "ffn.fc1.bias"),
    ("feed_forward.w_2.weight", "ffn.fc2.weight"),
    ("feed_forward.w_2.bias", "ffn.fc2.bias"),
    ("norm1.weight", "norm_attn.weight"),
    ("norm1.bias", "norm_attn.bias"),
    ("norm2.weight", "norm_ffn.weight"),
    ("norm2.bias", "norm_ffn.bias"),
];

const DECODER: &[(&str, &str)] = &[
    ("input_layernorm.weight", "norm_attn.weight"),
    ("post_attention_layernorm.weight", "norm_ffn.weight"),
    ("self_attn.q_proj.weight", "attn.q.weight"),
    ("self_attn.k_proj.weight", "attn.k.weight"),
    ("self_attn.v_proj.weight", "attn.v.weight"),
    ("self_attn.o_proj.weight", "attn.o.weight"),
    ("self_attn.q_norm.weight", "attn.q_norm.weight"),
    ("self_attn.k_norm.weight", "attn.k_norm.weight"),
    ("mlp.gate_proj.weight", "ffn.gate.weight"),
    ("mlp.up_proj.weight", "ffn.up.weight"),
    ("mlp.down_proj.weight", "ffn.down.weight"),
];

fn tensor_mapping() -> Vec<(String, String)> {
    let mut names = Vec::new();
    let mut blocks = vec![("encoders0.0".to_owned(), "encoders0.0".to_owned())];
    blocks.extend((0..49).map(|i| (format!("encoders.{i}"), format!("encoders.{i}"))));
    blocks.extend((0..20).map(|i| (format!("tp_encoders.{i}"), format!("tp_encoders.{i}"))));
    for (source, target) in blocks {
        for (from, to) in ENCODER {
            names.push((
                format!("audio_encoder.{source}.{from}"),
                format!("enc.{target}.{to}"),
            ));
        }
    }
    for name in [
        "after_norm.weight",
        "after_norm.bias",
        "tp_norm.weight",
        "tp_norm.bias",
    ] {
        names.push((format!("audio_encoder.{name}"), format!("enc.{name}")));
    }
    for name in [
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
    ] {
        names.push((format!("audio_adaptor.{name}"), format!("adaptor.{name}")));
    }
    for i in 0..2 {
        for (from, to) in ADAPTOR {
            names.push((
                format!("audio_adaptor.blocks.{i}.{from}"),
                format!("adaptor.blocks.{i}.{to}"),
            ));
        }
    }
    names.push((
        "llm.model.embed_tokens.weight".into(),
        "dec.token_embd.weight".into(),
    ));
    names.push((
        "llm.model.norm.weight".into(),
        "dec.output_norm.weight".into(),
    ));
    for i in 0..28 {
        for (from, to) in DECODER {
            names.push((
                format!("llm.model.layers.{i}.{from}"),
                format!("dec.layers.{i}.{to}"),
            ));
        }
    }
    names
}

fn per_tensor_bf16(name: &str) -> bool {
    !name.ends_with(".bias")
        && ![
            ".norm_attn.weight",
            ".norm_ffn.weight",
            ".output_norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".after_norm.weight",
            ".tp_norm.weight",
            ".attn.fsmn.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

#[derive(Debug)]
enum Value {
    U32(u32),
    F32(f32),
    Bool(bool),
    String(String),
    Strings(Vec<String>),
    Ints(Vec<u32>),
}

impl Value {
    fn write(&self, writer: &mut impl Write) -> Result<()> {
        let kind: u32 = match self {
            Self::U32(_) => 4,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Strings(_) | Self::Ints(_) => 9,
        };
        writer.write_all(&kind.to_le_bytes())?;
        match self {
            Self::U32(value) => writer.write_all(&value.to_le_bytes())?,
            Self::F32(value) => writer.write_all(&value.to_le_bytes())?,
            Self::Bool(value) => writer.write_all(&[u8::from(*value)])?,
            Self::String(value) => write_string(writer, value)?,
            Self::Strings(values) => {
                writer.write_all(&8u32.to_le_bytes())?;
                writer.write_all(&(values.len() as u64).to_le_bytes())?;
                for value in values {
                    write_string(writer, value)?;
                }
            }
            Self::Ints(values) => {
                writer.write_all(&5u32.to_le_bytes())?;
                writer.write_all(&(values.len() as u64).to_le_bytes())?;
                for value in values {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }
}

fn write_string(writer: &mut impl Write, text: &str) -> Result<()> {
    writer.write_all(&(text.len() as u64).to_le_bytes())?;
    writer.write_all(text.as_bytes())?;
    Ok(())
}

fn read_json(path: &Path) -> Result<Json> {
    serde_json::from_reader(
        File::open(path).with_context(|| format!("无法打开 {}", path.display()))?,
    )
    .with_context(|| format!("JSON 配置无效：{}", path.display()))
}

fn metadata(directory: &Path) -> Result<BTreeMap<String, Value>> {
    let yaml: serde_yaml::Value =
        serde_yaml::from_reader(File::open(directory.join("config.yaml"))?)?;
    let json = read_json(&directory.join("Qwen3-0.6B/config.json"))?;
    // This converter deliberately accepts only the requested model architecture.
    for (section, key, expected) in [
        ("audio_encoder_conf", "num_blocks", 50),
        ("audio_encoder_conf", "tp_blocks", 20),
        ("audio_encoder_conf", "output_size", 512),
        ("audio_encoder_conf", "attention_heads", 4),
        ("audio_encoder_conf", "linear_units", 2048),
        ("audio_encoder_conf", "kernel_size", 11),
        ("audio_encoder_conf", "sanm_shfit", 0),
        ("audio_adaptor_conf", "n_layer", 2),
        ("audio_adaptor_conf", "encoder_dim", 512),
        ("audio_adaptor_conf", "llm_dim", 1024),
        ("audio_adaptor_conf", "ffn_dim", 2048),
        ("audio_adaptor_conf", "downsample_rate", 1),
        ("frontend_conf", "fs", 16000),
        ("frontend_conf", "n_mels", 80),
        ("frontend_conf", "frame_length", 25),
        ("frontend_conf", "frame_shift", 10),
        ("frontend_conf", "lfr_m", 7),
        ("frontend_conf", "lfr_n", 6),
    ] {
        ensure!(
            yaml[section][key].as_u64() == Some(expected),
            "Fun-ASR-Nano 配置不受支持：{section}.{key}"
        );
    }
    ensure!(
        yaml["model"].as_str() == Some("FunASRNano")
            && yaml["audio_encoder_conf"]["normalize_before"].as_bool() == Some(true)
            && yaml["audio_encoder_conf"]["selfattention_layer_type"].as_str() == Some("sanm")
            && yaml["audio_adaptor_conf"]["use_low_frame_rate"].as_bool() == Some(true)
            && yaml["frontend_conf"]["window"].as_str() == Some("hamming")
            && yaml["frontend_conf"]["cmvn_file"].is_null(),
        "Fun-ASR-Nano 模型或前处理配置不受支持"
    );
    for (key, expected) in [
        ("num_hidden_layers", 28),
        ("hidden_size", 1024),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 8),
        ("head_dim", 128),
    ] {
        ensure!(
            json[key].as_u64() == Some(expected),
            "Qwen3 配置不受支持：{key}"
        );
    }
    ensure!(
        json["tie_word_embeddings"].as_bool() == Some(true),
        "Qwen3 必须共享输入和输出词嵌入"
    );
    let mut result = BTreeMap::new();
    for (key, value) in [
        ("general.architecture", "funasr_nano"),
        ("general.name", "Fun-ASR Nano"),
        ("general.basename", "fun-asr-nano"),
        ("general.version", "2512"),
        ("general.author", "Alibaba Group / FunAudioLLM"),
        (
            "general.url",
            "https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512",
        ),
        ("general.license", "other"),
        (
            "general.license.name",
            "FunASR Model Open Source License Agreement v1.1",
        ),
        ("stt.variant", "fun-asr-nano-2512"),
        ("stt.funasr_nano.encoder.attention_type", "sanm"),
        ("stt.funasr_nano.adaptor.activation", "relu"),
        ("stt.funasr_nano.decoder.activation", "silu"),
        ("stt.frontend.type", "kaldi_fbank_lfr"),
        ("stt.frontend.window", "hamming"),
        ("stt.frontend.normalize", "none"),
        ("stt.frontend.fbank_style", "kaldi_htk"),
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.pre", "qwen2"),
    ] {
        result.insert(key.to_owned(), Value::String(value.to_owned()));
    }
    for (key, value) in [
        ("general.file_type", 32),
        ("general.alignment", 32),
        ("stt.funasr_nano.encoder.n_blocks", 50),
        ("stt.funasr_nano.encoder.tp_blocks", 20),
        ("stt.funasr_nano.encoder.d_model", 512),
        ("stt.funasr_nano.encoder.d_input", 560),
        ("stt.funasr_nano.encoder.n_heads", 4),
        ("stt.funasr_nano.encoder.d_ff", 2048),
        ("stt.funasr_nano.encoder.kernel_size", 11),
        ("stt.funasr_nano.encoder.sanm_shift", 0),
        ("stt.funasr_nano.adaptor.n_blocks", 2),
        ("stt.funasr_nano.adaptor.encoder_dim", 512),
        ("stt.funasr_nano.adaptor.llm_dim", 1024),
        ("stt.funasr_nano.adaptor.pre_ffn_dim", 2048),
        ("stt.funasr_nano.adaptor.block_ffn_dim", 256),
        ("stt.funasr_nano.adaptor.n_heads", 8),
        ("stt.funasr_nano.adaptor.d_head", 128),
        ("stt.funasr_nano.adaptor.downsample_rate", 1),
        ("stt.frontend.num_mels", 80),
        ("stt.frontend.sample_rate", 16000),
        ("stt.frontend.n_fft", 400),
        ("stt.frontend.win_length", 400),
        ("stt.frontend.hop_length", 160),
        ("stt.frontend.lfr_m", 7),
        ("stt.frontend.lfr_n", 6),
    ] {
        result.insert(key.to_owned(), Value::U32(value));
    }
    for (target, source) in [
        ("n_layers", "num_hidden_layers"),
        ("hidden_size", "hidden_size"),
        ("intermediate_size", "intermediate_size"),
        ("n_heads", "num_attention_heads"),
        ("n_kv_heads", "num_key_value_heads"),
        ("head_dim", "head_dim"),
        ("vocab_size", "vocab_size"),
        ("max_position_embeddings", "max_position_embeddings"),
    ] {
        let value = json[source].as_u64().context("Qwen3 缺少整数配置")?;
        result.insert(
            format!("stt.funasr_nano.decoder.{target}"),
            Value::U32(value.try_into()?),
        );
    }
    for name in ["rms_norm_eps", "rope_theta"] {
        let value = json[name].as_f64().context("Qwen3 缺少浮点配置")? as f32;
        ensure!(value.is_finite() && value > 0.0, "Qwen3 配置无效：{name}");
        result.insert(format!("stt.funasr_nano.decoder.{name}"), Value::F32(value));
    }
    for (key, value) in [
        ("stt.capability.translate", false),
        ("stt.capability.lang_detect", false),
        ("stt.capability.streaming", false),
        ("stt.funasr_nano.encoder.normalize_before", true),
        ("stt.funasr_nano.adaptor.use_low_frame_rate", true),
        ("stt.funasr_nano.decoder.tie_word_embeddings", true),
        ("stt.frontend.upscale_samples", true),
        ("stt.frontend.snip_edges", true),
        ("stt.frontend.apply_cmvn", false),
    ] {
        result.insert(key.to_owned(), Value::Bool(value));
    }
    result.insert(
        "stt.funasr_nano.adaptor.layer_norm_eps".into(),
        Value::F32(1e-12),
    );
    result.insert("stt.frontend.dither".into(), Value::F32(0.0));
    result.insert(
        "general.languages".into(),
        Value::Strings(vec!["zh".into(), "en".into(), "ja".into()]),
    );
    tokenizer(
        directory,
        json["vocab_size"]
            .as_u64()
            .context("缺少 vocab_size")?
            .try_into()?,
        &mut result,
    )?;
    Ok(result)
}

fn tokenizer(
    directory: &Path,
    vocab_size: usize,
    metadata: &mut BTreeMap<String, Value>,
) -> Result<()> {
    ensure!(
        vocab_size > 0 && vocab_size <= 200_000,
        "Qwen3 词表大小不受支持"
    );
    let directory = directory.join("Qwen3-0.6B");
    let vocabulary = read_json(&directory.join("vocab.json"))?;
    let config = read_json(&directory.join("tokenizer_config.json"))?;
    let mut tokens: Vec<_> = (0..vocab_size).map(|i| format!("<|unused_{i}|>")).collect();
    let mut types = vec![1; vocab_size];
    let mut ids = BTreeMap::new();
    for (token, id) in vocabulary.as_object().context("Qwen3 词表无效")? {
        let id = usize::try_from(id.as_u64().context("Qwen3 token ID 无效")?)?;
        ensure!(id < vocab_size, "Qwen3 token ID 超出词表");
        tokens[id] = token.clone();
        ids.insert(token.clone(), id as u32);
    }
    for (id, entry) in config["added_tokens_decoder"]
        .as_object()
        .context("Qwen3 缺少特殊 token 配置")?
    {
        let id: usize = id.parse()?;
        ensure!(id < vocab_size, "Qwen3 特殊 token ID 超出词表");
        let token = entry["content"].as_str().context("Qwen3 特殊 token 无效")?;
        tokens[id] = token.to_owned();
        types[id] = if entry["special"].as_bool() == Some(true) {
            3
        } else {
            1
        };
        ids.insert(token.to_owned(), id as u32);
    }
    for (source, target) in [
        ("bos_token", "bos_token_id"),
        ("eos_token", "eos_token_id"),
        ("pad_token", "padding_token_id"),
    ] {
        if let Some(token) = config[source]
            .as_str()
            .or_else(|| config[source]["content"].as_str())
        {
            let id = *ids
                .get(token)
                .with_context(|| format!("词表中缺少 {source}"))?;
            metadata.insert(format!("tokenizer.ggml.{target}"), Value::U32(id));
        }
    }
    if let Some(template) = config["chat_template"].as_str() {
        metadata.insert(
            "tokenizer.chat_template".into(),
            Value::String(template.to_owned()),
        );
    }
    let merges = std::fs::read_to_string(directory.join("merges.txt"))?;
    let mut lines = merges.lines();
    let first = lines.next().context("Qwen3 BPE merges 为空")?;
    let mut merges = Vec::new();
    if !first.starts_with('#') {
        merges.push(first.to_owned());
    }
    merges.extend(lines.map(str::to_owned));
    metadata.insert("tokenizer.ggml.tokens".into(), Value::Strings(tokens));
    metadata.insert("tokenizer.ggml.token_type".into(), Value::Ints(types));
    metadata.insert("tokenizer.ggml.merges".into(), Value::Strings(merges));
    Ok(())
}

fn align(writer: &mut (impl Write + Seek)) -> Result<()> {
    let padding = (32 - writer.stream_position()? % 32) % 32;
    writer.write_all(&[0; 32][..padding as usize])?;
    Ok(())
}

/// Convert a verified official checkpoint to an atomically published GGUF cache file.
pub fn convert(directory: &Path, output: &Path) -> Result<()> {
    let metadata = metadata(directory)?;
    let checkpoint = directory.join("model.pt");
    // Official checkpoints use either a flat state dict or a `state_dict` wrapper.
    let tensors = PthTensors::new(&checkpoint, Some("state_dict"))
        .or_else(|_| PthTensors::new(&checkpoint, None))
        .context("无法读取 Fun-ASR-Nano PyTorch 权重")?;
    let infos = tensors.tensor_infos();
    ensure!(!infos.is_empty(), "Fun-ASR-Nano 权重中没有可读取的张量");
    let mapping = tensor_mapping();
    let mut consumed: HashSet<&str> = HashSet::new();
    for (source, _) in &mapping {
        ensure!(
            infos.contains_key(source),
            "Fun-ASR-Nano 缺少张量：{source}"
        );
        let info = &infos[source];
        ensure!(
            info.dtype == DType::BF16,
            "Fun-ASR-Nano 必须为官方 BF16 权重：{source}"
        );
        ensure!(
            info.layout.is_contiguous() && info.layout.start_offset() == 0,
            "Fun-ASR-Nano 张量布局不受支持：{source}"
        );
        consumed.insert(source);
    }
    for name in infos.keys() {
        ensure!(
            consumed.contains(name.as_str())
                || name == "llm.lm_head.weight"
                || name.starts_with("ctc_decoder.")
                || name.starts_with("ctc."),
            "Fun-ASR-Nano 出现未知张量：{name}"
        );
    }
    let embed = infos
        .get("llm.model.embed_tokens.weight")
        .context("缺少词嵌入")?;
    let head = infos.get("llm.lm_head.weight").context("缺少输出词嵌入")?;
    ensure!(
        embed.layout.dims() == head.layout.dims(),
        "Qwen3 输入和输出词嵌入形状不一致"
    );
    if embed.path != head.path || embed.layout != head.layout {
        let a = tensors
            .get("llm.model.embed_tokens.weight")?
            .context("缺少词嵌入")?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let b = tensors
            .get("llm.lm_head.weight")?
            .context("缺少输出词嵌入")?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        ensure!(
            a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "Qwen3 输出权重未与词嵌入共享，不能安全转换"
        );
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut temporary = NamedTempFile::new_in(parent)?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        writer.write_all(b"GGUF")?;
        writer.write_all(&3u32.to_le_bytes())?;
        writer.write_all(&(mapping.len() as u64).to_le_bytes())?;
        writer.write_all(&(metadata.len() as u64).to_le_bytes())?;
        for (name, value) in &metadata {
            write_string(&mut writer, name)?;
            value.write(&mut writer)?;
        }
        let mut offset = 0u64;
        for (source, target) in &mapping {
            let info = &infos[source];
            write_string(&mut writer, target)?;
            writer.write_all(&(info.layout.dims().len() as u32).to_le_bytes())?;
            for &dimension in info.layout.dims().iter().rev() {
                writer.write_all(&(dimension as u64).to_le_bytes())?;
            }
            let dtype: u32 = if per_tensor_bf16(target) { 30 } else { 0 };
            writer.write_all(&dtype.to_le_bytes())?;
            writer.write_all(&offset.to_le_bytes())?;
            let size = (info.layout.shape().elem_count() as u64)
                .checked_mul(if per_tensor_bf16(target) { 2 } else { 4 })
                .context("模型张量过大")?;
            offset = offset
                .checked_add(size)
                .context("模型文件过大")?
                .div_ceil(32)
                * 32;
        }
        align(&mut writer)?;
        tracing::info!(tensors = mapping.len(), path = %output.display(), "正在将 ASR 权重转换为原生 BF16 GGUF 缓存");
        for (source, target) in &mapping {
            let data = tensors
                .get(source)?
                .with_context(|| format!("无法读取张量 {source}"))?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            // Source BF16 values are exactly representable in FP32; this preserves all bits.
            if per_tensor_bf16(target) {
                for block in data.chunks(16_384) {
                    let bytes: Vec<_> = block
                        .iter()
                        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                        .collect();
                    writer.write_all(&bytes)?;
                }
            } else {
                for block in data.chunks(16_384) {
                    let bytes: Vec<_> =
                        block.iter().flat_map(|value| value.to_le_bytes()).collect();
                    writer.write_all(&bytes)?;
                }
            }
            align(&mut writer)?;
        }
        writer.flush()?;
    }
    temporary.as_file().sync_all()?;
    temporary
        .persist(output)
        .with_context(|| format!("无法保存 ASR 缓存：{}", output.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the downloaded default Hugging Face checkpoint; no GPU needed"]
    fn huggingface_checkpoint_matches_existing_native_bf16_weights() -> Result<()> {
        use sha2::{Digest, Sha256};
        use std::io::Read;

        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "models/huggingface/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/272c57b82523ada6fd87095e955f8e29100979ab",
        );
        let temporary = tempfile::tempdir()?;
        let output = temporary.path().join("hf-native-bf16.gguf");
        convert(&directory, &output)?;
        let mut file = File::open(output)?;
        assert_eq!(file.metadata()?.len(), 1_667_503_232);
        let mut digest = Sha256::new();
        let mut buffer = vec![0; 2 * 1024 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        assert_eq!(
            format!("{:x}", digest.finalize()),
            "9410f344a3f51d4fd9797d4fc7ef36ea66575878eea6828485f1df247e5f7e7c",
            "Hugging Face conversion must preserve the existing native BF16 weights and metadata"
        );
        Ok(())
    }

    #[test]
    fn model_mapping_and_precision_are_complete() {
        let names = tensor_mapping();
        assert_eq!(names.len(), 1260);
        assert_eq!(
            names
                .iter()
                .map(|(source, _)| source)
                .collect::<HashSet<_>>()
                .len(),
            1260
        );
        assert_eq!(
            names
                .iter()
                .map(|(_, target)| target)
                .collect::<HashSet<_>>()
                .len(),
            1260
        );
        assert!(per_tensor_bf16("dec.token_embd.weight"));
        assert!(per_tensor_bf16("dec.layers.27.attn.q.weight"));
        assert!(per_tensor_bf16("enc.encoders.0.attn.qkv.weight"));
        assert!(per_tensor_bf16("adaptor.linear1.weight"));
        for name in [
            "dec.layers.27.attn.q_norm.weight",
            "dec.layers.27.norm_attn.weight",
            "enc.encoders.0.attn.qkv.bias",
            "enc.after_norm.weight",
            "enc.tp_norm.weight",
            "enc.encoders.0.attn.fsmn.weight",
            "adaptor.blocks.0.norm_ffn.weight",
        ] {
            assert!(!per_tensor_bf16(name), "{name} must remain FP32");
        }
    }

    #[test]
    fn metadata_values_use_gguf_wire_types() -> Result<()> {
        let mut bytes = Vec::new();
        Value::Ints(vec![1, 3]).write(&mut bytes)?;
        assert_eq!(&bytes[0..4], &9u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &5u32.to_le_bytes());
        assert_eq!(&bytes[8..16], &2u64.to_le_bytes());
        assert_eq!(bytes.len(), 24);
        Ok(())
    }
}
