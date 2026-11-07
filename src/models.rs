use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::{
    config::{Config, ModelSource},
    runtime::Models,
};

const ASR_ID: &str = "FunAudioLLM/Fun-ASR-Nano-2512";
const VAD_ID: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";
const LLM_ID: &str = "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF";
const LLM_NAME: &str = "Hy-MT2-1.8B-Q6_K.gguf";
const HF_ASR_REVISION: &str = "272c57b82523ada6fd87095e955f8e29100979ab";
const HF_VAD_REVISION: &str = "df20e6b30c653645fa4ff125cacfcabd1020a669";
const HF_LLM_REVISION: &str = "a0c709d9fac510f2c807aa3af52872340dc37a4a";
const MAX_HF_INLINE_BYTES: u64 = 32 * 1024 * 1024;
const ASR_FILES: &[&str] = &[
    "model.pt",
    "config.yaml",
    "configuration.json",
    "multilingual.tiktoken",
    "Qwen3-0.6B/config.json",
    "Qwen3-0.6B/generation_config.json",
    "Qwen3-0.6B/merges.txt",
    "Qwen3-0.6B/tokenizer.json",
    "Qwen3-0.6B/tokenizer_config.json",
    "Qwen3-0.6B/vocab.json",
];
const VAD_FILES: &[&str] = &["model.pt", "config.yaml", "configuration.json", "am.mvn"];
const MANIFEST_NAME: &str = ".streamtrans-native-models.json";

#[derive(Clone, Copy)]
struct Specification<'a> {
    source: ModelSource,
    key: &'static str,
    id: &'static str,
    revision: &'a str,
    files: &'static [&'static str],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FileRecord {
    #[serde(rename = "Size")]
    size: u64,
    #[serde(rename = "Sha256")]
    sha256: String,
}

impl FileRecord {
    fn validate(&self) -> Result<()> {
        ensure!(self.size > 0, "模型文件大小必须大于零");
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|c| c.is_ascii_hexdigit()),
            "模型文件缺少有效 SHA256"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
struct CachedModel {
    id: String,
    revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resolved_revision: Option<String>,
    files: BTreeMap<String, FileRecord>,
}

#[derive(Deserialize, Serialize)]
struct Manifest {
    #[serde(default = "legacy_source")]
    source: ModelSource,
    version: u32,
    paths: Models,
    models: BTreeMap<String, CachedModel>,
}

fn legacy_source() -> ModelSource {
    ModelSource::ModelScope
}

fn specifications(config: &Config) -> [Specification<'_>; 3] {
    let (asr_id, asr_revision, vad_id, vad_revision, llm_id, llm_revision) =
        match config.model_source {
            ModelSource::HuggingFace => (
                ASR_ID,
                HF_ASR_REVISION,
                "funasr/fsmn-vad",
                HF_VAD_REVISION,
                "tencent/Hy-MT2-1.8B-GGUF",
                HF_LLM_REVISION,
            ),
            ModelSource::ModelScope => (
                ASR_ID,
                "05201c46f1c38592b1567f857c0d56eab3d0d8ef",
                VAD_ID,
                "f9a8b8274674755d925277e27063869038d41515",
                LLM_ID,
                "e97e6af0ec0950385644d7c0a8e5c899071f722b",
            ),
        };
    [
        Specification {
            source: config.model_source,
            key: "asr_dir",
            id: asr_id,
            revision: config.asr_revision.as_deref().unwrap_or(asr_revision),
            files: ASR_FILES,
        },
        Specification {
            source: config.model_source,
            key: "vad_dir",
            id: vad_id,
            revision: config.vad_revision.as_deref().unwrap_or(vad_revision),
            files: VAD_FILES,
        },
        Specification {
            source: config.model_source,
            key: "llm_file",
            id: llm_id,
            revision: config.llm_revision.as_deref().unwrap_or(llm_revision),
            files: &[LLM_NAME],
        },
    ]
}

fn model_path<'a>(models: &'a Models, key: &str) -> &'a Path {
    match key {
        "asr_dir" => &models.asr_dir,
        "vad_dir" => &models.vad_dir,
        "llm_file" => &models.llm_file,
        _ => unreachable!("only the three fixed model specifications are used"),
    }
}

fn file_path(models: &Models, key: &str, name: &str) -> PathBuf {
    let path = model_path(models, key);
    if key == "llm_file" {
        path.to_owned()
    } else {
        path.join(name)
    }
}

fn matching_revisions(manifest: &Manifest, specs: &[Specification<'_>]) -> bool {
    specs.iter().all(|spec| {
        manifest.source == spec.source
            && manifest
                .models
                .get(spec.key)
                .is_some_and(|model| model.id == spec.id && model.revision == spec.revision)
    })
}

fn manifest_path(cache: &Path, specs: &[Specification<'_>]) -> PathBuf {
    // Keep former model selections and their manifests intact when changing the LLM.
    let mut digest = Sha256::new();
    if specs[0].source == ModelSource::HuggingFace {
        digest.update(b"huggingface");
    }
    for spec in specs {
        for part in [spec.key, spec.id, spec.revision]
            .into_iter()
            .chain(spec.files.iter().copied())
        {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part.as_bytes());
        }
    }
    cache.join(format!(
        ".streamtrans-models-{}-{:x}.json",
        if specs[0].source == ModelSource::HuggingFace {
            "huggingface"
        } else {
            "native"
        },
        digest.finalize()
    ))
}

async fn verify_file(path: &Path, record: &FileRecord) -> Result<()> {
    record.validate()?;
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("模型文件缺失或无法读取：{}", path.display()))?;
    ensure!(
        file.metadata().await?.len() == record.size,
        "模型文件大小不符：{}",
        path.display()
    );
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    ensure!(
        format!("{:x}", digest.finalize()).eq_ignore_ascii_case(&record.sha256),
        "模型 SHA256 校验失败：{}，请删除该文件和缓存清单后重新下载",
        path.display()
    );
    Ok(())
}

async fn validate_manifest(
    manifest: &Manifest,
    specs: &[Specification<'_>],
    cache: &Path,
) -> Result<()> {
    ensure!(
        manifest.version == 1
            && manifest.models.len() == specs.len()
            && matching_revisions(manifest, specs),
        "模型缓存清单无效：模型标识或版本不符"
    );
    for spec in specs {
        let model = &manifest.models[spec.key];
        ensure!(
            model.files.len() == spec.files.len()
                && spec
                    .files
                    .iter()
                    .all(|name| model.files.contains_key(*name)),
            "模型缓存清单无效：{} 的文件集合不符",
            spec.id
        );
        let path = model_path(&manifest.paths, spec.key);
        ensure!(
            path.is_absolute(),
            "模型缓存清单无效：模型路径必须为绝对路径"
        );
        for name in spec.files {
            let path = file_path(&manifest.paths, spec.key, name);
            let resolved = tokio::fs::canonicalize(&path).await.with_context(|| {
                format!("模型文件缺失：{}，请删除缓存清单后重新下载", path.display())
            })?;
            ensure!(
                resolved.starts_with(cache),
                "模型缓存清单无效：文件路径超出缓存目录"
            );
            verify_file(&path, &model.files[*name]).await?;
        }
    }
    Ok(())
}

async fn cached_manifest(cache: &Path, specs: &[Specification<'_>]) -> Result<Option<Manifest>> {
    let mut paths = Vec::new();
    let native = manifest_path(cache, specs);
    if native.is_file() {
        paths.push(native);
    }
    if specs[0].source == ModelSource::ModelScope {
        let legacy_native = cache.join(MANIFEST_NAME);
        if legacy_native.is_file() {
            paths.push(legacy_native);
        }
        let mut directory = tokio::fs::read_dir(cache).await?;
        while let Some(entry) = directory.next_entry().await? {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(".streamtrans-models-")
                && !name.starts_with(".streamtrans-models-huggingface-")
                && name.ends_with(".json")
                && !paths.contains(&entry.path())
            {
                paths.push(entry.path());
            }
        }
    }
    for path in paths {
        let manifest: Manifest = serde_json::from_slice(&tokio::fs::read(&path).await?)
            .with_context(|| {
                format!(
                    "模型缓存清单无效：{}，请删除该清单后重新下载",
                    path.display()
                )
            })?;
        if !matching_revisions(&manifest, specs) {
            continue;
        }
        validate_manifest(&manifest, specs, cache)
            .await
            .with_context(|| format!("模型缓存检查失败：{}", path.display()))?;
        return Ok(Some(manifest));
    }
    Ok(None)
}

fn parse_metadata(
    payload: Value,
    spec: &Specification<'_>,
) -> Result<BTreeMap<String, FileRecord>> {
    ensure!(
        payload["Success"] == true,
        "无法读取 ModelScope 模型清单：{}，请检查网络和 revision",
        spec.id
    );
    let entries = payload["Data"]["Files"]
        .as_array()
        .with_context(|| format!("ModelScope 模型清单缺少 Files：{}", spec.id))?;
    let mut files = BTreeMap::new();
    for entry in entries {
        let Some(name) = entry["Path"].as_str() else {
            continue;
        };
        if entry["Type"] != "blob" || !spec.files.contains(&name) {
            continue;
        }
        let mut record: FileRecord = serde_json::from_value(entry.clone())
            .with_context(|| format!("ModelScope 文件元数据无效：{}/{name}", spec.id))?;
        record
            .validate()
            .with_context(|| format!("ModelScope 文件元数据无效：{}/{name}", spec.id))?;
        record.sha256.make_ascii_lowercase();
        ensure!(
            files.insert(name.to_owned(), record).is_none(),
            "ModelScope 文件元数据重复：{name}"
        );
    }
    for name in spec.files {
        ensure!(
            files.contains_key(*name),
            "ModelScope 模型 {} 缺少文件：{name}，请检查 revision",
            spec.id
        );
    }
    Ok(files)
}

#[derive(Debug)]
struct ModelMetadata {
    revision: String,
    files: BTreeMap<String, FileRecord>,
}

fn source_base(source: ModelSource) -> &'static str {
    match source {
        ModelSource::HuggingFace => "https://huggingface.co",
        ModelSource::ModelScope => "https://modelscope.cn",
    }
}

fn endpoint(
    base: &str,
    segments: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(base)?;
    url.path_segments_mut()
        .map_err(|_| anyhow::anyhow!("模型源 URL 不是层次化地址"))?
        .pop_if_empty()
        .extend(segments);
    Ok(url)
}

fn file_request(
    client: &reqwest::Client,
    spec: &Specification<'_>,
    name: &str,
    base: &str,
) -> Result<reqwest::RequestBuilder> {
    match spec.source {
        ModelSource::ModelScope => Ok(client
            .get(endpoint(
                base,
                ["api", "v1", "models"]
                    .into_iter()
                    .chain(spec.id.split('/'))
                    .chain(["repo"]),
            )?)
            .query(&[("Revision", spec.revision), ("FilePath", name)])),
        ModelSource::HuggingFace => Ok(client.get(endpoint(
            base,
            spec.id
                .split('/')
                .chain(["resolve", spec.revision])
                .chain(name.split('/')),
        )?)),
    }
}

fn validate_hf_revision(payload: &Value, requested: &str) -> Result<String> {
    let resolved = payload["sha"]
        .as_str()
        .context("Hugging Face 元数据缺少提交 SHA")?;
    ensure!(
        resolved.len() == 40 && resolved.bytes().all(|c| c.is_ascii_hexdigit()),
        "Hugging Face 元数据提交 SHA 无效"
    );
    if requested.len() == 40 && requested.bytes().all(|c| c.is_ascii_hexdigit()) {
        ensure!(
            resolved.eq_ignore_ascii_case(requested),
            "Hugging Face 返回的提交 SHA 与请求不符"
        );
    }
    Ok(resolved.to_ascii_lowercase())
}

async fn hf_metadata(
    client: &reqwest::Client,
    spec: &Specification<'_>,
    payload: Value,
    base: &str,
) -> Result<ModelMetadata> {
    let revision = validate_hf_revision(&payload, spec.revision)?;
    let entries = payload["siblings"]
        .as_array()
        .context("Hugging Face 元数据缺少 siblings 文件列表")?;
    let resolved = Specification {
        revision: &revision,
        ..*spec
    };
    let mut files = BTreeMap::new();
    for entry in entries {
        let Some(name) = entry["rfilename"].as_str() else {
            continue;
        };
        if !spec.files.contains(&name) {
            continue;
        }
        ensure!(
            !files.contains_key(name),
            "Hugging Face 文件元数据重复：{name}"
        );
        let size = entry["size"]
            .as_u64()
            .with_context(|| format!("Hugging Face 文件缺少大小：{name}"))?;
        let sha256 = if entry["lfs"].is_object() {
            ensure!(
                entry["lfs"]["size"].as_u64() == Some(size),
                "Hugging Face LFS 文件大小冲突：{name}"
            );
            entry["lfs"]["sha256"]
                .as_str()
                .context("Hugging Face LFS 文件缺少 SHA256")?
                .to_owned()
        } else {
            // Git blob IDs are SHA1, not SHA256; hash small pinned files ourselves.
            ensure!(
                size > 0 && size <= MAX_HF_INLINE_BYTES,
                "Hugging Face 非 LFS 文件超过 32 MiB 元数据读取上限：{name}"
            );
            let mut response = file_request(client, &resolved, name, base)?
                .timeout(Duration::from_secs(60))
                .send()
                .await?
                .error_for_status()?;
            ensure!(
                response.status() == reqwest::StatusCode::OK,
                "Hugging Face 必须返回完整文件（HTTP 200）"
            );
            if let Some(length) = response.content_length() {
                ensure!(
                    length == size,
                    "Hugging Face 下载文件大小与元数据不符：{name}"
                );
            }
            let mut digest = Sha256::new();
            let mut received = 0u64;
            while let Some(chunk) = response.chunk().await? {
                received = received
                    .checked_add(chunk.len() as u64)
                    .context("模型文件大小溢出")?;
                ensure!(
                    received <= size,
                    "Hugging Face 下载文件超过声明大小：{name}"
                );
                digest.update(&chunk);
            }
            ensure!(received == size, "Hugging Face 下载文件不完整：{name}");
            format!("{:x}", digest.finalize())
        };
        let mut record = FileRecord { size, sha256 };
        record
            .validate()
            .with_context(|| format!("Hugging Face 文件元数据无效：{}/{name}", spec.id))?;
        record.sha256.make_ascii_lowercase();
        files.insert(name.to_owned(), record);
    }
    for name in spec.files {
        ensure!(
            files.contains_key(*name),
            "Hugging Face 模型 {} 缺少文件：{name}，请检查 revision",
            spec.id
        );
    }
    Ok(ModelMetadata { revision, files })
}

async fn metadata_at(
    client: &reqwest::Client,
    spec: &Specification<'_>,
    base: &str,
) -> Result<ModelMetadata> {
    let request = match spec.source {
        ModelSource::ModelScope => client
            .get(endpoint(
                base,
                ["api", "v1", "models"]
                    .into_iter()
                    .chain(spec.id.split('/'))
                    .chain(["repo", "files"]),
            )?)
            .query(&[("Revision", spec.revision), ("Recursive", "true")]),
        ModelSource::HuggingFace => client
            .get(endpoint(
                base,
                ["api", "models"]
                    .into_iter()
                    .chain(spec.id.split('/'))
                    .chain(["revision", spec.revision]),
            )?)
            .query(&[("blobs", "true")]),
    };
    let response = request
        .timeout(Duration::from_secs(60))
        .send()
        .await?
        .error_for_status()?;
    ensure!(
        response.status() == reqwest::StatusCode::OK,
        "{} 元数据接口必须返回 HTTP 200",
        spec.source
    );
    let payload = response.json().await?;
    match spec.source {
        ModelSource::ModelScope => Ok(ModelMetadata {
            revision: spec.revision.to_owned(),
            files: parse_metadata(payload, spec)?,
        }),
        ModelSource::HuggingFace => hf_metadata(client, spec, payload, base).await,
    }
}

async fn metadata(client: &reqwest::Client, spec: &Specification<'_>) -> Result<ModelMetadata> {
    metadata_at(client, spec, source_base(spec.source)).await
}

fn snapshot_path(cache: &Path, spec: &Specification<'_>) -> PathBuf {
    // Revisions may be branch names containing slashes; keep them out of filesystem paths.
    let revision = if !matches!(spec.revision, "" | "." | "..")
        && spec
            .revision
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
    {
        spec.revision.to_owned()
    } else {
        format!("{:x}", Sha256::digest(spec.revision.as_bytes()))
    };
    let cache = if spec.source == ModelSource::HuggingFace {
        cache.join("huggingface")
    } else {
        cache.to_owned()
    };
    cache
        .join("models")
        .join(spec.id.replace('/', "--"))
        .join("snapshots")
        .join(revision)
}

async fn download_file(
    client: &reqwest::Client,
    spec: &Specification<'_>,
    name: &str,
    record: &FileRecord,
    path: &Path,
    cache: &Path,
) -> Result<()> {
    download_file_at(
        client,
        spec,
        name,
        record,
        path,
        cache,
        source_base(spec.source),
    )
    .await
}

async fn download_file_at(
    client: &reqwest::Client,
    spec: &Specification<'_>,
    name: &str,
    record: &FileRecord,
    path: &Path,
    cache: &Path,
    base: &str,
) -> Result<()> {
    record.validate()?;
    let parent = path.parent().context("模型文件没有父目录")?;
    tokio::fs::create_dir_all(parent).await?;
    ensure!(
        tokio::fs::canonicalize(parent).await?.starts_with(cache),
        "模型下载路径超出缓存目录"
    );
    if path.is_file() {
        ensure!(
            tokio::fs::canonicalize(path).await?.starts_with(cache),
            "模型文件路径超出缓存目录"
        );
        if verify_file(path, record).await.is_ok() {
            return Ok(());
        }
    }
    tracing::info!(
        source = %spec.source,
        model = spec.id,
        file = name,
        bytes = record.size,
        "下载模型文件"
    );
    let mut response = file_request(client, spec, name, base)?
        .send()
        .await?
        .error_for_status()?;
    ensure!(
        response.status() == reqwest::StatusCode::OK,
        "{} 必须返回完整文件（HTTP 200）",
        spec.source
    );
    if let Some(size) = response.content_length() {
        ensure!(
            size == record.size,
            "{} 下载文件大小与元数据不符：{name}",
            spec.source
        );
    }
    // Start interrupted files afresh; never trust a partial response without checking its range.
    let partial = tempfile::Builder::new()
        .prefix(".streamtrans-")
        .suffix(".partial")
        .tempfile_in(parent)?;
    let mut output = tokio::fs::File::from_std(partial.reopen()?);
    let mut digest = Sha256::new();
    let mut received = 0u64;
    let mut last_progress = Instant::now();
    while let Some(chunk) = response.chunk().await? {
        received = received
            .checked_add(chunk.len() as u64)
            .context("模型文件大小溢出")?;
        ensure!(
            received <= record.size,
            "{} 下载文件超过声明大小：{name}",
            spec.source
        );
        digest.update(&chunk);
        output.write_all(&chunk).await?;
        if received < record.size && last_progress.elapsed() >= Duration::from_secs(1) {
            tracing::info!(
                model = spec.id,
                file = name,
                "下载进度：{:.1}%（{:.2}/{:.2} MiB）",
                (received as f64 / record.size as f64 * 100.0).min(99.9),
                received as f64 / 1_048_576.0,
                record.size as f64 / 1_048_576.0,
            );
            last_progress = Instant::now();
        }
    }
    ensure!(
        received == record.size,
        "{} 下载文件不完整：{name}",
        spec.source
    );
    ensure!(
        format!("{:x}", digest.finalize()) == record.sha256,
        "{} 下载文件 SHA256 校验失败：{name}",
        spec.source
    );
    output.flush().await?;
    output.sync_all().await?;
    drop(output);
    partial
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("无法保存模型文件：{}", path.display()))?;
    tracing::info!(
        model = spec.id,
        file = name,
        "模型文件下载完成：100%（{:.2} MiB）",
        record.size as f64 / 1_048_576.0,
    );
    Ok(())
}

fn write_manifest(cache: &Path, manifest: &Manifest, specs: &[Specification<'_>]) -> Result<()> {
    use std::io::Write;
    let mut temporary = tempfile::Builder::new()
        .prefix(".streamtrans-")
        .suffix(".tmp")
        .tempfile_in(cache)?;
    serde_json::to_writer(&mut temporary, manifest)?;
    temporary.flush()?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(manifest_path(cache, specs))
        .map_err(|error| error.error)?;
    Ok(())
}

pub async fn download(config: &Config) -> Result<Models> {
    tokio::fs::create_dir_all(&config.cache_dir).await?;
    let cache = tokio::fs::canonicalize(&config.cache_dir).await?;
    let specs = specifications(config);
    tracing::info!(source = %config.model_source, cache = %cache.display(), "检查模型缓存与 SHA256");
    if let Some(manifest) = cached_manifest(&cache, &specs).await? {
        write_manifest(&cache, &manifest, &specs)?;
        return Ok(manifest.paths);
    }
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .build()?;
    let paths = Models {
        asr_dir: snapshot_path(&cache, &specs[0]),
        vad_dir: snapshot_path(&cache, &specs[1]),
        llm_file: snapshot_path(&cache, &specs[2]).join(LLM_NAME),
    };
    let mut manifest = Manifest {
        source: config.model_source,
        version: 1,
        paths,
        models: BTreeMap::new(),
    };
    for spec in &specs {
        let metadata = metadata(&client, spec)
            .await
            .with_context(|| format!("读取 {} 元数据失败：{}", spec.source, spec.id))?;
        let resolved = Specification {
            revision: &metadata.revision,
            ..*spec
        };
        let files = metadata.files;
        for name in spec.files {
            let path = file_path(&manifest.paths, spec.key, name);
            download_file(&client, &resolved, name, &files[*name], &path, &cache)
                .await
                .with_context(|| format!("下载 {} 模型失败：{}/{name}", spec.source, spec.id))?;
        }
        manifest.models.insert(
            spec.key.to_owned(),
            CachedModel {
                id: spec.id.to_owned(),
                revision: spec.revision.to_owned(),
                resolved_revision: (spec.source == ModelSource::HuggingFace)
                    .then_some(metadata.revision),
                files,
            },
        );
    }
    write_manifest(&cache, &manifest, &specs)?;
    Ok(manifest.paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use axum::{
        Router,
        body::Body,
        http::{Response, Uri},
    };
    use clap::Parser;
    use serde_json::json;

    fn record(data: &[u8]) -> FileRecord {
        FileRecord {
            size: data.len() as u64,
            sha256: format!("{:x}", Sha256::digest(data)),
        }
    }

    struct MockServer {
        base: String,
        requests: Arc<Mutex<Vec<String>>>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn mock_server(responses: BTreeMap<String, (u16, Vec<u8>)>) -> MockServer {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let app = Router::new().fallback(move |uri: Uri| {
            let key = uri.path_and_query().unwrap().as_str().to_owned();
            recorded.lock().unwrap().push(key.clone());
            let (status, body) = responses
                .get(&key)
                .cloned()
                .unwrap_or((404, b"missing".to_vec()));
            async move {
                Response::builder()
                    .status(status)
                    .body(Body::from(body))
                    .unwrap()
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        MockServer {
            base,
            requests,
            task,
        }
    }

    fn cached_fixture(cache: &Path, config: &Config) -> Manifest {
        let paths = Models {
            asr_dir: cache.join("asr"),
            vad_dir: cache.join("vad"),
            llm_file: cache.join("llm").join(LLM_NAME),
        };
        let mut models = BTreeMap::new();
        for spec in specifications(config) {
            let mut files = BTreeMap::new();
            for name in spec.files {
                let path = file_path(&paths, spec.key, name);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, name.as_bytes()).unwrap();
                files.insert((*name).to_owned(), record(name.as_bytes()));
            }
            models.insert(
                spec.key.to_owned(),
                CachedModel {
                    id: spec.id.to_owned(),
                    revision: spec.revision.to_owned(),
                    resolved_revision: None,
                    files,
                },
            );
        }
        Manifest {
            source: config.model_source,
            version: 1,
            paths,
            models,
        }
    }

    #[tokio::test]
    async fn checksum_rejects_corruption_with_unchanged_size() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("model.pt");
        std::fs::write(&path, b"yes").unwrap();
        verify_file(&path, &record(b"yes")).await.unwrap();
        std::fs::write(&path, b"bad").unwrap();
        assert!(
            verify_file(&path, &record(b"yes"))
                .await
                .unwrap_err()
                .to_string()
                .contains("SHA256")
        );
    }

    #[test]
    fn metadata_requires_complete_files_and_valid_hashes() {
        let spec = Specification {
            source: ModelSource::ModelScope,
            key: "llm_file",
            id: LLM_ID,
            revision: "fixed",
            files: &[LLM_NAME],
        };
        let entry = json!({"Path":LLM_NAME,"Type":"blob","Size":3,"Sha256":record(b"yes").sha256});
        assert!(
            parse_metadata(
                json!({"Success":true,"Data":{"Files":[entry.clone()]}}),
                &spec
            )
            .is_ok()
        );
        assert!(
            parse_metadata(
                json!({"Success":false,"Data":{"Files":[entry.clone()]}}),
                &spec
            )
            .is_err()
        );
        assert!(parse_metadata(json!({"Success":true,"Data":{"Files":[]}}), &spec).is_err());
        let mut invalid = entry.clone();
        invalid["Sha256"] = json!("bad");
        assert!(parse_metadata(json!({"Success":true,"Data":{"Files":[invalid]}}), &spec).is_err());
        assert!(
            parse_metadata(
                json!({"Success":true,"Data":{"Files":[entry.clone(),entry]}}),
                &spec
            )
            .is_err()
        );
    }

    #[test]
    fn source_defaults_and_explicit_revisions_select_the_right_repository() {
        let hf = Config::parse_from(["streamtrans"]);
        let hf_specs = specifications(&hf);
        assert_eq!(hf_specs[0].revision, HF_ASR_REVISION);
        assert_eq!(hf_specs[1].id, "funasr/fsmn-vad");
        assert_eq!(hf_specs[2].id, "tencent/Hy-MT2-1.8B-GGUF");
        let ms = Config::parse_from(["streamtrans", "--model-source", "modelscope"]);
        let ms_specs = specifications(&ms);
        assert_eq!(
            ms_specs[0].revision,
            "05201c46f1c38592b1567f857c0d56eab3d0d8ef"
        );
        assert_eq!(ms_specs[1].id, VAD_ID);
        assert_eq!(ms_specs[2].id, LLM_ID);
        let explicit = Config::parse_from([
            "streamtrans",
            "--asr-revision",
            "refs/pr/7",
            "--vad-revision",
            "vad-tag",
            "--llm-revision",
            "llm-tag",
        ]);
        let specs = specifications(&explicit);
        assert_eq!(specs[0].revision, "refs/pr/7");
        assert_eq!(specs[1].revision, "vad-tag");
        assert_eq!(specs[2].revision, "llm-tag");
        assert!(!snapshot_path(Path::new("cache"), &specs[0]).ends_with("refs/pr/7"));
    }

    #[tokio::test]
    async fn source_identity_is_part_of_manifest_matching_and_snapshot_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().canonicalize().unwrap();
        let config = Config::parse_from(["streamtrans"]);
        let hf_specs = specifications(&config);
        let ms_specs = hf_specs.map(|spec| Specification {
            source: ModelSource::ModelScope,
            ..spec
        });
        let current = cached_fixture(&cache, &config);
        assert_ne!(
            manifest_path(&cache, &hf_specs),
            manifest_path(&cache, &ms_specs)
        );
        assert_ne!(
            snapshot_path(&cache, &hf_specs[0]),
            snapshot_path(&cache, &ms_specs[0])
        );
        assert!(!matching_revisions(&current, &ms_specs));
        write_manifest(&cache, &current, &hf_specs).unwrap();
        assert!(cached_manifest(&cache, &ms_specs).await.unwrap().is_none());
        let mut legacy_json = serde_json::to_value(&current).unwrap();
        legacy_json.as_object_mut().unwrap().remove("source");
        let legacy: Manifest = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(legacy.source, ModelSource::ModelScope);
        assert!(!matching_revisions(&legacy, &hf_specs));
        assert!(matching_revisions(&legacy, &ms_specs));
    }

    #[tokio::test]
    async fn malformed_manifests_from_the_other_source_do_not_block_cache_lookup() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().canonicalize().unwrap();
        let hf = Config::parse_from(["streamtrans"]);
        let ms = Config::parse_from(["streamtrans", "--model-source", "modelscope"]);
        let hf_path = manifest_path(&cache, &specifications(&hf));
        std::fs::write(&hf_path, b"invalid").unwrap();
        assert!(
            cached_manifest(&cache, &specifications(&ms))
                .await
                .unwrap()
                .is_none()
        );
        std::fs::remove_file(&hf_path).unwrap();
        std::fs::write(cache.join(MANIFEST_NAME), b"invalid").unwrap();
        assert!(
            cached_manifest(&cache, &specifications(&hf))
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn huggingface_metadata_resolves_branches_and_hashes_inline_files() {
        let revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let spec = Specification {
            source: ModelSource::HuggingFace,
            key: "asr_dir",
            id: "org/model",
            revision: "refs/pr/7",
            files: &["model.pt", "sub/config.json"],
        };
        let payload = json!({"sha":revision,"siblings":[
            {"rfilename":"model.pt","size":3,"lfs":{"size":3,"sha256":record(b"yes").sha256}},
            {"rfilename":"sub/config.json","size":2,"blobId":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
        ]});
        let server = mock_server(BTreeMap::from([
            (
                "/api/models/org/model/revision/refs%2Fpr%2F7?blobs=true".into(),
                (200, serde_json::to_vec(&payload).unwrap()),
            ),
            (
                format!("/org/model/resolve/{revision}/sub/config.json"),
                (200, b"{}".to_vec()),
            ),
            (
                format!("/org/model/resolve/{revision}/model.pt"),
                (200, b"yes".to_vec()),
            ),
        ]))
        .await;
        let client = reqwest::Client::new();
        let metadata = metadata_at(&client, &spec, &server.base).await.unwrap();
        assert_eq!(metadata.revision, revision);
        assert_eq!(
            metadata.files["sub/config.json"].sha256,
            record(b"{}").sha256
        );
        assert_eq!(metadata.files["model.pt"].sha256, record(b"yes").sha256);
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().canonicalize().unwrap();
        let resolved = Specification {
            revision: &metadata.revision,
            ..spec
        };
        download_file_at(
            &client,
            &resolved,
            "model.pt",
            &metadata.files["model.pt"],
            &cache.join("model.pt"),
            &cache,
            &server.base,
        )
        .await
        .unwrap();
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(
            requests[1..]
                .iter()
                .all(|request| request.contains(revision) && !request.contains("refs"))
        );
    }

    #[tokio::test]
    async fn huggingface_metadata_rejects_mismatch_missing_duplicate_and_invalid_hashes() {
        let revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let spec = Specification {
            source: ModelSource::HuggingFace,
            key: "llm_file",
            id: "org/model",
            revision,
            files: &[LLM_NAME],
        };
        let entry =
            json!({"rfilename":LLM_NAME,"size":3,"lfs":{"size":3,"sha256":record(b"yes").sha256}});
        let valid = json!({"sha":revision,"siblings":[entry.clone()]});
        let client = reqwest::Client::new();
        assert!(
            hf_metadata(&client, &spec, valid.clone(), "https://unused.invalid")
                .await
                .is_ok()
        );
        let mut wrong_revision = valid.clone();
        wrong_revision["sha"] = json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        assert!(
            hf_metadata(&client, &spec, wrong_revision, "https://unused.invalid")
                .await
                .unwrap_err()
                .to_string()
                .contains("SHA")
        );
        for payload in [
            json!({"sha":revision,"siblings":[]}),
            json!({"sha":revision,"siblings":[entry.clone(),entry.clone()]}),
            json!({"sha":revision,"siblings":[{"rfilename":LLM_NAME,"size":3,"lfs":{"size":3,"sha256":"bad"}}]}),
            json!({"sha":revision,"siblings":[{"rfilename":LLM_NAME,"size":3,"lfs":{"size":4,"sha256":record(b"yes").sha256}}]}),
            json!({"sha":revision,"siblings":[{"rfilename":LLM_NAME,"size":MAX_HF_INLINE_BYTES+1,"blobId":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]}),
        ] {
            assert!(
                hf_metadata(&client, &spec, payload, "https://unused.invalid")
                    .await
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn both_sources_download_validate_and_preserve_files_on_failure() {
        for source in [ModelSource::HuggingFace, ModelSource::ModelScope] {
            let spec = Specification {
                source,
                key: "llm_file",
                id: "org/model",
                revision: "fixed",
                files: &["good"],
            };
            let client = reqwest::Client::new();
            let mut responses = BTreeMap::new();
            for (name, code, bytes) in [
                ("good", 200, b"yes".as_slice()),
                ("corrupt", 200, b"bad".as_slice()),
                ("short", 200, b"no".as_slice()),
                ("partial", 206, b"yes".as_slice()),
                ("oversize", 200, b"long".as_slice()),
            ] {
                let request = file_request(&client, &spec, name, "http://localhost")
                    .unwrap()
                    .build()
                    .unwrap();
                let url = request.url();
                let key = format!(
                    "{}{}",
                    url.path(),
                    url.query()
                        .map(|query| format!("?{query}"))
                        .unwrap_or_default()
                );
                responses.insert(key, (code, bytes.to_vec()));
            }
            let server = mock_server(responses).await;
            let directory = tempfile::tempdir().unwrap();
            let cache = directory.path().canonicalize().unwrap();
            let good = cache.join("good");
            download_file_at(
                &client,
                &spec,
                "good",
                &record(b"yes"),
                &good,
                &cache,
                &server.base,
            )
            .await
            .unwrap();
            assert_eq!(std::fs::read(&good).unwrap(), b"yes");
            let before = server.requests.lock().unwrap().len();
            download_file_at(
                &client,
                &spec,
                "good",
                &record(b"yes"),
                &good,
                &cache,
                &server.base,
            )
            .await
            .unwrap();
            assert_eq!(server.requests.lock().unwrap().len(), before);
            for name in ["corrupt", "short", "partial", "oversize", "missing"] {
                let target = cache.join(name);
                std::fs::write(&target, b"old").unwrap();
                let error = download_file_at(
                    &client,
                    &spec,
                    name,
                    &record(b"yes"),
                    &target,
                    &cache,
                    &server.base,
                )
                .await
                .unwrap_err();
                if name == "corrupt" {
                    assert!(error.to_string().contains("SHA256"));
                }
                assert_eq!(std::fs::read(target).unwrap(), b"old");
            }
            assert!(std::fs::read_dir(&cache).unwrap().all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".partial")
            }));
        }
    }

    #[tokio::test]
    async fn legacy_cache_migrates_without_network_and_corruption_fails_offline() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().canonicalize().unwrap();
        let mut config = Config::parse_from(["streamtrans", "--model-source", "modelscope"]);
        config.cache_dir = cache.clone();
        let manifest = cached_fixture(&cache, &config);
        let legacy = cache.join(".streamtrans-models-legacy.json");
        let mut legacy_json = serde_json::to_value(&manifest).unwrap();
        legacy_json.as_object_mut().unwrap().remove("source");
        std::fs::write(&legacy, serde_json::to_vec(&legacy_json).unwrap()).unwrap();
        let models = download(&config).await.unwrap();
        assert_eq!(models.llm_file, manifest.paths.llm_file);
        assert!(manifest_path(&cache, &specifications(&config)).is_file());
        std::fs::remove_file(legacy).unwrap();
        download(&config).await.unwrap();
        let path = models.asr_dir.join("model.pt");
        std::fs::write(&path, vec![b'x'; "model.pt".len()]).unwrap();
        let error = download(&config).await.unwrap_err();
        assert!(format!("{error:#}").contains("SHA256"));
    }

    #[tokio::test]
    async fn manifest_rejects_external_paths_or_changed_file_sets() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().canonicalize().unwrap();
        let config = Config::parse_from(["streamtrans"]);
        let mut manifest = cached_fixture(&cache, &config);
        let outside = tempfile::tempdir().unwrap();
        let external = outside.path().join(LLM_NAME);
        std::fs::write(&external, LLM_NAME.as_bytes()).unwrap();
        manifest.paths.llm_file = external;
        assert!(
            validate_manifest(&manifest, &specifications(&config), &cache)
                .await
                .is_err()
        );
        let mut manifest = cached_fixture(&cache, &config);
        manifest
            .models
            .get_mut("asr_dir")
            .unwrap()
            .files
            .remove("model.pt");
        assert!(
            validate_manifest(&manifest, &specifications(&config), &cache)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn model_switch_keeps_former_manifest_and_never_reuses_its_llm() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().canonicalize().unwrap();
        let config = Config::parse_from(["streamtrans"]);
        let specs = specifications(&config);
        let current = cached_fixture(&cache, &config);
        let mut former = cached_fixture(&cache, &config);
        let former_path = cache.join("MiniCPM5-2B-Q4_K_M.gguf");
        std::fs::write(&former_path, b"former model").unwrap();
        former.paths.llm_file = former_path.clone();
        former.models.insert(
            "llm_file".into(),
            CachedModel {
                id: "OpenBMB/MiniCPM5-2B-GGUF".into(),
                revision: "5cd297aab1231e38021260d1813963aff897c0da".into(),
                resolved_revision: None,
                files: BTreeMap::from([(
                    "MiniCPM5-2B-Q4_K_M.gguf".into(),
                    record(b"former model"),
                )]),
            },
        );
        let former_manifest = serde_json::to_vec(&former).unwrap();
        let former_manifest_path = cache.join(MANIFEST_NAME);
        std::fs::write(&former_manifest_path, &former_manifest).unwrap();
        assert!(cached_manifest(&cache, &specs).await.unwrap().is_none());
        write_manifest(&cache, &current, &specs).unwrap();
        assert_eq!(
            std::fs::read(former_manifest_path).unwrap(),
            former_manifest
        );
        assert_eq!(std::fs::read(former_path).unwrap(), b"former model");
        let reused = cached_manifest(&cache, &specs).await.unwrap().unwrap();
        assert_eq!(reused.paths.llm_file, current.paths.llm_file);
        assert!(reused.paths.llm_file.ends_with(LLM_NAME));
    }
}
