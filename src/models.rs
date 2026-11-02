use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::{config::Config, runtime::Models};

const ASR_ID: &str = "FunAudioLLM/Fun-ASR-Nano-2512";
const VAD_ID: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";
const LLM_ID: &str = "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF";
const LLM_NAME: &str = "Hy-MT2-1.8B-Q6_K.gguf";
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

struct Specification<'a> {
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
        ensure!(self.size > 0, "ModelScope 文件大小必须大于零");
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|c| c.is_ascii_hexdigit()),
            "ModelScope 文件缺少有效 SHA256"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
struct CachedModel {
    id: String,
    revision: String,
    files: BTreeMap<String, FileRecord>,
}

#[derive(Deserialize, Serialize)]
struct Manifest {
    version: u32,
    paths: Models,
    models: BTreeMap<String, CachedModel>,
}

fn specifications(config: &Config) -> [Specification<'_>; 3] {
    [
        Specification {
            key: "asr_dir",
            id: ASR_ID,
            revision: &config.asr_revision,
            files: ASR_FILES,
        },
        Specification {
            key: "vad_dir",
            id: VAD_ID,
            revision: &config.vad_revision,
            files: VAD_FILES,
        },
        Specification {
            key: "llm_file",
            id: LLM_ID,
            revision: &config.llm_revision,
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
        manifest
            .models
            .get(spec.key)
            .is_some_and(|model| model.id == spec.id && model.revision == spec.revision)
    })
}

fn manifest_path(cache: &Path, specs: &[Specification<'_>]) -> PathBuf {
    // Keep former model selections and their manifests intact when changing the LLM.
    let mut digest = Sha256::new();
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
        ".streamtrans-models-native-{:x}.json",
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
    let legacy_native = cache.join(MANIFEST_NAME);
    if legacy_native.is_file() {
        paths.push(legacy_native);
    }
    let mut directory = tokio::fs::read_dir(cache).await?;
    while let Some(entry) = directory.next_entry().await? {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(".streamtrans-models-")
            && name.ends_with(".json")
            && !paths.contains(&entry.path())
        {
            paths.push(entry.path());
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

async fn metadata(
    client: &reqwest::Client,
    spec: &Specification<'_>,
) -> Result<BTreeMap<String, FileRecord>> {
    let payload = client
        .get(format!(
            "https://modelscope.cn/api/v1/models/{}/repo/files",
            spec.id
        ))
        .query(&[("Revision", spec.revision), ("Recursive", "true")])
        .timeout(Duration::from_secs(60))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    parse_metadata(payload, spec)
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
        model = spec.id,
        file = name,
        bytes = record.size,
        "从 ModelScope 下载模型文件"
    );
    let mut response = client
        .get(format!(
            "https://modelscope.cn/api/v1/models/{}/repo",
            spec.id
        ))
        .query(&[("Revision", spec.revision), ("FilePath", name)])
        .send()
        .await?
        .error_for_status()?;
    ensure!(
        response.status() == reqwest::StatusCode::OK,
        "ModelScope 必须返回完整文件（HTTP 200）"
    );
    if let Some(size) = response.content_length() {
        ensure!(
            size == record.size,
            "ModelScope 下载文件大小与元数据不符：{name}"
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
    while let Some(chunk) = response.chunk().await? {
        received = received
            .checked_add(chunk.len() as u64)
            .context("模型文件大小溢出")?;
        ensure!(
            received <= record.size,
            "ModelScope 下载文件超过声明大小：{name}"
        );
        digest.update(&chunk);
        output.write_all(&chunk).await?;
    }
    ensure!(received == record.size, "ModelScope 下载文件不完整：{name}");
    ensure!(
        format!("{:x}", digest.finalize()) == record.sha256,
        "ModelScope 下载文件 SHA256 校验失败：{name}"
    );
    output.flush().await?;
    output.sync_all().await?;
    drop(output);
    partial
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("无法保存模型文件：{}", path.display()))?;
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
    tracing::info!(cache = %cache.display(), "检查 ModelScope 模型缓存与 SHA256");
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
        version: 1,
        paths,
        models: BTreeMap::new(),
    };
    for spec in &specs {
        let files = metadata(&client, spec)
            .await
            .with_context(|| format!("读取 ModelScope 元数据失败：{}", spec.id))?;
        for name in spec.files {
            let path = file_path(&manifest.paths, spec.key, name);
            download_file(&client, spec, name, &files[*name], &path, &cache)
                .await
                .with_context(|| format!("下载 ModelScope 模型失败：{}/{name}", spec.id))?;
        }
        manifest.models.insert(
            spec.key.to_owned(),
            CachedModel {
                id: spec.id.to_owned(),
                revision: spec.revision.to_owned(),
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
    use clap::Parser;
    use serde_json::json;

    fn record(data: &[u8]) -> FileRecord {
        FileRecord {
            size: data.len() as u64,
            sha256: format!("{:x}", Sha256::digest(data)),
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
                    files,
                },
            );
        }
        Manifest {
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

    #[tokio::test]
    async fn legacy_cache_migrates_without_network_and_corruption_fails_offline() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().canonicalize().unwrap();
        let mut config = Config::parse_from(["streamtrans"]);
        config.cache_dir = cache.clone();
        let manifest = cached_fixture(&cache, &config);
        let legacy = cache.join(".streamtrans-models-legacy.json");
        std::fs::write(&legacy, serde_json::to_vec(&manifest).unwrap()).unwrap();
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
