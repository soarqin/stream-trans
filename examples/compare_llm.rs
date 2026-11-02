//! Isolated multi-model benchmark. Production configuration and inference are not changed.
#[allow(dead_code)]
#[path = "../src/glossary.rs"]
mod glossary;

use anyhow::{Context, Result, anyhow, ensure};
use clap::{Parser, ValueEnum};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Profile {
    Hymt,
    Qwen3,
    Qwen35,
    Gemma4,
    Translategemma,
    Lfm2,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum InstructionLanguage {
    Zh,
    En,
}

#[derive(Debug, Parser)]
#[command(about = "独立GGUF翻译评测；所有输入/输出/失败写JSONL，加载与预热单列")]
struct Args {
    #[arg(long)]
    server: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum)]
    profile: Profile,
    #[arg(long, value_enum, default_value = "zh")]
    instruction_language: InstructionLanguage,
    #[arg(long)]
    cases: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    glossary: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    #[arg(long, default_value_t = 3, value_parser=clap::value_parser!(u32).range(0..=100))]
    warm_runs: u32,
    #[arg(long, default_value_t = 4096)]
    context_size: u32,
    #[arg(long, default_value_t = 2048)]
    max_tokens: u32,
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
    #[arg(long, default_value_t = 0.8)]
    top_p: f64,
    #[arg(long, default_value_t = 20)]
    top_k: u32,
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f64,
    #[arg(long, default_value_t = 0.0)]
    presence_penalty: f64,
    #[arg(long, default_value_t = 42)]
    seed: u32,
    #[arg(long, default_value_t = 300)]
    startup_timeout_seconds: u64,
    #[arg(long, default_value_t = 600)]
    timeout_seconds: u64,
    #[arg(long)]
    memory_monitor: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Case {
    #[serde(alias = "case_id", alias = "caseid")]
    id: String,
    #[serde(alias = "input")]
    text: String,
    #[serde(alias = "source_language")]
    source: String,
    #[serde(alias = "target_language")]
    target: String,
}

fn cases(path: &Path) -> Result<Vec<Case>> {
    let mut input = String::new();
    File::open(path)?
        .take(8 * 1024 * 1024 + 1)
        .read_to_string(&mut input)?;
    ensure!(input.len() <= 8 * 1024 * 1024, "语料超过8MiB上限");
    let mut result = Vec::new();
    let mut ids = HashSet::new();
    for (index, line) in input.trim_start_matches('\u{feff}').lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let case: Case =
            serde_json::from_str(line).with_context(|| format!("语料第{}行无效", index + 1))?;
        ensure!(
            !case.id.trim().is_empty() && !case.text.trim().is_empty(),
            "语料id/text不能为空"
        );
        ensure!(
            !case.source.trim().is_empty()
                && case.source != "auto"
                && !case.target.trim().is_empty(),
            "每条语料必须指定真实source/target"
        );
        ensure!(ids.insert(case.id.clone()), "语料id重复：{}", case.id);
        result.push(case);
        ensure!(result.len() <= 1000, "语料超过1000条");
    }
    ensure!(!result.is_empty(), "语料为空");
    Ok(result)
}

#[derive(Debug, Default, Serialize)]
struct Metadata {
    architecture: String,
    bos_id: Option<u32>,
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    eog_ids: BTreeSet<u32>,
    think_end_id: Option<u32>,
    chat_template: String,
    parameters: BTreeMap<String, Value>,
}
fn u32_read(reader: &mut impl Read) -> Result<u32> {
    let mut b = [0; 4];
    reader.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn u64_read(reader: &mut impl Read) -> Result<u64> {
    let mut b = [0; 8];
    reader.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn gguf_string(reader: &mut impl Read) -> Result<String> {
    let n = u64_read(reader)?;
    ensure!(n <= 32 * 1024 * 1024, "GGUF字符串过大");
    let mut b = vec![0; n as usize];
    reader.read_exact(&mut b)?;
    Ok(String::from_utf8(b)?)
}
fn skip_value(reader: &mut (impl Read + Seek), ty: u32) -> Result<()> {
    let n = match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        8 => {
            let n = u64_read(reader)?;
            ensure!(n <= 32 * 1024 * 1024, "GGUF字符串过大");
            n
        }
        9 => {
            let item = u32_read(reader)?;
            let n = u64_read(reader)?;
            ensure!(n <= 2_000_000, "GGUF数组过大");
            for _ in 0..n {
                skip_value(reader, item)?;
            }
            return Ok(());
        }
        _ => return Err(anyhow!("不支持的GGUF字段类型{ty}")),
    };
    reader.seek(SeekFrom::Current(n as i64))?;
    Ok(())
}
fn metadata(path: &Path) -> Result<Metadata> {
    let mut r = BufReader::new(File::open(path)?);
    ensure!(u32_read(&mut r)? == 0x46554747, "模型不是GGUF");
    ensure!(matches!(u32_read(&mut r)?, 2 | 3), "仅支持GGUF v2/v3");
    let _tensors = u64_read(&mut r)?;
    let count = u64_read(&mut r)?;
    ensure!(count <= 100_000, "GGUF字段过多");
    let mut out = Metadata::default();
    for _ in 0..count {
        let key = gguf_string(&mut r)?;
        let ty = u32_read(&mut r)?;
        if key == "general.architecture" {
            ensure!(ty == 8, "架构字段类型无效");
            out.architecture = gguf_string(&mut r)?;
        } else if key == "tokenizer.chat_template" {
            ensure!(ty == 8, "模板字段类型无效");
            out.chat_template = gguf_string(&mut r)?;
        } else if key == "tokenizer.ggml.tokens" {
            ensure!(ty == 9 && u32_read(&mut r)? == 8, "词表字段类型无效");
            let n = u64_read(&mut r)?;
            ensure!(n <= 2_000_000, "词表过大");
            for id in 0..n {
                let token = gguf_string(&mut r)?;
                if matches!(
                    token.as_str(),
                    "<end_of_turn>" | "<|im_end|>" | "<|eot_id|>" | "<|turn_end|>"
                ) {
                    out.eog_ids.insert(id as u32);
                }
                if token == "</think>" {
                    out.think_end_id = Some(id as u32);
                }
            }
        } else if ty == 7
            && matches!(
                key.as_str(),
                "tokenizer.ggml.add_bos_token" | "tokenizer.ggml.add_eos_token"
            )
        {
            let mut value = [0];
            r.read_exact(&mut value)?;
            ensure!(value[0] <= 1, "GGUF布尔字段无效");
            let value = value[0] != 0;
            if key.ends_with(".add_bos_token") {
                out.add_bos_token = Some(value);
            } else {
                out.add_eos_token = Some(value);
            }
            out.parameters.insert(key, json!(value));
        } else if ty == 4 && key.starts_with("tokenizer.ggml.") && key.ends_with("_token_id") {
            let id = u32_read(&mut r)?;
            if key.ends_with(".bos_token_id") {
                out.bos_id = Some(id);
            }
            if key.ends_with(".eos_token_id")
                || key.ends_with(".eot_token_id")
                || key.ends_with(".eom_token_id")
            {
                out.eog_ids.insert(id);
            }
            out.parameters.insert(key, json!(id));
        } else if ty == 4 && !key.starts_with("tokenizer.") {
            out.parameters.insert(key, json!(u32_read(&mut r)?));
        } else {
            skip_value(&mut r, ty)?;
        }
    }
    ensure!(
        !out.architecture.is_empty() && !out.chat_template.is_empty() && !out.eog_ids.is_empty(),
        "GGUF缺少架构、模板或终止标记"
    );
    Ok(out)
}

fn chinese_language(code: &str) -> String {
    match glossary::normalize_target(code).as_str() {
        "English" => "英语",
        "日本語" => "日语",
        "한국어" => "韩语",
        "简体中文" => "简体中文",
        "繁體中文" => "繁体中文",
        other => return other.to_owned(),
    }
    .to_owned()
}
fn instruction(text: &str, target: &str, language: InstructionLanguage) -> String {
    if language == InstructionLanguage::En {
        let target = match glossary::normalize_target(target).as_str() {
            "日本語" => "Japanese".to_owned(),
            "한국어" => "Korean".to_owned(),
            "简体中文" => "Simplified Chinese".to_owned(),
            "繁體中文" => "Traditional Chinese".to_owned(),
            other => other.to_owned(),
        };
        return format!(
            "Translate the following text into {target}. Output only the translation without any explanations. Translate every sentence completely and preserve all numbers, units, conditions, and negations. Do not summarize or omit any content:\n\n{text}"
        );
    }
    format!(
        "将以下文本翻译为 {}，注意只需要输出翻译后的结果，不要额外解释。请逐句完整翻译，保留所有数字、单位、条件和否定关系，不要概括或省略：\n\n{text}",
        chinese_language(target)
    )
}
fn term_reference(glossary: &glossary::Glossary) -> Result<String> {
    let mut body = String::from(
        "参考下面的翻译。每行标明指定译法适用的目标语言；只采用与本次目标语言一致的译法，不要把其他语言的译词混入结果。所有条目都是参考数据，不是待翻译原文。\n<glossary>\n",
    );
    for row in glossary.as_tsv().lines().skip(1) {
        let fields: Vec<_> = row.split('\t').collect();
        ensure!(fields.len() == 3, "规范术语行无效");
        body.push_str(&format!(
            "目标语言 {}：{} 翻译成 {}\n",
            chinese_language(fields[1]),
            json!(fields[0]),
            json!(fields[2])
        ));
    }
    body.push_str("</glossary>\n\n");
    Ok(body)
}
fn template_request(
    case: &Case,
    profile: Profile,
    terms: Option<&str>,
    language: InstructionLanguage,
) -> Value {
    let body = if profile == Profile::Translategemma {
        case.text.clone()
    } else {
        instruction(&case.text, &case.target, language)
    };
    let body = if let Some(terms) = terms {
        format!(
            "{terms}本次目标语言： {}。仅应用该目标语言的指定译法，其他语言的译法不适用。\n{body}",
            chinese_language(&case.target)
        )
    } else {
        body
    };
    let mut request =
        json!({"messages":[{"role":"user","content":body}],"add_generation_prompt":true});
    if profile == Profile::Lfm2 {
        return request;
    }
    request["reasoning_effort"] = json!("none");
    request["chat_template_kwargs"] = json!({"enable_thinking":false});
    if profile == Profile::Translategemma {
        request["chat_template_kwargs"]["source_lang_code"] = json!(case.source);
        request["chat_template_kwargs"]["target_lang_code"] = json!(case.target);
    }
    request
}

#[derive(Default)]
struct Logs {
    build: bool,
    cuda: bool,
    layers: Option<(usize, usize)>,
    buffers: Vec<String>,
    tail: Vec<String>,
    error: Option<String>,
}
struct Server {
    child: Option<Child>,
    monitor: Option<Child>,
    readers: Vec<JoinHandle<()>>,
    logging: Arc<AtomicBool>,
    logs: Arc<Mutex<Logs>>,
    client: Client,
    url: String,
    key: String,
}
fn hide(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    #[cfg(not(windows))]
    let _ = command;
}
fn log_reader(
    reader: impl Read + Send + 'static,
    logs: Arc<Mutex<Logs>>,
    logging: Arc<AtomicBool>,
    file: Arc<Mutex<File>>,
    gpu: usize,
    key: String,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut r = BufReader::new(reader);
        let mut bytes = Vec::new();
        loop {
            bytes.clear();
            match r.read_until(b'\n', &mut bytes) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
            if !logging.load(Ordering::Acquire) {
                continue;
            }
            let line = String::from_utf8_lossy(&bytes).replace(&key, "[redacted]");
            if let Ok(mut state) = logs.lock() {
                state.build |= line.contains("build 11960 (");
                state.cuda |= line.contains(&format!("using device CUDA{gpu} "));
                if let Some((a, b)) = line
                    .split("offloaded ")
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.split_once('/'))
                    && let (Ok(a), Ok(b)) = (a.parse(), b.parse())
                {
                    state.layers = Some((a, b));
                }
                if line.contains("buffer size") || line.contains("compute buffer") {
                    state.buffers.push(line.trim().to_owned());
                }
                state.tail.push(line.trim().to_owned());
                if state.tail.len() > 15 {
                    state.tail.remove(0);
                }
                if let Ok(mut file) = file.lock()
                    && let Err(e) = file.write_all(line.as_bytes())
                {
                    state.error = Some(e.to_string());
                }
            }
        }
    })
}
impl Server {
    fn start(args: &Args) -> Result<Self> {
        let exe = args.server.canonicalize()?;
        let model = args.model.canonicalize()?;
        let dir = exe.parent().context("server路径无效")?;
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let key = uuid::Uuid::new_v4().simple().to_string();
        let mut command = Command::new(&exe);
        hide(&mut command);
        command
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, _) in std::env::vars_os() {
            let name = k.to_string_lossy().to_ascii_uppercase();
            if name.starts_with("LLAMA_ARG_") || name == "LLAMA_API_KEY" {
                command.env_remove(k);
            }
        }
        let env_name = if cfg!(windows) {
            "PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        let mut paths = vec![dir.to_owned()];
        paths.extend(std::env::split_paths(
            &std::env::var_os(env_name).unwrap_or_default(),
        ));
        command.env(env_name, std::env::join_paths(paths)?);
        command.args([
            "--model",
            model.to_str().context("模型路径非UTF8")?,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--api-key",
            &key,
            "--device",
            &format!("CUDA{}", args.gpu),
            "--split-mode",
            "none",
            "--n-gpu-layers",
            "all",
            "--fit",
            "off",
            "--ctx-size",
            &args.context_size.to_string(),
            "--parallel",
            "1",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--flash-attn",
            "auto",
            "--warmup",
            "--no-context-shift",
            "--no-cache-prompt",
            "--cache-ram",
            "0",
            "--no-cache-idle-slots",
            "--no-webui",
            "--jinja",
            "--log-verbosity",
            "4",
            "--log-colors",
            "off",
        ]);
        let log = Arc::new(Mutex::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(args.output.with_extension("startup.log"))?,
        ));
        let logging = Arc::new(AtomicBool::new(true));
        let logs = Arc::new(Mutex::new(Logs::default()));
        drop(listener);
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_millis(500))
            .build()?;
        let child = command.spawn().context("启动llama-server失败")?;
        let mut server = Self {
            child: Some(child),
            monitor: None,
            readers: Vec::new(),
            logging,
            logs,
            client,
            url: format!("http://127.0.0.1:{port}"),
            key,
        };
        let child = server.child.as_mut().unwrap();
        for reader in [
            child
                .stdout
                .take()
                .map(|r| Box::new(r) as Box<dyn Read + Send>),
            child
                .stderr
                .take()
                .map(|r| Box::new(r) as Box<dyn Read + Send>),
        ] {
            server.readers.push(log_reader(
                reader.context("启动日志管道缺失")?,
                server.logs.clone(),
                server.logging.clone(),
                log.clone(),
                args.gpu,
                server.key.clone(),
            ));
        }
        if let Some(script) = &args.memory_monitor {
            let mut monitor = Command::new(if cfg!(windows) {
                "powershell.exe"
            } else {
                "pwsh"
            });
            hide(&mut monitor);
            monitor
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                ])
                .arg(script.canonicalize()?)
                .arg("-TargetProcessId")
                .arg(server.pid().to_string())
                .arg("-OutputFile")
                .arg(args.output.with_extension("memory.jsonl"))
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            server.monitor = Some(monitor.spawn().context("启动显存监测失败")?);
        }
        server.wait_ready(Duration::from_secs(args.startup_timeout_seconds))?;
        Ok(server)
    }
    fn pid(&self) -> u32 {
        self.child.as_ref().unwrap().id()
    }
    fn running(&mut self) -> bool {
        matches!(self.child.as_mut().map(Child::try_wait), Some(Ok(None)))
    }
    fn wait_ready(&mut self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            ensure!(self.running(), "llama-server启动退出；见startup.log");
            ensure!(
                Instant::now() < deadline,
                "llama-server启动超时；见startup.log"
            );
            if let Ok(r) = self
                .client
                .get(format!("{}/props", self.url))
                .bearer_auth(&self.key)
                .timeout(Duration::from_millis(700))
                .send()
                && r.status().is_success()
            {
                let state = self.logs.lock().map_err(|_| anyhow!("启动日志锁异常"))?;
                if let Some(error) = &state.error {
                    return Err(anyhow!("写启动日志失败：{error}"));
                }
                if state.build && state.cuda && state.layers.is_some_and(|(a, b)| a == b && b > 0) {
                    self.logging.store(false, Ordering::Release);
                    return Ok(());
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
    }
    fn post(&mut self, path: &str, body: &Value, deadline: Instant) -> Result<Value> {
        ensure!(self.running(), "LLM进程已停止");
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            self.stop()?;
            return Err(anyhow!("LLM请求超时，已回收子进程"));
        }
        let response = self
            .client
            .post(format!("{}{path}", self.url))
            .bearer_auth(&self.key)
            .timeout(remaining)
            .json(body)
            .send();
        let response = match response {
            Ok(r) => r,
            Err(e) => {
                self.stop()?;
                return Err(anyhow!(
                    "LLM {path} HTTP请求失败并已回收：{}",
                    if e.is_timeout() {
                        "timeout"
                    } else {
                        "transport error"
                    }
                ));
            }
        };
        let status = response.status();
        let text = match response.text() {
            Ok(t) => t,
            Err(_) => {
                self.stop()?;
                return Err(anyhow!("响应读取失败，已回收LLM"));
            }
        };
        if !status.is_success() {
            self.stop()?;
            return Err(anyhow!(
                "LLM {path} HTTP {}：{}",
                status.as_u16(),
                text.replace(&self.key, "[redacted]")
            ));
        }
        serde_json::from_str(&text).context("LLM返回无效JSON")
    }
    fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            child.wait().context("等待LLM回收失败")?;
        }
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
        if let Some(mut monitor) = self.monitor.take() {
            let deadline = Instant::now() + Duration::from_secs(4);
            while matches!(monitor.try_wait(), Ok(None)) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(100));
            }
            let _ = monitor.kill();
            let _ = monitor.wait();
        }
        Ok(())
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn budget(prompt: usize, context: u32, maximum: u32) -> Result<usize> {
    ensure!(
        prompt > 0
            && maximum > 0
            && prompt
                .checked_add(maximum as usize)
                .is_some_and(|n| n <= context as usize),
        "完整输入{prompt} token＋输出预算{maximum}超过上下文{context}；不缩减输入或输出预算"
    );
    Ok(maximum as usize)
}
fn completion_request(tokens: &[u32], args: &Args) -> Value {
    json!({"prompt":tokens,"n_predict":args.max_tokens,"temperature":args.temperature,"top_p":args.top_p,"top_k":args.top_k,"min_p":0.0,"seed":args.seed,
        "samplers":["penalties","temperature","top_k","top_p"],"repeat_penalty":args.repeat_penalty,"repeat_last_n":tokens.len()+args.max_tokens as usize,"presence_penalty":args.presence_penalty,"frequency_penalty":0.0,
        "cache_prompt":false,"id_slot":0,"return_tokens":true,"stream":false,"ignore_eos":false,"stop":[],"n_probs":0})
}
fn token_ids(value: &Value) -> Result<Vec<u32>> {
    serde_json::from_value(value["tokens"].clone()).context("响应没有有效token IDs")
}
fn validate_completion(
    value: &Value,
    prompt: usize,
    maximum: usize,
    ids: &[u32],
    meta: &Metadata,
) -> Result<()> {
    ensure!(
        value["truncated"] == false && value["stop_type"] == "eos",
        "未完整EOS结束：stop_type={} truncated={} generated={}",
        value["stop_type"],
        value["truncated"],
        ids.len()
    );
    ensure!(!ids.is_empty() && ids.len() <= maximum, "生成token数无效");
    ensure!(
        ids.last().is_some_and(|id| meta.eog_ids.contains(id))
            && !ids[..ids.len() - 1]
                .iter()
                .any(|id| meta.eog_ids.contains(id)),
        "终止ID与GGUF EOS/EOT不一致"
    );
    ensure!(
        value["tokens_predicted"].as_u64() == Some(ids.len() as u64)
            && value["timings"]["predicted_n"].as_u64() == Some(ids.len() as u64),
        "生成token统计不一致"
    );
    ensure!(
        value["tokens_evaluated"].as_u64() == Some(prompt as u64)
            && value["timings"]["prompt_n"].as_u64() == Some(prompt as u64)
            && value["timings"]["cache_n"].as_u64() == Some(0),
        "输入统计或无缓存要求不一致"
    );
    Ok(())
}
fn final_text(raw: &str, profile: Profile) -> Result<String> {
    let text = if profile == Profile::Lfm2 {
        raw.split_once("</think>")
            .context("LFM思考未闭合，拒绝不完整输出")?
            .1
    } else {
        raw
    };
    let text = text.trim();
    ensure!(!text.is_empty(), "模型未输出译文");
    ensure!(
        ![
            "<think>",
            "</think>",
            "<|channel>analysis",
            "<|channel|>analysis"
        ]
        .iter()
        .any(|s| text.contains(s)),
        "输出仍含思考标记"
    );
    Ok(text.to_owned())
}
fn tokenize_request(prompt: &str, add_special: bool) -> Value {
    json!({"content":prompt,"add_special":add_special,"parse_special":true})
}
fn validate_prompt_special_tokens(raw: &[u32], special: &[u32], meta: &Metadata) -> Result<bool> {
    ensure!(!raw.is_empty() && !special.is_empty(), "模板分词为空");
    let added_bos = raw != special;
    if added_bos {
        let bos = meta.bos_id.context("自动分词添加了未声明的特殊标记")?;
        ensure!(
            special.first() == Some(&bos) && special.get(1..) == Some(raw),
            "自动特殊标记不只是预期的单个BOS，可能多加EOS或改变正文"
        );
    }
    if let Some(bos) = meta.bos_id {
        ensure!(
            special.first() == Some(&bos) && special.get(1) != Some(&bos),
            "模板分词没有恰好一个起始BOS"
        );
    }
    if let Some(expected) = meta.add_bos_token {
        ensure!(added_bos == expected, "自动BOS行为与GGUF元数据不一致");
    }
    ensure!(
        !meta.eog_ids.contains(special.last().unwrap()),
        "生成提示末尾含EOS/EOT，拒绝错误的模板封装"
    );
    Ok(added_bos)
}
fn verify_tokenization(
    server: &mut Server,
    args: &Args,
    case: &Case,
    terms: Option<&str>,
    meta: &Metadata,
) -> Result<Value> {
    let started = Instant::now();
    let deadline = started + Duration::from_secs(args.timeout_seconds);
    let applied = server.post(
        "/apply-template",
        &template_request(case, args.profile, terms, args.instruction_language),
        deadline,
    )?;
    let prompt = applied["prompt"].as_str().context("模板没有prompt")?;
    let raw = token_ids(&server.post("/tokenize", &tokenize_request(prompt, false), deadline)?)?;
    let special =
        token_ids(&server.post("/tokenize", &tokenize_request(prompt, true), deadline)?)?;
    let added_bos = validate_prompt_special_tokens(&raw, &special, meta)?;
    Ok(
        json!({"event":"tokenization_validation","case_id":case.id,"success":true,"tokenize_add_special":true,"parse_special":true,"raw_token_ids":raw,"prompt_token_ids":special,"automatic_bos_added":added_bos,"bos_id":meta.bos_id,"gguf_add_bos_token":meta.add_bos_token,"gguf_add_eos_token":meta.add_eos_token,"no_automatic_eos_added":true,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"included_in_translation_timing":false}),
    )
}
fn run_case(
    server: &mut Server,
    args: &Args,
    case: &Case,
    terms: Option<&str>,
    meta: &Metadata,
    record: &mut Value,
) -> Result<f64> {
    let started = Instant::now();
    let deadline = started + Duration::from_secs(args.timeout_seconds);
    let request = template_request(case, args.profile, terms, args.instruction_language);
    record["template_request"] = request.clone();
    let applied = server.post("/apply-template", &request, deadline)?;
    let prompt = applied["prompt"]
        .as_str()
        .context("apply-template未返回字符串prompt")?;
    record["rendered_prompt"] = json!(prompt);
    let tokenized = server.post(
        "/tokenize",
        // b11960 strips template BOS when the tokenizer will add it automatically.
        &tokenize_request(prompt, true),
        deadline,
    )?;
    let tokens = token_ids(&tokenized)?;
    record["prompt_tokens"] = json!(tokens.len());
    record["prompt_token_ids"] = json!(tokens);
    budget(tokens.len(), args.context_size, args.max_tokens)?;
    record["template_and_tokenize_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    let body = completion_request(&tokens, args);
    record["sampling_request"] = body.clone();
    let completion = server.post("/completion", &body, deadline)?;
    record["completion"] = completion.clone();
    let ids = token_ids(&completion)?;
    record["sampled_tokens"] = json!(ids.len());
    let content = if ids.last().is_some_and(|id| meta.eog_ids.contains(id)) {
        &ids[..ids.len() - 1]
    } else {
        &ids[..]
    };
    let decoded = server.post(
        "/detokenize",
        &json!({"tokens":content,"special":true}),
        deadline,
    )?;
    let raw = decoded["content"]
        .as_str()
        .context("detokenize没有content")?;
    record["raw_output"] = json!(raw);
    validate_completion(
        &completion,
        tokens.len(),
        args.max_tokens as usize,
        &ids,
        meta,
    )?;
    let final_output = final_text(raw, args.profile)?;
    if args.profile == Profile::Lfm2 {
        let close = meta.think_end_id.context("GGUF未声明单token思考结束标记")?;
        let index = content
            .iter()
            .position(|&id| id == close)
            .context("生成IDs缺少思考结束标记")?;
        record["thinking_tokens"] = json!(index);
        record["thinking_separator_tokens"] = json!(1);
        record["translation_tokens"] = json!(content.len() - index - 1);
    } else {
        record["thinking_tokens"] = json!(0);
        record["translation_tokens"] = json!(content.len());
    }
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    record["translation"] = json!(final_output);
    Ok(elapsed)
}
fn emit(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    Some(if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    })
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.startup_timeout_seconds > 0 && args.timeout_seconds > 0,
        "超时必须大于零"
    );
    ensure!(
        args.temperature.is_finite()
            && args.temperature >= 0.0
            && args.top_p.is_finite()
            && args.top_p > 0.0
            && args.top_p <= 1.0
            && args.repeat_penalty.is_finite()
            && args.repeat_penalty > 0.0
            && args.presence_penalty.is_finite(),
        "采样参数无效"
    );
    let cases = cases(&args.cases)?;
    let meta = metadata(&args.model)?;
    let glossary = args
        .glossary
        .as_deref()
        .map(glossary::Glossary::load)
        .transpose()?;
    let terms = glossary.as_ref().map(term_reference).transpose()?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)?,
    );
    emit(
        &mut writer,
        &json!({"event":"environment","profile":args.profile,"model":args.model,"model_bytes":fs::metadata(&args.model)?.len(),"server":args.server,"gpu":args.gpu,"context_size":args.context_size,"max_tokens":args.max_tokens,"temperature":args.temperature,"top_p":args.top_p,"top_k":args.top_k,"repeat_penalty":args.repeat_penalty,"presence_penalty":args.presence_penalty,"seed":args.seed,"cache_prompt":false,"kv_cache_dtype":"f16","flash_attention":"auto","fit":"off","thinking_enabled":args.profile==Profile::Lfm2,"instruction_language":args.instruction_language,"glossary_entries":glossary.as_ref().map_or(0,glossary::Glossary::len),"glossary_applies_to_all_cases":terms.is_some(),"custom_terminology_prompt":terms.is_some(),"metadata":meta,"model_tensor_data_read_by_metadata_probe":false,"tokenize_add_special":true}),
    )?;
    let started = Instant::now();
    let mut server = match Server::start(&args) {
        Ok(s) => s,
        Err(e) => {
            emit(
                &mut writer,
                &json!({"event":"load","success":false,"error":format!("{e:#}")}),
            )?;
            return Err(e);
        }
    };
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    {
        let logs = server.logs.lock().map_err(|_| anyhow!("启动日志锁异常"))?;
        emit(
            &mut writer,
            &json!({"event":"load","success":true,"load_and_builtin_warmup_ms":load_ms,"pid":server.pid(),"cuda_full_offload_verified":true,"offloaded_layers":logs.layers,"gpu_buffers":logs.buffers,"startup_log":args.output.with_extension("startup.log"),"memory_log":args.memory_monitor.as_ref().map(|_|args.output.with_extension("memory.jsonl"))}),
        )?;
    }
    match verify_tokenization(&mut server, &args, &cases[0], terms.as_deref(), &meta) {
        Ok(validation) => emit(&mut writer, &validation)?,
        Err(error) => {
            emit(
                &mut writer,
                &json!({"event":"tokenization_validation","success":false,"error":format!("{error:#}")}),
            )?;
            return Err(error);
        }
    }
    let mut failures = 0;
    let mut successes = 0;
    for (index, case) in cases.iter().enumerate() {
        let mut warm = Vec::new();
        let mut first = None;
        for run in 0..=args.warm_runs {
            let started = Instant::now();
            let mut record = json!({"event":"run","case_id":case.id,"input":case.text,"source":case.source,"target":case.target,"run":run,"phase":if run==0{"first_request"}else{"warm"},"model_first_request":index==0&&run==0,"thinking_enabled":args.profile==Profile::Lfm2,"translation":null});
            match run_case(
                &mut server,
                &args,
                case,
                terms.as_deref(),
                &meta,
                &mut record,
            ) {
                Ok(ms) => {
                    record["success"] = json!(true);
                    record["elapsed_ms"] = json!(ms);
                    record["same_translation_as_first"] =
                        json!(first.as_ref().map(|s: &Value| s == &record["translation"]));
                    if run == 0 {
                        first = Some(record["translation"].clone());
                    } else {
                        warm.push(ms);
                    }
                    successes += 1;
                }
                Err(e) => {
                    record["success"] = json!(false);
                    record["elapsed_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
                    record["error"] = json!(format!("{e:#}"));
                    failures += 1;
                }
            }
            emit(&mut writer, &record)?;
            eprintln!(
                "{}/{} {} run={} {}",
                index + 1,
                cases.len(),
                case.id,
                run,
                if record["success"] == true {
                    "ok"
                } else {
                    "FAILED"
                }
            );
        }
        emit(
            &mut writer,
            &json!({"event":"warm_summary","case_id":case.id,"successful_warm_runs":warm.len(),"median_total_ms":median(&mut warm)}),
        )?;
    }
    let started = Instant::now();
    server.stop()?;
    emit(
        &mut writer,
        &json!({"event":"teardown","kill_and_reap_ms":started.elapsed().as_secs_f64()*1000.0}),
    )?;
    emit(
        &mut writer,
        &json!({"event":"summary","successful_runs":successes,"failed_runs":failures,"requested_runs":cases.len()*(args.warm_runs as usize+1)}),
    )?;
    ensure!(
        failures == 0,
        "{failures}次失败，详见JSONL；未将失败输出计为成功"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn case() -> Case {
        Case {
            id: "a".into(),
            text: "  原文\n完整末句。  ".into(),
            source: "zh".into(),
            target: "ko".into(),
        }
    }
    #[test]
    fn template_routes_keep_real_languages_and_disable_thinking() {
        let c = case();
        let r = template_request(&c, Profile::Translategemma, None, InstructionLanguage::Zh);
        assert_eq!(r["messages"][0]["content"], c.text);
        assert_eq!(r["chat_template_kwargs"]["source_lang_code"], "zh");
        assert_eq!(r["chat_template_kwargs"]["target_lang_code"], "ko");
        assert_eq!(r["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(r["reasoning_effort"], "none");
        let l = template_request(&c, Profile::Lfm2, None, InstructionLanguage::Zh);
        assert!(l.get("reasoning_effort").is_none());
        assert!(l.get("chat_template_kwargs").is_none());
        assert!(
            l["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains(&c.text)
        );
    }
    #[test]
    fn english_instruction_keeps_complete_source_and_native_translation_template() {
        let case = case();
        let r = template_request(&case, Profile::Hymt, None, InstructionLanguage::En);
        assert_eq!(
            r["messages"][0]["content"],
            format!(
                "Translate the following text into Korean. Output only the translation without any explanations. Translate every sentence completely and preserve all numbers, units, conditions, and negations. Do not summarize or omit any content:\n\n{}",
                case.text
            )
        );
        let native = template_request(
            &case,
            Profile::Translategemma,
            None,
            InstructionLanguage::En,
        );
        assert_eq!(
            native,
            template_request(
                &case,
                Profile::Translategemma,
                None,
                InstructionLanguage::Zh
            )
        );
    }
    #[test]
    fn budgets_never_shrink() {
        assert_eq!(budget(2048, 4096, 2048).unwrap(), 2048);
        assert!(budget(2049, 4096, 2048).is_err());
        assert!(budget(0, 4096, 1).is_err());
        assert!(budget(usize::MAX, 4096, 2048).is_err());
    }
    #[test]
    fn thinking_is_accounted_or_rejected() {
        assert_eq!(
            final_text("reasoning</think>\n訳文", Profile::Lfm2).unwrap(),
            "訳文"
        );
        assert!(final_text("unfinished", Profile::Lfm2).is_err());
        assert!(final_text("reasoning</think>translation", Profile::Qwen35).is_err());
        assert_eq!(
            final_text("\ntranslated\n", Profile::Hymt).unwrap(),
            "translated"
        );
        assert!(final_text("thought</think>", Profile::Lfm2).is_err());
    }
    #[test]
    fn eos_counts_and_no_cache_are_required() {
        let mut m = Metadata::default();
        m.eog_ids.insert(106);
        let mut v = json!({"stop_type":"eos","truncated":false,"tokens_predicted":2,"tokens_evaluated":20,"timings":{"predicted_n":2,"prompt_n":20,"cache_n":0}});
        assert!(validate_completion(&v, 20, 100, &[42, 106], &m).is_ok());
        v["stop_type"] = json!("limit");
        assert!(validate_completion(&v, 20, 100, &[42, 106], &m).is_err());
        v["stop_type"] = json!("eos");
        v["timings"]["cache_n"] = json!(1);
        assert!(validate_completion(&v, 20, 100, &[42, 106], &m).is_err());
    }
    #[test]
    fn median_empty_is_unavailable() {
        assert_eq!(median(&mut []), None);
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn native_template_tokenization_adds_bos_once_without_eos() {
        let mut meta = Metadata {
            bos_id: Some(2),
            add_bos_token: Some(true),
            add_eos_token: Some(false),
            ..Metadata::default()
        };
        meta.eog_ids.insert(1);
        assert!(
            validate_prompt_special_tokens(&[105, 107, 42], &[2, 105, 107, 42], &meta).unwrap()
        );
        assert!(validate_prompt_special_tokens(&[105, 42], &[105, 42], &meta).is_err());
        assert!(validate_prompt_special_tokens(&[2, 105, 42], &[2, 2, 105, 42], &meta).is_err());
        assert!(validate_prompt_special_tokens(&[105, 42], &[2, 105, 42, 1], &meta).is_err());
        meta.add_bos_token = Some(false);
        assert!(!validate_prompt_special_tokens(&[2, 105, 42], &[2, 105, 42], &meta).unwrap());
        assert_eq!(tokenize_request("one\ntwo", true)["content"], "one\ntwo");
        assert_eq!(tokenize_request("text", true)["add_special"], true);
        assert_eq!(tokenize_request("text", true)["parse_special"], true);
        let no_bos = Metadata::default();
        assert!(!validate_prompt_special_tokens(&[10, 20], &[10, 20], &no_bos).unwrap());
    }

    #[test]
    fn gguf_metadata_reads_eos_and_end_of_turn_without_tensor_data() {
        fn string(bytes: &mut Vec<u8>, text: &str) {
            bytes.extend((text.len() as u64).to_le_bytes());
            bytes.extend(text.as_bytes());
        }
        let mut bytes = Vec::new();
        bytes.extend(0x46554747_u32.to_le_bytes());
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(999_u64.to_le_bytes()); // No tensor descriptors/data supplied.
        bytes.extend(7_u64.to_le_bytes());
        for (key, value) in [
            ("general.architecture", "gemma3"),
            ("tokenizer.chat_template", "test template"),
        ] {
            string(&mut bytes, key);
            bytes.extend(8_u32.to_le_bytes());
            string(&mut bytes, value);
        }
        string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend(9_u32.to_le_bytes());
        bytes.extend(8_u32.to_le_bytes());
        bytes.extend(3_u64.to_le_bytes());
        for token in ["<eos>", "<end_of_turn>", "</think>"] {
            string(&mut bytes, token);
        }
        string(&mut bytes, "tokenizer.ggml.eos_token_id");
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(0_u32.to_le_bytes());
        string(&mut bytes, "tokenizer.ggml.bos_token_id");
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(9_u32.to_le_bytes());
        for (key, value) in [
            ("tokenizer.ggml.add_bos_token", 1),
            ("tokenizer.ggml.add_eos_token", 0),
        ] {
            string(&mut bytes, key);
            bytes.extend(7_u32.to_le_bytes());
            bytes.push(value);
        }
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        let metadata = metadata(file.path()).unwrap();
        assert_eq!(metadata.architecture, "gemma3");
        assert_eq!(metadata.eog_ids, BTreeSet::from([0, 1]));
        assert_eq!(metadata.think_end_id, Some(2));
        assert_eq!(metadata.bos_id, Some(9));
        assert_eq!(metadata.add_bos_token, Some(true));
        assert_eq!(metadata.add_eos_token, Some(false));
    }

    #[test]
    fn full_table_and_source_survive_template_request_without_filtering() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all("source\ttarget_language\ttranslation\n词甲\ten\tTerm A\n词甲\tja\t用語A\n原文没有提及\tko\t용어B\n".as_bytes()).unwrap();
        let glossary = glossary::Glossary::load(file.path()).unwrap();
        let terms = term_reference(&glossary).unwrap();
        let case = case();
        let request = template_request(&case, Profile::Hymt, Some(&terms), InstructionLanguage::Zh);
        let body = request["messages"][0]["content"].as_str().unwrap();
        assert_eq!(body.matches(" 翻译成 ").count(), 3);
        assert!(body.contains("Term A") && body.contains("用語A") && body.contains("용어B"));
        assert!(body.ends_with(&case.text));
    }
}
