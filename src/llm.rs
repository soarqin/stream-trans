use anyhow::{Context, Result, anyhow, ensure};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    net::TcpListener,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};
#[cfg(not(windows))]
use std::{
    io::{BufRead, BufReader, Read},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread::JoinHandle,
};
#[cfg(windows)]
#[path = "embedded_llama.rs"]
mod embedded_llama;
#[cfg(windows)]
use embedded_llama::ServerProcess;

use crate::glossary::Glossary;

// Tencent Hy-MT2-1.8B's pinned tokenizer and chat_template.jinja. The ordinary
// hy_end_of_sentence token is NOT this model's generated end-of-turn token.
const EOS_TOKEN: u32 = 120020;
const BOS: &str = "<｜hy_begin▁of▁sentence｜>";
const USER: &str = "<｜hy_User｜>";
const ASSISTANT: &str = "<｜hy_Assistant｜>";
const SYSTEM_END: &str = "<｜hy_place▁holder▁no▁3｜>";
const SOURCE_RESERVE_TOKENS: usize = 512;

pub struct Llm {
    process: ServerProcess,
    client: Client,
    endpoint: String,
    api_key: String,
    context_size: usize,
    max_tokens: usize,
    timeout: Duration,
    glossary: Option<PreparedGlossary>,
}

struct PreparedGlossary {
    tokens: Vec<u32>,
    info: GlossaryInfo,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct GlossaryInfo {
    pub entries: usize,
    pub table_tokens: usize,
    pub prefix_tokens: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SamplingParams {
    pub seed: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub repetition_penalty: f64,
}
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            seed: 42,
            temperature: 0.7,
            top_p: 0.6,
            top_k: Some(20),
            repetition_penalty: 1.05,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TranslationStats {
    pub glossary_entries: usize,
    pub glossary_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub evaluated_prompt_tokens: usize,
    pub prompt_tokens: usize,
    /// Actual sampled translation IDs excluding the terminal EOS.
    pub generated_tokens: usize,
    pub eos_tokens: usize,
    pub sampled_tokens: usize,
    pub translation_tokens: usize,
    pub token_count_method: &'static str,
    pub tokenize_ms: f64,
    /// Native server timing; the first sampled token belongs to prefill.
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_steps: usize,
    /// Native convention: (sampled tokens - first token) / decode time; includes EOS.
    pub decode_tokens_per_second: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}
#[derive(Debug, Serialize)]
pub struct MeasuredTranslation {
    pub translation: String,
    pub stats: TranslationStats,
}
#[derive(Deserialize)]
struct Tokenized {
    tokens: Vec<u32>,
}
#[derive(Deserialize)]
struct Detokenized {
    content: String,
}
#[derive(Deserialize)]
struct Completion {
    tokens: Vec<u32>,
    tokens_predicted: usize,
    tokens_evaluated: usize,
    stop_type: String,
    truncated: bool,
    timings: Timings,
}
#[derive(Deserialize)]
struct Timings {
    cache_n: usize,
    prompt_n: usize,
    prompt_ms: f64,
    predicted_n: usize,
    predicted_ms: f64,
    predicted_per_second: f64,
}

impl Llm {
    pub fn load(
        server_path: &Path,
        model_path: &Path,
        gpu: usize,
        context_size: u32,
        max_tokens: u32,
        startup_timeout: Duration,
        cancelled: &AtomicBool,
    ) -> Result<Self> {
        ensure!(
            context_size > 1 && max_tokens > 0,
            "LLM 上下文和输出上限必须大于零"
        );
        ensure!(!startup_timeout.is_zero(), "LLM 启动超时必须大于零");
        let server_path = server_path
            .canonicalize()
            .context("找不到打包的 llama.cpp 服务入口，请运行安装脚本")?;
        let model_path = model_path
            .canonicalize()
            .context("找不到 Hy-MT2-1.8B GGUF 模型")?;
        ensure!(!cancelled.load(Ordering::Acquire), "翻译模型启动已取消");
        #[cfg(windows)]
        crate::native_library::configure_cuda(&server_path, gpu.try_into()?)?;
        #[cfg(not(windows))]
        let directory = server_path.parent().context("llama-server 路径无效")?;
        let listener = TcpListener::bind(("127.0.0.1", 0)).context("无法分配 LLM 本地端口")?;
        let port = listener.local_addr()?.port();
        let api_key = uuid::Uuid::new_v4().simple().to_string();
        drop(listener);
        #[cfg(windows)]
        let process = ServerProcess::spawn(
            &server_path,
            &model_path,
            gpu,
            port,
            context_size,
            api_key.clone(),
        )?;
        #[cfg(not(windows))]
        let process = {
            let mut command = Command::new(&server_path);
            command
                .current_dir(directory)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            // Pin CLI settings instead of inheriting llama-specific environment defaults.
            for (key, _) in std::env::vars_os() {
                let name = key.to_string_lossy().to_ascii_uppercase();
                if name.starts_with("LLAMA_ARG_") || name == "LLAMA_API_KEY" {
                    command.env_remove(key);
                }
            }
            let mut paths = vec![directory.to_owned()];
            paths.extend(std::env::split_paths(
                &std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default(),
            ));
            command.env("LD_LIBRARY_PATH", std::env::join_paths(paths)?);
            command.args([
                "--model",
                model_path.to_str().context("模型路径不是有效 UTF-8")?,
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--api-key",
                &api_key,
                "--alias",
                "Hy-MT2-1.8B",
                "--device",
                &format!("CUDA{gpu}"),
                "--split-mode",
                "none",
                "--n-gpu-layers",
                "all",
                "--fit",
                "off",
                "--ctx-size",
                &context_size.to_string(),
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
                "--threads-http",
                "2",
                "--no-webui",
                "--log-verbosity",
                // b11960 maps library INFO (including GPU offload receipts) to TRACE=4.
                "4",
                "--log-colors",
                "off",
            ]);

            ServerProcess::spawn(&mut command, gpu, api_key.clone())?
        };
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_millis(500))
            .build()
            .context("无法创建 LLM 本地 HTTP 客户端")?;
        let mut model = Self {
            process,
            client,
            endpoint: format!("http://127.0.0.1:{port}"),
            api_key,
            context_size: context_size as usize,
            max_tokens: max_tokens as usize,
            timeout: Duration::from_secs(300),
            glossary: None,
        };
        model.wait_ready(startup_timeout, cancelled)?;
        // Reject a wrong/legacy model rather than silently use its incompatible IDs.
        let special = model.tokenize(
            &format!("{BOS}{USER}{ASSISTANT}<｜hy_place▁holder▁no▁2｜>{SYSTEM_END}"),
            Instant::now() + Duration::from_secs(10),
        )?;
        ensure!(
            special.tokens == [120000, 120006, 120007, EOS_TOKEN, 120021],
            "模型分词器与固定 Hy-MT2 模板不一致，请使用指定的 Hy-MT2-1.8B-Q6_K.gguf"
        );
        tracing::info!(
            gpu,
            offloaded_layers = model
                .process
                .logs
                .lock()
                .ok()
                .and_then(|logs| logs.offloaded)
                .map(|counts| counts.0),
            total_layers = model
                .process
                .logs
                .lock()
                .ok()
                .and_then(|logs| logs.offloaded)
                .map(|counts| counts.1),
            "Hy-MT2-1.8B 已由打包的 llama.cpp b11960 全层加载到 CUDA；KV=f16，Flash Attention=auto"
        );
        Ok(model)
    }

    fn wait_ready(&mut self, timeout: Duration, cancelled: &AtomicBool) -> Result<()> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .context("LLM 启动超时设置过大")?;
        let mut ready_since = None;
        loop {
            ensure!(!cancelled.load(Ordering::Acquire), "翻译模型启动已取消");
            if !self.healthy() {
                return Err(anyhow!(
                    "llama-server 启动失败；{}",
                    self.process.diagnostics()
                ));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(anyhow!(
                    "llama-server 启动超时；{}",
                    self.process.diagnostics()
                ));
            }
            if let Ok(response) = self
                .client
                .get(format!("{}/props", self.endpoint))
                .bearer_auth(&self.api_key)
                .timeout(remaining.min(Duration::from_secs(1)))
                .send()
                && response.status().is_success()
            {
                let mut logs = self
                    .process
                    .logs
                    .lock()
                    .map_err(|_| anyhow!("LLM 启动日志状态异常"))?;
                if logs.gpu_confirmed() {
                    logs.startup_complete = true;
                    return Ok(());
                }
                let since = ready_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= Duration::from_secs(2) {
                    return Err(anyhow!(
                        "llama-server 已启动，但未确认 b11960 和指定 CUDA 设备的全层 GPU 加载；{}",
                        logs.diagnostics()
                    ));
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
    }
    pub fn healthy(&mut self) -> bool {
        self.process.running()
    }
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
    pub fn set_glossary(
        &mut self,
        glossary: &Glossary,
        maximum_tokens: u32,
    ) -> Result<GlossaryInfo> {
        let deadline = Instant::now()
            .checked_add(self.timeout)
            .context("术语表编码超时设置过大")?;
        let table = self.tokenize(glossary.as_tsv(), deadline)?;
        ensure!(
            table.tokens.len() <= maximum_tokens as usize,
            "完整术语表需要 {} 个 token，超过 --glossary-max-tokens {}；请缩小术语表或在上下文容量允许时调整上限，不会自动筛选或截断条目",
            table.tokens.len(),
            maximum_tokens
        );
        let prefix = glossary_prefix(glossary.as_tsv())?;
        let tokens = self.tokenize(&prefix, deadline)?.tokens;
        ensure!(
            tokens.len() + SOURCE_RESERVE_TOKENS + self.max_tokens <= self.context_size,
            "术语前缀需要 {} 个 token，加上 {} 个原文预留和 {} 个输出预算，超过上下文 {}；请缩小术语表或调整 --context-size / --max-tokens",
            tokens.len(),
            SOURCE_RESERVE_TOKENS,
            self.max_tokens,
            self.context_size
        );
        // Verify full ID equality at the fixed text boundary, including language
        // changes. Only the changing suffix is encoded after initialization.
        for target in ["en", "ja", "zh"] {
            let probe = glossary_suffix("术语缓存校验。", target);
            let suffix = self.tokenize(&probe, deadline)?.tokens;
            let complete = self.tokenize(&format!("{prefix}{probe}"), deadline)?.tokens;
            ensure!(
                tokens.iter().chain(&suffix).eq(complete.iter()),
                "术语前缀与后缀 token 边界不一致，不能安全复用缓存"
            );
        }
        let info = GlossaryInfo {
            entries: glossary.len(),
            table_tokens: table.tokens.len(),
            prefix_tokens: tokens.len(),
        };
        self.glossary = Some(PreparedGlossary { tokens, info });
        tracing::info!(
            entries = info.entries,
            table_tokens = info.table_tokens,
            prefix_tokens = info.prefix_tokens,
            prompt_cache = true,
            "完整术语前缀已编码；首次翻译建立 KV 缓存，后续请求复用"
        );
        Ok(info)
    }
    fn tokenize(&mut self, content: &str, deadline: Instant) -> Result<Tokenized> {
        self.post(
            "/tokenize",
            &json!({"content":content,"add_special":false,"parse_special":true}),
            deadline,
        )
    }
    pub fn translate(&mut self, text: &str, target: &str) -> Result<String> {
        let result = self.translate_measured(text, target, SamplingParams::default())?;
        tracing::info!(
            glossary_entries = result.stats.glossary_entries,
            glossary_tokens = result.stats.glossary_tokens,
            cached_prompt_tokens = result.stats.cached_prompt_tokens,
            evaluated_prompt_tokens = result.stats.evaluated_prompt_tokens,
            prompt_tokens = result.stats.prompt_tokens,
            translation_tokens = result.stats.translation_tokens,
            decode_tokens_per_second = result.stats.decode_tokens_per_second,
            total_ms = result.stats.total_ms,
            "Hy-MT2 翻译耗时与 token 统计"
        );
        Ok(result.translation)
    }
    pub fn translate_measured(
        &mut self,
        text: &str,
        target: &str,
        sampling: SamplingParams,
    ) -> Result<MeasuredTranslation> {
        validate_sampling(sampling)?;
        ensure!(
            !text.trim().is_empty() && !target.trim().is_empty(),
            "翻译文本和目标语言不能为空"
        );
        let started = Instant::now();
        let deadline = started
            .checked_add(self.timeout)
            .context("LLM 推理超时设置过大")?;
        let prompt_cache = self.glossary.is_some();
        let changing = if prompt_cache {
            glossary_suffix(text, target)
        } else {
            translation_prompt(text, target)
        };
        let suffix = self.tokenize(&changing, deadline)?;
        let (tokenized, info) = if let Some(glossary) = &self.glossary {
            let mut tokens = Vec::with_capacity(glossary.tokens.len() + suffix.tokens.len());
            tokens.extend_from_slice(&glossary.tokens);
            tokens.extend(suffix.tokens);
            (Tokenized { tokens }, glossary.info)
        } else {
            (
                suffix,
                GlossaryInfo {
                    entries: 0,
                    table_tokens: 0,
                    prefix_tokens: 0,
                },
            )
        };
        let budget = output_budget(tokenized.tokens.len(), self.context_size, self.max_tokens)?;
        let tokenize_ms = milliseconds(started.elapsed());
        let completion: Completion = self.post(
            "/completion",
            &completion_request(&tokenized.tokens, budget, sampling, prompt_cache),
            deadline,
        )?;
        validate_completion(&completion, tokenized.tokens.len(), budget, prompt_cache)?;
        let postprocess = Instant::now();
        let content_ids = &completion.tokens[..completion.tokens.len() - 1];
        let decoded: Detokenized =
            self.post("/detokenize", &json!({"tokens":content_ids}), deadline)?;
        let translation = extract_translation(&decoded.content)?;
        let stats = TranslationStats {
            glossary_entries: info.entries,
            glossary_tokens: info.table_tokens,
            cached_prompt_tokens: completion.timings.cache_n,
            evaluated_prompt_tokens: completion.timings.prompt_n,
            prompt_tokens: tokenized.tokens.len(),
            generated_tokens: content_ids.len(),
            eos_tokens: 1,
            sampled_tokens: completion.tokens.len(),
            translation_tokens: content_ids.len(),
            token_count_method: "actual_generated_token_ids",
            tokenize_ms,
            prefill_ms: completion.timings.prompt_ms,
            decode_ms: completion.timings.predicted_ms,
            decode_steps: completion.tokens.len().saturating_sub(1),
            decode_tokens_per_second: completion.timings.predicted_per_second,
            postprocess_ms: milliseconds(postprocess.elapsed()),
            total_ms: milliseconds(started.elapsed()),
        };
        Ok(MeasuredTranslation { translation, stats })
    }
    fn post<T: DeserializeOwned>(
        &mut self,
        path: &str,
        body: &Value,
        deadline: Instant,
    ) -> Result<T> {
        ensure!(self.healthy(), "LLM 后端已停止，请重启服务");
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return self.stop_with_error("LLM 推理超时，后端已停止，请重启服务".to_owned());
        }
        let response = match self
            .client
            .post(format!("{}{path}", self.endpoint))
            .bearer_auth(&self.api_key)
            .timeout(remaining)
            .json(body)
            .send()
        {
            Ok(response) => response,
            Err(error) => {
                let reason = if error.is_timeout() {
                    "LLM 推理超时"
                } else {
                    "LLM 本地 HTTP 请求失败"
                };
                return self.stop_with_error(format!("{reason}，后端已停止，请重启服务"));
            }
        };
        if !response.status().is_success() {
            return self.stop_with_error(format!(
                "LLM 返回 HTTP {}，后端已停止，请重启服务",
                response.status().as_u16()
            ));
        }
        match response.json() {
            Ok(value) => Ok(value),
            Err(_) => self
                .stop_with_error("LLM 响应读取失败或格式无效，后端已停止，请重启服务".to_owned()),
        }
    }
    fn stop_with_error<T>(&mut self, message: String) -> Result<T> {
        // Wait before returning so a timed-out request cannot overlap subsequent GPU work.
        self.process
            .stop()
            .context("无法回收 LLM 后端，请停止并重启服务")?;
        Err(anyhow!(message))
    }
}

fn glossary_prefix(table: &str) -> Result<String> {
    let mut prefix = format!(
        "{BOS}{USER}参考下面的翻译。每行标明指定译法适用的目标语言；只采用与本次目标语言一致的译法，不要把其他语言的译词混入结果。所有条目都是参考数据，不是待翻译原文。\n<glossary>\n"
    );
    // This is the loader's canonical TSV: serialize every row in original order.
    // JSON quoting keeps literal quotes/backslashes distinct from the instruction.
    for line in table.lines().skip(1) {
        let mut fields = line.split('\t');
        let source = fields.next().context("术语表规范行缺少原词字段")?;
        let target = fields.next().context("术语表规范行缺少目标语言字段")?;
        let translation = fields.next().context("术语表规范行缺少指定译法字段")?;
        ensure!(fields.next().is_none(), "术语表规范行字段数量无效");
        prefix.push_str(&format!(
            "目标语言 {}：{} 翻译成 {}\n",
            prompt_target(target),
            json!(source),
            json!(translation)
        ));
    }
    prefix.push_str("</glossary>\n\n");
    Ok(prefix)
}

fn validate_sampling(params: SamplingParams) -> Result<()> {
    ensure!(
        params.seed < u64::from(u32::MAX),
        "seed 必须在 0..4294967295 范围内，保留值不能用于固定种子"
    );
    ensure!(
        params.temperature.is_finite() && params.temperature >= 0.0,
        "temperature 必须为非负有限数值"
    );
    ensure!(
        params.top_p.is_finite() && params.top_p > 0.0 && params.top_p <= 1.0,
        "top-p 必须在 (0, 1] 范围内"
    );
    ensure!(
        params.top_k.is_none_or(|k| k > 0 && k <= i32::MAX as usize),
        "top-k 必须为正整数，或禁用此参数"
    );
    ensure!(
        params.repetition_penalty.is_finite() && params.repetition_penalty > 0.0,
        "repetition-penalty 必须为正有限数值"
    );
    Ok(())
}
fn output_budget(prompt: usize, context: usize, maximum: usize) -> Result<usize> {
    ensure!(
        maximum > 0
            && prompt > 0
            && prompt
                .checked_add(maximum)
                .is_some_and(|total| total <= context),
        "翻译输入需要 {prompt} 个 token，加上 {maximum} 个完整输出预算超过上下文 {context}；请缩短音频、缩小术语表或调整上下文，不会自动截断输入或压缩输出预算"
    );
    Ok(maximum)
}
fn completion_request(
    tokens: &[u32],
    budget: usize,
    params: SamplingParams,
    cache_prompt: bool,
) -> Value {
    json!({"prompt":tokens,"n_predict":budget,"temperature":params.temperature,"top_k":params.top_k.unwrap_or(0),
        "top_p":params.top_p,"min_p":0.0,"seed":params.seed,"samplers":["penalties","temperature","top_k","top_p"],
        // b11960's HTTP schema rejects CLI's -1 sentinel. This nonnegative
        // window covers every prompt/generated token in the current request.
        "repeat_penalty":params.repetition_penalty,"repeat_last_n":tokens.len()+budget,"presence_penalty":0.0,"frequency_penalty":0.0,"cache_prompt":cache_prompt,"id_slot":0,
        "return_tokens":true,"stream":false,"ignore_eos":false,"stop":[],"n_probs":0})
}
fn validate_completion(
    value: &Completion,
    prompt: usize,
    budget: usize,
    cache_prompt: bool,
) -> Result<()> {
    ensure!(
        !value.truncated && value.stop_type == "eos",
        "LLM 未完整结束：stop_type={}，sampled_tokens={}，budget={}，truncated={}；译文可能被截断，请增加 --max-tokens、--context-size 或缩短语音段",
        value.stop_type,
        value.tokens_predicted,
        budget,
        value.truncated
    );
    ensure!(
        !value.tokens.is_empty() && value.tokens.len() <= budget,
        "LLM 返回的 token 数量无效"
    );
    ensure!(
        value.tokens_predicted == value.tokens.len()
            && value.timings.predicted_n == value.tokens.len(),
        "LLM 输出 token 统计不一致"
    );
    ensure!(
        value.tokens_evaluated == prompt
            && value.timings.prompt_n > 0
            && value.timings.prompt_n.checked_add(value.timings.cache_n) == Some(prompt)
            && (cache_prompt || value.timings.cache_n == 0),
        "LLM 输入 token 统计或缓存状态不一致"
    );
    ensure!(
        value.tokens.last() == Some(&EOS_TOKEN)
            && !value.tokens[..value.tokens.len() - 1].contains(&EOS_TOKEN),
        "LLM 未返回有效的终止 token，译文可能不完整"
    );
    ensure!(
        [
            value.timings.prompt_ms,
            value.timings.predicted_ms,
            value.timings.predicted_per_second
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0),
        "LLM 返回的计时数据无效"
    );
    Ok(())
}
fn translation_prompt(text: &str, target: &str) -> String {
    // Official Hy-MT2 template: no default system message, no thinking prefill.
    // Use the fixed Chinese instruction, as all current tested directions involve
    // Chinese. The model card also provides an English template for other pairs.
    format!("{BOS}{}", translation_suffix(text, target))
}
fn translation_suffix(text: &str, target: &str) -> String {
    let target = prompt_target(target);
    format!("{USER}{}", translation_instruction(text, &target))
}
fn glossary_suffix(text: &str, target: &str) -> String {
    let target = prompt_target(target);
    format!(
        "本次目标语言： {target}。仅应用该目标语言的指定译法，其他语言的译法不适用。\n{}",
        translation_instruction(text, &target)
    )
}
fn translation_instruction(text: &str, target: &str) -> String {
    format!(
        "将以下文本翻译为 {target}，注意只需要输出翻译后的结果，不要额外解释。请逐句完整翻译，保留所有数字、单位、条件和否定关系，不要概括或省略：\n\n{text}{ASSISTANT}"
    )
}
fn prompt_target(target: &str) -> String {
    // The API retains its existing names; the fixed Chinese instruction uses
    // full Chinese language names as recommended by the official model card.
    let normalized = crate::glossary::normalize_target(target);
    match normalized.as_str() {
        "English" => "英语".to_owned(),
        "简体中文" => "简体中文".to_owned(),
        "繁體中文" => "繁体中文".to_owned(),
        "日本語" => "日语".to_owned(),
        "한국어" => "韩语".to_owned(),
        "French" => "法语".to_owned(),
        "German" => "德语".to_owned(),
        "Spanish" => "西班牙语".to_owned(),
        "Russian" => "俄语".to_owned(),
        "Portuguese" => "葡萄牙语".to_owned(),
        "Arabic" => "阿拉伯语".to_owned(),
        "Italian" => "意大利语".to_owned(),
        _ => normalized,
    }
}
fn extract_translation(response: &str) -> Result<String> {
    let translation = response.trim();
    ensure!(!translation.is_empty(), "LLM 未返回译文");
    ensure!(
        !translation.contains("<｜hy_"),
        "LLM 译文包含未正确分离的控制标记"
    );
    Ok(translation.to_owned())
}
fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[derive(Default)]
struct StartupLogs {
    gpu: usize,
    api_key: String,
    build_confirmed: bool,
    cuda_confirmed: bool,
    offloaded: Option<(usize, usize)>,
    startup_complete: bool,
    tail: VecDeque<String>,
}
impl StartupLogs {
    fn record(&mut self, line: &str) {
        if self.startup_complete {
            return;
        }
        self.build_confirmed |= line.contains("build 11960 (");
        self.cuda_confirmed |= line.contains(&format!("using device CUDA{} ", self.gpu));
        if let Some(counts) = line
            .split("offloaded ")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            && let Some((loaded, total)) = counts.split_once('/')
            && let (Ok(loaded), Ok(total)) = (loaded.parse(), total.parse())
        {
            self.offloaded = Some((loaded, total));
        }
        let lower = line.to_ascii_lowercase();
        if ![
            "error",
            "failed",
            "cuda",
            "offload",
            "cannot",
            "unable",
            "exception",
            "abort",
            "build 11960",
        ]
        .iter()
        .any(|word| lower.contains(word))
        {
            return;
        }
        if [
            "prompt",
            "content",
            "transcription",
            "<think>",
            "</think>",
            "api-key",
            "api_key",
            "authorization",
        ]
        .iter()
        .any(|word| lower.contains(word))
        {
            return;
        }
        let safe = line
            .replace(&self.api_key, "[redacted]")
            .chars()
            .filter(|c| !c.is_control())
            .take(280)
            .collect();
        if self.tail.len() == 16 {
            self.tail.pop_front();
        }
        self.tail.push_back(safe);
    }
    fn gpu_confirmed(&self) -> bool {
        self.build_confirmed
            && self.cuda_confirmed
            && self.offloaded.is_some_and(|(n, total)| n > 0 && n == total)
    }
    fn diagnostics(&self) -> String {
        if self.tail.is_empty() {
            "没有可用的启动诊断，请检查打包文件和 NVIDIA 驱动".to_owned()
        } else {
            self.tail.iter().cloned().collect::<Vec<_>>().join(" | ")
        }
    }
}
#[cfg(not(windows))]
struct ServerProcess {
    child: Option<Child>,
    readers: Vec<JoinHandle<()>>,
    logs: Arc<Mutex<StartupLogs>>,
}
#[cfg(not(windows))]
impl ServerProcess {
    fn spawn(command: &mut Command, gpu: usize, api_key: String) -> Result<Self> {
        let child = command.spawn().context(
            "无法启动打包的 llama-server，请检查可执行文件、后端依赖库及共享 runtime 目录",
        )?;
        let mut process = Self {
            child: Some(child),
            readers: Vec::new(),
            logs: Arc::new(Mutex::new(StartupLogs {
                gpu,
                api_key,
                ..Default::default()
            })),
        };
        let child = process.child.as_mut().unwrap();
        let stdout = child.stdout.take().context("无法读取 llama-server 输出")?;
        let stderr = child.stderr.take().context("无法读取 llama-server 诊断")?;
        process
            .readers
            .push(Self::reader(stdout, process.logs.clone())?);
        process
            .readers
            .push(Self::reader(stderr, process.logs.clone())?);
        Ok(process)
    }
    fn reader<R: Read + Send + 'static>(
        pipe: R,
        logs: Arc<Mutex<StartupLogs>>,
    ) -> Result<JoinHandle<()>> {
        thread::Builder::new()
            .name("llama-log-drain".to_owned())
            .spawn(move || {
                let mut reader = BufReader::new(pipe);
                let mut bytes = Vec::new();
                loop {
                    bytes.clear();
                    match reader.read_until(b'\n', &mut bytes) {
                        Ok(0) => break,
                        Ok(_) => {
                            // Native dependencies may emit local-codepage bytes on Windows.
                            if let Ok(mut logs) = logs.lock() {
                                logs.record(&String::from_utf8_lossy(&bytes));
                            }
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
            })
            .context("无法启动 llama-server 日志读取线程")
    }
    fn running(&mut self) -> bool {
        match self.child.as_mut().map(Child::try_wait) {
            Some(Ok(None)) => true,
            _ => {
                let _ = self.stop();
                false
            }
        }
    }
    fn stop(&mut self) -> Result<()> {
        let waited = if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            child
                .wait()
                .map(|_| ())
                .context("无法等待 llama-server 后端退出")
        } else {
            Ok(())
        };
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
        waited
    }
    fn diagnostics(&self) -> String {
        self.logs
            .lock()
            .map(|logs| logs.diagnostics())
            .unwrap_or_else(|_| "启动诊断状态异常".to_owned())
    }
}
#[cfg(not(windows))]
impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_glossary_prefix_is_independent_of_source_and_target() {
        let table = "source\ttarget_language\ttranslation\n本地\tEnglish\ton-device\n语音识别\t日本語\t音声認識\n未提及\tFrench\tnon mentionné\n";
        let prefix = glossary_prefix(table).unwrap();
        assert!(prefix.starts_with(&format!("{BOS}{USER}")));
        assert!(prefix.ends_with("</glossary>\n\n"));
        assert!(prefix.contains("目标语言 英语：\"本地\" 翻译成 \"on-device\"\n"));
        assert!(prefix.contains("目标语言 日语：\"语音识别\" 翻译成 \"音声認識\"\n"));
        assert!(prefix.contains("目标语言 法语：\"未提及\" 翻译成 \"non mentionné\"\n"));
        assert_eq!(prefix.matches(" 翻译成 ").count(), 3);
        for target in ["en", "ja", "zh"] {
            let suffix = glossary_suffix("你好。", target);
            assert!(!suffix.contains(table));
            assert!(suffix.starts_with("本次目标语言： "));
            assert!(!suffix.contains(USER));
            let prompt = format!("{prefix}{suffix}");
            assert!(prompt.contains(&format!("\n\n你好。{ASSISTANT}")));
            assert!(prompt.contains("未提及"));
            assert_eq!(prompt.matches(BOS).count(), 1);
            assert_eq!(prompt.matches(USER).count(), 1);
            assert_eq!(prompt.matches(ASSISTANT).count(), 1);
            assert!(!prompt.contains(SYSTEM_END));
        }
    }

    #[test]
    fn complete_glossary_json_quotes_every_row_without_silent_discard() {
        let prefix = glossary_prefix(
            "source\ttarget_language\ttranslation\n含\"引号\\路径\tEnglish\tquoted \"term\"\n",
        )
        .unwrap();
        assert!(prefix.contains(r#"目标语言 英语："含\"引号\\路径" 翻译成 "quoted \"term\"""#));
        assert!(glossary_prefix("source\ttarget_language\ttranslation\nx\ten\n").is_err());
        assert!(
            glossary_prefix("source\ttarget_language\ttranslation\nx\ten\ty\textra\n").is_err()
        );
    }

    #[test]
    fn full_glossary_never_shrinks_output_or_silently_discards_entries() {
        assert_eq!(output_budget(2048, 4096, 2048).unwrap(), 2048);
        assert!(output_budget(2049, 4096, 2048).is_err());
        assert!(output_budget(usize::MAX, 4096, 2048).is_err());
        assert!(output_budget(0, 4096, 2048).is_err());
    }

    #[test]
    fn prompt_preserves_input_and_exact_template_prefixes() {
        let source = "  你好。\n下午三点开会。\n ";
        let prompt = translation_prompt(source, "ja");
        assert_eq!(
            prompt,
            format!(
                "{BOS}{USER}将以下文本翻译为 日语，注意只需要输出翻译后的结果，不要额外解释。请逐句完整翻译，保留所有数字、单位、条件和否定关系，不要概括或省略：\n\n{source}{ASSISTANT}"
            )
        );
        assert!(!prompt.contains("<think>"));
        assert!(!prompt.contains(SYSTEM_END));
        assert_eq!(prompt_target("zh-CN"), "简体中文");
        assert_eq!(prompt_target("zh-TW"), "繁体中文");
        assert_eq!(prompt_target("한국어"), "韩语");
        assert_eq!(prompt_target("Welsh"), "Welsh");
    }
    #[test]
    fn request_pins_sampling_cache_and_exact_prompt_ids() {
        let request = completion_request(
            &[120000, 120006, 8, 220],
            64,
            SamplingParams::default(),
            false,
        );
        assert_eq!(request["prompt"], json!([120000, 120006, 8, 220]));
        assert_eq!(
            request["samplers"],
            json!(["penalties", "temperature", "top_k", "top_p"])
        );
        assert_eq!(request["top_k"], 20);
        assert_eq!(request["top_p"], 0.6);
        assert_eq!(request["temperature"], 0.7);
        assert_eq!(request["repeat_penalty"], 1.05);
        assert_eq!(request["repeat_last_n"], 68);
        assert_eq!(request["min_p"], 0.0);
        assert_eq!(request["cache_prompt"], false);
        assert_eq!(request["return_tokens"], true);
        assert_eq!(request["ignore_eos"], false);
        assert_eq!(request["n_predict"], 64);
        let cached = completion_request(&[120000, 42, 120021], 64, SamplingParams::default(), true);
        assert_eq!(cached["cache_prompt"], true);
        assert_eq!(cached["id_slot"], 0);
        assert_eq!(cached["prompt"], json!([120000, 42, 120021]));
        assert_eq!(
            completion_request(
                &[0],
                1,
                SamplingParams {
                    top_k: None,
                    ..SamplingParams::default()
                },
                false
            )["top_k"],
            0
        );
    }
    #[test]
    fn budget_and_sampling_reject_invalid_boundaries() {
        assert_eq!(output_budget(99, 4096, 2048).unwrap(), 2048);
        assert!(output_budget(4090, 4096, 2048).is_err());
        assert!(output_budget(4096, 4096, 1).is_err());
        assert!(output_budget(0, 4096, 1).is_err());
        assert!(output_budget(99, 4096, 0).is_err());
        assert!(
            validate_sampling(SamplingParams {
                seed: u64::from(u32::MAX),
                ..SamplingParams::default()
            })
            .is_err()
        );
        assert!(
            validate_sampling(SamplingParams {
                repetition_penalty: 0.0,
                ..SamplingParams::default()
            })
            .is_err()
        );
        assert!(
            validate_sampling(SamplingParams {
                temperature: f64::NAN,
                ..SamplingParams::default()
            })
            .is_err()
        );
    }
    fn completion() -> Completion {
        serde_json::from_value(json!({"tokens":[1741,9,220,42,120020],"tokens_predicted":5,"tokens_evaluated":99,"stop_type":"eos","truncated":false,"timings":{"cache_n":0,"prompt_n":99,"prompt_ms":40.0,"predicted_n":5,"predicted_ms":50.0,"predicted_per_second":80.0}})).unwrap()
    }
    #[test]
    fn completion_requires_eos_and_consistent_actual_counts() {
        assert!(validate_completion(&completion(), 99, 2048, false).is_ok());
        let mut direct = completion();
        direct.tokens = vec![42, EOS_TOKEN];
        direct.tokens_predicted = 2;
        direct.timings.predicted_n = 2;
        assert!(validate_completion(&direct, 99, 2048, false).is_ok());
        direct.truncated = true;
        assert!(validate_completion(&direct, 99, 2048, false).is_err());
        let mut value = completion();
        value.stop_type = "limit".to_owned();
        assert!(validate_completion(&value, 99, 2048, false).is_err());
        let mut value = completion();
        value.tokens.pop();
        value.tokens_predicted = 4;
        value.timings.predicted_n = 4;
        assert!(validate_completion(&value, 99, 2048, false).is_err());
        let mut value = completion();
        value.timings.cache_n = 20;
        assert!(validate_completion(&value, 99, 2048, false).is_err());
        value.timings.prompt_n = 79;
        assert!(validate_completion(&value, 99, 2048, true).is_ok());
        assert!(validate_completion(&value, 99, 2048, false).is_err());
        value.timings.prompt_n = 0;
        value.timings.cache_n = 99;
        assert!(validate_completion(&value, 99, 2048, true).is_err());
        assert!(validate_completion(&completion(), 99, 4, false).is_err());
        for wrong_eos in [1, 130073, 120001, 120008] {
            let mut value = completion();
            *value.tokens.last_mut().unwrap() = wrong_eos;
            assert!(validate_completion(&value, 99, 2048, false).is_err());
        }
        let mut value = completion();
        value.tokens[0] = EOS_TOKEN;
        assert!(validate_completion(&value, 99, 2048, false).is_err());
    }
    #[test]
    fn direct_translation_preserves_content_and_rejects_native_control_markers() {
        assert_eq!(
            extract_translation("\nHello, world.\n").unwrap(),
            "Hello, world."
        );
        assert_eq!(
            extract_translation("意味、名前、123。\n第二段。").unwrap(),
            "意味、名前、123。\n第二段。"
        );
        for invalid in [
            "",
            " \n",
            "text<｜hy_User｜>",
            "text<｜hy_place▁holder▁no▁2｜>",
        ] {
            assert!(extract_translation(invalid).is_err());
        }
    }
    #[test]
    fn gpu_receipt_requires_build_device_and_all_layers() {
        let mut logs = StartupLogs {
            gpu: 0,
            api_key: "secret-key".to_owned(),
            ..Default::default()
        };
        logs.record("build 11960 (8642541b2) with Clang");
        logs.record("using device CUDA0 (NVIDIA)");
        logs.record("offloaded 42/43 layers to GPU");
        assert!(!logs.gpu_confirmed());
        logs.record("offloaded 43/43 layers to GPU");
        assert!(logs.gpu_confirmed());
        logs.record("error: prompt secret thinking");
        logs.record("error: secret-key");
        assert!(!logs.diagnostics().contains("secret-key"));
        assert!(!logs.diagnostics().contains("secret thinking"));
        logs.startup_complete = true;
        logs.record("error: private request text");
        assert!(!logs.diagnostics().contains("private request text"));
    }

    #[cfg(not(windows))]
    #[test]
    fn invalid_utf8_does_not_discard_later_gpu_receipts() {
        let logs = Arc::new(Mutex::new(StartupLogs {
            gpu: 0,
            api_key: "test-key".to_owned(),
            ..Default::default()
        }));
        let bytes = b"\xff\xfe local-codepage startup text\nbuild 11960 (8642541b2) with Clang\nusing device CUDA0 (NVIDIA)\noffloaded 43/43 layers to GPU\n".to_vec();
        ServerProcess::reader(std::io::Cursor::new(bytes), logs.clone())
            .unwrap()
            .join()
            .unwrap();
        assert!(logs.lock().unwrap().gpu_confirmed());
    }
}
