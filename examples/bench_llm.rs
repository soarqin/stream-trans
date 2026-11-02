//! Real translations through the packaged server; startup warmup is timed separately.
#[allow(dead_code)]
#[path = "../src/glossary.rs"]
mod glossary;
#[allow(dead_code)]
#[path = "../src/llm.rs"]
mod llm;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use llm::{Llm, SamplingParams};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Debug, Parser)]
#[command(about = "打包版 llama.cpp 翻译基准；首轮和热态分别输出 JSON Lines")]
struct Args {
    #[arg(long)]
    server: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    #[arg(long,default_value_t=3,value_parser=clap::value_parser!(u32).range(3..=100))]
    warm_runs: u32,
    #[arg(long, default_value_t = 4096)]
    context_size: u32,
    #[arg(long, default_value_t = 2048)]
    max_tokens: u32,
    /// JSONL: {"id":"case-1","text":"完整原文","target":"en"}; checks等字段不参与推理。
    #[arg(long)]
    cases: Option<PathBuf>,
    #[arg(long)]
    glossary: Option<PathBuf>,
    #[arg(long, default_value_t = 1024)]
    glossary_max_tokens: u32,
    #[arg(long, default_value_t = 300)]
    startup_timeout_seconds: u64,
    #[arg(long, default_value_t = 600)]
    timeout_seconds: u64,
    #[arg(
        long,
        default_value = "你好，这是一个本地语音识别和翻译测试。今天下午三点开会。"
    )]
    text: String,
    #[arg(long, default_value = "English")]
    target: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
    #[arg(long, default_value_t = 0.6)]
    top_p: f64,
    /// 0 disables Top-K; the production default is 20.
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f64,
}
#[derive(Debug, Deserialize)]
struct Case {
    #[serde(alias = "case_id", alias = "caseid")]
    id: String,
    #[serde(alias = "input")]
    text: String,
    target: String,
}
fn read_cases(path: &Path) -> Result<Vec<Case>> {
    const MAX_BYTES: u64 = 4 * 1024 * 1024;
    let mut input = String::new();
    File::open(path)
        .with_context(|| format!("无法打开基准语料：{}", path.display()))?
        .take(MAX_BYTES + 1)
        .read_to_string(&mut input)
        .context("基准语料必须是 UTF-8 JSON Lines")?;
    ensure!(input.len() as u64 <= MAX_BYTES, "基准语料超过 4 MiB 上限");
    parse_cases(&input)
}
fn parse_cases(input: &str) -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    let mut ids = HashSet::new();
    for (line, text) in input.trim_start_matches('\u{feff}').lines().enumerate() {
        if text.trim().is_empty() {
            continue;
        }
        let case: Case = serde_json::from_str(text)
            .with_context(|| format!("基准语料第 {} 行 JSON 无效", line + 1))?;
        ensure!(
            !case.id.trim().is_empty()
                && !case.text.trim().is_empty()
                && !case.target.trim().is_empty(),
            "基准语料第 {} 行 id、text、target 不能为空",
            line + 1
        );
        ensure!(ids.insert(case.id.clone()), "基准语料 id 重复：{}", case.id);
        cases.push(case);
        ensure!(cases.len() <= 1000, "基准语料超过 1000 条上限");
    }
    ensure!(!cases.is_empty(), "基准语料没有有效样本");
    Ok(cases)
}
fn emit(value: Value) {
    println!("{value}");
}
fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}
fn main() -> Result<()> {
    // Keep JSONL on stdout and the verified CUDA loading receipt on stderr.
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_writer(std::io::stderr)
        .try_init();
    let args = Args::parse();
    ensure!(
        args.startup_timeout_seconds > 0 && args.timeout_seconds > 0,
        "超时秒数必须大于零"
    );
    let cases = if let Some(path) = &args.cases {
        read_cases(path)?
    } else {
        vec![Case {
            id: "single".to_owned(),
            text: args.text,
            target: args.target,
        }]
    };
    let sampling = SamplingParams {
        seed: args.seed,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: (args.top_k > 0).then_some(args.top_k),
        repetition_penalty: args.repetition_penalty,
    };
    let glossary_started = Instant::now();
    let glossary = args
        .glossary
        .as_deref()
        .map(glossary::Glossary::load)
        .transpose()?;
    if let Some(glossary) = &glossary {
        emit(json!({"event":"glossary_load","entries":glossary.len(),
            "file_load_ms":glossary_started.elapsed().as_secs_f64()*1000.0}));
    }
    emit(
        json!({"event":"environment","engine":"llama.cpp b11960 CUDA","architecture":"Hy-MT2-1.8B","server":args.server,"model":args.model,
        "gpu":args.gpu,"context_size":args.context_size,"max_tokens":args.max_tokens,"sampling":sampling,
        "prompt_template":"official Chinese instruction plus explicit sentence-by-sentence completeness requirement; no default system message",
        "repeat_last_n":"prompt_tokens + output_budget; all tokens in the current request","samplers":["penalties","temperature","top_k","top_p"],
        "glossary_entries":glossary.as_ref().map_or(0, glossary::Glossary::len),
        "kv_cache_dtype":"f16","flash_attention":"auto","startup_warmup":true,"cache_prompt":glossary.is_some(),
        "glossary_max_tokens":args.glossary_max_tokens,
        "cases_file":args.cases,"case_count":cases.len(),"glossary_applies_to_all_cases":glossary.is_some(),
        "first_request_definition":"first request for each case, in listed order after one model load and built-in warmup; only the first case is also the model's first request; OS file cache is not flushed",
        "token_definition":"actual sampled translation IDs; terminal EOS reported separately",
        "decode_rate_definition":"native server timing: (sampled tokens - first token) / decode time; includes terminal EOS"}),
    );
    let started = Instant::now();
    let mut model = Llm::load(
        &args.server,
        &args.model,
        args.gpu,
        args.context_size,
        args.max_tokens,
        Duration::from_secs(args.startup_timeout_seconds),
    )?;
    model.set_timeout(Duration::from_secs(args.timeout_seconds));
    emit(
        json!({"event":"load","cuda_full_offload_verified":true,"load_and_builtin_warmup_ms":started.elapsed().as_secs_f64()*1000.0}),
    );
    if let Some(glossary) = glossary {
        let prepared = Instant::now();
        let info = model.set_glossary(&glossary, args.glossary_max_tokens)?;
        emit(json!({"event":"glossary_prepare","info":info,
            "tokenize_and_verify_ms":prepared.elapsed().as_secs_f64()*1000.0}));
    }
    let mut failed = 0;
    let mut completed = 0;
    'cases: for (case_index, case) in cases.iter().enumerate() {
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        let mut rates = Vec::new();
        let mut totals = Vec::new();
        let mut first_translation = None;
        for run in 0..=args.warm_runs {
            let started = Instant::now();
            match model.translate_measured(&case.text, &case.target, sampling) {
                Ok(result) => {
                    let same = first_translation
                        .as_ref()
                        .map(|text| text == &result.translation);
                    if run == 0 {
                        first_translation = Some(result.translation.clone());
                    } else {
                        prefill.push(result.stats.prefill_ms);
                        decode.push(result.stats.decode_ms);
                        rates.push(result.stats.decode_tokens_per_second);
                        totals.push(result.stats.total_ms);
                    }
                    completed += 1;
                    emit(
                        json!({"event":"run","success":true,"case_id":case.id,"input":case.text,
                        "target":case.target,"phase":if run==0 {"first_request"} else {"warm"},"run":run,
                        "model_first_request":case_index==0 && run==0,
                        "same_translation_as_first":same,"translation":result.translation,"stats":result.stats}),
                    );
                }
                Err(error) => {
                    failed += 1;
                    emit(
                        json!({"event":"run","success":false,"case_id":case.id,"input":case.text,
                        "target":case.target,"phase":if run==0 {"first_request"} else {"warm"},"run":run,
                        "translation":null,"stats":null,"error":format!("{error:#}"),
                        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0}),
                    );
                    if !model.healthy() {
                        emit(
                            json!({"event":"aborted","reason":"LLM 子进程已停止，剩余样本未执行"}),
                        );
                        break 'cases;
                    }
                }
            }
        }
        emit(
            json!({"event":"warm_summary","case_id":case.id,"requested_warm_runs":args.warm_runs,
            "successful_warm_runs":totals.len(),"median_prefill_ms":median(&mut prefill),
            "median_decode_ms":median(&mut decode),"median_decode_tokens_per_second":median(&mut rates),
            "median_total_ms":median(&mut totals)}),
        );
    }
    let started = Instant::now();
    drop(model);
    emit(json!({"event":"teardown","kill_and_reap_ms":started.elapsed().as_secs_f64()*1000.0}));
    emit(
        json!({"event":"summary","successful_runs":completed,"failed_runs":failed,
        "requested_runs":cases.len()*(args.warm_runs as usize+1)}),
    );
    ensure!(
        failed == 0,
        "有 {failed} 次翻译失败；失败或截断输出未计入成功耗时"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_to_three_warm_runs_and_production_sampling() {
        let args = Args::try_parse_from([
            "bench_llm",
            "--server",
            "llama-server",
            "--model",
            "model.gguf",
        ])
        .unwrap();
        assert_eq!(args.warm_runs, 3);
        assert_eq!(args.top_k, 20);
        assert_eq!(args.temperature, 0.7);
        assert_eq!(args.top_p, 0.6);
        assert_eq!(args.repetition_penalty, 1.05);
        assert_eq!(args.context_size, 4096);
        assert_eq!(args.max_tokens, 2048);
        assert!(args.cases.is_none());
    }
    #[test]
    fn cases_accept_existing_schema_aliases_and_preserve_full_source() {
        let cases = parse_cases("\u{feff}{\"id\":\"a\",\"text\":\"  第一段。\\n第二段。  \",\"target\":\"ja\",\"checks\":[\"all_numbers\"],\"glossary\":true}\n{\"case_id\":\"b\",\"input\":\"hello\",\"target\":\"zh\"}\n").unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].text, "  第一段。\n第二段。  ");
        assert_eq!(cases[1].id, "b");
        assert!(parse_cases("").is_err());
        assert!(parse_cases("{\"id\":\"a\",\"text\":\" \",\"target\":\"en\"}").is_err());
        assert!(parse_cases("{\"id\":\"a\",\"text\":\"x\",\"target\":\"en\"}\n{\"id\":\"a\",\"text\":\"y\",\"target\":\"ja\"}").is_err());
    }
    #[test]
    fn no_successful_warm_runs_has_no_fake_zero_latency() {
        assert_eq!(median(&mut []), None);
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }
}
