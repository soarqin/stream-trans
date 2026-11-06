use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderValue, header},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const MAX_ROWS: usize = 500;
const MAX_TEXT_BYTES: usize = 2 * 1024 * 1024;
const CSS_TEMPLATE: &str = include_str!("../assets/output.css");
const OUTPUT_HTML: &str = include_str!("../assets/output.html");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputRow {
    pub id: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub translation: Option<String>,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSnapshot {
    pub status: String,
    pub mode: String,
    pub show_original: bool,
    pub show_translation: bool,
    pub rows: Vec<OutputRow>,
    pub error: Option<String>,
}

impl Default for OutputSnapshot {
    fn default() -> Self {
        Self {
            status: "等待连接".to_owned(),
            mode: "translate".to_owned(),
            show_original: true,
            show_translation: true,
            rows: Vec::new(),
            error: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct OutputStore {
    inner: Arc<Mutex<OutputSnapshot>>,
}

impl OutputStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> OutputSnapshot {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn set_status(&self, status: impl Into<String>) {
        self.update(|state| state.status = short_message(status.into()));
    }

    pub fn set_error(&self, error: Option<String>) {
        self.update(|state| state.error = error.filter(|s| !s.is_empty()).map(short_message));
    }

    pub fn set_mode(&self, mode: impl Into<String>) {
        self.update(|state| state.mode = mode.into());
    }

    pub fn set_visibility(&self, show_original: bool, show_translation: bool) {
        self.update(|state| {
            state.show_original = show_original;
            state.show_translation = show_translation;
        });
    }

    pub fn clear(&self) {
        self.update(|state| {
            state.rows.clear();
            state.error = None;
        });
    }

    pub fn apply_event(&self, event: &Value) -> Result<()> {
        match event
            .get("type")
            .and_then(Value::as_str)
            .context("事件缺少 type")?
        {
            "ready" => {
                let mode = event
                    .get("mode")
                    .map(|value| value.as_str().context("输出模式必须为字符串"))
                    .transpose()?;
                ensure!(
                    mode.is_none_or(|m| matches!(m, "transcribe" | "translate")),
                    "输出模式无效"
                );
                self.update(|state| {
                    // Every server connection restarts segment IDs at 1.
                    state.rows.clear();
                    state.error = None;
                    state.status = "已连接".to_owned();
                    if let Some(mode) = mode {
                        state.mode = mode.to_owned();
                    }
                });
            }
            kind @ ("transcript" | "translation") => {
                let id = event
                    .get("segment_id")
                    .and_then(Value::as_u64)
                    .context("事件缺少 segment_id")?;
                let start_ms = event
                    .get("start_ms")
                    .and_then(Value::as_u64)
                    .context("事件缺少 start_ms")?;
                let end_ms = event
                    .get("end_ms")
                    .and_then(Value::as_u64)
                    .context("事件缺少 end_ms")?;
                ensure!(id > 0 && end_ms >= start_ms, "语音段 ID 或时间范围无效");
                let text = match event.get("text").and_then(Value::as_str) {
                    Some(text) => text.to_owned(),
                    None if kind == "translation" => String::new(),
                    None => anyhow::bail!("识别事件缺少 text"),
                };
                let translation = if kind == "translation" {
                    Some(
                        event
                            .get("translation")
                            .and_then(Value::as_str)
                            .context("翻译事件缺少 translation")?
                            .to_owned(),
                    )
                } else {
                    None
                };
                ensure!(
                    text.len() + translation.as_ref().map_or(0, String::len) <= MAX_TEXT_BYTES,
                    "单段结果超过 2 MiB 显示上限"
                );
                self.update(|state| {
                    if let Some(row) = state.rows.iter_mut().find(|row| row.id == id) {
                        let text_changed =
                            (kind == "transcript" || !text.is_empty()) && row.text != text;
                        let translation_changed =
                            translation.is_some() && row.translation != translation;
                        row.start_ms = start_ms;
                        row.end_ms = end_ms;
                        if kind == "transcript" || !text.is_empty() {
                            row.text = text;
                        }
                        if translation.is_some() {
                            row.translation = translation;
                        }
                        if text_changed || translation_changed {
                            row.updated_at_ms = unix_time_ms();
                        }
                    } else {
                        state.rows.push(OutputRow {
                            id,
                            start_ms,
                            end_ms,
                            text,
                            translation,
                            updated_at_ms: unix_time_ms(),
                        });
                        state.rows.sort_by_key(|row| row.id);
                    }
                    state.status = if kind == "transcript" {
                        "已收到原文"
                    } else {
                        "已收到译文"
                    }
                    .to_owned();
                });
            }
            "done" => self.set_status("已完成"),
            "error" => {
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .context("错误事件缺少 message")?;
                self.update(|state| {
                    state.status = "发生错误".to_owned();
                    state.error = Some(short_message(message.to_owned()));
                });
            }
            _ => anyhow::bail!("不支持的输出事件类型"),
        }
        Ok(())
    }

    fn update(&self, update: impl FnOnce(&mut OutputSnapshot)) {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        update(&mut state);
        if state.mode == "transcribe" {
            state.show_original = true;
            state.show_translation = false;
        }
        let mut bytes: usize = state.rows.iter().map(row_bytes).sum();
        while state.rows.len() > MAX_ROWS || bytes > MAX_TEXT_BYTES {
            bytes -= row_bytes(&state.rows.remove(0));
        }
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn row_bytes(row: &OutputRow) -> usize {
    row.text.len() + row.translation.as_ref().map_or(0, String::len)
}

fn short_message(mut text: String) -> String {
    if text.len() > 8192 {
        let mut end = 8192 - '…'.len_utf8();
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push('…');
    }
    text
}

pub fn render_text(snapshot: &OutputSnapshot) -> String {
    let original = snapshot.show_original || snapshot.mode == "transcribe";
    let translated = snapshot.show_translation && snapshot.mode != "transcribe";
    let mut blocks = Vec::new();
    for row in &snapshot.rows {
        let mut lines = Vec::new();
        if original && !row.text.is_empty() {
            lines.push(row.text.as_str());
        }
        if translated
            && let Some(text) = &row.translation
            && !text.is_empty()
        {
            lines.push(text.as_str());
        }
        if !lines.is_empty() {
            blocks.push(lines.join("\n"));
        }
    }
    if let Some(error) = &snapshot.error {
        blocks.push(format!("错误：{error}"));
    }
    blocks.join("\n\n")
}

#[derive(Clone)]
struct HttpState {
    store: OutputStore,
    css_path: PathBuf,
}

pub fn router(store: OutputStore, css_path: PathBuf) -> Router {
    Router::new()
        .route("/", get(page))
        .route("/api/state", get(snapshot))
        .route("/style.css", get(custom_css))
        .route("/style-template.css", get(template_css))
        .with_state(HttpState { store, css_path })
}

pub fn prepare_css(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).context("无法创建 CSS 模板目录")?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => file
            .write_all(CSS_TEMPLATE.as_bytes())
            .context("无法写入 CSS 模板"),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(path.is_file(), "CSS 路径不是文件：{}", path.display());
            Ok(())
        }
        Err(error) => Err(error).context("无法创建 CSS 模板文件"),
    }
}

async fn page() -> Response {
    no_store(Html(OUTPUT_HTML.replace("/*__DEFAULT_STYLE__*/", CSS_TEMPLATE)).into_response())
}

async fn snapshot(State(state): State<HttpState>) -> Response {
    no_store(Json(state.store.snapshot()).into_response())
}

async fn custom_css(State(state): State<HttpState>) -> Response {
    let css = tokio::fs::read_to_string(&state.css_path)
        .await
        .unwrap_or_else(|_| CSS_TEMPLATE.to_owned());
    css_response(css)
}

async fn template_css() -> Response {
    css_response(CSS_TEMPLATE.to_owned())
}

fn css_response(css: String) -> Response {
    no_store(([(header::CONTENT_TYPE, "text/css; charset=utf-8")], css).into_response())
}

fn no_store(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use serde_json::json;
    use tower::ServiceExt;

    fn event(id: u64, text: &str) -> Value {
        json!({"type":"transcript","segment_id":id,"start_ms":id*100,"end_ms":id*100+50,"text":text})
    }

    #[test]
    fn store_merges_segment_ids_and_resets_new_sessions_without_losing_settings() -> Result<()> {
        let store = OutputStore::new();
        store.apply_event(&event(1, "原文"))?;
        store.apply_event(&json!({"type":"translation","segment_id":1,"start_ms":100,"end_ms":150,"translation":"Translation"}))?;
        store.apply_event(&event(1, "修正原文"))?;
        assert_eq!(store.snapshot().rows.len(), 1);
        assert_eq!(render_text(&store.snapshot()), "修正原文\nTranslation");
        store.set_visibility(false, true);
        assert_eq!(render_text(&store.snapshot()), "Translation");
        store.set_visibility(true, false);
        assert_eq!(render_text(&store.snapshot()), "修正原文");
        store.set_visibility(false, false);
        assert_eq!(render_text(&store.snapshot()), "");
        store.set_error(Some("连接断开".to_owned()));
        assert!(render_text(&store.snapshot()).contains("错误：连接断开"));
        store.apply_event(&json!({"type":"ready","mode":"transcribe"}))?;
        let snapshot = store.snapshot();
        assert!(snapshot.rows.is_empty() && snapshot.error.is_none());
        assert!(snapshot.show_original && !snapshot.show_translation);
        store.apply_event(&event(1, "下一次录音"))?;
        assert_eq!(render_text(&store.snapshot()), "下一次录音");
        Ok(())
    }

    #[test]
    fn store_bounds_rows_and_text_bytes_and_rejects_invalid_updates() -> Result<()> {
        let store = OutputStore::new();
        for id in 1..=600 {
            store.apply_event(&event(id, "text"))?;
        }
        assert_eq!(store.snapshot().rows.len(), 500);
        assert_eq!(store.snapshot().rows[0].id, 101);
        store.clear();
        for id in 1..=450 {
            store.apply_event(&event(id, &"文".repeat(2000)))?;
        }
        assert!(store.snapshot().rows.iter().map(row_bytes).sum::<usize>() <= MAX_TEXT_BYTES);
        let count = store.snapshot().rows.len();
        assert!(
            store
                .apply_event(&json!({"type":"transcript","segment_id":1}))
                .is_err()
        );
        assert!(
            store
                .apply_event(&event(451, &"x".repeat(MAX_TEXT_BYTES + 1)))
                .is_err()
        );
        assert_eq!(store.snapshot().rows.len(), count);
        Ok(())
    }

    #[test]
    fn timestamps_change_only_with_result_content_and_keep_segment_order() -> Result<()> {
        let store = OutputStore::new();
        store.apply_event(&event(1, "第一句"))?;
        assert!(store.snapshot().rows[0].updated_at_ms > 0);
        store.inner.lock().unwrap().rows[0].updated_at_ms = 1;

        store.set_status("等待译文");
        store.set_visibility(false, true);
        store.apply_event(&event(1, "第一句"))?;
        assert_eq!(store.snapshot().rows[0].updated_at_ms, 1);

        store.apply_event(&event(1, "修正第一句"))?;
        assert!(store.snapshot().rows[0].updated_at_ms > 1);
        store.inner.lock().unwrap().rows[0].updated_at_ms = 1;
        let translation = json!({"type":"translation","segment_id":1,"start_ms":100,"end_ms":150,"translation":"First sentence"});
        store.apply_event(&translation)?;
        assert!(store.snapshot().rows[0].updated_at_ms > 1);
        store.inner.lock().unwrap().rows[0].updated_at_ms = 1;
        store.apply_event(&translation)?;
        assert_eq!(store.snapshot().rows[0].updated_at_ms, 1);

        store.apply_event(&event(2, "第二句"))?;
        store.inner.lock().unwrap().rows[1].updated_at_ms = 2;
        store.apply_event(&json!({"type":"translation","segment_id":1,"start_ms":100,"end_ms":150,"translation":"Revised first sentence"}))?;
        let snapshot = store.snapshot();
        assert_eq!(snapshot.rows.last().unwrap().id, 2);
        assert_eq!(snapshot.rows.last().unwrap().updated_at_ms, 2);
        assert!(snapshot.rows[0].updated_at_ms > 1);
        store.set_visibility(true, true);
        assert!(render_text(&store.snapshot()).contains("第二句"));
        assert!(render_text(&store.snapshot()).contains("Revised first sentence"));
        Ok(())
    }

    async fn get(app: &Router, uri: &str) -> (axum::http::HeaderMap, String) {
        let response = app
            .clone()
            .oneshot(Request::get(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let headers = response.headers().clone();
        let body = to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        (headers, String::from_utf8(body.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn http_is_read_only_no_store_and_renders_untrusted_text_as_data() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = OutputStore::new();
        let malicious = "<script>alert(1)</script><img src=x onerror=alert(1)>";
        store.apply_event(&event(1, malicious))?;
        let app = router(store, directory.path().join("missing.css"));
        let (headers, html) = get(&app, "/").await;
        assert!(
            headers[header::CONTENT_TYPE]
                .to_str()?
                .starts_with("text/html")
        );
        assert_eq!(headers[header::CACHE_CONTROL], "no-store");
        assert!(!html.contains(malicious));
        assert!(
            html.contains("textContent") && html.contains("<style>") && html.contains("<script>")
        );
        assert!(!html.contains("innerHTML") && !html.contains("/*__DEFAULT_STYLE__*/"));
        let (headers, json) = get(&app, "/api/state").await;
        assert!(
            headers[header::CONTENT_TYPE]
                .to_str()?
                .starts_with("application/json")
        );
        assert_eq!(headers[header::CACHE_CONTROL], "no-store");
        assert!(!headers.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));
        assert_eq!(
            serde_json::from_str::<Value>(&json)?["rows"][0]["text"],
            malicious
        );
        let response = app
            .oneshot(Request::post("/api/state").body(Body::empty())?)
            .await?;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        Ok(())
    }

    #[tokio::test]
    async fn css_template_is_not_overwritten_and_edits_and_missing_file_take_effect() -> Result<()>
    {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("styles/custom.css");
        prepare_css(&path)?;
        assert_eq!(std::fs::read_to_string(&path)?, CSS_TEMPLATE);
        std::fs::write(&path, ".translation { font-size: 42px; }")?;
        prepare_css(&path)?;
        let app = router(OutputStore::new(), path.clone());
        let (headers, custom) = get(&app, "/style.css").await;
        assert_eq!(custom, ".translation { font-size: 42px; }");
        assert!(
            headers[header::CONTENT_TYPE]
                .to_str()?
                .starts_with("text/css")
        );
        assert_eq!(headers[header::CACHE_CONTROL], "no-store");
        std::fs::write(&path, "body { font-size: 28px; }")?;
        assert_eq!(get(&app, "/style.css").await.1, "body { font-size: 28px; }");
        std::fs::remove_file(path)?;
        assert_eq!(get(&app, "/style.css").await.1, CSS_TEMPLATE);
        assert_eq!(get(&app, "/style-template.css").await.1, CSS_TEMPLATE);
        Ok(())
    }
}
