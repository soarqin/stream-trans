use std::{path::PathBuf, sync::mpsc, thread, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use futures_util::{SinkExt, StreamExt};
use tokio::{
    sync::{mpsc as async_mpsc, watch},
    time::{Instant, timeout},
};
use tokio_tungstenite::tungstenite::{Message, protocol::WebSocketConfig};
use url::Url;

use crate::{audio::Capture, output::OutputStore};

pub struct SessionOptions {
    pub api: String,
    pub device_id: String,
    pub target: Option<String>,
    pub wav: Option<PathBuf>,
}

pub struct Session {
    stop: watch::Sender<bool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Session {
    pub fn start(options: SessionOptions, store: OutputStore) -> Result<Self> {
        let url = stream_url(&options.api, options.target.as_deref())?;
        let (stop, receiver) = watch::channel(false);
        store.set_error(None);
        store.set_status("正在连接服务");
        store.set_mode(if options.target.is_some() {
            "translate"
        } else {
            "transcribe"
        });
        let thread = thread::Builder::new()
            .name("streamtrans-session".into())
            .spawn(move || {
                let result = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .context("无法启动网络线程")
                    .and_then(|runtime| {
                        runtime.block_on(run(options, url, store.clone(), receiver))
                    });
                if let Err(error) = result {
                    store.set_status("已停止：发生错误");
                    store.set_error(Some(format!("{error:#}")));
                }
            })
            .context("无法创建连接线程")?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }

    pub fn stop(&self) {
        let _ = self.stop.send(true);
    }
    pub fn is_finished(&self) -> bool {
        self.thread
            .as_ref()
            .is_none_or(|thread| thread.is_finished())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop();
        if self.is_finished()
            && let Some(thread) = self.thread.take()
        {
            let _ = thread.join();
        }
    }
}

pub fn stream_url(api: &str, target: Option<&str>) -> Result<Url> {
    let mut url = Url::parse(api.trim()).context("API 地址格式无效，请填写 http://主机:端口")?;
    let scheme = match url.scheme() {
        "http" | "ws" => "ws",
        "https" | "wss" => "wss",
        _ => bail!("API 地址仅支持 http、https、ws 或 wss"),
    };
    ensure!(url.host_str().is_some(), "API 地址缺少主机");
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "API 地址不能包含用户名或密码"
    );
    ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "请填写不含查询参数或片段的 API 基础地址"
    );
    url.set_scheme(scheme)
        .map_err(|_| anyhow::anyhow!("无法转换 WebSocket 地址"))?;
    let mut prefix = url.path().trim_end_matches('/').to_owned();
    for suffix in [
        "/v1/audio/translations/stream",
        "/v1/audio/transcriptions/stream",
        "/v1/audio/translations",
        "/v1/audio/transcriptions",
    ] {
        if let Some(base) = prefix.strip_suffix(suffix) {
            prefix = base.to_owned();
            break;
        }
    }
    let route = if target.is_some() {
        "translations"
    } else {
        "transcriptions"
    };
    url.set_path(&format!("{prefix}/v1/audio/{route}/stream"));
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("source_language", "auto");
        if let Some(target) = target {
            ensure!(!target.trim().is_empty(), "目标语言不能为空");
            query.append_pair("target_language", target);
        }
    }
    Ok(url)
}

async fn run(
    options: SessionOptions,
    url: Url,
    store: OutputStore,
    mut stop: watch::Receiver<bool>,
) -> Result<()> {
    let config = WebSocketConfig::default()
        .read_buffer_size(16 * 1024)
        .max_message_size(Some(2 * 1024 * 1024))
        .max_frame_size(Some(2 * 1024 * 1024));
    let connection = tokio::select! {
        result = timeout(Duration::from_secs(10), tokio_tungstenite::connect_async_with_config(url.as_str(), Some(config), true)) => {
            result.context("连接服务超时，请检查 API 地址与服务状态")?.context("无法连接服务")?
        }
        _ = stop.changed() => { store.set_status("已停止"); return Ok(()); }
    };
    let (mut socket, _) = connection;
    let ready = tokio::select! {
        result = timeout(Duration::from_secs(15), async {
            loop {
                match socket.next().await {
                    Some(Ok(Message::Text(text))) => {
                        let event: serde_json::Value = serde_json::from_str(&text).context("服务返回了无效 JSON")?;
                        if event["type"] == "error" { return Err(server_error(&event)); }
                        ensure!(event["type"] == "ready", "服务未返回 ready 事件");
                        return Ok(event);
                    }
                    Some(Ok(Message::Ping(payload))) => { socket.send(Message::Pong(payload)).await?; }
                    Some(Ok(Message::Close(_))) | None => bail!("服务在录音开始前关闭连接"),
                    Some(Err(error)) => return Err(error.into()),
                    _ => {}
                }
            }
        }) => result.context("等待服务就绪超时")??,
        _ = stop.changed() => { let _ = socket.close(None).await; store.set_status("已停止"); return Ok(()); }
    };
    store.apply_event(&ready)?;
    if *stop.borrow() {
        store.set_status("已停止");
        return Ok(());
    }

    let (sender, mut pcm) = async_mpsc::channel(64);
    let (errors_tx, errors_rx) = mpsc::channel();
    let mut capture = None;
    let mut replay = None;
    if let Some(path) = options.wav {
        let samples = read_wav(&path)?;
        replay = Some(tokio::spawn(async move {
            let base = Instant::now();
            for (index, chunk) in samples.chunks(1600).enumerate() {
                tokio::time::sleep_until(base + Duration::from_millis(index as u64 * 100)).await;
                let bytes = chunk
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect();
                if sender.send(bytes).await.is_err() {
                    break;
                }
            }
        }));
        store.set_status("正在回放音频，等待服务断句");
    } else {
        capture = Some(Capture::start(&options.device_id, sender, errors_tx)?);
        store.set_status("正在录音，等待服务断句");
    }
    let mut stopping = false;
    let mut finish_sent = false;
    let mut finish_deadline = None;
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    let result = async {
        loop {
            tokio::select! {
                changed = stop.changed(), if !stopping => {
                    if changed.is_err() || *stop.borrow() {
                        stopping = true;
                        store.set_status("已停止录音，正在处理剩余音频");
                        if let Some(capture) = capture.take() { capture.stop()?; }
                        if let Some(replay) = replay.take() { replay.abort(); }
                    }
                }
                bytes = pcm.recv(), if !finish_sent => {
                    if let Some(bytes) = bytes {
                        timeout(Duration::from_secs(5), socket.send(Message::Binary(bytes.into()))).await
                            .context("发送音频超时，已停止录音")?.context("发送音频失败")?;
                    } else {
                        if let Ok(error) = errors_rx.try_recv() { bail!("录音中断：{error}"); }
                        timeout(Duration::from_secs(5), socket.send(Message::Text(r#"{"type":"finish"}"#.into()))).await
                            .context("发送结束标记超时")??;
                        finish_sent = true;
                        finish_deadline = Some(Instant::now() + Duration::from_secs(330));
                        store.set_status("正在等待最后一段结果");
                    }
                }
                message = socket.next() => {
                    match message {
                        Some(Ok(Message::Text(text))) => {
                            let event: serde_json::Value = serde_json::from_str(&text).context("服务返回了无效 JSON")?;
                            if event["type"] == "error" { return Err(server_error(&event)); }
                            store.apply_event(&event)?;
                            if event["type"] == "done" {
                                if let Ok(error) = errors_rx.try_recv() { bail!("录音中断：{error}"); }
                                ensure!(finish_sent, "服务意外提前结束录音会话");
                                store.set_status("已完成");
                                return Ok(());
                            }
                        }
                        Some(Ok(Message::Ping(payload))) => {
                            timeout(Duration::from_secs(5), socket.send(Message::Pong(payload))).await.context("回应服务超时")??;
                        }
                        Some(Ok(Message::Close(_))) | None => bail!("连接在收到完整结果前关闭，请检查服务日志"),
                        Some(Err(error)) => return Err(anyhow::anyhow!(error).context("连接中断")),
                        _ => {}
                    }
                }
                _ = tick.tick() => {
                    if let Ok(error) = errors_rx.try_recv() { bail!("录音中断：{error}"); }
                    if finish_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                        bail!("等待最后一段结果超时，请检查服务状态");
                    }
                }
            }
        }
    }.await;
    if let Some(capture) = capture {
        let _ = capture.stop();
    }
    if let Some(replay) = replay {
        replay.abort();
    }
    let _ = timeout(Duration::from_secs(2), socket.close(None)).await;
    result
}

fn server_error(event: &serde_json::Value) -> anyhow::Error {
    let message = event["message"]
        .as_str()
        .or_else(|| event["error"].as_str())
        .unwrap_or("未知服务错误");
    anyhow::anyhow!("服务返回错误：{message}")
}

fn read_wav(path: &std::path::Path) -> Result<Vec<i16>> {
    ensure!(
        std::fs::metadata(path).context("无法读取回放音频")?.len() <= 32 * 1024 * 1024,
        "回放 WAV 不能超过 32 MiB"
    );
    let mut reader = hound::WavReader::open(path).context("无法打开回放 WAV")?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1
            && spec.sample_rate == 16_000
            && spec.bits_per_sample == 16
            && spec.sample_format == hound::SampleFormat::Int,
        "回放 WAV 必须为 16 kHz、单声道、16 位 PCM"
    );
    let samples = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(!samples.is_empty(), "回放音频为空");
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_and_proxy_prefix_are_preserved() {
        let url = stream_url("https://example.org/proxy/", Some("ja")).unwrap();
        assert_eq!(
            url.as_str(),
            "wss://example.org/proxy/v1/audio/translations/stream?source_language=auto&target_language=ja"
        );
        let url = stream_url("http://localhost:3000/v1/audio/translations/stream", None).unwrap();
        assert_eq!(
            url.as_str(),
            "ws://localhost:3000/v1/audio/transcriptions/stream?source_language=auto"
        );
    }

    #[test]
    fn malformed_or_credential_urls_are_rejected() {
        for api in [
            "localhost:3000",
            "file:///tmp",
            "http://user:secret@localhost",
            "http://localhost?target=ja",
            "http://localhost/#key",
        ] {
            assert!(stream_url(api, None).is_err(), "{api}");
        }
    }
}
