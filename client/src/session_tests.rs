//! Real loopback WebSocket tests. All audio comes from temporary synthetic WAVs.
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tempfile::NamedTempFile;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::{sleep, timeout},
};
use tokio_tungstenite::{
    WebSocketStream, accept_hdr_async,
    tungstenite::{
        Message,
        handshake::server::{Request, Response},
    },
};

use crate::{
    network::{Session, SessionOptions},
    output::{OutputSnapshot, OutputStore},
};

type Socket = WebSocketStream<TcpStream>;
const LIMIT: Duration = Duration::from_secs(5);

fn synthetic_wav(frames: usize) -> (NamedTempFile, Vec<u8>) {
    let file = NamedTempFile::new().unwrap();
    let mut writer = hound::WavWriter::create(
        file.path(),
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    let mut expected = Vec::with_capacity(frames * 2);
    for index in 0..frames {
        let sample = ((std::f64::consts::TAU * 1000.0 * index as f64 / 16_000.0).sin() * 20_000.0)
            .round() as i16;
        writer.write_sample(sample).unwrap();
        expected.extend_from_slice(&sample.to_le_bytes());
    }
    writer.finalize().unwrap();
    (file, expected)
}

async fn listener() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api = format!("http://{}", listener.local_addr().unwrap());
    (listener, api)
}

fn options(api: String, target: Option<&str>, wav: &NamedTempFile) -> SessionOptions {
    SessionOptions {
        api,
        request_key: None,
        target: target.map(str::to_owned),
        wav: Some(wav.path().to_path_buf()),
        // If the WAV branch accidentally opens a microphone, startup must fail.
        device_id: "test-must-never-open-a-microphone".into(),
    }
}

// Tungstenite fixes the handshake callback's error type to an unboxed HTTP response.
#[allow(clippy::result_large_err)]
async fn accept(listener: TcpListener, target: Option<&'static str>) -> Socket {
    accept_with_key(listener, target, None).await
}

// Tungstenite fixes the handshake callback's error type to an unboxed HTTP response.
#[allow(clippy::result_large_err)]
async fn accept_with_key(
    listener: TcpListener,
    target: Option<&'static str>,
    key: Option<&'static str>,
) -> Socket {
    let (stream, _) = timeout(LIMIT, listener.accept()).await.unwrap().unwrap();
    accept_hdr_async(stream, move |request: &Request, response: Response| {
        let authorization = request.headers().get("authorization");
        if let Some(key) = key {
            assert_eq!(authorization.unwrap(), format!("Bearer {key}").as_str());
            assert_eq!(request.headers().get_all("authorization").iter().count(), 1);
            assert!(!request.uri().to_string().contains(key));
        } else {
            assert!(authorization.is_none());
        }
        let url = url::Url::parse(&format!("ws://localhost{}", request.uri())).unwrap();
        let route = if target.is_some() {
            "translations"
        } else {
            "transcriptions"
        };
        assert_eq!(url.path(), format!("/v1/audio/{route}/stream"));
        let query: Vec<_> = url.query_pairs().collect();
        assert!(
            query
                .iter()
                .any(|(key, value)| key == "source_language" && value == "auto")
        );
        if let Some(target) = target {
            assert_eq!(query.len(), 2);
            assert!(
                query
                    .iter()
                    .any(|(key, value)| key == "target_language" && value == target)
            );
        } else {
            assert_eq!(query.len(), 1);
            assert!(!query.iter().any(|(key, _)| key == "target_language"));
        }
        Ok(response)
    })
    .await
    .unwrap()
}

async fn send(socket: &mut Socket, event: serde_json::Value) {
    socket
        .send(Message::Text(event.to_string().into()))
        .await
        .unwrap();
}

async fn receive(socket: &mut Socket) -> Message {
    timeout(LIMIT, socket.next())
        .await
        .expect("client message timeout")
        .expect("client disconnected")
        .expect("invalid client WebSocket message")
}

async fn snapshot_when(
    store: &OutputStore,
    condition: impl Fn(&OutputSnapshot) -> bool,
) -> OutputSnapshot {
    timeout(LIMIT, async {
        loop {
            let snapshot = store.snapshot();
            if condition(&snapshot) {
                return snapshot;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("client output was not updated")
}

async fn completed(session: &Session) {
    timeout(LIMIT, async {
        while !session.is_finished() {
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("session thread did not terminate");
}

async fn successful_session(target: Option<&'static str>, key: Option<&'static str>) {
    let (wav, expected) = synthetic_wav(3697);
    let (listener, api) = listener().await;
    let store = OutputStore::new();
    let observed_store = store.clone();
    let server = tokio::spawn(async move {
        let mut socket = accept_with_key(listener, target, key).await;
        send(
            &mut socket,
            json!({"type":"ready","mode":if target.is_some() {"translate"} else {"transcribe"}}),
        )
        .await;
        let mut pcm = Vec::new();
        let mut packets = 0;
        loop {
            match receive(&mut socket).await {
                Message::Binary(bytes) => {
                    assert!(!bytes.is_empty() && bytes.len() <= 3200 && bytes.len() % 2 == 0);
                    pcm.extend_from_slice(&bytes);
                    packets += 1;
                    if packets == 1 {
                        send(&mut socket, json!({"type":"transcript","segment_id":7,"start_ms":0,"end_ms":80,"text":"第一句测试","is_final":true})).await;
                        let snapshot =
                            snapshot_when(&observed_store, |snapshot| snapshot.rows.len() == 1)
                                .await;
                        assert_eq!(snapshot.rows[0].text, "第一句测试");
                        assert!(snapshot.rows[0].translation.is_none());
                    }
                    if packets == 2 && target.is_some() {
                        // Translation updates the existing row even when it does not repeat text.
                        send(&mut socket, json!({"type":"translation","segment_id":7,"start_ms":0,"end_ms":80,"translation":"最初のテストです。","target_language":"ja","is_final":true})).await;
                        let snapshot = snapshot_when(&observed_store, |snapshot| {
                            snapshot
                                .rows
                                .first()
                                .is_some_and(|row| row.translation.is_some())
                        })
                        .await;
                        assert_eq!(snapshot.rows.len(), 1);
                        assert_eq!(snapshot.rows[0].text, "第一句测试");
                    }
                }
                Message::Text(text) => {
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                        json!({"type":"finish"})
                    );
                    // All PCM, including the final short block, must precede finish.
                    assert_eq!(pcm, expected);
                    assert_eq!(packets, 3);
                    break;
                }
                message => panic!("unexpected message: {message:?}"),
            }
        }
        send(
            &mut socket,
            json!({"type":"done","duration_ms":231,"segments_count":1}),
        )
        .await;
        let _ = socket.close(None).await;
    });
    let mut options = options(api, target, &wav);
    options.request_key = key.map(str::to_owned);
    let session = Session::start(options, store.clone()).unwrap();
    timeout(LIMIT, server).await.unwrap().unwrap();
    completed(&session).await;
    let snapshot = store.snapshot();
    if let Some(key) = key {
        assert!(!serde_json::to_string(&snapshot).unwrap().contains(key));
    }
    assert_eq!(snapshot.status, "已完成");
    assert!(snapshot.error.is_none(), "{:?}", snapshot.error);
    assert_eq!(
        snapshot.mode,
        if target.is_some() {
            "translate"
        } else {
            "transcribe"
        }
    );
    assert_eq!(snapshot.rows.len(), 1);
    assert_eq!(snapshot.rows[0].id, 7);
    assert_eq!(snapshot.rows[0].start_ms, 0);
    assert_eq!(snapshot.rows[0].end_ms, 80);
    assert_eq!(snapshot.rows[0].text, "第一句测试");
    if target.is_some() {
        assert_eq!(
            snapshot.rows[0].translation.as_deref(),
            Some("最初のテストです。")
        );
    } else {
        assert!(snapshot.rows[0].translation.is_none());
        assert!(snapshot.show_original);
        assert!(!snapshot.show_translation);
    }
}

#[tokio::test]
async fn asr_only_replay_uses_transcription_route_without_target() {
    successful_session(None, None).await;
}

#[tokio::test]
async fn translation_replay_updates_one_row_and_finishes_after_all_pcm() {
    successful_session(Some("ja"), None).await;
}

#[tokio::test]
async fn request_key_authenticates_both_replay_modes_without_entering_output_state() {
    successful_session(None, Some("synthetic-client-key")).await;
    successful_session(Some("ja"), Some("synthetic-client-key")).await;
}

#[tokio::test]
async fn rejected_key_does_not_send_audio_or_echo_server_response_secrets() {
    const KEY: &str = "synthetic-rejected-client-key";
    let (wav, _) = synthetic_wav(16_000);
    let (listener, api) = listener().await;
    let store = OutputStore::new();
    let server = tokio::spawn(async move {
        let (mut stream, _) = timeout(LIMIT, listener.accept()).await.unwrap().unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            let mut byte = [0; 1];
            timeout(LIMIT, stream.read_exact(&mut byte))
                .await
                .unwrap()
                .unwrap();
            request.push(byte[0]);
            assert!(request.len() < 8192);
        }
        let request = String::from_utf8(request).unwrap();
        assert!(
            request
                .to_ascii_lowercase()
                .contains(&format!("authorization: bearer {KEY}"))
        );
        assert!(!request.lines().next().unwrap().contains(KEY));
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Length: {}\r\nX-Echo-Key: {KEY}\r\nConnection: close\r\n\r\n{KEY}",
            KEY.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        let mut bytes = [0; 64];
        let trailing = timeout(LIMIT, stream.read(&mut bytes)).await.unwrap();
        assert!(
            matches!(trailing, Ok(0))
                || trailing.as_ref().is_err_and(|error| {
                    matches!(
                        error.kind(),
                        std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted
                    )
                }),
            "unauthorized connection must close without sending audio"
        );
    });
    let mut options = options(api, None, &wav);
    options.request_key = Some(KEY.to_owned());
    let session = Session::start(options, store.clone()).unwrap();
    completed(&session).await;
    timeout(LIMIT, server).await.unwrap().unwrap();
    let snapshot = store.snapshot();
    let error = snapshot.error.as_deref().unwrap();
    assert!(error.contains("HTTP 401"));
    assert!(!error.contains(KEY));
    assert!(!serde_json::to_string(&snapshot).unwrap().contains(KEY));
    assert!(snapshot.rows.is_empty());
}

#[tokio::test]
async fn server_error_and_close_before_done_are_reported() {
    for return_error in [true, false] {
        let (wav, _) = synthetic_wav(16_000);
        let (listener, api) = listener().await;
        let store = OutputStore::new();
        let server = tokio::spawn(async move {
            let mut socket = accept(listener, None).await;
            send(&mut socket, json!({"type":"ready","mode":"transcribe"})).await;
            assert!(matches!(receive(&mut socket).await, Message::Binary(_)));
            if return_error {
                send(
                    &mut socket,
                    json!({"type":"error","message":"synthetic backend failure"}),
                )
                .await;
            }
            let _ = socket.close(None).await;
        });
        let session = Session::start(options(api, None, &wav), store.clone()).unwrap();
        timeout(LIMIT, server).await.unwrap().unwrap();
        completed(&session).await;
        let snapshot = store.snapshot();
        assert_eq!(snapshot.status, "已停止：发生错误");
        let error = snapshot.error.expect("failure must remain visible");
        assert!(
            error.contains(if return_error {
                "synthetic backend failure"
            } else {
                "完整结果前关闭"
            }),
            "{error}"
        );
    }
}

#[tokio::test]
async fn stopping_replay_sends_only_a_pcm_prefix_then_finish() {
    let (wav, expected) = synthetic_wav(32_000);
    let (listener, api) = listener().await;
    let store = OutputStore::new();
    let (received_tx, received_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let mut socket = accept(listener, None).await;
        send(&mut socket, json!({"type":"ready","mode":"transcribe"})).await;
        let mut pcm = Vec::new();
        let mut received_tx = Some(received_tx);
        loop {
            match receive(&mut socket).await {
                Message::Binary(bytes) => {
                    pcm.extend_from_slice(&bytes);
                    if pcm.len() >= 6400
                        && let Some(sender) = received_tx.take()
                    {
                        let _ = sender.send(());
                    }
                }
                Message::Text(text) => {
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                        json!({"type":"finish"})
                    );
                    assert!(pcm.len() >= 6400 && pcm.len() < expected.len());
                    assert_eq!(pcm, expected[..pcm.len()]);
                    break;
                }
                message => panic!("unexpected message: {message:?}"),
            }
        }
        send(
            &mut socket,
            json!({"type":"done","duration_ms":pcm.len()/32,"segments_count":0}),
        )
        .await;
        let _ = socket.close(None).await;
    });
    let session = Session::start(options(api, None, &wav), store.clone()).unwrap();
    timeout(LIMIT, received_rx).await.unwrap().unwrap();
    session.stop();
    timeout(LIMIT, server).await.unwrap().unwrap();
    completed(&session).await;
    assert!(store.snapshot().error.is_none());
    assert_eq!(store.snapshot().status, "已完成");
}
