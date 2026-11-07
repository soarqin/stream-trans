mod api;
mod asr;
#[cfg(windows)]
mod audio;
mod config;
mod convert_asr;
mod glossary;
mod llm;
mod models;
mod runtime;
mod stream;
mod vad;

use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use config::Config;
use runtime::Runtime;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "streamtrans=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let config = Config::parse();
    config.validate()?;
    let startup = async {
        Runtime::check_tools(&config).await?;
        let models = Runtime::download(&config).await?;
        if config.download_only {
            println!("{}", serde_json::to_string_pretty(&models)?);
            return Ok::<_, anyhow::Error>(None);
        }
        Runtime::start(&config, models).await.map(Some)
    };
    let Some(runtime) = (tokio::select! {
        result = startup => result?,
        _ = shutdown() => return Ok(()),
    }) else {
        return Ok(());
    };
    let runtime = Arc::new(runtime);
    let listener = match tokio::net::TcpListener::bind(config.bind).await {
        Ok(listener) => listener,
        Err(error) => {
            runtime.stop().await;
            return Err(error).context("HTTP 地址绑定失败");
        }
    };
    tracing::info!(address = %listener.local_addr()?, "模型已就绪：/v1/audio/transcriptions（仅识别）、/v1/audio/translations（识别与翻译）；流式接口在对应路径后加 /stream");
    let result = axum::serve(listener, api::router(runtime.clone(), config))
        .with_graceful_shutdown(shutdown())
        .await;
    runtime.stop().await;
    result?;
    Ok(())
}

async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}
