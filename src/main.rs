use starrocks_cost_prediction_service::{
    api::{AppState, build_router},
    config::AppConfig,
    ingest::tailer::FileTailer,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tracing_subscriber::EnvFilter;

struct NoDelayTcpListener(TcpListener);

impl axum::serve::Listener for NoDelayTcpListener {
    type Io = TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.0.accept().await {
                Ok((stream, addr)) => {
                    if let Err(err) = stream.set_nodelay(true) {
                        tracing::debug!(error = %err, "failed to set TCP_NODELAY on socket");
                    }
                    return (stream, addr);
                }
                Err(err) => {
                    tracing::warn!(error = %err, "TCP accept error; retrying");
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
        }
    }

    fn local_addr(&self) -> tokio::io::Result<Self::Addr> {
        self.0.local_addr()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::from_env();
    let bind_addr = config.bind_addr.clone();
    let retrain_interval_secs = config.retrain_interval_secs.max(1);
    let features_log_path = config.features_log_path.clone();

    let state = Arc::new(AppState::new(config)?);

    // Spawn background trainer task
    {
        let trainer_state = state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(retrain_interval_secs));
            interval.tick().await;
            loop {
                tokio::select! {
                    _ = trainer_state.retrain_notify.notified() => {},
                    _ = interval.tick() => {},
                }
                let engine = trainer_state.engine.clone();
                match tokio::task::spawn_blocking(move || engine.retrain_and_promote()).await {
                    Ok(Ok(report)) => {
                        if report.promoted {
                            tracing::info!(
                                model_version = report.model_version,
                                training_samples = report.training_samples,
                                pinball_p90 = report.holdout_metrics.pinball_p90,
                                "promoted challenger GBDT model set"
                            );
                        }
                    }
                    Ok(Err(err)) => {
                        tracing::warn!(error = %err, "background retrain failed");
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "background retrain task panicked");
                    }
                }
            }
        });
    }

    // Spawn optional fe.features.log file tailer task
    if let Some(log_path) = features_log_path {
        let tailer_state = state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(2));
            loop {
                ticker.tick().await;
                if !log_path.exists() {
                    continue;
                }
                match FileTailer::poll_once(&log_path, &tailer_state.store) {
                    Ok(samples) if !samples.is_empty() => {
                        let count = samples.len() as u64;
                        let mut should_retrain = false;
                        for sample in &samples {
                            if let Ok(trigger) = tailer_state.engine.observe_execution(sample)
                                && trigger
                            {
                                should_retrain = true;
                            }
                        }
                        tailer_state
                            .ingest_features_total
                            .fetch_add(count, Ordering::Relaxed);
                        if should_retrain {
                            tailer_state.retrain_notify.notify_one();
                        }
                    }
                    Ok(_) => {}
                    Err(err) => {
                        tracing::warn!(
                            path = %log_path.display(),
                            error = %err,
                            "file tailer poll error"
                        );
                    }
                }
            }
        });
    }

    let app = build_router(state.clone());
    let tcp_listener = TcpListener::bind(&bind_addr).await?;
    tracing::info!(
        bind_addr = %bind_addr,
        "StarRocks Cost Prediction Service listening with TCP_NODELAY enabled"
    );

    axum::serve(NoDelayTcpListener(tcp_listener), app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;

    Ok(())
}

async fn shutdown_signal(state: Arc<AppState>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("shutting down; persisting latest model snapshot to SQLite");
    if let Err(err) = state.engine.persist_snapshot() {
        tracing::warn!(error = %err, "failed to persist model snapshot on shutdown");
    }
}
