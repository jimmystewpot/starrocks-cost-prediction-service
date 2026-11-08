use crate::api::{AppState, build_router};
use crate::ingest::tailer::{FileTailer, MAX_LINES_PER_POLL};
use crate::model::ensemble::RetrainReport;
use crate::storage::sqlite::StorageError;
use std::future::Future;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};

pub struct NoDelayTcpListener(pub TcpListener);

impl axum::serve::Listener for NoDelayTcpListener {
    type Io = TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.0.accept().await {
                Ok((stream, addr)) => {
                    let _ = stream.set_nodelay(true);
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

pub async fn run_retrain_iteration(state: &Arc<AppState>) -> Result<RetrainReport, StorageError> {
    let engine = state.engine.clone();
    let report = tokio::task::spawn_blocking(move || engine.retrain_and_promote())
        .await
        .map_err(|err| StorageError::Io(std::io::Error::other(err)))??;
    if report.promoted {
        tracing::info!(
            model_version = report.model_version,
            training_samples = report.training_samples,
            pinball_p90 = report.holdout_metrics.pinball_p90,
            "promoted challenger GBDT model set"
        );
    }
    Ok(report)
}

pub async fn run_tailer_iteration(log_path: &Path, state: &Arc<AppState>) -> std::io::Result<u64> {
    let poll_path = log_path.to_path_buf();
    let poll_state = state.clone();
    tokio::task::spawn_blocking(move || {
        if !poll_path.exists() {
            return Ok(0);
        }
        let mut ingested = 0_u64;
        if let Some(batch) =
            FileTailer::read_batch_with_limit(&poll_path, &poll_state.store, MAX_LINES_PER_POLL)?
        {
            if !batch.samples.is_empty() {
                ingested = batch.samples.len() as u64;
                let should_retrain = poll_state
                    .engine
                    .observe_executions_batch(&batch.samples)
                    .map_err(std::io::Error::other)?;
                poll_state
                    .ingest_features_total
                    .fetch_add(ingested, Ordering::Relaxed);
                if should_retrain {
                    poll_state.retrain_notify.notify_one();
                }
            }
            FileTailer::commit_batch(&poll_path, &poll_state.store, &batch)?;
        }
        Ok(ingested)
    })
    .await
    .map_err(std::io::Error::other)?
}

pub fn spawn_background_tasks(
    state: Arc<AppState>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::new();
    let retrain_interval_secs = state.config.retrain_interval_secs.max(1);
    let features_log_path = state.config.features_log_path.clone();

    let trainer_state = state.clone();
    let mut trainer_shutdown = shutdown_rx.clone();
    handles.push(tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(retrain_interval_secs));
        interval.tick().await;
        loop {
            tokio::select! {
                _ = trainer_shutdown.changed() => break,
                _ = trainer_state.retrain_notify.notified() => {},
                _ = interval.tick() => {},
            }
            if let Err(err) = run_retrain_iteration(&trainer_state).await {
                tracing::warn!(error = %err, "background retrain failed");
            }
        }
    }));

    if let Some(log_path) = features_log_path {
        let tailer_state = state;
        handles.push(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(2));
            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => break,
                    _ = ticker.tick() => {},
                }
                if let Err(err) = run_tailer_iteration(&log_path, &tailer_state).await {
                    tracing::warn!(
                        path = %log_path.display(),
                        error = %err,
                        "file tailer poll error"
                    );
                }
            }
        }));
    }

    handles
}

pub async fn serve_with_shutdown<F>(
    listener: TcpListener,
    state: Arc<AppState>,
    shutdown_signal: F,
) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let bg_handles = spawn_background_tasks(state.clone(), shutdown_rx);
    let app = build_router(state.clone());

    let res = axum::serve(NoDelayTcpListener(listener), app)
        .with_graceful_shutdown(async move {
            shutdown_signal.await;
            let _ = shutdown_tx.send(true);
        })
        .await;

    for handle in bg_handles {
        let _ = handle.await;
    }

    tracing::info!("shutting down; persisting latest model snapshot to SQLite");
    if let Err(err) = state.engine.persist_snapshot() {
        tracing::warn!(error = %err, "failed to persist model snapshot on shutdown");
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;
    use axum::serve::Listener;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn test_nodelay_tcp_listener_sets_tcp_nodelay_and_graceful_shutdown_persists_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = AppConfig {
            db_path: dir.path().join("runtime_serve.db"),
            ..AppConfig::default()
        };
        let state = Arc::new(AppState::new(cfg).unwrap());

        let direct_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut direct_nodelay = NoDelayTcpListener(direct_listener);
        let direct_addr = direct_nodelay.local_addr().unwrap();
        let client_fut = TcpStream::connect(direct_addr);
        let ((accepted_stream, _), client_res) = tokio::join!(direct_nodelay.accept(), client_fut);
        let _client = client_res.unwrap();
        assert!(accepted_stream.nodelay().unwrap());

        let raw_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let nodelay_listener = NoDelayTcpListener(raw_listener);
        let addr = nodelay_listener.local_addr().unwrap();

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let server_state = state.clone();
        let server_handle = tokio::spawn(async move {
            serve_with_shutdown(nodelay_listener.0, server_state, async move {
                let _ = rx.await;
            })
            .await
        });

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(
                b"GET /health_check HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.unwrap();
        let resp = String::from_utf8_lossy(&buf);
        assert!(resp.contains("200 OK"));
        assert!(resp.ends_with("ok\n"));

        let _ = tx.send(());
        server_handle.await.unwrap().unwrap();

        let saved_blob = state.store.load_latest_snapshot_blob().unwrap();
        assert!(saved_blob.is_some());
    }

    #[tokio::test]
    async fn test_run_tailer_and_retrain_iterations_and_background_tasks() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("fe.features.log");
        let cfg = AppConfig {
            db_path: dir.path().join("runtime_bg.db"),
            features_log_path: Some(log_path.clone()),
            min_new_samples_for_retrain: 25,
            retrain_interval_secs: 1,
            ..AppConfig::default()
        };
        let state = Arc::new(AppState::new(cfg).unwrap());

        // Missing file returns Ok(0)
        assert_eq!(run_tailer_iteration(&log_path, &state).await.unwrap(), 0);

        // Write 35 lines to trigger retrain notification
        let mut content = String::new();
        for i in 0..35 {
            let mem = 20_000_000 + i * 1_000_000;
            content.push_str(&format!(
                "digest=d{i}|cpuCostNs=50000000|memCostBytes={mem}|scanBytes=4096000|scanRows=2000|returnRows=50|spilledBytes=0|time=15|state=EOF|catalog=default_catalog|database=db1|tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]\n"
            ));
        }
        std::fs::write(&log_path, &content).unwrap();

        let ingested = run_tailer_iteration(&log_path, &state).await.unwrap();
        assert_eq!(ingested, 35);
        assert_eq!(state.ingest_features_total.load(Ordering::Relaxed), 35);

        // Subsequent poll with no new lines returns Ok(0)
        assert_eq!(run_tailer_iteration(&log_path, &state).await.unwrap(), 0);

        // Run retrain iteration directly and verify model promotion
        let report = run_retrain_iteration(&state).await.unwrap();
        assert!(report.promoted);
        assert_eq!(report.model_version, 1);

        // Exercise background tasks spawn + notify + clean shutdown
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let handles = spawn_background_tasks(state.clone(), shutdown_rx);
        assert_eq!(handles.len(), 2);

        state.retrain_notify.notify_one();
        tokio::time::sleep(Duration::from_millis(50)).await;

        shutdown_tx.send(true).unwrap();
        for h in handles {
            h.await.unwrap();
        }
    }
}
