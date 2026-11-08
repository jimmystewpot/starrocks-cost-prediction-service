use starrocks_cost_prediction_service::{
    api::AppState, config::AppConfig, runtime::serve_with_shutdown,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::from_env();
    let bind_addr = config.bind_addr.clone();
    let state = Arc::new(AppState::new(config)?);

    let tcp_listener = TcpListener::bind(&bind_addr).await?;
    tracing::info!(
        bind_addr = %bind_addr,
        "StarRocks Cost Prediction Service listening with TCP_NODELAY enabled"
    );

    serve_with_shutdown(tcp_listener, state, shutdown_signal()).await?;
    Ok(())
}

async fn shutdown_signal() {
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
}
