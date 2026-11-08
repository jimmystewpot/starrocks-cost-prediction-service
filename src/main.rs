use starrocks_cost_prediction_service::{config::AppConfig, runtime::run_with_config};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    run_with_config(AppConfig::from_env(), async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
}
