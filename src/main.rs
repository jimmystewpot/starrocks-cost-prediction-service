use starrocks_cost_prediction_service::config::AppConfig;

fn main() {
    let config = AppConfig::from_env();
    tracing::info!(bind_addr = %config.bind_addr, "StarRocks Cost Prediction Service configured");
}
