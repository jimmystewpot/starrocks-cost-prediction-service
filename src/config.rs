use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub bind_addr: String,
    pub db_path: PathBuf,
    pub features_log_path: Option<PathBuf>,
    pub min_predicted_mem_bytes: u64,
    pub fallback_mem_bytes: u64,
    pub max_cluster_mem_multiplier: f64,
    pub max_sketch_entries: usize,
    pub min_new_samples_for_retrain: usize,
    pub retrain_interval_secs: u64,
    pub drift_q_error_threshold: f64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:8080".to_string(),
            db_path: PathBuf::from("data/cost_predictor.db"),
            features_log_path: None,
            min_predicted_mem_bytes: 1_048_576,
            fallback_mem_bytes: 67_108_864,
            max_cluster_mem_multiplier: 2.0,
            max_sketch_entries: 100_000,
            min_new_samples_for_retrain: 200,
            retrain_interval_secs: 300,
            drift_q_error_threshold: 1.2,
        }
    }
}

impl AppConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_BIND_ADDR") {
            cfg.bind_addr = v;
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_DB_PATH") {
            cfg.db_path = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_FEATURES_LOG_PATH")
            && !v.trim().is_empty()
        {
            cfg.features_log_path = Some(PathBuf::from(v));
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_MIN_MEM_BYTES")
            && let Ok(parsed) = v.parse()
        {
            cfg.min_predicted_mem_bytes = parsed;
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_FALLBACK_MEM_BYTES")
            && let Ok(parsed) = v.parse()
        {
            cfg.fallback_mem_bytes = parsed;
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_MAX_CLUSTER_MEM_MULTIPLIER")
            && let Ok(parsed) = v.parse()
        {
            cfg.max_cluster_mem_multiplier = parsed;
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_MAX_SKETCH_ENTRIES")
            && let Ok(parsed) = v.parse()
        {
            cfg.max_sketch_entries = parsed;
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_MIN_NEW_SAMPLES_FOR_RETRAIN")
            && let Ok(parsed) = v.parse()
        {
            cfg.min_new_samples_for_retrain = parsed;
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_RETRAIN_INTERVAL_SECS")
            && let Ok(parsed) = v.parse()
        {
            cfg.retrain_interval_secs = parsed;
        }
        if let Ok(v) = std::env::var("STARROCKS_COST_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD")
            && let Ok(parsed) = v.parse()
        {
            cfg.drift_q_error_threshold = parsed;
        }
        cfg
    }
}
