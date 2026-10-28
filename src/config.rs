use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

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

fn read_env_var(keys: &[&str]) -> Option<(String, String)> {
    for &key in keys {
        if let Ok(val) = std::env::var(key) {
            return Some((key.to_string(), val));
        }
    }
    None
}

fn parse_env_var<T: FromStr>(keys: &[&str], target: &mut T) {
    if let Some((key, raw)) = read_env_var(keys) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return;
        }
        match trimmed.parse::<T>() {
            Ok(parsed) => *target = parsed,
            Err(_) => {
                tracing::warn!(
                    env_var = %key,
                    raw_value = %raw,
                    "ignoring malformed environment variable value; using default"
                );
            }
        }
    }
}

impl AppConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Some((_, v)) = read_env_var(&[
            "STARROCKS_COST_PREDICTOR_BIND_ADDR",
            "SR_PREDICTOR_BIND_ADDR",
        ]) && !v.trim().is_empty()
        {
            cfg.bind_addr = v.trim().to_string();
        }

        if let Some((_, v)) =
            read_env_var(&["STARROCKS_COST_PREDICTOR_DB_PATH", "SR_PREDICTOR_DB_PATH"])
            && !v.trim().is_empty()
        {
            cfg.db_path = PathBuf::from(v.trim());
        }

        if let Some((_, v)) = read_env_var(&[
            "STARROCKS_COST_PREDICTOR_FEATURES_LOG_PATH",
            "SR_PREDICTOR_FEATURES_LOG_PATH",
        ]) && !v.trim().is_empty()
        {
            cfg.features_log_path = Some(PathBuf::from(v.trim()));
        }

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_MIN_MEM_BYTES",
                "STARROCKS_COST_PREDICTOR_MIN_PREDICTED_MEM_BYTES",
                "SR_PREDICTOR_MIN_PREDICTED_MEM_BYTES",
                "SR_PREDICTOR_MIN_MEM_BYTES",
            ],
            &mut cfg.min_predicted_mem_bytes,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_FALLBACK_MEM_BYTES",
                "SR_PREDICTOR_FALLBACK_MEM_BYTES",
            ],
            &mut cfg.fallback_mem_bytes,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_MAX_CLUSTER_MEM_MULTIPLIER",
                "STARROCKS_COST_PREDICTOR_MAX_MEM_MULTIPLIER",
                "SR_PREDICTOR_MAX_MEM_SAFETY_MULTIPLIER",
                "SR_PREDICTOR_MAX_CLUSTER_MEM_MULTIPLIER",
            ],
            &mut cfg.max_cluster_mem_multiplier,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_MAX_SKETCH_ENTRIES",
                "SR_PREDICTOR_MAX_SKETCH_ENTRIES",
            ],
            &mut cfg.max_sketch_entries,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_MIN_NEW_SAMPLES_FOR_RETRAIN",
                "STARROCKS_COST_PREDICTOR_MIN_SAMPLES_FOR_RETRAIN",
                "SR_PREDICTOR_MIN_NEW_SAMPLES_FOR_RETRAIN",
                "SR_PREDICTOR_MIN_SAMPLES_FOR_RETRAIN",
            ],
            &mut cfg.min_new_samples_for_retrain,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_RETRAIN_INTERVAL_SECS",
                "SR_PREDICTOR_RETRAIN_INTERVAL_SECS",
            ],
            &mut cfg.retrain_interval_secs,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD",
                "SR_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD",
            ],
            &mut cfg.drift_q_error_threshold,
        );

        cfg
    }
}
