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

fn read_env_var<F>(keys: &[&str], lookup: &F) -> Option<(String, String)>
where
    F: Fn(&str) -> Option<String>,
{
    for &key in keys {
        if let Some(val) = lookup(key) {
            return Some((key.to_string(), val));
        }
    }
    None
}

fn parse_env_var<T: FromStr, F>(keys: &[&str], target: &mut T, lookup: &F)
where
    F: Fn(&str) -> Option<String>,
{
    if let Some((key, raw)) = read_env_var(keys, lookup) {
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
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    pub fn from_lookup<F>(lookup: F) -> Self
    where
        F: Fn(&str) -> Option<String>,
    {
        let mut cfg = Self::default();

        if let Some((_, v)) = read_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_BIND_ADDR",
                "SR_PREDICTOR_BIND_ADDR",
            ],
            &lookup,
        ) && !v.trim().is_empty()
        {
            cfg.bind_addr = v.trim().to_string();
        }

        if let Some((_, v)) = read_env_var(
            &["STARROCKS_COST_PREDICTOR_DB_PATH", "SR_PREDICTOR_DB_PATH"],
            &lookup,
        ) && !v.trim().is_empty()
        {
            cfg.db_path = PathBuf::from(v.trim());
        }

        if let Some((_, v)) = read_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_FEATURES_LOG_PATH",
                "SR_PREDICTOR_FEATURES_LOG_PATH",
            ],
            &lookup,
        ) && !v.trim().is_empty()
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
            &lookup,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_FALLBACK_MEM_BYTES",
                "SR_PREDICTOR_FALLBACK_MEM_BYTES",
            ],
            &mut cfg.fallback_mem_bytes,
            &lookup,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_MAX_CLUSTER_MEM_MULTIPLIER",
                "STARROCKS_COST_PREDICTOR_MAX_MEM_MULTIPLIER",
                "SR_PREDICTOR_MAX_MEM_SAFETY_MULTIPLIER",
                "SR_PREDICTOR_MAX_CLUSTER_MEM_MULTIPLIER",
            ],
            &mut cfg.max_cluster_mem_multiplier,
            &lookup,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_MAX_SKETCH_ENTRIES",
                "SR_PREDICTOR_MAX_SKETCH_ENTRIES",
            ],
            &mut cfg.max_sketch_entries,
            &lookup,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_MIN_NEW_SAMPLES_FOR_RETRAIN",
                "STARROCKS_COST_PREDICTOR_MIN_SAMPLES_FOR_RETRAIN",
                "SR_PREDICTOR_MIN_NEW_SAMPLES_FOR_RETRAIN",
                "SR_PREDICTOR_MIN_SAMPLES_FOR_RETRAIN",
            ],
            &mut cfg.min_new_samples_for_retrain,
            &lookup,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_RETRAIN_INTERVAL_SECS",
                "SR_PREDICTOR_RETRAIN_INTERVAL_SECS",
            ],
            &mut cfg.retrain_interval_secs,
            &lookup,
        );

        parse_env_var(
            &[
                "STARROCKS_COST_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD",
                "SR_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD",
            ],
            &mut cfg.drift_q_error_threshold,
            &lookup,
        );

        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_app_config_from_env_and_empty_lookup_match_defaults() {
        let empty_cfg = AppConfig::from_lookup(|_| None);
        assert_eq!(empty_cfg, AppConfig::default());

        let env_cfg = AppConfig::from_env();
        assert!(!env_cfg.bind_addr.is_empty());
        assert!(env_cfg.min_predicted_mem_bytes > 0);
    }

    #[test]
    fn test_app_config_from_lookup_parses_primary_and_alias_keys() {
        let primary: HashMap<&str, &str> = HashMap::from([
            ("STARROCKS_COST_PREDICTOR_BIND_ADDR", " 127.0.0.1:9090 "),
            ("STARROCKS_COST_PREDICTOR_DB_PATH", "/tmp/primary.db"),
            (
                "STARROCKS_COST_PREDICTOR_FEATURES_LOG_PATH",
                "/var/log/fe.features.log",
            ),
            ("STARROCKS_COST_PREDICTOR_MIN_MEM_BYTES", "2097152"),
            ("STARROCKS_COST_PREDICTOR_FALLBACK_MEM_BYTES", "134217728"),
            ("STARROCKS_COST_PREDICTOR_MAX_CLUSTER_MEM_MULTIPLIER", "3.5"),
            ("STARROCKS_COST_PREDICTOR_MAX_SKETCH_ENTRIES", "50000"),
            (
                "STARROCKS_COST_PREDICTOR_MIN_NEW_SAMPLES_FOR_RETRAIN",
                "100",
            ),
            ("STARROCKS_COST_PREDICTOR_RETRAIN_INTERVAL_SECS", "60"),
            ("STARROCKS_COST_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD", "1.5"),
        ]);

        let cfg_primary = AppConfig::from_lookup(|k| primary.get(k).map(|v| (*v).to_string()));
        assert_eq!(cfg_primary.bind_addr, "127.0.0.1:9090");
        assert_eq!(cfg_primary.db_path, PathBuf::from("/tmp/primary.db"));
        assert_eq!(
            cfg_primary.features_log_path,
            Some(PathBuf::from("/var/log/fe.features.log"))
        );
        assert_eq!(cfg_primary.min_predicted_mem_bytes, 2_097_152);
        assert_eq!(cfg_primary.fallback_mem_bytes, 134_217_728);
        assert!((cfg_primary.max_cluster_mem_multiplier - 3.5).abs() < 1e-9);
        assert_eq!(cfg_primary.max_sketch_entries, 50_000);
        assert_eq!(cfg_primary.min_new_samples_for_retrain, 100);
        assert_eq!(cfg_primary.retrain_interval_secs, 60);
        assert!((cfg_primary.drift_q_error_threshold - 1.5).abs() < 1e-9);

        let aliases: HashMap<&str, &str> = HashMap::from([
            ("SR_PREDICTOR_BIND_ADDR", "127.0.0.1:9091"),
            ("SR_PREDICTOR_DB_PATH", "/tmp/alias.db"),
            (
                "SR_PREDICTOR_FEATURES_LOG_PATH",
                "/var/log/alias.features.log",
            ),
            ("SR_PREDICTOR_MIN_MEM_BYTES", "4194304"),
            ("SR_PREDICTOR_FALLBACK_MEM_BYTES", "268435456"),
            ("SR_PREDICTOR_MAX_MEM_SAFETY_MULTIPLIER", "1.75"),
            ("SR_PREDICTOR_MAX_SKETCH_ENTRIES", "25000"),
            ("SR_PREDICTOR_MIN_SAMPLES_FOR_RETRAIN", "50"),
            ("SR_PREDICTOR_RETRAIN_INTERVAL_SECS", "120"),
            ("SR_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD", "1.35"),
        ]);

        let cfg_alias = AppConfig::from_lookup(|k| aliases.get(k).map(|v| (*v).to_string()));
        assert_eq!(cfg_alias.bind_addr, "127.0.0.1:9091");
        assert_eq!(cfg_alias.db_path, PathBuf::from("/tmp/alias.db"));
        assert_eq!(
            cfg_alias.features_log_path,
            Some(PathBuf::from("/var/log/alias.features.log"))
        );
        assert_eq!(cfg_alias.min_predicted_mem_bytes, 4_194_304);
        assert_eq!(cfg_alias.fallback_mem_bytes, 268_435_456);
        assert!((cfg_alias.max_cluster_mem_multiplier - 1.75).abs() < 1e-9);
        assert_eq!(cfg_alias.max_sketch_entries, 25_000);
        assert_eq!(cfg_alias.min_new_samples_for_retrain, 50);
        assert_eq!(cfg_alias.retrain_interval_secs, 120);
        assert!((cfg_alias.drift_q_error_threshold - 1.35).abs() < 1e-9);
    }

    #[test]
    fn test_app_config_from_lookup_ignores_whitespace_and_malformed_numbers() {
        let malformed: HashMap<&str, &str> = HashMap::from([
            ("STARROCKS_COST_PREDICTOR_BIND_ADDR", "   "),
            ("STARROCKS_COST_PREDICTOR_DB_PATH", "   "),
            ("STARROCKS_COST_PREDICTOR_FEATURES_LOG_PATH", "   "),
            ("STARROCKS_COST_PREDICTOR_MIN_MEM_BYTES", "   "),
            (
                "STARROCKS_COST_PREDICTOR_FALLBACK_MEM_BYTES",
                "not_a_number",
            ),
            (
                "STARROCKS_COST_PREDICTOR_MAX_CLUSTER_MEM_MULTIPLIER",
                "not_a_number",
            ),
            (
                "STARROCKS_COST_PREDICTOR_MAX_SKETCH_ENTRIES",
                "not_a_number",
            ),
            (
                "STARROCKS_COST_PREDICTOR_MIN_NEW_SAMPLES_FOR_RETRAIN",
                "not_a_number",
            ),
            (
                "STARROCKS_COST_PREDICTOR_RETRAIN_INTERVAL_SECS",
                "not_a_number",
            ),
            (
                "STARROCKS_COST_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD",
                "not_a_number",
            ),
        ]);

        let cfg = AppConfig::from_lookup(|k| malformed.get(k).map(|v| (*v).to_string()));
        assert_eq!(cfg, AppConfig::default());
    }
}
