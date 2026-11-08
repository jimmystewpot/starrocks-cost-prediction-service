use serde::{Deserialize, Serialize};

pub const DENSE_FEATURE_DIM: usize = 39;
pub type DenseFeatureVector = [f64; DENSE_FEATURE_DIM];

#[derive(Clone, Debug, PartialEq)]
pub struct GbdtTrainingRow {
    pub features: DenseFeatureVector,
    pub log_mem: f64,
    pub log_scan_bytes: f64,
    pub log_return_rows: f64,
    pub log_cpu_ns: f64,
    pub spilled: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperatorSegmentKind {
    Scan,
    HashJoin,
    HashAgg,
    Exchange,
    Generic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorSegment {
    pub ordinal: i64,
    pub kind: OperatorSegmentKind,
    pub count: u64,
    pub mean_log_rows: f64,
    pub mean_row_size: f64,
    pub mean_log_cbo_mem: f64,
    pub mean_log_cbo_cpu: f64,
    pub mean_log_limit: f64,
    pub extra: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalPlanFeatures {
    pub tables: [String; 3],
    pub num_be_nodes: u64,
    pub avg_cpu_cores: u64,
    pub env_mem_log: u64,
    pub pipeline_dop: u64,
    pub segments: Vec<OperatorSegment>,
    pub plan_hash: u64,
    pub plan_features_24: [f64; 24],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySample {
    pub digest: String,
    pub plan_features: CanonicalPlanFeatures,
    pub effective_mem_bytes: u64,
    pub scan_bytes: u64,
    pub scan_rows: u64,
    pub return_rows: u64,
    pub cpu_cost_ns: u64,
    pub spilled_bytes: u64,
    pub query_time_ms: u64,
    pub state: String,
    pub timestamp_s: u64,
}

#[derive(Deserialize)]
struct PredictionOutputDe {
    predicted_mem_bytes_p90: u64,
    predicted_mem_bytes_p50: u64,
    predicted_scan_bytes: u64,
    predicted_return_rows: u64,
    predicted_cpu_cost_ns: u64,
    spill_probability: f64,
    confidence: f64,
    tier_weights: [f64; 3],
    dominant_tier: String,
    model_version: u64,
    latency_us: u64,
}

impl From<PredictionOutputDe> for PredictionOutput {
    fn from(de: PredictionOutputDe) -> Self {
        let dominant_tier: &'static str = match de.dominant_tier.as_str() {
            "sketch_tier1" => "sketch_tier1",
            "gbdt_tier2" => "gbdt_tier2",
            "rls_tier3" => "rls_tier3",
            "fallback" => "fallback",
            _ => Box::leak(de.dominant_tier.into_boxed_str()),
        };
        Self {
            predicted_mem_bytes_p90: de.predicted_mem_bytes_p90,
            predicted_mem_bytes_p50: de.predicted_mem_bytes_p50,
            predicted_scan_bytes: de.predicted_scan_bytes,
            predicted_return_rows: de.predicted_return_rows,
            predicted_cpu_cost_ns: de.predicted_cpu_cost_ns,
            spill_probability: de.spill_probability,
            confidence: de.confidence,
            tier_weights: de.tier_weights,
            dominant_tier,
            model_version: de.model_version,
            latency_us: de.latency_us,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PredictionOutput {
    pub predicted_mem_bytes_p90: u64,
    pub predicted_mem_bytes_p50: u64,
    pub predicted_scan_bytes: u64,
    pub predicted_return_rows: u64,
    pub predicted_cpu_cost_ns: u64,
    pub spill_probability: f64,
    pub confidence: f64,
    pub tier_weights: [f64; 3],
    pub dominant_tier: &'static str,
    pub model_version: u64,
    pub latency_us: u64,
}

impl<'de> Deserialize<'de> for PredictionOutput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let de = PredictionOutputDe::deserialize(deserializer)?;
        Ok(Self::from(de))
    }
}

pub fn watermark_mem_bytes(raw_bytes: u64, plan_hash: u64, min_bytes: u64) -> u64 {
    let base = raw_bytes.max(min_bytes);
    if base >= 1_048_576 {
        (base & !0x3FF) | (plan_hash & 0x3FF)
    } else {
        base
    }
}

pub fn compute_effective_mem_bytes(
    mem_cost_bytes: u64,
    spilled_bytes: u64,
    state: &str,
) -> Option<u64> {
    let normalized = state.trim().to_ascii_uppercase();
    if normalized == "EOF" || normalized == "OK" {
        Some(mem_cost_bytes.saturating_add(spilled_bytes))
    } else if normalized.starts_with("ERR") {
        if mem_cost_bytes == 0 && spilled_bytes == 0 {
            None
        } else {
            let sum = mem_cost_bytes.saturating_add(spilled_bytes);
            let penalized = (mem_cost_bytes as f64 * 1.5) as u64;
            Some(sum.max(penalized))
        }
    } else {
        None
    }
}

pub fn compute_cluster_max_mem_bytes(num_be_nodes: u64, env_mem_log: u64, multiplier: f64) -> u64 {
    let nodes = num_be_nodes.max(1) as f64;
    let per_be = ((env_mem_log.saturating_add(1)).min(40) as f64).exp();
    let total =
        (nodes * per_be * multiplier.max(0.1)).clamp(1_073_741_824.0, u64::MAX as f64 / 2.0);
    total as u64
}

pub fn memory_stratum(bytes: u64) -> u8 {
    if bytes < 10_000_000 {
        0
    } else if bytes < 100_000_000 {
        1
    } else if bytes < 1_000_000_000 {
        2
    } else if bytes < 10_000_000_000 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;

    #[test]
    fn test_watermark_mem_bytes_embeds_low_10_bits_of_plan_hash() {
        let plan_hash: u64 = 0xABCD_EF00_0000_02A5; // low 10 bits = 0x2A5 (677)
        let raw_bytes: u64 = 104_857_600; // 100 MiB
        let watermarked = watermark_mem_bytes(raw_bytes, plan_hash, 1_048_576);
        assert_eq!(watermarked & 0x3FF, 0x2A5);
        assert!((watermarked as i64 - raw_bytes as i64).abs() < 1024);

        // Sub-1MiB allocation bypasses watermark
        assert_eq!(watermark_mem_bytes(512_000, plan_hash, 256_000), 512_000);

        // PredictionOutput deserialization covers fallback and custom dominant_tier
        for tier in ["fallback", "custom_tier"] {
            let out = PredictionOutput {
                predicted_mem_bytes_p90: 1024,
                predicted_mem_bytes_p50: 512,
                predicted_scan_bytes: 100,
                predicted_return_rows: 10,
                predicted_cpu_cost_ns: 1000,
                spill_probability: 0.0,
                confidence: 0.5,
                tier_weights: [0.0, 0.0, 1.0],
                dominant_tier: tier,
                model_version: 0,
                latency_us: 5,
            };
            let json = serde_json::to_string(&out).unwrap();
            let decoded: PredictionOutput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.dominant_tier, tier);
        }
    }

    #[test]
    fn test_compute_effective_mem_bytes_handles_spill_and_oom_censoring() {
        // Normal EOF query adds mem + spill
        assert_eq!(
            compute_effective_mem_bytes(50_000_000, 20_000_000, "EOF"),
            Some(70_000_000)
        );
        // OOM / ERR query with non-zero memory applies 1.5x lower-bound penalty
        assert_eq!(
            compute_effective_mem_bytes(100_000_000, 0, "ERR"),
            Some(150_000_000)
        );
        // ERR query with 0 memory (syntax error) is rejected (None)
        assert_eq!(compute_effective_mem_bytes(0, 0, "ERR"), None);
    }

    #[test]
    fn test_compute_cluster_max_mem_bytes_compensates_for_java_log1p_truncation() {
        // 64 GiB BE node -> Java (long) Math.log1p(68719476736) == 24
        // exp(24) is only ~26.4 GiB, exp(25) is ~72.0 GiB
        let max_bytes = compute_cluster_max_mem_bytes(1, 24, 1.0);
        assert!(
            max_bytes > 68_719_476_736,
            "max_bytes={} must exceed 64 GiB",
            max_bytes
        );
    }

    #[test]
    fn test_memory_stratum_boundaries_and_default_config() {
        assert_eq!(memory_stratum(0), 0);
        assert_eq!(memory_stratum(9_999_999), 0);
        assert_eq!(memory_stratum(10_000_000), 1);
        assert_eq!(memory_stratum(99_999_999), 1);
        assert_eq!(memory_stratum(100_000_000), 2);
        assert_eq!(memory_stratum(999_999_999), 2);
        assert_eq!(memory_stratum(1_000_000_000), 3);
        assert_eq!(memory_stratum(9_999_999_999), 3);
        assert_eq!(memory_stratum(10_000_000_000), 4);

        let cfg = AppConfig::default();
        assert_eq!(cfg.bind_addr, "0.0.0.0:8080");
        assert_eq!(cfg.min_predicted_mem_bytes, 1_048_576);
        assert_eq!(cfg.fallback_mem_bytes, 67_108_864);
    }
}
