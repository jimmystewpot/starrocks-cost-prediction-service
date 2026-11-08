use crate::types::CanonicalPlanFeatures;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

const EWMA_ALPHA: f64 = 0.20;
const MIN_VAR_LOG_MEM: f64 = 0.04;
const Z_P90: f64 = 1.2816;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanSketchEntry {
    pub hit_count: u64,
    pub last_seen_epoch_s: u64,
    pub ewma_log_mem: f64,
    pub ewma_var_log_mem: f64,
    pub ewma_log_scan_bytes: f64,
    pub ewma_log_return_rows: f64,
    pub ewma_log_cpu_ns: f64,
    pub ewma_spill_rate: f64,
    pub ewma_ref_scan_log_rows: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchPrediction {
    pub log_mem_p50: f64,
    pub log_mem_p90: f64,
    pub log_scan_bytes: f64,
    pub log_return_rows: f64,
    pub log_cpu_ns: f64,
    pub spill_prob: f64,
    pub confidence: f64,
}

pub struct PlanSketchCache {
    map: DashMap<u64, PlanSketchEntry>,
    max_entries: usize,
}

fn ref_scan_log_rows(plan: &CanonicalPlanFeatures) -> f64 {
    if plan.plan_features_24[4] > 0.0 {
        plan.plan_features_24[4]
    } else {
        plan.plan_features_24[3]
    }
}

fn current_epoch_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl PlanSketchCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            map: DashMap::new(),
            max_entries: max_entries.max(1),
        }
    }

    pub fn predict(&self, plan: &CanonicalPlanFeatures) -> Option<SketchPrediction> {
        let entry = self.map.get(&plan.plan_hash)?;
        let scan_log_rows = ref_scan_log_rows(plan);
        let delta_r = (scan_log_rows - entry.ewma_ref_scan_log_rows).clamp(-2.0, 2.0);

        let log_mem_p50 = entry.ewma_log_mem + 0.5 * delta_r;
        let log_mem_p90 = log_mem_p50 + Z_P90 * entry.ewma_var_log_mem.max(MIN_VAR_LOG_MEM).sqrt();
        let log_scan_bytes = (entry.ewma_log_scan_bytes + 0.5 * delta_r).max(0.0);
        let log_return_rows = (entry.ewma_log_return_rows + 0.5 * delta_r).max(0.0);
        let log_cpu_ns = (entry.ewma_log_cpu_ns + 0.5 * delta_r).max(0.0);
        let spill_prob = entry.ewma_spill_rate.clamp(0.0, 1.0);

        let n = entry.hit_count as f64;
        let confidence = ((n / (n + 5.0))
            * (-0.20 * entry.ewma_var_log_mem).exp()
            * (-0.15 * delta_r.abs()).exp())
        .clamp(0.0, 0.95);

        Some(SketchPrediction {
            log_mem_p50,
            log_mem_p90,
            log_scan_bytes,
            log_return_rows,
            log_cpu_ns,
            spill_prob,
            confidence,
        })
    }

    pub fn update(
        &self,
        plan: &CanonicalPlanFeatures,
        effective_mem_bytes: u64,
        scan_bytes: u64,
        return_rows: u64,
        cpu_cost_ns: u64,
        spilled: bool,
    ) {
        let log_mem = (effective_mem_bytes as f64).ln_1p();
        let log_scan = (scan_bytes as f64).ln_1p();
        let log_rows = (return_rows as f64).ln_1p();
        let log_cpu = (cpu_cost_ns as f64).ln_1p();
        let spill_val = if spilled { 1.0 } else { 0.0 };
        let scan_log_rows = ref_scan_log_rows(plan);
        let now_s = current_epoch_s();

        self.map
            .entry(plan.plan_hash)
            .and_modify(|entry| {
                let diff = log_mem - entry.ewma_log_mem;
                entry.ewma_log_mem += EWMA_ALPHA * diff;
                entry.ewma_var_log_mem = ((1.0 - EWMA_ALPHA)
                    * (entry.ewma_var_log_mem + EWMA_ALPHA * diff * diff))
                    .max(MIN_VAR_LOG_MEM);
                entry.ewma_log_scan_bytes += EWMA_ALPHA * (log_scan - entry.ewma_log_scan_bytes);
                entry.ewma_log_return_rows += EWMA_ALPHA * (log_rows - entry.ewma_log_return_rows);
                entry.ewma_log_cpu_ns += EWMA_ALPHA * (log_cpu - entry.ewma_log_cpu_ns);
                entry.ewma_spill_rate = (entry.ewma_spill_rate
                    + EWMA_ALPHA * (spill_val - entry.ewma_spill_rate))
                    .clamp(0.0, 1.0);
                entry.ewma_ref_scan_log_rows +=
                    EWMA_ALPHA * (scan_log_rows - entry.ewma_ref_scan_log_rows);
                entry.hit_count = entry.hit_count.saturating_add(1);
                entry.last_seen_epoch_s = now_s;
            })
            .or_insert(PlanSketchEntry {
                hit_count: 1,
                last_seen_epoch_s: now_s,
                ewma_log_mem: log_mem,
                ewma_var_log_mem: MIN_VAR_LOG_MEM,
                ewma_log_scan_bytes: log_scan,
                ewma_log_return_rows: log_rows,
                ewma_log_cpu_ns: log_cpu,
                ewma_spill_rate: spill_val,
                ewma_ref_scan_log_rows: scan_log_rows,
            });

        if self.map.len() > self.max_entries {
            self.evict_if_needed(Some(plan.plan_hash));
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn export_entries(&self) -> Vec<(u64, PlanSketchEntry)> {
        self.map
            .iter()
            .map(|r| (*r.key(), r.value().clone()))
            .collect()
    }

    pub fn import_entries(&self, entries: Vec<(u64, PlanSketchEntry)>) {
        self.map.clear();
        for (hash, entry) in entries {
            self.map.insert(hash, entry);
        }
        self.evict_if_needed(None);
    }

    fn evict_if_needed(&self, exclude_key: Option<u64>) {
        while self.map.len() > self.max_entries {
            let victim = self
                .map
                .iter()
                .filter(|r| Some(*r.key()) != exclude_key)
                .take(64)
                .min_by_key(|r| (r.value().hit_count, r.value().last_seen_epoch_s, *r.key()))
                .map(|r| *r.key())
                .or_else(|| {
                    self.map
                        .iter()
                        .take(64)
                        .min_by_key(|r| {
                            (r.value().hit_count, r.value().last_seen_epoch_s, *r.key())
                        })
                        .map(|r| *r.key())
                });
            if let Some(key) = victim {
                self.map.remove(&key);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_plan(
        plan_hash: u64,
        scan_log_rows: f64,
        cbo_mem_log: f64,
    ) -> CanonicalPlanFeatures {
        let mut plan_features_24 = [0.0; 24];
        plan_features_24[3] = scan_log_rows; // scan_mean_log_rows
        plan_features_24[4] = scan_log_rows; // scan_est_log_sum_rows
        plan_features_24[9] = 1.0; // join_count
        plan_features_24[14] = 1.0; // agg_count
        plan_features_24[21] = cbo_mem_log; // cbo_est_total_mem_log
        plan_features_24[22] = 14.0; // cbo_est_total_cpu_log
        plan_features_24[23] = (8.0_f64).ln_1p(); // env_dop_log

        CanonicalPlanFeatures {
            tables: ["0".to_string(), "0".to_string(), "101".to_string()],
            num_be_nodes: 3,
            avg_cpu_cores: 16,
            env_mem_log: 24,
            pipeline_dop: 8,
            segments: Vec::new(),
            plan_hash,
            plan_features_24,
        }
    }

    #[test]
    fn test_plan_sketch_converges_and_scales_with_cardinality() {
        let sketch = PlanSketchCache::new(1000);
        let plan_base = make_test_plan(0x1234_5678_9ABC_DEF0, 14.0, 16.0);
        assert!(sketch.predict(&plan_base).is_none());

        for _ in 0..10 {
            sketch.update(
                &plan_base,
                100_000_000,
                50_000_000,
                10_000,
                500_000_000,
                false,
            );
        }
        let pred_base = sketch.predict(&plan_base).unwrap();
        let expected_log = (100_000_000.0_f64).ln_1p();
        assert!((pred_base.log_mem_p50 - expected_log).abs() < 0.05);
        assert!(pred_base.log_mem_p90 > pred_base.log_mem_p50);
        assert!(pred_base.confidence > 0.60);

        let plan_larger = make_test_plan(0x1234_5678_9ABC_DEF0, 18.0, 16.0);
        let pred_larger = sketch.predict(&plan_larger).unwrap();
        assert!(pred_larger.log_mem_p50 > pred_base.log_mem_p50);
    }

    #[test]
    fn test_plan_sketch_eviction_and_export_import() {
        let sketch = PlanSketchCache::new(2);
        let p1 = make_test_plan(1, 10.0, 15.0);
        let p2 = make_test_plan(2, 10.0, 15.0);
        let p3 = make_test_plan(3, 10.0, 15.0);

        sketch.update(&p1, 10_000_000, 1_000_000, 100, 1_000_000, false);
        sketch.update(&p2, 20_000_000, 2_000_000, 200, 2_000_000, true);
        sketch.update(&p2, 20_000_000, 2_000_000, 200, 2_000_000, true);
        assert_eq!(sketch.len(), 2);

        // Adding p3 should evict p1 (since p1 has hit_count 1 vs p2 hit_count 2)
        sketch.update(&p3, 30_000_000, 3_000_000, 300, 3_000_000, false);
        assert_eq!(sketch.len(), 2);
        assert!(sketch.predict(&p1).is_none());
        assert!(sketch.predict(&p2).is_some());
        assert!(sketch.predict(&p3).is_some());

        let exported = sketch.export_entries();
        assert_eq!(exported.len(), 2);

        let restored = PlanSketchCache::new(2);
        restored.import_entries(exported);
        assert_eq!(restored.len(), 2);
        assert!(restored.predict(&p2).is_some());
    }
}
