use crate::types::{CanonicalPlanFeatures, DENSE_FEATURE_DIM, DenseFeatureVector};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

const EWMA_ALPHA: f64 = 0.20;
const KNOWN_TABLE_MIN_EXECS: u64 = 5;
const DEFAULT_LOG_BYTES_PER_ROW: f64 = 4.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableProfile {
    pub exec_count: u64,
    pub ewma_log_scan_bytes: f64,
    pub ewma_log_bytes_per_row: f64,
    pub ewma_mem_multiplier_log: f64,
    pub ewma_spill_rate: f64,
}

#[derive(Debug, Default)]
pub struct TableCatalog {
    profiles: DashMap<String, TableProfile>,
}

impl TableCatalog {
    pub fn new() -> Self {
        Self {
            profiles: DashMap::new(),
        }
    }

    pub fn record_execution(
        &self,
        tables: &[String; 3],
        scan_bytes: u64,
        scan_rows: u64,
        actual_mem_bytes: u64,
        cbo_mem_log: f64,
        spilled: bool,
    ) {
        let log_scan_bytes = (1.0 + scan_bytes as f64).ln();
        let bytes_per_row = if scan_rows > 0 {
            scan_bytes as f64 / scan_rows as f64
        } else {
            54.0
        };
        let log_bytes_per_row = (1.0 + bytes_per_row.max(0.0)).ln();
        let log_actual_mem = (1.0 + actual_mem_bytes as f64).ln();
        let mem_multiplier_log = if cbo_mem_log > 0.0 {
            (log_actual_mem - cbo_mem_log).clamp(-10.0, 10.0)
        } else {
            0.0
        };
        let spill_val = if spilled { 1.0 } else { 0.0 };

        for table in tables {
            let trimmed = table.trim();
            if trimmed.is_empty() || trimmed == "0" {
                continue;
            }
            self.profiles
                .entry(trimmed.to_string())
                .and_modify(|p| {
                    p.exec_count = p.exec_count.saturating_add(1);
                    p.ewma_log_scan_bytes =
                        (1.0 - EWMA_ALPHA) * p.ewma_log_scan_bytes + EWMA_ALPHA * log_scan_bytes;
                    p.ewma_log_bytes_per_row = (1.0 - EWMA_ALPHA) * p.ewma_log_bytes_per_row
                        + EWMA_ALPHA * log_bytes_per_row;
                    p.ewma_mem_multiplier_log = (1.0 - EWMA_ALPHA) * p.ewma_mem_multiplier_log
                        + EWMA_ALPHA * mem_multiplier_log;
                    p.ewma_spill_rate =
                        (1.0 - EWMA_ALPHA) * p.ewma_spill_rate + EWMA_ALPHA * spill_val;
                })
                .or_insert(TableProfile {
                    exec_count: 1,
                    ewma_log_scan_bytes: log_scan_bytes,
                    ewma_log_bytes_per_row: log_bytes_per_row,
                    ewma_mem_multiplier_log: mem_multiplier_log,
                    ewma_spill_rate: spill_val,
                });
        }
    }

    pub fn build_dense_features(&self, plan: &CanonicalPlanFeatures) -> (DenseFeatureVector, f64) {
        let mut dense = [0.0_f64; DENSE_FEATURE_DIM];
        dense[..24].copy_from_slice(&plan.plan_features_24);

        let mut active_tables = 0usize;
        let mut known_tables = 0usize;

        for (i, table) in plan.tables.iter().enumerate() {
            let base = 24 + i * 5;
            let trimmed = table.trim();
            if trimmed.is_empty() || trimmed == "0" {
                dense[base] = 0.0;
                dense[base + 1] = 0.0;
                dense[base + 2] = DEFAULT_LOG_BYTES_PER_ROW;
                dense[base + 3] = 0.0;
                dense[base + 4] = 0.0;
                continue;
            }

            active_tables += 1;
            if let Some(profile) = self.profiles.get(trimmed)
                && profile.exec_count >= KNOWN_TABLE_MIN_EXECS
            {
                known_tables += 1;
                dense[base] = 1.0;
                dense[base + 1] = profile.ewma_log_scan_bytes;
                dense[base + 2] = profile.ewma_log_bytes_per_row;
                dense[base + 3] = profile.ewma_mem_multiplier_log;
                dense[base + 4] = profile.ewma_spill_rate;
            } else {
                dense[base] = 0.0;
                dense[base + 1] = 0.0;
                dense[base + 2] = DEFAULT_LOG_BYTES_PER_ROW;
                dense[base + 3] = 0.0;
                dense[base + 4] = 0.0;
            }
        }

        let known_table_ratio = if active_tables > 0 {
            known_tables as f64 / active_tables as f64
        } else {
            0.0
        };

        (dense, known_table_ratio)
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn export_profiles(&self) -> Vec<(String, TableProfile)> {
        self.profiles
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    pub fn import_profiles(&self, profiles: Vec<(String, TableProfile)>) {
        for (table_id, profile) in profiles {
            self.profiles.insert(table_id, profile);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::decoder::parse_feature_string;

    #[test]
    fn test_export_and_import_profiles_preserves_catalog_state() {
        let catalog1 = TableCatalog::new();
        assert!(catalog1.is_empty());
        let tables = ["0".to_string(), "1001".to_string(), "2002".to_string()];
        for _ in 0..6 {
            catalog1.record_execution(&tables, 20_000_000, 100_000, 80_000_000, 15.0, true);
        }
        assert_eq!(catalog1.len(), 2);

        let exported = catalog1.export_profiles();
        let catalog2 = TableCatalog::new();
        catalog2.import_profiles(exported);
        assert_eq!(catalog2.len(), 2);

        let plan = parse_feature_string(
            "tables=[0,1001,2002]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let (dense1, ratio1) = catalog1.build_dense_features(&plan);
        let (dense2, ratio2) = catalog2.build_dense_features(&plan);
        assert_eq!(dense1, dense2);
        assert!((ratio1 - 1.0).abs() < 1e-6);
        assert_eq!(ratio1, ratio2);
    }
}
