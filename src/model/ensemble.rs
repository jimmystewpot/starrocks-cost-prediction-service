use crate::config::AppConfig;
use crate::features::catalog::{TableCatalog, TableProfile};
use crate::model::gbdt::{GbdtHyperparams, GbdtModelSet, GbdtPrediction};
use crate::model::rls::{OnlineRlsCalibrator, RlsState};
use crate::model::sketch::{PlanSketchCache, PlanSketchEntry};
use crate::storage::sqlite::{SqliteStore, StorageError};
use crate::trainer::{should_promote_challenger, train_and_evaluate_challenger};
use crate::types::{
    CanonicalPlanFeatures, PredictionOutput, TelemetrySample, compute_cluster_max_mem_bytes,
    watermark_mem_bytes,
};
use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const Q_ERROR_EWMA_ALPHA: f64 = 0.05;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HoldoutMetrics {
    pub pinball_p90: f64,
    pub q_error_p50: f64,
    pub q_error_p90: f64,
    pub validation_samples: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrainReport {
    pub promoted: bool,
    pub model_version: u64,
    pub training_samples: usize,
    pub holdout_metrics: HoldoutMetrics,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelStatusReport {
    pub model_version: u64,
    pub gbdt_trained: bool,
    pub sketch_entries: usize,
    pub known_tables: usize,
    pub ingested_samples: u64,
    pub stratum_counts: [usize; 5],
    pub live_q_error_ewma: f64,
    pub circuit_breaker_tripped: bool,
    pub holdout_metrics: Option<HoldoutMetrics>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveGbdtSnapshot {
    pub version: u64,
    pub models: GbdtModelSet,
    pub holdout_metrics: HoldoutMetrics,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub model_version: u64,
    pub gbdt: Option<ActiveGbdtSnapshot>,
    pub sketch_entries: Vec<(u64, PlanSketchEntry)>,
    pub rls_state: RlsState,
    pub table_profiles: Vec<(String, TableProfile)>,
    pub ingested_samples: u64,
    pub live_q_error_ewma: f64,
}

pub struct EnsembleEngine {
    config: AppConfig,
    store: Arc<SqliteStore>,
    catalog: TableCatalog,
    sketch: PlanSketchCache,
    rls: OnlineRlsCalibrator,
    active_gbdt: ArcSwap<Option<ActiveGbdtSnapshot>>,
    model_version: AtomicU64,
    ingested_samples: AtomicU64,
    samples_since_retrain: AtomicUsize,
    live_q_error_ewma_bits: AtomicU64,
    circuit_breaker_tripped: AtomicBool,
}

fn log_to_u64(log_val: f64) -> u64 {
    if !log_val.is_finite() || log_val <= 0.0 {
        return 0;
    }
    let clamped = log_val.min(43.0);
    (clamped.exp() - 1.0).max(0.0).round() as u64
}

impl EnsembleEngine {
    pub fn new(config: AppConfig, store: Arc<SqliteStore>) -> Result<Arc<Self>, StorageError> {
        let catalog = TableCatalog::new();
        let sketch = PlanSketchCache::new(config.max_sketch_entries);
        let mut rls = OnlineRlsCalibrator::new();
        let mut gbdt_opt: Option<ActiveGbdtSnapshot> = None;
        let mut model_version = 0_u64;
        let mut ingested_samples = 0_u64;
        let mut live_q_error_ewma = 0.0_f64;

        if let Some((db_ver, _metrics_json, blob)) = store.load_latest_snapshot_blob()?
            && let Ok(snapshot) = rmp_serde::from_slice::<ModelSnapshot>(&blob)
        {
            catalog.import_profiles(snapshot.table_profiles);
            sketch.import_entries(snapshot.sketch_entries);
            rls = OnlineRlsCalibrator::from_state(snapshot.rls_state);
            gbdt_opt = snapshot.gbdt;
            model_version = db_ver.max(snapshot.model_version);
            ingested_samples = snapshot.ingested_samples;
            if snapshot.live_q_error_ewma.is_finite() {
                live_q_error_ewma = snapshot.live_q_error_ewma;
            }
        }

        Ok(Arc::new(Self {
            config,
            store,
            catalog,
            sketch,
            rls,
            active_gbdt: ArcSwap::from_pointee(gbdt_opt),
            model_version: AtomicU64::new(model_version),
            ingested_samples: AtomicU64::new(ingested_samples),
            samples_since_retrain: AtomicUsize::new(0),
            live_q_error_ewma_bits: AtomicU64::new(live_q_error_ewma.to_bits()),
            circuit_breaker_tripped: AtomicBool::new(false),
        }))
    }

    pub fn predict(&self, plan: &CanonicalPlanFeatures, csv_fast_path: bool) -> PredictionOutput {
        let start = Instant::now();

        // Tier 1: Structural Plan-Signature Sketch
        let t1 = self.sketch.predict(plan);
        let c_1 = t1
            .as_ref()
            .map(|p| p.confidence)
            .unwrap_or(0.0)
            .clamp(0.0, 0.95);

        // Tier 2: Multi-Target Pure-Rust GBDT Ensemble via lock-free ArcSwap load
        let gbdt_guard = self.active_gbdt.load();
        let (t2_opt, c_2, active_ver) = if let Some(ref snap) = **gbdt_guard {
            let (dense, known_table_ratio) = self.catalog.build_dense_features(plan);
            let conf2 = (0.45 + 0.45 * known_table_ratio).clamp(0.0, 0.90);
            let pred = if csv_fast_path {
                let p90 = snap.models.predict_mem_p90(&dense);
                GbdtPrediction {
                    log_mem_p90: p90,
                    log_mem_p50: p90,
                    log_scan_bytes: 0.0,
                    log_return_rows: 0.0,
                    log_cpu_ns: 0.0,
                    spill_prob: 0.0,
                }
            } else {
                snap.models.predict_all(&dense)
            };
            (Some(pred), conf2, snap.version)
        } else {
            (None, 0.0, self.model_version.load(Ordering::Relaxed))
        };

        // Tier 3: Online RLS CBO Calibrator
        let t3 = self.rls.predict(plan);

        // Continuous confidence weights
        let w_1 = c_1;
        let w_2 = ((1.0 - w_1) * c_2).clamp(0.0, 1.0 - w_1);
        let w_3 = (1.0 - w_1 - w_2).max(0.0);

        let (t1_p90, t1_p50, t1_scan, t1_rows, t1_cpu, t1_spill) = match t1 {
            Some(ref p) => (
                p.log_mem_p90,
                p.log_mem_p50,
                p.log_scan_bytes,
                p.log_return_rows,
                p.log_cpu_ns,
                p.spill_prob,
            ),
            None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };

        let (t2_p90, t2_p50, t2_scan, t2_rows, t2_cpu, t2_spill) = match t2_opt {
            Some(ref p) => (
                p.log_mem_p90,
                p.log_mem_p50,
                p.log_scan_bytes,
                p.log_return_rows,
                p.log_cpu_ns,
                p.spill_prob,
            ),
            None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };

        let log_mem_p90 = w_1 * t1_p90 + w_2 * t2_p90 + w_3 * t3.log_mem_p90;
        let log_mem_p50 = w_1 * t1_p50 + w_2 * t2_p50 + w_3 * t3.log_mem_p50;
        let log_scan_bytes = w_1 * t1_scan + w_2 * t2_scan + w_3 * t3.log_scan_bytes;
        let log_return_rows = w_1 * t1_rows + w_2 * t2_rows + w_3 * t3.log_return_rows;
        let log_cpu_ns = w_1 * t1_cpu + w_2 * t2_cpu + w_3 * t3.log_cpu_ns;
        let spill_probability = (w_1 * t1_spill + w_2 * t2_spill).clamp(0.0, 1.0);

        let max_mem = compute_cluster_max_mem_bytes(
            plan.num_be_nodes,
            plan.env_mem_log,
            self.config.max_cluster_mem_multiplier,
        )
        .max(self.config.min_predicted_mem_bytes);

        let raw_p50 = log_to_u64(log_mem_p50).clamp(self.config.min_predicted_mem_bytes, max_mem);
        let raw_p90 = log_to_u64(log_mem_p90)
            .max(raw_p50)
            .clamp(self.config.min_predicted_mem_bytes, max_mem);

        let predicted_mem_bytes_p90 =
            watermark_mem_bytes(raw_p90, plan.plan_hash, self.config.min_predicted_mem_bytes);
        let predicted_mem_bytes_p50 = raw_p50.min(predicted_mem_bytes_p90);

        let dominant_tier = if w_1 >= w_2 && w_1 >= w_3 && w_1 > 0.0 {
            "sketch_tier1"
        } else if w_2 >= w_1 && w_2 >= w_3 && w_2 > 0.0 {
            "gbdt_tier2"
        } else {
            "rls_tier3"
        };

        let confidence = (w_1 + w_2).clamp(0.10, 0.99);

        PredictionOutput {
            predicted_mem_bytes_p90,
            predicted_mem_bytes_p50,
            predicted_scan_bytes: if csv_fast_path {
                0
            } else {
                log_to_u64(log_scan_bytes)
            },
            predicted_return_rows: if csv_fast_path {
                0
            } else {
                log_to_u64(log_return_rows)
            },
            predicted_cpu_cost_ns: if csv_fast_path {
                0
            } else {
                log_to_u64(log_cpu_ns)
            },
            spill_probability,
            confidence,
            tier_weights: [w_1, w_2, w_3],
            dominant_tier,
            model_version: active_ver,
            latency_us: start.elapsed().as_micros() as u64,
        }
    }

    pub fn observe_execution(&self, sample: &TelemetrySample) -> Result<bool, StorageError> {
        // 1. Evaluate current prediction error for live drift detector before updating models
        let pre_pred = self.predict(&sample.plan_features, true);
        let log_pred = (1.0 + pre_pred.predicted_mem_bytes_p90 as f64).ln();
        let log_actual = (1.0 + sample.effective_mem_bytes as f64).ln();
        let log_q_err = (log_pred - log_actual).abs().clamp(0.0, 10.0);
        self.update_q_error_ewma(log_q_err);

        // 2. Update TableCatalog, Tier 1 Sketch, and Tier 3 RLS Calibrator
        let spilled = sample.spilled_bytes > 0;
        let cbo_mem_log = sample.plan_features.plan_features_24[21];
        self.catalog.record_execution(
            &sample.plan_features.tables,
            sample.scan_bytes,
            sample.scan_rows,
            sample.effective_mem_bytes,
            cbo_mem_log,
            spilled,
        );
        self.sketch.update(
            &sample.plan_features,
            sample.effective_mem_bytes,
            sample.scan_bytes,
            sample.return_rows,
            sample.cpu_cost_ns,
            spilled,
        );
        self.rls.update(
            &sample.plan_features,
            sample.effective_mem_bytes,
            sample.scan_bytes,
            sample.return_rows,
            sample.cpu_cost_ns,
        );

        // 3. Build 39-dim dense features and persist sample into SQLite stratified store
        let (dense, _) = self.catalog.build_dense_features(&sample.plan_features);
        self.store
            .insert_sample(sample.plan_features.plan_hash, &dense, sample)?;

        let total_ingested = self.ingested_samples.fetch_add(1, Ordering::Relaxed) + 1;
        let since_retrain = self.samples_since_retrain.fetch_add(1, Ordering::Relaxed) + 1;
        let min_samples = self.config.min_new_samples_for_retrain.max(1);
        let ewma = self.live_q_error_ewma();

        let should_retrain = since_retrain >= min_samples
            || (total_ingested as usize >= min_samples
                && ewma > self.config.drift_q_error_threshold);

        Ok(should_retrain)
    }

    pub fn retrain_and_promote(&self) -> Result<RetrainReport, StorageError> {
        let rows = self.store.load_training_rows()?;
        let training_samples = rows.len();
        let current_ver = self.model_version.load(Ordering::Relaxed);

        let params = GbdtHyperparams::default();
        let Some((challenger_models, challenger_metrics)) =
            train_and_evaluate_challenger(&rows, &params)
        else {
            return Ok(RetrainReport {
                promoted: false,
                model_version: current_ver,
                training_samples,
                holdout_metrics: HoldoutMetrics::default(),
            });
        };

        let champ_guard = self.active_gbdt.load();
        let champ_metrics = (**champ_guard).as_ref().map(|c| &c.holdout_metrics);

        if should_promote_challenger(champ_metrics, &challenger_metrics) {
            let new_version = self.model_version.fetch_add(1, Ordering::SeqCst) + 1;
            let new_active = ActiveGbdtSnapshot {
                version: new_version,
                models: challenger_models,
                holdout_metrics: challenger_metrics.clone(),
            };
            self.active_gbdt.store(Arc::new(Some(new_active)));
            self.samples_since_retrain.store(0, Ordering::Relaxed);
            self.live_q_error_ewma_bits
                .store(0.0_f64.to_bits(), Ordering::Relaxed);

            self.persist_snapshot()?;

            Ok(RetrainReport {
                promoted: true,
                model_version: new_version,
                training_samples,
                holdout_metrics: challenger_metrics,
            })
        } else {
            self.samples_since_retrain.store(0, Ordering::Relaxed);
            Ok(RetrainReport {
                promoted: false,
                model_version: current_ver,
                training_samples,
                holdout_metrics: challenger_metrics,
            })
        }
    }

    pub fn persist_snapshot(&self) -> Result<(), StorageError> {
        let version = self.model_version.load(Ordering::Relaxed);
        let gbdt = (**self.active_gbdt.load()).clone();
        let metrics_json = match gbdt.as_ref() {
            Some(g) => serde_json::to_string(&g.holdout_metrics)
                .map_err(|e| StorageError::Serialization(e.to_string()))?,
            None => "{}".to_string(),
        };

        let snapshot = ModelSnapshot {
            model_version: version,
            gbdt,
            sketch_entries: self.sketch.export_entries(),
            rls_state: self.rls.to_state(),
            table_profiles: self.catalog.export_profiles(),
            ingested_samples: self.ingested_samples.load(Ordering::Relaxed),
            live_q_error_ewma: self.live_q_error_ewma(),
        };

        let blob =
            rmp_serde::to_vec(&snapshot).map_err(|e| StorageError::Serialization(e.to_string()))?;
        self.store
            .save_snapshot_blob(version, &metrics_json, &blob)?;
        Ok(())
    }

    pub fn status(&self) -> ModelStatusReport {
        let gbdt_guard = self.active_gbdt.load();
        let (gbdt_trained, holdout_metrics) = match **gbdt_guard {
            Some(ref s) => (true, Some(s.holdout_metrics.clone())),
            None => (false, None),
        };
        let stratum_counts = self.store.sample_counts_by_stratum().unwrap_or([0; 5]);

        ModelStatusReport {
            model_version: self.model_version.load(Ordering::Relaxed),
            gbdt_trained,
            sketch_entries: self.sketch.len(),
            known_tables: self.catalog.len(),
            ingested_samples: self.ingested_samples.load(Ordering::Relaxed),
            stratum_counts,
            live_q_error_ewma: self.live_q_error_ewma(),
            circuit_breaker_tripped: self.is_circuit_breaker_tripped(),
            holdout_metrics,
        }
    }

    pub fn set_circuit_breaker(&self, tripped: bool) {
        self.circuit_breaker_tripped
            .store(tripped, Ordering::Relaxed);
    }

    pub fn is_circuit_breaker_tripped(&self) -> bool {
        self.circuit_breaker_tripped.load(Ordering::Relaxed)
    }

    pub fn catalog(&self) -> &TableCatalog {
        &self.catalog
    }

    fn live_q_error_ewma(&self) -> f64 {
        f64::from_bits(self.live_q_error_ewma_bits.load(Ordering::Relaxed))
    }

    fn update_q_error_ewma(&self, log_q_err: f64) {
        let prev = self.live_q_error_ewma();
        let next = if prev == 0.0 {
            log_q_err
        } else {
            (1.0 - Q_ERROR_EWMA_ALPHA) * prev + Q_ERROR_EWMA_ALPHA * log_q_err
        };
        self.live_q_error_ewma_bits
            .store(next.to_bits(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;
    use crate::features::decoder::parse_feature_string;

    #[test]
    fn test_ensemble_transitions_from_tier3_cold_start_to_tier1_and_tier2_gbdt() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let cfg = AppConfig {
            db_path: tmp.path().to_path_buf(),
            min_new_samples_for_retrain: 30,
            ..Default::default()
        };
        let store = Arc::new(SqliteStore::open(&cfg.db_path, 1000).unwrap());
        let engine = EnsembleEngine::new(cfg, store).unwrap();

        let plan_a = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let pred_cold = engine.predict(&plan_a, false);
        assert_eq!(pred_cold.dominant_tier, "rls_tier3");

        for i in 0..40 {
            let rows_log = 12 + (i % 6);
            let pipe = format!(
                "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,{},100,16,14,0,12,2,1,1]",
                rows_log
            );
            let pf = parse_feature_string(&pipe).unwrap();
            let sample = TelemetrySample {
                digest: format!("d{}", i),
                plan_features: pf,
                effective_mem_bytes: 50_000_000 + (rows_log as u64) * 10_000_000,
                scan_bytes: 20_000_000,
                scan_rows: 100_000,
                return_rows: 1_000,
                cpu_cost_ns: 200_000_000,
                spilled_bytes: 0,
                query_time_ms: 50,
                state: "EOF".to_string(),
                timestamp_s: 1700000000 + i as u64,
            };
            engine.observe_execution(&sample).unwrap();
        }
        let report = engine.retrain_and_promote().unwrap();
        assert!(report.promoted);
        assert_eq!(engine.status().model_version, 1);

        // Seen plan_a is now dominated by Tier 1 Sketch
        let pred_tier1 = engine.predict(&plan_a, false);
        assert_eq!(pred_tier1.dominant_tier, "sketch_tier1");

        // Unseen plan on known table 101 is dominated by Tier 2 GBDT
        let unseen_plan_on_101 = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[36,1,15,100,18,16,0,15,1,0,39,1,15,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let pred_gbdt = engine.predict(&unseen_plan_on_101, false);
        assert_eq!(pred_gbdt.dominant_tier, "gbdt_tier2");
        assert!(pred_gbdt.predicted_mem_bytes_p90 >= pred_gbdt.predicted_mem_bytes_p50);
    }

    #[test]
    fn test_ensemble_snapshot_persistence_and_hydration_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let cfg = AppConfig {
            db_path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let store = Arc::new(SqliteStore::open(&cfg.db_path, 1000).unwrap());
        let engine = EnsembleEngine::new(cfg.clone(), store.clone()).unwrap();

        for i in 0..30 {
            let rows_log = 10 + (i % 5);
            let pipe = format!(
                "tables=[0,0,202]|env=[3,16,24]|var=[8]|operators=[39,1,{},100,16,14,0,12,2,1,1]",
                rows_log
            );
            let pf = parse_feature_string(&pipe).unwrap();
            let sample = TelemetrySample {
                digest: format!("s{}", i),
                plan_features: pf,
                effective_mem_bytes: 80_000_000 + (rows_log as u64) * 5_000_000,
                scan_bytes: 15_000_000,
                scan_rows: 50_000,
                return_rows: 500,
                cpu_cost_ns: 100_000_000,
                spilled_bytes: 0,
                query_time_ms: 30,
                state: "EOF".to_string(),
                timestamp_s: 1700001000 + i as u64,
            };
            engine.observe_execution(&sample).unwrap();
        }
        let report = engine.retrain_and_promote().unwrap();
        assert!(report.promoted);

        let test_plan = parse_feature_string(
            "tables=[0,0,202]|env=[3,16,24]|var=[8]|operators=[36,1,14,100,17,15,0,14,1,0,39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let pred_before = engine.predict(&test_plan, false);

        // Hydrate a brand-new EnsembleEngine from SQLite snapshot
        let restored = EnsembleEngine::new(cfg, store).unwrap();
        let status = restored.status();
        assert_eq!(status.model_version, 1);
        assert!(status.gbdt_trained);
        assert_eq!(status.known_tables, 1);
        assert_eq!(status.sketch_entries, 1);

        let pred_after = restored.predict(&test_plan, false);
        assert_eq!(
            pred_before.predicted_mem_bytes_p90,
            pred_after.predicted_mem_bytes_p90
        );
        assert_eq!(pred_before.dominant_tier, pred_after.dominant_tier);

        // Verify circuit breaker toggle
        assert!(!restored.is_circuit_breaker_tripped());
        restored.set_circuit_breaker(true);
        assert!(restored.is_circuit_breaker_tripped());
        assert!(restored.status().circuit_breaker_tripped);
    }
}
