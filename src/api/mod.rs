pub mod admin;
pub mod ingest;
pub mod predict;

use crate::config::AppConfig;
use crate::ingest::correlator::InFlightCache;
use crate::model::ensemble::EnsembleEngine;
use crate::storage::sqlite::{SqliteStore, StorageError};
use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Notify, Semaphore};

pub use admin::CircuitBreakerRequest;
pub use predict::PredictRequest;

const DEFAULT_MAX_SAMPLES_PER_STRATUM: usize = 10_000;
const DEFAULT_INFLIGHT_CAPACITY: usize = 50_000;
const DEFAULT_INFLIGHT_TTL_SECS: u64 = 1_800;
const PREDICT_BODY_LIMIT_BYTES: usize = 256 * 1024;
const INGEST_BODY_LIMIT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_CONCURRENT_INGEST_REQUESTS: usize = 4;

pub struct AppState {
    pub config: AppConfig,
    pub store: Arc<SqliteStore>,
    pub engine: Arc<EnsembleEngine>,
    pub inflight: Arc<InFlightCache>,
    pub retrain_notify: Arc<Notify>,
    pub ingest_semaphore: Arc<Semaphore>,
    pub predict_csv_requests_total: AtomicU64,
    pub predict_csv_fallback_total: AtomicU64,
    pub v1_predict_requests_total: AtomicU64,
    pub ingest_features_total: AtomicU64,
    pub ingest_audit_total: AtomicU64,
    pub ingest_profile_total: AtomicU64,
    pub predict_latency_ns_sum: AtomicU64,
    pub predict_latency_count: AtomicU64,
    pub predict_latency_bucket_10us: AtomicU64,
    pub predict_latency_bucket_25us: AtomicU64,
    pub predict_latency_bucket_50us: AtomicU64,
    pub predict_latency_bucket_100us: AtomicU64,
    pub predict_latency_bucket_250us: AtomicU64,
    pub predict_latency_bucket_1000us: AtomicU64,
}

impl AppState {
    pub fn new(config: AppConfig) -> Result<Self, StorageError> {
        let store = Arc::new(SqliteStore::open(
            &config.db_path,
            DEFAULT_MAX_SAMPLES_PER_STRATUM,
        )?);
        let engine = EnsembleEngine::new(config.clone(), store.clone())?;
        let inflight = Arc::new(InFlightCache::new(
            DEFAULT_INFLIGHT_CAPACITY,
            DEFAULT_INFLIGHT_TTL_SECS,
        ));

        Ok(Self {
            config,
            store,
            engine,
            inflight,
            retrain_notify: Arc::new(Notify::new()),
            ingest_semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_INGEST_REQUESTS)),
            predict_csv_requests_total: AtomicU64::new(0),
            predict_csv_fallback_total: AtomicU64::new(0),
            v1_predict_requests_total: AtomicU64::new(0),
            ingest_features_total: AtomicU64::new(0),
            ingest_audit_total: AtomicU64::new(0),
            ingest_profile_total: AtomicU64::new(0),
            predict_latency_ns_sum: AtomicU64::new(0),
            predict_latency_count: AtomicU64::new(0),
            predict_latency_bucket_10us: AtomicU64::new(0),
            predict_latency_bucket_25us: AtomicU64::new(0),
            predict_latency_bucket_50us: AtomicU64::new(0),
            predict_latency_bucket_100us: AtomicU64::new(0),
            predict_latency_bucket_250us: AtomicU64::new(0),
            predict_latency_bucket_1000us: AtomicU64::new(0),
        })
    }

    #[inline]
    pub(crate) fn observe_latency_ns(&self, elapsed_ns: u64) {
        self.predict_latency_ns_sum
            .fetch_add(elapsed_ns, Ordering::Relaxed);
        self.predict_latency_count.fetch_add(1, Ordering::Relaxed);
        if elapsed_ns <= 10_000 {
            self.predict_latency_bucket_10us
                .fetch_add(1, Ordering::Relaxed);
        }
        if elapsed_ns <= 25_000 {
            self.predict_latency_bucket_25us
                .fetch_add(1, Ordering::Relaxed);
        }
        if elapsed_ns <= 50_000 {
            self.predict_latency_bucket_50us
                .fetch_add(1, Ordering::Relaxed);
        }
        if elapsed_ns <= 100_000 {
            self.predict_latency_bucket_100us
                .fetch_add(1, Ordering::Relaxed);
        }
        if elapsed_ns <= 250_000 {
            self.predict_latency_bucket_250us
                .fetch_add(1, Ordering::Relaxed);
        }
        if elapsed_ns <= 1_000_000 {
            self.predict_latency_bucket_1000us
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/predict_csv",
            post(predict::handle_predict_csv)
                .layer(DefaultBodyLimit::max(PREDICT_BODY_LIMIT_BYTES)),
        )
        .route("/health_check", get(admin::handle_health_check))
        .route(
            "/v1/predict",
            post(predict::handle_v1_predict).layer(DefaultBodyLimit::max(PREDICT_BODY_LIMIT_BYTES)),
        )
        .route(
            "/v1/ingest/features",
            post(ingest::handle_ingest_features)
                .layer(DefaultBodyLimit::max(INGEST_BODY_LIMIT_BYTES)),
        )
        .route(
            "/v1/ingest/audit",
            post(ingest::handle_ingest_audit).layer(DefaultBodyLimit::max(INGEST_BODY_LIMIT_BYTES)),
        )
        .route(
            "/v1/ingest/profile",
            post(ingest::handle_ingest_profile)
                .layer(DefaultBodyLimit::max(INGEST_BODY_LIMIT_BYTES)),
        )
        .route("/v1/models/status", get(admin::handle_models_status))
        .route("/v1/admin/retrain", post(admin::handle_admin_retrain))
        .route(
            "/v1/admin/circuit-breaker",
            post(admin::handle_admin_circuit_breaker),
        )
        .route("/metrics", get(admin::handle_metrics))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observe_latency_ns_populates_all_histogram_buckets() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let cfg = AppConfig {
            db_path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let state = AppState::new(cfg).unwrap();

        let latencies = [5_000, 20_000, 40_000, 80_000, 200_000, 500_000, 2_000_000];
        for lat in latencies {
            state.observe_latency_ns(lat);
        }

        assert_eq!(state.predict_latency_bucket_10us.load(Ordering::Relaxed), 1);
        assert_eq!(state.predict_latency_bucket_25us.load(Ordering::Relaxed), 2);
        assert_eq!(state.predict_latency_bucket_50us.load(Ordering::Relaxed), 3);
        assert_eq!(
            state.predict_latency_bucket_100us.load(Ordering::Relaxed),
            4
        );
        assert_eq!(
            state.predict_latency_bucket_250us.load(Ordering::Relaxed),
            5
        );
        assert_eq!(
            state.predict_latency_bucket_1000us.load(Ordering::Relaxed),
            6
        );
        assert_eq!(state.predict_latency_count.load(Ordering::Relaxed), 7);
        assert_eq!(
            state.predict_latency_ns_sum.load(Ordering::Relaxed),
            2_845_000
        );
    }
}
