use crate::config::AppConfig;
use crate::features::decoder::{parse_feature_string, parse_predict_csv};
use crate::ingest::correlator::InFlightCache;
use crate::ingest::parser::{AuditLogEvent, parse_features_log_line, parse_query_profile};
use crate::model::ensemble::EnsembleEngine;
use crate::storage::sqlite::{SqliteStore, StorageError};
use crate::types::{TelemetrySample, compute_effective_mem_bytes};
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Notify;

const DEFAULT_MAX_SAMPLES_PER_STRATUM: usize = 10_000;
const DEFAULT_INFLIGHT_CAPACITY: usize = 50_000;
const DEFAULT_INFLIGHT_TTL_SECS: u64 = 1_800;

pub struct AppState {
    pub config: AppConfig,
    pub store: Arc<SqliteStore>,
    pub engine: Arc<EnsembleEngine>,
    pub inflight: Arc<InFlightCache>,
    pub retrain_notify: Arc<Notify>,
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
    fn observe_latency_ns(&self, elapsed_ns: u64) {
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
        .route("/predict_csv", post(handle_predict_csv))
        .route("/health_check", get(handle_health_check))
        .route("/v1/predict", post(handle_v1_predict))
        .route("/v1/ingest/features", post(handle_ingest_features))
        .route("/v1/ingest/audit", post(handle_ingest_audit))
        .route("/v1/ingest/profile", post(handle_ingest_profile))
        .route("/v1/models/status", get(handle_models_status))
        .route("/v1/admin/retrain", post(handle_admin_retrain))
        .route(
            "/v1/admin/circuit-breaker",
            post(handle_admin_circuit_breaker),
        )
        .route("/metrics", get(handle_metrics))
        .with_state(state)
}

async fn handle_predict_csv(State(state): State<Arc<AppState>>, body: Bytes) -> impl IntoResponse {
    let start = Instant::now();
    state
        .predict_csv_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let Ok(csv_str) = std::str::from_utf8(&body) else {
        state
            .predict_csv_fallback_total
            .fetch_add(1, Ordering::Relaxed);
        tracing::warn!("predict_csv received non-UTF8 body; returning fallback_mem_bytes");
        state.observe_latency_ns(start.elapsed().as_nanos() as u64);
        return (StatusCode::OK, state.config.fallback_mem_bytes.to_string());
    };

    match parse_predict_csv(csv_str) {
        Ok(plan) => {
            let pred = state.engine.predict(&plan, true);
            state
                .inflight
                .record_prediction(pred.predicted_mem_bytes_p90, plan);
            state.observe_latency_ns(start.elapsed().as_nanos() as u64);
            (StatusCode::OK, pred.predicted_mem_bytes_p90.to_string())
        }
        Err(err) => {
            state
                .predict_csv_fallback_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(error = %err, "predict_csv failed to decode CSV; returning fallback_mem_bytes");
            state.observe_latency_ns(start.elapsed().as_nanos() as u64);
            (StatusCode::OK, state.config.fallback_mem_bytes.to_string())
        }
    }
}

async fn handle_health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.engine.is_circuit_breaker_tripped() {
        (StatusCode::SERVICE_UNAVAILABLE, "unhealthy\n")
    } else {
        (StatusCode::OK, "ok\n")
    }
}

#[derive(Deserialize)]
pub struct PredictRequest {
    #[serde(default)]
    pub feature_string: Option<String>,
    #[serde(default)]
    pub csv: Option<String>,
}

async fn handle_v1_predict(State(state): State<Arc<AppState>>, body: Bytes) -> impl IntoResponse {
    let start = Instant::now();
    state
        .v1_predict_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let body_str = match std::str::from_utf8(&body) {
        Ok(s) => s.trim(),
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": err.to_string() })),
            )
                .into_response();
        }
    };

    let plan_res = if body_str.starts_with('{') {
        match serde_json::from_str::<PredictRequest>(body_str) {
            Ok(req) => {
                if let Some(ref fs) = req.feature_string
                    && !fs.trim().is_empty()
                {
                    parse_feature_string(fs)
                } else if let Some(ref csv) = req.csv
                    && !csv.trim().is_empty()
                {
                    parse_predict_csv(csv)
                } else {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({ "error": "request must provide non-empty 'feature_string' or 'csv'" })),
                    )
                        .into_response();
                }
            }
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": err.to_string() })),
                )
                    .into_response();
            }
        }
    } else if body_str.contains("tables=") && body_str.contains("operators=") {
        parse_feature_string(body_str)
    } else if !body_str.is_empty() {
        parse_predict_csv(body_str)
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "request must provide non-empty 'feature_string' or 'csv'" })),
        )
            .into_response();
    };

    match plan_res {
        Ok(plan) => {
            let pred = state.engine.predict(&plan, false);
            state
                .inflight
                .record_prediction(pred.predicted_mem_bytes_p90, plan);
            state.observe_latency_ns(start.elapsed().as_nanos() as u64);
            (StatusCode::OK, Json(pred)).into_response()
        }
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct FeaturesJsonEnvelope {
    lines: Vec<String>,
}

async fn handle_ingest_features(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> impl IntoResponse {
    let body_str = String::from_utf8_lossy(&body);
    let trimmed = body_str.trim();
    if trimmed.is_empty() {
        return (
            StatusCode::OK,
            Json(serde_json::json!({ "ingested": 0, "errors": 0 })),
        );
    }

    let mut ingested = 0_u64;
    let mut errors = 0_u64;
    let mut notify_retrain = false;

    let process_line =
        |line: &str, ingested: &mut u64, errors: &mut u64, notify_retrain: &mut bool| {
            if line.trim().is_empty() {
                return;
            }
            match parse_features_log_line(line) {
                Ok(Some(sample)) => match state.engine.observe_execution(&sample) {
                    Ok(should_retrain) => {
                        *ingested += 1;
                        if should_retrain {
                            *notify_retrain = true;
                        }
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "failed to persist ingested feature sample");
                        *errors += 1;
                    }
                },
                Ok(None) => {}
                Err(err) => {
                    tracing::debug!(error = %err, "failed to parse features log line");
                    *errors += 1;
                }
            }
        };

    if trimmed.starts_with('{')
        && let Ok(env) = serde_json::from_str::<FeaturesJsonEnvelope>(trimmed)
    {
        for line in &env.lines {
            process_line(line, &mut ingested, &mut errors, &mut notify_retrain);
        }
    } else {
        for line in trimmed.lines() {
            process_line(line, &mut ingested, &mut errors, &mut notify_retrain);
        }
    }

    state
        .ingest_features_total
        .fetch_add(ingested, Ordering::Relaxed);
    if notify_retrain {
        state.retrain_notify.notify_one();
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ingested": ingested,
            "errors": errors,
        })),
    )
}

#[derive(Deserialize)]
#[serde(untagged)]
enum AuditIngestPayload {
    Single(AuditLogEvent),
    Batch(Vec<AuditLogEvent>),
    Envelope { events: Vec<AuditLogEvent> },
}

async fn handle_ingest_audit(State(state): State<Arc<AppState>>, body: Bytes) -> impl IntoResponse {
    let body_str = String::from_utf8_lossy(&body);
    let trimmed = body_str.trim();
    if trimmed.is_empty() {
        return (
            StatusCode::OK,
            Json(serde_json::json!({ "correlated": 0, "unmatched": 0 })),
        );
    }

    let events: Vec<AuditLogEvent> =
        if let Ok(parsed) = serde_json::from_str::<AuditIngestPayload>(trimmed) {
            match parsed {
                AuditIngestPayload::Single(ev) => vec![ev],
                AuditIngestPayload::Batch(list) => list,
                AuditIngestPayload::Envelope { events } => events,
            }
        } else {
            let mut list = Vec::new();
            for line in trimmed.lines() {
                let l = line.trim();
                if !l.is_empty()
                    && let Ok(ev) = serde_json::from_str::<AuditLogEvent>(l)
                {
                    list.push(ev);
                }
            }
            list
        };

    let mut correlated = 0_u64;
    let mut unmatched = 0_u64;
    let mut notify_retrain = false;

    for ev in &events {
        if let Some(sample) = state.inflight.correlate_audit_event(ev) {
            if let Ok(should_retrain) = state.engine.observe_execution(&sample) {
                correlated += 1;
                if should_retrain {
                    notify_retrain = true;
                }
            }
        } else {
            unmatched += 1;
        }
    }

    state
        .ingest_audit_total
        .fetch_add(correlated, Ordering::Relaxed);
    if notify_retrain {
        state.retrain_notify.notify_one();
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "correlated": correlated,
            "unmatched": unmatched,
        })),
    )
}

#[derive(Deserialize)]
struct ProfileEnvelope {
    #[serde(default)]
    digest: Option<String>,
    profile: String,
    #[serde(default)]
    feature_string: Option<String>,
}

async fn handle_ingest_profile(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> impl IntoResponse {
    let body_str = String::from_utf8_lossy(&body);
    let trimmed = body_str.trim();

    let (profile_payload, digest_opt, feature_str_opt) = if trimmed.starts_with('{')
        && let Ok(env) = serde_json::from_str::<ProfileEnvelope>(trimmed)
    {
        (env.profile, env.digest, env.feature_string)
    } else {
        (trimmed.to_string(), None, None)
    };

    match parse_query_profile(&profile_payload) {
        Ok(summary) => {
            state.ingest_profile_total.fetch_add(1, Ordering::Relaxed);

            if let Some(ref fs) = feature_str_opt
                && let Ok(plan_features) = parse_feature_string(fs)
                && let Some(effective_mem_bytes) = compute_effective_mem_bytes(
                    summary.peak_mem_bytes,
                    summary.spilled_bytes,
                    "EOF",
                )
            {
                let timestamp_s = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let sample = TelemetrySample {
                    digest: digest_opt.unwrap_or_else(|| "profile".to_string()),
                    plan_features,
                    effective_mem_bytes,
                    scan_bytes: summary.scan_bytes,
                    scan_rows: summary.scan_rows,
                    return_rows: 0,
                    cpu_cost_ns: 0,
                    spilled_bytes: summary.spilled_bytes,
                    query_time_ms: 0,
                    state: "EOF".to_string(),
                    timestamp_s,
                };
                if let Ok(true) = state.engine.observe_execution(&sample) {
                    state.retrain_notify.notify_one();
                }
            } else if !summary.tables.is_empty() && summary.peak_mem_bytes > 0 {
                let mut tables_arr = ["0".to_string(), "0".to_string(), "0".to_string()];
                for (i, t) in summary.tables.iter().take(3).enumerate() {
                    tables_arr[2 - i] = t.clone();
                }
                state.engine.catalog().record_execution(
                    &tables_arr,
                    summary.scan_bytes,
                    summary.scan_rows,
                    summary.peak_mem_bytes.saturating_add(summary.spilled_bytes),
                    0.0,
                    summary.spilled_bytes > 0,
                );
            }

            (StatusCode::OK, Json(summary)).into_response()
        }
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

async fn handle_models_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.engine.status()))
}

async fn handle_admin_retrain(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.clone();
    match tokio::task::spawn_blocking(move || engine.retrain_and_promote()).await {
        Ok(Ok(report)) => (StatusCode::OK, Json(report)).into_response(),
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Deserialize, Serialize)]
pub struct CircuitBreakerRequest {
    pub tripped: bool,
}

async fn handle_admin_circuit_breaker(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CircuitBreakerRequest>,
) -> impl IntoResponse {
    state.engine.set_circuit_breaker(req.tripped);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "circuit_breaker_tripped": req.tripped })),
    )
}

async fn handle_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let status = state.engine.status();
    let csv_reqs = state.predict_csv_requests_total.load(Ordering::Relaxed);
    let csv_fallbacks = state.predict_csv_fallback_total.load(Ordering::Relaxed);
    let v1_reqs = state.v1_predict_requests_total.load(Ordering::Relaxed);
    let ingest_feat = state.ingest_features_total.load(Ordering::Relaxed);
    let ingest_aud = state.ingest_audit_total.load(Ordering::Relaxed);
    let ingest_prof = state.ingest_profile_total.load(Ordering::Relaxed);
    let lat_sum = state.predict_latency_ns_sum.load(Ordering::Relaxed);
    let lat_cnt = state.predict_latency_count.load(Ordering::Relaxed);
    let b10 = state.predict_latency_bucket_10us.load(Ordering::Relaxed);
    let b25 = state.predict_latency_bucket_25us.load(Ordering::Relaxed);
    let b50 = state.predict_latency_bucket_50us.load(Ordering::Relaxed);
    let b100 = state.predict_latency_bucket_100us.load(Ordering::Relaxed);
    let b250 = state.predict_latency_bucket_250us.load(Ordering::Relaxed);
    let b1000 = state.predict_latency_bucket_1000us.load(Ordering::Relaxed);

    let mut out = String::with_capacity(2048);
    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_predict_csv_requests_total Total POST /predict_csv requests"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_predict_csv_requests_total counter"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_csv_requests_total {csv_reqs}"
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_predict_csv_fallback_total Total POST /predict_csv fallback responses"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_predict_csv_fallback_total counter"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_csv_fallback_total {csv_fallbacks}"
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_v1_predict_requests_total Total POST /v1/predict requests"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_v1_predict_requests_total counter"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_v1_predict_requests_total {v1_reqs}"
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_ingest_features_total Total ingested fe.features.log samples"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_ingest_features_total counter"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_ingest_features_total {ingest_feat}"
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_ingest_audit_total Total correlated audit log events"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_ingest_audit_total counter"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_ingest_audit_total {ingest_aud}"
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_ingest_profile_total Total ingested query profiles"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_ingest_profile_total counter"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_ingest_profile_total {ingest_prof}"
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_model_version Active GBDT ensemble model version"
    );
    let _ = writeln!(out, "# TYPE starrocks_cost_predictor_model_version gauge");
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_model_version {}",
        status.model_version
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_sketch_entries Active Tier-1 sketch cache entries"
    );
    let _ = writeln!(out, "# TYPE starrocks_cost_predictor_sketch_entries gauge");
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_sketch_entries {}",
        status.sketch_entries
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_known_tables Tracked tables in TableCatalog"
    );
    let _ = writeln!(out, "# TYPE starrocks_cost_predictor_known_tables gauge");
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_known_tables {}",
        status.known_tables
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_ingested_samples Total telemetry samples observed by engine"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_ingested_samples gauge"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_ingested_samples {}",
        status.ingested_samples
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_live_q_error_ewma Live log Q-error EWMA"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_live_q_error_ewma gauge"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_live_q_error_ewma {:.6}",
        status.live_q_error_ewma
    );

    let cb_val = u8::from(status.circuit_breaker_tripped);
    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_circuit_breaker_tripped Circuit breaker status (1=tripped, 0=healthy)"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_circuit_breaker_tripped gauge"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_circuit_breaker_tripped {cb_val}"
    );

    let _ = writeln!(
        out,
        "# HELP starrocks_cost_predictor_predict_latency_ns Prediction latency histogram in nanoseconds"
    );
    let _ = writeln!(
        out,
        "# TYPE starrocks_cost_predictor_predict_latency_ns histogram"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_bucket{{le=\"10000\"}} {b10}"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_bucket{{le=\"25000\"}} {b25}"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_bucket{{le=\"50000\"}} {b50}"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_bucket{{le=\"100000\"}} {b100}"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_bucket{{le=\"250000\"}} {b250}"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_bucket{{le=\"1000000\"}} {b1000}"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_bucket{{le=\"+Inf\"}} {lat_cnt}"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_sum {lat_sum}"
    );
    let _ = writeln!(
        out,
        "starrocks_cost_predictor_predict_latency_ns_count {lat_cnt}"
    );

    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
}
