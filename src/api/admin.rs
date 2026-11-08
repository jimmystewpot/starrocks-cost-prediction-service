use crate::api::AppState;
use axum::{
    Json,
    extract::State,
    http::{StatusCode, header},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::Ordering;

pub async fn handle_health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.engine.is_circuit_breaker_tripped() {
        (StatusCode::SERVICE_UNAVAILABLE, "unhealthy\n")
    } else {
        (StatusCode::OK, "ok\n")
    }
}

pub async fn handle_models_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.engine.status()))
}

pub async fn handle_admin_retrain(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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

pub async fn handle_admin_circuit_breaker(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CircuitBreakerRequest>,
) -> impl IntoResponse {
    state.engine.set_circuit_breaker(req.tripped);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "circuit_breaker_tripped": req.tripped })),
    )
}

pub async fn handle_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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
