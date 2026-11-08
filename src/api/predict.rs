use crate::api::AppState;
use crate::features::decoder::{parse_feature_string, parse_predict_csv};
use axum::{
    Json,
    body::Bytes,
    extract::{State, rejection::BytesRejection},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub async fn handle_predict_csv(
    State(state): State<Arc<AppState>>,
    body_res: Result<Bytes, BytesRejection>,
) -> impl IntoResponse {
    let start = Instant::now();
    state
        .predict_csv_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let body = match body_res {
        Ok(b) => b,
        Err(rejection) => {
            state
                .predict_csv_fallback_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                error = %rejection,
                "predict_csv body rejected; returning fallback_mem_bytes"
            );
            state.observe_latency_ns(start.elapsed().as_nanos() as u64);
            return (StatusCode::OK, state.config.fallback_mem_bytes.to_string());
        }
    };

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

#[derive(Deserialize)]
pub struct PredictRequest {
    #[serde(default)]
    pub feature_string: Option<String>,
    #[serde(default)]
    pub csv: Option<String>,
}

pub async fn handle_v1_predict(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> impl IntoResponse {
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
