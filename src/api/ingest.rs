use crate::api::AppState;
use crate::features::decoder::parse_feature_string;
use crate::ingest::parser::{AuditLogEvent, parse_features_log_line, parse_query_profile};
use crate::types::{TelemetrySample, compute_effective_mem_bytes};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::{StatusCode, header},
    response::IntoResponse,
};
use serde::Deserialize;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Deserialize)]
struct FeaturesJsonEnvelope {
    lines: Vec<String>,
}

pub async fn handle_ingest_features(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> impl IntoResponse {
    let Ok(_permit) = state.ingest_semaphore.clone().try_acquire_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "1")],
            Json(serde_json::json!({ "error": "ingestion rate limit exceeded; retry shortly" })),
        )
            .into_response();
    };

    let join_res = tokio::task::spawn_blocking(move || {
        let _permit = _permit;
        let body_str = String::from_utf8_lossy(&body);
        let trimmed = body_str.trim();
        if trimmed.is_empty() {
            return (0_u64, 0_u64);
        }

        let mut samples = Vec::new();
        let mut errors = 0_u64;

        let mut collect_line = |line: &str| {
            if line.trim().is_empty() {
                return;
            }
            match parse_features_log_line(line) {
                Ok(Some(sample)) => samples.push(sample),
                Ok(None) => {}
                Err(err) => {
                    tracing::debug!(error = %err, "failed to parse features log line");
                    errors += 1;
                }
            }
        };

        if trimmed.starts_with('{')
            && let Ok(env) = serde_json::from_str::<FeaturesJsonEnvelope>(trimmed)
        {
            for line in &env.lines {
                collect_line(line);
            }
        } else {
            for line in trimmed.lines() {
                collect_line(line);
            }
        }

        let ingested = match state.engine.observe_executions_batch(&samples) {
            Ok(should_retrain) => {
                if should_retrain {
                    state.retrain_notify.notify_one();
                }
                samples.len() as u64
            }
            Err(err) => {
                tracing::warn!(error = %err, "failed to persist ingested feature batch");
                errors += samples.len() as u64;
                0
            }
        };

        state
            .ingest_features_total
            .fetch_add(ingested, Ordering::Relaxed);

        (ingested, errors)
    })
    .await;

    match join_res {
        Ok((ingested, errors)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ingested": ingested,
                "errors": errors,
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum AuditIngestPayload {
    Single(AuditLogEvent),
    Batch(Vec<AuditLogEvent>),
    Envelope { events: Vec<AuditLogEvent> },
}

pub async fn handle_ingest_audit(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> impl IntoResponse {
    let Ok(_permit) = state.ingest_semaphore.clone().try_acquire_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "1")],
            Json(serde_json::json!({ "error": "ingestion rate limit exceeded; retry shortly" })),
        )
            .into_response();
    };

    let join_res = tokio::task::spawn_blocking(move || {
        let _permit = _permit;
        let body_str = String::from_utf8_lossy(&body);
        let trimmed = body_str.trim();
        if trimmed.is_empty() {
            return (0_u64, 0_u64);
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

        let mut correlated_samples = Vec::new();
        let mut unmatched = 0_u64;

        for ev in &events {
            if let Some(sample) = state.inflight.correlate_audit_event(ev) {
                correlated_samples.push(sample);
            } else {
                unmatched += 1;
            }
        }

        let correlated = match state.engine.observe_executions_batch(&correlated_samples) {
            Ok(should_retrain) => {
                if should_retrain {
                    state.retrain_notify.notify_one();
                }
                correlated_samples.len() as u64
            }
            Err(err) => {
                tracing::warn!(error = %err, "failed to persist correlated audit batch");
                0
            }
        };

        state
            .ingest_audit_total
            .fetch_add(correlated, Ordering::Relaxed);

        (correlated, unmatched)
    })
    .await;

    match join_res {
        Ok((correlated, unmatched)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "correlated": correlated,
                "unmatched": unmatched,
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct ProfileEnvelope {
    #[serde(default)]
    digest: Option<String>,
    profile: String,
    #[serde(default)]
    feature_string: Option<String>,
}

pub async fn handle_ingest_profile(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> impl IntoResponse {
    let Ok(_permit) = state.ingest_semaphore.clone().try_acquire_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "1")],
            Json(serde_json::json!({ "error": "ingestion rate limit exceeded; retry shortly" })),
        )
            .into_response();
    };

    let join_res = tokio::task::spawn_blocking(move || {
        let _permit = _permit;
        let body_str = String::from_utf8_lossy(&body);
        let trimmed = body_str.trim();

        let (profile_payload, digest_opt, feature_str_opt) = if trimmed.starts_with('{')
            && let Ok(env) = serde_json::from_str::<ProfileEnvelope>(trimmed)
        {
            (env.profile, env.digest, env.feature_string)
        } else {
            (trimmed.to_string(), None, None)
        };

        parse_query_profile(&profile_payload).inspect(|summary| {
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
                if let Ok(true) = state
                    .engine
                    .observe_executions_batch(std::slice::from_ref(&sample))
                {
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
        })
    })
    .await;

    match join_res {
        Ok(Ok(summary)) => (StatusCode::OK, Json(summary)).into_response(),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
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
