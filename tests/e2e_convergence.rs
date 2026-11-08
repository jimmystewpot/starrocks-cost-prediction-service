use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use starrocks_cost_prediction_service::{
    api::{AppState, build_router},
    config::AppConfig,
    model::ensemble::{ModelStatusReport, RetrainReport},
    types::PredictionOutput,
};
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn test_starrocks_wire_protocol_and_circuit_breaker() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let cfg = AppConfig {
        db_path: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let state = Arc::new(AppState::new(cfg).unwrap());
    let app = build_router(state.clone());

    // 1. GET /health_check -> 200 OK
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health_check")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // 2. POST /predict_csv with valid StarRocks CSV -> 200 OK and parseable double/long
    let csv = "tables_0,tables_1,tables_2,env_0,env_1,env_2,var_0,operators_0,operators_1,operators_2,operators_3,operators_4,operators_5,operators_6,operators_7,operators_8,operators_9,operators_10\n0,0,1001,3,16,24,8,39,1,15,100,18,16,0,12,3,2,2";
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict_csv")
                .header("content-type", "text/csv")
                .body(Body::from(csv))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap().trim();
    let predicted: f64 = body_str
        .parse()
        .expect("StarRocks parses Double.parseDouble");
    assert!(predicted >= 1_048_576.0);

    // 3. Malformed CSV still returns 200 OK (Zero-HTTP-500 policy)
    let bad_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict_csv")
                .body(Body::from("corrupt,csv"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bad_res.status(), StatusCode::OK);
    let fallback_bytes = bad_res.into_body().collect().await.unwrap().to_bytes();
    let fallback_str = std::str::from_utf8(&fallback_bytes).unwrap().trim();
    assert_eq!(fallback_str, "67108864");

    // 4. Trip circuit breaker -> /health_check returns 503 Service Unavailable
    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/circuit-breaker")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"tripped":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let hc_tripped = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health_check")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(hc_tripped.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn test_end_to_end_ingest_retrain_and_multi_target_prediction() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let cfg = AppConfig {
        db_path: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let state = Arc::new(AppState::new(cfg).unwrap());
    let app = build_router(state.clone());

    // 1. Ingest 60 fe.features.log lines via POST /v1/ingest/features
    let mut lines = String::new();
    for i in 0..60 {
        let rows_log = 12 + (i % 6);
        let mem_bytes = 60_000_000 + (rows_log as u64) * 10_000_000;
        let scan_bytes = 25_000_000 + (rows_log as u64) * 2_000_000;
        let return_rows = 500 + (rows_log as u64) * 100;
        let cpu_ns = 300_000_000 + (rows_log as u64) * 20_000_000;
        lines.push_str(&format!(
            "2026-10-28 21:00:01,112 INFO (qe|99) [AuditLogBuilder.java:185] digest=d{i}|cpuCostNs={cpu_ns}|memCostBytes={mem_bytes}|scanBytes={scan_bytes}|scanRows=150000|returnRows={return_rows}|spilledBytes=0|time=80|state=EOF|catalog=default_catalog|database=db1|tables=[0,0,1001]|env=[3,16,24]|var=[8]|operators=[39,1,{rows_log},100,18,16,0,12,3,2,2]\n"
        ));
    }

    let ingest_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/features")
                .header("content-type", "text/plain")
                .body(Body::from(lines))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ingest_res.status(), StatusCode::OK);
    let ingest_body = ingest_res.into_body().collect().await.unwrap().to_bytes();
    let ingest_json: serde_json::Value = serde_json::from_slice(&ingest_body).unwrap();
    assert_eq!(ingest_json["ingested"], 60);
    assert_eq!(ingest_json["errors"], 0);

    // 2. Trigger POST /v1/admin/retrain
    let retrain_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/retrain")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(retrain_res.status(), StatusCode::OK);
    let retrain_body = retrain_res.into_body().collect().await.unwrap().to_bytes();
    let retrain_report: RetrainReport = serde_json::from_slice(&retrain_body).unwrap();
    assert!(retrain_report.promoted);
    assert_eq!(retrain_report.model_version, 1);
    assert_eq!(retrain_report.training_samples, 60);

    // 3. Verify GET /v1/models/status reports gbdt_trained: true
    let status_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(status_res.status(), StatusCode::OK);
    let status_body = status_res.into_body().collect().await.unwrap().to_bytes();
    let status_report: ModelStatusReport = serde_json::from_slice(&status_body).unwrap();
    assert!(status_report.gbdt_trained);
    assert_eq!(status_report.model_version, 1);
    assert_eq!(status_report.ingested_samples, 60);
    assert_eq!(status_report.known_tables, 1);

    // 4. Verify POST /v1/predict returns accurate multi-target predictions
    let predict_req = serde_json::json!({
        "feature_string": "tables=[0,0,1001]|env=[3,16,24]|var=[8]|operators=[36,1,15,100,18,16,0,15,1,0,39,1,15,100,18,16,0,12,3,2,2]"
    });
    let pred_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(predict_req.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(pred_res.status(), StatusCode::OK);
    let pred_body = pred_res.into_body().collect().await.unwrap().to_bytes();
    let pred: PredictionOutput = serde_json::from_slice(&pred_body).unwrap();
    assert_eq!(pred.dominant_tier, "gbdt_tier2");
    assert!(pred.predicted_mem_bytes_p90 >= pred.predicted_mem_bytes_p50);
    assert!(pred.predicted_mem_bytes_p90 > 50_000_000);
    assert!(pred.predicted_scan_bytes > 10_000_000);
    assert!(pred.predicted_return_rows > 100);
    assert!(pred.predicted_cpu_cost_ns > 10_000_000);

    // 5. Verify POST /predict_csv + POST /v1/ingest/audit watermark correlation
    let csv = "tables_0,tables_1,tables_2,env_0,env_1,env_2,var_0,operators_0,operators_1,operators_2,operators_3,operators_4,operators_5,operators_6,operators_7,operators_8,operators_9,operators_10\n0,0,1001,3,16,24,8,39,1,15,100,18,16,0,12,3,2,2";
    let csv_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict_csv")
                .body(Body::from(csv))
                .unwrap(),
        )
        .await
        .unwrap();
    let csv_bytes = csv_res.into_body().collect().await.unwrap().to_bytes();
    let wm_mem: u64 = std::str::from_utf8(&csv_bytes)
        .unwrap()
        .trim()
        .parse()
        .unwrap();

    let audit_payload = serde_json::json!({
        "digest": "audit_q1",
        "predictMemBytes": wm_mem,
        "memCostBytes": 210_000_000_u64,
        "scanBytes": 50_000_000_u64,
        "scanRows": 150_000_u64,
        "returnRows": 2000_u64,
        "cpuCostNs": 600_000_000_u64,
        "spilledBytes": 0_u64,
        "queryTime": 90_u64,
        "state": "EOF",
        "tables": ["1001"]
    });
    let audit_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/audit")
                .header("content-type", "application/json")
                .body(Body::from(audit_payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(audit_res.status(), StatusCode::OK);
    let audit_body = audit_res.into_body().collect().await.unwrap().to_bytes();
    let audit_json: serde_json::Value = serde_json::from_slice(&audit_body).unwrap();
    assert_eq!(audit_json["correlated"], 1);

    // 6. Verify POST /v1/ingest/profile and GET /metrics
    let profile_text = "Query:\n  - PeakMemoryUsage: 128.00 MB\n  - Table: 1001\n  - BytesRead: 32.00 MB\n  - RowsRead: 100000\n";
    let prof_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/profile")
                .body(Body::from(profile_text))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(prof_res.status(), StatusCode::OK);

    let metrics_res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(metrics_res.status(), StatusCode::OK);
    let metrics_bytes = metrics_res.into_body().collect().await.unwrap().to_bytes();
    let metrics_str = std::str::from_utf8(&metrics_bytes).unwrap();
    assert!(metrics_str.contains("starrocks_cost_predictor_model_version 1"));
    assert!(metrics_str.contains("starrocks_cost_predictor_predict_csv_requests_total 1"));
}

#[tokio::test]
async fn test_predict_csv_oversized_payload_returns_200_fallback_not_413() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let cfg = AppConfig {
        db_path: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let state = Arc::new(AppState::new(cfg).unwrap());
    let app = build_router(state.clone());

    let mut oversized_csv = String::from(
        "tables_0,tables_1,tables_2,env_0,env_1,env_2,var_0,operators_0,operators_1,operators_2,operators_3,operators_4,operators_5,operators_6,operators_7,operators_8,operators_9,operators_10\n0,0,1001,3,16,24,8,39,1,15,100,18,16,0,12,3,2,2\n",
    );
    oversized_csv.push_str(&" ".repeat(300 * 1024));
    let oversized = oversized_csv.into_bytes();

    // 1. POST /predict_csv with >256 KiB payload MUST return 200 OK with fallback_mem_bytes (67108864), NEVER 413!
    let res_csv = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict_csv")
                .header("content-type", "text/csv")
                .body(Body::from(oversized.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res_csv.status(), StatusCode::OK);
    let body_bytes = res_csv.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap().trim();
    assert_eq!(body_str, "67108864");
    assert_eq!(
        state
            .predict_csv_fallback_total
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // 2. POST /v1/predict with >256 KiB payload MUST return 413 Payload Too Large
    let res_v1 = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(oversized))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res_v1.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn test_ingest_rate_limit_returns_429_when_semaphore_saturated() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let cfg = AppConfig {
        db_path: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let state = Arc::new(AppState::new(cfg).unwrap());
    let app = build_router(state.clone());

    // Exhaust all 4 ingestion permits to simulate 4 concurrent heavy ingestion batches in flight
    let _permits = state
        .ingest_semaphore
        .clone()
        .acquire_many_owned(4)
        .await
        .unwrap();

    // 5th concurrent ingestion request must immediately return 429 Too Many Requests with Retry-After: 1
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/features")
                .header("content-type", "text/plain")
                .body(Body::from("digest=d1|cpuCostNs=100|memCostBytes=2000000|scanBytes=100|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|tables=[0,0,1]|env=[1,8,24]|var=[1]|operators=[39,1,10,10,10,10,0,10,1,1,1]\n"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(res.headers().get("retry-after").unwrap(), "1");

    // Meanwhile, hot-path POST /predict_csv remains completely unaffected and responds 200 OK
    let csv = "tables_0,tables_1,tables_2,env_0,env_1,env_2,var_0,operators_0,operators_1,operators_2,operators_3,operators_4,operators_5,operators_6,operators_7,operators_8,operators_9,operators_10\n0,0,1001,3,16,24,8,39,1,15,100,18,16,0,12,3,2,2";
    let res_predict = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict_csv")
                .body(Body::from(csv))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res_predict.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_predict_endpoints_handle_non_utf8_raw_formats_and_bad_requests() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let cfg = AppConfig {
        db_path: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let state = Arc::new(AppState::new(cfg).unwrap());
    let app = build_router(state.clone());

    // 1. POST /predict_csv with invalid UTF-8 -> 200 OK with "67108864"
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict_csv")
                .header("content-type", "application/octet-stream")
                .body(Body::from(vec![0xFF, 0xFE, 0xFD]))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap().trim();
    assert_eq!(body_str, "67108864");

    // 2. POST /v1/predict with invalid UTF-8 -> 400 Bad Request
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/octet-stream")
                .body(Body::from(vec![0xFF, 0xFE, 0xFD]))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    // 3. POST /v1/predict with JSON {"csv": "0,0,1001,3,16,24,8,39,1,15,100,18,16,0,12,3,2,2"} -> 200 OK
    let json_req = serde_json::json!({
        "csv": "0,0,1001,3,16,24,8,39,1,15,100,18,16,0,12,3,2,2"
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(json_req.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let pred: PredictionOutput = serde_json::from_slice(&body_bytes).unwrap();
    assert!(pred.predicted_mem_bytes_p90 > 0);

    // 4. POST /v1/predict with raw pipe string -> 200 OK
    let raw_pipe =
        "tables=[0,0,1001]|env=[3,16,24]|var=[8]|operators=[39,1,15,100,18,16,0,12,3,2,2]";
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "text/plain")
                .body(Body::from(raw_pipe))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let pred: PredictionOutput = serde_json::from_slice(&body_bytes).unwrap();
    assert!(pred.predicted_mem_bytes_p90 > 0);

    // 5. POST /v1/predict with raw CSV string -> 200 OK
    let raw_csv = "0,0,1001,3,16,24,8,39,1,15,100,18,16,0,12,3,2,2";
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "text/csv")
                .body(Body::from(raw_csv))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let pred: PredictionOutput = serde_json::from_slice(&body_bytes).unwrap();
    assert!(pred.predicted_mem_bytes_p90 > 0);

    // 6. POST /v1/predict with empty body "", invalid JSON "{bad", empty JSON "{}",
    // and invalid feature string {"feature_string": "corrupt"} -> all return 400 Bad Request
    let bad_payloads = ["", "{bad", "{}", r#"{"feature_string": "corrupt"}"#];
    for payload in bad_payloads {
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/predict")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::BAD_REQUEST,
            "expected 400 for payload: {payload}"
        );
    }
}

#[tokio::test]
async fn test_ingest_endpoints_envelopes_empty_bodies_and_rate_limits() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let cfg = AppConfig {
        db_path: tmp.path().to_path_buf(),
        min_new_samples_for_retrain: 1,
        ..Default::default()
    };
    let state = Arc::new(AppState::new(cfg).unwrap());
    let app = build_router(state.clone());

    // 1. POST /v1/ingest/features with empty body "" -> 200 OK (ingested: 0, errors: 0)
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/features")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["ingested"], 0);
    assert_eq!(json["errors"], 0);

    // 2. POST /v1/ingest/features with {"lines": ["...", "state=RUNNING...", "", "corrupt_line"]} -> 200 OK (ingested: 1, errors: 1)
    let features_env = serde_json::json!({
        "lines": [
            "digest=d1|cpuCostNs=100|memCostBytes=2000000|scanBytes=100|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|tables=[0,0,1]|env=[1,8,24]|var=[1]|operators=[39,1,10,10,10,10,0,10,1,1,1]",
            "digest=d_run|cpuCostNs=100|memCostBytes=2000000|scanBytes=100|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=RUNNING|tables=[0,0,1]|env=[1,8,24]|var=[1]|operators=[39,1,10,10,10,10,0,10,1,1,1]",
            "",
            "corrupt_line"
        ]
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/features")
                .header("content-type", "application/json")
                .body(Body::from(features_env.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["ingested"], 1);
    assert_eq!(json["errors"], 1);

    // 2b. POST /v1/ingest/features with plain-text multi-line body including empty & corrupt lines
    let plain_features = "digest=d2|cpuCostNs=100|memCostBytes=2000000|scanBytes=100|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|tables=[0,0,1]|env=[1,8,24]|var=[1]|operators=[39,1,10,10,10,10,0,10,1,1,1]\n\ncorrupt_line\n";
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/features")
                .header("content-type", "text/plain")
                .body(Body::from(plain_features))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // 3. POST /v1/ingest/audit with empty body "" -> 200 OK (correlated: 0, unmatched: 0)
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/audit")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["correlated"], 0);
    assert_eq!(json["unmatched"], 0);

    // 4. POST /v1/ingest/audit with:
    //    - AuditIngestPayload::Batch ([...])
    let batch_payload = serde_json::json!([
        { "digest": "batch_event_1" }
    ]);
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/audit")
                .header("content-type", "application/json")
                .body(Body::from(batch_payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["correlated"], 0);
    assert_eq!(json["unmatched"], 1);

    //    - AuditIngestPayload::Envelope ({"events": [...]})
    let envelope_payload = serde_json::json!({
        "events": [
            { "digest": "envelope_event_1" }
        ]
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/audit")
                .header("content-type", "application/json")
                .body(Body::from(envelope_payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["correlated"], 0);
    assert_eq!(json["unmatched"], 1);

    //    - multi-line JSONL fallback, including unmatched events (unmatched: 1)
    let jsonl_payload = "{\"digest\": \"jsonl_event_1\"}\ncorrupt_line\n";
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/audit")
                .header("content-type", "application/x-ndjson")
                .body(Body::from(jsonl_payload))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["correlated"], 0);
    assert_eq!(json["unmatched"], 1);

    // 5. POST /v1/ingest/profile with ProfileEnvelope (without digest) -> 200 OK and increments ingested_samples
    let samples_before = state.engine.status().ingested_samples;
    let profile_env = serde_json::json!({
        "profile": "Query:\n  - PeakMemoryUsage: 64.00 MB\n  - BytesRead: 16.00 MB\n  - RowsRead: 1000\n",
        "feature_string": "tables=[0,0,1001]|env=[3,16,24]|var=[8]|operators=[39,1,15,100,18,16,0,12,3,2,2]"
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/profile")
                .header("content-type", "application/json")
                .body(Body::from(profile_env.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(state.engine.status().ingested_samples, samples_before + 1);

    // 6. POST /v1/ingest/profile with empty/unparseable profile "" -> 400 Bad Request
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/profile")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/profile")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"profile": ""}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    // 7. Hold all 4 state.ingest_semaphore permits and verify both
    //    POST /v1/ingest/audit and POST /v1/ingest/profile return 429 Too Many Requests with retry-after: 1
    let _permits = state
        .ingest_semaphore
        .clone()
        .acquire_many_owned(4)
        .await
        .unwrap();

    let res_audit = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/audit")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"digest":"d"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res_audit.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(res_audit.headers().get("retry-after").unwrap(), "1");

    let res_profile = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest/profile")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"profile":"Query:\n  - PeakMemoryUsage: 1MB\n"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res_profile.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(res_profile.headers().get("retry-after").unwrap(), "1");
}
