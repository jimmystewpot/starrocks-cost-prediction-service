# StarRocks Cost Prediction Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-learning, sub-millisecond Rust cost prediction service for StarRocks (`POST /predict_csv`, `GET /health_check`, `POST /v1/predict`, and multi-source async telemetry ingestion) backed by a 3-Tier ML Ensemble (Plan-Signature EWMA Sketch + Pure-Rust Multi-Target GBDT + Online RLS CBO Calibrator), embedded SQLite WAL storage, CI/CD workflows, Dependabot automation, and comprehensive root documentation (`README.md` and `AGENTS.md`).

**Architecture:** The service separates the synchronous CPU-bound inference plane (`axum` 0.8 + `ArcSwap` 1.9 + `DashMap` 6.2, `p99 < 250µs`, `TCP_NODELAY`) from the asynchronous ingestion and background training plane (`rayon` 1.12 + `rusqlite` 0.41 bundled WAL). Incoming StarRocks CSV and `fe.features.log` payloads are decoded by a version-resilient self-synchronizing segment parser into a 39-dimensional dense feature vector (`DenseFeatureVector`). Predictions blend Tier 1 (Plan-Signature EWMA Sketch), Tier 2 (Pure-Rust GBDT ensemble predicting `mem_p90`, `mem_p50`, `scan_bytes`, `return_rows`, `cpu_ns`, and `spill_prob`), and Tier 3 (Online RLS CBO Calibrator) with safety clamping and 10-bit watermark correlation for audit logs.

**Tech Stack:** Rust (Edition 2024, `rustc 1.99+`), `axum = "0.8.9"`, `tokio = "1.54.1"`, `arc-swap = "1.9.3"`, `dashmap = "6.2.1"`, `xxhash-rust = "0.8.19"` (`xxh3`), `rusqlite = "0.41.0"` (`bundled`), `rayon = "1.12.0"`, `serde = "1.0.229"`, `serde_json = "1.0.154"`, `rmp-serde = "1.3.1"`, `thiserror = "2.0.21"`, `tracing = "0.1.44"`, `tracing-subscriber = "0.3.23"`, `tower = "0.5.3"`, `http-body-util = "0.1.5"`, `tempfile = "3.27.0"`. GitHub Actions (`actions/checkout@v7`, `dtolnay/rust-toolchain@stable`, `Swatinem/rust-cache@v2`) & Dependabot v2 (`cargo` + `github-actions`).

**Spec:** `docs/superpowers/specs/2026-10-28-starrocks-cost-prediction-service-design.md`

## Global Constraints

- **Rust Edition & Latest Stable Crates:** Rust `edition = "2024"` in `Cargo.toml` with latest stable versions of all dependencies (`axum 0.8.9`, `tokio 1.54.1`, `arc-swap 1.9.3`, `dashmap 6.2.1`, `xxhash-rust 0.8.19`, `rusqlite 0.41.0`, `rayon 1.12.0`, `serde 1.0.229`, `serde_json 1.0.154`, `rmp-serde 1.3.1`, `thiserror 2.0.21`, `tracing 0.1.44`, `tracing-subscriber 0.3.23`, `tower 0.5.3`, `http-body-util 0.1.5`, `tempfile 3.27.0`).
- **Formatting, Linting & CI Gate:** All code must pass `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-targets`.
- **Zero-HTTP-500 Policy on `/predict_csv`:** `POST /predict_csv` must always return HTTP `200 OK` with a parseable numeric string (falling back to `config.fallback_mem_bytes` on malformed CSV) so StarRocks FE logs are never flooded with stack traces.
- **Dense Feature Dimensionality:** `DENSE_FEATURE_DIM = 39` (`24` synthesized plan/operator/environment features + `3 * 5 = 15` target-encoded table telemetry features).
- **Safety Clamping & Integer Truncation Compensation:** `env_2` is `(long) Math.log1p(memCapacityOfBE)`, so per-BE memory capacity upper bound is `exp((env_2 + 1).min(40) as f64)`. Final predicted memory bytes are clamped to `[config.min_predicted_mem_bytes (default 1_048_576), max_cluster_mem_bytes]`.
- **10-Bit In-Flight Audit Correlation Watermark:** When `predicted_mem_bytes >= 1_048_576`, encode `(plan_hash & 0x3FF)` into the lowest 10 bits: `(predicted_mem_bytes & !0x3FF) | (plan_hash & 0x3FF)` so post-execution `starrocks_audit_tbl__` records (`predictMemBytes`) correlate back to their in-flight `CanonicalPlanFeatures`.

## Review Focus

1. **StarRocks Version Skew / Unknown `OperatorType` Ordinals:** If a newer StarRocks version adds physical operators (shifting column counts in `operators_0..N`), the self-synchronizing segment walker in `src/features/decoder.rs` must still segment the operator vector without panicking or misaligning subsequent operators.
2. **Truncated `env_2` Log-Memory Upper Bound:** When `env_2 = 24` (from a 64 GiB BE node where `ln(68_719_476_736) = 24.953` was truncated to `24` by Java `(long)`), clamping must use `exp(env_2 + 1)` (`~72 GiB`) rather than `exp(env_2)` (`~26.4 GiB`) so valid 50 GiB predictions are not artificially clipped.
3. **Log4j Prefix & Malformed Lines in `fe.features.log`:** Production `fe.features.log` lines include a timestamp/level prefix (e.g., `2026-10-28 12:00:00.123 INFO (qe|1) [AuditLogBuilder.java:180] digest=...`) and may interleave blank or truncated lines; the parser must strip everything before `digest=` and skip corrupt lines without aborting batch ingestion.
4. **OOM-Killed Query Censoring (`state=ERR`):** Queries that fail with `state=ERR` and high memory usage (`memCostBytes > 0`) must apply the $1.5\times$ censored penalty (`max(mem + spill, mem * 1.5)`), whereas syntax/cancelled errors (`state=ERR` with `memCostBytes == 0`) must be ignored so they do not poison the training set with zero-byte labels.
5. **Singular Covariance Matrix in Online RLS (`cbo_est_total_mem_log == 0`):** When multiple queries have identical or zero features, RLS covariance matrix updates must remain finite (no `NaN`/`Inf` weights) via ridge regularization and diagonal clamping `P[i][i].clamp(1e-6, 1e4)`.

---

## Dependency Graph & Parallel Execution Waves

Tasks are structured into **4 Waves** to maximize parallel subagent execution after Task 1 initializes the workspace and shared types:

- **Wave 1 (Foundation — Sequential):**
  - **Task 1:** Cargo Workspace (Edition 2024, Latest Crates), Config, Core Domain Types & Shared Interfaces (`Cargo.toml`, `src/config.rs`, `src/types.rs`, `src/lib.rs`)
- **Wave 2 (Core Subsystems — 5 Parallel Subagents in Isolated Worktrees/Branches):**
  - **Task 2 (Subagent A):** StarRocks Feature Decoder, Self-Synchronizing Segment Walker & Table Telemetry Catalog (`src/features/`)
  - **Task 3 (Subagent B):** Tier 1 Plan-Signature Sketch & Tier 3 Online RLS CBO Calibrator (`src/model/sketch.rs`, `src/model/rls.rs`)
  - **Task 4 (Subagent C):** Tier 2 Pure-Rust GBDT Learner (Pinball P90 & Huber P50) (`src/model/gbdt.rs`)
  - **Task 5 (Subagent D):** Multi-Source Telemetry Parsers (`fe.features.log`, Audit JSON, Query Profile) & SQLite WAL Store (`src/ingest/parser.rs`, `src/storage/sqlite.rs`)
  - **Task 6 (Subagent E):** CI/CD GitHub Actions (Tests, `clippy`, `rustfmt`), Dependabot Configuration (`cargo` & `github-actions`), `README.md` & `AGENTS.md` (`.github/workflows/ci.yml`, `.github/dependabot.yml`, `README.md`, `AGENTS.md`)
- **Wave 3 (Ensemble & Orchestration — 2 Parallel Subagents):**
  - **Task 7 (Subagent F):** 3-Tier Ensemble Blender, `ArcSwap` Snapshot Manager, Drift Detector & Background Trainer (`src/model/ensemble.rs`, `src/trainer/mod.rs`)
  - **Task 8 (Subagent G):** In-Flight Prediction Correlator & Resilient File Tailer (`src/ingest/correlator.rs`, `src/ingest/tailer.rs`)
- **Wave 4 (API Server & End-to-End Verification — Sequential):**
  - **Task 9:** Axum 0.8 HTTP Server (`/predict_csv`, `/health_check`, `/v1/predict`, `/v1/ingest/*`, `/v1/admin/*`, `/metrics`), CLI Entrypoint, Full Verification (`fmt`, `clippy`, `test`) & End-to-End Convergence Suite (`src/api/mod.rs`, `src/main.rs`, `tests/e2e_convergence.rs`)

---

### Task 1: Project Foundation, Latest Crates, Config & Shared Domain Types

**Files:**
- Create: `Cargo.toml`
- Create: `src/config.rs`
- Create: `src/types.rs`
- Create: `src/lib.rs`
- Create: `src/main.rs`

**Interfaces:**
- Consumes: none (root foundation task)
- Produces:
  - `crate::config::AppConfig` (`bind_addr: String`, `db_path: PathBuf`, `features_log_path: Option<PathBuf>`, `min_predicted_mem_bytes: u64`, `fallback_mem_bytes: u64`, `max_cluster_mem_multiplier: f64`, `max_sketch_entries: usize`, `min_new_samples_for_retrain: usize`, `retrain_interval_secs: u64`, `drift_q_error_threshold: f64`)
  - `crate::types::DENSE_FEATURE_DIM: usize = 39`
  - `crate::types::DenseFeatureVector = Vec<f64>` (or `[f64; 39]` with serde array support — use `[f64; 39]` with `serde_big_array` or `Vec<f64>` / struct wrapper; Note: in `serde 1.0`, `[T; 39]` exceeds the built-in array trait impl limit of 32 unless serialized as slice/vec or `#[derive(Serialize, Deserialize)]` struct `DenseFeatures(pub [f64; 39])` with custom `Serialize`/`Deserialize` or `pub type DenseFeatureVector = [f64; 39]` serialized via `.as_slice()` in SQLite/MessagePack!)
  - `crate::types::OperatorSegmentKind` (`Scan`, `HashJoin`, `HashAgg`, `Exchange`, `Generic`)
  - `crate::types::OperatorSegment` (`ordinal: i64`, `kind: OperatorSegmentKind`, `count: u64`, `mean_log_rows: f64`, `mean_row_size: f64`, `mean_log_cbo_mem: f64`, `mean_log_cbo_cpu: f64`, `mean_log_limit: f64`, `extra: Vec<i64>`)
  - `crate::types::CanonicalPlanFeatures` (`tables: [String; 3]`, `num_be_nodes: u64`, `avg_cpu_cores: u64`, `env_mem_log: u64`, `pipeline_dop: u64`, `segments: Vec<OperatorSegment>`, `plan_hash: u64`, `plan_features_24: [f64; 24]`)
  - `crate::types::TelemetrySample` (`digest: String`, `plan_features: CanonicalPlanFeatures`, `effective_mem_bytes: u64`, `scan_bytes: u64`, `scan_rows: u64`, `return_rows: u64`, `cpu_cost_ns: u64`, `spilled_bytes: u64`, `query_time_ms: u64`, `state: String`, `timestamp_s: u64`)
  - `crate::types::PredictionOutput` (`predicted_mem_bytes_p90: u64`, `predicted_mem_bytes_p50: u64`, `predicted_scan_bytes: u64`, `predicted_return_rows: u64`, `predicted_cpu_cost_ns: u64`, `spill_probability: f64`, `confidence: f64`, `tier_weights: [f64; 3]`, `dominant_tier: &'static str`, `model_version: u64`, `latency_us: u64`)
  - `crate::types::watermark_mem_bytes(raw_bytes: u64, plan_hash: u64, min_bytes: u64) -> u64`
  - `crate::types::compute_effective_mem_bytes(mem_cost_bytes: u64, spilled_bytes: u64, state: &str) -> Option<u64>`
  - `crate::types::compute_cluster_max_mem_bytes(num_be_nodes: u64, env_mem_log: u64, multiplier: f64) -> u64`
  - `crate::types::memory_stratum(bytes: u64) -> u8`

- [ ] **Step 1: Create `Cargo.toml` with latest stable dependencies and stub modules with failing unit tests in `src/types.rs`**

Write `Cargo.toml`:
```toml
[package]
name = "starrocks-cost-prediction-service"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
description = "High-performance ML query cost prediction service for StarRocks CBO and Query Queue V2"
license = "Apache-2.0"

[dependencies]
axum = "0.8.9"
tokio = { version = "1.54.1", features = ["full"] }
serde = { version = "1.0.229", features = ["derive"] }
serde_json = "1.0.154"
rmp-serde = "1.3.1"
arc-swap = "1.9.3"
dashmap = "6.2.1"
xxhash-rust = { version = "0.8.19", features = ["xxh3"] }
rusqlite = { version = "0.41.0", features = ["bundled"] }
rayon = "1.12.0"
thiserror = "2.0.21"
tracing = "0.1.44"
tracing-subscriber = { version = "0.3.23", features = ["env-filter"] }

[dev-dependencies]
tower = { version = "0.5.3", features = ["util"] }
http-body-util = "0.1.5"
tempfile = "3.27.0"
```

And add unit tests in `src/types.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_mem_bytes_embeds_low_10_bits_of_plan_hash() {
        let plan_hash: u64 = 0xABCD_EF00_0000_02A5; // low 10 bits = 0x2A5 (677)
        let raw_bytes: u64 = 104_857_600; // 100 MiB
        let watermarked = watermark_mem_bytes(raw_bytes, plan_hash, 1_048_576);
        assert_eq!(watermarked & 0x3FF, 0x2A5);
        assert!((watermarked as i64 - raw_bytes as i64).abs() < 1024);
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
        assert!(max_bytes > 68_719_476_736, "max_bytes={} must exceed 64 GiB", max_bytes);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib types::tests`
Expected: FAIL (functions not implemented)

- [ ] **Step 3: Implement `src/config.rs`, `src/types.rs`, `src/lib.rs`, and `src/main.rs`**

Implement `AppConfig::default()`, `watermark_mem_bytes`, `compute_effective_mem_bytes`, `compute_cluster_max_mem_bytes`, `memory_stratum(bytes: u64) -> u8` (`0` for `<10MB`, `1` for `<100MB`, `2` for `<1GB`, `3` for `<10GB`, `4` for `>=10GB`), and the shared structs.

- [ ] **Step 4: Run test and clippy to verify it passes cleanly**

Run: `cargo test --lib types::tests && cargo clippy --all-targets -- -D warnings`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/
git commit -m "feat: initialize Rust 2024 crate with latest dependencies and core domain types"
```

---

### Task 2: StarRocks Feature Decoder, Self-Synchronizing Segment Walker & Table Catalog (Wave 2 — Subagent A)

**Files:**
- Create: `src/features/mod.rs`
- Create: `src/features/decoder.rs`
- Create: `src/features/catalog.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `crate::types::{CanonicalPlanFeatures, DenseFeatureVector, OperatorSegment, OperatorSegmentKind, DENSE_FEATURE_DIM}`
- Produces:
  - `crate::features::decoder::DecodeError` (`#[derive(Debug, thiserror::Error)]`)
  - `crate::features::decoder::parse_predict_csv(csv_body: &str) -> Result<CanonicalPlanFeatures, DecodeError>`
  - `crate::features::decoder::parse_feature_string(feature_str: &str) -> Result<CanonicalPlanFeatures, DecodeError>` (parses `tables=[...]|env=[...]|var=[...]|operators=[...]`)
  - `crate::features::decoder::decode_operator_segments(ops: &[i64]) -> Result<Vec<OperatorSegment>, DecodeError>`
  - `crate::features::catalog::TableProfile` (`#[derive(Clone, Debug, Serialize, Deserialize)]`)
  - `crate::features::catalog::TableCatalog` with:
    - `TableCatalog::new() -> Self`
    - `TableCatalog::record_execution(&self, tables: &[String; 3], scan_bytes: u64, scan_rows: u64, actual_mem_bytes: u64, cbo_mem_log: f64, spilled: bool)`
    - `TableCatalog::build_dense_features(&self, plan: &CanonicalPlanFeatures) -> (DenseFeatureVector, f64)` (returns the 39-element vector and `known_table_ratio` in `0.0..=1.0`)
    - `TableCatalog::len(&self) -> usize`
    - `TableCatalog::export_profiles(&self) -> Vec<(String, TableProfile)>`
    - `TableCatalog::import_profiles(&self, profiles: Vec<(String, TableProfile)>)`

- [ ] **Step 1: Write failing tests for segment decoding, summed-log correction, CSV/pipe parsing, and `TableCatalog` in `src/features/decoder.rs` and `src/features/catalog.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_synchronizing_segment_walker_handles_starrocks_operators() {
        let ops: Vec<i64> = vec![
            34, 1, 10, 100, 14, 12, 0, 2,
            35, 2, 20, 64, 28, 24, 0, 24, 4, 2, 10,
            36, 2, 22, 128, 32, 30, 0, 20, 2, 1,
            38, 0, 0, 0, 0, 0, 0,
            39, 1, 15, 100, 18, 16, 0, 12, 3, 2, 2,
        ];
        let segments = decode_operator_segments(&ops).expect("should segment cleanly");
        assert_eq!(segments.len(), 5);
        assert_eq!(segments[0].kind, OperatorSegmentKind::Exchange);
        assert_eq!(segments[1].kind, OperatorSegmentKind::HashAgg);
        assert!((segments[1].mean_log_cbo_mem - 14.0).abs() < 1e-6);
        assert_eq!(segments[2].kind, OperatorSegmentKind::HashJoin);
        assert_eq!(segments[3].kind, OperatorSegmentKind::Generic);
        assert_eq!(segments[4].kind, OperatorSegmentKind::Scan);
    }

    #[test]
    fn test_csv_and_pipe_feature_string_produce_identical_plan_hash_and_features() {
        let csv = "tables_0,tables_1,tables_2,env_0,env_1,env_2,var_0,operators_0,operators_1,operators_2,operators_3,operators_4,operators_5,operators_6,operators_7,operators_8,operators_9,operators_10,operators_11,operators_12,operators_13,operators_14,operators_15,operators_16,operators_17,operators_18\n0,1001,1002,3,16,24,8,34,1,10,100,14,12,0,2,35,1,15,64,18,16,0,16,2,1,5";
        let pipe = "tables=[0,1001,1002]|env=[3,16,24]|var=[8]|operators=[34,1,10,100,14,12,0,2,35,1,15,64,18,16,0,16,2,1,5]";
        let from_csv = parse_predict_csv(csv).unwrap();
        let from_pipe = parse_feature_string(pipe).unwrap();
        assert_eq!(from_csv.plan_hash, from_pipe.plan_hash);
        assert_eq!(from_csv.plan_features_24, from_pipe.plan_features_24);
    }

    #[test]
    fn test_plan_hash_is_invariant_to_row_counts_and_cbo_costs() {
        let pipe_small = "tables=[0,1001,1002]|env=[3,16,24]|var=[8]|operators=[36,1,10,100,14,12,0,12,1,0]";
        let pipe_large = "tables=[0,1001,1002]|env=[3,16,24]|var=[8]|operators=[36,1,22,100,24,20,0,20,1,0]";
        let f_small = parse_feature_string(pipe_small).unwrap();
        let f_large = parse_feature_string(pipe_large).unwrap();
        assert_eq!(f_small.plan_hash, f_large.plan_hash);
        assert!(f_large.plan_features_24[21] > f_small.plan_features_24[21]);
    }

    #[test]
    fn test_table_catalog_target_encoding_and_known_ratio() {
        let catalog = TableCatalog::new();
        let pipe = "tables=[0,1001,1002]|env=[3,16,24]|var=[8]|operators=[36,1,10,100,14,12,0,12,1,0]";
        let plan = parse_feature_string(pipe).unwrap();
        let (_, ratio_before) = catalog.build_dense_features(&plan);
        assert_eq!(ratio_before, 0.0);

        for _ in 0..5 {
            catalog.record_execution(&plan.tables, 10_000_000, 100_000, 50_000_000, 14.0, false);
        }
        let (dense, ratio_after) = catalog.build_dense_features(&plan);
        assert_eq!(dense.len(), DENSE_FEATURE_DIM);
        assert!((ratio_after - 1.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib features`
Expected: FAIL

- [ ] **Step 3: Implement `src/features/decoder.rs`, `src/features/catalog.rs`, and `src/features/mod.rs`**

Use dynamic programming / backtracking over step sizes `[7, 8, 10, 11]` in `decode_operator_segments(ops: &[i64])`: a transition from `pos` to `pos + len` is valid if `pos + len == ops.len()` or `(ops[pos + len] - ops[pos])` is in `1..=3` (preferring step delta `1` or `2`). Compute the 24 plan features and `xxh3_64` structural `plan_hash`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib features`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/ src/lib.rs
git commit -m "feat(features): add self-synchronizing StarRocks segment decoder and table catalog"
```

---

### Task 3: Tier 1 Plan-Signature Sketch & Tier 3 Online RLS Calibrator (Wave 2 — Subagent B)

**Files:**
- Create: `src/model/mod.rs`
- Create: `src/model/sketch.rs`
- Create: `src/model/rls.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `crate::types::CanonicalPlanFeatures`
- Produces:
  - `crate::model::sketch::PlanSketchEntry` (`#[derive(Clone, Debug, Serialize, Deserialize)]`)
  - `crate::model::sketch::PlanSketchCache` (`new(max_entries: usize) -> Self`, `predict(&self, plan: &CanonicalPlanFeatures) -> Option<SketchPrediction>`, `update(&self, plan: &CanonicalPlanFeatures, effective_mem_bytes: u64, scan_bytes: u64, return_rows: u64, cpu_cost_ns: u64, spilled: bool)`, `len(&self) -> usize`, `export_entries(&self) -> Vec<(u64, PlanSketchEntry)>`, `import_entries(&self, entries: Vec<(u64, PlanSketchEntry)>)`)
  - `crate::model::sketch::SketchPrediction` (`log_mem_p50: f64`, `log_mem_p90: f64`, `log_scan_bytes: f64`, `log_return_rows: f64`, `log_cpu_ns: f64`, `spill_prob: f64`, `confidence: f64`)
  - `crate::model::rls::RlsState` (`#[derive(Clone, Debug, Serialize, Deserialize)]`)
  - `crate::model::rls::OnlineRlsCalibrator` (`new() -> Self`, `predict(&self, plan: &CanonicalPlanFeatures) -> RlsPrediction`, `update(&self, plan: &CanonicalPlanFeatures, effective_mem_bytes: u64, scan_bytes: u64, return_rows: u64, cpu_cost_ns: u64)`, `to_state(&self) -> RlsState`, `from_state(state: RlsState) -> Self`)
  - `crate::model::rls::RlsPrediction` (`log_mem_p50: f64`, `log_mem_p90: f64`, `log_scan_bytes: f64`, `log_return_rows: f64`, `log_cpu_ns: f64`)

- [ ] **Step 1: Write failing tests for Tier 1 EWMA Sketch and Tier 3 Online RLS in `src/model/sketch.rs` and `src/model/rls.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::decoder::parse_feature_string;

    #[test]
    fn test_plan_sketch_converges_and_scales_with_cardinality() {
        let sketch = PlanSketchCache::new(1000);
        let plan_base = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]"
        ).unwrap();
        assert!(sketch.predict(&plan_base).is_none());

        for _ in 0..10 {
            sketch.update(&plan_base, 100_000_000, 50_000_000, 10_000, 500_000_000, false);
        }
        let pred_base = sketch.predict(&plan_base).unwrap();
        let expected_log = (100_000_000.0_f64).ln_1p();
        assert!((pred_base.log_mem_p50 - expected_log).abs() < 0.05);
        assert!(pred_base.log_mem_p90 > pred_base.log_mem_p50);
        assert!(pred_base.confidence > 0.60);

        let plan_larger = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,18,100,16,14,0,12,2,1,1]"
        ).unwrap();
        let pred_larger = sketch.predict(&plan_larger).unwrap();
        assert!(pred_larger.log_mem_p50 > pred_base.log_mem_p50);
    }

    #[test]
    fn test_rls_calibrator_cold_start_and_collinear_stability() {
        let rls = OnlineRlsCalibrator::new();
        let plan = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,18,14,0,12,2,1,1]"
        ).unwrap();
        let cold = rls.predict(&plan);
        assert!(cold.log_mem_p50.is_finite());

        let target_mem = ((plan.plan_features_24[21] + 2.0).exp() as u64).max(10_000_000);
        for _ in 0..100 {
            rls.update(&plan, target_mem, 50_000_000, 5_000, 100_000_000);
        }
        let warm = rls.predict(&plan);
        assert!(warm.log_mem_p50.is_finite() && !warm.log_mem_p50.is_nan());
        let target_log = (target_mem as f64).ln_1p();
        assert!((warm.log_mem_p50 - target_log).abs() < 0.25);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib model`
Expected: FAIL

- [ ] **Step 3: Implement `src/model/sketch.rs`, `src/model/rls.rs`, and `src/model/mod.rs`**

Implement `PlanSketchCache` using `DashMap<u64, PlanSketchEntry>` with EWMA $\alpha = 0.20$ and confidence $c_1 = \min(0.95, \frac{n}{n+5} \exp(-0.20 \cdot \text{var}) \exp(-0.15 \cdot |\Delta r|))$. Implement `OnlineRlsCalibrator` with `std::sync::RwLock` around the $6 \times 6$ covariance matrix $\mathbf{P}$, forgetting factor $\lambda = 0.995$, symmetry enforcement, and diagonal clamping $[10^{-6}, 10^4]$.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib model`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/ src/lib.rs
git commit -m "feat(model): implement Tier 1 PlanSketchCache and Tier 3 OnlineRlsCalibrator"
```

---

### Task 4: Tier 2 Pure-Rust Multi-Target GBDT Engine (Wave 2 — Subagent C)

**Files:**
- Create: `src/model/gbdt.rs`
- Modify: `src/model/mod.rs`

**Interfaces:**
- Consumes: `crate::types::{DenseFeatureVector, DENSE_FEATURE_DIM}`
- Produces:
  - `crate::model::gbdt::LossType` (`Huber { delta: f64 }`, `Pinball { alpha: f64 }`)
  - `crate::model::gbdt::GbdtHyperparams` (`n_estimators: usize`, `max_depth: usize`, `learning_rate: f64`, `min_samples_leaf: usize`)
  - `crate::model::gbdt::BoostedTreeRegressor` (`#[derive(Clone, Debug, Serialize, Deserialize)]`, `train(features: &[DenseFeatureVector], targets: &[f64], loss: LossType, params: &GbdtHyperparams) -> Self`, `predict(&self, x: &DenseFeatureVector) -> f64`)
  - `crate::model::gbdt::GbdtTrainingRow` (`features: DenseFeatureVector`, `log_mem: f64`, `log_scan_bytes: f64`, `log_return_rows: f64`, `log_cpu_ns: f64`, `spilled: f64`)
  - `crate::model::gbdt::GbdtModelSet` (`#[derive(Clone, Debug, Serialize, Deserialize)]`, `train_parallel(rows: &[GbdtTrainingRow], params: &GbdtHyperparams) -> Self`, `predict_mem_p90(&self, x: &DenseFeatureVector) -> f64`, `predict_all(&self, x: &DenseFeatureVector) -> GbdtPrediction`)
  - `crate::model::gbdt::GbdtPrediction` (`log_mem_p90: f64`, `log_mem_p50: f64`, `log_scan_bytes: f64`, `log_return_rows: f64`, `log_cpu_ns: f64`, `spill_prob: f64`)
  - `crate::model::gbdt::pinball_loss(y_true: f64, y_pred: f64, alpha: f64) -> f64`

- [ ] **Step 1: Write failing tests for GBDT Pinball (P90) and Huber (P50) regression in `src/model/gbdt.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DENSE_FEATURE_DIM;

    #[test]
    fn test_gbdt_learns_nonlinear_interaction_and_p90_exceeds_p50() {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for i in 0..160 {
            let mut x = [0.0_f64; DENSE_FEATURE_DIM];
            let scans = (i % 10) as f64;
            let is_bcast_join = if i % 2 == 0 { 1.0 } else { 0.0 };
            x[3] = scans;
            x[11] = is_bcast_join;
            let noise = ((i % 7) as f64 - 3.0) * 0.4;
            let y = 14.0 + 0.5 * scans + 2.5 * is_bcast_join + noise;
            features.push(x);
            targets.push(y);
        }
        let params = GbdtHyperparams {
            n_estimators: 60,
            max_depth: 4,
            learning_rate: 0.1,
            min_samples_leaf: 4,
        };
        let model_p50 = BoostedTreeRegressor::train(&features, &targets, LossType::Huber { delta: 1.0 }, &params);
        let model_p90 = BoostedTreeRegressor::train(&features, &targets, LossType::Pinball { alpha: 0.90 }, &params);

        let mut test_x = [0.0_f64; DENSE_FEATURE_DIM];
        test_x[3] = 6.0;
        test_x[11] = 1.0; // expected true center = 14.0 + 3.0 + 2.5 = 19.5
        let p50 = model_p50.predict(&test_x);
        let p90 = model_p90.predict(&test_x);
        assert!((p50 - 19.5).abs() < 0.5, "p50={} should be near 19.5", p50);
        assert!(p90 > p50 + 0.3, "p90={} must exceed p50={}", p90, p50);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib model::gbdt`
Expected: FAIL

- [ ] **Step 3: Implement `src/model/gbdt.rs` with flat cache-aligned `TreeNode` arrays and `rayon` parallel multi-target training**

Implement Histogram/Quantile-binned CART tree boosting over `DenseFeatureVector` with `LossType::Huber` and `LossType::Pinball` (computing the exact $\alpha$-quantile of residuals `y_i - F_{m-1}(x_i)` in each terminal leaf for Pinball loss so quantile trees converge rapidly and accurately).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib model::gbdt`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/gbdt.rs src/model/mod.rs
git commit -m "feat(model): implement pure-Rust multi-target GBDT with Pinball P90 and Huber loss"
```

---

### Task 5: Multi-Source Telemetry Parsers & SQLite WAL Stratified Store (Wave 2 — Subagent D)

**Files:**
- Create: `src/ingest/mod.rs`
- Create: `src/ingest/parser.rs`
- Create: `src/storage/mod.rs`
- Create: `src/storage/sqlite.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `crate::types::{TelemetrySample, DenseFeatureVector}`, `crate::features::decoder::parse_feature_string`, `crate::model::gbdt::GbdtTrainingRow`
- Produces:
  - `crate::ingest::parser::ParseError` (`#[derive(Debug, thiserror::Error)]`)
  - `crate::ingest::parser::parse_features_log_line(line: &str) -> Result<Option<TelemetrySample>, ParseError>`
  - `crate::ingest::parser::AuditLogEvent` (`#[derive(Clone, Debug, Serialize, Deserialize)]` with `digest: String`, `predict_mem_bytes: u64`, `mem_cost_bytes: u64`, `scan_bytes: u64`, `scan_rows: u64`, `return_rows: u64`, `cpu_cost_ns: u64`, `spilled_bytes: u64`, `query_time_ms: u64`, `state: String`, `tables: Vec<String>`)
  - `crate::ingest::parser::ProfileSummary` (`tables: Vec<String>`, `peak_mem_bytes: u64`, `scan_bytes: u64`, `scan_rows: u64`, `spilled_bytes: u64`)
  - `crate::ingest::parser::parse_query_profile(payload: &str) -> Result<ProfileSummary, ParseError>`
  - `crate::storage::sqlite::StorageError` (`#[derive(Debug, thiserror::Error)]`)
  - `crate::storage::sqlite::SqliteStore` (`open(path: &Path, max_per_stratum: usize) -> Result<Self, StorageError>`, `insert_sample(&self, plan_hash: u64, dense: &DenseFeatureVector, sample: &TelemetrySample) -> Result<(), StorageError>`, `load_training_rows(&self) -> Result<Vec<GbdtTrainingRow>, StorageError>`, `sample_counts_by_stratum(&self) -> Result<[usize; 5], StorageError>`, `save_snapshot_blob(&self, version: u64, metrics_json: &str, blob: &[u8]) -> Result<(), StorageError>`, `load_latest_snapshot_blob(&self) -> Result<Option<(u64, String, Vec<u8>)>, StorageError>`, `get_file_checkpoint(&self, path: &str) -> Result<Option<(u64, u64)>, StorageError>`, `set_file_checkpoint(&self, path: &str, inode: u64, offset: u64) -> Result<(), StorageError>`)

- [ ] **Step 1: Write failing tests for log4j-prefixed `fe.features.log` parsing, profile parsing, and SQLite cost-stratified retention in `src/ingest/parser.rs` and `src/storage/sqlite.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_features_log_line_with_log4j_prefix_and_oom_filtering() {
        let line = "2026-10-28 21:00:01,112 INFO (qe|99) [AuditLogBuilder.java:185] digest=abc123|cpuCostNs=500000|memCostBytes=40000000|scanBytes=10000000|scanRows=50000|returnRows=100|spilledBytes=10000000|time=120|state=EOF|catalog=default_catalog|database=db1|tables=[0,0,1001]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]";
        let sample = parse_features_log_line(line).unwrap().expect("valid sample");
        assert_eq!(sample.digest, "abc123");
        assert_eq!(sample.effective_mem_bytes, 50_000_000);

        let err_line = "digest=def|cpuCostNs=0|memCostBytes=0|scanBytes=0|scanRows=0|returnRows=0|spilledBytes=0|time=1|state=ERR|catalog=c|database=d|tables=[0,0,0]|env=[1,8,20]|var=[1]|operators=[39,0,0,0,0,0,0,0,0,0,0]";
        assert!(parse_features_log_line(err_line).unwrap().is_none());
    }

    #[test]
    fn test_sqlite_stratified_retention_preserves_large_queries_under_small_query_flood() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = SqliteStore::open(tmp.path(), 5).unwrap();
        let line_base = "digest=d1|cpuCostNs=100|memCostBytes=2000000000|scanBytes=1000|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|catalog=c|database=d|tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]";
        let mut sample = parse_features_log_line(line_base).unwrap().unwrap();
        let dense = [0.0_f64; 39];

        for _ in 0..3 {
            store.insert_sample(111, &dense, &sample).unwrap();
        }
        sample.effective_mem_bytes = 2_000_000;
        for _ in 0..20 {
            store.insert_sample(222, &dense, &sample).unwrap();
        }
        let counts = store.sample_counts_by_stratum().unwrap();
        assert_eq!(counts[0], 5, "stratum 0 capped at 5");
        assert_eq!(counts[3], 3, "stratum 3 (1-10GB) preserved all 3 heavy queries!");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib ingest storage`
Expected: FAIL

- [ ] **Step 3: Implement `src/ingest/parser.rs` and `src/storage/sqlite.rs`**

In `parse_features_log_line`, locate `"digest="` via `line.find("digest=")` to strip any log4j prefix, split key-value pairs by `'|'`, parse `tables=[...]...` via `parse_feature_string`, and compute `effective_mem_bytes` via `compute_effective_mem_bytes`. In `SqliteStore`, serialize `DenseFeatureVector` using `rmp_serde::to_vec(&dense.as_slice())` (or 39 little-endian `f64` bytes `39 * 8 = 312` bytes), enable `PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;`, and prune only within the matching `mem_stratum` when a stratum exceeds `max_per_stratum`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib ingest storage`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest/ src/storage/ src/lib.rs
git commit -m "feat(ingest,storage): add features/audit/profile parsers and stratified SQLite WAL store"
```

---

### Task 6: GitHub Actions CI, Dependabot, `README.md` & `AGENTS.md` (Wave 2 — Subagent E)

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/dependabot.yml`
- Create: `README.md`
- Create: `AGENTS.md`

**Interfaces:**
- Consumes: Spec `docs/superpowers/specs/2026-10-28-starrocks-cost-prediction-service-design.md`
- Produces:
  - `.github/workflows/ci.yml`: Runs on `push` (`main`) and `pull_request` with 3 jobs (`fmt`: `cargo fmt --all -- --check`, `clippy`: `cargo clippy --all-targets --all-features -- -D warnings`, `test`: `cargo test --all-targets --all-features`) using `actions/checkout@v7`, `dtolnay/rust-toolchain@stable`, and `Swatinem/rust-cache@v2`.
  - `.github/dependabot.yml`: Dependabot v2 config monitoring `package-ecosystem: "cargo"` (`directory: "/"`, weekly schedule) and `package-ecosystem: "github-actions"` (`directory: "/"`, weekly schedule).
  - `README.md`: Comprehensive open-source documentation covering StarRocks `fe.conf` setup (`enable_query_cost_prediction`, `query_cost_prediction_service_address`, `enable_plan_feature_collection`), the 3-Tier ML architecture, API reference (`/predict_csv`, `/health_check`, `/v1/predict`, `/v1/ingest/*`, `/v1/models/status`, `/v1/admin/*`, `/metrics`), configuration env vars, and quickstart instructions.
  - `AGENTS.md`: Developer & AI coding agent guide documenting architecture invariants, hot-path latency constraints (`p99 < 250µs`, zero-HTTP-500 policy on `/predict_csv`, `TCP_NODELAY`), module map, TDD workflow, and pre-commit commands (`cargo fmt`, `cargo clippy`, `cargo test`).

- [ ] **Step 1: Create `.github/workflows/ci.yml` and `.github/dependabot.yml`**

Write `.github/workflows/ci.yml` and `.github/dependabot.yml` with latest action versions (`actions/checkout@v7`, `dtolnay/rust-toolchain@stable`, `Swatinem/rust-cache@v2`).

- [ ] **Step 2: Create `README.md` and `AGENTS.md` in repository root**

Write thorough, accurate `README.md` and `AGENTS.md` matching the exact crate structure, StarRocks internals, and endpoints in the spec.

- [ ] **Step 3: Verify YAML syntax and markdown structure**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/dependabot.yml')); print('YAML valid')"` (or verify structure via script if PyYAML isn't installed) and verify files exist at root.
Expected: Valid YAML and markdown files in place.

- [ ] **Step 4: Commit**

```bash
git add .github/ README.md AGENTS.md
git commit -m "docs,ci: add GitHub Actions CI, Dependabot config, README.md, and AGENTS.md"
```

---

### Task 7: 3-Tier Ensemble Engine, `ArcSwap` Snapshot & Champion/Challenger Trainer (Wave 3 — Subagent F)

**Files:**
- Create: `src/model/ensemble.rs`
- Create: `src/trainer/mod.rs`
- Modify: `src/model/mod.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: Tasks 1–5 (`TableCatalog`, `PlanSketchCache`, `OnlineRlsCalibrator`, `GbdtModelSet`, `SqliteStore`, `AppConfig`)
- Produces:
  - `crate::model::ensemble::HoldoutMetrics` (`#[derive(Clone, Debug, Default, Serialize, Deserialize)]` with `pinball_p90: f64`, `q_error_p50: f64`, `q_error_p90: f64`, `validation_samples: usize`)
  - `crate::model::ensemble::RetrainReport` (`#[derive(Clone, Debug, Serialize, Deserialize)]` with `promoted: bool`, `model_version: u64`, `training_samples: usize`, `holdout_metrics: HoldoutMetrics`)
  - `crate::model::ensemble::ModelStatusReport` (`#[derive(Clone, Debug, Serialize, Deserialize)]` with `model_version: u64`, `gbdt_trained: bool`, `sketch_entries: usize`, `known_tables: usize`, `ingested_samples: u64`, `stratum_counts: [usize; 5]`, `live_q_error_ewma: f64`, `circuit_breaker_tripped: bool`, `holdout_metrics: Option<HoldoutMetrics>`)
  - `crate::model::ensemble::EnsembleEngine` with:
    - `EnsembleEngine::new(config: AppConfig, store: Arc<SqliteStore>) -> Result<Arc<Self>, StorageError>`
    - `EnsembleEngine::predict(&self, plan: &CanonicalPlanFeatures, csv_fast_path: bool) -> PredictionOutput`
    - `EnsembleEngine::observe_execution(&self, sample: &TelemetrySample) -> Result<bool, StorageError>`
    - `EnsembleEngine::retrain_and_promote(&self) -> Result<RetrainReport, StorageError>`
    - `EnsembleEngine::status(&self) -> ModelStatusReport`
    - `EnsembleEngine::set_circuit_breaker(&self, tripped: bool)`
    - `EnsembleEngine::is_circuit_breaker_tripped(&self) -> bool`
    - `EnsembleEngine::catalog(&self) -> &TableCatalog`

- [ ] **Step 1: Write failing tests for 3-tier confidence blending, drift detection, and Champion/Challenger promotion in `src/model/ensemble.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;
    use crate::features::decoder::parse_feature_string;

    #[test]
    fn test_ensemble_transitions_from_tier3_cold_start_to_tier1_and_tier2_gbdt() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut cfg = AppConfig::default();
        cfg.db_path = tmp.path().to_path_buf();
        cfg.min_new_samples_for_retrain = 30;
        let store = Arc::new(SqliteStore::open(&cfg.db_path, 1000).unwrap());
        let engine = EnsembleEngine::new(cfg, store).unwrap();

        let plan_a = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]"
        ).unwrap();
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

        let unseen_plan_on_101 = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[36,1,15,100,18,16,0,15,1,0,39,1,15,100,16,14,0,12,2,1,1]"
        ).unwrap();
        let pred_gbdt = engine.predict(&unseen_plan_on_101, false);
        assert_eq!(pred_gbdt.dominant_tier, "gbdt_tier2");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib model::ensemble`
Expected: FAIL

- [ ] **Step 3: Implement `src/model/ensemble.rs` and `src/trainer/mod.rs`**

Implement `EnsembleEngine` holding `ArcSwap<Option<ActiveGbdtSnapshot>>`, `PlanSketchCache`, `OnlineRlsCalibrator`, and `TableCatalog`. Use `rmp_serde` (MessagePack) to serialize/deserialize full `ModelSnapshot` blobs (`ActiveGbdtSnapshot`, `PlanSketchEntry` vec, `RlsState`, `TableProfile` vec) into `SqliteStore` so restarts hydrate in `< 50ms`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib model::ensemble`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/ensemble.rs src/trainer/ src/model/mod.rs src/lib.rs
git commit -m "feat(ensemble,trainer): implement 3-tier confidence blending, ArcSwap hot-reload, and Champion/Challenger trainer"
```

---

### Task 8: In-Flight Prediction Correlator & Resilient File Tailer (Wave 3 — Subagent G)

**Files:**
- Create: `src/ingest/correlator.rs`
- Create: `src/ingest/tailer.rs`
- Modify: `src/ingest/mod.rs`

**Interfaces:**
- Consumes: `crate::types::{CanonicalPlanFeatures, TelemetrySample}`, `crate::ingest::parser::{AuditLogEvent, parse_features_log_line}`, `crate::storage::sqlite::SqliteStore`
- Produces:
  - `crate::ingest::correlator::InFlightCache` (`new(capacity: usize, ttl_secs: u64) -> Self`, `record_prediction(&self, watermarked_mem_bytes: u64, plan: CanonicalPlanFeatures)`, `correlate_audit_event(&self, event: &AuditLogEvent) -> Option<TelemetrySample>`)
  - `crate::ingest::tailer::FileTailer` (`poll_once(log_path: &Path, store: &SqliteStore) -> Result<Vec<TelemetrySample>, std::io::Error>`)

- [ ] **Step 1: Write failing tests for 10-bit watermark correlation and log rotation tailing in `src/ingest/correlator.rs` and `src/ingest/tailer.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::decoder::parse_feature_string;
    use crate::types::watermark_mem_bytes;

    #[test]
    fn test_inflight_cache_correlates_audit_event_via_watermarked_mem_bytes() {
        let cache = InFlightCache::new(1000, 1800);
        let plan = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]"
        ).unwrap();
        let watermarked = watermark_mem_bytes(80_000_000, plan.plan_hash, 1_048_576);
        cache.record_prediction(watermarked, plan.clone());

        let audit = AuditLogEvent {
            digest: "sql_digest_1".to_string(),
            predict_mem_bytes: watermarked,
            mem_cost_bytes: 95_000_000,
            scan_bytes: 30_000_000,
            scan_rows: 200_000,
            return_rows: 50,
            cpu_cost_ns: 400_000_000,
            spilled_bytes: 5_000_000,
            query_time_ms: 80,
            state: "EOF".to_string(),
            tables: vec!["101".to_string()],
        };
        let correlated = cache.correlate_audit_event(&audit).expect("must correlate");
        assert_eq!(correlated.plan_features.plan_hash, plan.plan_hash);
        assert_eq!(correlated.effective_mem_bytes, 100_000_000);
    }

    #[test]
    fn test_file_tailer_checkpoints_offset_and_handles_log_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let log_path = dir.path().join("fe.features.log");
        let store = SqliteStore::open(&db_path, 100).unwrap();

        let line1 = "digest=d1|cpuCostNs=100|memCostBytes=10000000|scanBytes=1000|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|catalog=c|database=d|tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]\n";
        std::fs::write(&log_path, line1).unwrap();

        let batch1 = FileTailer::poll_once(&log_path, &store).unwrap();
        assert_eq!(batch1.len(), 1);
        let batch2 = FileTailer::poll_once(&log_path, &store).unwrap();
        assert_eq!(batch2.len(), 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib ingest`
Expected: FAIL

- [ ] **Step 3: Implement `src/ingest/correlator.rs` and `src/ingest/tailer.rs`**

Implement `InFlightCache` using `DashMap<u64, (CanonicalPlanFeatures, Instant)>` keyed by `watermarked_mem_bytes` with bounded eviction. Implement `FileTailer::poll_once` checking file metadata (`inode`, `len`); if `inode` changed or `len < saved_offset`, reset `offset = 0`, read complete `\n`-terminated lines, and persist `(inode, new_offset)` in `SqliteStore`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib ingest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest/
git commit -m "feat(ingest): add watermarked InFlightCache correlator and checkpointed FileTailer"
```

---

### Task 9: Axum 0.8 HTTP Server, Prometheus Metrics, CLI & End-to-End StarRocks Convergence Suite (Wave 4)

**Files:**
- Create: `src/api/mod.rs`
- Modify: `src/main.rs`
- Modify: `src/lib.rs`
- Create: `tests/e2e_convergence.rs`

**Interfaces:**
- Consumes: All modules from Tasks 1–8
- Produces:
  - `crate::api::AppState` & `crate::api::build_router(state: Arc<AppState>) -> axum::Router`
  - Endpoints:
    - `POST /predict_csv`
    - `GET /health_check`
    - `POST /v1/predict`
    - `POST /v1/ingest/features`
    - `POST /v1/ingest/audit`
    - `POST /v1/ingest/profile`
    - `GET /v1/models/status`
    - `POST /v1/admin/retrain`
    - `POST /v1/admin/circuit-breaker`
    - `GET /metrics`
  - TCP listener in `src/main.rs` setting `TCP_NODELAY(true)` on accepted sockets and running background trainer + file tailer tasks.

- [ ] **Step 1: Write failing integration and E2E tests in `tests/e2e_convergence.rs`**

```rust
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use starrocks_cost_prediction_service::{
    api::{build_router, AppState},
    config::AppConfig,
};
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn test_starrocks_wire_protocol_and_circuit_breaker() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let mut cfg = AppConfig::default();
    cfg.db_path = tmp.path().to_path_buf();
    let state = Arc::new(AppState::new(cfg).unwrap());
    let app = build_router(state.clone());

    // 1. GET /health_check -> 200 OK
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/health_check").body(Body::empty()).unwrap())
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
    let predicted: f64 = body_str.parse().expect("StarRocks parses Double.parseDouble");
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
        .oneshot(Request::builder().uri("/health_check").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(hc_tripped.status(), StatusCode::SERVICE_UNAVAILABLE);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test e2e_convergence`
Expected: FAIL (`api` module not implemented)

- [ ] **Step 3: Implement `src/api/mod.rs` and `src/main.rs` and add full E2E workload convergence test in `tests/e2e_convergence.rs`**

Implement all routes in `src/api/mod.rs`, latency histogram counters using `AtomicU64` in `AppState`, and a second test in `tests/e2e_convergence.rs` (`test_end_to_end_ingest_retrain_and_multi_target_prediction`) that ingests 60 `fe.features.log` lines, triggers `/v1/admin/retrain`, verifies `GET /v1/models/status` reports `gbdt_trained: true`, and verifies `POST /v1/predict` returns accurate `predicted_mem_bytes_p90`, `predicted_scan_bytes`, and `predicted_return_rows`.

- [ ] **Step 4: Run full formatting, linting, and test verification suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets --all-features`
Expected: PASS (0 warnings, all unit and E2E tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/ tests/
git commit -m "feat(api): add Axum 0.8 HTTP server, Prometheus metrics, and E2E StarRocks convergence suite"
```
