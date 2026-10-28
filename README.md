# StarRocks Cost Prediction Service

A high-performance, self-learning machine learning daemon written in Rust for StarRocks Cost-Based Optimizer (CBO) calibration, Query Queue V2 slot estimation, and Resource Group classification.

[![CI](https://github.com/jimmystewpot/starrocks-cost-prediction-service/actions/workflows/ci.yml/badge.svg)](https://github.com/jimmystewpot/starrocks-cost-prediction-service/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org)

---

## Table of Contents

- [Overview & Why StarRocks Needs ML Cost Prediction](#overview--why-starrocks-needs-ml-cost-prediction)
  - [The CBO Cardinality Skew Problem](#the-cbo-cardinality-skew-problem)
  - [Query Queue V2 MBE Slot Allocation](#query-queue-v2-mbe-slot-allocation)
  - [Resource Group Workload Management](#resource-group-workload-management)
- [StarRocks `fe.conf` Configuration](#starrocks-feconf-configuration)
- [Architecture & 3-Tier ML Ensemble](#architecture--3-tier-ml-ensemble)
  - [ASCII Architecture Overview](#ascii-architecture-overview)
  - [Tier 1: Structural Plan-Signature Sketch](#tier-1-structural-plan-signature-sketch)
  - [Tier 2: Pure-Rust Multi-Target GBDT Ensemble](#tier-2-pure-rust-multi-target-gbdt-ensemble)
  - [Tier 3: Online Recursive Least Squares (RLS) CBO Calibrator](#tier-3-online-recursive-least-squares-rls-cbo-calibrator)
  - [Confidence Blending & Safety Bounds](#confidence-blending--safety-bounds)
  - [10-Bit Audit Correlation Watermark](#10-bit-audit-correlation-watermark)
  - [Self-Synchronizing Segment Decoder](#self-synchronizing-segment-decoder)
  - [Embedded Analytical Storage (SQLite WAL)](#embedded-analytical-storage-sqlite-wal)
- [HTTP API Reference](#http-api-reference)
  - [`POST /predict_csv`](#post-predict_csv)
  - [`GET /health_check`](#get-health_check)
  - [`POST /v1/predict`](#post-v1predict)
  - [`POST /v1/ingest/features`](#post-v1ingestfeatures)
  - [`POST /v1/ingest/audit`](#post-v1ingestaudit)
  - [`POST /v1/ingest/profile`](#post-v1ingestprofile)
  - [`GET /v1/models/status`](#get-v1modelsstatus)
  - [`POST /v1/admin/retrain`](#post-v1adminretrain)
  - [`POST /v1/admin/circuit-breaker`](#post-v1admincircuit-breaker)
  - [`GET /metrics`](#get-metrics)
- [Configuration Reference](#configuration-reference)
- [Build, Test & Run](#build-test--run)
  - [Prerequisites](#prerequisites)
  - [Building from Source](#building-from-source)
  - [Running the Service](#running-the-service)
  - [Running Tests](#running-tests)
  - [Docker & Containerized Deployment](#docker--containerized-deployment)

---

## Overview & Why StarRocks Needs ML Cost Prediction

StarRocks features an advanced Cost-Based Optimizer (CBO) that computes analytical query execution plans using catalog metadata, table statistics (`row_count`, `data_size`), and column histograms. However, in modern high-concurrency production deployments, static CBO resource estimates face inherent statistical limitations.

### The CBO Cardinality Skew Problem

1. **Complex Joins & Skewed Distributions:** Multi-way joins, non-uniform key distributions, and correlated filters break statistical independence assumptions. Static estimates frequently diverge from actual execution memory and execution time by **1 to 3 orders of magnitude**.
2. **External Catalogs:** Queries querying external lakehouses (Apache Iceberg, Apache Hudi, Hive, Delta Lake) often lack up-to-date column histograms or partition statistics, forcing the CBO to rely on coarse heuristics.
3. **Complex Expressions & UDFs:** Scalar predicates, regular expressions, and nested subqueries cannot be reliably estimated prior to execution.

### Query Queue V2 MBE Slot Allocation

StarRocks Query Queue V2 prevents backend (BE) node memory exhaustion and thrashing. When configured with:
```ini
query_queue_slots_estimator_strategy = "MBE"
```
The query queue uses the Memory-Based Estimator (`MBE`) to determine how many concurrency slots a query must occupy:
$$\text{allocated\_slots} = \left\lceil \frac{\text{predictedMemBytes}}{\text{memBytesPerSlot}} \right\rceil$$

When the CBO underestimates memory consumption:
* Large queries are granted too few slots, leading to concurrent query over-admission, heavy memory spilling, and **Backend OOM process terminations**.
* When CBO overestimates memory consumption, queries consume excessive slots, leading to unnecessary queue starvation and artificial throughput bottlenecks.

The **StarRocks Cost Prediction Service** replaces the static CBO `planMemCost` with runtime P90 peak memory predictions (`coord.getPredictedCost()`), ensuring accurate slot reservation.

### Resource Group Workload Management

StarRocks Resource Groups classify and prioritize queries using resource limits. In `ResourceGroupMgr.java`, when cost prediction is enabled, StarRocks evaluates query `max_memory_cost` rules against `predictMemBytes`. Accurate predictions ensure:
* Heavy analytical queries are routed to dedicated high-memory queues or resource groups.
* Interactive dashboards and low-latency queries are protected from noisy neighbors.

---

## StarRocks `fe.conf` Configuration

To enable the cost prediction integration, add the following properties to each StarRocks Frontend (`conf/fe.conf`):

```ini
# ==============================================================================
# StarRocks Machine Learning Query Cost Prediction Configuration
# ==============================================================================

# Enable CBO cost prediction integration
enable_query_cost_prediction = true

# HTTP endpoint address of the StarRocks Cost Prediction Service
query_cost_prediction_service_address = "http://10.0.1.50:8080"

# Enable asynchronous plan feature telemetry logging (writes to fe.features.log)
enable_plan_feature_collection = true

# Optional: Health check polling interval in seconds (default: 30)
query_cost_predictor_healthchk_interval = 30
```

### How StarRocks Interacts with the Service

1. **Background Health Check (`GET /health_check`):**
   A dedicated daemon thread in `CostPredictor.java` polls `/health_check` every 30 seconds. If the endpoint responds with HTTP `200 OK`, `ServiceBasedCostPredictor.isAvailable()` evaluates to `true`.
2. **Compilation Hot-Path Prediction (`POST /predict_csv`):**
   In `StmtExecutor.java`, immediately before query deployment, the coordinator executes `predictor.tryPredictMemoryBytes(execPlan)`. It serializes the plan features to CSV and POSTs to `/predict_csv`.
3. **Offline Ingestion (`fe.features.log`):**
   Upon query completion, `AuditLogBuilder.java` appends the full plan feature vector and runtime execution metrics (`memCostBytes`, `cpuCostNs`, `scanBytes`, `returnRows`, `spilledBytes`) to `log/fe.features.log`. The cost prediction service continuously ingests this file to retrain models.

---

## Architecture & 3-Tier ML Ensemble

The service is packaged as a single standalone Rust binary with strictly separated execution planes:

### ASCII Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         STARROCKS FRONTEND (FE) NODES                                           │
│  ┌─────────────────────────────────────────┐       ┌─────────────────────────────────────────┐  │
│  │ StmtExecutor (Query Compile Hot Path)   │       │ Telemetry Sources (Post-Execution)      │  │
│  │ • POST /predict_csv (Max 2 Conns/FE!)   │       │ • fe.features.log (PlanFeatures+Actuals)│  │
│  │ • GET  /health_check (Every 30s)        │       │ • starrocks_audit_tbl__ / fe.audit.log  │  │
│  │ └───────────────────┬───────────────────┘       │ • FE HTTP /api/profile (Query Profiles) │  │
└────────────────────────┼───────────────────────────┴────────────────────┬────────────────────┘  │
                         │ Sync HTTP (TCP_NODELAY, p99 < 250µs)           │ Async Push / Tail / Poll
                         ▼                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                 starrocks-cost-predictor (Single Rust Binary)                                   │
│                                                                                                 │
│  ╔═════════════════════════════════════════╗       ╔═════════════════════════════════════════╗  │
│  ║ 1. HOT INFERENCE PLANE (Tokio / Axum)   ║       ║ 2. ASYNC INGESTION PLANE                ║  │
│  ║ • Zero-alloc CSV & Pipe Tokenizer       ║       ║ • File Tailer (notify + inode seek)     ║  │
│  ║ • Self-Synchronizing Segment Decoder    ║       ║ • HTTP Ingest API (/v1/ingest/*)        ║  │
│  ║ • Reads ModelSnapshot via ArcSwap       ║       ║ • In-Flight Prediction Correlator       ║  │
│  ║ • Pure-Sync CPU Execution (< 25µs)      ║       ║ • Spill & OOM Label Normalizer          ║  │
│  ╚═══════════════════╤═════════════════════╝       ╚════════════╤══════════════════╤═════════╝  │
│                      │ Lock-Free Pointer Load (~1ns)            │ Real-time        │ Batched    │
│                      ▼                                          ▼ (<1ms)           ▼ Append     │
│  ╔═══════════════════════════════════════════════════════════════════════╗  ┌────────────────┐  │
│  ║ 3. IN-MEMORY MODEL SNAPSHOT & LIVE SKETCHES                           ║  │ Embedded Store │  │
│  ║ ┌─────────────────────────┐ ┌───────────────────┐ ┌─────────────────┐ ║  │ (SQLite WAL    │  │
│  ║ │ Tier 1: Bounded Sketch  │ │ Tier 2: ArcSwap   │ │ Tier 3: Sharded │ ║  │  via rusqlite  │  │
│  ║ │ PlanHash ──► EWMA P50/90│ │ Multi-Target GBDT │ │ Online RLS CBO  │ ║  │  bundled +     │  │
│  ║ │ & Row-Scaling Factor    │ │ Ensemble (P50/P90)│ │ Calibrator      │ ║  │  Model Blobs)  │  │
│  ║ └─────────────────────────┘ └─────────▲─────────┘ └─────────────────┘ ║  └────────┬───────┘  │
│  ╚═══════════════════════════════════════╪═══════════════════════════════╝           │          │
│                                          │ Atomic Pointer Swap (ArcSwap::store)      │          │
│                                          │                                           ▼          │
│  ╔═══════════════════════════════════════╧═══════════════════════════════════════════════════╗  │
│  ║ 4. BACKGROUND TRAINING PLANE (Dedicated Rayon Thread Pool)                                ║  │
│  ║ • Triggered every N mins or M new queries (or Drift Detected in Q-Error)                  ║  │
│  ║ • Cost-stratified training window from embedded SQLite WAL store                          ║  │
│  ║ • Trains Pure-Rust GBDT trees (mem_p90, mem_p50, scan_bytes, return_rows, cpu_ns, spill)  ║  │
│  ║ • Validates holdout Q-Error before promoting snapshot to ArcSwap                          ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Tier 1: Structural Plan-Signature Sketch

* **Latency:** `~2µs` | **Storage:** Lock-free in-memory `DashMap` (bounded LRU, up to 100,000 templates).
* **Deterministic Template Hash:** Computes `xxh3_64` over the plan structure (tables, physical operator sequence, broadcast flags, distribution types, aggregate groupings), while excluding volatile runtime cardinalities and byte sizes. Identical SQL templates on growing tables hash to the exact same signature.
* **Cardinality Elasticity:** Adjusts historical memory consumption using input cardinality delta:
  $$\Delta r = \text{clamp}(\text{scan\_est\_log\_sum\_rows} - \text{ewma\_ref\_scan\_log\_rows}, -2.0, 2.0)$$
  $$\hat{y}_{\text{tier1, p50}} = \text{ewma\_log\_mem} + 0.5 \cdot \Delta r$$
  $$\hat{y}_{\text{tier1, p90}} = \hat{y}_{\text{tier1, p50}} + 1.2816 \sqrt{\max(0.04, \text{ewma\_var\_log\_mem})}$$

### Tier 2: Pure-Rust Multi-Target GBDT Ensemble

* **Latency:** `~15µs` | **Engine:** Native Rust Gradient Boosted Decision Tree learner (`src/model/gbdt.rs`).
* **Zero C++/FFI Dependencies:** Compiles in under 2 seconds, flat cache-aligned `Vec<TreeNode>` traversal, serialized with `rmp-serde` (MessagePack).
* **Multi-Target Loss Functions:**
  1. `mem_p90`: Pinball / Quantile Loss ($\alpha = 0.90$):
     $$\mathcal{L}_{0.90}(y, \hat{y}) = \max(0.90 \cdot (y - \hat{y}), 0.10 \cdot (\hat{y} - y))$$
     Penalizes memory under-prediction 9x more heavily than over-prediction to guarantee safe P90 bounds.
  2. `mem_p50`, `scan_bytes_p50`, `return_rows_p50`, `cpu_ns_p50`: Huber Loss ($\delta = 1.0$) for outlier-resistant median estimation.
  3. `spill_prob`: Binary logistic / squared-error classification for query memory spilling risk.

### Tier 3: Online Recursive Least Squares (RLS) CBO Calibrator

* **Latency:** `~1µs` | **Role:** Cold-start protection on fresh clusters or queries touching 100% unseen tables.
* **Mechanism:** 6-dimensional linear regressor with forgetting factor $\lambda = 0.995$:
  $$\mathbf{x}_{\text{rls}} = \begin{bmatrix} 1.0 & \text{cbo\_est\_total\_mem\_log} & \text{scan\_est\_log\_sum\_rows} & \ln(1 + \text{join\_count}) & \ln(1 + \text{agg\_count}) & \text{env\_dop\_log} \end{bmatrix}^T$$
* **Cold-Start Prior:** Initialized at $\mathbf{w}_0 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]^T$ and $\mathbf{P}_0 = 10 \cdot \mathbf{I}$. Directly predicts CBO estimates on uncalibrated tables, adding a $+0.693$ ($2\times$) safety margin until historical telemetry updates the weights.

### Confidence Blending & Safety Bounds

Predictions from all three tiers are continuously blended based on statistical confidence:
1. $c_1 \in [0, 0.95] = \min\left(0.95, \frac{n_{\text{hits}}}{n_{\text{hits}} + 5} \cdot e^{-0.20 \cdot \sigma^2} \cdot e^{-0.15 \cdot |\Delta r|}\right)$
2. $c_2 \in [0, 0.90] = \mathbb{I}(\text{gbdt\_trained}) \cdot (0.45 + 0.45 \cdot \text{known\_table\_ratio})$
3. Normalized weights: $w_1 = c_1$, $w_2 = (1 - w_1) c_2$, $w_3 = 1 - w_1 - w_2$.
4. **Safety Clamping:**
   * Upper bound derived from StarRocks hardware profile `env_2`:
     $$\text{max\_bytes} = \max\left(\text{default\_fallback}, \text{numBeNodes} \times \exp(\min(40, \text{env\_2} + 1)) \times 2.0\right)$$
   * Final memory is clamped to `[SR_PREDICTOR_MIN_PREDICTED_MEM_BYTES, max_bytes]`.

### 10-Bit Audit Correlation Watermark

StarRocks FE's `POST /predict_csv` protocol only returns a single integer (`long` bytes). To correlate subsequent audit logs (`fe.audit.log` / `starrocks_audit_tbl__`) with earlier in-flight query predictions, the service watermarks the lowest 10 bits:
$$\text{watermarked\_bytes} = (\text{bytes} \ \& \ \neg\text{0x3FF}) \mid (\text{plan\_hash} \ \& \ \text{0x3FF})$$
This jitters the response by less than $1\text{ KiB}$ (negligible on megabytes/gigabytes) while enabling collision-free in-flight correlation against the `InFlightCache` ring buffer.

### Self-Synchronizing Segment Decoder

StarRocks physical plan features iterate over `OperatorType` enums. Different operators emit vectors of varying lengths ($L \in \{7, 8, 10, 11\}$). Because operator ordinals increase monotonically, our decoder uses a 2-step lookahead dynamic programming path to establish operator boundaries. The decoder is **100% immune to StarRocks minor and major version upgrade skew**.

### Embedded Analytical Storage (SQLite WAL)

Instead of embedding heavy C++ analytical engines like DuckDB (which require 10+ minute compilation times and consume variable memory), the service embeds `rusqlite` with the `bundled` feature:
* Compiles in **5 seconds**.
* Uses $< 10\text{ MB}$ of RAM.
* Non-blocking readers with single-threaded WAL writer.
* Cost-stratified sample compaction maintains up to 50,000 samples across 5 memory strata (`<10MB`, `10-100MB`, `100MB-1GB`, `1-10GB`, `>10GB`), capping database size at 250,000 stratified samples.

---

## HTTP API Reference

### `POST /predict_csv`

StarRocks FE native prediction endpoint. Invoked synchronously on the query compile hot path.

* **Content-Type:** `text/csv`
* **Response:** `text/plain` containing predicted P90 peak memory bytes.
* **SRE Guarantee:** **Zero HTTP 500 policy**. Malformed CSV or parsing errors immediately return the safe default fallback (`67,108,864` bytes) with HTTP `200 OK`.

#### Example Request

```bash
curl -X POST http://127.0.0.1:8080/predict_csv \
  -H "Content-Type: text/csv" \
  --data-binary $'tables_0,tables_1,tables_2,env_0,env_1,env_2,var_0,operators_0,operators_1,operators_2,operators_3,operators_4,operators_5,operators_6,operators_7,operators_8,operators_9,operators_10\nlineitem,orders,0,3,16,24,8,1,1,18,128,15,14,0,100,5,2,1'
```

#### Example Response

```
536871234
```

---

### `GET /health_check`

StarRocks FE daemon health check endpoint.

* **Response:** `text/plain`
* **Status Codes:**
  * `200 OK`: Service healthy and circuit breaker closed.
  * `503 Service Unavailable`: Circuit breaker open. StarRocks FE will disable prediction until next poll.

#### Example Request

```bash
curl -i http://127.0.0.1:8080/health_check
```

#### Example Response

```http
HTTP/1.1 200 OK
content-type: text/plain; charset=utf-8
content-length: 2

OK
```

---

### `POST /v1/predict`

Extended multi-target prediction API for advanced integrations, query gateways, and diagnostic tooling.

* **Content-Type:** `application/json` or `text/csv`
* **Response:** `application/json`

#### Example Request

```bash
curl -X POST http://127.0.0.1:8080/v1/predict \
  -H "Content-Type: text/csv" \
  --data-binary $'tables_0,tables_1,tables_2,env_0,env_1,env_2,var_0,operators_0,operators_1,operators_2,operators_3,operators_4,operators_5,operators_6,operators_7,operators_8,operators_9,operators_10\nlineitem,orders,0,3,16,24,8,1,1,18,128,15,14,0,100,5,2,1'
```

#### Example Response

```json
{
  "predicted_mem_bytes_p90": 536871234,
  "predicted_mem_bytes_p50": 341829012,
  "predicted_scan_bytes": 1073741824,
  "predicted_return_rows": 150000,
  "predicted_cpu_cost_ns": 450000000,
  "spill_probability": 0.03,
  "confidence": 0.88,
  "tier_weights": [0.65, 0.30, 0.05],
  "dominant_tier": "sketch_tier1",
  "model_version": 4,
  "latency_us": 18
}
```

---

### `POST /v1/ingest/features`

Ingests line-delimited records from StarRocks `fe.features.log`.

* **Content-Type:** `text/plain`
* **Response:** `application/json` (`202 Accepted`)

#### Example Request

```bash
curl -X POST http://127.0.0.1:8080/v1/ingest/features \
  -H "Content-Type: text/plain" \
  -d 'digest=abc12345|cpuCostNs=125000000|memCostBytes=45000000|scanBytes=10485760|scanRows=50000|returnRows=100|spilledBytes=0|time=125|state=EOF|catalog=default_catalog|database=tpch|tables=[lineitem,orders,0]|env=[3,16,24]|var=[8]|operators=[1,1,18,128,15,14,0,100,5,2,1]'
```

#### Example Response

```json
{
  "accepted": 1
}
```

---

### `POST /v1/ingest/audit`

Ingests StarRocks query completion audit records from `starrocks_audit_tbl__` or `fe.audit.log`.

* **Content-Type:** `application/json`
* **Response:** `application/json`

#### Example Request

```bash
curl -X POST http://127.0.0.1:8080/v1/ingest/audit \
  -H "Content-Type: application/json" \
  -d '[{
    "queryId": "e0a29ef1-4b1a-11ef-8b29-00163e000001",
    "timestamp": 1730000000,
    "queryTime": 450,
    "memCostBytes": 104857600,
    "predictMemBytes": 104858450,
    "cpuCostNs": 950000000,
    "scanBytes": 52428800,
    "scanRows": 250000,
    "returnRows": 500,
    "spilledBytes": 0,
    "state": "EOF"
  }]'
```

#### Example Response

```json
{
  "correlated": 1,
  "unmatched": 0
}
```

---

### `POST /v1/ingest/profile`

Ingests detailed StarRocks Runtime Query Profiles (JSON or text tree) for operator-level telemetry calibration.

* **Content-Type:** `application/json` or `text/plain`
* **Response:** `application/json`

#### Example Request

```bash
curl -X POST http://127.0.0.1:8080/v1/ingest/profile \
  -H "Content-Type: application/json" \
  -d '{"query_id": "e0a29ef1-4b1a-11ef-8b29-00163e000001", "peak_memory": 104857600, "operators": []}'
```

#### Example Response

```json
{
  "status": "profile_ingested"
}
```

---

### `GET /v1/models/status`

Inspects active model snapshot metrics, cache sizes, and training telemetry.

* **Response:** `application/json`

#### Example Request

```bash
curl http://127.0.0.1:8080/v1/models/status
```

#### Example Response

```json
{
  "model_version": 4,
  "gbdt_trained": true,
  "sketch_entries": 1420,
  "known_tables": 38,
  "ingested_samples": 45120,
  "live_q_error_ewma": 1.14,
  "circuit_breaker_tripped": false,
  "holdout_metrics": {
    "mem_p90_pinball_loss": 0.082,
    "mem_p50_huber_loss": 0.041,
    "q_error_median": 1.12,
    "q_error_p90": 1.65
  }
}
```

---

### `POST /v1/admin/retrain`

Manually triggers an immediate background training run over stored SQLite WAL samples.

* **Response:** `application/json`

#### Example Request

```bash
curl -X POST http://127.0.0.1:8080/v1/admin/retrain
```

#### Example Response

```json
{
  "status": "retrain_dispatched"
}
```

---

### `POST /v1/admin/circuit-breaker`

Manually trip or reset the service circuit breaker for operational maintenance or rolling upgrades.

* **Content-Type:** `application/json`
* **Response:** `application/json`

#### Example Request

```bash
curl -X POST http://127.0.0.1:8080/v1/admin/circuit-breaker \
  -H "Content-Type: application/json" \
  -d '{"tripped": true}'
```

#### Example Response

```json
{
  "circuit_breaker_tripped": true
}
```

---

### `GET /metrics`

Exposes service operational metrics in Prometheus text exposition format.

* **Content-Type:** `text/plain; version=0.0.4`

#### Example Request

```bash
curl http://127.0.0.1:8080/metrics
```

#### Example Output

```text
# HELP predict_requests_total Total number of prediction requests processed
# TYPE predict_requests_total counter
predict_requests_total{endpoint="/predict_csv",status="200"} 12450
# HELP predict_latency_microseconds Latency histogram of inference requests
# TYPE predict_latency_microseconds histogram
predict_latency_microseconds_bucket{le="25"} 11200
predict_latency_microseconds_bucket{le="50"} 12350
predict_latency_microseconds_bucket{le="100"} 12440
predict_latency_microseconds_bucket{le="+Inf"} 12450
predict_latency_microseconds_sum 249000
predict_latency_microseconds_count 12450
# HELP model_sketch_entries_total Active plan sketch entries in memory
# TYPE model_sketch_entries_total gauge
model_sketch_entries_total 1420
# HELP model_q_error_ewma Live EWMA Q-Error across all predictions
# TYPE model_q_error_ewma gauge
model_q_error_ewma 1.14
```

---

## Configuration Reference

The service is configured entirely via environment variables. Standard prefixes `SR_PREDICTOR_*` and `STARROCKS_COST_PREDICTOR_*` are supported interchangeably:

| Variable | Default Value | Type | Description & Production Sizing |
|---|---|---|---|
| `SR_PREDICTOR_BIND_ADDR` | `0.0.0.0:8080` | String | TCP listen address and port for HTTP server. |
| `SR_PREDICTOR_DB_PATH` | `data/cost_predictor.db` | Path | SQLite WAL embedded database file location. |
| `SR_PREDICTOR_FEATURES_LOG_PATH` | *(None)* | Path | Optional path to `fe.features.log` for automatic background file tailing. |
| `SR_PREDICTOR_MIN_PREDICTED_MEM_BYTES` | `1048576` (1 MiB) | Integer | Absolute lower bound clamp for predicted peak memory bytes. |
| `SR_PREDICTOR_FALLBACK_MEM_BYTES` | `67108864` (64 MiB) | Integer | Safe fallback value returned when malformed or unparseable inputs occur. |
| `SR_PREDICTOR_MAX_CLUSTER_MEM_MULTIPLIER` | `2.0` | Float | Multiplier applied to hardware BE profile (`numBeNodes * exp(env_2 + 1)`) to establish upper memory clamp. |
| `SR_PREDICTOR_MAX_SKETCH_ENTRIES` | `100000` | Integer | Maximum capacity of Tier 1 plan-hash sketch cache before LRU eviction. (~15 MB RAM for 100k entries). |
| `SR_PREDICTOR_MIN_NEW_SAMPLES_FOR_RETRAIN` | `200` | Integer | Minimum newly ingested queries required before triggering automated GBDT retrain. |
| `SR_PREDICTOR_RETRAIN_INTERVAL_SECS` | `300` (5 min) | Integer | Periodic timer interval in seconds for background model retraining evaluations. |
| `SR_PREDICTOR_DRIFT_Q_ERROR_THRESHOLD` | `1.2` | Float | EWMA log Q-Error threshold above which drift retraining is immediately triggered ($e^{1.2} \approx 3.32\times$ median error). |

---

## Build, Test & Run

### Prerequisites

* **Rust Toolchain:** Rust 1.85+ (Edition 2024). Install via [rustup](https://rustup.rs):
  ```bash
  rustup update stable
  ```
* **C Compiler:** Standard `cc` or `clang` for `rusqlite` bundled SQLite compilation.

### Building from Source

```bash
# Clone the repository
git clone https://github.com/jimmystewpot/starrocks-cost-prediction-service.git
cd starrocks-cost-prediction-service

# Build optimized release binary
cargo build --release

# The compiled binary is located at target/release/starrocks-cost-predictor
```

### Running the Service

```bash
# Set configuration options
export SR_PREDICTOR_BIND_ADDR="0.0.0.0:8080"
export SR_PREDICTOR_DB_PATH="data/cost_predictor.db"
export SR_PREDICTOR_FEATURES_LOG_PATH="/opt/starrocks/fe/log/fe.features.log"

# Run the daemon
./target/release/starrocks-cost-predictor
```

### Running Tests

```bash
# Run unit tests, integration tests, and benchmarks
cargo test --all-targets --all-features

# Verify formatting
cargo fmt --all -- --check

# Run Clippy linter
cargo clippy --all-targets --all-features -- -D warnings
```

### Docker & Containerized Deployment

A lightweight distroless or Alpine container can run the single static binary:

```dockerfile
FROM rust:1.85-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/starrocks-cost-predictor /app/starrocks-cost-predictor
EXPOSE 8080
VOLUME ["/app/data"]
ENTRYPOINT ["/app/starrocks-cost-predictor"]
```

Run container:
```bash
docker run -d \
  --name starrocks-cost-predictor \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -e SR_PREDICTOR_BIND_ADDR="0.0.0.0:8080" \
  starrocks-cost-predictor:latest
```
