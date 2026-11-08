# AGENTS.md — Developer & AI Agent Guide

This document is the authoritative engineering manual and operational specification for developers and AI coding agents working on the **StarRocks Cost Prediction Service** (`starrocks-cost-prediction-service`).

---

## 1. Project Purpose & Architectural Invariants

The StarRocks Cost Prediction Service is a standalone, ultra-low-latency, self-learning daemon written in Rust (Edition 2024). It serves runtime resource cost estimates (P90 memory, P50 memory, scan bytes, return rows, CPU nanoseconds, spill probability) to the StarRocks Frontend (FE) for Query Queue V2 `MBE` slot allocation and Resource Group classification.

### Critical StarRocks FE Hot-Path Invariants

All developers and AI agents modifying HTTP handlers, serialization, or inference paths MUST adhere to the following non-negotiable operational invariants:

#### 1. Apache HttpClient Default Concurrency Cap (2 Connections Per FE Node)
In StarRocks FE (`CostPredictor.java`), the HTTP client is initialized as:
```java
private final CloseableHttpClient httpClient = HttpClients.createDefault();
```
* `HttpClients.createDefault()` configures a `PoolingHttpClientConnectionManager` with **`defaultMaxPerRoute = 2`** and `maxTotal = 20`.
* Because all requests from a StarRocks FE node target the single `query_cost_prediction_service_address` route, **at most 2 concurrent HTTP requests can be in flight per FE node**. All other concurrent queries on that FE wait in an in-memory queue.
* **Throughput Math:**
  * At a $5\text{ ms}$ response latency, an FE node is choked at $\frac{2}{0.005} = 400\text{ queries/sec}$.
  * At a $25\ \mu\text{s}$ response latency, the same 2 persistent connections sustain $\frac{2}{0.000025} = 80,000\text{ queries/sec}$.
* **Invariant:** Hot-path inference (`/predict_csv` and `/v1/predict`) must execute in **pure synchronous CPU operations** (< $25\ \mu\text{s}$, p99 < $250\ \mu\text{s}$). Never block on file I/O, database queries, network calls, mutex acquisition, or Tokio task yielding inside the request handler.

#### 2. StarRocks FE Has No Socket Timeout
* `HttpClients.createDefault()` sets **NO socket read timeout** (`SO_TIMEOUT = 0`, indefinite).
* If the Rust service hangs, deadlocks, or fails to flush a response, the StarRocks FE query compilation thread **hangs indefinitely**, freezing client queries.
* **Invariant:** Every request handler must be bounded, panic-safe, and self-contained.

#### 3. TCP_NODELAY & HTTP/1.1 Keep-Alive Idle Timeouts
* The HTTP listener must explicitly set **`TCP_NODELAY = true`** on every accepted socket. Because responses to `/predict_csv` are tiny plain-text ASCII numbers (~10–12 bytes), failing to set `TCP_NODELAY` causes Nagle's algorithm to trigger delayed-ACK stalls of up to $40\text{ ms}$.
* StarRocks polls `/health_check` every 30 seconds (`query_cost_predictor_healthchk_interval = 30`). HTTP/1.1 Keep-Alive idle timeout on the server must be configured to at least **`90s`** (3x the poll interval) to keep persistent connections open and avoid connection re-establishment latency.

#### 4. The Zero-HTTP-500 Policy on `/predict_csv`
* StarRocks FE's `CostPredictor.java` catches exceptions and falls back to `0L`, disabling ML cost estimation for that query.
* Under NO circumstances should `POST /predict_csv` return HTTP 500, 400, or drop connections on corrupt CSV, malformed lines, unexpected tokens, or NaN/Inf floats.
* **Invariant:** If tokenization, decoding, or feature extraction fails, the handler MUST log a warning at `DEBUG`/`WARN` level and immediately return HTTP `200 OK` with `config.fallback_mem_bytes` (default `67,108,864` = 64 MiB).

---

## 2. Codebase Module Structure

```
src/
├── config.rs           # Configuration parser for environment variables with fallback defaults
├── types.rs            # Core domain types, DenseFeatureVector (39), watermark & penalty formulas
├── lib.rs              # Crate library root and module exports
├── main.rs             # Daemon entrypoint, signal handling, server startup
│
├── features/           # Feature extraction & telemetry catalog
│   ├── mod.rs          # Feature submodule re-exports
│   ├── decoder.rs      # Self-synchronizing segment DP decoder & 24 canonical plan features
│   └── catalog.rs      # TableCatalog with DashMap for target-encoded table profiles (15 features)
│
├── model/              # 3-Tier ML Model Ensemble
│   ├── mod.rs          # Model submodule re-exports
│   ├── ensemble.rs     # EnsembleEngine, ModelSnapshot, and ArcSwap pointer management
│   ├── sketch.rs       # Tier 1: Plan-hash EWMA sketch cache with cardinality scaling
│   ├── gbdt.rs         # Tier 2: Pure-Rust GBDT regressor (Huber loss & P90 Pinball loss)
│   └── rls.rs          # Tier 3: Online Recursive Least Squares (RLS) CBO calibrator
│
├── ingest/             # Asynchronous query telemetry ingestion
│   ├── mod.rs          # Ingestion submodule re-exports
│   ├── parser.rs       # fe.features.log parser, AuditLogEvent parser, and profile parser
│   ├── correlator.rs   # 10-bit watermarked InFlightCache correlator
│   └── tailer.rs       # Resilient checkpointed FileTailer for fe.features.log
│
├── storage/            # Embedded analytical persistence
│   ├── mod.rs          # Storage submodule re-exports
│   └── sqlite.rs       # SQLite WAL embedded engine, stratified sample compaction, model snapshots
│
├── trainer/            # Background retraining pipeline (dedicated Rayon thread pool)
│   └── mod.rs          # Holdout split, P90 Pinball & Q-Error evaluation, champion/challenger gate
│
└── api/                # Axum HTTP routes & handlers
    ├── mod.rs          # AppState, ingestion concurrency semaphore, and router construction
    ├── predict.rs      # Hot-path /predict_csv and /v1/predict handlers
    ├── ingest.rs       # Rate-limited, batched /v1/ingest/{features,audit,profile} handlers
    └── admin.rs        # /health_check, /v1/models/status, /v1/admin/*, and /metrics handlers
```

---

## 3. Mathematical Invariants & Engineering Formulas

### 3.1 Summed-Log Normalization (`decoder.rs`)
In StarRocks `PlanFeatures.AggregatedFeature.merge()`, when multiple operators of the same type exist in a plan (e.g. $k = 3$ `PHYSICAL_HASH_JOIN` operators), StarRocks sums their integer log features:
$$S = \sum_{i=1}^k \lfloor \ln(1 + x_i) \rfloor$$

If used naively, this creates severe non-linear distortion. For every operator segment with $\text{count} > 0$, the decoder reconstructs:
* **Mean Log Feature:** $\bar{z} = \frac{S}{\text{count}}$ (geometric mean in log space).
* **Estimated Log of Sum:** $\tilde{z}_{\text{sum}} = \bar{z} + \ln(\text{count})$.

### 3.2 Hardware Capacity `env_2` Java Truncation Compensation (`types.rs`)
In StarRocks `PlanFeatures.java`, warehouse backend memory capacity is emitted as:
```java
(long) Math.log1p(memCapacityOfBE)
```
Java's integer cast truncates the natural logarithm:
* For a 64 GiB BE node: $6.87 \times 10^{10}\text{ bytes} \implies \ln(1 + 6.87 \times 10^{10}) \approx 24.95 \implies 24$.
* Reversing via $\exp(24)$ yields only $2.64 \times 10^{10}\text{ bytes} = 26.4\text{ GiB}$ (underestimating by $59\%$).
* Reversing via $\exp(24 + 1) = \exp(25)$ yields $7.20 \times 10^{10}\text{ bytes} = 72.0\text{ GiB}$ (safe upper bound).
* **Invariant:** Cluster memory upper bound must be computed as:
  $$\text{max\_bytes} = \text{numBeNodes} \times \exp(\min(40, \text{env\_2} + 1)) \times \text{multiplier}$$

### 3.3 10-Bit Audit Correlation Watermark (`types.rs`)
To correlate `/predict_csv` predictions with subsequent `fe.audit.log` query completion records without modifying StarRocks FE wire protocol:
$$\text{watermarked\_bytes} = (\text{raw\_bytes} \ \& \ \neg\text{0x3FF}) \mid (\text{plan\_hash} \ \& \ \text{0x3FF})$$
* Embeds the lowest 10 bits of `plan_hash` into the byte value.
* Maximum jitter is $1,023\text{ bytes}$ (< $0.1\%$ on allocations $\ge 1\text{ MiB}$).
* Correlated in `InFlightCache` ring buffer ($50,000$ slots, 30-minute TTL).

### 3.4 Effective Memory Ground Truth (`types.rs`)
Telemetry labels must account for memory spilling and OOM aborts:
* **Normal Completion (`state` is `EOF` or `OK`):**
  $$\text{effective\_mem} = \text{memCostBytes} + \text{spilledBytes}$$
* **OOM / Query Failure (`state` starts with `ERR`):**
  $$\text{effective\_mem} = \max(\text{memCostBytes} + \text{spilledBytes}, \text{memCostBytes} \times 1.5)$$
  *(Applies a 50% penalty to simulate the memory requirement that would have avoided the OOM).*

### 3.5 Quantile Pinball Loss ($\alpha = 0.90$) (`gbdt.rs`)
For P90 memory peak predictions, under-prediction causes query thrashing and OOM, whereas over-prediction merely wastes a queue slot. The GBDT ensemble uses asymmetric Pinball loss:
$$\mathcal{L}_{\alpha}(y, \hat{y}) = \begin{cases} \alpha (y - \hat{y}) & \text{if } y \ge \hat{y} \\ (1 - \alpha)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$
With $\alpha = 0.90$, under-estimation errors are penalized $9\times$ more heavily than over-estimation errors.

### 3.6 Online RLS Covariance Stabilization (`rls.rs`)
Recursive Least Squares with forgetting factor $\lambda = 0.995$:
* Covariance matrix update: $\mathbf{P} \leftarrow \frac{1}{\lambda} \left(\mathbf{P} - \frac{\mathbf{P} \mathbf{x} \mathbf{x}^T \mathbf{P}}{\lambda + \mathbf{x}^T \mathbf{P} \mathbf{x}}\right)$.
* To prevent numerical breakdown over millions of iterations:
  * Symmetry is enforced on every update: $\mathbf{P} \leftarrow \frac{1}{2}(\mathbf{P} + \mathbf{P}^T)$.
  * Diagonal elements are clamped: $P_{ii} \in [10^{-6}, 10^4]$.

---

## 4. TDD & Verification Commands

Before creating a pull request or completing any task, run the following verification pipeline:

```bash
# 1. Format verification (must produce zero diff)
cargo fmt --all -- --check

# 2. Strict Clippy linting (warnings are treated as errors)
cargo clippy --all-targets --all-features -- -D warnings

# 3. Unit, integration, and doc tests
cargo test --all-targets --all-features
```

### Guidelines for Adding Features
* **Zero FFI / C++ Dependencies:** Do not add external C++ libraries. The binary must compile quickly and reliably using pure Rust and bundled SQLite.
* **No Dynamic Allocation in Hot Path:** Avoid heap allocations (`String`, `Vec`, `Box`) inside the prediction loop. Reuse pre-allocated arrays (`DenseFeatureVector = [f64; 39]`).
* **ArcSwap for Model Swaps:** Model updates are published via atomic pointer swap (`ArcSwap::store`). Readers load the pointer with `ArcSwap::load()`, which compiles to a single CPU load instruction (~1ns) without locking.
* **Test Coverage:** Every bugfix or new feature must be accompanied by unit tests covering edge cases (e.g., zero operators, empty CSV, infinite values, out-of-order ordinals).
