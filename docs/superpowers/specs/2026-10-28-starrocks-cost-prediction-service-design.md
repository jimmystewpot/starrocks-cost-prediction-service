# StarRocks Cost Prediction Service — Architectural Specification

**Date:** 2026-10-28  
**Status:** Reviewed & Approved (Principal Engineer / SRE Hardened)  
**Language / Runtime:** Rust (Edition 2021, Single Static Binary)

---

## 1. Executive Summary & Problem Statement

StarRocks includes a Cost-Based Optimizer (CBO) that estimates query CPU, memory, and cardinality costs using table and column statistics (`row_count`, `data_size`, `ndv`, histograms). However, in production distributed SQL workloads—especially with complex multi-way joins, skewed data distributions, complex scalar predicates, or external catalog tables (Iceberg, Hive, Hudi)—static CBO estimates frequently diverge from actual runtime resource consumption by 1 to 3 orders of magnitude.

To address this, StarRocks Frontend (FE) exposes configuration hooks in `fe.conf`:
```ini
enable_query_cost_prediction = true
query_cost_prediction_service_address = "http://<cost-prediction-service>:8080"
enable_plan_feature_collection = true
```

This specification defines an open-source, high-performance, self-learning **StarRocks Cost Prediction Service** written in Rust. The service ingests query execution telemetry (`fe.features.log`, `fe.audit.log` / `starrocks_audit_tbl__`, and Runtime Query Profiles), maintains an asynchronous continuous machine learning pipeline over embedded analytical storage (SQLite WAL / Parquet storage + optional S3 sync), and serves ultra-low-latency (`p99 < 250µs`, target `~20µs`) predictions for both StarRocks' native `/predict_csv` endpoint and an extended `/v1/predict` multi-target data-size and resource estimation API.

---

## 2. Deep-Dive Analysis of StarRocks FE Internals

Inspection of the StarRocks FE source code (`com.starrocks.sql.optimizer.cost.feature.*`, `StmtExecutor.java`, `SlotEstimatorFactory.java`, `ResourceGroupMgr.java`, `AuditLogBuilder.java`) reveals critical behavioral contracts and operational hazards that govern this design:

### 2.1 Wire Protocol (`CostPredictor.java`)
1. **Health Check (`GET /health_check`):**
   * Polled by a single-threaded `ScheduledExecutorService` daemon every `Config.query_cost_predictor_healthchk_interval` seconds (default **30s**, non-mutable).
   * Sets `lastHealthCheckStatusCode = response.getStatusLine().getStatusCode()`.
   * `ServiceBasedCostPredictor.isAvailable()` returns `Config.enable_query_cost_prediction && lastHealthCheckStatusCode == 200`.
2. **Prediction Request (`POST /predict_csv`):**
   * Triggered synchronously inside `StmtExecutor.handleQueryStmt()` on the query compilation thread right before fragment deployment (`coord.execWithQueryDeployExecutor(context)`).
   * **Content-Type:** `text/csv`.
   * **Request Payload:** Two lines: `PlanFeatures.featuresHeader() + "\n" + planFeatures.toFeatureCsv()`.
   * **Response Payload:** Plain-text string parsed via `(long) Double.parseDouble(responseBody)`, representing predicted peak memory consumption in bytes (`memCostBytes`).
   * **Fallback on Error:** `tryPredictMemoryBytes(ExecPlan)` catches `Exception`, logs a warning, and returns `0L`.

### 2.2 Critical Timeout, Connection Pool & Concurrency Hazards in StarRocks FE
In `CostPredictor.java`, StarRocks initializes its HTTP client via:
```java
private final CloseableHttpClient httpClient = HttpClients.createDefault();
```
1. **No Socket or Connect Timeout:** `HttpClients.createDefault()` does **not** set a socket read timeout (`SO_TIMEOUT` defaults to infinite/OS-level) or connection lease timeout.
2. **Apache HttpClient Default Route Concurrency Cap (2 Connections!):** `HttpClients.createDefault()` uses `PoolingHttpClientConnectionManager` with `defaultMaxPerRoute = 2` and `maxTotal = 20`. Because all requests from a StarRocks FE go to a single `query_cost_prediction_service_address` route, **at most 2 concurrent HTTP requests can be in flight per FE node**, while all other concurrent queries on that FE wait in the connection pool queue!
3. **Unconditional Invocation on Hot Path:** In `StmtExecutor.java`, `if (Config.enable_query_cost_prediction)` calls `predictor.tryPredictMemoryBytes(execPlan)` directly without checking `predictor.isAvailable()`.
4. **Architectural Consequences:**
   * Because at most 2 connections are reused via HTTP/1.1 Keep-Alive per FE node, even a `5ms` response time limits an FE node to `400 queries/sec` before queueing delay explodes! At `25µs` response time, 2 persistent connections can sustain `80,000 queries/sec`.
   * The Rust HTTP server must configure `TCP_NODELAY = true` (disabling Nagle's algorithm so small HTTP responses aren't delayed by 40ms delayed-ACK timers!) and aggressive HTTP/1.1 Keep-Alive idle timeouts (`> 60s`, longer than the 30s health check interval).
   * The inference path must be **100% CPU-bound and synchronous inside the handler** (no internal async task hops, no mutexes, no I/O) so it completes in `< 50µs`.

### 2.3 Downstream Consumers inside StarRocks FE
1. **Query Queue V2 (`SlotEstimatorFactory.MemoryBasedSlotsEstimator`):**
   When `query_queue_slots_estimator_strategy = "MBE"`, if `isAvailable() && coord.getPredictedCost() > 0`, StarRocks replaces the CBO `planMemCost` with `coord.getPredictedCost()` to compute required concurrency slots (`predictedMemBytes / memBytesPerSlot`).
2. **Resource Group Classifier (`ResourceGroupMgr.chooseResourceGroup`):**
   When `isAvailable()`, replaces `planMemCosts` with `predictMemBytes` when evaluating `max_memory_cost` rules.
3. **Audit & Explain (`DefaultCoordinator.java` & `AuditEvent`):**
   Records `predictMemBytes` in the audit event and prints `predicted memory cost: <bytes>` in `EXPLAIN SCHEDULER`.

### 2.4 Feature Vector Layout (`PlanFeatures.java` & `OperatorFeatures.java`)
The CSV row consists of comma-separated tokens across 4 sections:
1. `tables_0, tables_1, tables_2` (3 columns): Top 3 scanned tables ordered by `rowCount` descending (left-padded with `"0"` if fewer than 3 tables).
   * **Important Format Difference:** In `/predict_csv`, `PlanFeatures.toFeatureCsv()` outputs `tables_0,tables_1,tables_2` as the first 3 CSV columns. In `fe.features.log`, `PlanFeatures.toFeatureString()` outputs `tables=[t0,t1,t2]|env=[e0,e1,e2]|var=[v0]|operators=[op0,op1,...]`. Both are produced from the exact same `extractTopTables()` and `operatorFeatureVectors` methods!
2. `env_0, env_1, env_2` (3 columns): Warehouse backend hardware profile:
   * `env_0`: `numBeNodes` (integer)
   * `env_1`: `avgCpuCoreOfBE` (integer)
   * `env_2`: `(long) Math.log1p(memCapacityOfBE)` (integer-truncated natural log of bytes! Note: because Java casts `(long) Math.log1p(...)`, a 64 GiB BE node has $\ln(6.87 \times 10^{10}) \approx 24.95 \rightarrow 24$. Reversing via $\exp(24)$ is $26.4\text{ GiB}$, while $\exp(25)$ is $72.0\text{ GiB}$ — so recovering an upper bound for `memCapacityOfBE` requires $\exp(\text{env\_2} + 1)$).
3. `var_0` (1 column): Session `pipeline_dop` (degree of parallelism).
4. `operators_0 ... operators_{N-1}` ($N$ columns): Iterates over all physical `OperatorType` enum values (`OperatorType.PHYSICAL.ordinal() + 1` to `OperatorType.SCALAR.ordinal() - 1`, skipping `PHYSICAL_MERGE_JOIN`).
   * Every physical operator type emits a fixed-size segment of length `2 + vectorLength(opType)`:
     * `[0]`: `opType.ordinal()`
     * `[1]`: `count` (number of occurrences of this physical operator in the plan)
     * `[2..6]` (Base 5 features, summed across all occurrences of `opType` in the plan):
       * `sum( (long) Math.log1p(outputRowCount) )`
       * `sum( (long) avgRowSize )`
       * `sum( (long) Math.log1p(cboMemoryCost) )`
       * `sum( (long) Math.log1p(cboCpuCost) )`
       * `sum( limit != DEFAULT ? (long) Math.log1p(limit) : 0 )`
     * **Specialized Operator Extra Features (also summed across occurrences):**
       * **Physical Scan Operators (`+4` = 9 features, all collapsed into `PHYSICAL_OLAP_SCAN`):** `numTablets`, `numPartitions`, `numBinaryPredicates`, `numPredicateColumns`.
       * **`PHYSICAL_HASH_JOIN` (`+3` = 8 features):** `encodeLargeSize(rightSize)`, `isBroadcast (1/0)`, `joinOperator.ordinal()`.
       * **`PHYSICAL_HASH_AGG` (`+4` = 9 features):** `encodeLargeSize(inputRows)`, `numGroupByColumns`, `numAggregations`, `(long) aggRatio` where `aggRatio = (1 + inputRows) / (1 + outputRowCount)`.
       * **`PHYSICAL_DISTRIBUTION` (`+1` = 6 features):** `distributionType.ordinal()`.

---

## 3. System Architecture & Plane Separation

The service is packaged as a single standalone Rust binary (`starrocks-cost-predictor`) with four strictly isolated execution planes:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         STARROCKS FRONTEND (FE) NODES                                           │
│  ┌─────────────────────────────────────────┐       ┌─────────────────────────────────────────┐  │
│  │ StmtExecutor (Query Compile Hot Path)   │       │ Telemetry Sources (Post-Execution)      │  │
│  │ • POST /predict_csv (Max 2 Conns/FE!)   │       │ • fe.features.log (PlanFeatures+Actuals)│  │
│  │ • GET  /health_check (Every 30s)        │       │ • starrocks_audit_tbl__ / fe.audit.log  │  │
│  └───────────────────┬─────────────────────┘       │ • FE HTTP /api/profile (Query Profiles) │  │
└──────────────────────┼─────────────────────────────┴────────────────────┬────────────────────┘  │
                       │ Sync HTTP (TCP_NODELAY, p99 < 250µs)             │ Async Push / Tail / Poll
                       ▼                                                  ▼
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
│  ║ • Queries cost-stratified training window from embedded SQLite WAL store                  ║  │
│  ║ • Trains Pure-Rust GBDT trees (mem_p90, mem_p50, scan_bytes, return_rows, cpu_ns, spill)  ║  │
│  ║ • Validates holdout Q-Error before promoting snapshot to ArcSwap + optional S3 sync       ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 SRE Hardening: Why Embedded SQLite WAL (`rusqlite` bundled) Over C++ DuckDB
* **Compile Speed & Memory Predictability:** Embedding `duckdb-rs` compiles ~500k lines of C++ (10+ minute build times) and allocates large internal buffer pools that compete with the OS page cache. By contrast, `rusqlite` with the `bundled` feature compiles in **~5 seconds**, uses $< 10\text{ MB}$ of RAM, supports concurrent non-blocking readers with a single background batch writer in `WAL` (`Write-Ahead Logging`) mode, and stores our 250,000 stratified feature vectors in a single portable file (`data/cost_predictor.db`).

### 3.2 Thread Pool & Network Socket Hardening
* **TCP Socket Tuning:** The HTTP listener explicitly sets `TCP_NODELAY(true)` on every accepted connection and configures HTTP/1.1 `keep_alive` timeout to `90s` (3x the StarRocks 30s health check period) so StarRocks FE's 2-connection pool never pays TCP handshake or Nagle-delay penalties.
* **Payload Size Limit:** Requests to `/predict_csv` are capped at `64 KiB` (typical StarRocks CSV is `~1.5 KiB`) to prevent OOM from rogue clients.
* **Training Thread Pool (`rayon`):** Initialized with `num_threads = max(1, num_cpus / 4)` so background tree building never saturates CPU cores needed by Tokio workers.

---

## 4. Feature Engineering & Self-Synchronizing Segment Decoder

### 4.1 Solving StarRocks Version Skew via Structural Segment DP / Lookahead
In StarRocks, `PlanFeatures` iterates over `OperatorType` between `PHYSICAL` and `SCALAR` (excluding `PHYSICAL_MERGE_JOIN`). Each operator emits a vector of length $L \in \{7, 8, 10, 11\}$ (i.e. $2 + 5 = 7$ for generic operators, $2 + 6 = 8$ for Exchange, $2 + 8 = 10$ for HashJoin, $2 + 9 = 11$ for Scan and HashAgg).

Notice that the first element of every operator segment is `opType.ordinal()`, which is **strictly monotonically increasing** (specifically, `prev_ordinal + 1`, or `prev_ordinal + 2` across the skipped `PHYSICAL_MERGE_JOIN`)!
Our decoder (`src/features/decoder.rs`) implements a **Self-Synchronizing Ordinal Segment Walker**:
1. Given an `operators` slice of `i64` values, it starts at index `pos = 0` with `first_ordinal = ops[0]`.
2. To determine the length $L \in \{7, 8, 10, 11\}$ of the current segment at `pos`, it checks which candidate step $L \in \{7, 8, 10, 11\}$ lands on `pos + L == ops.len()` OR where `ops[pos + L]` equals `ops[pos] + 1` or `ops[pos] + 2` (and resolves any local ambiguity via 2-step lookahead / dynamic programming path from `0` to `ops.len()`)!
3. Once segmented, the decoder classifies each segment by its segment length $L$ and internal invariants:
   * $L = 8 \implies$ **Exchange (`PHYSICAL_DISTRIBUTION`)**
   * $L = 10 \implies$ **HashJoin (`PHYSICAL_HASH_JOIN`)**
   * $L = 11$ and earliest ordinal in the scan block $\implies$ **Scan (`PHYSICAL_OLAP_SCAN` collapsed bucket)**
   * $L = 11$ and appears before/outside the scan block $\implies$ **HashAgg (`PHYSICAL_HASH_AGG`)**
   * $L = 7 \implies$ **Generic Physical Operator** (Project, Sort, TopN, Window, Filter, Union, CTE, etc.)

This guarantees **100% immunity to StarRocks minor/major version upgrades** even if StarRocks adds or reorders enum values!

### 4.2 Correcting the "Summed Log" Mathematical Artifact
In StarRocks' `PlanFeatures.AggregatedFeature.merge()`, when multiple operators of the same type exist in a plan (e.g., $k = 3$ `PHYSICAL_HASH_JOIN` operators), StarRocks sums their `(long) Math.log1p()` values:
$$S = \sum_{i=1}^k \lfloor \ln(1 + x_i) \rfloor$$
For every operator segment with `count > 0`, the decoder computes:
* **Mean Log Feature:** $\bar{z} = \frac{S}{\text{count}}$ (geometric mean in log-space).
* **Estimated Log of Sum:** $\tilde{z}_{\text{sum}} = \bar{z} + \ln(\text{count})$.
* **24 Canonical Dense Features (`DenseFeatureVector`, `f64` array of length 39 = 24 plan features + 15 table features):**
  1. `total_physical_ops`
  2. `active_operator_types`
  3. `scan_count`
  4. `scan_mean_log_rows`
  5. `scan_est_log_sum_rows`
  6. `scan_mean_row_size`
  7. `scan_total_tablets_log` ($\ln(1 + \text{numTablets})$)
  8. `scan_total_partitions_log` ($\ln(1 + \text{numPartitions})$)
  9. `scan_total_predicates`
  10. `join_count`
  11. `broadcast_join_count`
  12. `partition_join_count` (`join_count - broadcast_join_count`)
  13. `join_mean_right_log_size`
  14. `join_est_sum_right_log_size`
  15. `agg_count`
  16. `agg_mean_log_input_rows`
  17. `agg_total_group_bys`
  18. `agg_total_aggregations`
  19. `agg_mean_log_ratio` ($\ln(1 + \text{aggRatio} / \text{agg\_count})$)
  20. `exchange_count`
  21. `generic_ops_count`
  22. `cbo_est_total_mem_log` ($\ln\left(1 + \sum_{\text{active ops}} \text{count}_j \cdot \exp(\bar{z}_{\text{mem}, j})\right)$)
  23. `cbo_est_total_cpu_log` ($\ln\left(1 + \sum_{\text{active ops}} \text{count}_j \cdot \exp(\bar{z}_{\text{cpu}, j})\right)$)
  24. `env_dop_log` ($\ln(1 + \text{dop}) + \ln(1 + \text{numBeNodes})$)

### 4.3 Target-Encoded Table Telemetry Catalog (`tables_0..2`)
The service maintains an in-memory `TableCatalog` (`DashMap<String, TableProfile>`) updated continuously from ingested executions. For each of the 3 table slots ($i \in \{0, 1, 2\}$), 5 features (total 15 features) are appended to form the 39-element `DenseFeatureVector`:
1. `table_i_known` (`1.0` if non-`"0"` and `exec_count >= 5`, else `0.0`)
2. `table_i_ewma_log_scan_bytes` (default `0.0` if unknown)
3. `table_i_ewma_log_bytes_per_row` (default `4.0` $\approx 54\text{ B/row}$ if unknown)
4. `table_i_mem_multiplier_log` (historical EWMA of $\ln(1 + \text{actualMem}) - \text{cbo\_est\_total\_mem\_log}$, default `0.0`)
5. `table_i_spill_rate` (historical fraction of queries scanning this table with `spilledBytes > 0`, default `0.0`)

---

## 5. The Cascading 3-Tier ML Architecture

All regression targets (`mem_bytes`, `scan_bytes`, `return_rows`, `cpu_cost_ns`) are modeled in **natural log-space**:
$$y = \ln(1 + \text{actual\_value}), \quad \hat{V} = \max(0.0, \exp(\hat{y}) - 1.0)$$

Ground-truth memory labels incorporate spill compensation and OOM censoring:
* **Normal Completion (`state IN ('EOF', 'OK')`):**
  $$\text{effective\_mem\_bytes} = \text{memCostBytes} + \text{spilledBytes}$$
* **OOM / Memory Limit Exceeded (`state = 'ERR'` and error/memory threshold met):**
  $$\text{effective\_mem\_bytes} = \max(\text{memCostBytes} + \text{spilledBytes}, \text{memCostBytes} \times 1.5)$$

### 5.1 Tier 1: Structural Plan-Signature Sketch (`~2µs` Latency, Zero Training Lag)
Captures recurring query templates:
* **Signature Hash (`u64` via `xxhash-rust` `xxh3_64`):** Computed deterministically over:
  * Sorted `tables_0..2` strings
  * Sequence of `(segment_index, count, is_broadcast_count, join_type_sum, agg_group_by_sum)` for all segments where `count > 0`.
  * Excludes row counts, byte sizes, and CBO cost numbers so identical SQL templates on growing tables produce the exact same `u64` signature.
* **Tracked State per Signature (`PlanSketchEntry`):**
  * `hit_count: u64`, `last_seen_epoch_s: u64`
  * `ewma_log_mem: f64`, `ewma_var_log_mem: f64` (EWMA factor $\alpha = 0.20$; variance floor `0.04`)
  * `ewma_log_scan_bytes: f64`, `ewma_log_return_rows: f64`, `ewma_log_cpu_ns: f64`, `ewma_spill_rate: f64`
  * `ewma_ref_scan_log_rows: f64` (reference input cardinality)
* **Cardinality-Scaled Inference:**
  Let $\Delta r = \text{clamp}(\text{scan\_est\_log\_sum\_rows} - \text{ewma\_ref\_scan\_log\_rows}, -2.0, 2.0)$ and elasticity $\gamma = 0.5$:
  $$\hat{y}_{\text{tier1, p50}} = \text{ewma\_log\_mem} + \gamma \cdot \Delta r$$
  $$\hat{y}_{\text{tier1, p90}} = \hat{y}_{\text{tier1, p50}} + 1.2816 \sqrt{\max(0.04, \text{ewma\_var\_log\_mem})}$$
* **Bounded LRU Eviction:** Capped at `max_sketch_entries` (default `100,000`).

### 5.2 Tier 2: Multi-Target Pure-Rust GBDT Ensemble (`~15µs` Latency)
Generalizes across unseen query plans and multi-table interactions:
* **Engine:** Self-contained Pure-Rust Histogram/Exact Gradient Boosted Decision Tree regressor (`src/model/gbdt.rs`) with flat cache-aligned contiguous `Vec<TreeNode>` arrays and zero C++/FFI dependencies, supporting both **Huber loss (P50)** and **Pinball / Quantile loss ($\alpha = 0.90$ for P90)**!
  *(Why an idiomatic native Rust GBDT implementation in `src/model/gbdt.rs`? External crates either pull in C++ compilers or lack native quantile pinball loss + zero-copy `serde` serialization. A clean 250-line exact/binned CART boosted tree learner with Pinball & Huber gradients in pure Rust compiles in 1 second, serializes directly via `bincode`, and runs 100 trees of depth 5 in `< 3µs`!)*
* **Trained Ensembles in `GbdtModelSet`:**
  1. `mem_p90`: Pinball loss ($\alpha = 0.90$) on $\ln(1 + \text{effective\_mem\_bytes})$.
  2. `mem_p50`: Huber loss ($\delta = 1.0$) on $\ln(1 + \text{effective\_mem\_bytes})$.
  3. `scan_bytes_p50`: Huber loss on $\ln(1 + \text{scanBytes})$.
  4. `return_rows_p50`: Huber loss on $\ln(1 + \text{returnRows})$.
  5. `cpu_ns_p50`: Huber loss on $\ln(1 + \text{cpuCostNs})$.
  6. `spill_prob`: Logistic / squared-error probability tree on $\mathbb{I}(\text{spilledBytes} > 0)$.

### 5.3 Tier 3: Online Recursive Least Squares (RLS) CBO Calibrator (`~1µs` Latency)
Provides immediate cold-start protection on a brand-new cluster or when a query touches 100% unseen tables:
* 6-dimensional linear regressor $\mathbf{w} \in \mathbb{R}^6$ with $6 \times 6$ covariance matrix $\mathbf{P}$ and forgetting factor $\lambda = 0.995$:
  $$\mathbf{x}_{\text{rls}} = \begin{bmatrix} 1.0 & \text{cbo\_est\_total\_mem\_log} & \text{scan\_est\_log\_sum\_rows} & \ln(1 + \text{join\_count}) & \ln(1 + \text{agg\_count}) & \text{env\_dop\_log} \end{bmatrix}^T$$
* **Cold-Start Prior:** Initialized at $\mathbf{w}_0 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]^T$ and $\mathbf{P}_0 = 10.0 \cdot \mathbf{I}_6$. When `cbo_est_total_mem_log > 0`, it initially predicts the CBO estimate directly (`p90` adds $+0.693 \approx 2\times$ safety margin until empirical residual variance is learned).
* **Covariance Stabilization:** Every update enforces symmetry $\mathbf{P} \leftarrow \frac{1}{2}(\mathbf{P} + \mathbf{P}^T)$ and diagonal clamping $P_{ii} \in [10^{-6}, 10^4]$ so long-running clusters never suffer floating-point drift.

### 5.4 Continuous Confidence Blending & Safety Clamping
1. **Tier 1 Confidence ($c_1 \in [0, 0.95]$):**
   $$c_1 = \min\left(0.95, \frac{n_{\text{hits}}}{n_{\text{hits}} + 5} \cdot \exp(-0.20 \cdot \text{ewma\_var\_log\_mem}) \cdot \exp(-0.15 \cdot |\Delta r|)\right)$$
2. **Tier 2 Confidence ($c_2 \in [0, 0.90]$):**
   $$c_2 = \mathbb{I}(\text{gbdt\_trained}) \cdot (0.45 + 0.45 \cdot \text{known\_table\_ratio})$$
3. **Normalized Weights:**
   * $w_1 = c_1$
   * $w_2 = (1.0 - w_1) \cdot c_2$
   * $w_3 = 1.0 - w_1 - w_2$
   * $\hat{y}_{\text{final}} = w_1 \hat{y}_1 + w_2 \hat{y}_2 + w_3 \hat{y}_3$
4. **Safety Bounds:**
   * Upper bound from `env_2`: Since `env_2` is $\lfloor \ln(1 + \text{memCapacityOfBE}) \rfloor$, per-BE upper bound is $\exp(\min(40, \text{env\_2} + 1))$. Cluster upper bound is `max_bytes = max(default_max_mem_bytes, numBeNodes * exp(env_2 + 1) * 2.0)`.
   * Final predicted bytes are clamped to `[config.min_predicted_mem_bytes (default 1 MiB = 1_048_576), max_bytes]`.

---

## 6. Multi-Source Ingestion, Correlation & SQLite WAL Storage

### 6.1 Supported Ingestion Sources
1. **Source A — `fe.features.log` (File Tailer + `POST /v1/ingest/features`):**
   * Parses pipe-delimited records written by `AuditLogBuilder.java`:
     `digest=...|cpuCostNs=...|memCostBytes=...|scanBytes=...|scanRows=...|returnRows=...|spilledBytes=...|time=...|state=...|catalog=...|database=...|tables=[...]|env=[...]|var=[...]|operators=[...]`
   * Tolerates optional log4j timestamp/level prefixes before `digest=`.
   * File tailer tracks `(inode, byte_offset)` to resume cleanly across log rotations.
2. **Source B — `fe.audit.log` / `starrocks_audit_tbl__` (`POST /v1/ingest/audit`):**
   * **In-Flight Prediction Correlation Cache (`InFlightCache`):** Bounded ring-buffer (capacity `50,000`, TTL `30m`). When `/predict_csv` runs, we add a tiny deterministic 10-bit fingerprint `(plan_hash & 0x3FF)` to the returned `predicted_mem_bytes` (jittering the byte count by $< 1\text{ KiB}$ out of megabytes/gigabytes!). When the completed audit event arrives with `predictMemBytes`, matching `(predictMemBytes)` in `InFlightCache` is practically collision-free!
3. **Source C — Runtime Query Profiles (`POST /v1/ingest/profile`):**
   * Parses StarRocks Query Profiles (JSON or text tree) to extract operator-level `PeakMemoryUsage`, `RowsReturned`, and `SpillBytes`, updating `TableCatalog` and operator statistics.

### 6.2 Embedded SQLite WAL Schema & Cost-Stratified Retention
Database (`data/cost_predictor.db`) initialized with `PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;`:
* `telemetry_samples`:
  * `id INTEGER PRIMARY KEY AUTOINCREMENT`
  * `ingested_at_s INTEGER NOT NULL`
  * `plan_hash INTEGER NOT NULL`
  * `mem_stratum INTEGER NOT NULL` (`0`: `<10MB`, `1`: `10-100MB`, `2`: `100MB-1GB`, `3`: `1-10GB`, `4`: `>10GB`)
  * `features_blob BLOB NOT NULL` (serialized `DenseFeatureVector`)
  * `effective_mem_bytes INTEGER NOT NULL`
  * `scan_bytes INTEGER NOT NULL`
  * `return_rows INTEGER NOT NULL`
  * `cpu_cost_ns INTEGER NOT NULL`
  * `spilled_bytes INTEGER NOT NULL`
* `model_snapshots`:
  * `version INTEGER PRIMARY KEY AUTOINCREMENT`
  * `created_at_s INTEGER NOT NULL`
  * `metrics_json TEXT NOT NULL`
  * `snapshot_blob BLOB NOT NULL` (serialized `ModelSnapshot`)
* **Cost-Stratified Compaction:** Retains up to `50,000` samples per `mem_stratum` (`250,000` total max).

### 6.3 Continuous Training, Drift Detection & Champion/Challenger Gate
* **Live Drift Detector:** Computes Q-Error $q = \max\left(\frac{\hat{V} + 1}{V + 1}, \frac{V + 1}{\hat{V} + 1}\right)$ on each ingested query against the active ensemble's prediction, maintaining an EWMA of $\ln(q)$. If `ewma_log_q_error` exceeds `drift_threshold` (default `1.2`, i.e., $>3.3\times$ error) after at least 50 samples, a drift retrain flag is raised.
* **Champion/Challenger Validation:**
  * Trains `GbdtModelSet` on 85% of samples, evaluates on 15% holdout.
  * Promotes the new `GbdtModelSet` into `ArcSwap` if no model is currently trained OR if Challenger holdout P90 Pinball Loss $\le 1.05 \times$ Champion holdout loss.

---

## 7. API Specification

### 7.1 StarRocks Native Endpoints
| Method & Path | Content-Type | Description | Latency Target |
|---|---|---|---|
| `POST /predict_csv` | `text/csv` | Accepts 2-line StarRocks CSV (`header\nrow`). Returns plain-text integer string of predicted memory bytes (`p90`). Never returns HTTP 500 on malformed input (returns safe fallback with `200 OK`). | `p50 < 25µs`, `p99 < 250µs` |
| `GET /health_check` | `text/plain` | Returns `200 OK` when healthy and circuit breaker is closed. Returns `503 Service Unavailable` when circuit breaker is tripped. | `< 20µs` |

### 7.2 Extended Prediction, Ingestion & Operational Endpoints
| Method & Path | Description |
|---|---|
| `POST /v1/predict` | Accepts StarRocks CSV (`text/csv`) or JSON. Returns JSON: `predicted_mem_bytes_p90`, `predicted_mem_bytes_p50`, `predicted_scan_bytes`, `predicted_return_rows`, `predicted_cpu_cost_ns`, `spill_probability`, `confidence`, `tier_weights`, `dominant_tier`, `model_version`, `latency_us`. |
| `POST /v1/ingest/features` | Ingests single or multi-line `fe.features.log` text payloads. Returns `202 Accepted` with `{ "accepted": N }`. |
| `POST /v1/ingest/audit` | Ingests JSON array/object of StarRocks audit events. Correlates with `InFlightCache` and updates models. |
| `POST /v1/ingest/profile` | Ingests StarRocks Runtime Query Profile JSON/text payloads. |
| `GET /v1/models/status` | Returns JSON status: active `model_version`, `gbdt_trained`, `sketch_entries`, `known_tables`, `ingested_samples`, `live_q_error_ewma`, `circuit_breaker_tripped`, `holdout_metrics`. |
| `POST /v1/admin/retrain` | Triggers an immediate background GBDT retrain over stored samples and returns updated model status. |
| `POST /v1/admin/circuit-breaker` | Accepts `{ "tripped": bool }`. Toggles `/health_check` between `200 OK` and `503 Service Unavailable`. |
| `GET /metrics` | Prometheus text exposition format metrics. |
