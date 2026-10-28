# StarRocks Cost Prediction Service — Architectural Specification

**Date:** 2026-10-28  
**Status:** Draft for Review  
**Language / Runtime:** Rust (Edition 2021/2024, Single Binary)

---

## 1. Executive Summary & Problem Statement

StarRocks includes a Cost-Based Optimizer (CBO) that estimates query CPU, memory, and cardinality costs using table and column statistics (`row_count`, `data_size`, `ndv`, histograms). However, in production distributed SQL workloads—especially with complex multi-way joins, skewed data distributions, complex scalar predicates, or external catalog tables (Iceberg, Hive, Hudi)—static CBO estimates frequently diverge from actual runtime resource consumption by 1 to 3 orders of magnitude.

To address this, StarRocks Frontend (FE) exposes two configuration hooks in `fe.conf`:
```ini
enable_query_cost_prediction = true
query_cost_prediction_service_address = "http://<cost-prediction-service>:8080"
enable_plan_feature_collection = true
```

This specification defines an open-source, high-performance, self-learning **StarRocks Cost Prediction Service** written in Rust. The service ingests query execution telemetry (`fe.features.log`, `fe.audit.log` / `starrocks_audit_tbl__`, and Runtime Query Profiles), maintains an asynchronous continuous machine learning pipeline over embedded analytical storage (DuckDB + optional S3 sync), and serves ultra-low-latency (`p99 < 500µs`, target `~20µs`) predictions for both StarRocks' native `/predict_csv` endpoint and an extended `/v1/predict` multi-target data-size and resource estimation API.

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

### 2.2 Critical Timeout & Concurrency Hazard in StarRocks FE
In `CostPredictor.java`, StarRocks initializes its HTTP client via:
```java
private final CloseableHttpClient httpClient = HttpClients.createDefault();
```
* **No Socket or Connect Timeout:** `HttpClients.createDefault()` does **not** set a socket read timeout (`SO_TIMEOUT` defaults to infinite/OS-level) or connection lease timeout.
* **Unconditional Invocation on Hot Path:** In `StmtExecutor.java`, `if (Config.enable_query_cost_prediction)` calls `predictor.tryPredictMemoryBytes(execPlan)` directly without checking `predictor.isAvailable()`.
* **Architectural Consequence:** If the prediction service stalls, experiences garbage collection pauses, or blocks on model training locks, **every `SELECT` query compilation thread in StarRocks FE will block**, exhausting FE connection pools. Therefore, the inference plane must be **wait-free, zero-I/O, CPU-bounded, and protected by a strict internal server-side deadline (`< 2ms`)**.

### 2.3 Downstream Consumers inside StarRocks FE
1. **Query Queue V2 (`SlotEstimatorFactory.MemoryBasedSlotsEstimator`):**
   When `query_queue_slots_estimator_strategy = "MBE"`, if `isAvailable() && coord.getPredictedCost() > 0`, StarRocks replaces the CBO `planMemCost` with `coord.getPredictedCost()` to compute required concurrency slots (`predictedMemBytes / memBytesPerSlot`).
2. **Resource Group Classifier (`ResourceGroupMgr.chooseResourceGroup`):**
   When `isAvailable()`, replaces `planMemCosts` with `predictMemBytes` when evaluating `max_memory_cost` rules (e.g., routing large queries away from `short_query` resource groups).
3. **Audit & Explain (`DefaultCoordinator.java` & `AuditEvent`):**
   Records `predictMemBytes` in the audit event and prints `predicted memory cost: <bytes>` in `EXPLAIN SCHEDULER`.

### 2.4 Feature Vector Layout (`PlanFeatures.java` & `OperatorFeatures.java`)
The CSV row consists of comma-separated tokens across 4 sections:
1. `tables_0, tables_1, tables_2` (3 columns): Top 3 scanned tables ordered by `rowCount` descending (left-padded with `"0"` if fewer than 3 tables). For native `OlapTable`, this is the numeric `Table.getId()` string; for external catalog tables, it is `"catalogDbName.catalogTableName"`.
2. `env_0, env_1, env_2` (3 columns): Warehouse backend hardware profile:
   * `env_0`: `numBeNodes` (integer)
   * `env_1`: `avgCpuCoreOfBE` (integer)
   * `env_2`: `(long) Math.log1p(memCapacityOfBE)` (log-bytes)
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
       * **Physical Scan Operators (`+4` = 9 features, all collapsed into `PHYSICAL_OLAP_SCAN`):** `numTablets`, `numPartitions`, `numBinaryPredicates`, `numPredicateColumns`. (All other `PHYSICAL_*_SCAN` slots remain `count = 0` with 9 zeros).
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
│  │ • POST /predict_csv (No Socket Timeout!)│       │ • fe.features.log (PlanFeatures+Actuals)│  │
│  │ • GET  /health_check (Every 30s)        │       │ • starrocks_audit_tbl__ / fe.audit.log  │  │
│  └───────────────────┬─────────────────────┘       │ • FE HTTP /api/profile (Query Profiles) │  │
└──────────────────────┼─────────────────────────────┴────────────────────┬────────────────────┘  │
                       │ Sync HTTP (Target: p99 < 500µs)                  │ Async Push / Tail / Poll
                       ▼                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                 starrocks-cost-predictor (Single Rust Binary)                                   │
│                                                                                                 │
│  ╔═════════════════════════════════════════╗       ╔═════════════════════════════════════════╗  │
│  ║ 1. HOT INFERENCE PLANE (Tokio / Axum)   ║       ║ 2. ASYNC INGESTION PLANE                ║  │
│  ║ • Zero-alloc CSV Tokenizer              ║       ║ • File Tailer (notify + inode seek)     ║  │
│  ║ • Schema-Fingerprinted Feature Decoder  ║       ║ • HTTP Ingest API (/v1/ingest/*)        ║  │
│  ║ • Reads ModelSnapshot via ArcSwap       ║       ║ • StarRocks Audit & Profile Poller      ║  │
│  ║ • Server-Side Timeout Guard (2ms cap)   ║       ║ • Spill-Adjusted Label Normalizer       ║  │
│  ╚═══════════════════╤═════════════════════╝       ╚════════════╤══════════════════╤═════════╝  │
│                      │ Lock-Free Pointer Load (~1ns)            │ Real-time        │ Batched    │
│                      ▼                                          ▼ (<1ms)           ▼ Append     │
│  ╔═══════════════════════════════════════════════════════════════════════╗  ┌────────────────┐  │
│  ║ 3. IN-MEMORY MODEL SNAPSHOT & LIVE SKETCHES                           ║  │ Embedded Store │  │
│  ║ ┌─────────────────────────┐ ┌───────────────────┐ ┌─────────────────┐ ║  │ (DuckDB +      │  │
│  ║ │ Tier 1: DashMap Sketch  │ │ Tier 2: ArcSwap   │ │ Tier 3: Atomic  │ ║  │  Model Blobs + │  │
│  ║ │ PlanHash ──► EWMA P50/90│ │ Multi-Target GBDT │ │ Online RLS CBO  │ ║  │  Opt. S3 Sync) │  │
│  ║ │ & Row-Scaling Factor    │ │ Ensemble (P50/P90)│ │ Calibrator      │ ║  └────────┬───────┘  │
│  ║ └─────────────────────────┘ └─────────▲─────────┘ └─────────────────┘ ║           │          │
│  ╚═══════════════════════════════════════╪═══════════════════════════════╝           │          │
│                                          │ Atomic Pointer Swap (ArcSwap::store)      │          │
│                                          │                                           ▼          │
│  ╔═══════════════════════════════════════╧═══════════════════════════════════════════════════╗  │
│  ║ 4. BACKGROUND TRAINING PLANE (Dedicated Rayon Thread Pool)                                ║  │
│  ║ • Triggered every N mins or M new queries (or Drift Detected in Q-Error)                  ║  │
│  ║ • Queries cost-stratified training window from DuckDB                                     ║  │
│  ║ • Trains Pure-Rust GBDT trees (mem_p90, mem_p50, scan_bytes, return_rows, cpu_ns, spill)  ║  │
│  ║ • Validates holdout Q-Error before promoting snapshot to ArcSwap + optional S3 sync       ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Thread Pool & Resource Isolation
* **Hot HTTP Runtime (`tokio`):** Dedicated multi-threaded async runtime handling `/predict_csv`, `/v1/predict`, `/health_check`, and `/metrics`. Never performs blocking disk I/O or lock acquisition.
* **Ingestion Runtime:** Bounded `tokio::sync::mpsc` channels receive raw telemetry events, perform fast in-memory updates to Tier 1 (`DashMap`) and Tier 3 (atomic/sharded state), and flush batches to DuckDB via a dedicated single-writer background thread using DuckDB's bulk `Appender` API.
* **Training Thread Pool (`rayon`):** Configured with a bounded thread count (`max(1, num_cpus / 4)` by default) so background GBDT training cannot starve the HTTP runtime of CPU caches or scheduling quanta.

---

## 4. Feature Engineering & Version-Resilient Decoding

### 4.1 Handling StarRocks Version Skew & Dynamic Operator Segments
Because StarRocks generates `operators_0..N` by iterating over `OperatorType.values()` between `PHYSICAL` and `SCALAR`, adding a new physical operator enum in a future StarRocks release shifts column offsets.

Our decoder (`src/features/decoder.rs`) solves this without fragile hardcoded column indices:
1. **Header Validation & Fast Path:** Checks the number of `operators_*` columns. If it matches a known StarRocks version profile (e.g., StarRocks 3.3, 3.4, 3.5), it uses a precomputed index lookup table (`O(1)` array indexing).
2. **Self-Synchronizing Segment Scanner (Fallback):** Because every operator block in `PlanFeatures` begins with `[opType.ordinal(), count, ...]`, and active operators have `count > 0` while inactive operators are `[ordinal, 0, 0, ...]`, the parser can dynamically walk the segment boundaries and align operator blocks into a canonical internal representation (`CanonicalPlanFeatures`).

### 4.2 Correcting the "Summed Log" Mathematical Artifact
In StarRocks' `PlanFeatures.AggregatedFeature.merge()`, when multiple operators of the same type exist in a plan (e.g., $k = 3$ `PHYSICAL_HASH_JOIN` operators), StarRocks sums their `Math.log1p()` values:
$$S = \sum_{i=1}^k \log(1 + x_i)$$
Passing $S$ raw to a model conflates operator count with exponential size ($\exp(S) \approx \prod x_i$).
For every operator block with `count > 0`, the decoder computes:
* **Mean Log Feature:** $\bar{z} = \frac{S}{\text{count}}$ (representing the geometric mean in log-space).
* **Estimated Log of Sum:** $\tilde{z}_{\text{sum}} = \bar{z} + \ln(\text{count})$ (approximating $\log(\sum x_i)$).
* **Synthesized Plan-Level Interaction Features (24 dense features):**
  * `total_physical_operators`, `distinct_operator_types`
  * `scan_count`, `scan_mean_log_rows`, `scan_total_tablets`, `scan_total_partitions`, `scan_total_predicates`
  * `join_count`, `broadcast_join_count`, `partition_join_count`, `join_max_right_log_size`
  * `agg_count`, `agg_mean_log_input_rows`, `agg_total_group_bys`, `agg_mean_ratio`
  * `exchange_count`, `sort_topn_count`, `window_count`, `cte_count`
  * `cbo_total_mem_log_est`, `cbo_total_cpu_log_est`, `cbo_max_op_mem_log`
  * `env_be_nodes`, `env_total_cores` (`numBeNodes * avgCpuCoreOfBE`), `env_be_mem_log`, `var_pipeline_dop`

### 4.3 Target-Encoded Table Telemetry Catalog (`tables_0..2`)
Raw `tables_0..2` values (e.g., `"10492"` or `"hive.sales.fact_orders"`) are high-cardinality categorical identifiers. The service maintains an in-memory `DashMap<TableIdentifier, TableProfile>` updated continuously from ingested executions. For each of the 3 table slots, the feature enricher emits:
1. `table_i_known` (`1.0` if seen $\ge 5$ times, else `0.0`)
2. `table_i_ewma_log_scan_bytes` (historical EWMA of $\log(1 + \text{scanBytes})$ when this table is scanned)
3. `table_i_ewma_bytes_per_row` (historical $\frac{\text{scanBytes}}{\max(1, \text{scanRows})}$)
4. `table_i_mem_multiplier` (historical EWMA of $\ln\left(\frac{1 + \text{actualMemBytes}}{1 + \text{cboMemBytes}}\right)$ for queries involving this table)
5. `table_i_spill_rate` (fraction of historical queries touching this table that spilled to disk)

---

## 5. The Cascading 3-Tier ML Architecture

All regression targets spanning multiple orders of magnitude (`mem_bytes`, `scan_bytes`, `return_rows`, `cpu_cost_ns`) are modeled in **natural log-space**:
$$y = \ln(1 + \text{actual\_value}), \quad \hat{V} = \max(0, \exp(\hat{y}) - 1)$$

For memory prediction (`memCostBytes`), ground-truth labels incorporate spill compensation and OOM censoring:
* **Normal Completion (`state IN ('EOF', 'OK')`):**
  $$\text{effective\_mem\_bytes} = \text{memCostBytes} + \text{spilledBytes}$$
* **OOM / Memory Limit Exceeded (`state = 'ERR'` with memory error signature):**
  $$\text{effective\_mem\_bytes} = \max(\text{memCostBytes} + \text{spilledBytes}, \text{memCostBytes} \times 1.5)$$

### 5.1 Tier 1: Structural Plan-Signature Sketch (`~2µs` Latency, Zero Training Lag)
Captures recurring query templates (BI dashboards, scheduled reports, parameterized app queries):
* **Signature Hash (`u64` via `xxh3`):** Computed over `(sorted(tables_0..2), [(op_ordinal, count, join_is_broadcast, join_type, agg_group_by_cols) for active ops])`. Crucially, this **excludes** row counts and cost numbers so the same SQL template hashes identically even as underlying table row counts grow.
* **Tracked State per Signature (`PlanSketchEntry`):**
  * `hit_count: u64`, `last_seen_epoch_s: u64`
  * `ewma_log_mem: f64`, `ewma_var_log_mem: f64` (using Welford/EWMA with $\alpha = 0.20$)
  * `ewma_log_scan_bytes: f64`, `ewma_log_return_rows: f64`, `ewma_log_cpu_ns: f64`
  * `ewma_cbo_log_rows: f64` (reference input cardinality)
* **Cardinality-Scaled Inference:**
  If the current query's `scan_mean_log_rows` differs from `ewma_cbo_log_rows` by $\Delta r$, Tier 1 scales its log-prediction proportionally:
  $$\hat{y}_{\text{tier1, p50}} = \text{ewma\_log\_mem} + \gamma \cdot \text{clamp}(\Delta r, -2.0, 2.0)$$
  $$\hat{y}_{\{ \text{tier1, p90} \}} = \hat{y}_{\text{tier1, p50}} + 1.2816 \sqrt{\text{ewma\_var\_log\_mem}}$$
* **Eviction:** Bounded capacity (default 100,000 templates) using segmented LRU / TinyLFU (`moka` cache or sharded LRU).

### 5.2 Tier 2: Multi-Target Pure-Rust GBDT Ensemble (`~15µs` Latency)
Generalizes across unseen query plans, ad-hoc SQL, and multi-table interactions:
* **Engine:** Pure-Rust Gradient Boosted Decision Trees (`forust-ml`) compiled natively into the binary (with optional ONNX export/import support).
* **Model Targets (Trained in parallel on the `rayon` pool):**
  1. `gbdt_mem_p90`: Quantile loss ($\alpha = 0.90$) on $\ln(1 + \text{effective\_mem\_bytes})$. Primary model for `/predict_csv` so Query Queue V2 slots safely cover 90th-percentile peak memory.
  2. `gbdt_mem_p50`: Huber / L1 loss ($\alpha = 0.50$) on $\ln(1 + \text{effective\_mem\_bytes})$.
  3. `gbdt_scan_bytes`: Huber loss on $\ln(1 + \text{scanBytes})$.
  4. `gbdt_return_rows`: Huber loss on $\ln(1 + \text{returnRows})$.
  5. `gbdt_cpu_ns`: Huber loss on $\ln(1 + \text{cpuCostNs})$.
  6. `gbdt_spill_prob`: Log-loss binary classifier on `spilledBytes > 0`.
* **Hyperparameters (Tuned for Fast Inference & Tabular Generalization):**
  * `n_estimators = 150`, `max_depth = 6`, `learning_rate = 0.08`, `min_leaf_weight = 5.0`.
  * Evaluating 150 depth-6 trees in flat contiguous Rust arrays takes $\sim 3\mu\text{s}$ per target ($\sim 15\mu\text{s}$ for all targets, or $\sim 3\mu\text{s}$ on `/predict_csv` which only needs `gbdt_mem_p90`).

### 5.3 Tier 3: Online Recursive Least Squares (RLS) CBO Calibrator (`~1µs` Latency)
Provides immediate cold-start protection on a brand-new cluster or when a query touches 100% unseen tables:
* Models the relationship between StarRocks' internal CBO estimates (extracted from the CSV) and actual log-resources using a 6-dimensional linear parameter vector $\mathbf{w} \in \mathbb{R}^6$ and $6 \times 6$ inverse covariance matrix $\mathbf{P}$ with forgetting factor $\lambda = 0.995$:
  $$\mathbf{x}_{\text{rls}} = \begin{bmatrix} 1 & \text{cbo\_total\_mem\_log} & \text{scan\_mean\_log\_rows} & \ln(1 + \text{join\_count}) & \ln(1 + \text{agg\_count}) & \ln(1 + \text{ dop }) \end{bmatrix}^T$$
* Initialized prior at boot ($\mathbf{w}_0 = [0, 1, 0, 0, 0, 0]^T$, i.e., trusting the StarRocks CBO memory estimate 1:1 until execution data arrives) and updated in $O(d^2) = 36$ floating-point operations per ingested query.

### 5.4 Continuous Confidence Blending & Safety Clamping
Rather than brittle step-function cutoffs, the ensemble blends log-space predictions $\hat{y}_1, \hat{y}_2, \hat{y}_3$:
1. **Tier 1 Confidence ($c_1 \in [0, 0.95]$):**
   $$c_1 = \frac{n_{\text{hits}}}{n_{\text{hits}} + 5} \cdot \exp(-0.25 \cdot \text{ewma\_var\_log\_mem}) \cdot \exp(-0.1 \cdot |\Delta r|)$$
2. **Tier 2 Confidence ($c_2 \in [0, 0.90]$):**
   $$c_2 = \mathbb{I}(\text{gbdt\_ready}) \cdot \left(0.4 + 0.5 \cdot \text{known\_table\_ratio}\right) \cdot \exp(-\text{holdout\_med\_qerror\_excess})$$
3. **Normalized Weights:**
   * $w_1 = c_1$
   * $w_2 = (1 - w_1) \cdot c_2$
   * $w_3 = 1 - w_1 - w_2$
   * $\hat{y}_{\text{final}} = w_1 \hat{y}_1 + w_2 \hat{y}_2 + w_3 \hat{y}_3$
4. **Safety Bounds:**
   Final bytes $\hat{B} = \exp(\hat{y}_{\text{final}}) - 1$ are clamped to `[config.min_predicted_mem_bytes (default 1 MiB), env_mem_capacity_bytes * config.max_cluster_mem_multiplier (default 2.0)]`.

---

## 6. Multi-Source Ingestion, Correlation & DuckDB Storage

### 6.1 Supported Ingestion Sources
1. **Source A — `fe.features.log` (File Tailer + `POST /v1/ingest/features`):**
   * Parses pipe-delimited records written by `AuditLogBuilder.java`:
     `digest=...|cpuCostNs=...|memCostBytes=...|scanBytes=...|scanRows=...|returnRows=...|spilledBytes=...|time=...|state=...|catalog=...|database=...|tables=[...]|env=[...]|var=[...]|operators=[...]`
   * Built-in file tailer tracks `(dev, inode, byte_offset)` in DuckDB to survive restarts and log rollovers (`fe.features.log.20261028-1`).
2. **Source B — `fe.audit.log` / `starrocks_audit_tbl__` (Poller + `POST /v1/ingest/audit`):**
   * Supports users who only collect standard audit logs without `fe.features.log`.
   * **In-Flight Prediction Correlation Cache:** Whenever `/predict_csv` or `/v1/predict` is called, the service stores the incoming `CanonicalPlanFeatures` in a bounded TTL ring-buffer (`InFlightCache`, capacity 50,000, TTL 30 minutes) indexed by `(predicted_mem_bytes, db_or_top_table)`. Since StarRocks records the exact `predictMemBytes` returned by `/predict_csv` into `AuditEvent.predictMemBytes`, incoming audit records are matched back to their pre-execution `CanonicalPlanFeatures` vector.
   * Even if an audit record cannot be matched to an in-flight prediction, its table-level scan statistics (`scanBytes`, `scanRows`, `memCostBytes`) still update the `TableProfile` catalog and Tier 3 calibrator.
3. **Source C — Runtime Query Profiles (`POST /v1/ingest/profile` + Optional FE HTTP Pull):**
   * Parses StarRocks Query Profiles (JSON or text tree format) to extract per-operator `PeakMemoryUsage`, `RowsReturned`, `BuildHashTableTime`, and `SpillBytes`.
   * Refines per-table scan selectivity and operator-specific calibration factors.

### 6.2 DuckDB Schema & Cost-Stratified Retention
Embedded DuckDB database (`data/cost_predictor.duckdb`) persists across restarts:
* `telemetry_samples`: Stores extracted dense feature vectors, raw metadata (`digest`, `tables`, `state`, `timestamp`), and actual target labels (`effective_mem_bytes`, `scan_bytes`, `return_rows`, `cpu_cost_ns`, `spilled_bytes`, `query_time_ms`).
* `table_profiles`: Persists aggregated table statistics for fast startup hydration.
* `model_snapshots`: Stores versioned, zstd-compressed `bincode`/`msgpack` blobs of trained GBDT ensembles, Tier 1 sketches, Tier 3 RLS matrices, and holdout evaluation metrics (`q_error_p50`, `q_error_p90`, `pinball_loss_p90`).
* **Cost-Stratified Compaction:**
  A periodic maintenance task bounds `telemetry_samples` to `max_stored_samples` (default `250,000` rows, `~50–80 MB` on disk). Rather than pure FIFO eviction—which allows high-frequency 1MB point-lookups to evict rare 50GB analytical joins—eviction partitions rows into 5 memory cost strata (`<10MB`, `10–100MB`, `100MB–1GB`, `1–10GB`, `>10GB`) and retains up to `50,000` recent samples per stratum.

### 6.3 Continuous Training & Champion/Challenger Promotion
The background trainer wakes up on a configurable tick (default every `30s`) and triggers a GBDT training run on the `rayon` pool when either:
1. **Volume + Time Trigger:** At least `min_new_samples_for_retrain` (default `200`) have arrived AND `retrain_interval_secs` (default `300s`) has elapsed since the last training run (or immediately once the first `100` samples arrive during initial bootstrap).
2. **Concept Drift Trigger:** An online Page-Hinkley / EWMA monitor tracking live prediction Q-Error ($\max(\hat{y}/y, y/\hat{y})$) detects that recent 100-query median Q-Error exceeds historical baseline by $> 35\%$.

**Champion/Challenger Validation Gate:**
* Splits training data into an 85% stratified training set and a 15% recent/stratified validation holdout.
* Computes validation **P90 Pinball Loss** and **Median / P90 Q-Error**.
* Promotes the Challenger model to `ArcSwap<ModelSnapshot>` only if validation loss is finite, predictions on standard canary vectors are within valid bounds, and holdout loss does not regress by more than a configurable tolerance (default `5%`) relative to the active Champion evaluated on the same holdout window.
* Optionally uploads the promoted snapshot to `s3://<bucket>/<prefix>/latest.snapshot` so horizontally scaled read replicas can poll and hot-swap models without duplicating training work.

---

## 7. API Specification

### 7.1 StarRocks Native Endpoints
| Method & Path | Content-Type | Description | Latency Target |
|---|---|---|---|
| `POST /predict_csv` | `text/csv` | Accepts 2-line StarRocks CSV (`header\nrow`). Returns plain-text ASCII integer string of predicted memory bytes (default `p90` quantile). | `p50 < 25µs`, `p99 < 250µs` |
| `GET /health_check` | `text/plain` | Returns `200 OK` when healthy and circuit breaker is closed. Returns `503 Service Unavailable` if circuit breaker is tripped (instructing StarRocks FE to fall back to native CBO). | `< 20µs` |

### 7.2 Extended Prediction, Ingestion & Operational Endpoints
| Method & Path | Description |
|---|---|
| `POST /v1/predict` | Accepts either StarRocks CSV (`text/csv`) or structured JSON (`application/json`). Returns multi-target prediction JSON: `predicted_mem_bytes_p90`, `predicted_mem_bytes_p50`, `predicted_scan_bytes`, `predicted_return_rows`, `predicted_cpu_cost_ns`, `spill_probability`, `confidence`, `tier_weights`, `dominant_tier`, `model_version`, `latency_us`. |
| `POST /v1/ingest/features` | Ingests batch or single lines of `fe.features.log` (raw text or NDJSON). Returns `202 Accepted`. |
| `POST /v1/ingest/audit` | Ingests StarRocks audit log events (JSON/CSV). Correlates with in-flight predictions and updates models. |
| `POST /v1/ingest/profile` | Ingests StarRocks Runtime Query Profile payloads to refine operator and table statistics. |
| `GET /v1/models/status` | Returns active model version, training sample counts per stratum, holdout Q-Error metrics, Tier 1 cache hit rates, and table catalog size. |
| `POST /v1/admin/retrain` | Forces an asynchronous background GBDT retrain cycle. |
| `POST /v1/admin/circuit-breaker` | Sets `{ "tripped": true | false }` to gracefully toggle `/health_check` status for StarRocks FE fallback. |
| `GET /metrics` | Prometheus metrics exposition (`sr_predictor_inference_duration_us`, `sr_predictor_tier_hits_total`, `sr_predictor_live_q_error`, `sr_predictor_ingest_rows_total`, `sr_predictor_model_version`). |

---

## 8. Error Handling & Resilience Guardrails

1. **Zero-HTTP-500 Policy on `/predict_csv`:**
   Because `CostPredictor.java` logs a full stack trace warning on every non-200 HTTP response, if a malformed CSV arrives or internal timeout (`2ms`) triggers, `/predict_csv` logs a rate-limited internal metric and returns a safe fallback estimate (Tier 3 CBO estimate if parseable, or `config.fallback_mem_bytes` / `0`) with HTTP `200 OK`.
2. **Automatic Drift Circuit-Breaker (Optional):**
   If live predictions exhibit severe persistent error and the operator enables `auto_circuit_breaker_on_severe_drift = true`, `/health_check` automatically returns `503` so StarRocks reverts to native CBO until the next valid model retrains.
3. **Graceful Shutdown & Fast Hydration:**
   On `SIGTERM`/`SIGINT`, flushes pending ingestion buffer to DuckDB and writes a final `ModelSnapshot` checkpoint. On startup, loads the latest `ModelSnapshot` into `ArcSwap` before binding the TCP listener (`< 50ms`).

---

## 9. Testing & Verification Strategy

1. **StarRocks Wire-Protocol & Feature Equivalence Tests:**
   * Unit tests using exact CSV payloads and `fe.features.log` lines generated by StarRocks' `PlanFeatures.java` and `AuditLogBuilder.java` (including StarRocks `CostPredictorTest.java` cases).
   * Validates summed-log normalization, scan operator collapsing, and version-skewed `OperatorType` column counts.
2. **3-Tier ML Convergence & Q-Error Simulation Test:**
   * Synthetic workload generator simulating repeated dashboard queries (Tier 1), multi-table joins on known tables with non-linear broadcast join memory spikes (Tier 2 GBDT), and unseen cold-start queries (Tier 3 RLS).
   * Verifies that after ingesting 250 synthetic queries, background retraining automatically promotes a Tier 2 GBDT snapshot and reduces P90 memory & scan-bytes Q-Error by $> 3\times$ compared to uncalibrated CBO estimates.
3. **Concurrency & Latency Benchmarks (`criterion` + load test):**
   * Verifies that `/predict_csv` maintains `p99 < 250µs` while a simultaneous background GBDT training job is executing and swapping `ArcSwap` snapshots under 1,000+ req/sec concurrent load.
