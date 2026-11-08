use crate::types::{
    DENSE_FEATURE_DIM, DenseFeatureVector, GbdtTrainingRow, TelemetrySample, memory_stratum,
};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const MAX_RETAINED_SNAPSHOTS: usize = 5;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(String),
}

pub struct SqliteStore {
    conn: Mutex<Connection>,
    max_per_stratum: usize,
    stratum_counts: [AtomicUsize; 5],
}

impl SqliteStore {
    pub fn open(path: &Path, max_per_stratum: usize) -> Result<Self, StorageError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA busy_timeout = 5000;

             CREATE TABLE IF NOT EXISTS telemetry_samples (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 plan_hash INTEGER NOT NULL,
                 mem_stratum INTEGER NOT NULL,
                 features_blob BLOB NOT NULL,
                 log_mem REAL NOT NULL,
                 log_scan_bytes REAL NOT NULL,
                 log_return_rows REAL NOT NULL,
                 log_cpu_ns REAL NOT NULL,
                 spilled REAL NOT NULL,
                 created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_telemetry_stratum ON telemetry_samples(mem_stratum, id);

             CREATE TABLE IF NOT EXISTS model_snapshots (
                 version INTEGER PRIMARY KEY,
                 metrics_json TEXT NOT NULL,
                 snapshot_blob BLOB NOT NULL,
                 created_at INTEGER NOT NULL DEFAULT (unixepoch())
             );

             CREATE TABLE IF NOT EXISTS file_checkpoints (
                 file_path TEXT PRIMARY KEY,
                 inode INTEGER NOT NULL,
                 byte_offset INTEGER NOT NULL
             );",
        )?;

        let stratum_counts: [AtomicUsize; 5] = std::array::from_fn(|_| AtomicUsize::new(0));
        {
            let mut stmt = conn.prepare(
                "SELECT mem_stratum, COUNT(*) FROM telemetry_samples GROUP BY mem_stratum",
            )?;
            let rows = stmt.query_map([], |row| {
                let s: i64 = row.get(0)?;
                let c: i64 = row.get(1)?;
                Ok((s, c))
            })?;
            for r in rows {
                let (s, c) = r?;
                if (0..5).contains(&s) {
                    stratum_counts[s as usize].store(c.max(0) as usize, Ordering::Relaxed);
                }
            }
        }

        Ok(Self {
            conn: Mutex::new(conn),
            max_per_stratum: max_per_stratum.max(1),
            stratum_counts,
        })
    }

    fn lock_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn insert_sample(
        &self,
        plan_hash: u64,
        dense: &DenseFeatureVector,
        sample: &TelemetrySample,
    ) -> Result<(), StorageError> {
        self.insert_samples_batch(&[(plan_hash, *dense, sample)])
    }

    pub fn insert_samples_batch(
        &self,
        batch: &[(u64, DenseFeatureVector, &TelemetrySample)],
    ) -> Result<(), StorageError> {
        if batch.is_empty() {
            return Ok(());
        }

        let conn = self.lock_conn();
        let tx = conn.unchecked_transaction()?;

        let mut pending_counts: [usize; 5] =
            std::array::from_fn(|i| self.stratum_counts[i].load(Ordering::Relaxed));
        let mut touched_strata = [false; 5];

        {
            let mut insert_stmt = tx.prepare_cached(
                "INSERT INTO telemetry_samples (
                    plan_hash, mem_stratum, features_blob,
                    log_mem, log_scan_bytes, log_return_rows, log_cpu_ns, spilled, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;

            let mut features_blob = [0u8; DENSE_FEATURE_DIM * 8];
            for &(plan_hash, ref dense, sample) in batch {
                let stratum_idx = (memory_stratum(sample.effective_mem_bytes) as usize).min(4);
                let stratum = stratum_idx as i64;
                for (i, &val) in dense.iter().enumerate() {
                    features_blob[i * 8..(i + 1) * 8].copy_from_slice(&val.to_le_bytes());
                }

                let log_mem = (1.0 + sample.effective_mem_bytes as f64).ln();
                let log_scan_bytes = (1.0 + sample.scan_bytes as f64).ln();
                let log_return_rows = (1.0 + sample.return_rows as f64).ln();
                let log_cpu_ns = (1.0 + sample.cpu_cost_ns as f64).ln();
                let spilled = if sample.spilled_bytes > 0 { 1.0 } else { 0.0 };

                insert_stmt.execute(params![
                    plan_hash as i64,
                    stratum,
                    &features_blob[..],
                    log_mem,
                    log_scan_bytes,
                    log_return_rows,
                    log_cpu_ns,
                    spilled,
                    sample.timestamp_s as i64,
                ])?;

                pending_counts[stratum_idx] += 1;
                touched_strata[stratum_idx] = true;
            }
        }

        for stratum_idx in 0..5 {
            if touched_strata[stratum_idx] && pending_counts[stratum_idx] > self.max_per_stratum {
                let excess = (pending_counts[stratum_idx] - self.max_per_stratum) as i64;
                let mut del_stmt = tx.prepare_cached(
                    "DELETE FROM telemetry_samples WHERE id IN (
                        SELECT id FROM telemetry_samples WHERE mem_stratum = ?1 ORDER BY id ASC LIMIT ?2
                     )",
                )?;
                let deleted = del_stmt.execute(params![stratum_idx as i64, excess])?;
                pending_counts[stratum_idx] = pending_counts[stratum_idx].saturating_sub(deleted);
            }
        }

        tx.commit()?;
        for (stratum_idx, &count) in pending_counts.iter().enumerate() {
            if touched_strata[stratum_idx] {
                self.stratum_counts[stratum_idx].store(count, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    pub fn load_training_rows(&self) -> Result<Vec<GbdtTrainingRow>, StorageError> {
        let conn = self.lock_conn();
        let mut stmt = conn.prepare(
            "SELECT features_blob, log_mem, log_scan_bytes, log_return_rows, log_cpu_ns, spilled
             FROM telemetry_samples ORDER BY id ASC",
        )?;

        let rows_iter = stmt.query_map([], |row| {
            let blob: Vec<u8> = row.get(0)?;
            let log_mem: f64 = row.get(1)?;
            let log_scan_bytes: f64 = row.get(2)?;
            let log_return_rows: f64 = row.get(3)?;
            let log_cpu_ns: f64 = row.get(4)?;
            let spilled: f64 = row.get(5)?;
            Ok((
                blob,
                log_mem,
                log_scan_bytes,
                log_return_rows,
                log_cpu_ns,
                spilled,
            ))
        })?;

        let mut out = Vec::new();
        for item in rows_iter {
            let (blob, log_mem, log_scan_bytes, log_return_rows, log_cpu_ns, spilled) = item?;
            let features = decode_dense_features(&blob)?;
            out.push(GbdtTrainingRow {
                features,
                log_mem,
                log_scan_bytes,
                log_return_rows,
                log_cpu_ns,
                spilled,
            });
        }

        Ok(out)
    }

    pub fn sample_counts_by_stratum(&self) -> Result<[usize; 5], StorageError> {
        Ok(std::array::from_fn(|i| {
            self.stratum_counts[i].load(Ordering::Relaxed)
        }))
    }

    pub fn save_snapshot_blob(
        &self,
        version: u64,
        metrics_json: &str,
        blob: &[u8],
    ) -> Result<(), StorageError> {
        let conn = self.lock_conn();
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO model_snapshots (version, metrics_json, snapshot_blob)
             VALUES (?1, ?2, ?3)",
            params![version as i64, metrics_json, blob],
        )?;
        tx.execute(
            "DELETE FROM model_snapshots WHERE version NOT IN (
                SELECT version FROM model_snapshots ORDER BY version DESC LIMIT ?1
             )",
            params![MAX_RETAINED_SNAPSHOTS as i64],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn load_latest_snapshot_blob(
        &self,
    ) -> Result<Option<(u64, String, Vec<u8>)>, StorageError> {
        let conn = self.lock_conn();
        let res = conn
            .query_row(
                "SELECT version, metrics_json, snapshot_blob
                 FROM model_snapshots
                 ORDER BY version DESC
                 LIMIT 1",
                [],
                |row| {
                    let v: i64 = row.get(0)?;
                    let m: String = row.get(1)?;
                    let b: Vec<u8> = row.get(2)?;
                    Ok((v.max(0) as u64, m, b))
                },
            )
            .optional()?;
        Ok(res)
    }

    pub fn snapshot_count(&self) -> Result<usize, StorageError> {
        let conn = self.lock_conn();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM model_snapshots", [], |row| row.get(0))?;
        Ok(count.max(0) as usize)
    }

    pub fn get_file_checkpoint(&self, path: &str) -> Result<Option<(u64, u64)>, StorageError> {
        let conn = self.lock_conn();
        let res = conn
            .query_row(
                "SELECT inode, byte_offset FROM file_checkpoints WHERE file_path = ?1",
                params![path],
                |row| {
                    let inode: i64 = row.get(0)?;
                    let offset: i64 = row.get(1)?;
                    Ok((inode as u64, offset as u64))
                },
            )
            .optional()?;
        Ok(res)
    }

    pub fn set_file_checkpoint(
        &self,
        path: &str,
        inode: u64,
        offset: u64,
    ) -> Result<(), StorageError> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT OR REPLACE INTO file_checkpoints (file_path, inode, byte_offset)
             VALUES (?1, ?2, ?3)",
            params![path, inode as i64, offset as i64],
        )?;
        Ok(())
    }
}

fn decode_dense_features(blob: &[u8]) -> Result<DenseFeatureVector, StorageError> {
    if blob.len() == DENSE_FEATURE_DIM * 8 {
        let mut features = [0.0_f64; DENSE_FEATURE_DIM];
        for (i, chunk) in blob.chunks_exact(8).enumerate() {
            let bytes: [u8; 8] = chunk.try_into().unwrap();
            features[i] = f64::from_le_bytes(bytes);
        }
        return Ok(features);
    }

    let vec: Vec<f64> =
        rmp_serde::from_slice(blob).map_err(|e| StorageError::Serialization(e.to_string()))?;
    if vec.len() != DENSE_FEATURE_DIM {
        return Err(StorageError::Serialization(format!(
            "expected {} features, got {}",
            DENSE_FEATURE_DIM,
            vec.len()
        )));
    }
    let mut features = [0.0_f64; DENSE_FEATURE_DIM];
    features.copy_from_slice(&vec);
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::parser::parse_features_log_line;

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
        assert_eq!(
            counts[3], 3,
            "stratum 3 (1-10GB) preserved all 3 heavy queries!"
        );
    }

    #[test]
    fn test_sqlite_training_rows_snapshots_and_file_checkpoints_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = SqliteStore::open(tmp.path(), 10).unwrap();
        let line = "digest=d1|cpuCostNs=500000|memCostBytes=40000000|scanBytes=10000000|scanRows=50000|returnRows=100|spilledBytes=10000000|time=120|state=EOF|catalog=c|database=d|tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]";
        let sample = parse_features_log_line(line).unwrap().unwrap();
        let mut dense = [0.0_f64; 39];
        dense[0] = 3.5;
        dense[38] = -1.25;

        store.insert_sample(999, &dense, &sample).unwrap();
        let rows = store.load_training_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].features[0], 3.5);
        assert_eq!(rows[0].features[38], -1.25);
        assert!((rows[0].spilled - 1.0).abs() < 1e-12);

        assert!(store.load_latest_snapshot_blob().unwrap().is_none());
        store
            .save_snapshot_blob(1, r#"{"pinball":0.2}"#, &[1, 2, 3])
            .unwrap();
        store
            .save_snapshot_blob(2, r#"{"pinball":0.1}"#, &[4, 5, 6])
            .unwrap();
        let (ver, metrics, blob) = store.load_latest_snapshot_blob().unwrap().unwrap();
        assert_eq!(ver, 2);
        assert_eq!(metrics, r#"{"pinball":0.1}"#);
        assert_eq!(blob, vec![4, 5, 6]);

        assert!(
            store
                .get_file_checkpoint("/var/log/fe.features.log")
                .unwrap()
                .is_none()
        );
        store
            .set_file_checkpoint("/var/log/fe.features.log", 12345, 67890)
            .unwrap();
        assert_eq!(
            store
                .get_file_checkpoint("/var/log/fe.features.log")
                .unwrap(),
            Some((12345, 67890))
        );
    }

    #[test]
    fn test_sqlite_snapshot_pruning_retains_latest_five() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = SqliteStore::open(tmp.path(), 10).unwrap();
        for v in 1..=8 {
            store
                .save_snapshot_blob(v, &format!(r#"{{"version":{v}}}"#), &[v as u8; 16])
                .unwrap();
        }
        let (latest_ver, metrics, blob) = store.load_latest_snapshot_blob().unwrap().unwrap();
        assert_eq!(latest_ver, 8);
        assert_eq!(metrics, r#"{"version":8}"#);
        assert_eq!(blob, vec![8u8; 16]);
        assert_eq!(
            store.snapshot_count().unwrap(),
            5,
            "model_snapshots must prune older snapshots and retain only the latest 5"
        );

        // Also verify sample_counts_by_stratum hydration across reopen
        let line = "digest=d1|cpuCostNs=500000|memCostBytes=40000000|scanBytes=10000000|scanRows=50000|returnRows=100|spilledBytes=10000000|time=120|state=EOF|catalog=c|database=d|tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]";
        let sample = parse_features_log_line(line).unwrap().unwrap();
        let dense = [0.0_f64; 39];
        store.insert_sample(1, &dense, &sample).unwrap();
        store.insert_sample(2, &dense, &sample).unwrap();
        let counts_before = store.sample_counts_by_stratum().unwrap();
        drop(store);

        let reopened = SqliteStore::open(tmp.path(), 10).unwrap();
        assert_eq!(reopened.sample_counts_by_stratum().unwrap(), counts_before);
        assert_eq!(reopened.snapshot_count().unwrap(), 5);
    }

    #[test]
    fn test_insert_samples_batch_commits_and_compacts_in_single_transaction() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = SqliteStore::open(tmp.path(), 5).unwrap();
        let line_stratum3 = "digest=d1|cpuCostNs=100|memCostBytes=2000000000|scanBytes=1000|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|catalog=c|database=d|tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]";
        let sample_stratum3 = parse_features_log_line(line_stratum3).unwrap().unwrap();
        let mut sample_stratum0 = sample_stratum3.clone();
        sample_stratum0.effective_mem_bytes = 2_000_000;

        let mut samples = Vec::with_capacity(25);
        for _ in 0..3 {
            samples.push(sample_stratum3.clone());
        }
        for _ in 0..22 {
            samples.push(sample_stratum0.clone());
        }

        let dense = [0.0_f64; 39];
        let batch: Vec<(u64, DenseFeatureVector, &TelemetrySample)> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| (1000 + i as u64, dense, s))
            .collect();

        store.insert_samples_batch(&batch).unwrap();

        let counts = store.sample_counts_by_stratum().unwrap();
        assert_eq!(counts[0], 5, "stratum 0 compacted to 5 in batch");
        assert_eq!(counts[3], 3, "stratum 3 retained all 3 in batch");
        assert_eq!(counts, [5, 0, 0, 3, 0]);

        let rows = store.load_training_rows().unwrap();
        assert_eq!(rows.len(), 8);
    }

    #[test]
    fn test_sqlite_batch_empty_and_corrupt_blob_error() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = SqliteStore::open(tmp.path(), 5).unwrap();

        // 1. insert_samples_batch with empty slice
        store.insert_samples_batch(&[]).unwrap();
        assert_eq!(store.load_training_rows().unwrap().len(), 0);

        // 2. Corrupt features_blob error handling in load_training_rows
        {
            let conn = store.lock_conn();
            conn.execute(
                "INSERT INTO telemetry_samples (
                    plan_hash, mem_stratum, features_blob,
                    log_mem, log_scan_bytes, log_return_rows, log_cpu_ns, spilled, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    1_i64,
                    0_i64,
                    b"bad_msgpack" as &[u8],
                    10.0_f64,
                    10.0_f64,
                    10.0_f64,
                    10.0_f64,
                    0.0_f64,
                    1000_i64,
                ],
            )
            .unwrap();
        }

        let res = store.load_training_rows();
        assert!(matches!(res, Err(StorageError::Serialization(_))));

        // 3. Test legacy MessagePack Vec<f64> decoding (both valid 39-len and invalid len)
        let valid_msgpack = rmp_serde::to_vec(&vec![2.5_f64; DENSE_FEATURE_DIM]).unwrap();
        let decoded = decode_dense_features(&valid_msgpack).unwrap();
        assert_eq!(decoded[0], 2.5);
        assert_eq!(decoded[DENSE_FEATURE_DIM - 1], 2.5);

        let wrong_len_msgpack = rmp_serde::to_vec(&vec![1.0_f64; 5]).unwrap();
        assert!(matches!(
            decode_dense_features(&wrong_len_msgpack),
            Err(StorageError::Serialization(_))
        ));
    }
}
