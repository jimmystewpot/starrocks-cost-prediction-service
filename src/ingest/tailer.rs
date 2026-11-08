use crate::ingest::parser::parse_features_log_line;
use crate::storage::sqlite::SqliteStore;
use crate::types::TelemetrySample;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub const MAX_LINES_PER_POLL: usize = 5_000;
pub const MAX_LINE_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub struct TailBatch {
    pub samples: Vec<TelemetrySample>,
    pub inode: u64,
    pub new_offset: u64,
}

pub struct FileTailer;

impl FileTailer {
    pub fn poll_once(
        log_path: &Path,
        store: &SqliteStore,
    ) -> Result<Vec<TelemetrySample>, std::io::Error> {
        Self::poll_once_with_limit(log_path, store, MAX_LINES_PER_POLL)
    }

    pub fn poll_once_with_limit(
        log_path: &Path,
        store: &SqliteStore,
        max_lines: usize,
    ) -> Result<Vec<TelemetrySample>, std::io::Error> {
        let Some(batch) = Self::read_batch_with_limit(log_path, store, max_lines)? else {
            return Ok(Vec::new());
        };
        Self::commit_batch(log_path, store, &batch)?;
        Ok(batch.samples)
    }

    pub fn read_batch_with_limit(
        log_path: &Path,
        store: &SqliteStore,
        max_lines: usize,
    ) -> Result<Option<TailBatch>, std::io::Error> {
        Self::read_batch_with_limits(log_path, store, max_lines, MAX_LINE_BYTES)
    }

    pub fn read_batch_with_limits(
        log_path: &Path,
        store: &SqliteStore,
        max_lines: usize,
        max_line_bytes: usize,
    ) -> Result<Option<TailBatch>, std::io::Error> {
        let mut file = File::open(log_path)?;
        let meta = file.metadata()?;
        let current_inode = file_inode(&meta);
        let file_len = meta.len();

        let path_key = log_path.to_string_lossy();
        let saved = store
            .get_file_checkpoint(&path_key)
            .map_err(std::io::Error::other)?;

        let mut offset = match saved {
            Some((saved_inode, saved_offset))
                if saved_inode == current_inode && file_len >= saved_offset =>
            {
                saved_offset
            }
            _ => 0,
        };

        if file_len == offset {
            if saved != Some((current_inode, offset)) {
                store
                    .set_file_checkpoint(&path_key, current_inode, offset)
                    .map_err(std::io::Error::other)?;
            }
            return Ok(None);
        }

        file.seek(SeekFrom::Start(offset))?;
        let mut reader = BufReader::new(file);
        let mut line_buf = Vec::new();
        let mut samples = Vec::new();
        let line_limit = max_lines.max(1);
        let byte_limit = max_line_bytes.max(64);
        let mut lines_processed = 0usize;

        while lines_processed < line_limit {
            line_buf.clear();
            let bytes_read = (&mut reader)
                .take((byte_limit + 1) as u64)
                .read_until(b'\n', &mut line_buf)?;
            if bytes_read == 0 {
                break;
            }
            if bytes_read > byte_limit && !line_buf.ends_with(b"\n") {
                let mut discard = Vec::new();
                let mut skipped = bytes_read as u64;
                let mut hit_newline = false;
                loop {
                    discard.clear();
                    let n = (&mut reader).take(65_536).read_until(b'\n', &mut discard)?;
                    if n == 0 {
                        break;
                    }
                    skipped += n as u64;
                    if discard.ends_with(b"\n") {
                        hit_newline = true;
                        break;
                    }
                }
                if !hit_newline {
                    break;
                }
                offset += skipped;
                lines_processed += 1;
                tracing::warn!(
                    path = %path_key,
                    skipped_bytes = skipped,
                    "skipping oversized features log line"
                );
                continue;
            }
            if !line_buf.ends_with(b"\n") {
                // Partial trailing line; do not advance checkpoint until newline arrives.
                break;
            }

            offset += bytes_read as u64;
            lines_processed += 1;
            let line_str = String::from_utf8_lossy(&line_buf);
            match parse_features_log_line(&line_str) {
                Ok(Some(sample)) => samples.push(sample),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(
                        path = %path_key,
                        error = %err,
                        "skipping malformed features log line"
                    );
                }
            }
        }

        Ok(Some(TailBatch {
            samples,
            inode: current_inode,
            new_offset: offset,
        }))
    }

    pub fn commit_batch(
        log_path: &Path,
        store: &SqliteStore,
        batch: &TailBatch,
    ) -> Result<(), std::io::Error> {
        let path_key = log_path.to_string_lossy();
        store
            .set_file_checkpoint(&path_key, batch.inode, batch.new_offset)
            .map_err(std::io::Error::other)
    }
}

#[cfg(unix)]
fn file_inode(meta: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    meta.ino()
}

#[cfg(not(unix))]
fn file_inode(_meta: &std::fs::Metadata) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
        assert_eq!(batch1[0].digest, "d1");

        let batch2 = FileTailer::poll_once(&log_path, &store).unwrap();
        assert_eq!(batch2.len(), 0);

        // Append a partial line without trailing newline: checkpoint must NOT advance past it.
        let partial_part1 = "digest=d2|cpuCostNs=200|memCostBytes=20000000|scanBytes=2000|scanRows=20|returnRows=2|spilledBytes=0|time=20|state=EOF";
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&log_path)
                .unwrap();
            f.write_all(partial_part1.as_bytes()).unwrap();
        }
        let batch_partial = FileTailer::poll_once(&log_path, &store).unwrap();
        assert_eq!(batch_partial.len(), 0);

        // Complete the partial line with a newline and verify it is ingested.
        let partial_part2 = "|catalog=c|database=d|tables=[0,0,102]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]\n";
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&log_path)
                .unwrap();
            f.write_all(partial_part2.as_bytes()).unwrap();
        }
        let batch_completed = FileTailer::poll_once(&log_path, &store).unwrap();
        assert_eq!(batch_completed.len(), 1);
        assert_eq!(batch_completed[0].digest, "d2");

        // Simulate log truncation (shorter file at same path) -> resets offset to 0.
        let truncated_line = "digest=d3|cpuCostNs=300|memCostBytes=30000000|scanBytes=3000|scanRows=30|returnRows=3|spilledBytes=0|time=30|state=EOF|catalog=c|database=d|tables=[0,0,103]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]\n";
        std::fs::write(&log_path, truncated_line).unwrap();
        let batch_truncated = FileTailer::poll_once(&log_path, &store).unwrap();
        assert_eq!(batch_truncated.len(), 1);
        assert_eq!(batch_truncated[0].digest, "d3");
    }

    #[test]
    fn test_file_tailer_respects_max_lines_per_poll() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_limit.db");
        let log_path = dir.path().join("fe.features.log");
        let store = SqliteStore::open(&db_path, 100).unwrap();

        let mut content = String::new();
        for i in 1..=5 {
            content.push_str(&format!(
                "digest=d{i}|cpuCostNs=100|memCostBytes=10000000|scanBytes=1000|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|catalog=c|database=d|tables=[0,0,10{i}]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]\n"
            ));
        }
        std::fs::write(&log_path, content).unwrap();

        let poll1 = FileTailer::poll_once_with_limit(&log_path, &store, 2).unwrap();
        assert_eq!(poll1.len(), 2);
        assert_eq!(poll1[0].digest, "d1");
        assert_eq!(poll1[1].digest, "d2");

        let poll2 = FileTailer::poll_once_with_limit(&log_path, &store, 2).unwrap();
        assert_eq!(poll2.len(), 2);
        assert_eq!(poll2[0].digest, "d3");
        assert_eq!(poll2[1].digest, "d4");

        let poll3 = FileTailer::poll_once_with_limit(&log_path, &store, 2).unwrap();
        assert_eq!(poll3.len(), 1);
        assert_eq!(poll3[0].digest, "d5");

        let poll4 = FileTailer::poll_once_with_limit(&log_path, &store, 2).unwrap();
        assert!(poll4.is_empty());
    }

    #[test]
    fn test_file_tailer_skips_oversized_corrupt_line_and_commits_checkpoint_only_on_commit() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_oversized.db");
        let log_path = dir.path().join("fe.features.log");
        let store = SqliteStore::open(&db_path, 100).unwrap();

        let line1 = "digest=d1|cpuCostNs=100|memCostBytes=10000000|scanBytes=1000|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|catalog=c|database=d|tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]\n";
        let oversized = format!("digest=corrupt|{}\n", "X".repeat(2000));
        let line2 = "digest=d2|cpuCostNs=100|memCostBytes=10000000|scanBytes=1000|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=EOF|catalog=c|database=d|tables=[0,0,102]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]\n";

        std::fs::write(&log_path, format!("{line1}{oversized}{line2}")).unwrap();

        let batch = FileTailer::read_batch_with_limits(&log_path, &store, 10, 512)
            .unwrap()
            .expect("should read batch");
        assert_eq!(batch.samples.len(), 2);
        assert_eq!(batch.samples[0].digest, "d1");
        assert_eq!(batch.samples[1].digest, "d2");

        // Checkpoint must NOT be saved before commit_batch is called
        assert!(
            store
                .get_file_checkpoint(&log_path.to_string_lossy())
                .unwrap()
                .is_none()
        );

        FileTailer::commit_batch(&log_path, &store, &batch).unwrap();
        assert_eq!(
            store
                .get_file_checkpoint(&log_path.to_string_lossy())
                .unwrap()
                .unwrap()
                .1,
            batch.new_offset
        );

        // Verify empty file checkpoint initialization, non-finalized line (Ok(None)), malformed line (Err), and oversized line without newline
        let empty_log = dir.path().join("empty.features.log");
        std::fs::write(&empty_log, "").unwrap();
        assert!(
            FileTailer::read_batch_with_limits(&empty_log, &store, 10, 512)
                .unwrap()
                .is_none()
        );

        let running_line = "digest=d_run|cpuCostNs=100|memCostBytes=10000000|scanBytes=1000|scanRows=10|returnRows=1|spilledBytes=0|time=10|state=RUNNING|catalog=c|database=d|tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]\n";
        let malformed_line = "digest=d_bad|state=EOF|memCostBytes=1000|tables=[1]|env=[1]\n";
        let oversized_no_newline = format!("digest=partial|{}", "Y".repeat(1200));
        std::fs::write(
            &empty_log,
            format!("{running_line}{malformed_line}{oversized_no_newline}"),
        )
        .unwrap();
        let batch2 = FileTailer::read_batch_with_limits(&empty_log, &store, 10, 512)
            .unwrap()
            .expect("should read batch with skipped lines");
        assert!(batch2.samples.is_empty());
    }
}
