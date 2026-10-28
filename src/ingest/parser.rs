use crate::features::decoder::{DecodeError, parse_feature_string};
use crate::types::{TelemetrySample, compute_effective_mem_bytes};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("decode error: {0}")]
    Decode(#[from] DecodeError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid format: {0}")]
    InvalidFormat(String),
}

fn default_eof_state() -> String {
    "EOF".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditLogEvent {
    pub digest: String,
    #[serde(default, alias = "predictMemBytes")]
    pub predict_mem_bytes: u64,
    #[serde(default, alias = "memCostBytes")]
    pub mem_cost_bytes: u64,
    #[serde(default, alias = "scanBytes")]
    pub scan_bytes: u64,
    #[serde(default, alias = "scanRows")]
    pub scan_rows: u64,
    #[serde(default, alias = "returnRows")]
    pub return_rows: u64,
    #[serde(default, alias = "cpuCostNs")]
    pub cpu_cost_ns: u64,
    #[serde(default, alias = "spilledBytes")]
    pub spilled_bytes: u64,
    #[serde(default, alias = "queryTime", alias = "time")]
    pub query_time_ms: u64,
    #[serde(default = "default_eof_state")]
    pub state: String,
    #[serde(default)]
    pub tables: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProfileSummary {
    pub tables: Vec<String>,
    pub peak_mem_bytes: u64,
    pub scan_bytes: u64,
    pub scan_rows: u64,
    pub spilled_bytes: u64,
}

pub fn parse_features_log_line(line: &str) -> Result<Option<TelemetrySample>, ParseError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let digest_pos = trimmed
        .find("digest=")
        .ok_or_else(|| ParseError::InvalidFormat("missing 'digest=' prefix".to_string()))?;
    let payload = &trimmed[digest_pos..];

    let mut digest = String::new();
    let mut cpu_cost_ns: u64 = 0;
    let mut mem_cost_bytes: u64 = 0;
    let mut scan_bytes: u64 = 0;
    let mut scan_rows: u64 = 0;
    let mut return_rows: u64 = 0;
    let mut spilled_bytes: u64 = 0;
    let mut query_time_ms: u64 = 0;
    let mut state = String::from("EOF");
    let mut feature_parts = Vec::with_capacity(4);

    for part in payload.split('|') {
        let part = part.trim();
        if let Some(v) = part.strip_prefix("digest=") {
            digest = v.to_string();
        } else if let Some(v) = part.strip_prefix("cpuCostNs=") {
            cpu_cost_ns = parse_u64_field("cpuCostNs", v)?;
        } else if let Some(v) = part.strip_prefix("memCostBytes=") {
            mem_cost_bytes = parse_u64_field("memCostBytes", v)?;
        } else if let Some(v) = part.strip_prefix("scanBytes=") {
            scan_bytes = parse_u64_field("scanBytes", v)?;
        } else if let Some(v) = part.strip_prefix("scanRows=") {
            scan_rows = parse_u64_field("scanRows", v)?;
        } else if let Some(v) = part.strip_prefix("returnRows=") {
            return_rows = parse_u64_field("returnRows", v)?;
        } else if let Some(v) = part.strip_prefix("spilledBytes=") {
            spilled_bytes = parse_u64_field("spilledBytes", v)?;
        } else if let Some(v) = part.strip_prefix("time=") {
            query_time_ms = parse_u64_field("time", v)?;
        } else if let Some(v) = part.strip_prefix("state=") {
            state = v.to_string();
        } else if part.starts_with("tables=")
            || part.starts_with("env=")
            || part.starts_with("var=")
            || part.starts_with("operators=")
        {
            feature_parts.push(part);
        }
    }

    let Some(effective_mem_bytes) =
        compute_effective_mem_bytes(mem_cost_bytes, spilled_bytes, &state)
    else {
        return Ok(None);
    };

    let feature_str = feature_parts.join("|");
    let plan_features = parse_feature_string(&feature_str)?;

    let timestamp_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Ok(Some(TelemetrySample {
        digest,
        plan_features,
        effective_mem_bytes,
        scan_bytes,
        scan_rows,
        return_rows,
        cpu_cost_ns,
        spilled_bytes,
        query_time_ms,
        state,
        timestamp_s,
    }))
}

fn parse_u64_field(name: &str, raw: &str) -> Result<u64, ParseError> {
    raw.trim()
        .parse::<i64>()
        .map(|v| v.max(0) as u64)
        .map_err(|e| ParseError::InvalidFormat(format!("invalid {name}='{raw}': {e}")))
}

#[derive(Deserialize)]
struct JsonProfileHelper {
    #[serde(default)]
    tables: Vec<String>,
    #[serde(default, alias = "peakMemBytes", alias = "PeakMemoryUsage")]
    peak_mem_bytes: u64,
    #[serde(default, alias = "scanBytes", alias = "BytesRead")]
    scan_bytes: u64,
    #[serde(default, alias = "scanRows", alias = "RowsRead")]
    scan_rows: u64,
    #[serde(default, alias = "spilledBytes", alias = "SpillBytes")]
    spilled_bytes: u64,
}

pub fn parse_query_profile(payload: &str) -> Result<ProfileSummary, ParseError> {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return Err(ParseError::InvalidFormat("empty query profile".to_string()));
    }

    if trimmed.starts_with('{') {
        let helper: JsonProfileHelper = serde_json::from_str(trimmed)?;
        return Ok(ProfileSummary {
            tables: helper.tables,
            peak_mem_bytes: helper.peak_mem_bytes,
            scan_bytes: helper.scan_bytes,
            scan_rows: helper.scan_rows,
            spilled_bytes: helper.spilled_bytes,
        });
    }

    let mut summary = ProfileSummary::default();
    for line in trimmed.lines() {
        let l = line.trim().trim_start_matches('-').trim();
        if let Some((key, val)) = l.split_once(':') {
            let key = key.trim();
            let val = val.trim();
            if key.eq_ignore_ascii_case("Table") || key.eq_ignore_ascii_case("OlapScanNode") {
                let table_name = val.split_whitespace().next().unwrap_or("").trim();
                if !table_name.is_empty() && !summary.tables.iter().any(|t| t == table_name) {
                    summary.tables.push(table_name.to_string());
                }
            } else if key.contains("PeakMemoryUsage") || key.eq_ignore_ascii_case("peak_mem_bytes")
            {
                if let Some(bytes) = parse_size_or_count(val) {
                    summary.peak_mem_bytes = summary.peak_mem_bytes.max(bytes);
                }
            } else if key.contains("BytesRead")
                || key.contains("ScanBytes")
                || key.eq_ignore_ascii_case("scan_bytes")
            {
                if let Some(bytes) = parse_size_or_count(val) {
                    summary.scan_bytes = summary.scan_bytes.saturating_add(bytes);
                }
            } else if key.contains("RowsRead")
                || key.contains("RowsReturned")
                || key.eq_ignore_ascii_case("scan_rows")
            {
                if let Some(rows) = parse_size_or_count(val) {
                    summary.scan_rows = summary.scan_rows.saturating_add(rows);
                }
            } else if key.contains("SpillBytes")
                || key.contains("SpilledBytes")
                || key.eq_ignore_ascii_case("spilled_bytes")
            {
                if let Some(bytes) = parse_size_or_count(val) {
                    summary.spilled_bytes = summary.spilled_bytes.saturating_add(bytes);
                }
            }
        }
    }

    Ok(summary)
}

fn parse_size_or_count(raw: &str) -> Option<u64> {
    let clean = raw.split('(').next().unwrap_or(raw).trim();
    let mut num_end = 0;
    for (idx, ch) in clean.char_indices() {
        if ch.is_ascii_digit() || ch == '.' {
            num_end = idx + ch.len_utf8();
        } else if num_end > 0 {
            break;
        }
    }
    if num_end == 0 {
        return None;
    }
    let num: f64 = clean[..num_end].parse().ok()?;
    let unit = clean[num_end..].trim().to_ascii_uppercase();
    let multiplier = match unit.as_str() {
        "KB" | "KIB" | "K" => 1024.0,
        "MB" | "MIB" | "M" => 1024.0 * 1024.0,
        "GB" | "GIB" | "G" => 1024.0 * 1024.0 * 1024.0,
        "TB" | "TIB" | "T" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => 1.0,
    };
    Some((num * multiplier).max(0.0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_features_log_line_with_log4j_prefix_and_oom_filtering() {
        let line = "2026-10-28 21:00:01,112 INFO (qe|99) [AuditLogBuilder.java:185] digest=abc123|cpuCostNs=500000|memCostBytes=40000000|scanBytes=10000000|scanRows=50000|returnRows=100|spilledBytes=10000000|time=120|state=EOF|catalog=default_catalog|database=db1|tables=[0,0,1001]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]";
        let sample = parse_features_log_line(line)
            .unwrap()
            .expect("valid sample");
        assert_eq!(sample.digest, "abc123");
        assert_eq!(sample.effective_mem_bytes, 50_000_000);

        let err_line = "digest=def|cpuCostNs=0|memCostBytes=0|scanBytes=0|scanRows=0|returnRows=0|spilledBytes=0|time=1|state=ERR|catalog=c|database=d|tables=[0,0,0]|env=[1,8,20]|var=[1]|operators=[39,0,0,0,0,0,0,0,0,0,0]";
        assert!(parse_features_log_line(err_line).unwrap().is_none());
    }

    #[test]
    fn test_parse_query_profile_json_and_text_formats() {
        let json_profile = r#"{"tables":["1001","1002"],"peak_mem_bytes":104857600,"scan_bytes":52428800,"scan_rows":120000,"spilled_bytes":4096}"#;
        let p1 = parse_query_profile(json_profile).unwrap();
        assert_eq!(p1.tables, vec!["1001", "1002"]);
        assert_eq!(p1.peak_mem_bytes, 104_857_600);
        assert_eq!(p1.scan_bytes, 52_428_800);
        assert_eq!(p1.scan_rows, 120_000);
        assert_eq!(p1.spilled_bytes, 4096);

        let text_profile = "\
Query:
  - PeakMemoryUsage: 64.00 MB
  - OlapScanNode: orders
  - Table: lineitem
  - BytesRead: 10.00 MB
  - RowsRead: 25000
  - SpillBytes: 2.00 MB
";
        let p2 = parse_query_profile(text_profile).unwrap();
        assert_eq!(p2.tables, vec!["orders", "lineitem"]);
        assert_eq!(p2.peak_mem_bytes, 64 * 1024 * 1024);
        assert_eq!(p2.scan_bytes, 10 * 1024 * 1024);
        assert_eq!(p2.scan_rows, 25_000);
        assert_eq!(p2.spilled_bytes, 2 * 1024 * 1024);
    }

    #[test]
    fn test_audit_log_event_deserialization_with_camel_case_aliases() {
        let raw = r#"{"digest":"d99","predictMemBytes":67108864,"memCostBytes":80000000,"scanBytes":12000,"scanRows":500,"returnRows":10,"cpuCostNs":900000,"spilledBytes":0,"queryTime":45,"tables":["1001"]}"#;
        let ev: AuditLogEvent = serde_json::from_str(raw).unwrap();
        assert_eq!(ev.digest, "d99");
        assert_eq!(ev.predict_mem_bytes, 67_108_864);
        assert_eq!(ev.mem_cost_bytes, 80_000_000);
        assert_eq!(ev.query_time_ms, 45);
        assert_eq!(ev.state, "EOF");
        assert_eq!(ev.tables, vec!["1001"]);
    }
}
