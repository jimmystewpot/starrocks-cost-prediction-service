use crate::types::{CanonicalPlanFeatures, OperatorSegment, OperatorSegmentKind};
use xxhash_rust::xxh3::xxh3_64;

const SEGMENT_LENGTHS: [usize; 4] = [7, 8, 10, 11];

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("empty input")]
    EmptyInput,
    #[error("invalid CSV format: {0}")]
    InvalidCsvFormat(String),
    #[error("invalid feature string: {0}")]
    InvalidFeatureString(String),
    #[error("integer parse error: {0}")]
    ParseInt(#[from] std::num::ParseIntError),
    #[error("invalid operator segments: {0}")]
    InvalidOperatorSegments(String),
}

pub fn parse_predict_csv(csv_body: &str) -> Result<CanonicalPlanFeatures, DecodeError> {
    let mut non_empty_lines = csv_body.lines().map(str::trim).filter(|l| !l.is_empty());
    let first_line = non_empty_lines.next().ok_or(DecodeError::EmptyInput)?;

    let data_line = if first_line.starts_with("tables_0") || first_line.contains("operators_0") {
        non_empty_lines.next().ok_or_else(|| {
            DecodeError::InvalidCsvFormat("CSV header present but data row is missing".to_string())
        })?
    } else {
        first_line
    };

    let mut cols = data_line.split(',').map(str::trim);
    let mut header = [""; 7];
    for (i, slot) in header.iter_mut().enumerate() {
        *slot = cols.next().ok_or_else(|| {
            DecodeError::InvalidCsvFormat(format!("expected at least 7 columns, got {i}"))
        })?;
    }

    let tables = [
        header[0].to_string(),
        header[1].to_string(),
        header[2].to_string(),
    ];
    let num_be_nodes = header[3].parse::<i64>()?.max(0) as u64;
    let avg_cpu_cores = header[4].parse::<i64>()?.max(0) as u64;
    let env_mem_log = header[5].parse::<i64>()?.max(0) as u64;
    let pipeline_dop = header[6].parse::<i64>()?.max(0) as u64;

    let mut ops = Vec::with_capacity(64);
    for token in cols {
        if !token.is_empty() {
            ops.push(token.parse::<i64>()?);
        }
    }

    build_canonical_plan_features(
        tables,
        num_be_nodes,
        avg_cpu_cores,
        env_mem_log,
        pipeline_dop,
        &ops,
    )
}

pub fn parse_feature_string(feature_str: &str) -> Result<CanonicalPlanFeatures, DecodeError> {
    let trimmed = feature_str.trim();
    if trimmed.is_empty() {
        return Err(DecodeError::EmptyInput);
    }

    let mut tables_opt: Option<[String; 3]> = None;
    let mut env_opt: Option<(u64, u64, u64)> = None;
    let mut var_opt: Option<u64> = None;
    let mut ops_opt: Option<Vec<i64>> = None;

    for part in trimmed.split('|') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("tables=") {
            let inner = strip_brackets(rest);
            let mut items: Vec<String> = if inner.is_empty() {
                Vec::new()
            } else {
                inner.split(',').map(|s| s.trim().to_string()).collect()
            };
            while items.len() < 3 {
                items.insert(0, "0".to_string());
            }
            tables_opt = Some([items[0].clone(), items[1].clone(), items[2].clone()]);
        } else if let Some(rest) = part.strip_prefix("env=") {
            let inner = strip_brackets(rest);
            let vals = parse_i64_list(inner)?;
            if vals.len() < 3 {
                return Err(DecodeError::InvalidFeatureString(format!(
                    "expected 3 env values, got {}",
                    vals.len()
                )));
            }
            env_opt = Some((
                vals[0].max(0) as u64,
                vals[1].max(0) as u64,
                vals[2].max(0) as u64,
            ));
        } else if let Some(rest) = part.strip_prefix("var=") {
            let inner = strip_brackets(rest);
            let vals = parse_i64_list(inner)?;
            if vals.is_empty() {
                return Err(DecodeError::InvalidFeatureString(
                    "expected at least 1 var value".to_string(),
                ));
            }
            var_opt = Some(vals[0].max(0) as u64);
        } else if let Some(rest) = part.strip_prefix("operators=") {
            let inner = strip_brackets(rest);
            ops_opt = Some(parse_i64_list(inner)?);
        }
    }

    let tables = tables_opt
        .ok_or_else(|| DecodeError::InvalidFeatureString("missing tables=[...]".to_string()))?;
    let (num_be_nodes, avg_cpu_cores, env_mem_log) = env_opt
        .ok_or_else(|| DecodeError::InvalidFeatureString("missing env=[...]".to_string()))?;
    let pipeline_dop = var_opt
        .ok_or_else(|| DecodeError::InvalidFeatureString("missing var=[...]".to_string()))?;
    let ops = ops_opt
        .ok_or_else(|| DecodeError::InvalidFeatureString("missing operators=[...]".to_string()))?;

    build_canonical_plan_features(
        tables,
        num_be_nodes,
        avg_cpu_cores,
        env_mem_log,
        pipeline_dop,
        &ops,
    )
}

fn strip_brackets(s: &str) -> &str {
    let s = s.trim();
    let s = s.strip_prefix('[').unwrap_or(s);
    let s = s.strip_suffix(']').unwrap_or(s);
    s.trim()
}

fn parse_i64_list(s: &str) -> Result<Vec<i64>, DecodeError> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for tok in s.split(',') {
        let tok = tok.trim();
        if !tok.is_empty() {
            out.push(tok.parse::<i64>()?);
        }
    }
    Ok(out)
}

pub fn decode_operator_segments(ops: &[i64]) -> Result<Vec<OperatorSegment>, DecodeError> {
    if ops.is_empty() {
        return Ok(Vec::new());
    }
    if ops.len() < 7 {
        return Err(DecodeError::InvalidOperatorSegments(format!(
            "operator slice length {} is shorter than minimum segment length 7",
            ops.len()
        )));
    }

    let mut path = Vec::new();
    let mut failed = vec![false; ops.len() + 1];
    if !walk_segments(ops, 0, &mut path, &mut failed) {
        return Err(DecodeError::InvalidOperatorSegments(
            "failed to synchronize operator segment boundaries".to_string(),
        ));
    }

    let first_hash_join_idx = path.iter().position(|&(_, len)| len == 10);
    let l11_count = path.iter().filter(|&&(_, len)| len == 11).count();
    let first_l11_idx = path.iter().position(|&(_, len)| len == 11);

    let mut segments = Vec::with_capacity(path.len());
    for (idx, &(pos, len)) in path.iter().enumerate() {
        let ordinal = ops[pos];
        let count = ops[pos + 1] as u64;
        let kind = match len {
            8 => OperatorSegmentKind::Exchange,
            10 => OperatorSegmentKind::HashJoin,
            7 => OperatorSegmentKind::Generic,
            11 => {
                if let Some(hj_idx) = first_hash_join_idx {
                    if idx < hj_idx {
                        OperatorSegmentKind::HashAgg
                    } else {
                        OperatorSegmentKind::Scan
                    }
                } else if (l11_count > 1 && Some(idx) == first_l11_idx) || ordinal <= 36 {
                    OperatorSegmentKind::HashAgg
                } else {
                    OperatorSegmentKind::Scan
                }
            }
            _ => unreachable!(),
        };

        let (mean_log_rows, mean_row_size, mean_log_cbo_mem, mean_log_cbo_cpu, mean_log_limit) =
            if count > 0 {
                let c = count as f64;
                (
                    ops[pos + 2] as f64 / c,
                    ops[pos + 3] as f64 / c,
                    ops[pos + 4] as f64 / c,
                    ops[pos + 5] as f64 / c,
                    ops[pos + 6] as f64 / c,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };

        let extra = ops[pos + 7..pos + len].to_vec();
        segments.push(OperatorSegment {
            ordinal,
            kind,
            count,
            mean_log_rows,
            mean_row_size,
            mean_log_cbo_mem,
            mean_log_cbo_cpu,
            mean_log_limit,
            extra,
        });
    }

    Ok(segments)
}

fn walk_segments(
    ops: &[i64],
    pos: usize,
    path: &mut Vec<(usize, usize)>,
    failed: &mut [bool],
) -> bool {
    if pos == ops.len() {
        return true;
    }
    if failed[pos] || pos + 7 > ops.len() {
        return false;
    }

    let cur_ord = ops[pos];
    let count = ops[pos + 1];
    if cur_ord < 0 || count < 0 {
        failed[pos] = true;
        return false;
    }

    // Base 5 features must be non-negative
    if ops[pos + 2..pos + 7].iter().any(|&v| v < 0) {
        failed[pos] = true;
        return false;
    }

    // Collect valid candidate lengths and sort by transition priority
    let mut candidates: Vec<(i64, usize)> = Vec::with_capacity(4);
    for &len in &SEGMENT_LENGTHS {
        if pos + len > ops.len() {
            continue;
        }
        if count == 0 {
            if ops[pos + 2..pos + len].iter().any(|&v| v != 0) {
                continue;
            }
        } else if ops[pos + 7..pos + len].iter().any(|&v| v < 0) {
            continue;
        }

        if pos + len == ops.len() {
            candidates.push((0, len));
        } else if pos + len + 7 <= ops.len() {
            let next_ord = ops[pos + len];
            let next_count = ops[pos + len + 1];
            let delta = next_ord - cur_ord;
            if (1..=5).contains(&delta) && next_count >= 0 {
                candidates.push((delta, len));
            }
        }
    }

    candidates.sort_by_key(|&(delta, len)| (delta, len));

    for (_, len) in candidates {
        path.push((pos, len));
        if walk_segments(ops, pos + len, path, failed) {
            return true;
        }
        path.pop();
    }

    failed[pos] = true;
    false
}

fn build_canonical_plan_features(
    tables: [String; 3],
    num_be_nodes: u64,
    avg_cpu_cores: u64,
    env_mem_log: u64,
    pipeline_dop: u64,
    ops: &[i64],
) -> Result<CanonicalPlanFeatures, DecodeError> {
    let segments = decode_operator_segments(ops)?;
    let plan_hash = compute_plan_hash(&tables, &segments);
    let plan_features_24 = compute_plan_features_24(&segments, num_be_nodes, pipeline_dop);

    Ok(CanonicalPlanFeatures {
        tables,
        num_be_nodes,
        avg_cpu_cores,
        env_mem_log,
        pipeline_dop,
        segments,
        plan_hash,
        plan_features_24,
    })
}

fn compute_plan_hash(tables: &[String; 3], segments: &[OperatorSegment]) -> u64 {
    let mut sorted_tables = tables.clone();
    sorted_tables.sort();

    let mut buf = Vec::with_capacity(128);
    for t in &sorted_tables {
        buf.extend_from_slice(t.as_bytes());
        buf.push(0xFF);
    }

    for seg in segments {
        if seg.count == 0 {
            continue;
        }
        buf.extend_from_slice(&seg.ordinal.to_le_bytes());
        buf.extend_from_slice(&seg.count.to_le_bytes());
        match seg.kind {
            OperatorSegmentKind::HashJoin if seg.extra.len() >= 3 => {
                buf.extend_from_slice(&seg.extra[1].to_le_bytes());
                buf.extend_from_slice(&seg.extra[2].to_le_bytes());
            }
            OperatorSegmentKind::HashAgg if seg.extra.len() >= 3 => {
                buf.extend_from_slice(&seg.extra[1].to_le_bytes());
                buf.extend_from_slice(&seg.extra[2].to_le_bytes());
            }
            OperatorSegmentKind::Exchange if !seg.extra.is_empty() => {
                buf.extend_from_slice(&seg.extra[0].to_le_bytes());
            }
            OperatorSegmentKind::Scan if seg.extra.len() >= 4 => {
                buf.extend_from_slice(&seg.extra[2].to_le_bytes());
                buf.extend_from_slice(&seg.extra[3].to_le_bytes());
            }
            _ => {}
        }
    }

    xxh3_64(&buf)
}

fn compute_plan_features_24(
    segments: &[OperatorSegment],
    num_be_nodes: u64,
    pipeline_dop: u64,
) -> [f64; 24] {
    let mut total_physical_ops = 0.0_f64;
    let mut active_operator_types = 0.0_f64;

    let mut scan_count = 0.0_f64;
    let mut scan_weighted_log_rows = 0.0_f64;
    let mut scan_weighted_row_size = 0.0_f64;
    let mut scan_tablets_sum = 0.0_f64;
    let mut scan_partitions_sum = 0.0_f64;
    let mut scan_total_predicates = 0.0_f64;

    let mut join_count = 0.0_f64;
    let mut broadcast_join_count = 0.0_f64;
    let mut join_right_log_size_sum = 0.0_f64;

    let mut agg_count = 0.0_f64;
    let mut agg_input_rows_log_sum = 0.0_f64;
    let mut agg_total_group_bys = 0.0_f64;
    let mut agg_total_aggregations = 0.0_f64;
    let mut agg_ratio_sum = 0.0_f64;

    let mut exchange_count = 0.0_f64;
    let mut generic_ops_count = 0.0_f64;

    let mut cbo_mem_sum = 0.0_f64;
    let mut cbo_cpu_sum = 0.0_f64;

    for seg in segments {
        if seg.count == 0 {
            continue;
        }
        let c = seg.count as f64;
        total_physical_ops += c;
        active_operator_types += 1.0;

        cbo_mem_sum += c * (seg.mean_log_cbo_mem.min(40.0).exp() - 1.0).max(0.0);
        cbo_cpu_sum += c * (seg.mean_log_cbo_cpu.min(40.0).exp() - 1.0).max(0.0);

        match seg.kind {
            OperatorSegmentKind::Scan => {
                scan_count += c;
                scan_weighted_log_rows += c * seg.mean_log_rows;
                scan_weighted_row_size += c * seg.mean_row_size;
                let tablets = seg.extra.first().copied().unwrap_or(0).max(0) as f64;
                let partitions = seg.extra.get(1).copied().unwrap_or(0).max(0) as f64;
                let bin_preds = seg.extra.get(2).copied().unwrap_or(0).max(0) as f64;
                let pred_cols = seg.extra.get(3).copied().unwrap_or(0).max(0) as f64;
                scan_tablets_sum += tablets;
                scan_partitions_sum += partitions;
                scan_total_predicates += bin_preds + pred_cols;
            }
            OperatorSegmentKind::HashJoin => {
                join_count += c;
                let right_log_size = seg.extra.first().copied().unwrap_or(0).max(0) as f64;
                let bcast = seg.extra.get(1).copied().unwrap_or(0).max(0) as f64;
                join_right_log_size_sum += right_log_size;
                broadcast_join_count += bcast;
            }
            OperatorSegmentKind::HashAgg => {
                agg_count += c;
                let input_rows_log = seg.extra.first().copied().unwrap_or(0).max(0) as f64;
                let group_bys = seg.extra.get(1).copied().unwrap_or(0).max(0) as f64;
                let aggs = seg.extra.get(2).copied().unwrap_or(0).max(0) as f64;
                let ratio = seg.extra.get(3).copied().unwrap_or(0).max(0) as f64;
                agg_input_rows_log_sum += input_rows_log;
                agg_total_group_bys += group_bys;
                agg_total_aggregations += aggs;
                agg_ratio_sum += ratio;
            }
            OperatorSegmentKind::Exchange => {
                exchange_count += c;
            }
            OperatorSegmentKind::Generic => {
                generic_ops_count += c;
            }
        }
    }

    let scan_mean_log_rows = if scan_count > 0.0 {
        scan_weighted_log_rows / scan_count
    } else {
        0.0
    };
    let scan_est_log_sum_rows = if scan_count > 0.0 {
        scan_mean_log_rows + scan_count.ln()
    } else {
        0.0
    };
    let scan_mean_row_size = if scan_count > 0.0 {
        scan_weighted_row_size / scan_count
    } else {
        0.0
    };
    let scan_total_tablets_log = (1.0 + scan_tablets_sum).ln();
    let scan_total_partitions_log = (1.0 + scan_partitions_sum).ln();

    let partition_join_count = (join_count - broadcast_join_count).max(0.0);
    let join_mean_right_log_size = if join_count > 0.0 {
        join_right_log_size_sum / join_count
    } else {
        0.0
    };
    let join_est_sum_right_log_size = if join_count > 0.0 {
        join_mean_right_log_size + join_count.ln()
    } else {
        0.0
    };

    let agg_mean_log_input_rows = if agg_count > 0.0 {
        agg_input_rows_log_sum / agg_count
    } else {
        0.0
    };
    let agg_mean_log_ratio = if agg_count > 0.0 {
        (1.0 + (agg_ratio_sum / agg_count)).ln()
    } else {
        0.0
    };

    let cbo_est_total_mem_log = if total_physical_ops > 0.0 {
        (1.0 + cbo_mem_sum).ln()
    } else {
        0.0
    };
    let cbo_est_total_cpu_log = if total_physical_ops > 0.0 {
        (1.0 + cbo_cpu_sum).ln()
    } else {
        0.0
    };
    let env_dop_log = (1.0 + pipeline_dop as f64).ln() + (1.0 + num_be_nodes as f64).ln();

    [
        total_physical_ops,          // 0
        active_operator_types,       // 1
        scan_count,                  // 2
        scan_mean_log_rows,          // 3
        scan_est_log_sum_rows,       // 4
        scan_mean_row_size,          // 5
        scan_total_tablets_log,      // 6
        scan_total_partitions_log,   // 7
        scan_total_predicates,       // 8
        join_count,                  // 9
        broadcast_join_count,        // 10
        partition_join_count,        // 11
        join_mean_right_log_size,    // 12
        join_est_sum_right_log_size, // 13
        agg_count,                   // 14
        agg_mean_log_input_rows,     // 15
        agg_total_group_bys,         // 16
        agg_total_aggregations,      // 17
        agg_mean_log_ratio,          // 18
        exchange_count,              // 19
        generic_ops_count,           // 20
        cbo_est_total_mem_log,       // 21
        cbo_est_total_cpu_log,       // 22
        env_dop_log,                 // 23
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::catalog::TableCatalog;
    use crate::types::DENSE_FEATURE_DIM;

    #[test]
    fn test_self_synchronizing_segment_walker_handles_starrocks_operators() {
        let ops: Vec<i64> = vec![
            34, 1, 10, 100, 14, 12, 0, 2, 35, 2, 20, 64, 28, 24, 0, 24, 4, 2, 10, 36, 2, 22, 128,
            32, 30, 0, 20, 2, 1, 38, 0, 0, 0, 0, 0, 0, 39, 1, 15, 100, 18, 16, 0, 12, 3, 2, 2,
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
        let pipe_small =
            "tables=[0,1001,1002]|env=[3,16,24]|var=[8]|operators=[36,1,10,100,14,12,0,12,1,0]";
        let pipe_large =
            "tables=[0,1001,1002]|env=[3,16,24]|var=[8]|operators=[36,1,22,100,24,20,0,20,1,0]";
        let f_small = parse_feature_string(pipe_small).unwrap();
        let f_large = parse_feature_string(pipe_large).unwrap();
        assert_eq!(f_small.plan_hash, f_large.plan_hash);
        assert!(f_large.plan_features_24[21] > f_small.plan_features_24[21]);
    }

    #[test]
    fn test_table_catalog_target_encoding_and_known_ratio() {
        let catalog = TableCatalog::new();
        let pipe =
            "tables=[0,1001,1002]|env=[3,16,24]|var=[8]|operators=[36,1,10,100,14,12,0,12,1,0]";
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
