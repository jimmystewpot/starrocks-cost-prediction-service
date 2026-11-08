use crate::ingest::parser::AuditLogEvent;
use crate::types::{CanonicalPlanFeatures, TelemetrySample, compute_effective_mem_bytes};
use dashmap::DashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct InFlightCache {
    by_watermarked: DashMap<u64, (CanonicalPlanFeatures, Instant)>,
    by_low10: DashMap<u64, (u64, CanonicalPlanFeatures, Instant)>,
    capacity: usize,
    ttl: Duration,
}

impl InFlightCache {
    pub fn new(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            by_watermarked: DashMap::new(),
            by_low10: DashMap::new(),
            capacity: capacity.max(1),
            ttl: Duration::from_secs(ttl_secs.max(1)),
        }
    }

    pub fn record_prediction(&self, watermarked_mem_bytes: u64, plan: CanonicalPlanFeatures) {
        let now = Instant::now();
        if self.by_watermarked.len() >= self.capacity
            && !self.by_watermarked.contains_key(&watermarked_mem_bytes)
        {
            self.evict_one_or_expired(now);
        }

        let low10 = watermarked_mem_bytes & 0x3FF;
        self.by_watermarked
            .insert(watermarked_mem_bytes, (plan.clone(), now));
        self.by_low10
            .insert(low10, (watermarked_mem_bytes, plan, now));
    }

    pub fn correlate_audit_event(&self, event: &AuditLogEvent) -> Option<TelemetrySample> {
        if event.predict_mem_bytes == 0 {
            return None;
        }
        let effective_mem_bytes =
            compute_effective_mem_bytes(event.mem_cost_bytes, event.spilled_bytes, &event.state)?;
        let now = Instant::now();

        let mut matched_plan = None;

        if let Some(entry) = self.by_watermarked.get(&event.predict_mem_bytes) {
            let (plan, created_at) = entry.value();
            if now.duration_since(*created_at) <= self.ttl {
                matched_plan = Some(plan.clone());
            } else {
                drop(entry);
                self.by_watermarked.remove(&event.predict_mem_bytes);
                self.by_low10
                    .remove_if(&(event.predict_mem_bytes & 0x3FF), |_, (wm, _, _)| {
                        *wm == event.predict_mem_bytes
                    });
            }
        }

        if matched_plan.is_none() {
            let low10 = event.predict_mem_bytes & 0x3FF;
            if let Some(entry) = self.by_low10.get(&low10) {
                let (wm_key, plan, created_at) = entry.value();
                if now.duration_since(*created_at) <= self.ttl {
                    if tables_compatible(&event.tables, &plan.tables) {
                        matched_plan = Some(plan.clone());
                    }
                } else {
                    let wm_key = *wm_key;
                    drop(entry);
                    self.by_low10
                        .remove_if(&low10, |_, (wm, _, _)| *wm == wm_key);
                    self.by_watermarked.remove(&wm_key);
                }
            }
        }

        let plan_features = matched_plan?;
        let timestamp_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Some(TelemetrySample {
            digest: event.digest.clone(),
            plan_features,
            effective_mem_bytes,
            scan_bytes: event.scan_bytes,
            scan_rows: event.scan_rows,
            return_rows: event.return_rows,
            cpu_cost_ns: event.cpu_cost_ns,
            spilled_bytes: event.spilled_bytes,
            query_time_ms: event.query_time_ms,
            state: event.state.clone(),
            timestamp_s,
        })
    }

    pub fn len(&self) -> usize {
        self.by_watermarked.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_watermarked.is_empty()
    }

    fn evict_one_or_expired(&self, now: Instant) {
        let mut expired_keys = Vec::new();
        let mut oldest_key: Option<(u64, Instant)> = None;

        for (inspected, entry) in self.by_watermarked.iter().enumerate() {
            let key = *entry.key();
            let created_at = entry.value().1;
            if now.duration_since(created_at) > self.ttl {
                expired_keys.push(key);
            } else if oldest_key.is_none_or(|(_, old_ts)| created_at < old_ts) {
                oldest_key = Some((key, created_at));
            }
            if inspected >= 64 && (!expired_keys.is_empty() || oldest_key.is_some()) {
                break;
            }
        }

        if !expired_keys.is_empty() {
            for k in expired_keys {
                self.by_watermarked.remove(&k);
                self.by_low10
                    .remove_if(&(k & 0x3FF), |_, (wm, _, _)| *wm == k);
            }
        } else if let Some((k, _)) = oldest_key {
            self.by_watermarked.remove(&k);
            self.by_low10
                .remove_if(&(k & 0x3FF), |_, (wm, _, _)| *wm == k);
        }
    }
}

fn tables_compatible(audit_tables: &[String], plan_tables: &[String; 3]) -> bool {
    if audit_tables.is_empty() {
        return true;
    }
    let has_non_zero_plan_table = plan_tables.iter().any(|t| !t.is_empty() && t != "0");
    if !has_non_zero_plan_table {
        return true;
    }
    audit_tables.iter().any(|at| {
        !at.is_empty()
            && at != "0"
            && plan_tables
                .iter()
                .any(|pt| pt == at || at.ends_with(&format!(".{pt}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::decoder::parse_feature_string;
    use crate::types::watermark_mem_bytes;

    #[test]
    fn test_inflight_cache_ttl_expiry_and_edge_cases() {
        let cache = InFlightCache::new(2, 1);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        // correlate_audit_event with predict_mem_bytes == 0 -> None
        let audit_zero_predict = AuditLogEvent {
            digest: "d0".to_string(),
            predict_mem_bytes: 0,
            mem_cost_bytes: 50_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "EOF".to_string(),
            tables: vec!["db.t1".to_string()],
        };
        assert!(cache.correlate_audit_event(&audit_zero_predict).is_none());

        // correlate_audit_event with state == "RUNNING" -> None
        let audit_running = AuditLogEvent {
            digest: "d_running".to_string(),
            predict_mem_bytes: 67_108_864,
            mem_cost_bytes: 50_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "RUNNING".to_string(),
            tables: vec!["db.t1".to_string()],
        };
        assert!(cache.correlate_audit_event(&audit_running).is_none());

        // Test tables_compatible via correlate_audit_event low10 fallback:
        // 1. when audit.tables is empty (vec![])
        let plan_with_tables = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let wm_with_tables = watermark_mem_bytes(50_000_000, plan_with_tables.plan_hash, 1_048_576);
        cache.record_prediction(wm_with_tables, plan_with_tables.clone());

        // Low-10 fallback occurs when high bits differ but low 10 bits match
        let shifted_wm_empty_tables = (80_000_000 & !0x3FF) | (wm_with_tables & 0x3FF);
        let audit_empty_tables = AuditLogEvent {
            digest: "d_empty_tbl".to_string(),
            predict_mem_bytes: shifted_wm_empty_tables,
            mem_cost_bytes: 50_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "EOF".to_string(),
            tables: vec![],
        };
        let sample_empty_tables = cache
            .correlate_audit_event(&audit_empty_tables)
            .expect("low10 fallback must succeed when audit.tables is empty");
        assert_eq!(
            sample_empty_tables.plan_features.plan_hash,
            plan_with_tables.plan_hash
        );

        // Incompatible audit tables should return None
        let audit_incompat = AuditLogEvent {
            digest: "d_incompat".to_string(),
            predict_mem_bytes: shifted_wm_empty_tables,
            mem_cost_bytes: 50_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "EOF".to_string(),
            tables: vec!["unrelated_table_999".to_string()],
        };
        assert!(cache.correlate_audit_event(&audit_incompat).is_none());

        // 2. when plan.tables are all "0" (["0", "0", "0"])
        let plan_all_zeros = parse_feature_string(
            "tables=[0,0,0]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let wm_all_zeros = watermark_mem_bytes(60_000_000, plan_all_zeros.plan_hash, 1_048_576);
        cache.record_prediction(wm_all_zeros, plan_all_zeros.clone());

        let shifted_wm_zero_tables = (85_000_000 & !0x3FF) | (wm_all_zeros & 0x3FF);
        let audit_zero_tables = AuditLogEvent {
            digest: "d_zero_tbl".to_string(),
            predict_mem_bytes: shifted_wm_zero_tables,
            mem_cost_bytes: 60_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "EOF".to_string(),
            tables: vec!["db.arbitrary_table".to_string()],
        };
        let sample_zero_tables = cache
            .correlate_audit_event(&audit_zero_tables)
            .expect("low10 fallback must succeed when plan.tables are all zeros");
        assert_eq!(
            sample_zero_tables.plan_features.plan_hash,
            plan_all_zeros.plan_hash
        );

        // Record entries for direct lookup, fallback lookup, and capacity eviction
        let plan_exp1 = parse_feature_string(
            "tables=[0,0,501]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let plan_exp2 = parse_feature_string(
            "tables=[0,0,502]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let plan_exp3 = parse_feature_string(
            "tables=[0,0,503]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();

        // Use cache (capacity 2, ttl 1s) to test direct & fallback TTL expiry
        let wm_exp1 = watermark_mem_bytes(70_000_000, plan_exp1.plan_hash, 1_048_576);
        let wm_exp2 = watermark_mem_bytes(75_000_000, plan_exp2.plan_hash, 1_048_576);
        cache.record_prediction(wm_exp1, plan_exp1.clone());
        cache.record_prediction(wm_exp2, plan_exp2.clone());

        // Prepare eviction cache (capacity 2, ttl 1s) with 2 entries before sleeping
        let evict_cache = InFlightCache::new(2, 1);
        let wm_evict1 = watermark_mem_bytes(90_000_000, plan_exp1.plan_hash, 1_048_576);
        let wm_evict2 = watermark_mem_bytes(95_000_000, plan_exp2.plan_hash, 1_048_576);
        let wm_evict3 = watermark_mem_bytes(100_000_000, plan_exp3.plan_hash, 1_048_576);
        evict_cache.record_prediction(wm_evict1, plan_exp1.clone());
        evict_cache.record_prediction(wm_evict2, plan_exp2.clone());
        assert_eq!(evict_cache.len(), 2);

        // Sleep to exceed 1-second TTL
        std::thread::sleep(Duration::from_millis(1100));

        // Direct by_watermarked lookup detects expired entry, removes it, and returns None
        let audit_exp1 = AuditLogEvent {
            digest: "d_exp1".to_string(),
            predict_mem_bytes: wm_exp1,
            mem_cost_bytes: 70_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "EOF".to_string(),
            tables: vec!["501".to_string()],
        };
        assert!(cache.correlate_audit_event(&audit_exp1).is_none());
        assert!(!cache.by_watermarked.contains_key(&wm_exp1));

        // Fallback by_low10 lookup detects expired entry, removes it, and returns None
        let shifted_wm_exp2 = (88_000_000 & !0x3FF) | (wm_exp2 & 0x3FF);
        let audit_exp2 = AuditLogEvent {
            digest: "d_exp2".to_string(),
            predict_mem_bytes: shifted_wm_exp2,
            mem_cost_bytes: 75_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "EOF".to_string(),
            tables: vec!["502".to_string()],
        };
        assert!(cache.correlate_audit_event(&audit_exp2).is_none());
        assert!(!cache.by_watermarked.contains_key(&wm_exp2));

        // Filling cache past capacity triggers evict_one_or_expired and purges expired_keys
        evict_cache.record_prediction(wm_evict3, plan_exp3.clone());
        assert_eq!(evict_cache.len(), 1);
        assert!(!evict_cache.by_watermarked.contains_key(&wm_evict1));
        assert!(!evict_cache.by_watermarked.contains_key(&wm_evict2));
        assert!(evict_cache.by_watermarked.contains_key(&wm_evict3));

        // Also verify eviction on `cache` directly: record 2 fresh entries, sleep, and record 3rd
        let wm_c1 = watermark_mem_bytes(110_000_000, plan_exp1.plan_hash, 1_048_576);
        let wm_c2 = watermark_mem_bytes(115_000_000, plan_exp2.plan_hash, 1_048_576);
        let wm_c3 = watermark_mem_bytes(120_000_000, plan_exp3.plan_hash, 1_048_576);
        cache.record_prediction(wm_c1, plan_exp1);
        cache.record_prediction(wm_c2, plan_exp2);
        assert_eq!(cache.len(), 2);
        std::thread::sleep(Duration::from_millis(1100));
        cache.record_prediction(wm_c3, plan_exp3);
        assert_eq!(cache.len(), 1);
        assert!(!cache.by_watermarked.contains_key(&wm_c1));
        assert!(!cache.by_watermarked.contains_key(&wm_c2));
        assert!(cache.by_watermarked.contains_key(&wm_c3));
    }

    #[test]
    fn test_inflight_cache_correlates_audit_event_via_watermarked_mem_bytes() {
        let cache = InFlightCache::new(1000, 1800);
        let plan = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let watermarked = watermark_mem_bytes(80_000_000, plan.plan_hash, 1_048_576);
        cache.record_prediction(watermarked, plan.clone());
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

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
    fn test_inflight_cache_low10_fallback_and_capacity_eviction() {
        let cache = InFlightCache::new(2, 1800);
        let plan1 = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let plan2 = parse_feature_string(
            "tables=[0,0,202]|env=[3,16,24]|var=[8]|operators=[39,1,15,100,16,15,0,12,2,1,1]",
        )
        .unwrap();
        let plan3 = parse_feature_string(
            "tables=[0,0,303]|env=[3,16,24]|var=[8]|operators=[39,1,16,100,16,16,0,12,2,1,1]",
        )
        .unwrap();

        let wm1 = watermark_mem_bytes(50_000_000, plan1.plan_hash, 1_048_576);
        let wm2 = watermark_mem_bytes(60_000_000, plan2.plan_hash, 1_048_576);
        let wm3 = watermark_mem_bytes(70_000_000, plan3.plan_hash, 1_048_576);

        cache.record_prediction(wm1, plan1);
        cache.record_prediction(wm2, plan2.clone());
        cache.record_prediction(wm3, plan3.clone());
        assert_eq!(cache.len(), 2);

        // Fallback via 10-bit watermark when high bits differ slightly
        let shifted_wm3 = (90_000_000 & !0x3FF) | (plan3.plan_hash & 0x3FF);
        let audit = AuditLogEvent {
            digest: "d3".to_string(),
            predict_mem_bytes: shifted_wm3,
            mem_cost_bytes: 70_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "EOF".to_string(),
            tables: vec!["db.303".to_string()],
        };
        let sample = cache
            .correlate_audit_event(&audit)
            .expect("low10 fallback must correlate");
        assert_eq!(sample.plan_features.plan_hash, plan3.plan_hash);
    }

    #[test]
    fn test_inflight_cache_eviction_preserves_colliding_low10_entry() {
        let cache = InFlightCache::new(2, 1800);
        let plan1 = parse_feature_string(
            "tables=[0,0,101]|env=[3,16,24]|var=[8]|operators=[39,1,14,100,16,14,0,12,2,1,1]",
        )
        .unwrap();
        let plan2 = parse_feature_string(
            "tables=[0,0,202]|env=[3,16,24]|var=[8]|operators=[39,1,15,100,16,15,0,12,2,1,1]",
        )
        .unwrap();
        let plan3 = parse_feature_string(
            "tables=[0,0,303]|env=[3,16,24]|var=[8]|operators=[39,1,16,100,16,16,0,12,2,1,1]",
        )
        .unwrap();

        let shared_low10: u64 = 0x2AB;
        let wm1 = (50_000_000 & !0x3FF) | shared_low10;
        let wm2 = (60_000_000 & !0x3FF) | shared_low10;
        let wm3 = (70_000_000 & !0x3FF) | ((shared_low10 + 1) & 0x3FF);

        cache.record_prediction(wm1, plan1);
        std::thread::sleep(Duration::from_millis(2));
        cache.record_prediction(wm2, plan2.clone());
        std::thread::sleep(Duration::from_millis(2));
        // Recording wm3 evicts wm1 (the oldest entry). It must NOT evict wm2 from by_low10.
        cache.record_prediction(wm3, plan3);
        assert_eq!(cache.len(), 2);

        let shifted_wm2 = (85_000_000 & !0x3FF) | shared_low10;
        let audit = AuditLogEvent {
            digest: "d2".to_string(),
            predict_mem_bytes: shifted_wm2,
            mem_cost_bytes: 60_000_000,
            scan_bytes: 1000,
            scan_rows: 10,
            return_rows: 1,
            cpu_cost_ns: 100,
            spilled_bytes: 0,
            query_time_ms: 10,
            state: "EOF".to_string(),
            tables: vec!["db.202".to_string()],
        };
        let sample = cache
            .correlate_audit_event(&audit)
            .expect("wm2 in by_low10 must survive eviction of colliding wm1");
        assert_eq!(sample.plan_features.plan_hash, plan2.plan_hash);
    }
}
