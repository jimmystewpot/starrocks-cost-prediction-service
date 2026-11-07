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
