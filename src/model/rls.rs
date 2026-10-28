use crate::types::CanonicalPlanFeatures;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

const FORGETTING_FACTOR: f64 = 0.995;
const WEIGHT_MIN: f64 = -50.0;
const WEIGHT_MAX: f64 = 50.0;
const P_DIAG_MIN: f64 = 1e-6;
const P_DIAG_MAX: f64 = 1e4;
const P_OFFDIAG_MAX: f64 = 1e4;
const MIN_RESID_VAR_MEM: f64 = 0.09;
const Z_P90: f64 = 1.2816;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlsState {
    pub w_mem: [f64; 6],
    pub w_scan: [f64; 6],
    pub w_rows: [f64; 6],
    pub w_cpu: [f64; 6],
    pub p: [[f64; 6]; 6],
    pub ewma_resid_var_mem: f64,
    pub update_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlsPrediction {
    pub log_mem_p50: f64,
    pub log_mem_p90: f64,
    pub log_scan_bytes: f64,
    pub log_return_rows: f64,
    pub log_cpu_ns: f64,
}

pub struct OnlineRlsCalibrator {
    state: RwLock<RlsState>,
}

fn extract_features(plan: &CanonicalPlanFeatures) -> [f64; 6] {
    [
        1.0,
        plan.plan_features_24[21],
        plan.plan_features_24[4].max(plan.plan_features_24[3]),
        (1.0 + plan.plan_features_24[9].max(0.0)).ln(),
        (1.0 + plan.plan_features_24[14].max(0.0)).ln(),
        plan.plan_features_24[23],
    ]
}

fn dot(w: &[f64; 6], x: &[f64; 6]) -> f64 {
    let mut sum = 0.0;
    for i in 0..6 {
        sum += w[i] * x[i];
    }
    sum
}

fn update_weights(w: &mut [f64; 6], k: &[f64; 6], x: &[f64; 6], y: f64) {
    let pred = dot(w, x);
    let err = y - pred;
    for i in 0..6 {
        let next = w[i] + k[i] * err;
        w[i] = if next.is_finite() {
            next.clamp(WEIGHT_MIN, WEIGHT_MAX)
        } else {
            0.0
        };
    }
}

impl OnlineRlsCalibrator {
    pub fn new() -> Self {
        let mut p = [[0.0; 6]; 6];
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = 10.0;
        }
        Self {
            state: RwLock::new(RlsState {
                w_mem: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                w_scan: [2.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                w_rows: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                w_cpu: [4.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                p,
                ewma_resid_var_mem: 0.30,
                update_count: 0,
            }),
        }
    }

    pub fn predict(&self, plan: &CanonicalPlanFeatures) -> RlsPrediction {
        let x = extract_features(plan);
        let state = self.state.read().unwrap();

        let log_mem_p50 = if x[1] == 0.0 && state.update_count == 0 {
            18.0
        } else {
            dot(&state.w_mem, &x).clamp(0.0, 50.0)
        };
        let log_mem_p90 =
            log_mem_p50 + Z_P90 * state.ewma_resid_var_mem.max(MIN_RESID_VAR_MEM).sqrt();
        let log_scan_bytes = dot(&state.w_scan, &x).clamp(0.0, 50.0);
        let log_return_rows = dot(&state.w_rows, &x).clamp(0.0, 50.0);
        let log_cpu_ns = dot(&state.w_cpu, &x).clamp(0.0, 60.0);

        RlsPrediction {
            log_mem_p50,
            log_mem_p90,
            log_scan_bytes,
            log_return_rows,
            log_cpu_ns,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn update(
        &self,
        plan: &CanonicalPlanFeatures,
        effective_mem_bytes: u64,
        scan_bytes: u64,
        return_rows: u64,
        cpu_cost_ns: u64,
    ) {
        let x = extract_features(plan);
        let y_mem = (effective_mem_bytes as f64).ln_1p();
        let y_scan = (scan_bytes as f64).ln_1p();
        let y_rows = (return_rows as f64).ln_1p();
        let y_cpu = (cpu_cost_ns as f64).ln_1p();

        let mut state = self.state.write().unwrap();

        // Compute P * x and x^T * P
        let mut px = [0.0; 6];
        let mut xtp = [0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                px[i] += state.p[i][j] * x[j];
                xtp[j] += x[i] * state.p[i][j];
            }
        }

        let xpx = dot(&x, &px);
        let denom = (FORGETTING_FACTOR + xpx).max(1e-9);
        let mut k = [0.0; 6];
        for i in 0..6 {
            k[i] = px[i] / denom;
        }

        // Track EWMA residual variance of memory before weight update
        let err_mem = y_mem - dot(&state.w_mem, &x);
        if err_mem.is_finite() {
            state.ewma_resid_var_mem = (0.95 * state.ewma_resid_var_mem
                + 0.05 * (err_mem * err_mem).min(25.0))
            .clamp(MIN_RESID_VAR_MEM, 25.0);
        }

        update_weights(&mut state.w_mem, &k, &x, y_mem);
        update_weights(&mut state.w_scan, &k, &x, y_scan);
        update_weights(&mut state.w_rows, &k, &x, y_rows);
        update_weights(&mut state.w_cpu, &k, &x, y_cpu);

        let mut next_p = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                next_p[i][j] = (state.p[i][j] - k[i] * xtp[j]) / FORGETTING_FACTOR;
            }
        }

        for i in 0..6 {
            for j in 0..6 {
                let sym = 0.5 * (next_p[i][j] + next_p[j][i]);
                state.p[i][j] = if i == j {
                    if sym.is_finite() {
                        sym.clamp(P_DIAG_MIN, P_DIAG_MAX)
                    } else {
                        10.0
                    }
                } else if sym.is_finite() {
                    sym.clamp(-P_OFFDIAG_MAX, P_OFFDIAG_MAX)
                } else {
                    0.0
                };
            }
        }

        state.update_count = state.update_count.saturating_add(1);
    }

    pub fn to_state(&self) -> RlsState {
        self.state.read().unwrap().clone()
    }

    pub fn from_state(state: RlsState) -> Self {
        Self {
            state: RwLock::new(state),
        }
    }
}

impl Default for OnlineRlsCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_plan(
        plan_hash: u64,
        scan_log_rows: f64,
        cbo_mem_log: f64,
    ) -> CanonicalPlanFeatures {
        let mut plan_features_24 = [0.0; 24];
        plan_features_24[3] = scan_log_rows; // scan_mean_log_rows
        plan_features_24[4] = scan_log_rows; // scan_est_log_sum_rows
        plan_features_24[9] = 1.0; // join_count
        plan_features_24[14] = 1.0; // agg_count
        plan_features_24[21] = cbo_mem_log; // cbo_est_total_mem_log
        plan_features_24[22] = 14.0; // cbo_est_total_cpu_log
        plan_features_24[23] = (8.0_f64).ln_1p(); // env_dop_log

        CanonicalPlanFeatures {
            tables: ["0".to_string(), "0".to_string(), "101".to_string()],
            num_be_nodes: 3,
            avg_cpu_cores: 16,
            env_mem_log: 24,
            pipeline_dop: 8,
            segments: Vec::new(),
            plan_hash,
            plan_features_24,
        }
    }

    #[test]
    fn test_rls_calibrator_cold_start_and_collinear_stability() {
        let rls = OnlineRlsCalibrator::new();
        let plan = make_test_plan(0xABCD_1234, 14.0, 18.0);
        let cold = rls.predict(&plan);
        assert!(cold.log_mem_p50.is_finite());

        let target_mem = ((plan.plan_features_24[21] + 2.0).exp() as u64).max(10_000_000);
        for _ in 0..100 {
            rls.update(&plan, target_mem, 50_000_000, 5_000, 100_000_000);
        }
        let warm = rls.predict(&plan);
        assert!(warm.log_mem_p50.is_finite() && !warm.log_mem_p50.is_nan());
        let target_log = (target_mem as f64).ln_1p();
        assert!((warm.log_mem_p50 - target_log).abs() < 0.25);
    }

    #[test]
    fn test_rls_calibrator_zero_cbo_fallback_and_state_roundtrip() {
        let rls = OnlineRlsCalibrator::new();
        let zero_cbo_plan = make_test_plan(0x9999, 12.0, 0.0);
        let cold_zero = rls.predict(&zero_cbo_plan);
        assert!((cold_zero.log_mem_p50 - 18.0).abs() < 1e-9);
        assert!(cold_zero.log_mem_p90 > cold_zero.log_mem_p50);

        rls.update(&zero_cbo_plan, 64_000_000, 10_000_000, 1_000, 50_000_000);
        let state = rls.to_state();
        assert_eq!(state.update_count, 1);

        let restored = OnlineRlsCalibrator::from_state(state);
        let pred1 = rls.predict(&zero_cbo_plan);
        let pred2 = restored.predict(&zero_cbo_plan);
        assert!((pred1.log_mem_p50 - pred2.log_mem_p50).abs() < 1e-12);
    }
}
