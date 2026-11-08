use crate::model::ensemble::HoldoutMetrics;
use crate::model::gbdt::{GbdtHyperparams, GbdtModelSet, pinball_loss};
use crate::types::GbdtTrainingRow;

pub fn split_train_validation(
    rows: &[GbdtTrainingRow],
) -> (Vec<GbdtTrainingRow>, Vec<GbdtTrainingRow>) {
    if rows.len() < 5 {
        return (rows.to_vec(), rows.to_vec());
    }
    let mut train = Vec::with_capacity(rows.len() * 4 / 5 + 1);
    let mut val = Vec::with_capacity(rows.len() / 5 + 1);
    for (i, row) in rows.iter().enumerate() {
        if i % 5 == 4 {
            val.push(row.clone());
        } else {
            train.push(row.clone());
        }
    }
    (train, val)
}

pub fn evaluate_model_set(models: &GbdtModelSet, val_rows: &[GbdtTrainingRow]) -> HoldoutMetrics {
    if val_rows.is_empty() {
        return HoldoutMetrics::default();
    }

    let mut total_pinball = 0.0_f64;
    let mut q_errors = Vec::with_capacity(val_rows.len());

    for row in val_rows {
        let pred = models.predict_all(&row.features);
        let loss_p90 = pinball_loss(row.log_mem, pred.log_mem_p90, 0.90);
        if loss_p90.is_finite() {
            total_pinball += loss_p90;
        }
        let log_q = (pred.log_mem_p50 - row.log_mem).abs().clamp(0.0, 20.0);
        q_errors.push(log_q.exp());
    }

    q_errors.sort_by(f64::total_cmp);
    let n = q_errors.len();
    let idx_p50 = (((n - 1) as f64) * 0.50).round() as usize;
    let idx_p90 = (((n - 1) as f64) * 0.90).round() as usize;

    HoldoutMetrics {
        pinball_p90: total_pinball / (n as f64),
        q_error_p50: q_errors[idx_p50.min(n - 1)],
        q_error_p90: q_errors[idx_p90.min(n - 1)],
        validation_samples: n,
    }
}

pub fn should_promote_challenger(
    champion: Option<&HoldoutMetrics>,
    challenger: &HoldoutMetrics,
) -> bool {
    if challenger.validation_samples == 0 || !challenger.pinball_p90.is_finite() {
        return false;
    }
    match champion {
        None => true,
        Some(champ) => challenger.pinball_p90 <= champ.pinball_p90 * 1.05,
    }
}

pub fn train_and_evaluate_challenger(
    rows: &[GbdtTrainingRow],
    params: &GbdtHyperparams,
) -> Option<(GbdtModelSet, HoldoutMetrics)> {
    if rows.len() < 5 {
        return None;
    }
    let (train_rows, val_rows) = split_train_validation(rows);
    let models = GbdtModelSet::train_parallel(&train_rows, params);
    let metrics = evaluate_model_set(&models, &val_rows);
    Some((models, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DENSE_FEATURE_DIM;

    #[test]
    fn test_trainer_split_evaluation_and_champion_challenger_gate() {
        let mut rows = Vec::new();
        for i in 0..40 {
            let mut features = [0.0_f64; DENSE_FEATURE_DIM];
            features[0] = (i % 8) as f64;
            features[1] = if i % 2 == 0 { 1.0 } else { 0.0 };
            rows.push(GbdtTrainingRow {
                features,
                log_mem: 16.0 + 0.5 * features[0] + features[1],
                log_scan_bytes: 15.0 + 0.3 * features[0],
                log_return_rows: 8.0 + 0.2 * features[0],
                log_cpu_ns: 19.0 + 0.4 * features[0],
                spilled: 0.0,
            });
        }

        let (train, val) = split_train_validation(&rows);
        assert_eq!(train.len() + val.len(), 40);
        assert_eq!(val.len(), 8);

        let params = GbdtHyperparams::default();
        let (_models, metrics) =
            train_and_evaluate_challenger(&rows, &params).expect("must train challenger");
        assert_eq!(metrics.validation_samples, 8);
        assert!(metrics.pinball_p90.is_finite() && metrics.pinball_p90 >= 0.0);
        assert!(metrics.q_error_p50 >= 1.0);
        assert!(metrics.q_error_p90 >= metrics.q_error_p50);

        // No champion -> always promote
        assert!(should_promote_challenger(None, &metrics));

        // Challenger within 5% of champion -> promote
        let champ = HoldoutMetrics {
            pinball_p90: 0.50,
            q_error_p50: 1.2,
            q_error_p90: 1.8,
            validation_samples: 8,
        };
        let slightly_worse = HoldoutMetrics {
            pinball_p90: 0.52,
            ..champ.clone()
        };
        let much_worse = HoldoutMetrics {
            pinball_p90: 0.60,
            ..champ.clone()
        };
        assert!(should_promote_challenger(Some(&champ), &slightly_worse));
        assert!(!should_promote_challenger(Some(&champ), &much_worse));

        // Edge cases: < 5 rows in split, empty validation rows, 0 validation samples in gate
        let (short_train, short_val) = split_train_validation(&rows[..3]);
        assert_eq!(short_train.len(), 3);
        assert_eq!(short_val.len(), 3);
        let empty_eval = evaluate_model_set(&_models, &[]);
        assert_eq!(empty_eval.validation_samples, 0);
        assert!(!should_promote_challenger(None, &empty_eval));
    }
}
