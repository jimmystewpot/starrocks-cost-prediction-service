pub use crate::types::GbdtTrainingRow;
use crate::types::{DENSE_FEATURE_DIM, DenseFeatureVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub fn trainer_thread_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 4))
        .unwrap_or(2)
}

fn trainer_thread_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(trainer_thread_pool_size())
            .thread_name(|idx| format!("sr-gbdt-trainer-{idx}"))
            .build()
            .expect("failed to build GBDT Rayon thread pool")
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LossType {
    Huber { delta: f64 },
    Pinball { alpha: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GbdtHyperparams {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub learning_rate: f64,
    pub min_samples_leaf: usize,
}

impl Default for GbdtHyperparams {
    fn default() -> Self {
        Self {
            n_estimators: 80,
            max_depth: 5,
            learning_rate: 0.10,
            min_samples_leaf: 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TreeNode {
    pub is_leaf: bool,
    pub feature_idx: usize,
    pub threshold: f64,
    pub left_child: usize,
    pub right_child: usize,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoostedTreeRegressor {
    pub base_value: f64,
    pub learning_rate: f64,
    pub trees: Vec<Vec<TreeNode>>,
    pub loss: LossType,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GbdtPrediction {
    pub log_mem_p90: f64,
    pub log_mem_p50: f64,
    pub log_scan_bytes: f64,
    pub log_return_rows: f64,
    pub log_cpu_ns: f64,
    pub spill_prob: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GbdtModelSet {
    pub mem_p90: BoostedTreeRegressor,
    pub mem_p50: BoostedTreeRegressor,
    pub scan_bytes_p50: BoostedTreeRegressor,
    pub return_rows_p50: BoostedTreeRegressor,
    pub cpu_ns_p50: BoostedTreeRegressor,
    pub spill_prob: BoostedTreeRegressor,
}

pub fn pinball_loss(y_true: f64, y_pred: f64, alpha: f64) -> f64 {
    let residual = y_true - y_pred;
    if residual >= 0.0 {
        alpha * residual
    } else {
        (alpha - 1.0) * residual
    }
}

fn compute_quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let clamped_q = q.clamp(0.0, 1.0);
    let idx = (((sorted.len() - 1) as f64) * clamped_q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_leaf_value(indices: &[usize], true_residuals: &[f64], loss: LossType) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    match loss {
        LossType::Huber { .. } => {
            let sum: f64 = indices.iter().map(|&i| true_residuals[i]).sum();
            sum / (indices.len() as f64)
        }
        LossType::Pinball { alpha } => {
            let leaf_residuals: Vec<f64> = indices.iter().map(|&i| true_residuals[i]).collect();
            compute_quantile(&leaf_residuals, alpha)
        }
    }
}

struct TreeBuildContext<'a> {
    features: &'a [DenseFeatureVector],
    pseudo_residuals: &'a [f64],
    true_residuals: &'a [f64],
    loss: LossType,
    params: &'a GbdtHyperparams,
}

fn build_tree_node(
    nodes: &mut Vec<TreeNode>,
    indices: &[usize],
    depth: usize,
    ctx: &TreeBuildContext<'_>,
) -> usize {
    let node_idx = nodes.len();
    let leaf_value = compute_leaf_value(indices, ctx.true_residuals, ctx.loss);
    nodes.push(TreeNode {
        is_leaf: true,
        feature_idx: 0,
        threshold: 0.0,
        left_child: 0,
        right_child: 0,
        value: leaf_value,
    });

    let min_leaf = ctx.params.min_samples_leaf.max(1);
    if depth >= ctx.params.max_depth || indices.len() < 2 * min_leaf {
        return node_idx;
    }

    let n = indices.len();
    let total_g: f64 = indices.iter().map(|&i| ctx.pseudo_residuals[i]).sum();
    let parent_score = (total_g * total_g) / (n as f64);

    let mut best_gain = 1e-12_f64;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut found_split = false;

    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);

    for f in 0..DENSE_FEATURE_DIM {
        pairs.clear();
        for &i in indices {
            pairs.push((ctx.features[i][f], ctx.pseudo_residuals[i]));
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        if pairs[0].0 >= pairs[n - 1].0 {
            continue;
        }

        let mut left_sum = 0.0;
        for k in 0..(n - min_leaf) {
            left_sum += pairs[k].1;
            let left_count = k + 1;
            if left_count < min_leaf {
                continue;
            }
            if (pairs[k + 1].0 - pairs[k].0).abs() <= 1e-12 {
                continue;
            }
            let right_count = n - left_count;
            let right_sum = total_g - left_sum;
            let gain = (left_sum * left_sum) / (left_count as f64)
                + (right_sum * right_sum) / (right_count as f64)
                - parent_score;
            if gain > best_gain {
                best_gain = gain;
                best_feature = f;
                best_threshold = 0.5 * (pairs[k].0 + pairs[k + 1].0);
                found_split = true;
            }
        }
    }

    if !found_split {
        return node_idx;
    }

    let mut left_indices = Vec::with_capacity(n);
    let mut right_indices = Vec::with_capacity(n);
    for &i in indices {
        if ctx.features[i][best_feature] <= best_threshold {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }

    if left_indices.len() < min_leaf || right_indices.len() < min_leaf {
        return node_idx;
    }

    let left_child = build_tree_node(nodes, &left_indices, depth + 1, ctx);
    let right_child = build_tree_node(nodes, &right_indices, depth + 1, ctx);

    nodes[node_idx] = TreeNode {
        is_leaf: false,
        feature_idx: best_feature,
        threshold: best_threshold,
        left_child,
        right_child,
        value: leaf_value,
    };

    node_idx
}

fn predict_single_tree(nodes: &[TreeNode], x: &DenseFeatureVector) -> f64 {
    if nodes.is_empty() {
        return 0.0;
    }
    let mut idx = 0;
    while !nodes[idx].is_leaf {
        let node = &nodes[idx];
        if x[node.feature_idx] <= node.threshold {
            idx = node.left_child;
        } else {
            idx = node.right_child;
        }
    }
    nodes[idx].value
}

impl BoostedTreeRegressor {
    pub fn train(
        features: &[DenseFeatureVector],
        targets: &[f64],
        loss: LossType,
        params: &GbdtHyperparams,
    ) -> Self {
        let n = features.len().min(targets.len());
        if n == 0 {
            return Self {
                base_value: 0.0,
                learning_rate: params.learning_rate,
                trees: Vec::new(),
                loss,
            };
        }

        let targets = &targets[..n];
        let features = &features[..n];

        let base_value = match loss {
            LossType::Huber { .. } => compute_quantile(targets, 0.50),
            LossType::Pinball { alpha } => compute_quantile(targets, alpha),
        };

        let mut preds = vec![base_value; n];
        let mut true_residuals = vec![0.0_f64; n];
        let mut pseudo_residuals = vec![0.0_f64; n];
        let all_indices: Vec<usize> = (0..n).collect();
        let mut trees = Vec::with_capacity(params.n_estimators);

        for _ in 0..params.n_estimators {
            for i in 0..n {
                let r = targets[i] - preds[i];
                true_residuals[i] = r;
                pseudo_residuals[i] = match loss {
                    LossType::Huber { delta } => r.clamp(-delta, delta),
                    LossType::Pinball { alpha } => {
                        if r >= 0.0 {
                            alpha
                        } else {
                            alpha - 1.0
                        }
                    }
                };
            }

            let mut nodes = Vec::new();
            let ctx = TreeBuildContext {
                features,
                pseudo_residuals: &pseudo_residuals,
                true_residuals: &true_residuals,
                loss,
                params,
            };
            build_tree_node(&mut nodes, &all_indices, 0, &ctx);

            for (pred, feat) in preds.iter_mut().zip(features.iter()) {
                *pred += params.learning_rate * predict_single_tree(&nodes, feat);
            }

            trees.push(nodes);
        }

        Self {
            base_value,
            learning_rate: params.learning_rate,
            trees,
            loss,
        }
    }

    pub fn predict(&self, x: &DenseFeatureVector) -> f64 {
        let mut sum = self.base_value;
        for tree in &self.trees {
            sum += self.learning_rate * predict_single_tree(tree, x);
        }
        sum
    }
}

impl GbdtModelSet {
    pub fn train_parallel(rows: &[GbdtTrainingRow], params: &GbdtHyperparams) -> Self {
        let features: Vec<DenseFeatureVector> = rows.iter().map(|r| r.features).collect();
        let log_mem: Vec<f64> = rows.iter().map(|r| r.log_mem).collect();
        let log_scan_bytes: Vec<f64> = rows.iter().map(|r| r.log_scan_bytes).collect();
        let log_return_rows: Vec<f64> = rows.iter().map(|r| r.log_return_rows).collect();
        let log_cpu_ns: Vec<f64> = rows.iter().map(|r| r.log_cpu_ns).collect();
        let spilled: Vec<f64> = rows.iter().map(|r| r.spilled).collect();

        let specs: [(&[f64], LossType); 6] = [
            (&log_mem, LossType::Pinball { alpha: 0.90 }),
            (&log_mem, LossType::Huber { delta: 1.0 }),
            (&log_scan_bytes, LossType::Huber { delta: 1.0 }),
            (&log_return_rows, LossType::Huber { delta: 1.0 }),
            (&log_cpu_ns, LossType::Huber { delta: 1.0 }),
            (&spilled, LossType::Huber { delta: 0.5 }),
        ];

        let mut models: Vec<BoostedTreeRegressor> = trainer_thread_pool().install(|| {
            specs
                .into_par_iter()
                .map(|(targets, loss)| {
                    BoostedTreeRegressor::train(&features, targets, loss, params)
                })
                .collect()
        });

        let spill_prob = models.pop().expect("spill_prob model");
        let cpu_ns_p50 = models.pop().expect("cpu_ns_p50 model");
        let return_rows_p50 = models.pop().expect("return_rows_p50 model");
        let scan_bytes_p50 = models.pop().expect("scan_bytes_p50 model");
        let mem_p50 = models.pop().expect("mem_p50 model");
        let mem_p90 = models.pop().expect("mem_p90 model");

        Self {
            mem_p90,
            mem_p50,
            scan_bytes_p50,
            return_rows_p50,
            cpu_ns_p50,
            spill_prob,
        }
    }

    pub fn predict_mem_p90(&self, x: &DenseFeatureVector) -> f64 {
        self.mem_p90.predict(x)
    }

    pub fn predict_all(&self, x: &DenseFeatureVector) -> GbdtPrediction {
        let log_mem_p50 = self.mem_p50.predict(x);
        let log_mem_p90 = self.mem_p90.predict(x).max(log_mem_p50);
        GbdtPrediction {
            log_mem_p90,
            log_mem_p50,
            log_scan_bytes: self.scan_bytes_p50.predict(x),
            log_return_rows: self.return_rows_p50.predict(x),
            log_cpu_ns: self.cpu_ns_p50.predict(x),
            spill_prob: self.spill_prob.predict(x).clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DENSE_FEATURE_DIM;

    #[test]
    fn test_gbdt_learns_nonlinear_interaction_and_p90_exceeds_p50() {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for i in 0..160 {
            let mut x = [0.0_f64; DENSE_FEATURE_DIM];
            let scans = (i % 10) as f64;
            let is_bcast_join = if i % 2 == 0 { 1.0 } else { 0.0 };
            x[3] = scans;
            x[11] = is_bcast_join;
            let noise = ((i % 7) as f64 - 3.0) * 0.4;
            let y = 14.0 + 0.5 * scans + 2.5 * is_bcast_join + noise;
            features.push(x);
            targets.push(y);
        }
        let params = GbdtHyperparams {
            n_estimators: 60,
            max_depth: 4,
            learning_rate: 0.1,
            min_samples_leaf: 4,
        };
        let model_p50 = BoostedTreeRegressor::train(
            &features,
            &targets,
            LossType::Huber { delta: 1.0 },
            &params,
        );
        let model_p90 = BoostedTreeRegressor::train(
            &features,
            &targets,
            LossType::Pinball { alpha: 0.90 },
            &params,
        );

        let mut test_x = [0.0_f64; DENSE_FEATURE_DIM];
        test_x[3] = 6.0;
        test_x[11] = 1.0; // expected true center = 14.0 + 3.0 + 2.5 = 19.5
        let p50 = model_p50.predict(&test_x);
        let p90 = model_p90.predict(&test_x);
        assert!((p50 - 19.5).abs() < 0.5, "p50={} should be near 19.5", p50);
        assert!(p90 > p50 + 0.3, "p90={} must exceed p50={}", p90, p50);
    }

    #[test]
    fn test_gbdt_model_set_parallel_training_and_pinball_loss() {
        let mut rows = Vec::new();
        for i in 0..80 {
            let mut features = [0.0_f64; DENSE_FEATURE_DIM];
            features[0] = (i % 8) as f64;
            features[1] = if i % 2 == 0 { 1.0 } else { 0.0 };
            let noise = ((i % 5) as f64 - 2.0) * 0.25;
            rows.push(GbdtTrainingRow {
                features,
                log_mem: 16.0 + 0.4 * features[0] + 1.5 * features[1] + noise,
                log_scan_bytes: 18.0 + 0.5 * features[0],
                log_return_rows: 8.0 + 0.2 * features[0],
                log_cpu_ns: 20.0 + 0.3 * features[0] + features[1],
                spilled: features[1],
            });
        }

        let params = GbdtHyperparams::default();
        assert_eq!(params.n_estimators, 80);
        assert_eq!(params.max_depth, 5);
        assert!((params.learning_rate - 0.10).abs() < 1e-12);
        assert_eq!(params.min_samples_leaf, 4);

        let model_set = GbdtModelSet::train_parallel(&rows, &params);
        let mut x = [0.0_f64; DENSE_FEATURE_DIM];
        x[0] = 4.0;
        x[1] = 1.0;
        let pred = model_set.predict_all(&x);
        assert!(pred.log_mem_p90 >= pred.log_mem_p50);
        assert!((pred.log_mem_p90 - model_set.predict_mem_p90(&x)).abs() < 1e-9);
        assert!((0.0..=1.0).contains(&pred.spill_prob));
        assert!(pred.spill_prob > 0.7);

        // Verify pinball loss asymmetry at alpha = 0.90
        let under_loss = pinball_loss(20.0, 18.0, 0.90); // underprediction by 2.0 -> 1.8
        let over_loss = pinball_loss(20.0, 22.0, 0.90); // overprediction by 2.0 -> 0.2
        assert!((under_loss - 1.8).abs() < 1e-12);
        assert!((over_loss - 0.2).abs() < 1e-12);

        // Verify isolated Rayon trainer thread pool bounds and thread naming
        let pool_size = trainer_thread_pool_size();
        assert!((1..=4).contains(&pool_size));
        let thread_name = trainer_thread_pool().install(|| {
            std::thread::current()
                .name()
                .unwrap_or_default()
                .to_string()
        });
        assert!(
            thread_name.starts_with("sr-gbdt-trainer-"),
            "unexpected thread name: {thread_name}"
        );
    }
}
