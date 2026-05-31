//! Integration tests for `predictor::tree`.

use oxiz_smtcomp::benchmark::BenchmarkStatus;
use oxiz_smtcomp::predictor::tree::TreeNode;
use oxiz_smtcomp::predictor::{
    Dataset, DifficultyModel, Features, RegressionTree, Sample, TrainingConfig,
};
use rand::SeedableRng;

fn make_sample(atom: f64, rt: f64) -> Sample {
    Sample {
        features: Features {
            atom_count: atom,
            ..Default::default()
        },
        runtime_seconds: rt,
        status: BenchmarkStatus::Sat,
    }
}

fn fit_tree(max_depth: usize, min_split: usize, ds: &Dataset) -> RegressionTree {
    let mut model = RegressionTree::new(max_depth, min_split);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    model.fit(ds, &config, &mut rng);
    model
}

#[test]
fn test_tree_fits_pure_split() {
    // Two perfectly separable clusters
    let mut ds = Dataset::new();
    for _ in 0..10 {
        ds.push(make_sample(0.0, 0.05)); // trivial
        ds.push(make_sample(100.0, 30.0)); // hard
    }
    let model = fit_tree(4, 2, &ds);
    assert!(model.is_fitted);
    assert!(model.root.is_some());

    let q_trivial = Features {
        atom_count: 0.0,
        ..Default::default()
    };
    let rt_trivial = model.predict_runtime(&q_trivial);
    assert!(
        rt_trivial < 1.0,
        "Expected trivial prediction, got {rt_trivial}"
    );

    let q_hard = Features {
        atom_count: 100.0,
        ..Default::default()
    };
    let rt_hard = model.predict_runtime(&q_hard);
    assert!(rt_hard > 1.0, "Expected hard prediction, got {rt_hard}");
}

#[test]
fn test_tree_max_depth_respected() {
    let mut ds = Dataset::new();
    for i in 0..30 {
        ds.push(make_sample(i as f64, i as f64 * 0.5));
    }
    let max_d = 3usize;
    let model = fit_tree(max_d, 2, &ds);

    if let Some(ref root) = model.root {
        let actual_depth = RegressionTree::tree_depth(root);
        assert!(
            actual_depth <= max_d,
            "Tree depth {actual_depth} exceeds max {max_d}"
        );
    }
}

#[test]
fn test_tree_min_samples_split_respected() {
    // 3 samples, min_split = 4 → root must be a leaf
    let mut ds = Dataset::new();
    ds.push(make_sample(0.0, 0.05));
    ds.push(make_sample(50.0, 5.0));
    ds.push(make_sample(100.0, 30.0));

    let model = fit_tree(10, 4, &ds);
    assert!(model.is_fitted);
    assert!(
        matches!(model.root, Some(TreeNode::Leaf { .. })),
        "Expected root to be a Leaf when n < min_samples_split"
    );
}

#[test]
fn test_tree_predictions_are_non_negative() {
    let mut ds = Dataset::new();
    for i in 0..20 {
        ds.push(make_sample(i as f64 * 5.0, 0.1 + i as f64 * 0.5));
    }
    let model = fit_tree(5, 2, &ds);
    for s in &ds.samples {
        let rt = model.predict_runtime(&s.features);
        assert!(rt >= 0.0, "Negative prediction: {rt}");
        assert!(rt.is_finite(), "Non-finite prediction: {rt}");
    }
}

#[test]
fn test_tree_empty_dataset() {
    let ds = Dataset::new();
    let model = fit_tree(5, 2, &ds);
    assert!(model.is_fitted);
    assert!(model.root.is_none());
    let f = Features::default();
    assert_eq!(model.predict_runtime(&f), 0.0);
}
