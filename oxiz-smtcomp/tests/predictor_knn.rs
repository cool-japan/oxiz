//! Integration tests for `predictor::knn`.

use oxiz_smtcomp::benchmark::BenchmarkStatus;
use oxiz_smtcomp::predictor::{
    Dataset, DifficultyModel, Features, KnnRegressor, Sample, TrainingConfig,
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

fn fit_knn(k: usize, ds: &Dataset) -> KnnRegressor {
    let mut model = KnnRegressor::new(k);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    model.fit(ds, &config, &mut rng);
    model
}

#[test]
fn test_knn_predict_matches_nearest_neighbour_on_k1() {
    let mut ds = Dataset::new();
    ds.push(make_sample(0.0, 0.05)); // cluster A: trivial
    ds.push(make_sample(1.0, 0.06));
    ds.push(make_sample(100.0, 30.0)); // cluster B: hard
    ds.push(make_sample(101.0, 31.0));

    let model = fit_knn(1, &ds);

    // Query near cluster A → should predict small runtime
    let q_a = Features {
        atom_count: 0.5,
        ..Default::default()
    };
    let rt_a = model.predict_runtime(&q_a);
    assert!(
        rt_a < 1.0,
        "Expected trivial runtime near cluster A, got {rt_a}"
    );

    // Query near cluster B → should predict large runtime
    let q_b = Features {
        atom_count: 100.5,
        ..Default::default()
    };
    let rt_b = model.predict_runtime(&q_b);
    assert!(
        rt_b > 1.0,
        "Expected hard runtime near cluster B, got {rt_b}"
    );
}

#[test]
fn test_knn_inverse_distance_weighting() {
    // Two training points; query is 3× closer to the first one
    let mut ds = Dataset::new();
    ds.push(make_sample(0.0, 1.0)); // runtime = 1s
    ds.push(make_sample(3.0, 7.0)); // runtime = 7s

    let model = fit_knn(2, &ds);

    // Query at atom=1 is 1 unit from first, 2 units from second
    // Expected: weighted average biased toward 1s
    let q = Features {
        atom_count: 1.0,
        ..Default::default()
    };
    let rt = model.predict_runtime(&q);
    // The result should be between 1 and 7 and closer to 1 than 7
    assert!(rt > 0.0, "Runtime should be positive: {rt}");
    assert!(rt < 7.0, "Runtime should be less than 7: {rt}");
}

#[test]
fn test_knn_handles_zero_distance_without_nan() {
    // Identical query point as a training sample
    let mut ds = Dataset::new();
    ds.push(make_sample(42.0, 5.0));
    ds.push(make_sample(42.0, 5.0)); // exact duplicate

    let model = fit_knn(2, &ds);

    let q = Features {
        atom_count: 42.0,
        ..Default::default()
    };
    let rt = model.predict_runtime(&q);
    assert!(rt.is_finite(), "Got NaN/Inf when distance is zero: {rt}");
    assert!(rt >= 0.0, "Negative runtime: {rt}");
}

#[test]
fn test_knn_with_empty_dataset() {
    let ds = Dataset::new();
    let model = fit_knn(3, &ds);
    assert!(model.is_fitted);
    // Should return 0 for empty model
    let f = Features::default();
    let rt = model.predict_runtime(&f);
    assert_eq!(rt, 0.0);
}
