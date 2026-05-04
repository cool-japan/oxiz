//! Integration tests for `predictor::persistence`.

use oxiz_smtcomp::benchmark::BenchmarkStatus;
use oxiz_smtcomp::predictor::{
    Dataset, DifficultyModel, Features, KnnRegressor, LinearRegressor, RegressionTree, Sample,
    TrainingConfig, load_from_file, save_to_file,
};
use rand::SeedableRng;
use std::env;

fn make_tiny_dataset() -> Dataset {
    let mut ds = Dataset::new();
    for i in 0..8 {
        ds.push(Sample {
            features: Features { atom_count: i as f64 * 5.0, ..Default::default() },
            runtime_seconds: 0.1 * (i + 1) as f64,
            status: BenchmarkStatus::Sat,
        });
    }
    ds
}

fn predictions(model: &dyn DifficultyModel, ds: &Dataset) -> Vec<f64> {
    ds.samples
        .iter()
        .map(|s| model.predict_runtime(&s.features))
        .collect()
}

#[test]
fn test_persistence_round_trip_linear() {
    let ds = make_tiny_dataset();
    let mut model = LinearRegressor::new(1e-3);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    model.fit(&ds, &config, &mut rng);

    let mut path = env::temp_dir();
    path.push("oxiz_persist_linear.json");
    save_to_file(&model, &path).expect("save_to_file failed");
    let loaded = load_from_file(&path).expect("load_from_file failed");

    let p1 = predictions(&model, &ds);
    let p2 = predictions(loaded.as_ref(), &ds);
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-9, "Prediction mismatch: {a} vs {b}");
    }
}

#[test]
fn test_persistence_round_trip_knn() {
    let ds = make_tiny_dataset();
    let mut model = KnnRegressor::new(3);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    model.fit(&ds, &config, &mut rng);

    let mut path = env::temp_dir();
    path.push("oxiz_persist_knn.json");
    save_to_file(&model, &path).expect("save_to_file failed");
    let loaded = load_from_file(&path).expect("load_from_file failed");

    let p1 = predictions(&model, &ds);
    let p2 = predictions(loaded.as_ref(), &ds);
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-9, "Prediction mismatch: {a} vs {b}");
    }
}

#[test]
fn test_persistence_round_trip_tree() {
    let ds = make_tiny_dataset();
    let mut model = RegressionTree::new(4, 2);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    model.fit(&ds, &config, &mut rng);

    let mut path = env::temp_dir();
    path.push("oxiz_persist_tree.json");
    save_to_file(&model, &path).expect("save_to_file failed");
    let loaded = load_from_file(&path).expect("load_from_file failed");

    let p1 = predictions(&model, &ds);
    let p2 = predictions(loaded.as_ref(), &ds);
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-9, "Prediction mismatch: {a} vs {b}");
    }
}

#[test]
fn test_persistence_rejects_unknown_kind() {
    // Write an envelope with an unknown kind
    let json = r#"{
        "oxiz_predictor_version": "0.2.2",
        "kind": "nonexistent_model",
        "payload": {}
    }"#;
    let mut path = env::temp_dir();
    path.push("oxiz_persist_unknown_kind.json");
    std::fs::write(&path, json).expect("write failed");

    let result = load_from_file(&path);
    let err = result.err().expect("should have failed for unknown kind");
    assert!(
        err.to_string().contains("unknown predictor kind"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_persistence_rejects_version_mismatch() {
    let json = r#"{
        "oxiz_predictor_version": "0.0.0",
        "kind": "linear",
        "payload": {}
    }"#;
    let mut path = env::temp_dir();
    path.push("oxiz_persist_version_mismatch.json");
    std::fs::write(&path, json).expect("write failed");

    let result = load_from_file(&path);
    let err = result.err().expect("should have failed for version mismatch");
    assert!(
        err.to_string().contains("version mismatch"),
        "Unexpected error: {err}"
    );
}
