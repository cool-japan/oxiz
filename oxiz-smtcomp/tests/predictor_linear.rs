//! Integration tests for `predictor::linear`.

use oxiz_smtcomp::benchmark::BenchmarkStatus;
use oxiz_smtcomp::predictor::{
    Dataset, DifficultyClass, DifficultyModel, Features, LinearRegressor, Sample, TrainingConfig,
    load_from_file, save_to_file,
};
use rand::SeedableRng;
use std::env;

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

fn make_linear_dataset(n: usize) -> Dataset {
    let mut ds = Dataset::new();
    for i in 0..n {
        let atom = i as f64 * 2.0;
        // Roughly: runtime grows with atom count
        let rt = (0.01 * atom).exp_m1().max(0.001);
        ds.push(make_sample(atom, rt));
    }
    ds
}

#[test]
fn test_linear_fits_synthetic_linear_data() {
    let ds = make_linear_dataset(50);
    let mut model = LinearRegressor::new(1e-3);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let report = model.fit(&ds, &config, &mut rng);
    assert!(model.is_fitted);
    assert!(report.epochs > 0);
    assert!(report.mae_seconds.is_finite());
    assert!(report.mae_seconds >= 0.0);
    // Final loss should be non-negative
    assert!(report.final_loss >= 0.0);
}

#[test]
fn test_linear_predicts_class_aligns_with_runtime() {
    let ds = make_linear_dataset(50);
    let mut model = LinearRegressor::new(1e-3);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    model.fit(&ds, &config, &mut rng);

    // Predictions should be non-negative
    for sample in &ds.samples {
        let rt = model.predict_runtime(&sample.features);
        let class = model.predict_class(&sample.features);
        assert!(rt >= 0.0, "Negative predicted runtime: {rt}");
        assert_eq!(class, DifficultyClass::from_runtime_seconds(rt));
    }
}

#[test]
fn test_linear_save_load_preserves_predictions() {
    let ds = make_linear_dataset(30);
    let mut model = LinearRegressor::new(1e-3);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    model.fit(&ds, &config, &mut rng);

    let mut path = env::temp_dir();
    path.push("oxiz_predictor_linear_test.json");
    save_to_file(&model, &path).expect("save failed");

    let loaded = load_from_file(&path).expect("load failed");

    // Predictions must be bit-identical after round-trip
    for sample in &ds.samples {
        let p1 = model.predict_runtime(&sample.features);
        let p2 = loaded.predict_runtime(&sample.features);
        assert!(
            (p1 - p2).abs() < 1e-9,
            "Prediction mismatch after round-trip: {p1} vs {p2}"
        );
    }
}

#[test]
fn test_linear_empty_dataset_is_handled() {
    let ds = Dataset::new();
    let mut model = LinearRegressor::new(1e-3);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let report = model.fit(&ds, &config, &mut rng);
    assert!(model.is_fitted);
    assert_eq!(report.epochs, 0);
    // Should still predict something finite
    let f = Features::default();
    assert!(model.predict_runtime(&f).is_finite());
}
