//! ML-based difficulty prediction for SMT benchmarks.
//!
//! This module provides three regression models — [`LinearRegressor`],
//! [`KnnRegressor`], and [`RegressionTree`] — all sharing the [`DifficultyModel`]
//! trait.  Models predict expected solver runtime (seconds) and difficulty
//! class from a [`Features`] vector derived from a benchmark's metadata.
//!
//! ## Quick start
//!
//! ```no_run
//! use oxiz_smtcomp::predictor::{default_model, DifficultyModel, Features};
//!
//! let model = default_model();
//! let features = Features::default();
//! let runtime = model.predict_runtime(&features);
//! let class   = model.predict_class(&features);
//! println!("Predicted {runtime:.2}s — class: {class}");
//! ```

pub mod class;
pub mod dataset;
pub mod features;
pub mod knn;
pub mod linear;
pub mod models;
pub mod persistence;
pub mod report;
pub mod tree;

pub use class::DifficultyClass;
pub use dataset::{Dataset, Sample};
pub use features::{FEATURE_DIM, FeatureNormalizer, Features};
pub use knn::KnnRegressor;
pub use linear::LinearRegressor;
pub use models::DifficultyModel;
pub use persistence::{load_from_file, save_to_file};
pub use report::{PredictorStats, TrainingConfig, TrainingReport};
pub use tree::RegressionTree;

use rand::SeedableRng;

/// Returns a pre-trained [`LinearRegressor`] from a small synthetic seed dataset.
///
/// The model is ready to use immediately without any historical data.  Its
/// accuracy is modest because the training corpus is synthetic, but it provides
/// a reasonable cold-start predictor.  Fine-tune by calling `fit()` on real
/// [`crate::benchmark::SingleResult`] data wrapped in a [`Dataset`].
#[must_use]
pub fn default_model() -> LinearRegressor {
    let dataset = synthetic_seed_dataset();
    let mut model = LinearRegressor::new(1e-3);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFE_BABE);
    model.fit(&dataset, &config, &mut rng);
    model
}

/// Train a fresh [`LinearRegressor`] on the given dataset.
///
/// Uses a fixed random seed for reproducibility.  Returns a fitted model.
#[must_use]
pub fn train_model(dataset: &Dataset) -> LinearRegressor {
    let mut model = LinearRegressor::new(1e-3);
    let config = TrainingConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x0000_1234);
    model.fit(dataset, &config, &mut rng);
    model
}

/// Build a small synthetic [`Dataset`] that seeds the cold-start predictor.
///
/// Covers a range of logics, structural profiles, and runtimes spanning all
/// five difficulty classes.  No file I/O is performed — all `BenchmarkMeta`
/// objects are constructed in memory.
fn synthetic_seed_dataset() -> Dataset {
    use crate::benchmark::BenchmarkStatus;
    use crate::loader::BenchmarkMeta;
    use crate::logic_detector::StructuralFeatures;
    use std::path::PathBuf;

    struct SyntheticEntry {
        logic: &'static str,
        file_size: u64,
        atom_count: u32,
        clause_count: u32,
        max_term_depth: u32,
        max_quantifier_nesting: u32,
        max_ite_depth: u32,
        max_let_depth: u32,
        bv_width: u32,
        runtime_seconds: f64,
    }

    let entries = [
        // QF_LIA — trivial
        SyntheticEntry {
            logic: "QF_LIA",
            file_size: 200,
            atom_count: 2,
            clause_count: 1,
            max_term_depth: 3,
            max_quantifier_nesting: 0,
            max_ite_depth: 0,
            max_let_depth: 0,
            bv_width: 0,
            runtime_seconds: 0.03,
        },
        SyntheticEntry {
            logic: "QF_LIA",
            file_size: 400,
            atom_count: 4,
            clause_count: 2,
            max_term_depth: 4,
            max_quantifier_nesting: 0,
            max_ite_depth: 0,
            max_let_depth: 0,
            bv_width: 0,
            runtime_seconds: 0.05,
        },
        // QF_LIA — easy
        SyntheticEntry {
            logic: "QF_LIA",
            file_size: 2000,
            atom_count: 20,
            clause_count: 10,
            max_term_depth: 8,
            max_quantifier_nesting: 0,
            max_ite_depth: 1,
            max_let_depth: 0,
            bv_width: 0,
            runtime_seconds: 0.3,
        },
        SyntheticEntry {
            logic: "QF_LIA",
            file_size: 5000,
            atom_count: 40,
            clause_count: 20,
            max_term_depth: 10,
            max_quantifier_nesting: 0,
            max_ite_depth: 2,
            max_let_depth: 1,
            bv_width: 0,
            runtime_seconds: 0.7,
        },
        // QF_LIA — medium
        SyntheticEntry {
            logic: "QF_LIA",
            file_size: 20000,
            atom_count: 200,
            clause_count: 80,
            max_term_depth: 15,
            max_quantifier_nesting: 0,
            max_ite_depth: 3,
            max_let_depth: 2,
            bv_width: 0,
            runtime_seconds: 3.0,
        },
        SyntheticEntry {
            logic: "QF_LIA",
            file_size: 50000,
            atom_count: 500,
            clause_count: 200,
            max_term_depth: 20,
            max_quantifier_nesting: 0,
            max_ite_depth: 5,
            max_let_depth: 3,
            bv_width: 0,
            runtime_seconds: 8.0,
        },
        // QF_BV — trivial/easy
        SyntheticEntry {
            logic: "QF_BV",
            file_size: 300,
            atom_count: 3,
            clause_count: 1,
            max_term_depth: 4,
            max_quantifier_nesting: 0,
            max_ite_depth: 0,
            max_let_depth: 0,
            bv_width: 32,
            runtime_seconds: 0.04,
        },
        SyntheticEntry {
            logic: "QF_BV",
            file_size: 3000,
            atom_count: 30,
            clause_count: 12,
            max_term_depth: 10,
            max_quantifier_nesting: 0,
            max_ite_depth: 1,
            max_let_depth: 0,
            bv_width: 64,
            runtime_seconds: 0.5,
        },
        // QF_BV — hard
        SyntheticEntry {
            logic: "QF_BV",
            file_size: 100000,
            atom_count: 1000,
            clause_count: 400,
            max_term_depth: 30,
            max_quantifier_nesting: 0,
            max_ite_depth: 8,
            max_let_depth: 4,
            bv_width: 256,
            runtime_seconds: 25.0,
        },
        SyntheticEntry {
            logic: "QF_BV",
            file_size: 200000,
            atom_count: 2000,
            clause_count: 800,
            max_term_depth: 40,
            max_quantifier_nesting: 0,
            max_ite_depth: 10,
            max_let_depth: 6,
            bv_width: 512,
            runtime_seconds: 50.0,
        },
        // QF_FP — medium/hard
        SyntheticEntry {
            logic: "QF_FP",
            file_size: 10000,
            atom_count: 100,
            clause_count: 40,
            max_term_depth: 15,
            max_quantifier_nesting: 0,
            max_ite_depth: 2,
            max_let_depth: 1,
            bv_width: 64,
            runtime_seconds: 5.0,
        },
        SyntheticEntry {
            logic: "QF_FP",
            file_size: 80000,
            atom_count: 800,
            clause_count: 300,
            max_term_depth: 25,
            max_quantifier_nesting: 0,
            max_ite_depth: 6,
            max_let_depth: 3,
            bv_width: 64,
            runtime_seconds: 40.0,
        },
        // QF_S (strings) — easy/medium
        SyntheticEntry {
            logic: "QF_S",
            file_size: 1500,
            atom_count: 15,
            clause_count: 6,
            max_term_depth: 7,
            max_quantifier_nesting: 0,
            max_ite_depth: 1,
            max_let_depth: 0,
            bv_width: 0,
            runtime_seconds: 0.6,
        },
        SyntheticEntry {
            logic: "QF_S",
            file_size: 15000,
            atom_count: 150,
            clause_count: 60,
            max_term_depth: 18,
            max_quantifier_nesting: 0,
            max_ite_depth: 3,
            max_let_depth: 2,
            bv_width: 0,
            runtime_seconds: 6.0,
        },
        // QF_DT (datatypes) — medium/hard
        SyntheticEntry {
            logic: "QF_DT",
            file_size: 8000,
            atom_count: 80,
            clause_count: 30,
            max_term_depth: 12,
            max_quantifier_nesting: 0,
            max_ite_depth: 2,
            max_let_depth: 2,
            bv_width: 0,
            runtime_seconds: 4.0,
        },
        SyntheticEntry {
            logic: "QF_DT",
            file_size: 60000,
            atom_count: 600,
            clause_count: 250,
            max_term_depth: 22,
            max_quantifier_nesting: 0,
            max_ite_depth: 5,
            max_let_depth: 4,
            bv_width: 0,
            runtime_seconds: 35.0,
        },
        // QF_AUFBV — hard/very hard
        SyntheticEntry {
            logic: "QF_AUFBV",
            file_size: 150000,
            atom_count: 1500,
            clause_count: 600,
            max_term_depth: 35,
            max_quantifier_nesting: 0,
            max_ite_depth: 12,
            max_let_depth: 5,
            bv_width: 128,
            runtime_seconds: 45.0,
        },
        SyntheticEntry {
            logic: "QF_AUFBV",
            file_size: 500000,
            atom_count: 5000,
            clause_count: 2000,
            max_term_depth: 60,
            max_quantifier_nesting: 0,
            max_ite_depth: 20,
            max_let_depth: 10,
            bv_width: 256,
            runtime_seconds: 120.0,
        },
        // LIA (with quantifiers) — medium/very hard
        SyntheticEntry {
            logic: "LIA",
            file_size: 12000,
            atom_count: 120,
            clause_count: 50,
            max_term_depth: 18,
            max_quantifier_nesting: 2,
            max_ite_depth: 2,
            max_let_depth: 1,
            bv_width: 0,
            runtime_seconds: 7.0,
        },
        SyntheticEntry {
            logic: "LIA",
            file_size: 250000,
            atom_count: 2500,
            clause_count: 1000,
            max_term_depth: 50,
            max_quantifier_nesting: 5,
            max_ite_depth: 8,
            max_let_depth: 6,
            bv_width: 0,
            runtime_seconds: 180.0,
        },
        // UFLIA — very hard
        SyntheticEntry {
            logic: "UFLIA",
            file_size: 300000,
            atom_count: 3000,
            clause_count: 1200,
            max_term_depth: 55,
            max_quantifier_nesting: 4,
            max_ite_depth: 10,
            max_let_depth: 8,
            bv_width: 0,
            runtime_seconds: 90.0,
        },
        // NIA — very hard
        SyntheticEntry {
            logic: "NIA",
            file_size: 200000,
            atom_count: 2000,
            clause_count: 800,
            max_term_depth: 45,
            max_quantifier_nesting: 3,
            max_ite_depth: 7,
            max_let_depth: 5,
            bv_width: 0,
            runtime_seconds: 150.0,
        },
        // QF_NIA — hard
        SyntheticEntry {
            logic: "QF_NIA",
            file_size: 80000,
            atom_count: 800,
            clause_count: 300,
            max_term_depth: 28,
            max_quantifier_nesting: 0,
            max_ite_depth: 6,
            max_let_depth: 3,
            bv_width: 0,
            runtime_seconds: 55.0,
        },
        // QF_LRA — medium
        SyntheticEntry {
            logic: "QF_LRA",
            file_size: 25000,
            atom_count: 250,
            clause_count: 100,
            max_term_depth: 18,
            max_quantifier_nesting: 0,
            max_ite_depth: 3,
            max_let_depth: 2,
            bv_width: 0,
            runtime_seconds: 4.5,
        },
        // QF_UF — easy
        SyntheticEntry {
            logic: "QF_UF",
            file_size: 4000,
            atom_count: 40,
            clause_count: 15,
            max_term_depth: 9,
            max_quantifier_nesting: 0,
            max_ite_depth: 1,
            max_let_depth: 0,
            bv_width: 0,
            runtime_seconds: 0.4,
        },
        // QF_ABV — medium/hard
        SyntheticEntry {
            logic: "QF_ABV",
            file_size: 35000,
            atom_count: 350,
            clause_count: 140,
            max_term_depth: 22,
            max_quantifier_nesting: 0,
            max_ite_depth: 5,
            max_let_depth: 3,
            bv_width: 64,
            runtime_seconds: 9.0,
        },
        SyntheticEntry {
            logic: "QF_ABV",
            file_size: 120000,
            atom_count: 1200,
            clause_count: 480,
            max_term_depth: 38,
            max_quantifier_nesting: 0,
            max_ite_depth: 10,
            max_let_depth: 6,
            bv_width: 128,
            runtime_seconds: 42.0,
        },
        // QF_ALIA — hard
        SyntheticEntry {
            logic: "QF_ALIA",
            file_size: 90000,
            atom_count: 900,
            clause_count: 360,
            max_term_depth: 32,
            max_quantifier_nesting: 0,
            max_ite_depth: 9,
            max_let_depth: 4,
            bv_width: 0,
            runtime_seconds: 30.0,
        },
        // BV — very hard
        SyntheticEntry {
            logic: "BV",
            file_size: 400000,
            atom_count: 4000,
            clause_count: 1600,
            max_term_depth: 70,
            max_quantifier_nesting: 3,
            max_ite_depth: 15,
            max_let_depth: 8,
            bv_width: 64,
            runtime_seconds: 200.0,
        },
        // QF_LIA — very hard (large)
        SyntheticEntry {
            logic: "QF_LIA",
            file_size: 400000,
            atom_count: 4000,
            clause_count: 1600,
            max_term_depth: 65,
            max_quantifier_nesting: 0,
            max_ite_depth: 15,
            max_let_depth: 9,
            bv_width: 0,
            runtime_seconds: 80.0,
        },
    ];

    let mut ds = Dataset::new();

    for (idx, entry) in entries.iter().enumerate() {
        let sf = StructuralFeatures {
            max_term_depth: entry.max_term_depth,
            atom_count: entry.atom_count,
            clause_count: entry.clause_count,
            max_quantifier_nesting: entry.max_quantifier_nesting,
            bv_width_histogram: if entry.bv_width > 0 {
                vec![(entry.bv_width, 1)]
            } else {
                vec![]
            },
            array_dim_histogram: vec![],
            max_ite_depth: entry.max_ite_depth,
            max_let_depth: entry.max_let_depth,
        };

        let meta = BenchmarkMeta {
            path: PathBuf::from(format!("synthetic://seed/{idx}")),
            logic: Some(entry.logic.to_string()),
            expected_status: None,
            file_size: entry.file_size,
            category: None,
            structural_features: Some(sf),
        };

        let features = Features::from_meta(&meta);
        ds.push(Sample {
            features,
            runtime_seconds: entry.runtime_seconds,
            status: BenchmarkStatus::Sat,
        });
    }

    ds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_is_fitted() {
        let model = default_model();
        assert!(model.is_fitted);
    }

    #[test]
    fn test_default_model_predicts_finite() {
        let model = default_model();
        let f = Features::default();
        let rt = model.predict_runtime(&f);
        assert!(rt.is_finite());
        assert!(rt >= 0.0);
    }

    #[test]
    fn test_synthetic_seed_dataset_size() {
        let ds = synthetic_seed_dataset();
        assert!(ds.len() >= 20, "Seed dataset too small: {}", ds.len());
    }

    #[test]
    fn test_train_model_from_real_dataset() {
        let seed = synthetic_seed_dataset();
        let model = train_model(&seed);
        assert!(model.is_fitted);
        let f = Features::default();
        assert!(model.predict_runtime(&f).is_finite());
    }
}
