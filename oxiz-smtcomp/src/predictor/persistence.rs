//! JSON-based model persistence.
//!
//! Models are saved with a version-tagged envelope so that future changes to
//! the serialisation schema can be detected at load time.

use super::knn::KnnRegressor;
use super::linear::LinearRegressor;
use super::models::DifficultyModel;
use super::tree::RegressionTree;

/// Version string embedded in every saved model file.
const CURRENT_VERSION: &str = "0.2.2";

/// Top-level envelope wrapping any serialised model.
#[derive(serde::Serialize, serde::Deserialize)]
struct Envelope {
    /// Crate version at save time.
    oxiz_predictor_version: String,
    /// Model kind discriminator ("linear", "knn", "tree").
    kind: String,
    /// The serialised model as a JSON value.
    payload: serde_json::Value,
}

/// Persist `model` to a JSON file at `path`.
///
/// The file is written atomically in the sense that the entire content is
/// produced before `write` is called; on error the file may be partially
/// written.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if serialisation or the file write fails.
pub fn save_to_file(model: &dyn DifficultyModel, path: &std::path::Path) -> std::io::Result<()> {
    let model_json = model.to_json();
    let payload: serde_json::Value = serde_json::from_str(&model_json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let envelope = Envelope {
        oxiz_predictor_version: CURRENT_VERSION.to_string(),
        kind: model.name().to_string(),
        payload,
    };

    let json = serde_json::to_string_pretty(&envelope)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    std::fs::write(path, json)
}

/// Load a model from the JSON file at `path`.
///
/// The `kind` field in the envelope determines which concrete type is
/// deserialised.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if:
/// * the file cannot be read,
/// * the JSON is malformed,
/// * the version does not match [`CURRENT_VERSION`], or
/// * the `kind` is not one of "linear", "knn", "tree".
pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Box<dyn DifficultyModel>> {
    let bytes = std::fs::read(path)?;

    let envelope: Envelope = serde_json::from_slice(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    if envelope.oxiz_predictor_version != CURRENT_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "version mismatch: expected {CURRENT_VERSION}, got {}",
                envelope.oxiz_predictor_version
            ),
        ));
    }

    let payload_str = serde_json::to_string(&envelope.payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    match envelope.kind.as_str() {
        "linear" => {
            let model = LinearRegressor::from_json(&payload_str)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            Ok(Box::new(model))
        }
        "knn" => {
            let model = KnnRegressor::from_json(&payload_str)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            Ok(Box::new(model))
        }
        "tree" => {
            let model = RegressionTree::from_json(&payload_str)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            Ok(Box::new(model))
        }
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown predictor kind: {other}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkStatus;
    use crate::predictor::dataset::{Dataset, Sample};
    use crate::predictor::features::Features;
    use crate::predictor::report::TrainingConfig;
    use rand::SeedableRng;
    use std::env;

    fn make_tiny_dataset() -> Dataset {
        let mut ds = Dataset::new();
        for i in 0..5 {
            ds.push(Sample {
                features: Features {
                    atom_count: i as f64 * 10.0,
                    ..Default::default()
                },
                runtime_seconds: 0.1 * (i + 1) as f64,
                status: BenchmarkStatus::Sat,
            });
        }
        ds
    }

    #[test]
    fn test_round_trip_linear() {
        let ds = make_tiny_dataset();
        let mut model = LinearRegressor::new(1e-3);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        model.fit(&ds, &config, &mut rng);

        let mut path = env::temp_dir();
        path.push("oxiz_test_linear.json");
        save_to_file(&model, &path).expect("save failed");

        let loaded = load_from_file(&path).expect("load failed");
        let f = Features::default();
        let p1 = model.predict_runtime(&f);
        let p2 = loaded.predict_runtime(&f);
        assert!((p1 - p2).abs() < 1e-9);
    }

    #[test]
    fn test_round_trip_knn() {
        use crate::predictor::knn::KnnRegressor;
        let ds = make_tiny_dataset();
        let mut model = KnnRegressor::new(2);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        model.fit(&ds, &config, &mut rng);

        let mut path = env::temp_dir();
        path.push("oxiz_test_knn.json");
        save_to_file(&model, &path).expect("save failed");

        let loaded = load_from_file(&path).expect("load failed");
        let f = Features::default();
        let p1 = model.predict_runtime(&f);
        let p2 = loaded.predict_runtime(&f);
        assert!((p1 - p2).abs() < 1e-9);
    }

    #[test]
    fn test_round_trip_tree() {
        use crate::predictor::tree::RegressionTree;
        let ds = make_tiny_dataset();
        let mut model = RegressionTree::new(4, 2);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        model.fit(&ds, &config, &mut rng);

        let mut path = env::temp_dir();
        path.push("oxiz_test_tree.json");
        save_to_file(&model, &path).expect("save failed");

        let loaded = load_from_file(&path).expect("load failed");
        let f = Features::default();
        let p1 = model.predict_runtime(&f);
        let p2 = loaded.predict_runtime(&f);
        assert!((p1 - p2).abs() < 1e-9);
    }

    #[test]
    fn test_rejects_unknown_kind() {
        let envelope = Envelope {
            oxiz_predictor_version: CURRENT_VERSION.to_string(),
            kind: "nonexistent_model".to_string(),
            payload: serde_json::Value::Object(Default::default()),
        };
        let json = serde_json::to_string_pretty(&envelope).expect("serialize");
        let mut path = env::temp_dir();
        path.push("oxiz_test_unknown_kind.json");
        std::fs::write(&path, &json).expect("write");
        let result = load_from_file(&path);
        let err = result.err().expect("should fail for unknown kind");
        assert!(err.to_string().contains("unknown predictor kind"));
    }

    #[test]
    fn test_rejects_version_mismatch() {
        let envelope = Envelope {
            oxiz_predictor_version: "0.0.0".to_string(),
            kind: "linear".to_string(),
            payload: serde_json::Value::Object(Default::default()),
        };
        let json = serde_json::to_string_pretty(&envelope).expect("serialize");
        let mut path = env::temp_dir();
        path.push("oxiz_test_version_mismatch.json");
        std::fs::write(&path, &json).expect("write");
        let result = load_from_file(&path);
        let err = result.err().expect("should fail for version mismatch");
        assert!(err.to_string().contains("version mismatch"));
    }
}
