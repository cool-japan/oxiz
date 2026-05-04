//! Object-safe trait shared by all difficulty models.
//!
//! Concrete implementations live in [`super::linear`], [`super::knn`], and
//! [`super::tree`].  Deserialization (`from_json`) is deliberately **not** on
//! this trait so that it can remain object-safe.  The persistence layer
//! handles dispatch.

use super::class::DifficultyClass;
use super::dataset::Dataset;
use super::features::Features;
use super::report::{TrainingConfig, TrainingReport};

/// A trainable difficulty predictor.
///
/// All implementations must be `Send + Sync` so they can be shared across
/// rayon threads.  The trait is object-safe: `from_json` is omitted and `rng`
/// is passed as `&mut dyn rand::Rng`.
pub trait DifficultyModel: Send + Sync {
    /// Short name used as the `kind` discriminator in serialized envelopes.
    fn name(&self) -> &'static str;

    /// Predict solver runtime in seconds for the given feature vector.
    fn predict_runtime(&self, features: &Features) -> f64;

    /// Predict the difficulty class for the given feature vector.
    ///
    /// Default implementation delegates to [`predict_runtime`] and converts
    /// via [`DifficultyClass::from_runtime_seconds`].
    fn predict_class(&self, features: &Features) -> DifficultyClass {
        DifficultyClass::from_runtime_seconds(self.predict_runtime(features))
    }

    /// Fit the model on the given dataset.
    ///
    /// The `rng` parameter uses `&mut dyn rand::Rng` to keep the trait
    /// object-safe (no generic parameter on the trait itself).
    fn fit(
        &mut self,
        dataset: &Dataset,
        config: &TrainingConfig,
        rng: &mut dyn rand::Rng,
    ) -> TrainingReport;

    /// Serialise the model to a JSON string.
    ///
    /// This is intentionally infallible for concrete types whose fields are
    /// always JSON-representable (`Vec<f64>`, `f64`, `bool`, …).
    fn to_json(&self) -> String;
}
