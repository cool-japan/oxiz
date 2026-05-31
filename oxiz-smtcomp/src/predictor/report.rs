//! Training configuration, results, and evaluation statistics.

use super::class::DifficultyClass;

/// Hyper-parameters for iterative model training.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    /// Maximum number of gradient-descent epochs (default 1000).
    pub max_epochs: usize,
    /// Initial learning rate (default 0.01).
    pub learning_rate: f64,
    /// L2 regularisation coefficient applied to model weights (default 1e-3).
    pub l2_regularization: f64,
    /// Absolute loss improvement threshold for early stopping (default 1e-6).
    pub convergence_tol: f64,
    /// Number of consecutive epochs without sufficient improvement before
    /// early-stopping triggers (default 3).
    pub convergence_patience: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            learning_rate: 0.01,
            l2_regularization: 1e-3,
            convergence_tol: 1e-6,
            convergence_patience: 3,
        }
    }
}

/// Summary of a completed training run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingReport {
    /// Final training loss (MSE of log1p-runtime).
    pub final_loss: f64,
    /// Number of epochs actually executed before convergence.
    pub epochs: usize,
    /// Mean absolute error on the training set in seconds.
    pub mae_seconds: f64,
    /// Fraction of training samples whose predicted class matches actual class.
    pub class_accuracy: f64,
    /// Mean MAE from k-fold cross-validation, if performed.
    pub k_fold_mean_mae: Option<f64>,
    /// Wall-clock time spent training.
    pub time_to_train: std::time::Duration,
}

/// Aggregate evaluation statistics for a trained predictor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PredictorStats {
    /// Number of samples evaluated.
    pub count: usize,
    /// Mean absolute error in seconds.
    pub mae_seconds: f64,
    /// Fraction of samples with correct difficulty class.
    pub class_accuracy: f64,
    /// Fraction of predictions classified as `Trivial`.
    pub trivial_fraction: f64,
    /// Fraction of predictions classified as `Easy`.
    pub easy_fraction: f64,
    /// Fraction of predictions classified as `Medium`.
    pub medium_fraction: f64,
    /// Fraction of predictions classified as `Hard`.
    pub hard_fraction: f64,
    /// Fraction of predictions classified as `VeryHard`.
    pub very_hard_fraction: f64,
}

impl PredictorStats {
    /// Compute aggregate statistics from a slice of `(predicted_runtime_s, actual_class)` pairs.
    ///
    /// `predictions[i].0` is the model's predicted runtime in seconds.
    /// `predictions[i].1` is the ground-truth [`DifficultyClass`].
    #[must_use]
    pub fn from_predictions(predictions: &[(f64, DifficultyClass)]) -> Self {
        let count = predictions.len();
        if count == 0 {
            return Self {
                count: 0,
                mae_seconds: 0.0,
                class_accuracy: 0.0,
                trivial_fraction: 0.0,
                easy_fraction: 0.0,
                medium_fraction: 0.0,
                hard_fraction: 0.0,
                very_hard_fraction: 0.0,
            };
        }

        let n = count as f64;
        let mut total_ae = 0.0_f64;
        let mut correct = 0usize;
        let mut trivial = 0usize;
        let mut easy = 0usize;
        let mut medium = 0usize;
        let mut hard = 0usize;
        let mut very_hard = 0usize;

        for &(predicted_rt, actual_class) in predictions {
            let predicted_class = DifficultyClass::from_runtime_seconds(predicted_rt);

            // Approximate actual runtime from class mid-point for MAE calculation
            let actual_rt = class_midpoint(actual_class);
            total_ae += (predicted_rt - actual_rt).abs();

            if predicted_class == actual_class {
                correct += 1;
            }

            match actual_class {
                DifficultyClass::Trivial => trivial += 1,
                DifficultyClass::Easy => easy += 1,
                DifficultyClass::Medium => medium += 1,
                DifficultyClass::Hard => hard += 1,
                DifficultyClass::VeryHard => very_hard += 1,
            }
        }

        Self {
            count,
            mae_seconds: total_ae / n,
            class_accuracy: correct as f64 / n,
            trivial_fraction: trivial as f64 / n,
            easy_fraction: easy as f64 / n,
            medium_fraction: medium as f64 / n,
            hard_fraction: hard as f64 / n,
            very_hard_fraction: very_hard as f64 / n,
        }
    }

    /// Alias for `class_accuracy` — fraction of correctly classified samples.
    #[must_use]
    pub fn solve_rate(&self) -> f64 {
        self.class_accuracy
    }
}

/// Representative runtime (seconds) for each difficulty class, used for MAE
/// approximation when exact runtimes are unavailable.
fn class_midpoint(c: DifficultyClass) -> f64 {
    match c {
        DifficultyClass::Trivial => 0.05,
        DifficultyClass::Easy => 0.55,
        DifficultyClass::Medium => 5.5,
        DifficultyClass::Hard => 35.0,
        DifficultyClass::VeryHard => 120.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_stats_empty() {
        let stats = PredictorStats::from_predictions(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mae_seconds, 0.0);
    }

    #[test]
    fn test_predictor_stats_perfect_prediction() {
        // If we predict the class midpoint perfectly for trivial...
        let predictions = vec![(0.05, DifficultyClass::Trivial)];
        let stats = PredictorStats::from_predictions(&predictions);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.class_accuracy, 1.0);
        assert!((stats.mae_seconds).abs() < 1e-9);
    }

    #[test]
    fn test_training_config_defaults() {
        let cfg = TrainingConfig::default();
        assert_eq!(cfg.max_epochs, 1000);
        assert!((cfg.learning_rate - 0.01).abs() < 1e-12);
        assert_eq!(cfg.convergence_patience, 3);
    }
}
