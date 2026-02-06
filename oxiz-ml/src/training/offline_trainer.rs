//! Offline Model Training
//!
//! Train models on collected datasets.

use super::{DataSet, TrainingStats};
use crate::models::{Model, ModelError};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Validation split
    pub validation_split: f64,
    /// Early stopping patience
    pub patience: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.01,
            batch_size: 32,
            validation_split: 0.2,
            patience: 10,
        }
    }
}

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Training statistics
    pub stats: TrainingStats,
    /// Training losses per epoch
    pub train_losses: Vec<f64>,
    /// Validation losses per epoch
    pub val_losses: Vec<f64>,
    /// Best epoch
    pub best_epoch: usize,
}

/// Offline trainer
pub struct OfflineTrainer {
    /// Configuration
    config: TrainingConfig,
}

impl OfflineTrainer {
    /// Create a new offline trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(TrainingConfig::default())
    }

    /// Train a model on a dataset
    pub fn train<M: Model>(
        &self,
        model: &mut M,
        dataset: &DataSet,
    ) -> Result<TrainingResult, ModelError> {
        let start = std::time::Instant::now();

        // Split dataset
        let (train_set, val_set) = dataset.split(1.0 - self.config.validation_split);

        let mut train_losses = Vec::new();
        let mut val_losses = Vec::new();
        let mut best_val_loss = f64::INFINITY;
        let mut best_epoch = 0;
        let mut patience_counter = 0;

        // Training loop
        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Train on all examples
            for example in &train_set.examples {
                let loss = model.train(&example.features, &example.target)?;
                epoch_loss += loss;
                num_batches += 1;
            }

            let avg_train_loss = epoch_loss / num_batches as f64;
            train_losses.push(avg_train_loss);

            // Validation
            let val_loss = self.validate(model, &val_set)?;
            val_losses.push(val_loss);

            // Early stopping check
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    break;
                }
            }
        }

        let training_time = start.elapsed().as_secs_f64();

        let stats = TrainingStats {
            num_examples: dataset.len(),
            epochs: train_losses.len(),
            final_loss: *train_losses.last().unwrap_or(&0.0),
            training_time,
        };

        Ok(TrainingResult {
            stats,
            train_losses,
            val_losses,
            best_epoch,
        })
    }

    /// Validate model on dataset
    fn validate<M: Model>(&self, model: &M, dataset: &DataSet) -> Result<f64, ModelError> {
        let mut total_loss = 0.0;

        for example in &dataset.examples {
            let prediction = model.predict(&example.features);

            // Compute MSE loss
            let loss: f64 = prediction
                .iter()
                .zip(&example.target)
                .map(|(pred, target)| (pred - target).powi(2))
                .sum::<f64>()
                / prediction.len() as f64;

            total_loss += loss;
        }

        Ok(total_loss / dataset.len() as f64)
    }

    /// Get configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LinearRegression;

    #[test]
    fn test_offline_trainer_creation() {
        let trainer = OfflineTrainer::default_config();
        assert_eq!(trainer.config.epochs, 100);
    }

    #[test]
    fn test_offline_trainer_train() {
        let trainer = OfflineTrainer::default_config();
        let mut model = LinearRegression::new(2);

        let mut dataset = DataSet::new("test".to_string());
        for i in 0..50 {
            dataset.add_example(super::super::TrainingExample::new(
                vec![i as f64, (i + 1) as f64],
                vec![(i * 2) as f64],
            ));
        }

        let result = trainer.train(&mut model, &dataset);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.stats.epochs > 0);
    }
}
