//! Online Learning
//!
//! Update models incrementally during solving.

use crate::MLStats;
use crate::models::{Model, ModelError};

/// Online learning configuration
#[derive(Debug, Clone)]
pub struct OnlineConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Update frequency (update every N examples)
    pub update_frequency: usize,
    /// Mini-batch size for updates
    pub mini_batch_size: usize,
}

impl Default for OnlineConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001, // Lower LR for online learning
            update_frequency: 10,
            mini_batch_size: 5,
        }
    }
}

/// Online learner
pub struct OnlineLearner {
    /// Configuration
    config: OnlineConfig,
    /// Buffer for mini-batch updates
    buffer: Vec<(Vec<f64>, Vec<f64>)>,
    /// Update counter
    update_count: usize,
    /// Statistics
    stats: MLStats,
}

impl OnlineLearner {
    /// Create a new online learner
    pub fn new(config: OnlineConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            update_count: 0,
            stats: MLStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(OnlineConfig::default())
    }

    /// Add a training example (may trigger update)
    pub fn add_example<M: Model>(
        &mut self,
        model: &mut M,
        features: Vec<f64>,
        target: Vec<f64>,
    ) -> Result<Option<f64>, ModelError> {
        self.buffer.push((features, target));

        // Check if we should update
        if self.buffer.len() >= self.config.mini_batch_size
            && self
                .update_count
                .is_multiple_of(self.config.update_frequency)
        {
            let loss = self.update(model)?;
            self.buffer.clear();
            Ok(Some(loss))
        } else {
            self.update_count += 1;
            Ok(None)
        }
    }

    /// Force an update with current buffer
    pub fn update<M: Model>(&mut self, model: &mut M) -> Result<f64, ModelError> {
        if self.buffer.is_empty() {
            return Ok(0.0);
        }

        let start = std::time::Instant::now();

        let mut total_loss = 0.0;

        for (features, target) in &self.buffer {
            let loss = model.train(features, target)?;
            total_loss += loss;
        }

        let avg_loss = total_loss / self.buffer.len() as f64;

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.record_training_time(elapsed);
        self.update_count += 1;

        Ok(avg_loss)
    }

    /// Get statistics
    pub fn stats(&self) -> &MLStats {
        &self.stats
    }

    /// Reset buffer and statistics
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.update_count = 0;
        self.stats = MLStats::default();
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get update count
    pub fn update_count(&self) -> usize {
        self.update_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LinearRegression;

    #[test]
    fn test_online_learner_creation() {
        let learner = OnlineLearner::default_config();
        assert_eq!(learner.update_count, 0);
    }

    #[test]
    fn test_online_learner_add_example() {
        let mut learner = OnlineLearner::default_config();
        let mut model = LinearRegression::new(2);

        let result = learner.add_example(&mut model, vec![1.0, 2.0], vec![3.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_online_learner_buffering() {
        let mut learner = OnlineLearner::default_config();
        let mut model = LinearRegression::new(2);

        // Add examples until mini-batch is full
        for i in 0..5 {
            learner
                .add_example(
                    &mut model,
                    vec![i as f64, (i + 1) as f64],
                    vec![(i * 2) as f64],
                )
                .unwrap();
        }

        // Buffer should trigger update after 5 examples
        assert!(learner.buffer_size() <= 5);
    }

    #[test]
    fn test_online_learner_reset() {
        let mut learner = OnlineLearner::default_config();
        let mut model = LinearRegression::new(2);

        learner
            .add_example(&mut model, vec![1.0, 2.0], vec![3.0])
            .unwrap();
        learner.reset();

        assert_eq!(learner.buffer_size(), 0);
        assert_eq!(learner.update_count, 0);
    }
}
