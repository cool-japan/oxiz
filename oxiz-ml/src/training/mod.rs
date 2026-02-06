//! Training Infrastructure
//!
//! Collect training data and train models offline or online.

mod data_collection;
mod offline_trainer;
mod online_learning;

pub use data_collection::{DataCollector, DataSet, TrainingExample};
pub use offline_trainer::{OfflineTrainer, TrainingConfig, TrainingResult};
pub use online_learning::{OnlineConfig, OnlineLearner};

/// Training statistics
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    /// Number of training examples
    pub num_examples: usize,
    /// Number of epochs completed
    pub epochs: usize,
    /// Final training loss
    pub final_loss: f64,
    /// Training time (seconds)
    pub training_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_stats_default() {
        let stats = TrainingStats::default();
        assert_eq!(stats.num_examples, 0);
    }
}
