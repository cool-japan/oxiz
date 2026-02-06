//! Training Data Collection
//!
//! Collect and store training data from solver runs.

use serde::{Deserialize, Serialize};

/// A single training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input features
    pub features: Vec<f64>,
    /// Target output
    pub target: Vec<f64>,
    /// Optional metadata
    pub metadata: Option<String>,
}

impl TrainingExample {
    /// Create a new training example
    pub fn new(features: Vec<f64>, target: Vec<f64>) -> Self {
        Self {
            features,
            target,
            metadata: None,
        }
    }

    /// Create with metadata
    pub fn with_metadata(features: Vec<f64>, target: Vec<f64>, metadata: String) -> Self {
        Self {
            features,
            target,
            metadata: Some(metadata),
        }
    }
}

/// Dataset for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSet {
    /// Training examples
    pub examples: Vec<TrainingExample>,
    /// Dataset name
    pub name: String,
}

impl DataSet {
    /// Create a new empty dataset
    pub fn new(name: String) -> Self {
        Self {
            examples: Vec::new(),
            name,
        }
    }

    /// Add an example
    pub fn add_example(&mut self, example: TrainingExample) {
        self.examples.push(example);
    }

    /// Get number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Shuffle examples
    pub fn shuffle(&mut self) {
        // Simple pseudo-random shuffle
        let n = self.examples.len();
        for i in 0..n {
            let j = (i * 2654435761) % n;
            self.examples.swap(i, j);
        }
    }

    /// Split into train/test sets
    pub fn split(&self, train_fraction: f64) -> (DataSet, DataSet) {
        let train_size = (self.examples.len() as f64 * train_fraction) as usize;

        let mut train_set = DataSet::new(format!("{}_train", self.name));
        let mut test_set = DataSet::new(format!("{}_test", self.name));

        for (i, example) in self.examples.iter().enumerate() {
            if i < train_size {
                train_set.add_example(example.clone());
            } else {
                test_set.add_example(example.clone());
            }
        }

        (train_set, test_set)
    }

    /// Save to JSON file
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Serialization error: {}", e))?;

        std::fs::write(path, json).map_err(|e| format!("IO error: {}", e))?;

        Ok(())
    }

    /// Load from JSON file
    pub fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("IO error: {}", e))?;

        let dataset =
            serde_json::from_str(&json).map_err(|e| format!("Deserialization error: {}", e))?;

        Ok(dataset)
    }
}

/// Data collector for live collection during solving
pub struct DataCollector {
    /// Current dataset
    dataset: DataSet,
    /// Maximum dataset size
    max_size: usize,
}

impl DataCollector {
    /// Create a new data collector
    pub fn new(name: String, max_size: usize) -> Self {
        Self {
            dataset: DataSet::new(name),
            max_size,
        }
    }

    /// Collect a training example
    pub fn collect(&mut self, features: Vec<f64>, target: Vec<f64>) {
        if self.dataset.len() >= self.max_size {
            // Remove oldest example
            self.dataset.examples.remove(0);
        }

        self.dataset
            .add_example(TrainingExample::new(features, target));
    }

    /// Get collected dataset
    pub fn dataset(&self) -> &DataSet {
        &self.dataset
    }

    /// Get mutable dataset
    pub fn dataset_mut(&mut self) -> &mut DataSet {
        &mut self.dataset
    }

    /// Clear collected data
    pub fn clear(&mut self) {
        self.dataset.examples.clear();
    }

    /// Get number of collected examples
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_example() {
        let example = TrainingExample::new(vec![1.0, 2.0], vec![3.0]);
        assert_eq!(example.features.len(), 2);
        assert_eq!(example.target.len(), 1);
    }

    #[test]
    fn test_dataset() {
        let mut dataset = DataSet::new("test".to_string());
        assert!(dataset.is_empty());

        dataset.add_example(TrainingExample::new(vec![1.0], vec![2.0]));
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_dataset_split() {
        let mut dataset = DataSet::new("test".to_string());

        for i in 0..10 {
            dataset.add_example(TrainingExample::new(vec![i as f64], vec![i as f64 * 2.0]));
        }

        let (train, test) = dataset.split(0.8);
        assert_eq!(train.len(), 8);
        assert_eq!(test.len(), 2);
    }

    #[test]
    fn test_data_collector() {
        let mut collector = DataCollector::new("test".to_string(), 100);

        collector.collect(vec![1.0], vec![2.0]);
        collector.collect(vec![3.0], vec![4.0]);

        assert_eq!(collector.len(), 2);
    }

    #[test]
    fn test_data_collector_max_size() {
        let mut collector = DataCollector::new("test".to_string(), 2);

        collector.collect(vec![1.0], vec![1.0]);
        collector.collect(vec![2.0], vec![2.0]);
        collector.collect(vec![3.0], vec![3.0]);

        assert_eq!(collector.len(), 2);
    }
}
