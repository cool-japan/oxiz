//! ML-Guided Clause Deletion Policy
#![allow(clippy::non_canonical_partial_ord_impl)] // Custom ordering logic
//!
//! Decide which clauses to delete based on ML predictions.

use super::{ClauseFeatures, ClauseFeedback, ClauseId, UsefulnessPredictor};
use crate::MLStats;
use std::cmp::Ordering;

/// Deletion decision for a clause
#[derive(Debug, Clone, Copy)]
pub struct DeletionDecision {
    /// Clause ID
    pub clause_id: ClauseId,
    /// Should delete this clause?
    pub should_delete: bool,
    /// Priority (lower = more likely to delete)
    pub priority: f64,
}

impl DeletionDecision {
    /// Create a new deletion decision
    pub fn new(clause_id: ClauseId, should_delete: bool, priority: f64) -> Self {
        Self {
            clause_id,
            should_delete,
            priority,
        }
    }
}

/// For priority queue (max-heap, we want to keep high-priority clauses)
#[derive(Debug, Clone, Copy)]
struct ClausePriority {
    clause_id: ClauseId,
    priority: f64,
}

impl PartialEq for ClausePriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for ClausePriority {}

impl PartialOrd for ClausePriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse for min-heap behavior
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for ClausePriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Deletion policy configuration
#[derive(Debug, Clone)]
pub struct DeletionConfig {
    /// Target clause database size
    pub target_size: usize,
    /// Deletion fraction (what fraction to delete when over target)
    pub deletion_fraction: f64,
    /// Use ML for deletion decisions
    pub use_ml: bool,
    /// Minimum usefulness score to keep
    pub min_usefulness: f64,
    /// Always keep glue clauses (LBD <= 2)
    pub keep_glue: bool,
}

impl Default for DeletionConfig {
    fn default() -> Self {
        Self {
            target_size: 10_000,
            deletion_fraction: 0.5,
            use_ml: true,
            min_usefulness: 0.4,
            keep_glue: true,
        }
    }
}

/// ML-guided clause deletion policy
pub struct DeletionPolicy {
    /// Usefulness predictor
    predictor: Option<UsefulnessPredictor>,
    /// Configuration
    config: DeletionConfig,
    /// Statistics
    stats: MLStats,
}

impl DeletionPolicy {
    /// Create a new deletion policy
    pub fn new(config: DeletionConfig, predictor: Option<UsefulnessPredictor>) -> Self {
        Self {
            predictor,
            config,
            stats: MLStats::default(),
        }
    }

    /// Create with default configuration and ML predictor
    pub fn default_config() -> Self {
        let config = DeletionConfig::default();
        let predictor = if config.use_ml {
            Some(UsefulnessPredictor::default_config())
        } else {
            None
        };

        Self::new(config, predictor)
    }

    /// Decide which clauses to delete from a set
    pub fn select_clauses_to_delete(
        &mut self,
        clauses: &[(ClauseId, ClauseFeatures)],
    ) -> Vec<ClauseId> {
        if clauses.len() <= self.config.target_size {
            return Vec::new();
        }

        let num_to_delete = ((clauses.len() - self.config.target_size) as f64
            * self.config.deletion_fraction) as usize;

        let start = std::time::Instant::now();

        // Compute priorities for all clauses
        let mut priorities: Vec<(ClauseId, f64, bool)> = clauses
            .iter()
            .map(|(id, features)| {
                let (priority, is_glue) = self.compute_priority(*id, features);
                (*id, priority, is_glue)
            })
            .collect();

        // Sort by priority (ascending = delete first)
        priorities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Select clauses to delete, respecting keep_glue setting
        let to_delete: Vec<ClauseId> = priorities
            .iter()
            .filter(|(_, _, is_glue)| !self.config.keep_glue || !is_glue)
            .take(num_to_delete)
            .map(|(id, _, _)| *id)
            .collect();

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.record_prediction_time(elapsed);

        to_delete
    }

    /// Compute priority for a clause (lower = more likely to delete)
    fn compute_priority(&mut self, clause_id: ClauseId, features: &ClauseFeatures) -> (f64, bool) {
        // Check if it's a glue clause (LBD <= 2)
        let lbd = if !features.features.is_empty() {
            (features.features[0] * 20.0) as usize
        } else {
            10
        };
        let is_glue = lbd <= 2;

        // If using ML predictor
        if let Some(ref mut predictor) = self.predictor {
            let prediction = predictor.predict_usefulness(features);

            // Priority = usefulness score (higher = keep, lower = delete)
            let priority = if is_glue && self.config.keep_glue {
                1.0 // Maximum priority for glue clauses
            } else {
                prediction.score
            };

            (priority, is_glue)
        } else {
            // Fallback: use LBD and activity
            let lbd_score = 1.0 - (lbd as f64 / 20.0);
            let activity = if features.features.len() > 2 {
                features.features[2]
            } else {
                0.5
            };

            let priority = if is_glue && self.config.keep_glue {
                1.0
            } else {
                (lbd_score + activity) / 2.0
            };

            (priority, is_glue)
        }
    }

    /// Should delete a specific clause?
    pub fn should_delete(&mut self, features: &ClauseFeatures) -> bool {
        let (priority, is_glue) = self.compute_priority(0, features);

        if self.config.keep_glue && is_glue {
            return false;
        }

        priority < self.config.min_usefulness
    }

    /// Learn from feedback about deleted/kept clauses
    pub fn learn_from_feedback(&mut self, features: &ClauseFeatures, feedback: ClauseFeedback) {
        if let Some(ref mut predictor) = self.predictor {
            predictor.learn_from_feedback(features, feedback);

            if feedback.was_useful {
                self.stats.record_correct();
            } else {
                self.stats.record_incorrect();
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &MLStats {
        &self.stats
    }

    /// Get predictor statistics if available
    pub fn predictor_stats(&self) -> Option<&MLStats> {
        self.predictor.as_ref().map(|p| p.stats())
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MLStats::default();
        if let Some(ref mut predictor) = self.predictor {
            predictor.reset_stats();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deletion_decision() {
        let decision = DeletionDecision::new(1, true, 0.3);
        assert_eq!(decision.clause_id, 1);
        assert!(decision.should_delete);
    }

    #[test]
    fn test_deletion_policy_creation() {
        let policy = DeletionPolicy::default_config();
        assert!(policy.predictor.is_some());
    }

    #[test]
    fn test_deletion_policy_select_clauses() {
        let mut policy = DeletionPolicy::default_config();

        // Create some test clauses
        let clauses: Vec<(ClauseId, ClauseFeatures)> = (0..100)
            .map(|i| {
                let features = ClauseFeatures::extract(
                    (i % 10) + 1, // varying LBD
                    10,
                    0.5,
                    i,
                    5,
                    2,
                    20,
                    15,
                );
                (i, features)
            })
            .collect();

        // Target size is 10000, so with 100 clauses, should delete none
        let to_delete = policy.select_clauses_to_delete(&clauses);
        assert!(to_delete.is_empty());
    }

    #[test]
    fn test_deletion_policy_should_delete() {
        let mut policy = DeletionPolicy::default_config();

        // Low quality clause (high LBD, low activity)
        let features = ClauseFeatures::extract(15, 50, 0.1, 1000, 0, 10, 50, 40);

        // Prediction may vary, but should be callable
        let _should_delete = policy.should_delete(&features);
    }

    #[test]
    fn test_deletion_policy_glue_clauses() {
        let config = DeletionConfig {
            keep_glue: true,
            ..Default::default()
        };

        let mut policy = DeletionPolicy::new(config, None);

        // Glue clause (LBD = 2)
        let features = ClauseFeatures::extract(2, 5, 0.8, 10, 5, 2, 10, 5);

        let should_delete = policy.should_delete(&features);
        assert!(!should_delete); // Should not delete glue clauses
    }
}
