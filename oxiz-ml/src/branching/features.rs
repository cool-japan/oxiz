//! Feature Extraction for Branching Decisions
//!
//! Extract relevant features from solver state for ML-guided branching.

use super::VarId;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// Features for a variable in branching context
#[derive(Debug, Clone)]
pub struct BranchingFeatures {
    /// Raw feature vector (standardized to fixed size)
    pub features: Vec<f64>,
}

impl BranchingFeatures {
    /// Create from raw feature vector
    pub fn from_vec(features: Vec<f64>) -> Self {
        Self { features }
    }

    /// Get number of features
    pub fn dim(&self) -> usize {
        self.features.len()
    }

    /// Normalize features to [0, 1] range
    pub fn normalize(&mut self) {
        if self.features.is_empty() {
            return;
        }

        let min = self.features.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .features
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            // All features are the same, set to 0.5
            self.features.fill(0.5);
        } else {
            for f in &mut self.features {
                *f = (*f - min) / (max - min);
            }
        }
    }

    /// Standardize features (zero mean, unit variance)
    pub fn standardize(&mut self) {
        if self.features.is_empty() {
            return;
        }

        let mean = self.features.iter().sum::<f64>() / self.features.len() as f64;
        let variance = self
            .features
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.features.len() as f64;
        let std = variance.sqrt();

        if std < 1e-10 {
            self.features.fill(0.0);
        } else {
            for f in &mut self.features {
                *f = (*f - mean) / std;
            }
        }
    }
}

/// Variable statistics for feature extraction
#[derive(Debug, Clone, Default)]
pub struct VarStats {
    /// Number of times variable appeared in conflicts
    pub conflict_count: usize,
    /// Number of times variable was involved in propagation
    pub propagation_count: usize,
    /// Number of times variable was a decision
    pub decision_count: usize,
    /// Activity score (VSIDS-style)
    pub activity: f64,
    /// Positive phase count (how often assigned true)
    pub positive_count: usize,
    /// Negative phase count (how often assigned false)
    pub negative_count: usize,
    /// Average decision level when assigned
    pub avg_decision_level: f64,
    /// Number of times in learned clauses
    pub learned_clause_count: usize,
    /// Average LBD of clauses containing this variable
    pub avg_lbd: f64,
    /// Number of times backtracked over
    pub backtrack_count: usize,
}

impl VarStats {
    /// Update with new conflict
    pub fn update_conflict(&mut self) {
        self.conflict_count += 1;
    }

    /// Update with new propagation
    pub fn update_propagation(&mut self) {
        self.propagation_count += 1;
    }

    /// Update with new decision
    pub fn update_decision(&mut self, polarity: bool) {
        self.decision_count += 1;
        if polarity {
            self.positive_count += 1;
        } else {
            self.negative_count += 1;
        }
    }

    /// Update activity score
    pub fn bump_activity(&mut self, amount: f64) {
        self.activity += amount;
    }

    /// Decay activity
    pub fn decay_activity(&mut self, factor: f64) {
        self.activity *= factor;
    }

    /// Get phase consistency (preference for true vs false)
    pub fn phase_consistency(&self) -> f64 {
        let total = (self.positive_count + self.negative_count) as f64;
        if total < 1.0 {
            0.5
        } else {
            self.positive_count as f64 / total
        }
    }

    /// Get conflict participation rate
    pub fn conflict_rate(&self) -> f64 {
        let total = (self.conflict_count + self.propagation_count + self.decision_count) as f64;
        if total < 1.0 {
            0.0
        } else {
            self.conflict_count as f64 / total
        }
    }
}

/// Feature extractor for branching decisions
pub struct FeatureExtractor {
    /// Variable statistics
    var_stats: FxHashMap<VarId, VarStats>,
    /// Global statistics
    total_conflicts: usize,
    total_propagations: usize,
    total_decisions: usize,
    /// Feature dimension (fixed)
    feature_dim: usize,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(feature_dim: usize) -> Self {
        Self {
            var_stats: FxHashMap::default(),
            total_conflicts: 0,
            total_propagations: 0,
            total_decisions: 0,
            feature_dim,
        }
    }

    /// Extract features for a variable
    pub fn extract(&self, var: VarId) -> BranchingFeatures {
        let stats = self.var_stats.get(&var).cloned().unwrap_or_default();

        // Build feature vector (15 features as per spec)
        let mut features = Vec::with_capacity(self.feature_dim);

        // 1. Normalized activity
        features.push(stats.activity);

        // 2. Conflict participation rate
        features.push(stats.conflict_rate());

        // 3. Propagation rate
        let total = (stats.conflict_count + stats.propagation_count + stats.decision_count) as f64;
        let prop_rate = if total > 0.0 {
            stats.propagation_count as f64 / total
        } else {
            0.0
        };
        features.push(prop_rate);

        // 4. Decision rate
        let decision_rate = if total > 0.0 {
            stats.decision_count as f64 / total
        } else {
            0.0
        };
        features.push(decision_rate);

        // 5. Phase consistency
        features.push(stats.phase_consistency());

        // 6. Average decision level (normalized)
        features.push(stats.avg_decision_level / 100.0); // Assume max depth ~100

        // 7. Learned clause participation rate
        let total_learned = if self.total_conflicts > 0 {
            self.total_conflicts as f64
        } else {
            1.0
        };
        features.push(stats.learned_clause_count as f64 / total_learned);

        // 8. Average LBD (normalized)
        features.push(stats.avg_lbd / 20.0); // Assume max LBD ~20

        // 9. Backtrack count (normalized)
        features.push((1.0 + stats.backtrack_count as f64).ln() / 10.0);

        // 10. Global conflict ratio
        let global_conflict_ratio = if self.total_conflicts + self.total_propagations > 0 {
            self.total_conflicts as f64 / (self.total_conflicts + self.total_propagations) as f64
        } else {
            0.0
        };
        features.push(global_conflict_ratio);

        // 11. Positive assignment bias
        let pos_bias =
            stats.positive_count as f64 / (stats.positive_count + stats.negative_count + 1) as f64;
        features.push(pos_bias);

        // 12. Negative assignment bias
        let neg_bias =
            stats.negative_count as f64 / (stats.positive_count + stats.negative_count + 1) as f64;
        features.push(neg_bias);

        // 13. Relative activity (compared to max)
        let max_activity = self
            .var_stats
            .values()
            .map(|s| s.activity)
            .fold(0.0f64, f64::max);
        let rel_activity = if max_activity > 0.0 {
            stats.activity / max_activity
        } else {
            0.0
        };
        features.push(rel_activity);

        // 14. Conflict/decision ratio
        let conflict_decision_ratio = if stats.decision_count > 0 {
            stats.conflict_count as f64 / stats.decision_count as f64
        } else {
            0.0
        };
        features.push(conflict_decision_ratio);

        // 15. Recency (inverse of backtrack count)
        let recency = 1.0 / (1.0 + stats.backtrack_count as f64);
        features.push(recency);

        // Pad or truncate to feature_dim
        features.resize(self.feature_dim, 0.0);

        BranchingFeatures { features }
    }

    /// Update statistics for a variable after conflict
    pub fn update_conflict(&mut self, var: VarId, lbd: f64) {
        self.total_conflicts += 1;
        let stats = self.var_stats.entry(var).or_default();
        stats.update_conflict();
        stats.learned_clause_count += 1;

        // Update average LBD
        let n = stats.learned_clause_count as f64;
        stats.avg_lbd = (stats.avg_lbd * (n - 1.0) + lbd) / n;
    }

    /// Update statistics for a variable after propagation
    pub fn update_propagation(&mut self, var: VarId) {
        self.total_propagations += 1;
        self.var_stats.entry(var).or_default().update_propagation();
    }

    /// Update statistics for a variable after decision
    pub fn update_decision(&mut self, var: VarId, polarity: bool, level: usize) {
        self.total_decisions += 1;
        let stats = self.var_stats.entry(var).or_default();
        stats.update_decision(polarity);

        // Update average decision level
        let n = stats.decision_count as f64;
        stats.avg_decision_level = (stats.avg_decision_level * (n - 1.0) + level as f64) / n;
    }

    /// Update activity scores
    pub fn bump_activity(&mut self, var: VarId, amount: f64) {
        self.var_stats.entry(var).or_default().bump_activity(amount);
    }

    /// Decay all activity scores
    pub fn decay_all_activities(&mut self, factor: f64) {
        for stats in self.var_stats.values_mut() {
            stats.decay_activity(factor);
        }
    }

    /// Update backtrack count
    pub fn update_backtrack(&mut self, var: VarId) {
        self.var_stats.entry(var).or_default().backtrack_count += 1;
    }

    /// Get variable statistics
    pub fn get_stats(&self, var: VarId) -> Option<&VarStats> {
        self.var_stats.get(&var)
    }

    /// Clear all statistics
    pub fn clear(&mut self) {
        self.var_stats.clear();
        self.total_conflicts = 0;
        self.total_propagations = 0;
        self.total_decisions = 0;
    }

    /// Get number of tracked variables
    pub fn num_variables(&self) -> usize {
        self.var_stats.len()
    }
}

/// Feature cache for performance
pub struct FeatureCache {
    /// Cached features
    cache: HashMap<VarId, (BranchingFeatures, u64)>,
    /// Cache timestamp counter
    timestamp: u64,
    /// Maximum cache size
    max_size: usize,
}

impl FeatureCache {
    /// Create a new feature cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            timestamp: 0,
            max_size,
        }
    }

    /// Get cached features
    pub fn get(&self, var: VarId) -> Option<&BranchingFeatures> {
        self.cache.get(&var).map(|(features, _)| features)
    }

    /// Insert features into cache
    pub fn insert(&mut self, var: VarId, features: BranchingFeatures) {
        self.timestamp += 1;

        // Evict old entries if cache is full
        if self.cache.len() >= self.max_size {
            // Remove oldest entry
            if let Some((&oldest_var, _)) = self.cache.iter().min_by_key(|(_, (_, ts))| ts) {
                self.cache.remove(&oldest_var);
            }
        }

        self.cache.insert(var, (features, self.timestamp));
    }

    /// Invalidate cache for a variable
    pub fn invalidate(&mut self, var: VarId) {
        self.cache.remove(&var);
    }

    /// Clear entire cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.timestamp = 0;
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        // Would need to track hits/misses
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branching_features_normalize() {
        let mut features = BranchingFeatures::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        features.normalize();

        assert!((features.features[0] - 0.0).abs() < 1e-10);
        assert!((features.features[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_var_stats() {
        let mut stats = VarStats::default();

        stats.update_conflict();
        stats.update_propagation();
        stats.update_decision(true);

        assert_eq!(stats.conflict_count, 1);
        assert_eq!(stats.propagation_count, 1);
        assert_eq!(stats.decision_count, 1);
        assert_eq!(stats.positive_count, 1);
    }

    #[test]
    fn test_var_stats_phase_consistency() {
        let mut stats = VarStats::default();

        stats.update_decision(true);
        stats.update_decision(true);
        stats.update_decision(false);

        let consistency = stats.phase_consistency();
        assert!((consistency - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_feature_extractor() {
        let extractor = FeatureExtractor::new(15);

        let features = extractor.extract(0);
        assert_eq!(features.dim(), 15);
        assert!(features.features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_feature_extractor_updates() {
        let mut extractor = FeatureExtractor::new(15);

        extractor.update_conflict(0, 3.0);
        extractor.update_propagation(0);
        extractor.update_decision(0, true, 5);

        let stats = extractor.get_stats(0).unwrap();
        assert_eq!(stats.conflict_count, 1);
        assert_eq!(stats.propagation_count, 1);
        assert_eq!(stats.decision_count, 1);
    }

    #[test]
    fn test_feature_cache() {
        let mut cache = FeatureCache::new(10);

        let features = BranchingFeatures::from_vec(vec![1.0, 2.0, 3.0]);
        cache.insert(0, features);

        assert!(cache.get(0).is_some());
        assert!(cache.get(1).is_none());

        cache.invalidate(0);
        assert!(cache.get(0).is_none());
    }

    #[test]
    fn test_feature_cache_eviction() {
        let mut cache = FeatureCache::new(2);

        cache.insert(0, BranchingFeatures::from_vec(vec![1.0]));
        cache.insert(1, BranchingFeatures::from_vec(vec![2.0]));
        cache.insert(2, BranchingFeatures::from_vec(vec![3.0]));

        // Cache should have only 2 entries (oldest evicted)
        assert_eq!(cache.cache.len(), 2);
    }
}
