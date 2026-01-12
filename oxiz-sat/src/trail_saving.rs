//! Trail saving and restoration for SAT solving
//!
//! This module implements trail saving, a technique where successful decision
//! sequences (trails) are saved and can be reused to guide future search.
//! This can help the solver quickly return to promising areas of the search space.
//!
//! References:
//! - "Trail Saving on Backtrack" (IJCAI 2009)
//! - "Improving CDCL SAT Solvers through Trail Saving"

use crate::literal::{Lit, Var};

/// A saved trail representing a sequence of decisions
#[derive(Debug, Clone)]
pub struct SavedTrail {
    /// Sequence of decision literals
    pub decisions: Vec<Lit>,
    /// Quality score (e.g., depth reached, conflicts avoided)
    pub quality: f64,
    /// Number of times this trail was useful
    pub use_count: usize,
}

impl SavedTrail {
    /// Create a new saved trail
    #[must_use]
    pub fn new(decisions: Vec<Lit>, quality: f64) -> Self {
        Self {
            decisions,
            quality,
            use_count: 0,
        }
    }

    /// Get the length of the trail
    #[must_use]
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Check if trail is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }
}

/// Statistics for trail saving
#[derive(Debug, Clone, Default)]
pub struct TrailSavingStats {
    /// Number of trails saved
    pub trails_saved: usize,
    /// Number of trails reused
    pub trails_reused: usize,
    /// Number of successful trail applications
    pub successful_applications: usize,
    /// Number of failed trail applications (led to early conflict)
    pub failed_applications: usize,
    /// Total saved trail length
    pub total_trail_length: usize,
}

impl TrailSavingStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Trail Saving Statistics:");
        println!("  Trails saved: {}", self.trails_saved);
        println!("  Trails reused: {}", self.trails_reused);
        println!(
            "  Successful applications: {}",
            self.successful_applications
        );
        println!("  Failed applications: {}", self.failed_applications);
        if self.trails_saved > 0 {
            let avg_length = self.total_trail_length as f64 / self.trails_saved as f64;
            println!("  Average trail length: {:.1}", avg_length);
        }
        if self.trails_reused > 0 {
            let success_rate =
                100.0 * self.successful_applications as f64 / self.trails_reused as f64;
            println!("  Success rate: {:.1}%", success_rate);
        }
    }
}

/// Trail saving manager
///
/// Maintains a collection of successful decision trails and provides
/// mechanisms to save and restore them during search.
#[derive(Debug)]
pub struct TrailSavingManager {
    /// Collection of saved trails
    saved_trails: Vec<SavedTrail>,
    /// Maximum number of trails to keep
    max_trails: usize,
    /// Minimum quality threshold for saving a trail
    min_quality: f64,
    /// Current trail being built
    current_trail: Vec<Lit>,
    /// Statistics
    stats: TrailSavingStats,
}

impl Default for TrailSavingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TrailSavingManager {
    /// Create a new trail saving manager with default settings
    ///
    /// Defaults:
    /// - Max trails: 10
    /// - Min quality: 1.0
    #[must_use]
    pub fn new() -> Self {
        Self {
            saved_trails: Vec::new(),
            max_trails: 10,
            min_quality: 1.0,
            current_trail: Vec::new(),
            stats: TrailSavingStats::default(),
        }
    }

    /// Create with custom settings
    #[must_use]
    pub fn with_config(max_trails: usize, min_quality: f64) -> Self {
        Self {
            saved_trails: Vec::new(),
            max_trails,
            min_quality,
            current_trail: Vec::new(),
            stats: TrailSavingStats::default(),
        }
    }

    /// Record a decision in the current trail
    pub fn record_decision(&mut self, lit: Lit) {
        self.current_trail.push(lit);
    }

    /// Save the current trail with given quality score
    ///
    /// Quality could be based on:
    /// - Decision level reached
    /// - Number of conflicts avoided
    /// - Search depth
    pub fn save_current_trail(&mut self, quality: f64) {
        // Only save if quality is sufficient
        if quality < self.min_quality {
            return;
        }

        // Don't save empty trails
        if self.current_trail.is_empty() {
            return;
        }

        let trail = SavedTrail::new(self.current_trail.clone(), quality);
        self.stats.total_trail_length += trail.len();
        self.stats.trails_saved += 1;

        // Add to collection
        self.saved_trails.push(trail);

        // Sort by quality (best first)
        self.saved_trails.sort_by(|a, b| {
            b.quality
                .partial_cmp(&a.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep only top trails
        if self.saved_trails.len() > self.max_trails {
            self.saved_trails.truncate(self.max_trails);
        }
    }

    /// Clear the current trail being built
    pub fn clear_current_trail(&mut self) {
        self.current_trail.clear();
    }

    /// Get the best saved trail
    ///
    /// Returns None if no trails are saved
    #[must_use]
    pub fn get_best_trail(&mut self) -> Option<&mut SavedTrail> {
        if self.saved_trails.is_empty() {
            None
        } else {
            self.stats.trails_reused += 1;
            Some(&mut self.saved_trails[0])
        }
    }

    /// Get a trail by index (0 = best quality)
    #[must_use]
    pub fn get_trail(&mut self, index: usize) -> Option<&mut SavedTrail> {
        if index < self.saved_trails.len() {
            self.stats.trails_reused += 1;
            Some(&mut self.saved_trails[index])
        } else {
            None
        }
    }

    /// Record that a trail application was successful
    pub fn record_success(&mut self) {
        self.stats.successful_applications += 1;
    }

    /// Record that a trail application failed
    pub fn record_failure(&mut self) {
        self.stats.failed_applications += 1;
    }

    /// Get all saved trails
    #[must_use]
    pub fn trails(&self) -> &[SavedTrail] {
        &self.saved_trails
    }

    /// Get number of saved trails
    #[must_use]
    pub fn num_trails(&self) -> usize {
        self.saved_trails.len()
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &TrailSavingStats {
        &self.stats
    }

    /// Remove low-quality trails
    ///
    /// Removes trails with use_count = 0 and quality below threshold
    pub fn prune_trails(&mut self, quality_threshold: f64) {
        self.saved_trails
            .retain(|trail| trail.use_count > 0 || trail.quality >= quality_threshold);
    }

    /// Clear all saved trails
    pub fn clear(&mut self) {
        self.saved_trails.clear();
        self.current_trail.clear();
    }

    /// Check if a variable appears in any saved trail
    ///
    /// Useful for deciding variable deletion
    #[must_use]
    pub fn is_var_in_trails(&self, var: Var) -> bool {
        self.saved_trails
            .iter()
            .any(|trail| trail.decisions.iter().any(|&lit| lit.var() == var))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trail_saving_manager_creation() {
        let manager = TrailSavingManager::new();
        assert_eq!(manager.num_trails(), 0);
        assert_eq!(manager.max_trails, 10);
    }

    #[test]
    fn test_record_and_save_trail() {
        let mut manager = TrailSavingManager::new();

        manager.record_decision(Lit::pos(Var::new(0)));
        manager.record_decision(Lit::neg(Var::new(1)));
        manager.save_current_trail(10.0);

        assert_eq!(manager.num_trails(), 1);
        assert_eq!(manager.stats().trails_saved, 1);
    }

    #[test]
    fn test_quality_threshold() {
        let mut manager = TrailSavingManager::new();

        // Low quality trail should not be saved
        manager.record_decision(Lit::pos(Var::new(0)));
        manager.save_current_trail(0.5);
        assert_eq!(manager.num_trails(), 0);

        // High quality trail should be saved
        manager.clear_current_trail();
        manager.record_decision(Lit::pos(Var::new(1)));
        manager.save_current_trail(10.0);
        assert_eq!(manager.num_trails(), 1);
    }

    #[test]
    fn test_max_trails_limit() {
        let mut manager = TrailSavingManager::with_config(3, 1.0);

        // Save 5 trails with different qualities
        for i in 0..5 {
            manager.record_decision(Lit::pos(Var::new(i)));
            manager.save_current_trail((i + 1) as f64);
            manager.clear_current_trail();
        }

        // Should keep only top 3
        assert_eq!(manager.num_trails(), 3);

        // Best quality should be first
        if let Some(best) = manager.get_best_trail() {
            assert_eq!(best.quality, 5.0);
        }
    }

    #[test]
    fn test_get_best_trail() {
        let mut manager = TrailSavingManager::new();

        manager.record_decision(Lit::pos(Var::new(0)));
        manager.save_current_trail(5.0);
        manager.clear_current_trail();

        manager.record_decision(Lit::pos(Var::new(1)));
        manager.save_current_trail(10.0);

        if let Some(best) = manager.get_best_trail() {
            assert_eq!(best.quality, 10.0);
            assert_eq!(manager.stats().trails_reused, 1);
        }
    }

    #[test]
    fn test_trail_use_tracking() {
        let mut manager = TrailSavingManager::new();

        manager.record_decision(Lit::pos(Var::new(0)));
        manager.save_current_trail(5.0);

        if let Some(trail) = manager.get_best_trail() {
            trail.use_count += 1;
            assert_eq!(trail.use_count, 1);
        }

        manager.record_success();
        assert_eq!(manager.stats().successful_applications, 1);
    }

    #[test]
    fn test_prune_trails() {
        let mut manager = TrailSavingManager::new();

        // Add trail with low quality, never used
        manager.record_decision(Lit::pos(Var::new(0)));
        manager.save_current_trail(2.0);
        manager.clear_current_trail();

        // Add trail with high quality
        manager.record_decision(Lit::pos(Var::new(1)));
        manager.save_current_trail(10.0);

        // Prune trails with quality < 5.0 and use_count = 0
        manager.prune_trails(5.0);

        assert_eq!(manager.num_trails(), 1);
        if let Some(trail) = manager.get_best_trail() {
            assert_eq!(trail.quality, 10.0);
        }
    }

    #[test]
    fn test_is_var_in_trails() {
        let mut manager = TrailSavingManager::new();

        manager.record_decision(Lit::pos(Var::new(0)));
        manager.record_decision(Lit::neg(Var::new(1)));
        manager.save_current_trail(5.0);

        assert!(manager.is_var_in_trails(Var::new(0)));
        assert!(manager.is_var_in_trails(Var::new(1)));
        assert!(!manager.is_var_in_trails(Var::new(2)));
    }

    #[test]
    fn test_clear() {
        let mut manager = TrailSavingManager::new();

        manager.record_decision(Lit::pos(Var::new(0)));
        manager.save_current_trail(5.0);

        manager.clear();

        assert_eq!(manager.num_trails(), 0);
        assert!(manager.current_trail.is_empty());
    }

    #[test]
    fn test_empty_trail_not_saved() {
        let mut manager = TrailSavingManager::new();

        // Try to save empty trail
        manager.save_current_trail(10.0);

        assert_eq!(manager.num_trails(), 0);
    }
}
