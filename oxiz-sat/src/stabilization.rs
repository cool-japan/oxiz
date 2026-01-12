//! Stabilization modes for search strategy
//!
//! This module implements focused and diversification modes to balance
//! exploitation and exploration in the search strategy.

/// Search mode for stabilization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Focused mode: exploit current search direction
    /// - Use saved phases aggressively
    /// - Lower restart intervals
    /// - Keep learned clauses longer
    Focused,

    /// Diversification mode: explore new search space
    /// - Randomize decisions more
    /// - Higher restart intervals
    /// - More aggressive clause deletion
    Diversification,
}

/// Stabilization strategy configuration
#[derive(Debug, Clone)]
pub struct StabilizationConfig {
    /// Number of conflicts in focused mode before switching
    pub focused_conflicts: u64,

    /// Number of conflicts in diversification mode before switching
    pub diversification_conflicts: u64,

    /// Minimum focused period (conflicts)
    pub min_focused: u64,

    /// Maximum focused period (conflicts)
    pub max_focused: u64,

    /// Focused mode phase saving weight (0.0 = random, 1.0 = always use saved)
    pub focused_phase_weight: f64,

    /// Diversification mode phase saving weight
    pub diversification_phase_weight: f64,

    /// Focused mode randomization probability
    pub focused_random_prob: f64,

    /// Diversification mode randomization probability
    pub diversification_random_prob: f64,

    /// Enable dynamic adjustment
    pub dynamic_adjustment: bool,
}

impl StabilizationConfig {
    /// Create default configuration
    pub fn default_config() -> Self {
        Self {
            focused_conflicts: 5000,
            diversification_conflicts: 1000,
            min_focused: 1000,
            max_focused: 100000,
            focused_phase_weight: 0.95,
            diversification_phase_weight: 0.5,
            focused_random_prob: 0.01,
            diversification_random_prob: 0.1,
            dynamic_adjustment: true,
        }
    }

    /// Create aggressive focused configuration
    pub fn aggressive_focused() -> Self {
        Self {
            focused_conflicts: 10000,
            diversification_conflicts: 500,
            min_focused: 5000,
            max_focused: 200000,
            focused_phase_weight: 0.98,
            diversification_phase_weight: 0.3,
            focused_random_prob: 0.005,
            diversification_random_prob: 0.15,
            dynamic_adjustment: true,
        }
    }

    /// Create balanced configuration
    pub fn balanced() -> Self {
        Self::default_config()
    }
}

impl Default for StabilizationConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Stabilization manager
pub struct StabilizationManager {
    /// Current search mode
    mode: SearchMode,

    /// Configuration
    config: StabilizationConfig,

    /// Conflicts in current mode
    conflicts_in_mode: u64,

    /// Total conflicts in focused mode
    total_focused_conflicts: u64,

    /// Total conflicts in diversification mode
    total_diversification_conflicts: u64,

    /// Number of mode switches
    num_switches: u64,

    /// Progress score in current mode
    progress_score: f64,

    /// Best progress score in current focused period
    best_focused_progress: f64,
}

impl StabilizationManager {
    /// Create a new stabilization manager
    pub fn new(config: StabilizationConfig) -> Self {
        Self {
            mode: SearchMode::Focused,
            config,
            conflicts_in_mode: 0,
            total_focused_conflicts: 0,
            total_diversification_conflicts: 0,
            num_switches: 0,
            progress_score: 0.0,
            best_focused_progress: 0.0,
        }
    }

    /// Get current search mode
    pub fn mode(&self) -> SearchMode {
        self.mode
    }

    /// Record a conflict and check if mode should switch
    pub fn on_conflict(&mut self) -> bool {
        self.conflicts_in_mode += 1;

        match self.mode {
            SearchMode::Focused => {
                self.total_focused_conflicts += 1;
                if self.conflicts_in_mode >= self.config.focused_conflicts {
                    self.switch_to_diversification();
                    return true;
                }
            }
            SearchMode::Diversification => {
                self.total_diversification_conflicts += 1;
                if self.conflicts_in_mode >= self.config.diversification_conflicts {
                    self.switch_to_focused();
                    return true;
                }
            }
        }

        false
    }

    /// Switch to focused mode
    fn switch_to_focused(&mut self) {
        self.mode = SearchMode::Focused;
        self.conflicts_in_mode = 0;
        self.num_switches += 1;
        self.best_focused_progress = 0.0;
    }

    /// Switch to diversification mode
    fn switch_to_diversification(&mut self) {
        self.mode = SearchMode::Diversification;
        self.conflicts_in_mode = 0;
        self.num_switches += 1;
    }

    /// Record progress (e.g., based on learned clauses or LBD)
    pub fn record_progress(&mut self, score: f64) {
        self.progress_score = score;

        if self.mode == SearchMode::Focused && score > self.best_focused_progress {
            self.best_focused_progress = score;

            // Dynamically extend focused period if making good progress
            if self.config.dynamic_adjustment {
                let extension = (self.config.focused_conflicts as f64 * 0.1) as u64;
                if self.config.focused_conflicts + extension <= self.config.max_focused {
                    self.config.focused_conflicts += extension;
                }
            }
        }
    }

    /// Get phase saving weight for current mode
    pub fn phase_weight(&self) -> f64 {
        match self.mode {
            SearchMode::Focused => self.config.focused_phase_weight,
            SearchMode::Diversification => self.config.diversification_phase_weight,
        }
    }

    /// Get randomization probability for current mode
    pub fn random_prob(&self) -> f64 {
        match self.mode {
            SearchMode::Focused => self.config.focused_random_prob,
            SearchMode::Diversification => self.config.diversification_random_prob,
        }
    }

    /// Get restart interval multiplier for current mode
    pub fn restart_multiplier(&self) -> f64 {
        match self.mode {
            SearchMode::Focused => 0.8,         // Shorter restarts in focused mode
            SearchMode::Diversification => 1.5, // Longer restarts in diversification
        }
    }

    /// Get clause deletion aggressiveness for current mode
    pub fn deletion_aggressiveness(&self) -> f64 {
        match self.mode {
            SearchMode::Focused => 0.5,         // Keep more clauses in focused mode
            SearchMode::Diversification => 1.5, // Delete more aggressively
        }
    }

    /// Force switch to a specific mode
    pub fn force_mode(&mut self, mode: SearchMode) {
        if self.mode != mode {
            self.mode = mode;
            self.conflicts_in_mode = 0;
            self.num_switches += 1;
        }
    }

    /// Get statistics
    pub fn stats(&self) -> StabilizationStats {
        StabilizationStats {
            current_mode: self.mode,
            conflicts_in_mode: self.conflicts_in_mode,
            total_focused_conflicts: self.total_focused_conflicts,
            total_diversification_conflicts: self.total_diversification_conflicts,
            num_switches: self.num_switches,
            progress_score: self.progress_score,
        }
    }

    /// Reset the manager
    pub fn reset(&mut self) {
        self.mode = SearchMode::Focused;
        self.conflicts_in_mode = 0;
        self.total_focused_conflicts = 0;
        self.total_diversification_conflicts = 0;
        self.num_switches = 0;
        self.progress_score = 0.0;
        self.best_focused_progress = 0.0;
    }
}

impl Default for StabilizationManager {
    fn default() -> Self {
        Self::new(StabilizationConfig::default())
    }
}

/// Statistics for stabilization
#[derive(Debug, Clone)]
pub struct StabilizationStats {
    /// Current mode
    pub current_mode: SearchMode,
    /// Conflicts in current mode
    pub conflicts_in_mode: u64,
    /// Total focused conflicts
    pub total_focused_conflicts: u64,
    /// Total diversification conflicts
    pub total_diversification_conflicts: u64,
    /// Number of mode switches
    pub num_switches: u64,
    /// Current progress score
    pub progress_score: f64,
}

impl StabilizationStats {
    /// Get focused ratio
    pub fn focused_ratio(&self) -> f64 {
        let total = self.total_focused_conflicts + self.total_diversification_conflicts;
        if total == 0 {
            return 0.5;
        }
        self.total_focused_conflicts as f64 / total as f64
    }

    /// Get average conflicts per mode
    pub fn avg_conflicts_per_mode(&self) -> f64 {
        if self.num_switches == 0 {
            return 0.0;
        }
        let total = self.total_focused_conflicts + self.total_diversification_conflicts;
        total as f64 / self.num_switches as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stabilization_basic() {
        let config = StabilizationConfig::default_config();
        let manager = StabilizationManager::new(config);

        assert_eq!(manager.mode(), SearchMode::Focused);
    }

    #[test]
    fn test_mode_switching() {
        let config = StabilizationConfig {
            focused_conflicts: 10,
            diversification_conflicts: 5,
            ..StabilizationConfig::default_config()
        };

        let mut manager = StabilizationManager::new(config);

        // Start in focused mode
        assert_eq!(manager.mode(), SearchMode::Focused);

        // After 10 conflicts, should switch to diversification
        for _ in 0..10 {
            manager.on_conflict();
        }
        assert_eq!(manager.mode(), SearchMode::Diversification);

        // After 5 more conflicts, should switch back to focused
        for _ in 0..5 {
            manager.on_conflict();
        }
        assert_eq!(manager.mode(), SearchMode::Focused);
    }

    #[test]
    fn test_phase_weight() {
        let config = StabilizationConfig::default_config();
        let manager = StabilizationManager::new(config.clone());

        assert_eq!(manager.phase_weight(), config.focused_phase_weight);

        let mut manager = StabilizationManager::new(config.clone());
        manager.force_mode(SearchMode::Diversification);
        assert_eq!(manager.phase_weight(), config.diversification_phase_weight);
    }

    #[test]
    fn test_stats() {
        let mut config = StabilizationConfig::default_config();
        config.focused_conflicts = 5;
        let mut manager = StabilizationManager::new(config);

        for _ in 0..10 {
            manager.on_conflict();
        }

        let stats = manager.stats();
        assert!(stats.num_switches > 0);
        assert!(stats.total_focused_conflicts > 0);
    }

    #[test]
    fn test_progress_tracking() {
        let config = StabilizationConfig::default_config();
        let mut manager = StabilizationManager::new(config);

        manager.record_progress(0.5);
        assert_eq!(manager.progress_score, 0.5);

        manager.record_progress(0.8);
        assert_eq!(manager.progress_score, 0.8);
    }
}
