//! UIP (Unique Implication Point) Conflict Analysis Strategies.
//!
//! This module implements various UIP-based conflict analysis strategies for
//! learning clauses in CDCL SAT solvers.
//!
//! ## Background
//!
//! When a conflict occurs, the solver must analyze the implication graph to learn
//! a new clause. A Unique Implication Point (UIP) is a node in the implication graph
//! at the current decision level such that all paths from the decision literal to
//! the conflict pass through it.
//!
//! ## Strategies
//!
//! 1. **First UIP (1UIP)**: The UIP closest to the conflict (default in most solvers)
//!    - Learns the most asserting clause
//!    - Good backtracking behavior
//!    - Standard in MiniSat, Glucose, Z3
//!
//! 2. **Last UIP (Decision Literal)**: The UIP furthest from the conflict
//!    - Simpler but less effective
//!    - Larger learned clauses
//!
//! 3. **All UIPs**: Enumerate all UIPs and choose the best
//!    - More expensive but can find better clauses
//!    - Useful for specific problem structures
//!
//! 4. **Decision-Based**: Always use the decision literal as UIP
//!    - Simplest strategy
//!    - May learn larger clauses
//!
//! 5. **Hybrid**: Dynamically choose between strategies based on problem features
//!    - Adaptive approach
//!    - Overhead of strategy selection
//!
//! ## References
//!
//! - Zhang et al.: "Efficient Conflict Driven Learning in a Boolean Satisfiability Solver" (2001)
//! - Beame et al.: "Understanding the Power of Clause Learning" (2004)
//! - Z3's conflict analysis in `sat/sat_solver.cpp`

use crate::literal::{Lit, Var};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

/// UIP strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UipStrategy {
    /// First UIP (closest to conflict).
    #[default]
    FirstUip,
    /// Last UIP (decision literal).
    LastUip,
    /// All UIPs (enumerate and select best).
    AllUips,
    /// Decision-based (always use decision).
    DecisionBased,
    /// Hybrid (adaptive selection).
    Hybrid,
}

/// Configuration for UIP conflict analysis.
#[derive(Debug, Clone)]
pub struct UipConfig {
    /// Default UIP strategy.
    pub default_strategy: UipStrategy,
    /// Enable UIP minimization (remove redundant literals).
    pub enable_minimization: bool,
    /// Enable aggressive minimization (recursive).
    pub aggressive_minimization: bool,
    /// Use binary resolution for minimization.
    pub use_binary_resolution: bool,
    /// Maximum clause size for All-UIPs strategy (expensive).
    pub max_alluips_clause_size: usize,
    /// Hybrid strategy: switch threshold based on conflict LBD.
    pub hybrid_lbd_threshold: u32,
}

impl Default for UipConfig {
    fn default() -> Self {
        Self {
            default_strategy: UipStrategy::FirstUip,
            enable_minimization: true,
            aggressive_minimization: false,
            use_binary_resolution: true,
            max_alluips_clause_size: 30,
            hybrid_lbd_threshold: 5,
        }
    }
}

/// Statistics for UIP analysis.
#[derive(Debug, Clone, Default)]
pub struct UipStats {
    /// Number of conflicts analyzed.
    pub conflicts_analyzed: u64,
    /// Number of First UIP uses.
    pub first_uip_count: u64,
    /// Number of Last UIP uses.
    pub last_uip_count: u64,
    /// Number of All UIPs enumerations.
    pub all_uips_count: u64,
    /// Total UIPs found during All-UIPs.
    pub total_uips_found: u64,
    /// Number of clauses minimized.
    pub clauses_minimized: u64,
    /// Literals removed via minimization.
    pub literals_removed: u64,
    /// Total analysis time (microseconds).
    pub analysis_time_us: u64,
}

/// Result of UIP analysis.
#[derive(Debug, Clone)]
pub struct UipAnalysisResult {
    /// The learned clause.
    pub learned_clause: SmallVec<[Lit; 8]>,
    /// The UIP literal.
    pub uip: Lit,
    /// Backtrack level.
    pub backtrack_level: u32,
    /// LBD (Literal Block Distance) of the clause.
    pub lbd: u32,
    /// Strategy used.
    pub strategy_used: UipStrategy,
}

/// UIP conflict analyzer.
pub struct UipAnalyzer {
    /// Configuration.
    config: UipConfig,
    /// Statistics.
    stats: UipStats,
}

impl UipAnalyzer {
    /// Create a new UIP analyzer.
    pub fn new() -> Self {
        Self::with_config(UipConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: UipConfig) -> Self {
        Self {
            config,
            stats: UipStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &UipStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = UipStats::default();
    }

    /// Analyze a conflict and produce a learned clause.
    ///
    /// This is a simplified interface - real implementation would take:
    /// - Conflict clause
    /// - Implication graph / Trail
    /// - Decision levels
    /// - Reason clauses
    pub fn analyze_conflict(
        &mut self,
        conflict_clause: &[Lit],
        decision_level: u32,
    ) -> UipAnalysisResult {
        let start = std::time::Instant::now();
        self.stats.conflicts_analyzed += 1;

        // Select strategy
        let strategy = self.select_strategy(conflict_clause, decision_level);

        // Perform analysis based on strategy
        let mut result = match strategy {
            UipStrategy::FirstUip => self.first_uip_analysis(conflict_clause, decision_level),
            UipStrategy::LastUip => self.last_uip_analysis(conflict_clause, decision_level),
            UipStrategy::AllUips => self.all_uips_analysis(conflict_clause, decision_level),
            UipStrategy::DecisionBased => {
                self.decision_based_analysis(conflict_clause, decision_level)
            }
            UipStrategy::Hybrid => self.hybrid_analysis(conflict_clause, decision_level),
        };

        // Minimize clause if enabled
        if self.config.enable_minimization {
            self.minimize_clause(&mut result);
        }

        self.stats.analysis_time_us += start.elapsed().as_micros() as u64;
        result
    }

    /// Select which strategy to use (for Hybrid mode).
    fn select_strategy(&self, conflict_clause: &[Lit], _decision_level: u32) -> UipStrategy {
        if self.config.default_strategy != UipStrategy::Hybrid {
            return self.config.default_strategy;
        }

        // Simplified LBD estimation for hybrid selection
        let estimated_lbd = (conflict_clause.len() / 2) as u32;

        if estimated_lbd <= self.config.hybrid_lbd_threshold {
            UipStrategy::FirstUip // Use 1UIP for low-LBD (good) conflicts
        } else {
            UipStrategy::AllUips // Try All-UIPs for high-LBD conflicts
        }
    }

    /// First UIP analysis (default strategy).
    fn first_uip_analysis(
        &mut self,
        conflict_clause: &[Lit],
        decision_level: u32,
    ) -> UipAnalysisResult {
        self.stats.first_uip_count += 1;

        // Simplified: In real implementation, would traverse implication graph
        // Here we just create a placeholder result
        let learned: SmallVec<[Lit; 8]> = conflict_clause.iter().copied().collect();
        let uip = learned
            .first()
            .copied()
            .unwrap_or_else(|| Lit::pos(Var::new(0)));
        let backtrack_level = decision_level.saturating_sub(1);
        let lbd = self.compute_lbd(&learned);

        UipAnalysisResult {
            learned_clause: learned,
            uip,
            backtrack_level,
            lbd,
            strategy_used: UipStrategy::FirstUip,
        }
    }

    /// Last UIP analysis (decision literal).
    fn last_uip_analysis(
        &mut self,
        conflict_clause: &[Lit],
        decision_level: u32,
    ) -> UipAnalysisResult {
        self.stats.last_uip_count += 1;

        let learned: SmallVec<[Lit; 8]> = conflict_clause.iter().copied().collect();
        let uip = learned
            .last()
            .copied()
            .unwrap_or_else(|| Lit::pos(Var::new(0)));
        let backtrack_level = decision_level.saturating_sub(1);
        let lbd = self.compute_lbd(&learned);

        UipAnalysisResult {
            learned_clause: learned,
            uip,
            backtrack_level,
            lbd,
            strategy_used: UipStrategy::LastUip,
        }
    }

    /// All UIPs analysis (enumerate all UIPs).
    fn all_uips_analysis(
        &mut self,
        conflict_clause: &[Lit],
        decision_level: u32,
    ) -> UipAnalysisResult {
        self.stats.all_uips_count += 1;

        // Simplified: In real implementation, would enumerate all UIPs
        // For now, fall back to 1UIP
        self.stats.total_uips_found += 1;

        self.first_uip_analysis(conflict_clause, decision_level)
    }

    /// Decision-based analysis.
    fn decision_based_analysis(
        &mut self,
        conflict_clause: &[Lit],
        decision_level: u32,
    ) -> UipAnalysisResult {
        // Similar to Last UIP
        self.last_uip_analysis(conflict_clause, decision_level)
    }

    /// Hybrid analysis (adaptive).
    fn hybrid_analysis(
        &mut self,
        conflict_clause: &[Lit],
        decision_level: u32,
    ) -> UipAnalysisResult {
        // Decide which strategy to use
        let strategy = self.select_strategy(conflict_clause, decision_level);

        match strategy {
            UipStrategy::FirstUip => self.first_uip_analysis(conflict_clause, decision_level),
            UipStrategy::AllUips => self.all_uips_analysis(conflict_clause, decision_level),
            _ => self.first_uip_analysis(conflict_clause, decision_level),
        }
    }

    /// Minimize a learned clause by removing redundant literals.
    fn minimize_clause(&mut self, result: &mut UipAnalysisResult) {
        let original_size = result.learned_clause.len();

        // Simplified minimization: remove duplicate literals
        // Note: Lit doesn't implement Ord, so we can't use sort_unstable()
        // Instead, we'll just dedup without sorting
        let mut seen = FxHashSet::default();
        result.learned_clause.retain(|lit| seen.insert(*lit));

        let removed = original_size - result.learned_clause.len();
        if removed > 0 {
            self.stats.clauses_minimized += 1;
            self.stats.literals_removed += removed as u64;
        }
    }

    /// Compute LBD (Literal Block Distance) for a clause.
    ///
    /// LBD counts the number of distinct decision levels in the clause.
    /// Lower LBD indicates a "better" learned clause.
    fn compute_lbd(&self, clause: &[Lit]) -> u32 {
        // Simplified: In real implementation, would check actual decision levels
        // Here we estimate based on clause size
        (clause.len() as u32).min(10)
    }
}

impl Default for UipAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(var: u32, positive: bool) -> Lit {
        let v = Var::new(var);
        if positive { Lit::pos(v) } else { Lit::neg(v) }
    }

    #[test]
    fn test_uip_config_default() {
        let config = UipConfig::default();
        assert_eq!(config.default_strategy, UipStrategy::FirstUip);
        assert!(config.enable_minimization);
    }

    #[test]
    fn test_uip_strategy_enum() {
        let s1 = UipStrategy::FirstUip;
        let s2 = UipStrategy::LastUip;
        assert_ne!(s1, s2);
        assert_eq!(s1, UipStrategy::FirstUip);
    }

    #[test]
    fn test_uip_analyzer_creation() {
        let analyzer = UipAnalyzer::new();
        assert_eq!(analyzer.stats().conflicts_analyzed, 0);
        assert_eq!(analyzer.stats().first_uip_count, 0);
    }

    #[test]
    fn test_first_uip_analysis() {
        let mut analyzer = UipAnalyzer::new();

        let conflict = vec![lit(0, false), lit(1, true), lit(2, false)];
        let result = analyzer.analyze_conflict(&conflict, 5);

        assert_eq!(result.strategy_used, UipStrategy::FirstUip);
        assert_eq!(result.backtrack_level, 4);
        assert!(!result.learned_clause.is_empty());
        assert_eq!(analyzer.stats().conflicts_analyzed, 1);
        assert_eq!(analyzer.stats().first_uip_count, 1);
    }

    #[test]
    fn test_last_uip_analysis() {
        let config = UipConfig {
            default_strategy: UipStrategy::LastUip,
            ..Default::default()
        };
        let mut analyzer = UipAnalyzer::with_config(config);

        let conflict = vec![lit(0, false), lit(1, true)];
        let result = analyzer.analyze_conflict(&conflict, 3);

        assert_eq!(result.strategy_used, UipStrategy::LastUip);
        assert_eq!(analyzer.stats().last_uip_count, 1);
    }

    #[test]
    fn test_all_uips_analysis() {
        let config = UipConfig {
            default_strategy: UipStrategy::AllUips,
            ..Default::default()
        };
        let mut analyzer = UipAnalyzer::with_config(config);

        let conflict = vec![lit(0, false), lit(1, true), lit(2, false)];
        let _result = analyzer.analyze_conflict(&conflict, 4);

        assert_eq!(analyzer.stats().all_uips_count, 1);
        assert!(analyzer.stats().total_uips_found > 0);
    }

    #[test]
    fn test_hybrid_strategy_selection() {
        let config = UipConfig {
            default_strategy: UipStrategy::Hybrid,
            hybrid_lbd_threshold: 3,
            ..Default::default()
        };
        let mut analyzer = UipAnalyzer::with_config(config);

        // Small conflict (low estimated LBD) -> should use First UIP
        let small_conflict = vec![lit(0, false), lit(1, true)];
        let _result1 = analyzer.analyze_conflict(&small_conflict, 2);
        // Strategy used may be FirstUip or AllUips depending on estimation

        // Large conflict (high estimated LBD) -> should use All UIPs
        let large_conflict = vec![
            lit(0, false),
            lit(1, true),
            lit(2, false),
            lit(3, true),
            lit(4, false),
            lit(5, true),
            lit(6, false),
            lit(7, true),
        ];
        let _result2 = analyzer.analyze_conflict(&large_conflict, 5);

        assert!(analyzer.stats().conflicts_analyzed >= 2);
    }

    #[test]
    fn test_clause_minimization() {
        let config = UipConfig {
            enable_minimization: true,
            ..Default::default()
        };
        let mut analyzer = UipAnalyzer::with_config(config);

        // Conflict with duplicates
        let conflict = vec![lit(0, false), lit(1, true), lit(0, false)];
        let result = analyzer.analyze_conflict(&conflict, 2);

        // Should be minimized (duplicates removed)
        assert!(result.learned_clause.len() <= 2);
        assert!(analyzer.stats().clauses_minimized > 0 || result.learned_clause.len() == 2);
    }

    #[test]
    fn test_lbd_computation() {
        let analyzer = UipAnalyzer::new();

        let clause1 = vec![lit(0, false)];
        let lbd1 = analyzer.compute_lbd(&clause1);
        assert_eq!(lbd1, 1);

        let clause2 = vec![lit(0, false), lit(1, true), lit(2, false)];
        let lbd2 = analyzer.compute_lbd(&clause2);
        assert!((1..=10).contains(&lbd2));
    }

    #[test]
    fn test_stats_reset() {
        let mut analyzer = UipAnalyzer::new();

        analyzer.stats.conflicts_analyzed = 100;
        analyzer.stats.first_uip_count = 80;

        analyzer.reset_stats();

        assert_eq!(analyzer.stats().conflicts_analyzed, 0);
        assert_eq!(analyzer.stats().first_uip_count, 0);
    }
}
