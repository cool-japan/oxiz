//! Large Neighborhood Search (LNS) for optimization.
//!
//! LNS is a metaheuristic optimization technique that works by:
//! 1. Starting with an initial solution
//! 2. Iteratively destroying part of the solution (using destroy operators)
//! 3. Repairing the destroyed part (using repair operators)
//! 4. Accepting the new solution based on criteria
//! 5. Using restart strategies to escape local optima
//!
//! Reference: Z3's optimization strategies and local search techniques

use crate::maxsat::Weight;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during LNS
#[derive(Error, Debug)]
pub enum LnsError {
    /// No solution found
    #[error("no solution found")]
    NoSolution,
    /// Iteration limit exceeded
    #[error("iteration limit exceeded")]
    IterationLimit,
    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result of LNS optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LnsResult {
    /// Found optimal or near-optimal solution
    Optimal,
    /// Found feasible solution but not proven optimal
    Feasible,
    /// No solution exists
    Unsatisfiable,
    /// Could not determine
    Unknown,
}

/// Configuration for LNS
#[derive(Debug, Clone)]
pub struct LnsConfig {
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Maximum time in seconds (0 = unlimited)
    pub max_time_sec: u32,
    /// Neighborhood destruction ratio (0.0 to 1.0)
    pub destroy_ratio: f64,
    /// Enable adaptive destruction ratio
    pub adaptive_destroy: bool,
    /// Restart threshold (restart after N non-improving iterations)
    pub restart_threshold: u32,
    /// Maximum number of restarts
    pub max_restarts: u32,
    /// Random seed
    pub random_seed: u64,
}

impl Default for LnsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            max_time_sec: 300,  // 5 minutes
            destroy_ratio: 0.3, // Destroy 30% of solution
            adaptive_destroy: true,
            restart_threshold: 100,
            max_restarts: 5,
            random_seed: 42,
        }
    }
}

/// Statistics from LNS
#[derive(Debug, Clone, Default)]
pub struct LnsStats {
    /// Number of iterations performed
    pub iterations: u32,
    /// Number of improvements found
    pub improvements: u32,
    /// Number of restarts performed
    pub restarts: u32,
    /// Number of destroy operations
    pub destroy_ops: u32,
    /// Number of repair operations
    pub repair_ops: u32,
    /// Best cost found
    pub best_cost: Option<Weight>,
}

/// A partial assignment of variables
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Variable assignments (variable_id -> value)
    pub values: HashMap<u32, bool>,
}

impl Assignment {
    /// Create a new empty assignment
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Create from a map
    pub fn from_map(values: HashMap<u32, bool>) -> Self {
        Self { values }
    }

    /// Assign a variable
    pub fn assign(&mut self, var: u32, value: bool) {
        self.values.insert(var, value);
    }

    /// Get assignment for a variable
    pub fn get(&self, var: u32) -> Option<bool> {
        self.values.get(&var).copied()
    }

    /// Remove assignment for a variable
    pub fn unassign(&mut self, var: u32) {
        self.values.remove(&var);
    }

    /// Get all assigned variables
    pub fn assigned_vars(&self) -> HashSet<u32> {
        self.values.keys().copied().collect()
    }

    /// Get number of assigned variables
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clear all assignments
    pub fn clear(&mut self) {
        self.values.clear();
    }
}

impl Default for Assignment {
    fn default() -> Self {
        Self::new()
    }
}

/// Neighborhood operator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborhoodOp {
    /// Random destruction
    Random,
    /// Destroy variables involved in violated constraints
    ViolationBased,
    /// Destroy variables with high impact on objective
    ObjectiveBased,
    /// Destroy a contiguous block of variables
    BlockBased,
    /// Adaptive (choose based on recent performance)
    Adaptive,
}

/// Restart strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Never restart
    Never,
    /// Restart after N non-improving iterations
    Static,
    /// Geometric restart (Luby sequence)
    Luby,
    /// Adaptive restart based on performance
    Adaptive,
}

/// Neighborhood operator
pub struct NeighborhoodOperator {
    /// Operator type
    op_type: NeighborhoodOp,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl NeighborhoodOperator {
    /// Create a new neighborhood operator
    pub fn new(op_type: NeighborhoodOp, seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            op_type,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Destroy part of an assignment
    ///
    /// Returns the set of variables that were unassigned
    pub fn destroy(&mut self, assignment: &mut Assignment, ratio: f64) -> HashSet<u32> {
        match self.op_type {
            NeighborhoodOp::Random => self.destroy_random(assignment, ratio),
            NeighborhoodOp::ViolationBased => self.destroy_violation_based(assignment, ratio),
            NeighborhoodOp::ObjectiveBased => self.destroy_objective_based(assignment, ratio),
            NeighborhoodOp::BlockBased => self.destroy_block_based(assignment, ratio),
            NeighborhoodOp::Adaptive => self.destroy_random(assignment, ratio), // Default to random
        }
    }

    /// Random destruction
    fn destroy_random(&mut self, assignment: &mut Assignment, ratio: f64) -> HashSet<u32> {
        let count = (assignment.len() as f64 * ratio).ceil() as usize;
        let mut destroyed = HashSet::new();

        let vars: Vec<u32> = assignment.assigned_vars().into_iter().collect();
        let mut indices: Vec<usize> = (0..vars.len()).collect();

        // Shuffle and select first `count` variables
        use rand::prelude::SliceRandom;
        indices.shuffle(&mut self.rng);

        for &idx in indices.iter().take(count) {
            let var = vars[idx];
            assignment.unassign(var);
            destroyed.insert(var);
        }

        destroyed
    }

    /// Violation-based destruction (destroy variables in violated constraints)
    fn destroy_violation_based(&mut self, assignment: &mut Assignment, ratio: f64) -> HashSet<u32> {
        // Simplified implementation: random for now
        // In a full implementation, this would analyze constraint violations
        self.destroy_random(assignment, ratio)
    }

    /// Objective-based destruction (destroy high-impact variables)
    fn destroy_objective_based(&mut self, assignment: &mut Assignment, ratio: f64) -> HashSet<u32> {
        // Simplified implementation: random for now
        // In a full implementation, this would analyze objective impact
        self.destroy_random(assignment, ratio)
    }

    /// Block-based destruction (destroy contiguous block)
    fn destroy_block_based(&mut self, assignment: &mut Assignment, ratio: f64) -> HashSet<u32> {
        let count = (assignment.len() as f64 * ratio).ceil() as usize;
        let mut destroyed = HashSet::new();

        let vars: Vec<u32> = assignment.assigned_vars().into_iter().collect();
        if vars.is_empty() {
            return destroyed;
        }

        // Select a random starting position
        let start = self.rng.random_range(0..vars.len());

        // Destroy a contiguous block (wrapping around)
        for i in 0..count.min(vars.len()) {
            let idx = (start + i) % vars.len();
            let var = vars[idx];
            assignment.unassign(var);
            destroyed.insert(var);
        }

        destroyed
    }
}

/// Restart manager
pub struct RestartManager {
    /// Strategy
    strategy: RestartStrategy,
    /// Threshold for static restart
    threshold: u32,
    /// Current non-improving iterations
    non_improving: u32,
    /// Restart count
    restart_count: u32,
    /// Luby sequence index
    luby_index: u32,
}

impl RestartManager {
    /// Create a new restart manager
    pub fn new(strategy: RestartStrategy, threshold: u32) -> Self {
        Self {
            strategy,
            threshold,
            non_improving: 0,
            restart_count: 0,
            luby_index: 0,
        }
    }

    /// Record an iteration (with or without improvement)
    pub fn record_iteration(&mut self, improved: bool) {
        if improved {
            self.non_improving = 0;
        } else {
            self.non_improving += 1;
        }
    }

    /// Check if restart should occur
    pub fn should_restart(&mut self) -> bool {
        match self.strategy {
            RestartStrategy::Never => false,
            RestartStrategy::Static => {
                if self.non_improving >= self.threshold {
                    self.non_improving = 0;
                    self.restart_count += 1;
                    true
                } else {
                    false
                }
            }
            RestartStrategy::Luby => {
                let luby_value = self.luby_sequence(self.luby_index);
                if self.non_improving >= luby_value * self.threshold {
                    self.non_improving = 0;
                    self.restart_count += 1;
                    self.luby_index += 1;
                    true
                } else {
                    false
                }
            }
            RestartStrategy::Adaptive => {
                // Simple adaptive: exponential backoff
                let threshold = self.threshold * (1 << self.restart_count.min(10));
                if self.non_improving >= threshold {
                    self.non_improving = 0;
                    self.restart_count += 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Get restart count
    pub fn restart_count(&self) -> u32 {
        self.restart_count
    }

    /// Reset restart manager
    pub fn reset(&mut self) {
        self.non_improving = 0;
        self.restart_count = 0;
        self.luby_index = 0;
    }

    /// Compute Luby sequence value
    ///
    /// The Luby sequence is: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
    fn luby_sequence(&self, index: u32) -> u32 {
        // Standard Luby sequence implementation
        // L(1) = 1, L(n) = 2^(k-1) if n = 2^k - 1, else L(n - 2^(k-1) + 1)
        let mut i = index;
        let mut size = 1u32;

        // Find the size of the sequence block containing index i
        while size < i + 1 {
            size = 2 * size + 1;
        }

        // Navigate down to find the actual value
        while size - 1 != i {
            size = (size - 1) >> 1;
            if i >= size {
                i -= size;
            }
        }

        (size + 1) >> 1
    }
}

/// LNS solver
pub struct LnsSolver {
    /// Configuration
    config: LnsConfig,
    /// Statistics
    stats: LnsStats,
    /// Current best assignment
    best_assignment: Option<Assignment>,
    /// Neighborhood operator
    operator: NeighborhoodOperator,
    /// Restart manager
    restart_manager: RestartManager,
}

impl LnsSolver {
    /// Create a new LNS solver
    pub fn new() -> Self {
        Self::with_config(LnsConfig::default())
    }

    /// Create a new LNS solver with configuration
    pub fn with_config(config: LnsConfig) -> Self {
        let operator = NeighborhoodOperator::new(NeighborhoodOp::Random, config.random_seed);
        let restart_manager =
            RestartManager::new(RestartStrategy::Static, config.restart_threshold);

        Self {
            config,
            stats: LnsStats::default(),
            best_assignment: None,
            operator,
            restart_manager,
        }
    }

    /// Set neighborhood operator type
    pub fn set_operator(&mut self, op_type: NeighborhoodOp) {
        self.operator = NeighborhoodOperator::new(op_type, self.config.random_seed);
    }

    /// Set restart strategy
    pub fn set_restart_strategy(&mut self, strategy: RestartStrategy) {
        self.restart_manager = RestartManager::new(strategy, self.config.restart_threshold);
    }

    /// Initialize with an assignment
    pub fn initialize(&mut self, initial: Assignment) {
        self.best_assignment = Some(initial);
        self.stats = LnsStats::default();
    }

    /// Perform one LNS iteration
    ///
    /// Returns true if an improvement was found
    pub fn iterate(&mut self) -> Result<bool, LnsError> {
        if self.best_assignment.is_none() {
            return Err(LnsError::NoSolution);
        }

        self.stats.iterations += 1;

        // Get current assignment
        let mut current = self
            .best_assignment
            .as_ref()
            .expect("best_assignment set after initial solve")
            .clone();

        // Destroy part of the solution
        let destroy_ratio = if self.config.adaptive_destroy {
            // Adaptive: increase ratio if stuck
            let base = self.config.destroy_ratio;
            let stuck_factor = (self.restart_manager.non_improving as f64 / 100.0).min(0.3);
            (base + stuck_factor).min(0.9)
        } else {
            self.config.destroy_ratio
        };

        let _destroyed = self.operator.destroy(&mut current, destroy_ratio);
        self.stats.destroy_ops += 1;

        // Repair would be done here using an SMT solver
        // For now, we just record the operation
        self.stats.repair_ops += 1;

        // Check if this is an improvement
        // In a real implementation, we'd evaluate the cost
        let improved = false; // Placeholder

        self.restart_manager.record_iteration(improved);

        if improved {
            self.best_assignment = Some(current);
            self.stats.improvements += 1;
        }

        // Check for restart
        if self.restart_manager.should_restart()
            && self.restart_manager.restart_count() <= self.config.max_restarts
        {
            self.stats.restarts += 1;
            // Restart: randomize or use a different initial solution
        }

        Ok(improved)
    }

    /// Get the best assignment found
    pub fn best_assignment(&self) -> Option<&Assignment> {
        self.best_assignment.as_ref()
    }

    /// Get statistics
    pub fn stats(&self) -> &LnsStats {
        &self.stats
    }

    /// Check if iteration limit reached
    pub fn at_iteration_limit(&self) -> bool {
        self.stats.iterations >= self.config.max_iterations
    }

    /// Check if restart limit reached
    pub fn at_restart_limit(&self) -> bool {
        self.stats.restarts >= self.config.max_restarts
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.stats = LnsStats::default();
        self.best_assignment = None;
        self.restart_manager.reset();
    }
}

impl Default for LnsSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assignment() {
        let mut assignment = Assignment::new();
        assert!(assignment.is_empty());

        assignment.assign(1, true);
        assignment.assign(2, false);
        assignment.assign(3, true);

        assert_eq!(assignment.len(), 3);
        assert_eq!(assignment.get(1), Some(true));
        assert_eq!(assignment.get(2), Some(false));
        assert_eq!(assignment.get(3), Some(true));
        assert_eq!(assignment.get(4), None);

        assignment.unassign(2);
        assert_eq!(assignment.len(), 2);
        assert_eq!(assignment.get(2), None);
    }

    #[test]
    fn test_destroy_random() {
        let mut operator = NeighborhoodOperator::new(NeighborhoodOp::Random, 42);
        let mut assignment = Assignment::new();

        for i in 1..=10 {
            assignment.assign(i, i % 2 == 0);
        }

        let destroyed = operator.destroy(&mut assignment, 0.3);
        assert_eq!(destroyed.len(), 3); // 30% of 10
        assert_eq!(assignment.len(), 7); // 7 remaining
    }

    #[test]
    fn test_destroy_block() {
        let mut operator = NeighborhoodOperator::new(NeighborhoodOp::BlockBased, 42);
        let mut assignment = Assignment::new();

        for i in 1..=10 {
            assignment.assign(i, true);
        }

        let destroyed = operator.destroy(&mut assignment, 0.4);
        assert_eq!(destroyed.len(), 4); // 40% of 10
        assert_eq!(assignment.len(), 6);
    }

    #[test]
    fn test_restart_manager_static() {
        let mut manager = RestartManager::new(RestartStrategy::Static, 5);

        for _ in 0..4 {
            manager.record_iteration(false);
            assert!(!manager.should_restart());
        }

        manager.record_iteration(false);
        assert!(manager.should_restart()); // 5th non-improving iteration
        assert_eq!(manager.restart_count(), 1);
    }

    #[test]
    fn test_restart_manager_with_improvement() {
        let mut manager = RestartManager::new(RestartStrategy::Static, 5);

        manager.record_iteration(false);
        manager.record_iteration(false);
        manager.record_iteration(true); // Improvement resets counter
        manager.record_iteration(false);
        manager.record_iteration(false);

        assert!(!manager.should_restart()); // Only 2 since last improvement
    }

    #[test]
    fn test_luby_sequence() {
        let manager = RestartManager::new(RestartStrategy::Luby, 1);

        // First few Luby values: 1, 1, 2, 1, 1, 2, 4, 1, ...
        assert_eq!(manager.luby_sequence(0), 1);
        assert_eq!(manager.luby_sequence(1), 1);
        assert_eq!(manager.luby_sequence(2), 2);
        assert_eq!(manager.luby_sequence(3), 1);
        assert_eq!(manager.luby_sequence(4), 1);
        assert_eq!(manager.luby_sequence(5), 2);
        assert_eq!(manager.luby_sequence(6), 4);
        assert_eq!(manager.luby_sequence(7), 1);
    }

    #[test]
    fn test_lns_solver_basic() {
        let mut solver = LnsSolver::new();

        let mut initial = Assignment::new();
        for i in 1..=5 {
            initial.assign(i, true);
        }

        solver.initialize(initial);
        assert!(solver.best_assignment().is_some());
        assert_eq!(solver.best_assignment().unwrap().len(), 5);
    }

    #[test]
    fn test_lns_iterations() {
        let config = LnsConfig {
            max_iterations: 10,
            destroy_ratio: 0.3,
            ..Default::default()
        };

        let mut solver = LnsSolver::with_config(config);

        let mut initial = Assignment::new();
        for i in 1..=10 {
            initial.assign(i, true);
        }

        solver.initialize(initial);

        for _ in 0..5 {
            let _ = solver.iterate();
        }

        assert_eq!(solver.stats().iterations, 5);
        assert_eq!(solver.stats().destroy_ops, 5);
        assert_eq!(solver.stats().repair_ops, 5);
    }

    #[test]
    fn test_adaptive_destroy_ratio() {
        let config = LnsConfig {
            adaptive_destroy: true,
            destroy_ratio: 0.2,
            ..Default::default()
        };

        let solver = LnsSolver::with_config(config);
        assert!(solver.config.adaptive_destroy);
    }

    #[test]
    fn test_lns_limits() {
        let config = LnsConfig {
            max_iterations: 5,
            max_restarts: 2,
            ..Default::default()
        };

        let mut solver = LnsSolver::with_config(config);
        let initial = Assignment::new();
        solver.initialize(initial);

        for _ in 0..10 {
            if solver.at_iteration_limit() {
                break;
            }
            let _ = solver.iterate();
        }

        assert!(solver.at_iteration_limit());
        assert_eq!(solver.stats().iterations, 5);
    }
}
