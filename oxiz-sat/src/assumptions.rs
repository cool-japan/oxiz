//! Enhanced assumption-based solving interface
//!
//! This module provides advanced assumption management for incremental solving,
//! including assumption levels, tracking, and efficient core extraction.

use crate::literal::Lit;
use std::collections::{HashMap, HashSet};

#[cfg(test)]
use crate::literal::Var;

/// Assumption level for hierarchical assumption management
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssumptionLevel(pub u32);

impl AssumptionLevel {
    /// Create a new assumption level
    pub const fn new(level: u32) -> Self {
        Self(level)
    }

    /// Get the root level (0)
    pub const fn root() -> Self {
        Self(0)
    }
}

/// Assumption with metadata
#[derive(Debug, Clone)]
pub struct Assumption {
    /// The assumed literal
    pub lit: Lit,
    /// Level at which this assumption was made
    pub level: AssumptionLevel,
    /// Optional user data
    pub user_data: Option<u64>,
}

impl Assumption {
    /// Create a new assumption
    pub fn new(lit: Lit, level: AssumptionLevel) -> Self {
        Self {
            lit,
            level,
            user_data: None,
        }
    }

    /// Create an assumption with user data
    pub fn with_data(lit: Lit, level: AssumptionLevel, data: u64) -> Self {
        Self {
            lit,
            level,
            user_data: Some(data),
        }
    }
}

/// Assumption stack with level management
pub struct AssumptionStack {
    /// Stack of assumptions
    assumptions: Vec<Assumption>,
    /// Current assumption level
    current_level: AssumptionLevel,
    /// Map from literal to assumption index
    lit_to_index: HashMap<Lit, usize>,
    /// Conflicting assumptions in the last UNSAT result
    conflict_set: Vec<Lit>,
}

impl AssumptionStack {
    /// Create a new assumption stack
    pub fn new() -> Self {
        Self {
            assumptions: Vec::new(),
            current_level: AssumptionLevel::root(),
            lit_to_index: HashMap::new(),
            conflict_set: Vec::new(),
        }
    }

    /// Push an assumption at the current level
    pub fn push(&mut self, lit: Lit) -> Result<(), String> {
        // Check for conflicting assumptions
        if self.lit_to_index.contains_key(&lit.negate()) {
            return Err(format!(
                "Conflicting assumption: {:?} conflicts with existing assumption",
                lit
            ));
        }

        let assumption = Assumption::new(lit, self.current_level);
        self.lit_to_index.insert(lit, self.assumptions.len());
        self.assumptions.push(assumption);
        Ok(())
    }

    /// Push an assumption with user data
    pub fn push_with_data(&mut self, lit: Lit, data: u64) -> Result<(), String> {
        if self.lit_to_index.contains_key(&lit.negate()) {
            return Err(format!(
                "Conflicting assumption: {:?} conflicts with existing assumption",
                lit
            ));
        }

        let assumption = Assumption::with_data(lit, self.current_level, data);
        self.lit_to_index.insert(lit, self.assumptions.len());
        self.assumptions.push(assumption);
        Ok(())
    }

    /// Create a new assumption level
    pub fn new_level(&mut self) {
        self.current_level = AssumptionLevel::new(self.current_level.0 + 1);
    }

    /// Pop all assumptions at the current level and above
    pub fn pop_level(&mut self) {
        if self.current_level.0 == 0 {
            return;
        }

        // Remove assumptions from current level
        self.assumptions.retain(|a| {
            let keep = a.level < self.current_level;
            if !keep {
                self.lit_to_index.remove(&a.lit);
            }
            keep
        });

        // Decrement level
        self.current_level = AssumptionLevel::new(self.current_level.0 - 1);
    }

    /// Clear all assumptions
    pub fn clear(&mut self) {
        self.assumptions.clear();
        self.lit_to_index.clear();
        self.current_level = AssumptionLevel::root();
        self.conflict_set.clear();
    }

    /// Get all assumptions as a slice
    pub fn get_all(&self) -> &[Assumption] {
        &self.assumptions
    }

    /// Get assumptions as literals
    pub fn get_literals(&self) -> Vec<Lit> {
        self.assumptions.iter().map(|a| a.lit).collect()
    }

    /// Get assumptions at a specific level
    pub fn get_at_level(&self, level: AssumptionLevel) -> Vec<&Assumption> {
        self.assumptions
            .iter()
            .filter(|a| a.level == level)
            .collect()
    }

    /// Check if a literal is assumed
    pub fn is_assumed(&self, lit: Lit) -> bool {
        self.lit_to_index.contains_key(&lit)
    }

    /// Get the level of an assumption
    pub fn get_level(&self, lit: Lit) -> Option<AssumptionLevel> {
        self.lit_to_index
            .get(&lit)
            .and_then(|&idx| self.assumptions.get(idx))
            .map(|a| a.level)
    }

    /// Set the conflict set from UNSAT core
    pub fn set_conflict(&mut self, core: Vec<Lit>) {
        self.conflict_set = core;
    }

    /// Get the conflict set
    pub fn get_conflict(&self) -> &[Lit] {
        &self.conflict_set
    }

    /// Get the current level
    pub fn current_level(&self) -> AssumptionLevel {
        self.current_level
    }

    /// Get number of assumptions
    pub fn len(&self) -> usize {
        self.assumptions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.assumptions.is_empty()
    }
}

impl Default for AssumptionStack {
    fn default() -> Self {
        Self::new()
    }
}

/// Assumption core minimizer
pub struct AssumptionCoreMinimizer {
    /// Assumptions that must be in the core
    fixed_assumptions: HashSet<Lit>,
}

impl AssumptionCoreMinimizer {
    /// Create a new core minimizer
    pub fn new() -> Self {
        Self {
            fixed_assumptions: HashSet::new(),
        }
    }

    /// Add a fixed assumption that must be in the core
    pub fn add_fixed(&mut self, lit: Lit) {
        self.fixed_assumptions.insert(lit);
    }

    /// Minimize a core using deletion-based minimization
    pub fn minimize_deletion(&self, core: &[Lit]) -> Vec<Lit> {
        // Simple deletion: try removing each assumption and see if still UNSAT
        // This is a placeholder - actual implementation would need solver access
        let mut minimal = core.to_vec();
        minimal.retain(|&lit| self.fixed_assumptions.contains(&lit));
        minimal
    }

    /// Minimize a core using QuickXplain algorithm
    pub fn minimize_quickxplain(&self, core: &[Lit]) -> Vec<Lit> {
        // QuickXplain is a divide-and-conquer algorithm for core minimization
        // This is a placeholder implementation
        core.to_vec()
    }

    /// Check if a literal is fixed in the core
    pub fn is_fixed(&self, lit: Lit) -> bool {
        self.fixed_assumptions.contains(&lit)
    }
}

impl Default for AssumptionCoreMinimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Assumption-based solving context
pub struct AssumptionContext {
    /// Assumption stack
    stack: AssumptionStack,
    /// Core minimizer
    minimizer: AssumptionCoreMinimizer,
    /// Track assumption usage statistics
    stats: AssumptionStats,
}

impl AssumptionContext {
    /// Create a new assumption context
    pub fn new() -> Self {
        Self {
            stack: AssumptionStack::new(),
            minimizer: AssumptionCoreMinimizer::new(),
            stats: AssumptionStats::default(),
        }
    }

    /// Get the assumption stack
    pub fn stack(&self) -> &AssumptionStack {
        &self.stack
    }

    /// Get mutable assumption stack
    pub fn stack_mut(&mut self) -> &mut AssumptionStack {
        &mut self.stack
    }

    /// Get the core minimizer
    pub fn minimizer(&self) -> &AssumptionCoreMinimizer {
        &self.minimizer
    }

    /// Get mutable core minimizer
    pub fn minimizer_mut(&mut self) -> &mut AssumptionCoreMinimizer {
        &mut self.minimizer
    }

    /// Get statistics
    pub fn stats(&self) -> &AssumptionStats {
        &self.stats
    }

    /// Record a solve with assumptions
    pub fn record_solve(&mut self, num_assumptions: usize, is_sat: bool) {
        self.stats.total_calls += 1;
        self.stats.total_assumptions += num_assumptions;
        if !is_sat {
            self.stats.unsat_calls += 1;
        }
    }

    /// Record a core extraction
    pub fn record_core(&mut self, core_size: usize, original_size: usize) {
        self.stats.core_extractions += 1;
        self.stats.total_core_size += core_size;
        self.stats.total_original_size += original_size;
    }
}

impl Default for AssumptionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for assumption-based solving
#[derive(Debug, Clone, Default)]
pub struct AssumptionStats {
    /// Total number of solve calls with assumptions
    pub total_calls: u64,
    /// Number of UNSAT results
    pub unsat_calls: u64,
    /// Total assumptions across all calls
    pub total_assumptions: usize,
    /// Number of core extractions
    pub core_extractions: u64,
    /// Total core sizes
    pub total_core_size: usize,
    /// Total original assumption set sizes
    pub total_original_size: usize,
}

impl AssumptionStats {
    /// Get average number of assumptions per call
    pub fn avg_assumptions(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.total_assumptions as f64 / self.total_calls as f64
    }

    /// Get average core size
    pub fn avg_core_size(&self) -> f64 {
        if self.core_extractions == 0 {
            return 0.0;
        }
        self.total_core_size as f64 / self.core_extractions as f64
    }

    /// Get average minimization ratio
    pub fn avg_minimization_ratio(&self) -> f64 {
        if self.total_original_size == 0 {
            return 1.0;
        }
        self.total_core_size as f64 / self.total_original_size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assumption_stack_basic() {
        let mut stack = AssumptionStack::new();

        let lit = Lit::pos(Var(0));
        assert!(stack.push(lit).is_ok());
        assert_eq!(stack.len(), 1);
        assert!(stack.is_assumed(lit));
    }

    #[test]
    fn test_assumption_stack_conflict() {
        let mut stack = AssumptionStack::new();

        let lit = Lit::pos(Var(0));
        stack.push(lit).unwrap();

        let neg_lit = Lit::neg(Var(0));
        assert!(stack.push(neg_lit).is_err());
    }

    #[test]
    fn test_assumption_levels() {
        let mut stack = AssumptionStack::new();

        let lit1 = Lit::pos(Var(0));
        stack.push(lit1).unwrap();

        stack.new_level();
        let lit2 = Lit::pos(Var(1));
        stack.push(lit2).unwrap();

        assert_eq!(stack.len(), 2);

        stack.pop_level();
        assert_eq!(stack.len(), 1);
        assert!(stack.is_assumed(lit1));
        assert!(!stack.is_assumed(lit2));
    }

    #[test]
    fn test_assumption_with_data() {
        let mut stack = AssumptionStack::new();

        let lit = Lit::pos(Var(0));
        stack.push_with_data(lit, 42).unwrap();

        assert_eq!(stack.assumptions[0].user_data, Some(42));
    }

    #[test]
    fn test_assumption_context() {
        let mut ctx = AssumptionContext::new();

        ctx.record_solve(5, false);
        ctx.record_core(3, 5);

        assert_eq!(ctx.stats().total_calls, 1);
        assert_eq!(ctx.stats().unsat_calls, 1);
        assert_eq!(ctx.stats().avg_core_size(), 3.0);
    }
}
