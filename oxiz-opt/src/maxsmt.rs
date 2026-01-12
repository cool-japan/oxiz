//! MaxSMT solver.
//!
//! This module implements MaxSMT (Maximum Satisfiability Modulo Theories),
//! which extends MaxSAT to SMT formulas. It handles:
//! - Soft constraints with weights over SMT formulas
//! - Integration with theory solvers
//! - Incremental optimization
//!
//! Reference: Z3's `opt/maxsmt.cpp`

use crate::maxsat::{MaxSatConfig, MaxSatError, MaxSatResult, Weight};
use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use thiserror::Error;

/// Errors that can occur during MaxSMT solving
#[derive(Error, Debug)]
pub enum MaxSmtError {
    /// Hard constraints unsatisfiable
    #[error("hard constraints unsatisfiable")]
    Unsatisfiable,
    /// MaxSAT level error
    #[error("maxsat error: {0}")]
    MaxSat(#[from] MaxSatError),
    /// Theory conflict
    #[error("theory conflict")]
    TheoryConflict,
    /// Resource limit
    #[error("resource limit")]
    ResourceLimit,
}

/// Result of MaxSMT solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaxSmtResult {
    /// Optimal solution found
    Optimal,
    /// Solution found but optimality not proven
    Satisfiable,
    /// No solution exists
    Unsatisfiable,
    /// Could not determine
    Unknown,
}

impl From<MaxSatResult> for MaxSmtResult {
    fn from(r: MaxSatResult) -> Self {
        match r {
            MaxSatResult::Optimal => MaxSmtResult::Optimal,
            MaxSatResult::Satisfiable => MaxSmtResult::Satisfiable,
            MaxSatResult::Unsatisfiable => MaxSmtResult::Unsatisfiable,
            MaxSatResult::Unknown => MaxSmtResult::Unknown,
        }
    }
}

/// Unique identifier for a soft SMT constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SoftSmtId(pub u32);

impl SoftSmtId {
    /// Create a new soft SMT ID
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// A soft SMT constraint
#[derive(Debug, Clone)]
pub struct SoftSmtConstraint {
    /// Unique identifier
    pub id: SoftSmtId,
    /// The term representing the constraint
    pub term: TermId,
    /// Weight of this soft constraint
    pub weight: Weight,
    /// Whether this constraint is currently satisfied
    satisfied: bool,
}

impl SoftSmtConstraint {
    /// Create a new soft SMT constraint
    pub fn new(id: SoftSmtId, term: TermId, weight: Weight) -> Self {
        Self {
            id,
            term,
            weight,
            satisfied: false,
        }
    }

    /// Check if satisfied
    pub fn is_satisfied(&self) -> bool {
        self.satisfied
    }

    /// Set satisfaction status
    pub fn set_satisfied(&mut self, satisfied: bool) {
        self.satisfied = satisfied;
    }
}

/// Configuration for MaxSMT solver
#[derive(Debug, Clone)]
pub struct MaxSmtConfig {
    /// Underlying MaxSAT configuration
    pub maxsat: MaxSatConfig,
    /// Enable theory-aware optimization
    pub theory_aware: bool,
    /// Maximum iterations
    pub max_iterations: u32,
}

impl Default for MaxSmtConfig {
    fn default() -> Self {
        Self {
            maxsat: MaxSatConfig::default(),
            theory_aware: true,
            max_iterations: 100000,
        }
    }
}

/// Statistics from MaxSMT solving
#[derive(Debug, Clone, Default)]
pub struct MaxSmtStats {
    /// Number of SMT solver calls
    pub smt_calls: u32,
    /// Number of cores extracted
    pub cores_extracted: u32,
    /// Number of theory propagations
    pub theory_propagations: u32,
}

/// MaxSMT solver
///
/// This solver handles optimization of soft SMT constraints.
/// It uses a core-guided approach similar to MaxSAT but
/// integrates with theory solvers for SMT-level reasoning.
#[derive(Debug)]
pub struct MaxSmtSolver {
    /// Hard constraints (must be satisfied)
    hard_constraints: Vec<TermId>,
    /// Soft constraints with weights
    soft_constraints: Vec<SoftSmtConstraint>,
    /// Next soft ID
    next_soft_id: u32,
    /// Configuration
    #[allow(dead_code)]
    config: MaxSmtConfig,
    /// Statistics
    stats: MaxSmtStats,
    /// Current lower bound
    lower_bound: Weight,
    /// Current upper bound
    upper_bound: Weight,
    /// Term to Boolean variable mapping (for SAT encoding)
    term_to_var: FxHashMap<TermId, u32>,
    /// Next Boolean variable
    next_var: u32,
    /// Soft constraint groups (for stratified solving)
    groups: FxHashMap<String, Vec<SoftSmtId>>,
}

impl Default for MaxSmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxSmtSolver {
    /// Create a new MaxSMT solver
    pub fn new() -> Self {
        Self::with_config(MaxSmtConfig::default())
    }

    /// Create a new MaxSMT solver with configuration
    pub fn with_config(config: MaxSmtConfig) -> Self {
        Self {
            hard_constraints: Vec::new(),
            soft_constraints: Vec::new(),
            next_soft_id: 0,
            config,
            stats: MaxSmtStats::default(),
            lower_bound: Weight::zero(),
            upper_bound: Weight::Infinite,
            term_to_var: FxHashMap::default(),
            next_var: 0,
            groups: FxHashMap::default(),
        }
    }

    /// Add a hard constraint
    pub fn add_hard(&mut self, term: TermId) {
        self.hard_constraints.push(term);
    }

    /// Add a soft constraint with unit weight
    pub fn add_soft(&mut self, term: TermId) -> SoftSmtId {
        self.add_soft_weighted(term, Weight::one())
    }

    /// Add a soft constraint with weight
    pub fn add_soft_weighted(&mut self, term: TermId, weight: Weight) -> SoftSmtId {
        let id = SoftSmtId(self.next_soft_id);
        self.next_soft_id += 1;

        let constraint = SoftSmtConstraint::new(id, term, weight.clone());
        self.soft_constraints.push(constraint);

        // Update upper bound
        self.upper_bound = self.upper_bound.add(&weight);

        id
    }

    /// Add a soft constraint to a group
    pub fn add_soft_to_group(&mut self, term: TermId, weight: Weight, group: &str) -> SoftSmtId {
        let id = self.add_soft_weighted(term, weight);
        self.groups.entry(group.to_string()).or_default().push(id);
        id
    }

    /// Get the number of hard constraints
    pub fn num_hard(&self) -> usize {
        self.hard_constraints.len()
    }

    /// Get the number of soft constraints
    pub fn num_soft(&self) -> usize {
        self.soft_constraints.len()
    }

    /// Get the lower bound
    pub fn lower_bound(&self) -> &Weight {
        &self.lower_bound
    }

    /// Get the upper bound
    pub fn upper_bound(&self) -> &Weight {
        &self.upper_bound
    }

    /// Get statistics
    pub fn stats(&self) -> &MaxSmtStats {
        &self.stats
    }

    /// Get the cost (sum of weights of unsatisfied soft constraints)
    pub fn cost(&self) -> Weight {
        self.soft_constraints
            .iter()
            .filter(|c| !c.is_satisfied())
            .fold(Weight::zero(), |acc, c| acc.add(&c.weight))
    }

    /// Allocate a Boolean variable for a term
    #[allow(dead_code)]
    fn allocate_var(&mut self, term: TermId) -> u32 {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }
        let var = self.next_var;
        self.next_var += 1;
        self.term_to_var.insert(term, var);
        var
    }

    /// Solve the MaxSMT problem
    ///
    /// This is the main entry point for MaxSMT optimization.
    pub fn solve(&mut self) -> Result<MaxSmtResult, MaxSmtError> {
        // Trivial case: no soft constraints
        if self.soft_constraints.is_empty() {
            return self.check_hard_satisfiable();
        }

        // Check if hard constraints alone are satisfiable
        match self.check_hard_satisfiable()? {
            MaxSmtResult::Unsatisfiable => return Err(MaxSmtError::Unsatisfiable),
            MaxSmtResult::Unknown => return Ok(MaxSmtResult::Unknown),
            _ => {}
        }

        // Use stratified solving if we have groups with different weights
        if self.has_different_weights() {
            return self.solve_stratified();
        }

        // Use core-guided approach
        self.solve_core_guided()
    }

    /// Check if hard constraints are satisfiable
    fn check_hard_satisfiable(&mut self) -> Result<MaxSmtResult, MaxSmtError> {
        // This would integrate with the actual SMT solver
        // For now, return Unknown as placeholder
        self.stats.smt_calls += 1;
        Ok(MaxSmtResult::Unknown)
    }

    /// Check if weights differ
    fn has_different_weights(&self) -> bool {
        if self.soft_constraints.is_empty() {
            return false;
        }
        let first = &self.soft_constraints[0].weight;
        self.soft_constraints.iter().any(|c| &c.weight != first)
    }

    /// Solve using stratified approach
    fn solve_stratified(&mut self) -> Result<MaxSmtResult, MaxSmtError> {
        // Collect unique weight levels
        let mut weight_levels: Vec<Weight> = self
            .soft_constraints
            .iter()
            .map(|c| c.weight.clone())
            .collect();
        weight_levels.sort();
        weight_levels.dedup();
        weight_levels.reverse(); // Highest weight first

        // Solve for each level
        for level in weight_levels {
            let active_ids: Vec<SoftSmtId> = self
                .soft_constraints
                .iter()
                .filter(|c| c.weight >= level)
                .map(|c| c.id)
                .collect();

            if !active_ids.is_empty() {
                self.stats.smt_calls += 1;
            }
        }

        Ok(MaxSmtResult::Unknown)
    }

    /// Solve using core-guided approach
    fn solve_core_guided(&mut self) -> Result<MaxSmtResult, MaxSmtError> {
        // This would implement the full core-guided MaxSMT algorithm
        // Similar to MaxSAT Fu-Malik but with SMT integration
        // Placeholder - would integrate with SMT solver
        // 1. Check satisfiability with current assumptions
        // 2. If SAT, we have a candidate solution
        // 3. If UNSAT, extract core and relax

        self.stats.smt_calls += 1;

        Ok(MaxSmtResult::Unknown)
    }

    /// Get satisfied soft constraint IDs
    pub fn satisfied(&self) -> impl Iterator<Item = SoftSmtId> + '_ {
        self.soft_constraints
            .iter()
            .filter(|c| c.is_satisfied())
            .map(|c| c.id)
    }

    /// Get unsatisfied soft constraint IDs
    pub fn unsatisfied(&self) -> impl Iterator<Item = SoftSmtId> + '_ {
        self.soft_constraints
            .iter()
            .filter(|c| !c.is_satisfied())
            .map(|c| c.id)
    }

    /// Get the weight of a soft constraint
    pub fn weight(&self, id: SoftSmtId) -> Option<&Weight> {
        self.soft_constraints.get(id.0 as usize).map(|c| &c.weight)
    }

    /// Check if a soft constraint is satisfied
    pub fn is_satisfied(&self, id: SoftSmtId) -> bool {
        self.soft_constraints
            .get(id.0 as usize)
            .is_some_and(|c| c.is_satisfied())
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.hard_constraints.clear();
        self.soft_constraints.clear();
        self.next_soft_id = 0;
        self.stats = MaxSmtStats::default();
        self.lower_bound = Weight::zero();
        self.upper_bound = Weight::Infinite;
        self.term_to_var.clear();
        self.next_var = 0;
        self.groups.clear();
    }
}

/// Theory-aware core extraction
///
/// Represents a core extracted from an unsatisfiable SMT query,
/// including both Boolean and theory-level information.
#[derive(Debug, Clone)]
pub struct SmtCore {
    /// Soft constraint IDs in this core
    pub soft_ids: SmallVec<[SoftSmtId; 8]>,
    /// Theory lemmas involved in the conflict
    pub theory_lemmas: Vec<TermId>,
    /// Minimum weight in the core
    pub min_weight: Weight,
}

impl SmtCore {
    /// Create a new SMT core
    pub fn new(soft_ids: impl IntoIterator<Item = SoftSmtId>) -> Self {
        let ids: SmallVec<[SoftSmtId; 8]> = soft_ids.into_iter().collect();
        Self {
            soft_ids: ids,
            theory_lemmas: Vec::new(),
            min_weight: Weight::Infinite,
        }
    }

    /// Get the size of this core
    pub fn size(&self) -> usize {
        self.soft_ids.len()
    }

    /// Check if this core is empty
    pub fn is_empty(&self) -> bool {
        self.soft_ids.is_empty()
    }

    /// Add a theory lemma
    pub fn add_lemma(&mut self, lemma: TermId) {
        self.theory_lemmas.push(lemma);
    }

    /// Set the minimum weight
    pub fn set_min_weight(&mut self, weight: Weight) {
        self.min_weight = weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_smt_id() {
        let id = SoftSmtId::new(42);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn test_soft_constraint() {
        let id = SoftSmtId::new(0);
        let term = TermId::from(1);
        let mut constraint = SoftSmtConstraint::new(id, term, Weight::from(5));

        assert_eq!(constraint.id, id);
        assert_eq!(constraint.term, term);
        assert_eq!(constraint.weight, Weight::from(5));
        assert!(!constraint.is_satisfied());

        constraint.set_satisfied(true);
        assert!(constraint.is_satisfied());
    }

    #[test]
    fn test_maxsmt_solver_new() {
        let solver = MaxSmtSolver::new();
        assert_eq!(solver.num_hard(), 0);
        assert_eq!(solver.num_soft(), 0);
    }

    #[test]
    fn test_add_hard() {
        let mut solver = MaxSmtSolver::new();
        solver.add_hard(TermId::from(1));
        solver.add_hard(TermId::from(2));
        assert_eq!(solver.num_hard(), 2);
    }

    #[test]
    fn test_add_soft() {
        let mut solver = MaxSmtSolver::new();
        let id1 = solver.add_soft(TermId::from(1));
        let id2 = solver.add_soft_weighted(TermId::from(2), Weight::from(5));

        assert_eq!(id1.raw(), 0);
        assert_eq!(id2.raw(), 1);
        assert_eq!(solver.num_soft(), 2);

        assert_eq!(solver.weight(id1), Some(&Weight::one()));
        assert_eq!(solver.weight(id2), Some(&Weight::from(5)));
    }

    #[test]
    fn test_groups() {
        let mut solver = MaxSmtSolver::new();
        solver.add_soft_to_group(TermId::from(1), Weight::one(), "g1");
        solver.add_soft_to_group(TermId::from(2), Weight::one(), "g1");
        solver.add_soft_to_group(TermId::from(3), Weight::one(), "g2");

        assert_eq!(solver.groups.get("g1").map(|v| v.len()), Some(2));
        assert_eq!(solver.groups.get("g2").map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_cost() {
        let mut solver = MaxSmtSolver::new();
        solver.add_soft_weighted(TermId::from(1), Weight::from(3));
        solver.add_soft_weighted(TermId::from(2), Weight::from(5));

        // All unsatisfied initially
        assert_eq!(solver.cost(), Weight::from(8));

        // Mark one as satisfied
        solver.soft_constraints[0].set_satisfied(true);
        assert_eq!(solver.cost(), Weight::from(5));

        // Mark both as satisfied
        solver.soft_constraints[1].set_satisfied(true);
        assert_eq!(solver.cost(), Weight::zero());
    }

    #[test]
    fn test_reset() {
        let mut solver = MaxSmtSolver::new();
        solver.add_hard(TermId::from(1));
        solver.add_soft(TermId::from(2));

        solver.reset();

        assert_eq!(solver.num_hard(), 0);
        assert_eq!(solver.num_soft(), 0);
    }

    #[test]
    fn test_smt_core() {
        let mut core = SmtCore::new([SoftSmtId(0), SoftSmtId(1), SoftSmtId(2)]);

        assert_eq!(core.size(), 3);
        assert!(!core.is_empty());

        core.add_lemma(TermId::from(10));
        core.set_min_weight(Weight::from(5));

        assert_eq!(core.theory_lemmas.len(), 1);
        assert_eq!(core.min_weight, Weight::from(5));
    }

    #[test]
    fn test_maxsmt_result_from_maxsat() {
        assert_eq!(
            MaxSmtResult::from(MaxSatResult::Optimal),
            MaxSmtResult::Optimal
        );
        assert_eq!(
            MaxSmtResult::from(MaxSatResult::Satisfiable),
            MaxSmtResult::Satisfiable
        );
        assert_eq!(
            MaxSmtResult::from(MaxSatResult::Unsatisfiable),
            MaxSmtResult::Unsatisfiable
        );
        assert_eq!(
            MaxSmtResult::from(MaxSatResult::Unknown),
            MaxSmtResult::Unknown
        );
    }
}
