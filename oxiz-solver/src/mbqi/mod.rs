//! Model-Based Quantifier Instantiation (MBQI)
//!
//! This module implements a comprehensive MBQI system for handling quantified formulas
//! in SMT solving. MBQI is a powerful technique that uses candidate models to guide
//! the instantiation of quantified formulas.
//!
//! # Algorithm Overview
//!
//! Model-Based Quantifier Instantiation works by:
//!
//! 1. **Model Extraction**: Get a partial model M from the SAT solver
//! 2. **Model Completion**: Complete the partial model to handle all function interpretations
//! 3. **Quantifier Specialization**: For each universal quantifier ∀x.φ(x), specialize φ with M
//! 4. **Counterexample Search**: Look for assignments that falsify φ under M
//! 5. **Instantiation**: If counterexample x' found where ¬φ(x') holds, add φ(x') as lemma
//! 6. **Refinement**: Repeat until no counterexamples exist or resource limits reached
//!
//! # Key Features
//!
//! - **Finite Model Finding**: Restricts infinite domains to finite universes
//! - **Model Completion**: Completes partial models using macro solving
//! - **Projection Functions**: Maps values to representative terms
//! - **Lazy Instantiation**: Generates instances on-demand
//! - **Conflict-Driven Learning**: Learns from failed instantiation attempts
//! - **Symmetry Breaking**: Reduces redundant search in symmetric problems
//!
//! # References
//!
//! - Ge, Y., & de Moura, L. (2009). "Complete instantiation for quantified formulas
//!   in satisfiability modulo theories." CAV 2009.
//! - Reynolds, A., et al. (2013). "Quantifier instantiation techniques for finite
//!   model finding in SMT." CADE 2013.
//!
//! # Module Organization
//!
//! - `model_completion`: Algorithms for completing partial models
//! - `counterexample`: Counter-example generation and refinement
//! - `instantiation`: Model-based instantiation logic
//! - `finite_model`: Finite model finder with bounded enumeration
//! - `lazy_instantiation`: Lazy quantifier instantiation strategies
//! - `integration`: Integration with the main solver
//! - `heuristics`: MBQI-specific heuristics and strategies
//! - `macros`: Macro definitions and utilities

#![allow(missing_docs)]
#![allow(dead_code)]

use lasso::Spur;
use oxiz_core::ast::TermId;
use oxiz_core::sort::SortId;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::fmt;

pub mod counterexample;
pub mod finite_model;
pub mod heuristics;
pub mod instantiation;
pub mod integration;
pub mod lazy_instantiation;
pub mod macros;
pub mod model_completion;
pub mod patterns;

// Re-export key types
pub use counterexample::{CounterExample, CounterExampleGenerator, RefinementStrategy};
pub use finite_model::{FiniteModel, FiniteModelFinder, SymmetryBreaker, UniverseSize};
pub use heuristics::{InstantiationHeuristic, MBQIHeuristics, SelectionStrategy, TriggerSelection};
pub use instantiation::{
    InstantiationContext, InstantiationEngine, InstantiationPattern, QuantifierInstantiator,
};
pub use integration::{MBQIIntegration, SolverCallback};
pub use lazy_instantiation::{LazyInstantiator, LazyStrategy, MatchingContext};
pub use model_completion::{
    MacroSolver, ModelCompleter, ModelFixer, ProjectionFunction, UninterpretedSortHandler,
};

/// A quantified formula tracked by MBQI
#[derive(Debug, Clone)]
pub struct QuantifiedFormula {
    /// The original quantified term
    pub term: TermId,
    /// Bound variables (name, sort)
    pub bound_vars: SmallVec<[(Spur, SortId); 4]>,
    /// The body of the quantifier
    pub body: TermId,
    /// Whether this is universal (true) or existential (false)
    pub is_universal: bool,
    /// Quantifier nesting depth
    pub nesting_depth: u32,
    /// Number of times this quantifier has been instantiated
    pub instantiation_count: usize,
    /// Maximum allowed instantiations for this quantifier
    pub max_instantiations: usize,
    /// Generation number (for tracking term age)
    pub generation: u32,
    /// Weight for prioritization
    pub weight: f64,
    /// Patterns for E-matching (if available)
    pub patterns: Vec<Vec<TermId>>,
    /// Whether this quantifier has been flattened
    pub is_flattened: bool,
    /// Cached specialized body (None if not yet specialized)
    pub specialized_body: Option<TermId>,
}

impl QuantifiedFormula {
    /// Create a new tracked quantified formula
    pub fn new(
        term: TermId,
        bound_vars: SmallVec<[(Spur, SortId); 4]>,
        body: TermId,
        is_universal: bool,
    ) -> Self {
        Self {
            term,
            bound_vars,
            body,
            is_universal,
            nesting_depth: 0,
            instantiation_count: 0,
            max_instantiations: 1000,
            generation: 0,
            weight: 1.0,
            patterns: Vec::new(),
            is_flattened: false,
            specialized_body: None,
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        term: TermId,
        bound_vars: SmallVec<[(Spur, SortId); 4]>,
        body: TermId,
        is_universal: bool,
        max_instantiations: usize,
        weight: f64,
    ) -> Self {
        let mut qf = Self::new(term, bound_vars, body, is_universal);
        qf.max_instantiations = max_instantiations;
        qf.weight = weight;
        qf
    }

    /// Check if we can instantiate more
    pub fn can_instantiate(&self) -> bool {
        self.instantiation_count < self.max_instantiations
    }

    /// Get the number of bound variables
    pub fn num_vars(&self) -> usize {
        self.bound_vars.len()
    }

    /// Get variable name by index
    pub fn var_name(&self, idx: usize) -> Option<Spur> {
        self.bound_vars.get(idx).map(|(name, _)| *name)
    }

    /// Get variable sort by index
    pub fn var_sort(&self, idx: usize) -> Option<SortId> {
        self.bound_vars.get(idx).map(|(_, sort)| *sort)
    }

    /// Record an instantiation
    pub fn record_instantiation(&mut self) {
        self.instantiation_count += 1;
    }

    /// Calculate priority score (higher = more important)
    pub fn priority_score(&self) -> f64 {
        // Combine multiple factors:
        // - Weight (user-specified importance)
        // - Inverse of instantiation count (prefer less-instantiated)
        // - Inverse of nesting depth (prefer simpler quantifiers)
        let inst_factor = 1.0 / (1.0 + self.instantiation_count as f64);
        let depth_factor = 1.0 / (1.0 + self.nesting_depth as f64);
        self.weight * inst_factor * depth_factor
    }
}

/// An instantiation of a quantified formula
#[derive(Debug, Clone)]
pub struct Instantiation {
    /// The quantifier that was instantiated
    pub quantifier: TermId,
    /// The substitution used (variable name -> term)
    pub substitution: FxHashMap<Spur, TermId>,
    /// The resulting ground term (body with substitution applied)
    pub result: TermId,
    /// Generation at which this instantiation was created
    pub generation: u32,
    /// Reason for instantiation (for proof generation)
    pub reason: InstantiationReason,
    /// Cost/weight of this instantiation
    pub cost: f64,
}

impl Instantiation {
    /// Create a new instantiation
    pub fn new(
        quantifier: TermId,
        substitution: FxHashMap<Spur, TermId>,
        result: TermId,
        generation: u32,
    ) -> Self {
        Self {
            quantifier,
            substitution,
            result,
            generation,
            reason: InstantiationReason::ModelBased,
            cost: 1.0,
        }
    }

    /// Create with reason
    pub fn with_reason(
        quantifier: TermId,
        substitution: FxHashMap<Spur, TermId>,
        result: TermId,
        generation: u32,
        reason: InstantiationReason,
    ) -> Self {
        Self {
            quantifier,
            substitution,
            result,
            generation,
            reason,
            cost: 1.0,
        }
    }

    /// Get the binding as a sorted vector for hashing
    pub fn binding_key(&self) -> Vec<(Spur, TermId)> {
        let mut key: Vec<_> = self.substitution.iter().map(|(&k, &v)| (k, v)).collect();
        key.sort_by_key(|(name, _)| *name);
        key
    }
}

/// Reason for creating an instantiation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstantiationReason {
    /// Model-based instantiation
    ModelBased,
    /// E-matching pattern instantiation
    EMatching,
    /// Conflict-driven instantiation
    Conflict,
    /// Enumerative instantiation
    Enumerative,
    /// User-provided instantiation
    User,
    /// Theory-specific instantiation
    Theory,
}

impl fmt::Display for InstantiationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelBased => write!(f, "model-based"),
            Self::EMatching => write!(f, "e-matching"),
            Self::Conflict => write!(f, "conflict"),
            Self::Enumerative => write!(f, "enumerative"),
            Self::User => write!(f, "user"),
            Self::Theory => write!(f, "theory"),
        }
    }
}

/// Result of MBQI check
#[derive(Debug, Clone)]
pub enum MBQIResult {
    /// No quantifiers to process
    NoQuantifiers,
    /// All quantifiers satisfied under the model
    Satisfied,
    /// Found new instantiations to add
    NewInstantiations(Vec<Instantiation>),
    /// Found a conflict (quantifier cannot be satisfied)
    Conflict {
        quantifier: TermId,
        reason: Vec<TermId>,
    },
    /// Reached instantiation limit
    InstantiationLimit,
    /// Unknown (resource limits or incompleteness)
    Unknown,
}

impl MBQIResult {
    /// Check if the result is satisfiable
    pub fn is_sat(&self) -> bool {
        matches!(self, Self::Satisfied)
    }

    /// Check if the result is unsatisfiable
    pub fn is_unsat(&self) -> bool {
        matches!(self, Self::Conflict { .. })
    }

    /// Check if new instantiations were found
    pub fn has_instantiations(&self) -> bool {
        matches!(self, Self::NewInstantiations(_))
    }

    /// Get the number of new instantiations
    pub fn num_instantiations(&self) -> usize {
        match self {
            Self::NewInstantiations(insts) => insts.len(),
            _ => 0,
        }
    }
}

/// Statistics about MBQI
#[derive(Debug, Clone, Default)]
pub struct MBQIStats {
    /// Number of tracked quantifiers
    pub num_quantifiers: usize,
    /// Total instantiations generated
    pub total_instantiations: usize,
    /// Maximum allowed instantiations
    pub max_instantiations: usize,
    /// Unique instantiations (after deduplication)
    pub unique_instantiations: usize,
    /// Number of MBQI check calls
    pub num_checks: usize,
    /// Number of successful model completions
    pub num_completions: usize,
    /// Number of counterexamples found
    pub num_counterexamples: usize,
    /// Number of conflicts generated
    pub num_conflicts: usize,
    /// Total time spent in MBQI (microseconds)
    pub total_time_us: u64,
    /// Time spent in model completion (microseconds)
    pub completion_time_us: u64,
    /// Time spent in counterexample search (microseconds)
    pub cex_search_time_us: u64,
}

impl MBQIStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get average instantiations per check
    pub fn avg_instantiations_per_check(&self) -> f64 {
        if self.num_checks == 0 {
            0.0
        } else {
            self.total_instantiations as f64 / self.num_checks as f64
        }
    }

    /// Get average time per check (microseconds)
    pub fn avg_time_per_check_us(&self) -> f64 {
        if self.num_checks == 0 {
            0.0
        } else {
            self.total_time_us as f64 / self.num_checks as f64
        }
    }
}

impl fmt::Display for MBQIStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MBQI Statistics:")?;
        writeln!(f, "  Quantifiers: {}", self.num_quantifiers)?;
        writeln!(f, "  Total checks: {}", self.num_checks)?;
        writeln!(f, "  Total instantiations: {}", self.total_instantiations)?;
        writeln!(f, "  Unique instantiations: {}", self.unique_instantiations)?;
        writeln!(
            f,
            "  Avg inst/check: {:.2}",
            self.avg_instantiations_per_check()
        )?;
        writeln!(f, "  Counterexamples: {}", self.num_counterexamples)?;
        writeln!(f, "  Conflicts: {}", self.num_conflicts)?;
        writeln!(
            f,
            "  Total time: {:.2}ms",
            self.total_time_us as f64 / 1000.0
        )?;
        writeln!(
            f,
            "  Completion time: {:.2}ms",
            self.completion_time_us as f64 / 1000.0
        )?;
        writeln!(
            f,
            "  CEX search time: {:.2}ms",
            self.cex_search_time_us as f64 / 1000.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lasso::Key;

    #[test]
    fn test_quantified_formula_creation() {
        let qf = QuantifiedFormula::new(TermId::new(1), SmallVec::new(), TermId::new(2), true);
        assert!(qf.is_universal);
        assert_eq!(qf.instantiation_count, 0);
        assert!(qf.can_instantiate());
    }

    #[test]
    fn test_quantified_formula_priority() {
        let mut qf = QuantifiedFormula::with_params(
            TermId::new(1),
            SmallVec::new(),
            TermId::new(2),
            true,
            100,
            2.0,
        );
        let initial_priority = qf.priority_score();
        qf.record_instantiation();
        let after_priority = qf.priority_score();
        // Priority should decrease after instantiation
        assert!(after_priority < initial_priority);
    }

    #[test]
    fn test_instantiation_binding_key() {
        let mut subst = FxHashMap::default();
        subst.insert(
            Spur::try_from_usize(1).expect("valid spur"),
            TermId::new(10),
        );
        subst.insert(
            Spur::try_from_usize(2).expect("valid spur"),
            TermId::new(20),
        );

        let inst = Instantiation::new(TermId::new(1), subst, TermId::new(3), 0);
        let key = inst.binding_key();
        assert_eq!(key.len(), 2);
        // Should be sorted
        assert!(key[0].0 <= key[1].0);
    }

    #[test]
    fn test_mbqi_result_predicates() {
        let sat = MBQIResult::Satisfied;
        assert!(sat.is_sat());
        assert!(!sat.is_unsat());
        assert!(!sat.has_instantiations());

        let conflict = MBQIResult::Conflict {
            quantifier: TermId::new(1),
            reason: vec![],
        };
        assert!(!conflict.is_sat());
        assert!(conflict.is_unsat());

        let inst = MBQIResult::NewInstantiations(vec![]);
        assert!(inst.has_instantiations());
        assert_eq!(inst.num_instantiations(), 0);
    }

    #[test]
    fn test_stats_initialization() {
        let stats = MBQIStats::new();
        assert_eq!(stats.num_quantifiers, 0);
        assert_eq!(stats.total_instantiations, 0);
        assert_eq!(stats.avg_instantiations_per_check(), 0.0);
    }

    #[test]
    fn test_stats_averages() {
        let mut stats = MBQIStats::new();
        stats.num_checks = 10;
        stats.total_instantiations = 50;
        stats.total_time_us = 1000;

        assert_eq!(stats.avg_instantiations_per_check(), 5.0);
        assert_eq!(stats.avg_time_per_check_us(), 100.0);
    }

    #[test]
    fn test_instantiation_reason_display() {
        assert_eq!(
            format!("{}", InstantiationReason::ModelBased),
            "model-based"
        );
        assert_eq!(format!("{}", InstantiationReason::EMatching), "e-matching");
        assert_eq!(format!("{}", InstantiationReason::Conflict), "conflict");
    }
}
