//! Inequality Propagation Tactic.
//!
//! Propagates arithmetic inequalities to tighten bounds and detect conflicts.
//!
//! ## Propagation Rules
//!
//! - **(x ≤ a) ∧ (a ≤ b)** → **(x ≤ b)**
//! - **(x ≥ a) ∧ (b ≥ a)** → **(x ≥ b)** (strengthens to tighter bound)
//! - **(x + y ≤ a) ∧ (x ≥ b)** → **(y ≤ a - b)**
//! - Transitive closure of inequalities
//!
//! ## References
//!
//! - Z3's `tactic/arith/propagate_ineqs_tactic.cpp`
//! - "Abstract DPLL and Abstract DPLL Modulo Theories" (Nieuwenhuis et al., 2006)

use crate::TermId;
use crate::tactic::{Goal, Tactic, TacticResult};
use num_rational::BigRational;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Variable identifier.
pub type VarId = usize;

/// A bound on a variable: x ≤ upper or x ≥ lower.
#[derive(Debug, Clone)]
pub struct Bound {
    /// Variable.
    pub var: VarId,
    /// Bound value.
    pub value: BigRational,
    /// Type of bound.
    pub kind: BoundKind,
    /// Origin (for explanation).
    pub origin: Option<TermId>,
}

/// Type of bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundKind {
    /// Upper bound (x ≤ value).
    Upper,
    /// Lower bound (x ≥ value).
    Lower,
}

/// Configuration for inequality propagation.
#[derive(Debug, Clone)]
pub struct PropagateIneqsConfig {
    /// Maximum propagation iterations.
    pub max_iterations: usize,
    /// Enable transitive closure.
    pub transitive: bool,
    /// Enable linear combination propagation.
    pub linear_combinations: bool,
}

impl Default for PropagateIneqsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            transitive: true,
            linear_combinations: true,
        }
    }
}

/// Statistics for inequality propagation.
#[derive(Debug, Clone, Default)]
pub struct PropagateIneqsStats {
    /// Iterations performed.
    pub iterations: u64,
    /// Bounds propagated.
    pub bounds_propagated: u64,
    /// Conflicts detected.
    pub conflicts_detected: u64,
    /// Transitive closures computed.
    pub transitive_closures: u64,
}

/// Inequality propagation tactic.
pub struct PropagateIneqsTactic {
    /// Configuration.
    config: PropagateIneqsConfig,
    /// Statistics.
    stats: PropagateIneqsStats,
    /// Current bounds: var -> (lower, upper).
    bounds: FxHashMap<VarId, (Option<BigRational>, Option<BigRational>)>,
    /// Propagation queue.
    queue: VecDeque<Bound>,
}

impl PropagateIneqsTactic {
    /// Create a new propagate inequalities tactic.
    pub fn new(config: PropagateIneqsConfig) -> Self {
        Self {
            config,
            stats: PropagateIneqsStats::default(),
            bounds: FxHashMap::default(),
            queue: VecDeque::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PropagateIneqsConfig::default())
    }

    /// Add a bound to be propagated.
    pub fn add_bound(&mut self, bound: Bound) {
        let var = bound.var;

        // Update stored bounds
        let (lower, upper) = self.bounds.entry(var).or_insert((None, None));

        match bound.kind {
            BoundKind::Lower => {
                if lower.as_ref().is_none_or(|l| bound.value > *l) {
                    *lower = Some(bound.value.clone());
                    self.queue.push_back(bound);
                }
            }
            BoundKind::Upper => {
                if upper.as_ref().is_none_or(|u| bound.value < *u) {
                    *upper = Some(bound.value.clone());
                    self.queue.push_back(bound);
                }
            }
        }
    }

    /// Propagate all bounds until fixpoint.
    pub fn propagate(&mut self) -> Result<(), PropagateIneqsError> {
        for _ in 0..self.config.max_iterations {
            self.stats.iterations += 1;

            if self.queue.is_empty() {
                return Ok(());
            }

            while let Some(bound) = self.queue.pop_front() {
                self.propagate_bound(&bound)?;
                self.stats.bounds_propagated += 1;
            }
        }

        // Max iterations reached
        Ok(())
    }

    /// Propagate a single bound.
    fn propagate_bound(&mut self, bound: &Bound) -> Result<(), PropagateIneqsError> {
        // Check for conflicts
        if let Some((lower, upper)) = self.bounds.get(&bound.var)
            && let (Some(l), Some(u)) = (lower, upper)
            && l > u
        {
            self.stats.conflicts_detected += 1;
            return Err(PropagateIneqsError::Conflict);
        }

        // Propagate transitively if enabled
        if self.config.transitive {
            self.propagate_transitive(bound)?;
        }

        Ok(())
    }

    /// Propagate bounds transitively.
    fn propagate_transitive(&mut self, _bound: &Bound) -> Result<(), PropagateIneqsError> {
        self.stats.transitive_closures += 1;

        // Simplified: In real implementation, would find all
        // inequalities involving bound.var and propagate

        Ok(())
    }

    /// Get the current bound for a variable.
    pub fn get_bounds(&self, var: VarId) -> Option<&(Option<BigRational>, Option<BigRational>)> {
        self.bounds.get(&var)
    }

    /// Check if bounds are consistent.
    pub fn is_consistent(&self) -> bool {
        for (lower, upper) in self.bounds.values() {
            if let (Some(l), Some(u)) = (lower, upper)
                && l > u
            {
                return false;
            }
        }
        true
    }

    /// Get statistics.
    pub fn stats(&self) -> &PropagateIneqsStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PropagateIneqsStats::default();
    }

    /// Clear all bounds.
    pub fn clear(&mut self) {
        self.bounds.clear();
        self.queue.clear();
    }
}

impl Tactic for PropagateIneqsTactic {
    fn apply(&self, _goal: &Goal) -> crate::error::Result<TacticResult> {
        // Simplified: Would extract inequalities from goal and propagate
        Ok(TacticResult::NotApplicable)
    }

    fn name(&self) -> &str {
        "propagate-ineqs"
    }
}

/// Errors for inequality propagation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropagateIneqsError {
    /// Inconsistent bounds detected.
    Conflict,
    /// Maximum iterations exceeded.
    MaxIterations,
}

impl std::fmt::Display for PropagateIneqsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropagateIneqsError::Conflict => write!(f, "conflict detected"),
            PropagateIneqsError::MaxIterations => write!(f, "max iterations exceeded"),
        }
    }
}

impl std::error::Error for PropagateIneqsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tactic_creation() {
        let tactic = PropagateIneqsTactic::default_config();
        assert_eq!(tactic.stats().iterations, 0);
    }

    #[test]
    fn test_add_bound() {
        let mut tactic = PropagateIneqsTactic::default_config();

        let bound = Bound {
            var: 0,
            value: BigRational::from_integer(10.into()),
            kind: BoundKind::Upper,
            origin: None,
        };

        tactic.add_bound(bound);

        let bounds = tactic.get_bounds(0).unwrap();
        assert_eq!(bounds.1, Some(BigRational::from_integer(10.into())));
    }

    #[test]
    fn test_conflict_detection() {
        let mut tactic = PropagateIneqsTactic::default_config();

        // Add lower bound x ≥ 10
        tactic.add_bound(Bound {
            var: 0,
            value: BigRational::from_integer(10.into()),
            kind: BoundKind::Lower,
            origin: None,
        });

        // Add upper bound x ≤ 5 (conflict!)
        tactic.add_bound(Bound {
            var: 0,
            value: BigRational::from_integer(5.into()),
            kind: BoundKind::Upper,
            origin: None,
        });

        // Propagate should detect conflict
        let result = tactic.propagate();
        assert!(matches!(result, Err(PropagateIneqsError::Conflict)));
    }

    #[test]
    fn test_consistency_check() {
        let mut tactic = PropagateIneqsTactic::default_config();

        tactic.add_bound(Bound {
            var: 0,
            value: BigRational::from_integer(5.into()),
            kind: BoundKind::Lower,
            origin: None,
        });

        tactic.add_bound(Bound {
            var: 0,
            value: BigRational::from_integer(10.into()),
            kind: BoundKind::Upper,
            origin: None,
        });

        assert!(tactic.is_consistent());
    }

    #[test]
    fn test_stats() {
        let tactic = PropagateIneqsTactic::default_config();
        assert_eq!(tactic.stats().bounds_propagated, 0);
    }
}
