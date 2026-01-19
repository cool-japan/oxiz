//! Objective function optimization (OMT).
//!
//! This module implements Optimization Modulo Theories for optimizing
//! objective functions over SMT formulas. It supports:
//! - Linear arithmetic objectives
//! - Binary search optimization
//! - Linear search optimization
//! - Bit-vector objectives
//!
//! Reference: Z3's `opt/optsmt.cpp`

use crate::maxsat::Weight;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Errors that can occur during objective optimization
#[derive(Error, Debug)]
pub enum ObjectiveError {
    /// No solution exists
    #[error("unsatisfiable")]
    Unsatisfiable,
    /// Objective is unbounded
    #[error("objective unbounded")]
    Unbounded,
    /// Timeout or resource limit
    #[error("resource limit")]
    ResourceLimit,
    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result of objective optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectiveResult {
    /// Optimal value found
    Optimal(Weight),
    /// Solution found but optimality not proven
    Satisfiable(Weight),
    /// Objective is unbounded
    Unbounded,
    /// No solution exists
    Unsatisfiable,
    /// Could not determine within limits
    Unknown,
}

/// Unique identifier for an objective
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectiveId(pub u32);

impl ObjectiveId {
    /// Create a new objective ID
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

impl From<u32> for ObjectiveId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for ObjectiveId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl From<ObjectiveId> for u32 {
    fn from(id: ObjectiveId) -> Self {
        id.0
    }
}

impl From<ObjectiveId> for usize {
    fn from(id: ObjectiveId) -> Self {
        id.0 as usize
    }
}

/// Kind of objective
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveKind {
    /// Minimize the objective
    Minimize,
    /// Maximize the objective
    Maximize,
}

/// An objective function to optimize
#[derive(Debug, Clone)]
pub struct Objective {
    /// Unique identifier
    pub id: ObjectiveId,
    /// The term representing the objective
    pub term: TermId,
    /// Optimization direction
    pub kind: ObjectiveKind,
    /// Priority (lower = higher priority, for lexicographic)
    pub priority: u32,
}

impl Objective {
    /// Create a minimization objective
    pub fn minimize(id: ObjectiveId, term: TermId) -> Self {
        Self {
            id,
            term,
            kind: ObjectiveKind::Minimize,
            priority: 0,
        }
    }

    /// Create a maximization objective
    pub fn maximize(id: ObjectiveId, term: TermId) -> Self {
        Self {
            id,
            term,
            kind: ObjectiveKind::Maximize,
            priority: 0,
        }
    }

    /// Check if this is a minimization objective
    pub fn is_minimize(&self) -> bool {
        matches!(self.kind, ObjectiveKind::Minimize)
    }

    /// Check if this is a maximization objective
    pub fn is_maximize(&self) -> bool {
        matches!(self.kind, ObjectiveKind::Maximize)
    }
}

/// Bound on an objective value
#[derive(Debug, Clone)]
pub struct ObjectiveBound {
    /// The bound value
    pub value: Weight,
    /// Whether the bound is strict (<) or non-strict (<=)
    pub strict: bool,
}

impl ObjectiveBound {
    /// Create a non-strict bound
    pub fn at_most(value: Weight) -> Self {
        Self {
            value,
            strict: false,
        }
    }

    /// Create a strict bound
    pub fn less_than(value: Weight) -> Self {
        Self {
            value,
            strict: true,
        }
    }
}

/// Configuration for objective optimization
#[derive(Debug, Clone)]
pub struct ObjectiveConfig {
    /// Use binary search (vs linear search)
    pub binary_search: bool,
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Precision for rational objectives
    pub precision: Option<BigRational>,
    /// Initial lower bound
    pub initial_lower: Option<Weight>,
    /// Initial upper bound
    pub initial_upper: Option<Weight>,
}

impl Default for ObjectiveConfig {
    fn default() -> Self {
        Self {
            binary_search: true,
            max_iterations: 100000,
            precision: None,
            initial_lower: None,
            initial_upper: None,
        }
    }
}

/// Statistics from objective optimization
#[derive(Debug, Clone, Default)]
pub struct ObjectiveStats {
    /// Number of solver calls
    pub solver_calls: u32,
    /// Number of bound updates
    pub bound_updates: u32,
    /// Number of binary search iterations
    pub binary_iterations: u32,
    /// Number of linear search iterations
    pub linear_iterations: u32,
}

/// Linear objective representation
///
/// Represents an objective of the form: c0 + c1*x1 + c2*x2 + ... + cn*xn
#[derive(Debug, Clone)]
pub struct LinearObjective {
    /// Constant term
    pub constant: BigRational,
    /// Coefficients for variables: variable ID -> coefficient
    pub coefficients: FxHashMap<u32, BigRational>,
}

impl LinearObjective {
    /// Create a zero objective
    pub fn zero() -> Self {
        Self {
            constant: BigRational::zero(),
            coefficients: FxHashMap::default(),
        }
    }

    /// Create a constant objective
    pub fn constant(value: impl Into<BigRational>) -> Self {
        Self {
            constant: value.into(),
            coefficients: FxHashMap::default(),
        }
    }

    /// Create a single variable objective
    pub fn var(id: u32) -> Self {
        let mut coefficients = FxHashMap::default();
        coefficients.insert(id, BigRational::one());
        Self {
            constant: BigRational::zero(),
            coefficients,
        }
    }

    /// Create a variable with coefficient
    pub fn term(id: u32, coeff: impl Into<BigRational>) -> Self {
        let mut coefficients = FxHashMap::default();
        coefficients.insert(id, coeff.into());
        Self {
            constant: BigRational::zero(),
            coefficients,
        }
    }

    /// Add another objective to this one
    pub fn add(&self, other: &LinearObjective) -> Self {
        let mut result = self.clone();
        result.constant = &result.constant + &other.constant;
        for (&var, coeff) in &other.coefficients {
            let entry = result
                .coefficients
                .entry(var)
                .or_insert_with(BigRational::zero);
            *entry = &*entry + coeff;
        }
        result
    }

    /// Multiply by a constant
    pub fn scale(&self, factor: &BigRational) -> Self {
        let mut result = Self {
            constant: &self.constant * factor,
            coefficients: FxHashMap::default(),
        };
        for (&var, coeff) in &self.coefficients {
            result.coefficients.insert(var, coeff * factor);
        }
        result
    }

    /// Negate the objective
    pub fn negate(&self) -> Self {
        self.scale(&(-BigRational::one()))
    }

    /// Check if this is a constant objective
    pub fn is_constant(&self) -> bool {
        self.coefficients.is_empty() || self.coefficients.values().all(|c| c.is_zero())
    }

    /// Get the constant value (if applicable)
    pub fn as_constant(&self) -> Option<&BigRational> {
        if self.is_constant() {
            Some(&self.constant)
        } else {
            None
        }
    }

    /// Evaluate the objective given variable assignments
    pub fn evaluate(&self, assignments: &FxHashMap<u32, BigRational>) -> BigRational {
        let mut result = self.constant.clone();
        for (&var, coeff) in &self.coefficients {
            if let Some(val) = assignments.get(&var) {
                result = &result + &(coeff * val);
            }
        }
        result
    }
}

impl Default for LinearObjective {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<BigRational> for LinearObjective {
    /// Create a constant linear objective from a BigRational.
    fn from(value: BigRational) -> Self {
        Self::constant(value)
    }
}

impl From<BigInt> for LinearObjective {
    /// Create a constant linear objective from a BigInt.
    fn from(value: BigInt) -> Self {
        Self::constant(BigRational::from(value))
    }
}

impl From<i64> for LinearObjective {
    /// Create a constant linear objective from an i64.
    fn from(value: i64) -> Self {
        Self::constant(BigRational::from(BigInt::from(value)))
    }
}

impl From<(u32, i64)> for LinearObjective {
    /// Create a linear objective with a single term: coefficient * var.
    ///
    /// # Example
    /// ```
    /// use oxiz_opt::LinearObjective;
    /// // Creates 3 * x_0
    /// let obj: LinearObjective = (0u32, 3i64).into();
    /// ```
    fn from((var_id, coeff): (u32, i64)) -> Self {
        Self::term(var_id, BigRational::from(BigInt::from(coeff)))
    }
}

impl<const N: usize> From<[(u32, i64); N]> for LinearObjective {
    /// Create a linear objective from an array of (variable_id, coefficient) pairs.
    ///
    /// # Example
    /// ```
    /// use oxiz_opt::LinearObjective;
    /// // Creates 2*x_0 + 3*x_1
    /// let obj: LinearObjective = [(0u32, 2i64), (1, 3)].into();
    /// ```
    fn from(terms: [(u32, i64); N]) -> Self {
        let mut coefficients = FxHashMap::default();
        for (var_id, coeff) in terms {
            coefficients.insert(var_id, BigRational::from(BigInt::from(coeff)));
        }
        Self {
            constant: BigRational::zero(),
            coefficients,
        }
    }
}

impl From<Vec<(u32, i64)>> for LinearObjective {
    /// Create a linear objective from a vector of (variable_id, coefficient) pairs.
    fn from(terms: Vec<(u32, i64)>) -> Self {
        let mut coefficients = FxHashMap::default();
        for (var_id, coeff) in terms {
            coefficients.insert(var_id, BigRational::from(BigInt::from(coeff)));
        }
        Self {
            constant: BigRational::zero(),
            coefficients,
        }
    }
}

/// Objective optimizer using binary/linear search
#[derive(Debug)]
pub struct ObjectiveOptimizer {
    /// Objective to optimize
    objective: Objective,
    /// Linear representation (if available)
    linear: Option<LinearObjective>,
    /// Current lower bound
    lower_bound: Option<Weight>,
    /// Current upper bound
    upper_bound: Option<Weight>,
    /// Best solution found
    best_value: Option<Weight>,
    /// Configuration
    config: ObjectiveConfig,
    /// Statistics
    stats: ObjectiveStats,
}

impl ObjectiveOptimizer {
    /// Create a new optimizer for an objective
    pub fn new(objective: Objective) -> Self {
        Self::with_config(objective, ObjectiveConfig::default())
    }

    /// Create a new optimizer with configuration
    pub fn with_config(objective: Objective, config: ObjectiveConfig) -> Self {
        Self {
            objective,
            linear: None,
            lower_bound: config.initial_lower.clone(),
            upper_bound: config.initial_upper.clone(),
            best_value: None,
            config,
            stats: ObjectiveStats::default(),
        }
    }

    /// Set the linear representation of the objective
    pub fn set_linear(&mut self, linear: LinearObjective) {
        self.linear = Some(linear);
    }

    /// Update the lower bound
    pub fn update_lower(&mut self, value: Weight) {
        match (&self.lower_bound, &value) {
            (None, _) => self.lower_bound = Some(value),
            (Some(current), _) if value > *current => {
                self.lower_bound = Some(value);
                self.stats.bound_updates += 1;
            }
            _ => {}
        }
    }

    /// Update the upper bound
    pub fn update_upper(&mut self, value: Weight) {
        match (&self.upper_bound, &value) {
            (None, _) => self.upper_bound = Some(value),
            (Some(current), _) if value < *current => {
                self.upper_bound = Some(value);
                self.stats.bound_updates += 1;
            }
            _ => {}
        }
    }

    /// Record a solution with the given objective value
    pub fn record_solution(&mut self, value: Weight) {
        match self.objective.kind {
            ObjectiveKind::Minimize => {
                self.update_upper(value.clone());
                match &self.best_value {
                    None => self.best_value = Some(value),
                    Some(current) if value < *current => self.best_value = Some(value),
                    _ => {}
                }
            }
            ObjectiveKind::Maximize => {
                self.update_lower(value.clone());
                match &self.best_value {
                    None => self.best_value = Some(value),
                    Some(current) if value > *current => self.best_value = Some(value),
                    _ => {}
                }
            }
        }
    }

    /// Get the current best value
    pub fn best_value(&self) -> Option<&Weight> {
        self.best_value.as_ref()
    }

    /// Get the lower bound
    pub fn lower_bound(&self) -> Option<&Weight> {
        self.lower_bound.as_ref()
    }

    /// Get the upper bound
    pub fn upper_bound(&self) -> Option<&Weight> {
        self.upper_bound.as_ref()
    }

    /// Get statistics
    pub fn stats(&self) -> &ObjectiveStats {
        &self.stats
    }

    /// Check if optimization is complete
    pub fn is_complete(&self) -> bool {
        match (&self.lower_bound, &self.upper_bound) {
            (Some(lower), Some(upper)) => lower >= upper,
            _ => false,
        }
    }

    /// Get the next bound to try using binary search
    pub fn next_binary_bound(&self) -> Option<Weight> {
        let lower = self.lower_bound.as_ref()?;
        let upper = self.upper_bound.as_ref()?;

        // Compute midpoint
        match (lower, upper) {
            (Weight::Int(l), Weight::Int(u)) => {
                if l >= u {
                    return None;
                }
                let mid = (l + u) / BigInt::from(2);
                Some(Weight::Int(mid))
            }
            (Weight::Rational(l), Weight::Rational(u)) => {
                if l >= u {
                    return None;
                }
                let mid = (l + u) / BigRational::from(BigInt::from(2));
                Some(Weight::Rational(mid))
            }
            _ => None,
        }
    }

    /// Get the next bound to try using linear search
    pub fn next_linear_bound(&self) -> Option<Weight> {
        match self.objective.kind {
            ObjectiveKind::Minimize => {
                // Try to improve upper bound
                self.upper_bound.as_ref().map(|u| {
                    match u {
                        Weight::Int(n) => Weight::Int(n - BigInt::one()),
                        Weight::Rational(r) => {
                            // Use precision or small decrement
                            let delta = self.config.precision.clone().unwrap_or_else(|| {
                                BigRational::new(BigInt::one(), BigInt::from(1000))
                            });
                            Weight::Rational(r - delta)
                        }
                        Weight::Infinite => Weight::Infinite,
                    }
                })
            }
            ObjectiveKind::Maximize => {
                // Try to improve lower bound
                self.lower_bound.as_ref().map(|l| match l {
                    Weight::Int(n) => Weight::Int(n + BigInt::one()),
                    Weight::Rational(r) => {
                        let delta =
                            self.config.precision.clone().unwrap_or_else(|| {
                                BigRational::new(BigInt::one(), BigInt::from(1000))
                            });
                        Weight::Rational(r + delta)
                    }
                    Weight::Infinite => Weight::Infinite,
                })
            }
        }
    }

    /// Reset the optimizer
    pub fn reset(&mut self) {
        self.lower_bound = self.config.initial_lower.clone();
        self.upper_bound = self.config.initial_upper.clone();
        self.best_value = None;
        self.stats = ObjectiveStats::default();
    }
}

/// Manager for multiple objectives (lexicographic optimization)
#[derive(Debug)]
pub struct ObjectiveManager {
    /// Objectives sorted by priority
    objectives: Vec<ObjectiveOptimizer>,
    /// Current objective index being optimized
    current_index: usize,
}

impl ObjectiveManager {
    /// Create a new objective manager
    pub fn new() -> Self {
        Self {
            objectives: Vec::new(),
            current_index: 0,
        }
    }

    /// Add an objective
    pub fn add(&mut self, objective: Objective) -> ObjectiveId {
        let id = objective.id;
        let optimizer = ObjectiveOptimizer::new(objective);
        self.objectives.push(optimizer);
        // Sort by priority
        self.objectives.sort_by_key(|o| o.objective.priority);
        id
    }

    /// Get the current objective being optimized
    pub fn current(&self) -> Option<&ObjectiveOptimizer> {
        self.objectives.get(self.current_index)
    }

    /// Get the current objective mutably
    pub fn current_mut(&mut self) -> Option<&mut ObjectiveOptimizer> {
        self.objectives.get_mut(self.current_index)
    }

    /// Move to the next objective
    pub fn advance_to_next(&mut self) -> bool {
        if self.current_index + 1 < self.objectives.len() {
            self.current_index += 1;
            true
        } else {
            false
        }
    }

    /// Check if all objectives are optimized
    pub fn is_complete(&self) -> bool {
        self.current_index >= self.objectives.len()
            || self.objectives.iter().all(|o| o.is_complete())
    }

    /// Reset the manager
    pub fn reset(&mut self) {
        self.current_index = 0;
        for opt in &mut self.objectives {
            opt.reset();
        }
    }

    /// Get all objective values
    pub fn values(&self) -> Vec<Option<Weight>> {
        self.objectives
            .iter()
            .map(|o| o.best_value.clone())
            .collect()
    }
}

impl Default for ObjectiveManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objective_id() {
        let id = ObjectiveId::new(42);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn test_objective_kinds() {
        let min = Objective::minimize(ObjectiveId::new(0), TermId::from(1));
        let max = Objective::maximize(ObjectiveId::new(1), TermId::from(2));

        assert!(min.is_minimize());
        assert!(!min.is_maximize());
        assert!(max.is_maximize());
        assert!(!max.is_minimize());
    }

    #[test]
    fn test_linear_objective_constant() {
        let obj = LinearObjective::constant(BigRational::from(BigInt::from(5)));
        assert!(obj.is_constant());
        assert_eq!(obj.as_constant(), Some(&BigRational::from(BigInt::from(5))));
    }

    #[test]
    fn test_linear_objective_var() {
        let obj = LinearObjective::var(0);
        assert!(!obj.is_constant());
        assert_eq!(obj.coefficients.get(&0), Some(&BigRational::one()));
    }

    #[test]
    fn test_linear_objective_add() {
        let a = LinearObjective::term(0, BigRational::from(BigInt::from(2)));
        let b = LinearObjective::term(1, BigRational::from(BigInt::from(3)));
        let sum = a.add(&b);

        assert_eq!(
            sum.coefficients.get(&0),
            Some(&BigRational::from(BigInt::from(2)))
        );
        assert_eq!(
            sum.coefficients.get(&1),
            Some(&BigRational::from(BigInt::from(3)))
        );
    }

    #[test]
    fn test_linear_objective_scale() {
        let obj = LinearObjective::term(0, BigRational::from(BigInt::from(2)));
        let scaled = obj.scale(&BigRational::from(BigInt::from(3)));

        assert_eq!(
            scaled.coefficients.get(&0),
            Some(&BigRational::from(BigInt::from(6)))
        );
    }

    #[test]
    fn test_linear_objective_evaluate() {
        let obj = LinearObjective {
            constant: BigRational::from(BigInt::from(1)),
            coefficients: {
                let mut m = FxHashMap::default();
                m.insert(0, BigRational::from(BigInt::from(2)));
                m.insert(1, BigRational::from(BigInt::from(3)));
                m
            },
        };

        let mut assignments = FxHashMap::default();
        assignments.insert(0, BigRational::from(BigInt::from(4)));
        assignments.insert(1, BigRational::from(BigInt::from(5)));

        // 1 + 2*4 + 3*5 = 1 + 8 + 15 = 24
        assert_eq!(
            obj.evaluate(&assignments),
            BigRational::from(BigInt::from(24))
        );
    }

    #[test]
    fn test_objective_optimizer_bounds() {
        let obj = Objective::minimize(ObjectiveId::new(0), TermId::from(1));
        let mut opt = ObjectiveOptimizer::new(obj);

        opt.update_lower(Weight::from(0));
        opt.update_upper(Weight::from(100));

        assert_eq!(opt.lower_bound(), Some(&Weight::from(0)));
        assert_eq!(opt.upper_bound(), Some(&Weight::from(100)));
    }

    #[test]
    fn test_objective_optimizer_record_minimize() {
        let obj = Objective::minimize(ObjectiveId::new(0), TermId::from(1));
        let mut opt = ObjectiveOptimizer::new(obj);

        opt.record_solution(Weight::from(50));
        assert_eq!(opt.best_value(), Some(&Weight::from(50)));

        opt.record_solution(Weight::from(30));
        assert_eq!(opt.best_value(), Some(&Weight::from(30)));

        opt.record_solution(Weight::from(40));
        assert_eq!(opt.best_value(), Some(&Weight::from(30))); // Still 30, not worse
    }

    #[test]
    fn test_objective_optimizer_record_maximize() {
        let obj = Objective::maximize(ObjectiveId::new(0), TermId::from(1));
        let mut opt = ObjectiveOptimizer::new(obj);

        opt.record_solution(Weight::from(50));
        assert_eq!(opt.best_value(), Some(&Weight::from(50)));

        opt.record_solution(Weight::from(70));
        assert_eq!(opt.best_value(), Some(&Weight::from(70)));

        opt.record_solution(Weight::from(60));
        assert_eq!(opt.best_value(), Some(&Weight::from(70))); // Still 70, not worse
    }

    #[test]
    fn test_objective_manager() {
        let mut manager = ObjectiveManager::new();

        let _id1 = manager.add(Objective {
            id: ObjectiveId::new(0),
            term: TermId::from(1),
            kind: ObjectiveKind::Minimize,
            priority: 1,
        });

        let _id2 = manager.add(Objective {
            id: ObjectiveId::new(1),
            term: TermId::from(2),
            kind: ObjectiveKind::Maximize,
            priority: 0, // Higher priority
        });

        // Current should be the one with priority 0
        assert!(manager.current().is_some());

        // Move to next
        assert!(manager.advance_to_next());
        assert!(!manager.advance_to_next()); // No more
    }

    #[test]
    fn test_binary_search_midpoint() {
        let obj = Objective::minimize(ObjectiveId::new(0), TermId::from(1));
        let mut opt = ObjectiveOptimizer::new(obj);

        opt.update_lower(Weight::from(0));
        opt.update_upper(Weight::from(100));

        let mid = opt.next_binary_bound();
        assert_eq!(mid, Some(Weight::from(50)));
    }

    // Tests for From implementations

    #[test]
    fn test_objective_id_from_u32() {
        let id: ObjectiveId = ObjectiveId::from(5u32);
        assert_eq!(id.raw(), 5);
    }

    #[test]
    fn test_objective_id_from_usize() {
        let id: ObjectiveId = ObjectiveId::from(10usize);
        assert_eq!(id.raw(), 10);
    }

    #[test]
    fn test_objective_id_to_u32() {
        let id = ObjectiveId::new(7);
        let n: u32 = id.into();
        assert_eq!(n, 7);
    }

    #[test]
    fn test_objective_id_to_usize() {
        let id = ObjectiveId::new(9);
        let n: usize = id.into();
        assert_eq!(n, 9);
    }

    #[test]
    fn test_objective_id_roundtrip() {
        let original = 42u32;
        let id: ObjectiveId = original.into();
        let back: u32 = id.into();
        assert_eq!(original, back);
    }

    #[test]
    fn test_linear_objective_from_bigrational() {
        let r = BigRational::from(BigInt::from(5));
        let obj: LinearObjective = r.clone().into();
        assert!(obj.is_constant());
        assert_eq!(obj.constant, r);
    }

    #[test]
    fn test_linear_objective_from_bigint() {
        let n = BigInt::from(10);
        let obj: LinearObjective = n.clone().into();
        assert!(obj.is_constant());
        assert_eq!(obj.constant, BigRational::from(n));
    }

    #[test]
    fn test_linear_objective_from_i64() {
        let obj: LinearObjective = LinearObjective::from(42i64);
        assert!(obj.is_constant());
        assert_eq!(obj.constant, BigRational::from(BigInt::from(42)));
    }

    #[test]
    fn test_linear_objective_from_single_term_tuple() {
        let obj: LinearObjective = (0u32, 3i64).into();
        assert!(!obj.is_constant());
        assert_eq!(
            obj.coefficients.get(&0),
            Some(&BigRational::from(BigInt::from(3)))
        );
    }

    #[test]
    fn test_linear_objective_from_array() {
        let obj: LinearObjective = [(0u32, 2i64), (1, 3), (2, -1)].into();
        assert!(!obj.is_constant());
        assert_eq!(
            obj.coefficients.get(&0),
            Some(&BigRational::from(BigInt::from(2)))
        );
        assert_eq!(
            obj.coefficients.get(&1),
            Some(&BigRational::from(BigInt::from(3)))
        );
        assert_eq!(
            obj.coefficients.get(&2),
            Some(&BigRational::from(BigInt::from(-1)))
        );
    }

    #[test]
    fn test_linear_objective_from_vec() {
        let terms = vec![(0u32, 1i64), (1, 2), (2, 3)];
        let obj: LinearObjective = terms.into();
        assert!(!obj.is_constant());
        assert_eq!(obj.coefficients.len(), 3);
    }

    #[test]
    fn test_linear_objective_default() {
        let obj = LinearObjective::default();
        assert!(obj.is_constant());
        assert!(obj.constant.is_zero());
    }
}
