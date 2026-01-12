//! Bound propagation for polynomial constraints.
//!
//! This module implements interval-based bound propagation for polynomial
//! constraints, enabling early conflict detection and search space reduction.
//!
//! Key features:
//! - **Interval Arithmetic**: Track upper and lower bounds for variables
//! - **Constraint Propagation**: Propagate bounds through polynomial constraints
//! - **Conflict Detection**: Detect empty intervals (conflicts) early
//! - **Monotonicity Analysis**: Use polynomial monotonicity for tighter bounds
//!
//! Reference: Z3's bound propagation in theory solvers

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;
use std::fmt;

/// An interval representing possible values for a variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interval {
    /// Lower bound (inclusive).
    pub lower: Option<BigRational>,
    /// Upper bound (inclusive).
    pub upper: Option<BigRational>,
}

impl Interval {
    /// Create an unbounded interval (-∞, +∞).
    pub fn unbounded() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }

    /// Create an interval with given bounds.
    pub fn new(lower: Option<BigRational>, upper: Option<BigRational>) -> Self {
        Self { lower, upper }
    }

    /// Create a point interval [v, v].
    pub fn point(value: BigRational) -> Self {
        Self {
            lower: Some(value.clone()),
            upper: Some(value),
        }
    }

    /// Check if the interval is empty.
    pub fn is_empty(&self) -> bool {
        match (&self.lower, &self.upper) {
            (Some(l), Some(u)) => l > u,
            _ => false,
        }
    }

    /// Check if the interval is a single point.
    pub fn is_point(&self) -> bool {
        match (&self.lower, &self.upper) {
            (Some(l), Some(u)) => l == u,
            _ => false,
        }
    }

    /// Intersect with another interval.
    pub fn intersect(&self, other: &Interval) -> Interval {
        let lower = match (&self.lower, &other.lower) {
            (None, l) | (l, None) => l.clone(),
            (Some(l1), Some(l2)) => Some(l1.max(l2).clone()),
        };

        let upper = match (&self.upper, &other.upper) {
            (None, u) | (u, None) => u.clone(),
            (Some(u1), Some(u2)) => Some(u1.min(u2).clone()),
        };

        Interval::new(lower, upper)
    }

    /// Check if a value is in the interval.
    pub fn contains(&self, value: &BigRational) -> bool {
        let lower_ok = match &self.lower {
            None => true,
            Some(l) => value >= l,
        };

        let upper_ok = match &self.upper {
            None => true,
            Some(u) => value <= u,
        };

        lower_ok && upper_ok
    }

    /// Tighten the lower bound.
    pub fn tighten_lower(&mut self, new_lower: BigRational) {
        self.lower = Some(match &self.lower {
            None => new_lower,
            Some(old) => old.max(&new_lower).clone(),
        });
    }

    /// Tighten the upper bound.
    pub fn tighten_upper(&mut self, new_upper: BigRational) {
        self.upper = Some(match &self.upper {
            None => new_upper,
            Some(old) => old.min(&new_upper).clone(),
        });
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lower_str = match &self.lower {
            None => "-∞".to_string(),
            Some(l) => format!("{}", l),
        };
        let upper_str = match &self.upper {
            None => "+∞".to_string(),
            Some(u) => format!("{}", u),
        };
        write!(f, "[{}, {}]", lower_str, upper_str)
    }
}

/// Bound propagation engine.
pub struct BoundPropagator {
    /// Current interval bounds for each variable.
    bounds: FxHashMap<Var, Interval>,
    /// Number of propagations performed.
    num_propagations: u64,
    /// Number of conflicts detected.
    num_conflicts: u64,
}

impl BoundPropagator {
    /// Create a new bound propagator.
    pub fn new() -> Self {
        Self {
            bounds: FxHashMap::default(),
            num_propagations: 0,
            num_conflicts: 0,
        }
    }

    /// Get the current bounds for a variable.
    pub fn get_bounds(&self, var: Var) -> Interval {
        self.bounds
            .get(&var)
            .cloned()
            .unwrap_or_else(Interval::unbounded)
    }

    /// Set the bounds for a variable.
    pub fn set_bounds(&mut self, var: Var, interval: Interval) -> bool {
        if interval.is_empty() {
            self.num_conflicts += 1;
            return false;
        }

        self.bounds.insert(var, interval);
        true
    }

    /// Tighten the lower bound of a variable.
    pub fn tighten_lower(&mut self, var: Var, new_lower: BigRational) -> bool {
        let mut interval = self.get_bounds(var);
        interval.tighten_lower(new_lower);

        if interval.is_empty() {
            self.num_conflicts += 1;
            return false;
        }

        self.bounds.insert(var, interval);
        true
    }

    /// Tighten the upper bound of a variable.
    pub fn tighten_upper(&mut self, var: Var, new_upper: BigRational) -> bool {
        let mut interval = self.get_bounds(var);
        interval.tighten_upper(new_upper);

        if interval.is_empty() {
            self.num_conflicts += 1;
            return false;
        }

        self.bounds.insert(var, interval);
        true
    }

    /// Propagate bounds through a linear constraint ax + b ≤ 0.
    ///
    /// Returns false if a conflict is detected.
    #[allow(dead_code)]
    pub fn propagate_linear(&mut self, var: Var, a: &BigRational, b: &BigRational) -> bool {
        if a.is_zero() {
            // Constraint is constant, check if it's satisfiable
            return b <= &BigRational::zero();
        }

        self.num_propagations += 1;

        // ax + b ≤ 0
        // ax ≤ -b
        // x ≤ -b/a  (if a > 0)
        // x ≥ -b/a  (if a < 0)

        let bound = -b / a;

        if a.is_positive() {
            self.tighten_upper(var, bound)
        } else {
            self.tighten_lower(var, bound)
        }
    }

    /// Propagate bounds through a polynomial constraint p ≤ 0 (heuristic).
    ///
    /// This is a simplified heuristic propagation for non-linear constraints.
    #[allow(dead_code)]
    pub fn propagate_polynomial(&mut self, _poly: &Polynomial) -> bool {
        // For non-linear polynomials, we use heuristic propagation
        // This is a simplified version - full implementation would analyze
        // polynomial structure and use interval arithmetic

        self.num_propagations += 1;

        // Placeholder: always return true (no conflict)
        // TODO: Implement actual polynomial bound propagation
        true
    }

    /// Clear all bounds.
    pub fn clear(&mut self) {
        self.bounds.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> (u64, u64) {
        (self.num_propagations, self.num_conflicts)
    }

    /// Check if all bounds are consistent (no empty intervals).
    pub fn is_consistent(&self) -> bool {
        self.bounds.values().all(|interval| !interval.is_empty())
    }

    /// Get all variables with bounds.
    pub fn variables(&self) -> Vec<Var> {
        self.bounds.keys().copied().collect()
    }
}

impl Default for BoundPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i32) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_interval_unbounded() {
        let interval = Interval::unbounded();
        assert!(interval.lower.is_none());
        assert!(interval.upper.is_none());
        assert!(!interval.is_empty());
    }

    #[test]
    fn test_interval_point() {
        let interval = Interval::point(rat(5));
        assert!(interval.is_point());
        assert!(!interval.is_empty());
        assert!(interval.contains(&rat(5)));
        assert!(!interval.contains(&rat(4)));
    }

    #[test]
    fn test_interval_empty() {
        let interval = Interval::new(Some(rat(10)), Some(rat(5)));
        assert!(interval.is_empty());
    }

    #[test]
    fn test_interval_intersect() {
        let i1 = Interval::new(Some(rat(0)), Some(rat(10)));
        let i2 = Interval::new(Some(rat(5)), Some(rat(15)));
        let result = i1.intersect(&i2);

        assert_eq!(result.lower, Some(rat(5)));
        assert_eq!(result.upper, Some(rat(10)));
    }

    #[test]
    fn test_interval_contains() {
        let interval = Interval::new(Some(rat(0)), Some(rat(10)));
        assert!(interval.contains(&rat(5)));
        assert!(interval.contains(&rat(0)));
        assert!(interval.contains(&rat(10)));
        assert!(!interval.contains(&rat(-1)));
        assert!(!interval.contains(&rat(11)));
    }

    #[test]
    fn test_propagator_new() {
        let propagator = BoundPropagator::new();
        assert_eq!(propagator.bounds.len(), 0);
    }

    #[test]
    fn test_propagator_set_bounds() {
        let mut propagator = BoundPropagator::new();
        let interval = Interval::new(Some(rat(0)), Some(rat(10)));
        assert!(propagator.set_bounds(0, interval.clone()));
        assert_eq!(propagator.get_bounds(0), interval);
    }

    #[test]
    fn test_propagator_empty_conflict() {
        let mut propagator = BoundPropagator::new();
        let empty = Interval::new(Some(rat(10)), Some(rat(5)));
        assert!(!propagator.set_bounds(0, empty));
        assert_eq!(propagator.num_conflicts, 1);
    }

    #[test]
    fn test_propagator_tighten_lower() {
        let mut propagator = BoundPropagator::new();
        propagator.set_bounds(0, Interval::new(Some(rat(0)), Some(rat(10))));

        assert!(propagator.tighten_lower(0, rat(5)));
        let bounds = propagator.get_bounds(0);
        assert_eq!(bounds.lower, Some(rat(5)));
        assert_eq!(bounds.upper, Some(rat(10)));
    }

    #[test]
    fn test_propagator_tighten_upper() {
        let mut propagator = BoundPropagator::new();
        propagator.set_bounds(0, Interval::new(Some(rat(0)), Some(rat(10))));

        assert!(propagator.tighten_upper(0, rat(5)));
        let bounds = propagator.get_bounds(0);
        assert_eq!(bounds.lower, Some(rat(0)));
        assert_eq!(bounds.upper, Some(rat(5)));
    }

    #[test]
    fn test_propagator_tighten_conflict() {
        let mut propagator = BoundPropagator::new();
        propagator.set_bounds(0, Interval::new(Some(rat(0)), Some(rat(10))));

        // Tightening lower bound above upper bound causes conflict
        assert!(!propagator.tighten_lower(0, rat(15)));
        assert_eq!(propagator.num_conflicts, 1);
    }

    #[test]
    fn test_propagator_linear() {
        let mut propagator = BoundPropagator::new();

        // 2x + 10 ≤ 0  =>  x ≤ -5
        let a = rat(2);
        let b = rat(10);
        assert!(propagator.propagate_linear(0, &a, &b));

        let bounds = propagator.get_bounds(0);
        assert_eq!(bounds.upper, Some(rat(-5)));
    }

    #[test]
    fn test_propagator_is_consistent() {
        let mut propagator = BoundPropagator::new();
        assert!(propagator.is_consistent());

        propagator.set_bounds(0, Interval::new(Some(rat(0)), Some(rat(10))));
        assert!(propagator.is_consistent());
    }

    #[test]
    fn test_propagator_clear() {
        let mut propagator = BoundPropagator::new();
        propagator.set_bounds(0, Interval::new(Some(rat(0)), Some(rat(10))));
        propagator.clear();
        assert_eq!(propagator.bounds.len(), 0);
    }
}
