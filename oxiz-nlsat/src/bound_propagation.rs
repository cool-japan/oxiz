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
use num_traits::{One, Signed, Zero};
use oxiz_math::polynomial::{Monomial, Polynomial, Var};
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

    /// Propagate bounds through a polynomial constraint p ≤ 0.
    ///
    /// Uses interval arithmetic to:
    /// 1. Detect infeasibility: if the interval of p is entirely above 0, return `false`.
    /// 2. Tighten bounds on degree-1 variables: for a term `a·v + rest(others) ≤ 0`,
    ///    compute the interval of `rest` and derive a new bound for `v`.
    ///
    /// For monomials containing variables with no recorded bounds (unbounded), their
    /// interval contribution is treated as `(-∞, +∞)`, which prevents conflict
    /// detection but is always sound.  Returns `false` only when a definitive
    /// contradiction is found; returns `true` otherwise (no-conflict / tightened).
    #[allow(dead_code)]
    pub fn propagate_polynomial(&mut self, poly: &Polynomial) -> bool {
        self.num_propagations += 1;

        // If the polynomial is identically zero, the constraint 0 ≤ 0 is trivially satisfied.
        if poly.is_zero() {
            return true;
        }

        // If the polynomial is a non-zero constant (e.g. 5), the constraint is c ≤ 0.
        // It is infeasible iff c > 0.
        if poly.is_constant() {
            let c = poly.constant_value();
            if c > BigRational::zero() {
                self.num_conflicts += 1;
                return false;
            }
            return true;
        }

        // Compute the interval of the whole polynomial via interval arithmetic.
        let poly_interval = self.eval_polynomial_interval(poly);

        // If the entire interval is above 0, i.e. lower bound > 0, then p ≤ 0 is infeasible.
        if let Some(ref lb) = poly_interval.lower
            && lb > &BigRational::zero()
        {
            self.num_conflicts += 1;
            return false;
        }

        // Attempt linear-variable bound tightening.
        // For each variable `v` that appears at degree exactly 1 in the polynomial:
        //   isolate its contribution as  a·v + rest(others) ≤ 0
        //   then derive a bound on v from the interval of rest(others).
        let poly_terms = poly.terms();
        for term in poly_terms {
            let vps = term.monomial.vars();
            // Only handle the linear-in-one-variable case: exactly one VarPower with power == 1.
            if vps.len() != 1 || vps[0].power != 1 {
                continue;
            }
            let var = vps[0].var;
            let a = &term.coeff;

            // Build the rest interval: sum intervals of all other terms.
            let rest_interval = self.eval_polynomial_interval_except(poly, var, 1);

            // Derive a bound on v.
            //   a > 0:  a·v ≤ -rest_lower  →  v ≤ (-rest_lower)/a  (upper bound)
            //   a < 0:  a·v ≤ -rest_upper  →  v ≥ (-rest_upper)/a  (sign flips)
            if a.is_positive() {
                if let Some(ref rest_lb) = rest_interval.lower {
                    let new_upper = (-rest_lb.clone()) / a.clone();
                    if !self.tighten_upper(var, new_upper) {
                        return false;
                    }
                }
            } else if a.is_negative()
                && let Some(ref rest_ub) = rest_interval.upper
            {
                let new_lower = (-rest_ub.clone()) / a.clone();
                if !self.tighten_lower(var, new_lower) {
                    return false;
                }
            }
        }

        true
    }

    /// Evaluate the interval of a polynomial given current variable bounds.
    ///
    /// Uses interval arithmetic: sums the intervals of each monomial.
    fn eval_polynomial_interval(&self, poly: &Polynomial) -> Interval {
        let mut sum = Interval::point(BigRational::zero());
        for term in poly.terms() {
            let monomial_interval = self.eval_monomial_interval(&term.monomial, &term.coeff);
            sum = interval_add(&sum, &monomial_interval);
        }
        sum
    }

    /// Evaluate the interval of the polynomial, excluding the term for `skip_var^skip_power`.
    fn eval_polynomial_interval_except(
        &self,
        poly: &Polynomial,
        skip_var: Var,
        skip_power: u32,
    ) -> Interval {
        let mut sum = Interval::point(BigRational::zero());
        for term in poly.terms() {
            let vps = term.monomial.vars();
            if vps.len() == 1 && vps[0].var == skip_var && vps[0].power == skip_power {
                continue;
            }
            let monomial_interval = self.eval_monomial_interval(&term.monomial, &term.coeff);
            sum = interval_add(&sum, &monomial_interval);
        }
        sum
    }

    /// Evaluate the interval of a single monomial `coeff · ∏ vᵢ^pᵢ`.
    fn eval_monomial_interval(&self, monomial: &Monomial, coeff: &BigRational) -> Interval {
        let mut interval = Interval::point(coeff.clone());
        for vp in monomial.vars() {
            let var_interval = self.eval_var_power_interval(vp.var, vp.power);
            interval = interval_mul(&interval, &var_interval);
        }
        interval
    }

    /// Compute the interval of `v^p` from the variable's current bounds.
    ///
    /// For power 0 returns [1, 1].  For power 1 returns the variable's interval.
    /// For higher powers uses monotonicity / sign analysis.  Unbounded → unbounded.
    fn eval_var_power_interval(&self, var: Var, power: u32) -> Interval {
        if power == 0 {
            return Interval::point(BigRational::one());
        }
        let var_bounds = self.get_bounds(var);
        if power == 1 {
            return var_bounds;
        }
        let (lb, ub) = match (var_bounds.lower, var_bounds.upper) {
            (Some(l), Some(u)) => (l, u),
            _ => return Interval::unbounded(),
        };
        let lb_p = rational_pow(&lb, power);
        let ub_p = rational_pow(&ub, power);
        if power.is_multiple_of(2) {
            let zero = BigRational::zero();
            if lb <= zero && zero <= ub {
                // Interval straddles 0: even power minimum is 0.
                let hi = lb_p.max(ub_p);
                Interval::new(Some(zero), Some(hi))
            } else {
                let lo = lb_p.clone().min(ub_p.clone());
                let hi = lb_p.max(ub_p);
                Interval::new(Some(lo), Some(hi))
            }
        } else {
            // Odd power: monotonically increasing.
            let lo = lb_p.clone().min(ub_p.clone());
            let hi = lb_p.max(ub_p);
            Interval::new(Some(lo), Some(hi))
        }
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

// ---------------------------------------------------------------------------
// Interval arithmetic helpers (module-private)
// ---------------------------------------------------------------------------

/// Add two intervals: [a, b] + [c, d] = [a+c, b+d].
fn interval_add(lhs: &Interval, rhs: &Interval) -> Interval {
    let lower = match (&lhs.lower, &rhs.lower) {
        (Some(l1), Some(l2)) => Some(l1.clone() + l2.clone()),
        _ => None,
    };
    let upper = match (&lhs.upper, &rhs.upper) {
        (Some(u1), Some(u2)) => Some(u1.clone() + u2.clone()),
        _ => None,
    };
    Interval::new(lower, upper)
}

/// Multiply two intervals via the four-corner method.
///
/// [a, b] × [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
///
/// If either interval is unbounded, the result is unbounded (safe over-approximation).
fn interval_mul(lhs: &Interval, rhs: &Interval) -> Interval {
    match (&lhs.lower, &lhs.upper, &rhs.lower, &rhs.upper) {
        (Some(l1), Some(u1), Some(l2), Some(u2)) => {
            let corners = [
                l1.clone() * l2.clone(),
                l1.clone() * u2.clone(),
                u1.clone() * l2.clone(),
                u1.clone() * u2.clone(),
            ];
            let lo = corners
                .iter()
                .min_by(|a, b| a.cmp(b))
                .cloned()
                .unwrap_or_else(BigRational::zero);
            let hi = corners
                .iter()
                .max_by(|a, b| a.cmp(b))
                .cloned()
                .unwrap_or_else(BigRational::zero);
            Interval::new(Some(lo), Some(hi))
        }
        _ => Interval::unbounded(),
    }
}

/// Compute `base^exp` for a `BigRational` base and `u32` exponent using fast exponentiation.
fn rational_pow(base: &BigRational, exp: u32) -> BigRational {
    if exp == 0 {
        return BigRational::one();
    }
    let mut result = BigRational::one();
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result *= b.clone();
        }
        b = b.clone() * b;
        e >>= 1;
    }
    result
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

    // -----------------------------------------------------------------------
    // Tests for propagate_polynomial
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagate_polynomial_zero_poly_no_conflict() {
        // 0 ≤ 0 is always satisfiable.
        let mut propagator = BoundPropagator::new();
        let zero = Polynomial::zero();
        assert!(propagator.propagate_polynomial(&zero));
    }

    #[test]
    fn test_propagate_polynomial_positive_constant_conflict() {
        // 5 ≤ 0 is infeasible.
        let mut propagator = BoundPropagator::new();
        let five = Polynomial::constant(BigRational::from_integer(BigInt::from(5)));
        assert!(!propagator.propagate_polynomial(&five));
        assert_eq!(propagator.num_conflicts, 1);
    }

    #[test]
    fn test_propagate_polynomial_negative_constant_no_conflict() {
        // -3 ≤ 0 is always satisfied.
        let mut propagator = BoundPropagator::new();
        let neg_three = Polynomial::constant(BigRational::from_integer(BigInt::from(-3)));
        assert!(propagator.propagate_polynomial(&neg_three));
        assert_eq!(propagator.num_conflicts, 0);
    }

    #[test]
    fn test_propagate_polynomial_linear_tightens_upper_bound() {
        // Constraint: x + 3 ≤ 0  →  x ≤ -3.
        // With no prior bounds, after propagation x should have upper bound -3.
        // rest_interval for x is [3, 3] (the constant part),
        // so new_upper = (-3) / 1 = -3.
        let mut propagator = BoundPropagator::new();
        let x_poly = Polynomial::from_var(0);
        let three_poly = Polynomial::constant(BigRational::from_integer(BigInt::from(3)));
        let poly = Polynomial::add(&x_poly, &three_poly);

        let ok = propagator.propagate_polynomial(&poly);
        assert!(ok);
        let bounds = propagator.get_bounds(0);
        assert_eq!(
            bounds.upper,
            Some(BigRational::from_integer(BigInt::from(-3)))
        );
    }

    #[test]
    fn test_propagate_polynomial_fully_bounded_infeasible() {
        // Constraint: x ≤ 0 with x bounded to [5, 10].
        // x's interval is [5, 10], polynomial interval lower = 5 > 0 → conflict.
        let mut propagator = BoundPropagator::new();
        propagator.set_bounds(0, Interval::new(Some(rat(5)), Some(rat(10))));
        let x_poly = Polynomial::from_var(0);
        assert!(!propagator.propagate_polynomial(&x_poly));
        assert!(propagator.num_conflicts > 0);
    }

    #[test]
    fn test_propagate_polynomial_unbounded_var_no_false_conflict() {
        // x ≤ 0 with x unbounded: we cannot determine infeasibility, return true.
        let mut propagator = BoundPropagator::new();
        let x_poly = Polynomial::from_var(0);
        assert!(propagator.propagate_polynomial(&x_poly));
    }
}
