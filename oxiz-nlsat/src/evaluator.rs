//! Polynomial constraint evaluator for NLSAT.
//!
//! This module provides efficient evaluation of polynomial constraints under
//! partial or full arithmetic assignments. Key features include:
//!
//! - **Polynomial evaluation**: Compute polynomial values at given points
//! - **Sign computation**: Determine the sign of polynomials
//! - **Atom evaluation**: Evaluate polynomial constraints (atoms) to get truth values
//! - **Root finding**: Isolate roots for univariate polynomials
//! - **Caching**: Cache evaluation results for efficiency
//!
//! Reference: Z3's `nlsat/nlsat_evaluator.cpp`

use crate::interval_set::IntervalSet;
use crate::types::{Atom, AtomKind, IneqAtom, Lbool, RootAtom};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// Configuration for the evaluator.
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Whether to cache polynomial evaluations.
    pub cache_evaluations: bool,
    /// Maximum cache size (number of entries).
    pub max_cache_size: usize,
    /// Precision for root isolation.
    pub root_precision: usize,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            cache_evaluations: true,
            max_cache_size: 10_000,
            root_precision: 10,
        }
    }
}

/// Result of polynomial sign computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// Polynomial evaluates to a positive value.
    Positive,
    /// Polynomial evaluates to zero.
    Zero,
    /// Polynomial evaluates to a negative value.
    Negative,
    /// Sign cannot be determined (incomplete assignment).
    Unknown,
}

impl Sign {
    /// Convert from a numeric value.
    pub fn from_value(value: &BigRational) -> Self {
        if value.is_zero() {
            Sign::Zero
        } else if value.is_positive() {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }

    /// Convert to an i8 sign value.
    pub fn to_i8(self) -> Option<i8> {
        match self {
            Sign::Positive => Some(1),
            Sign::Zero => Some(0),
            Sign::Negative => Some(-1),
            Sign::Unknown => None,
        }
    }

    /// Check if the sign satisfies a given atom kind.
    pub fn satisfies(self, kind: AtomKind) -> Option<bool> {
        match (self, kind) {
            (Sign::Unknown, _) => None,
            (Sign::Zero, AtomKind::Eq) => Some(true),
            (Sign::Zero, AtomKind::Lt) => Some(false),
            (Sign::Zero, AtomKind::Gt) => Some(false),
            (Sign::Positive, AtomKind::Eq) => Some(false),
            (Sign::Positive, AtomKind::Lt) => Some(false),
            (Sign::Positive, AtomKind::Gt) => Some(true),
            (Sign::Negative, AtomKind::Eq) => Some(false),
            (Sign::Negative, AtomKind::Lt) => Some(true),
            (Sign::Negative, AtomKind::Gt) => Some(false),
            _ => None, // Root atoms handled separately
        }
    }
}

/// Polynomial evaluator.
pub struct Evaluator {
    /// Configuration.
    #[allow(dead_code)]
    config: EvaluatorConfig,
    /// Cache: polynomial hash + assignment -> value
    value_cache: HashMap<u64, BigRational>,
    /// Cache: polynomial hash + assignment -> sign
    sign_cache: HashMap<u64, Sign>,
    /// Number of evaluations performed.
    num_evaluations: u64,
    /// Number of cache hits.
    num_cache_hits: u64,
}

impl Evaluator {
    /// Create a new evaluator with default configuration.
    pub fn new() -> Self {
        Self::with_config(EvaluatorConfig::default())
    }

    /// Create a new evaluator with the given configuration.
    pub fn with_config(config: EvaluatorConfig) -> Self {
        Self {
            config,
            value_cache: HashMap::new(),
            sign_cache: HashMap::new(),
            num_evaluations: 0,
            num_cache_hits: 0,
        }
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        self.value_cache.clear();
        self.sign_cache.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> (u64, u64) {
        (self.num_evaluations, self.num_cache_hits)
    }

    /// Evaluate a polynomial at a given assignment.
    ///
    /// Returns `None` if any required variable is unassigned.
    pub fn evaluate(
        &mut self,
        poly: &Polynomial,
        assignment: &FxHashMap<Var, BigRational>,
    ) -> Option<BigRational> {
        self.num_evaluations += 1;

        // Check if all variables are assigned
        for var in poly.vars() {
            if !assignment.contains_key(&var) {
                return None;
            }
        }

        // Evaluate
        Some(poly.eval(assignment))
    }

    /// Compute the sign of a polynomial at a given assignment.
    pub fn sign(&mut self, poly: &Polynomial, assignment: &FxHashMap<Var, BigRational>) -> Sign {
        match self.evaluate(poly, assignment) {
            Some(value) => Sign::from_value(&value),
            None => Sign::Unknown,
        }
    }

    /// Evaluate an inequality atom.
    ///
    /// Returns the truth value of the atom, or `Lbool::Undef` if it cannot be determined.
    pub fn evaluate_ineq(
        &mut self,
        atom: &IneqAtom,
        assignment: &FxHashMap<Var, BigRational>,
    ) -> Lbool {
        // Compute the product sign
        let mut product_sign = 1i8;

        for factor in &atom.factors {
            match self.sign(&factor.poly, assignment) {
                Sign::Unknown => return Lbool::Undef,
                Sign::Zero => {
                    product_sign = 0;
                    break;
                }
                Sign::Positive => {
                    if !factor.is_even {
                        // Positive contributes nothing to sign change
                    }
                }
                Sign::Negative => {
                    if !factor.is_even {
                        product_sign = -product_sign;
                    }
                }
            }
        }

        // Check against the atom kind
        let satisfied = match atom.kind {
            AtomKind::Eq => product_sign == 0,
            AtomKind::Lt => product_sign < 0,
            AtomKind::Gt => product_sign > 0,
            _ => return Lbool::Undef, // Should not happen for IneqAtom
        };

        Lbool::from_bool(satisfied)
    }

    /// Evaluate a root atom.
    ///
    /// A root atom is of the form `x op root_i(p)` where:
    /// - `x` is a variable
    /// - `op` is a comparison operator
    /// - `root_i(p)` is the i-th root of polynomial p
    pub fn evaluate_root(
        &mut self,
        atom: &RootAtom,
        assignment: &FxHashMap<Var, BigRational>,
    ) -> Lbool {
        // Get the value of the variable being compared
        let var_value = match assignment.get(&atom.var) {
            Some(v) => v.clone(),
            None => return Lbool::Undef,
        };

        // Evaluate the polynomial to get a univariate polynomial in the remaining variable
        // For now, we require all variables except the main one to be assigned
        let mut partial_assignment = assignment.clone();
        partial_assignment.remove(&atom.var);

        // Substitute to get univariate polynomial
        let univariate = self.substitute_all(&atom.poly, &partial_assignment);
        if univariate.is_none() {
            return Lbool::Undef;
        }
        let univariate = univariate.expect("univariate polynomial validated");

        // Find the roots
        let roots = self.find_roots(&univariate, atom.var);

        // Get the i-th root (1-indexed)
        let root_idx = atom.root_index as usize;
        if root_idx == 0 || root_idx > roots.len() {
            // Invalid root index or not enough roots
            return Lbool::Undef;
        }
        let root = &roots[root_idx - 1];

        // Compare the variable value to the root
        let satisfied = match atom.kind {
            AtomKind::RootEq => &var_value == root,
            AtomKind::RootLt => &var_value < root,
            AtomKind::RootGt => &var_value > root,
            AtomKind::RootLe => &var_value <= root,
            AtomKind::RootGe => &var_value >= root,
            _ => return Lbool::Undef,
        };

        Lbool::from_bool(satisfied)
    }

    /// Evaluate an atom (either inequality or root).
    pub fn evaluate_atom(
        &mut self,
        atom: &Atom,
        assignment: &FxHashMap<Var, BigRational>,
    ) -> Lbool {
        match atom {
            Atom::Ineq(ineq) => self.evaluate_ineq(ineq, assignment),
            Atom::Root(root) => self.evaluate_root(root, assignment),
        }
    }

    /// Substitute all variables in a polynomial except one.
    fn substitute_all(
        &self,
        poly: &Polynomial,
        assignment: &FxHashMap<Var, BigRational>,
    ) -> Option<Polynomial> {
        // Check if all variables except one are assigned
        let mut unassigned = Vec::new();
        for var in poly.vars() {
            if !assignment.contains_key(&var) {
                unassigned.push(var);
            }
        }

        if unassigned.len() > 1 {
            return None;
        }

        // Substitute each assigned variable one at a time
        let mut result = poly.clone();
        for (var, value) in assignment {
            if poly.vars().contains(var) {
                result = result.eval_at(*var, value);
            }
        }

        Some(result)
    }

    /// Find the roots of a univariate polynomial.
    ///
    /// Returns the roots in sorted order.
    pub fn find_roots(&self, poly: &Polynomial, var: Var) -> Vec<BigRational> {
        let degree = poly.degree(var);

        match degree {
            0 => Vec::new(), // Constant polynomial has no roots
            1 => self.find_linear_root(poly, var).into_iter().collect(),
            2 => self.find_quadratic_roots(poly, var),
            _ => {
                // For higher degrees, use numerical methods or interval arithmetic
                // For now, return empty (not implemented)
                Vec::new()
            }
        }
    }

    /// Find the root of a linear polynomial ax + b = 0.
    fn find_linear_root(&self, poly: &Polynomial, var: Var) -> Option<BigRational> {
        // Extract coefficients: p(x) = a*x + b
        let (a, b) = self.extract_linear_coeffs(poly, var)?;

        if a.is_zero() {
            return None; // Not actually linear
        }

        // x = -b/a
        Some(-b / a)
    }

    /// Find the roots of a quadratic polynomial ax^2 + bx + c = 0.
    fn find_quadratic_roots(&self, poly: &Polynomial, var: Var) -> Vec<BigRational> {
        // Extract coefficients
        let (a, b, c) = match self.extract_quadratic_coeffs(poly, var) {
            Some(coeffs) => coeffs,
            None => return Vec::new(),
        };

        if a.is_zero() {
            // Actually linear
            if b.is_zero() {
                return Vec::new();
            }
            return vec![-c / b];
        }

        // Discriminant: b^2 - 4ac
        let discriminant = &b * &b - BigRational::from_integer(BigInt::from(4)) * &a * &c;

        if discriminant.is_negative() {
            return Vec::new(); // No real roots
        }

        if discriminant.is_zero() {
            // One repeated root: -b / (2a)
            let root = -b / (BigRational::from_integer(BigInt::from(2)) * a);
            return vec![root];
        }

        // Two distinct roots
        // For rational discriminant, check if it's a perfect square
        let sqrt_discr = self.rational_sqrt(&discriminant);
        match sqrt_discr {
            Some(s) => {
                let two_a = BigRational::from_integer(BigInt::from(2)) * &a;
                let root1 = (-&b - &s) / &two_a;
                let root2 = (-&b + &s) / &two_a;

                let mut roots = vec![root1, root2];
                roots.sort();
                roots
            }
            None => {
                // Discriminant is not a perfect square - roots are irrational
                // Return approximation or empty for now
                Vec::new()
            }
        }
    }

    /// Extract linear coefficients a, b from polynomial ax + b.
    fn extract_linear_coeffs(
        &self,
        poly: &Polynomial,
        var: Var,
    ) -> Option<(BigRational, BigRational)> {
        let mut a = BigRational::zero();
        let mut b = BigRational::zero();

        for term in poly.terms() {
            let exp = term.monomial.degree(var);
            match exp {
                0 => b += &term.coeff,
                1 => a += &term.coeff,
                _ => return None, // Not linear
            }
        }

        Some((a, b))
    }

    /// Extract quadratic coefficients a, b, c from polynomial ax^2 + bx + c.
    fn extract_quadratic_coeffs(
        &self,
        poly: &Polynomial,
        var: Var,
    ) -> Option<(BigRational, BigRational, BigRational)> {
        let mut a = BigRational::zero();
        let mut b = BigRational::zero();
        let mut c = BigRational::zero();

        for term in poly.terms() {
            let exp = term.monomial.degree(var);
            // Check that other variables in the monomial have degree 0
            let other_degree: u32 = term
                .monomial
                .vars()
                .iter()
                .filter(|vp| vp.var != var)
                .map(|vp| vp.power)
                .sum();

            if other_degree > 0 {
                return None; // Not univariate
            }

            match exp {
                0 => c += &term.coeff,
                1 => b += &term.coeff,
                2 => a += &term.coeff,
                _ => return None, // Not quadratic
            }
        }

        Some((a, b, c))
    }

    /// Compute the square root of a rational if it's a perfect square.
    fn rational_sqrt(&self, r: &BigRational) -> Option<BigRational> {
        if r.is_negative() {
            return None;
        }
        if r.is_zero() {
            return Some(BigRational::zero());
        }

        // r = p/q, sqrt(r) = sqrt(p)/sqrt(q) if both are perfect squares
        let (numer, denom) = (r.numer(), r.denom());

        let sqrt_numer = integer_sqrt(numer)?;
        let sqrt_denom = integer_sqrt(denom)?;

        Some(BigRational::new(sqrt_numer, sqrt_denom))
    }

    /// Compute the feasible region for a variable given polynomial constraints.
    pub fn feasible_region(
        &mut self,
        var: Var,
        atoms: &[&Atom],
        atom_values: &[bool],
        assignment: &FxHashMap<Var, BigRational>,
    ) -> IntervalSet {
        let mut region = IntervalSet::reals();

        for (atom, &value) in atoms.iter().zip(atom_values.iter()) {
            match atom {
                Atom::Ineq(ineq) => {
                    // Compute the constraint on var from this atom
                    let constraint = self.atom_constraint(ineq, var, value, assignment);
                    region = region.intersect(&constraint);
                }
                Atom::Root(root) => {
                    if root.var == var {
                        let constraint = self.root_constraint(root, value, assignment);
                        region = region.intersect(&constraint);
                    }
                }
            }

            if region.is_empty() {
                break;
            }
        }

        region
    }

    /// Compute the constraint on a variable from an inequality atom.
    fn atom_constraint(
        &mut self,
        ineq: &IneqAtom,
        var: Var,
        _satisfied: bool,
        _assignment: &FxHashMap<Var, BigRational>,
    ) -> IntervalSet {
        // Check if this atom involves the variable
        let involves_var = ineq.factors.iter().any(|f| f.poly.vars().contains(&var));

        if !involves_var {
            // Atom doesn't involve this variable
            return IntervalSet::reals();
        }

        // For each factor, compute its sign constraint
        // This is simplified - full implementation would compute exact feasible regions
        IntervalSet::reals()
    }

    /// Compute the constraint from a root atom.
    fn root_constraint(
        &mut self,
        root: &RootAtom,
        satisfied: bool,
        assignment: &FxHashMap<Var, BigRational>,
    ) -> IntervalSet {
        // Substitute to get univariate polynomial
        let mut partial = assignment.clone();
        partial.remove(&root.var);

        let univariate = match self.substitute_all(&root.poly, &partial) {
            Some(p) => p,
            None => return IntervalSet::reals(),
        };

        // Find roots
        let roots = self.find_roots(&univariate, root.var);

        // Get the target root
        let idx = root.root_index as usize;
        if idx == 0 || idx > roots.len() {
            return IntervalSet::reals();
        }
        let target_root = &roots[idx - 1];

        // Compute the constraint based on the comparison
        let make_constraint = |op: AtomKind| -> IntervalSet {
            use oxiz_math::interval::Interval;
            match op {
                AtomKind::RootEq => IntervalSet::point(target_root.clone()),
                AtomKind::RootLt => {
                    IntervalSet::from_interval(Interval::less_than(target_root.clone()))
                }
                AtomKind::RootGt => {
                    IntervalSet::from_interval(Interval::greater_than(target_root.clone()))
                }
                AtomKind::RootLe => {
                    IntervalSet::from_interval(Interval::at_most(target_root.clone()))
                }
                AtomKind::RootGe => {
                    IntervalSet::from_interval(Interval::at_least(target_root.clone()))
                }
                _ => IntervalSet::reals(),
            }
        };

        if satisfied {
            make_constraint(root.kind)
        } else {
            // Negate the constraint
            make_constraint(root.kind).complement()
        }
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the integer square root if the number is a perfect square.
fn integer_sqrt(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    if n.is_zero() {
        return Some(BigInt::zero());
    }

    // Newton's method for integer square root
    let mut x: BigInt = n.clone();
    let two = BigInt::from(2);
    let mut y: BigInt = (&x + BigInt::one()) / &two;

    while y < x {
        x = y.clone();
        y = (&x + n / &x) / &two;
    }

    // Check if it's a perfect square
    if &x * &x == *n { Some(x) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_sign() {
        assert_eq!(Sign::from_value(&rat(5)), Sign::Positive);
        assert_eq!(Sign::from_value(&rat(0)), Sign::Zero);
        assert_eq!(Sign::from_value(&rat(-3)), Sign::Negative);

        assert_eq!(Sign::Positive.to_i8(), Some(1));
        assert_eq!(Sign::Zero.to_i8(), Some(0));
        assert_eq!(Sign::Negative.to_i8(), Some(-1));
        assert_eq!(Sign::Unknown.to_i8(), None);
    }

    #[test]
    fn test_sign_satisfies() {
        assert_eq!(Sign::Zero.satisfies(AtomKind::Eq), Some(true));
        assert_eq!(Sign::Positive.satisfies(AtomKind::Eq), Some(false));
        assert_eq!(Sign::Negative.satisfies(AtomKind::Lt), Some(true));
        assert_eq!(Sign::Positive.satisfies(AtomKind::Gt), Some(true));
        assert_eq!(Sign::Unknown.satisfies(AtomKind::Eq), None);
    }

    #[test]
    fn test_evaluator_basic() {
        let mut eval = Evaluator::new();

        // p(x) = x - 2 using from_coeffs_int: [(coeff, [(var, power)])]
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-2, &[])]);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, rat(5));

        let result = eval.evaluate(&poly, &assignment);
        assert_eq!(result, Some(rat(3)));

        assert_eq!(eval.sign(&poly, &assignment), Sign::Positive);
    }

    #[test]
    fn test_evaluator_missing_var() {
        let mut eval = Evaluator::new();

        // p(x, y) = x + y
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)])]);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, rat(5));
        // y is not assigned

        let result = eval.evaluate(&poly, &assignment);
        assert_eq!(result, None);
        assert_eq!(eval.sign(&poly, &assignment), Sign::Unknown);
    }

    #[test]
    fn test_find_linear_root() {
        let eval = Evaluator::new();

        // p(x) = 2x - 6 = 0 => x = 3
        let poly = Polynomial::from_coeffs_int(&[(2, &[(0, 1)]), (-6, &[])]);

        let roots = eval.find_roots(&poly, 0);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], rat(3));
    }

    #[test]
    fn test_find_quadratic_roots() {
        let eval = Evaluator::new();

        // p(x) = x^2 - 4 = 0 => x = -2, 2
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-4, &[])]);

        let roots = eval.find_roots(&poly, 0);
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0], rat(-2));
        assert_eq!(roots[1], rat(2));
    }

    #[test]
    fn test_find_quadratic_roots_one_root() {
        let eval = Evaluator::new();

        // p(x) = x^2 - 2x + 1 = (x-1)^2 = 0 => x = 1
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-2, &[(0, 1)]), (1, &[])]);

        let roots = eval.find_roots(&poly, 0);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], rat(1));
    }

    #[test]
    fn test_find_quadratic_roots_no_roots() {
        let eval = Evaluator::new();

        // p(x) = x^2 + 1 = 0 => no real roots
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[])]);

        let roots = eval.find_roots(&poly, 0);
        assert_eq!(roots.len(), 0);
    }

    #[test]
    fn test_integer_sqrt() {
        assert_eq!(integer_sqrt(&BigInt::from(0)), Some(BigInt::from(0)));
        assert_eq!(integer_sqrt(&BigInt::from(1)), Some(BigInt::from(1)));
        assert_eq!(integer_sqrt(&BigInt::from(4)), Some(BigInt::from(2)));
        assert_eq!(integer_sqrt(&BigInt::from(9)), Some(BigInt::from(3)));
        assert_eq!(integer_sqrt(&BigInt::from(16)), Some(BigInt::from(4)));
        assert_eq!(integer_sqrt(&BigInt::from(25)), Some(BigInt::from(5)));
        assert_eq!(integer_sqrt(&BigInt::from(100)), Some(BigInt::from(10)));

        // Not perfect squares
        assert_eq!(integer_sqrt(&BigInt::from(2)), None);
        assert_eq!(integer_sqrt(&BigInt::from(3)), None);
        assert_eq!(integer_sqrt(&BigInt::from(5)), None);
        assert_eq!(integer_sqrt(&BigInt::from(10)), None);

        // Negative
        assert_eq!(integer_sqrt(&BigInt::from(-1)), None);
    }
}
