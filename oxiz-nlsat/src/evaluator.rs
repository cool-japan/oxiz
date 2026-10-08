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
        let Some(univariate) = self.substitute_all(&atom.poly, &partial_assignment) else {
            return Lbool::Undef;
        };

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
                // For higher degrees, find exact rational roots via the rational root theorem.
                // We must verify each candidate by evaluating the polynomial.
                find_rational_roots_univariate(poly, var)
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

    /// Compute the feasible region on `var` imposed by an inequality atom under
    /// the (partial) `assignment`, given whether the atom is asserted `satisfied`.
    ///
    /// The atom's polynomial is specialised by substituting every assigned
    /// variable other than `var`; the univariate specialisation's real roots
    /// partition the line into sign-invariant cells, and the feasible region is
    /// the union of the cells whose sign satisfies the atom kind and polarity.
    ///
    /// Soundness: exact rational root isolation only captures rational roots. To
    /// avoid ever excluding a genuinely feasible point (which would be unsound
    /// when the region is used to detect infeasibility), the exact sign-cell
    /// construction is used *only* when the number of exact rational roots
    /// equals the true number of distinct real roots (from the Sturm sequence);
    /// otherwise the sound over-approximation `IntervalSet::reals()` is returned.
    fn atom_constraint(
        &mut self,
        ineq: &IneqAtom,
        var: Var,
        satisfied: bool,
        assignment: &FxHashMap<Var, BigRational>,
    ) -> IntervalSet {
        use crate::cad::SturmSequence;

        // Only single-factor atoms are handled precisely; leave the rest
        // unconstrained (a sound over-approximation).
        if ineq.factors.len() != 1 {
            return IntervalSet::reals();
        }
        let factor = &ineq.factors[0];
        if !factor.poly.vars().contains(&var) {
            return IntervalSet::reals();
        }

        // Substitute every assigned variable other than `var`.
        let mut sub = factor.poly.clone();
        for v in factor.poly.vars() {
            if v != var {
                match assignment.get(&v) {
                    Some(val) => sub = sub.eval_at(v, val),
                    None => return IntervalSet::reals(), // another variable unassigned
                }
            }
        }

        // Constant after substitution: the atom is decided outright.
        if sub.is_constant() {
            let value = sub.eval(&FxHashMap::default());
            let holds = match ineq.kind {
                AtomKind::Eq => value.is_zero(),
                AtomKind::Lt => value.is_negative(),
                AtomKind::Gt => value.is_positive(),
                _ => return IntervalSet::reals(),
            };
            let ok = if satisfied { holds } else { !holds };
            return if ok {
                IntervalSet::reals()
            } else {
                IntervalSet::empty()
            };
        }

        if !sub.is_univariate() || sub.degree(var) == 0 {
            return IntervalSet::reals();
        }

        // Exact sign cells require every real root to be rational.
        let sturm = SturmSequence::new(&sub, var);
        let total_roots = sturm.count_roots() as usize;
        let roots = self.find_roots(&sub, var);
        if roots.len() != total_roots {
            // At least one irrational real root: no exact rational boundary is
            // available, so fall back to the sound over-approximation.
            return IntervalSet::reals();
        }

        let signs = self.signs_between_roots(&sub, var, &roots);
        sign_constraint_set(ineq.kind, satisfied, &roots, &signs)
    }

    /// Sign (`-1`, `0`, `1`) of a univariate polynomial evaluated at `val`.
    fn sign_at(&self, poly: &Polynomial, var: Var, val: &BigRational) -> i8 {
        let mut eval_map = FxHashMap::default();
        eval_map.insert(var, val.clone());
        let value = poly.eval(&eval_map);
        if value.is_zero() {
            0
        } else if value.is_positive() {
            1
        } else {
            -1
        }
    }

    /// Signs of `poly` on the cells `(-∞, r_0), (r_0, r_1), …, (r_{n-1}, +∞)`
    /// delimited by the (sorted, distinct) rational `roots`.
    fn signs_between_roots(&self, poly: &Polynomial, var: Var, roots: &[BigRational]) -> Vec<i8> {
        if roots.is_empty() {
            return vec![self.sign_at(poly, var, &BigRational::zero())];
        }

        let mut signs = Vec::with_capacity(roots.len() + 1);
        let before = &roots[0] - BigRational::one();
        signs.push(self.sign_at(poly, var, &before));

        for window in roots.windows(2) {
            let mid = (&window[0] + &window[1]) / BigRational::from_integer(BigInt::from(2));
            signs.push(self.sign_at(poly, var, &mid));
        }

        let last = &roots[roots.len() - 1];
        let after = last + BigRational::one();
        signs.push(self.sign_at(poly, var, &after));

        signs
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

/// Build the feasible interval set for a univariate atom from the sign of its
/// polynomial on each cell delimited by `roots`.
///
/// `signs[i]` is the sign on the `i`-th cell (there are `roots.len() + 1` cells).
/// The result is the union of cells whose sign satisfies `kind` under polarity
/// `satisfied`.
fn sign_constraint_set(
    kind: AtomKind,
    satisfied: bool,
    roots: &[BigRational],
    signs: &[i8],
) -> IntervalSet {
    match (kind, satisfied) {
        (AtomKind::Eq, true) => IntervalSet::sign_set(roots, signs, 0),
        (AtomKind::Eq, false) => IntervalSet::sign_set(roots, signs, 0).complement(),
        (AtomKind::Lt, true) => IntervalSet::sign_set(roots, signs, -1),
        (AtomKind::Lt, false) => {
            // ¬(p < 0) ⇔ p ≥ 0 ⇔ p > 0 ∨ p = 0.
            IntervalSet::sign_set(roots, signs, 1).union(&IntervalSet::sign_set(roots, signs, 0))
        }
        (AtomKind::Gt, true) => IntervalSet::sign_set(roots, signs, 1),
        (AtomKind::Gt, false) => {
            // ¬(p > 0) ⇔ p ≤ 0 ⇔ p < 0 ∨ p = 0.
            IntervalSet::sign_set(roots, signs, -1).union(&IntervalSet::sign_set(roots, signs, 0))
        }
        _ => IntervalSet::reals(),
    }
}

/// Find all exact rational roots of a univariate polynomial using the rational root theorem.
///
/// For a polynomial with rational coefficients `a_n x^n + ... + a_0`, any rational root
/// p/q satisfies p | (numerator of a_0) and q | (numerator of a_n) after clearing denominators.
fn find_rational_roots_univariate(poly: &Polynomial, var: Var) -> Vec<BigRational> {
    let degree = poly.degree(var) as usize;
    if degree == 0 {
        return Vec::new();
    }

    // Collect rational coefficients coeff[k] = coefficient of var^k
    let rat_coeffs: Vec<BigRational> = (0..=degree)
        .map(|k| poly.univ_coeff(var, k as u32))
        .collect();

    // Scale by LCM of denominators to obtain integer coefficients
    let lcm_denom: BigInt = rat_coeffs.iter().fold(BigInt::from(1i64), |acc, r| {
        let g = gcd_bigint_eval(acc.abs(), r.denom().abs());
        (acc.abs() / g * r.denom()).abs()
    });

    let int_coeffs: Vec<BigInt> = rat_coeffs
        .iter()
        .map(|r| r.numer() * (&lcm_denom / r.denom()))
        .collect();

    rational_roots_from_int_coeffs(poly, var, &int_coeffs)
}

/// GCD helper (Euclidean) for non-negative BigInts.
fn gcd_bigint_eval(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let t = &a % &b;
        a = b;
        b = t;
    }
    if a.is_zero() { BigInt::one() } else { a }
}

/// Test rational root candidates from integer coefficients.
///
/// The `a0 == 0` deflation is a loop rather than a recursive call: the
/// number of deflation steps is the multiplicity of the root at zero, i.e.
/// the polynomial degree, which comes straight from the input. The return
/// type is a plain `Vec` with no channel for a depth error.
fn rational_roots_from_int_coeffs(
    poly: &Polynomial,
    var: Var,
    int_coeffs: &[BigInt],
) -> Vec<BigRational> {
    let mut roots = Vec::new();
    let mut coeffs: &[BigInt] = int_coeffs;

    // Peel off the factors of x. Each step drops the (zero) constant term,
    // so the slice shrinks by one and the loop always terminates.
    while coeffs.len() >= 2 && coeffs[0].is_zero() {
        // x=0 is a root
        roots.push(BigRational::zero());
        coeffs = &coeffs[1..];
    }

    let n = coeffs.len();
    if n < 2 {
        roots.sort();
        roots.dedup();
        return roots;
    }

    let a0 = &coeffs[0];
    let an = &coeffs[n - 1];

    // Divisors of a0 and an. If either set could not be enumerated within
    // the trial-division budget, report only the roots found by deflation
    // rather than testing an incomplete candidate set. Callers already
    // treat this list as "the rational roots we could establish" (irrational
    // roots of a degree>=3 polynomial are never in it either).
    let (Some(divisors_a0), Some(divisors_an)) = (pos_divisors(a0.abs()), pos_divisors(an.abs()))
    else {
        roots.sort();
        roots.dedup();
        return roots;
    };

    let mut eval_map = rustc_hash::FxHashMap::default();
    for p in &divisors_a0 {
        for q in &divisors_an {
            if q.is_zero() {
                continue;
            }
            for &sign in &[1i64, -1i64] {
                let candidate = BigRational::new(p * BigInt::from(sign), q.clone());
                eval_map.clear();
                eval_map.insert(var, candidate.clone());
                if poly.eval(&eval_map).is_zero() {
                    roots.push(candidate);
                }
            }
        }
    }

    roots.sort();
    roots.dedup();
    roots
}

/// Trial-division budget for divisor enumeration.
///
/// Enumerating the divisors of `n` by trial division costs `sqrt(n)` bignum
/// modulos. `n` here is a polynomial coefficient taken straight from
/// `.smt2` input, so it is entirely attacker-chosen: a 40-digit prime
/// coefficient would need ~10²⁰ bignum modulos, i.e. the process never
/// returns. This budget covers every `n` below `TRIAL_DIVISION_BUDGET²`
/// (10¹⁰) exactly, and turns anything larger into an honest "cannot
/// enumerate" rather than a hang.
const TRIAL_DIVISION_BUDGET: u64 = 100_000;

/// Return all positive divisors of a positive `BigInt`.
///
/// `None` means the trial-division budget was exhausted, so the divisor set
/// could not be enumerated *completely*. Callers must not fall back to the
/// partial list: the rational-root theorem only rules candidates in or out
/// when the divisor sets are complete, and a partial list would silently
/// change which candidates get tested.
fn pos_divisors(n: BigInt) -> Option<Vec<BigInt>> {
    if n.is_zero() {
        return Some(vec![BigInt::one()]);
    }
    let mut divs = Vec::new();
    let mut i = BigInt::one();
    let mut steps = 0u64;
    loop {
        if &i * &i > n {
            break;
        }
        if steps >= TRIAL_DIVISION_BUDGET {
            return None;
        }
        steps += 1;
        let r = &n % &i;
        let q = &n / &i;
        if r.is_zero() {
            divs.push(i.clone());
            if q != i {
                divs.push(q);
            }
        }
        i += BigInt::one();
    }
    Some(divs)
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

    /// Small inputs are still enumerated exactly.
    #[test]
    fn test_pos_divisors_small() {
        let divisors = pos_divisors(BigInt::from(12)).expect("12 is within the budget");
        let mut sorted = divisors;
        sorted.sort();
        assert_eq!(
            sorted,
            vec![1, 2, 3, 4, 6, 12]
                .into_iter()
                .map(BigInt::from)
                .collect::<Vec<_>>()
        );
    }

    /// A 40-digit prime coefficient used to make this trial-divide ~10²⁰
    /// times, i.e. never return. It must now report "cannot enumerate"
    /// quickly instead of hanging.
    #[test]
    fn test_pos_divisors_huge_prime_reports_budget_exhaustion() {
        // 2^127 - 1 (a Mersenne prime): sqrt is ~1.3e19 trial divisions.
        let mersenne = (BigInt::from(1u8) << 127) - BigInt::from(1u8);
        assert_eq!(pos_divisors(mersenne), None);
    }

    /// Semantic pin: rational roots are still found for ordinary input,
    /// and the deflation of a root at zero is exhaustive.
    #[test]
    fn test_rational_roots_with_multiplicity_at_zero() {
        // x^3 * (x - 2) = x^4 - 2x^3, roots 0 (multiplicity 3) and 2.
        let int_coeffs: Vec<BigInt> = vec![0, 0, 0, -2, 1].into_iter().map(BigInt::from).collect();
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 4)]), (-2, &[(0, 3)])]);
        let roots = rational_roots_from_int_coeffs(&poly, 0, &int_coeffs);
        assert_eq!(roots, vec![rat(0), rat(2)]);
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

    // ─── atom_constraint / feasible_region (previously an all-reals stub) ────

    fn ineq_atom(poly: Polynomial, kind: AtomKind) -> Atom {
        let max_var = poly.vars().into_iter().max().unwrap_or(0);
        Atom::Ineq(IneqAtom {
            kind,
            factors: vec![crate::types::PolyFactor {
                poly,
                is_even: false,
            }],
            max_var,
            bool_var: 0,
        })
    }

    #[test]
    fn test_atom_constraint_linear_gt_is_open_ray() {
        let mut eval = Evaluator::new();
        // x - 2 > 0  ⇔  x > 2.
        let atom = ineq_atom(
            Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-2, &[])]),
            AtomKind::Gt,
        );
        let assignment = FxHashMap::default();
        let region = eval.feasible_region(0, &[&atom], &[true], &assignment);

        // Previously this returned all reals (unconstrained); it must now be
        // the open ray (2, ∞).
        assert!(!region.is_reals(), "x > 2 must actually constrain x");
        assert!(region.contains(&rat(3)), "3 satisfies x > 2");
        assert!(
            !region.contains(&rat(2)),
            "2 does not satisfy x > 2 (strict)"
        );
        assert!(!region.contains(&rat(1)), "1 does not satisfy x > 2");
    }

    #[test]
    fn test_atom_constraint_intersection_bounds_variable() {
        let mut eval = Evaluator::new();
        // x > 0  and  x - 4 < 0  ⇒  0 < x < 4.
        let gt = ineq_atom(Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]), AtomKind::Gt);
        let lt = ineq_atom(
            Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-4, &[])]),
            AtomKind::Lt,
        );
        let assignment = FxHashMap::default();
        let region = eval.feasible_region(0, &[&gt, &lt], &[true, true], &assignment);

        assert!(region.contains(&rat(2)), "2 ∈ (0, 4)");
        assert!(!region.contains(&rat(0)), "0 ∉ (0, 4)");
        assert!(!region.contains(&rat(4)), "4 ∉ (0, 4)");
        assert!(!region.contains(&rat(5)), "5 ∉ (0, 4)");
    }

    #[test]
    fn test_atom_constraint_negated_polarity() {
        let mut eval = Evaluator::new();
        // ¬(x - 2 > 0)  ⇔  x ≤ 2.
        let atom = ineq_atom(
            Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-2, &[])]),
            AtomKind::Gt,
        );
        let assignment = FxHashMap::default();
        let region = eval.feasible_region(0, &[&atom], &[false], &assignment);

        assert!(region.contains(&rat(2)), "2 satisfies x ≤ 2");
        assert!(region.contains(&rat(1)), "1 satisfies x ≤ 2");
        assert!(!region.contains(&rat(3)), "3 does not satisfy x ≤ 2");
    }

    #[test]
    fn test_atom_constraint_irrational_root_is_sound_overapprox() {
        let mut eval = Evaluator::new();
        // x^2 - 2 > 0 has irrational roots ±√2; the exact rational-root path
        // cannot place the boundaries, so the region must stay a sound
        // over-approximation (all reals) rather than wrongly excluding, say, 2.
        let atom = ineq_atom(
            Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-2, &[])]),
            AtomKind::Gt,
        );
        let assignment = FxHashMap::default();
        let region = eval.feasible_region(0, &[&atom], &[true], &assignment);
        assert!(
            region.contains(&rat(2)),
            "x = 2 satisfies x^2 > 2 and must remain feasible"
        );
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
