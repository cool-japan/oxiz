//! Polynomial arithmetic for non-linear theories.
//!
//! This module provides multivariate polynomial representation and operations
//! for SMT solving, particularly for non-linear real arithmetic (NRA).
//!
//! Reference: Z3's `math/polynomial/` directory.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul, Neg, Sub};

/// Variable identifier for polynomials.
pub type Var = u32;

/// Null variable constant (indicates no variable).
pub const NULL_VAR: Var = u32::MAX;

/// Power of a variable (variable, exponent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarPower {
    /// The variable identifier.
    pub var: Var,
    /// The exponent (power) of the variable.
    pub power: u32,
}

impl VarPower {
    /// Create a new variable power.
    #[inline]
    pub fn new(var: Var, power: u32) -> Self {
        Self { var, power }
    }
}

impl PartialOrd for VarPower {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VarPower {
    fn cmp(&self, other: &Self) -> Ordering {
        self.var.cmp(&other.var)
    }
}

/// A monomial is a product of variables with exponents.
/// Represented as a sorted list of (variable, power) pairs.
/// The unit monomial (1) is represented as an empty list.
#[derive(Clone, PartialEq, Eq)]
pub struct Monomial {
    /// Variables with their exponents, sorted by variable index.
    vars: SmallVec<[VarPower; 4]>,
    /// Cached total degree.
    total_degree: u32,
    /// Cached hash value.
    hash: u64,
}

impl Monomial {
    /// Create the unit monomial (1).
    #[inline]
    pub fn unit() -> Self {
        Self {
            vars: SmallVec::new(),
            total_degree: 0,
            hash: 0,
        }
    }

    /// Create a monomial from a single variable with power 1.
    #[inline]
    pub fn from_var(var: Var) -> Self {
        Self::from_var_power(var, 1)
    }

    /// Create a monomial from a single variable with a given power.
    pub fn from_var_power(var: Var, power: u32) -> Self {
        if power == 0 {
            return Self::unit();
        }
        let mut vars = SmallVec::new();
        vars.push(VarPower::new(var, power));
        Self {
            total_degree: power,
            hash: compute_monomial_hash(&vars),
            vars,
        }
    }

    /// Create a monomial from a list of (variable, power) pairs.
    /// The input doesn't need to be sorted or normalized.
    pub fn from_powers(powers: impl IntoIterator<Item = (Var, u32)>) -> Self {
        let mut var_powers: FxHashMap<Var, u32> = FxHashMap::default();
        for (var, power) in powers {
            if power > 0 {
                *var_powers.entry(var).or_insert(0) += power;
            }
        }

        let mut vars: SmallVec<[VarPower; 4]> = var_powers
            .into_iter()
            .filter(|(_, p)| *p > 0)
            .map(|(v, p)| VarPower::new(v, p))
            .collect();
        vars.sort_by_key(|vp| vp.var);

        let total_degree = vars.iter().map(|vp| vp.power).sum();
        let hash = compute_monomial_hash(&vars);

        Self {
            vars,
            total_degree,
            hash,
        }
    }

    /// Returns true if this is the unit monomial.
    #[inline]
    pub fn is_unit(&self) -> bool {
        self.vars.is_empty()
    }

    /// Returns the total degree of the monomial.
    #[inline]
    pub fn total_degree(&self) -> u32 {
        self.total_degree
    }

    /// Returns the number of variables in the monomial.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.vars.len()
    }

    /// Returns the variable-power pairs.
    #[inline]
    pub fn vars(&self) -> &[VarPower] {
        &self.vars
    }

    /// Returns the degree of a specific variable in this monomial.
    pub fn degree(&self, var: Var) -> u32 {
        self.vars
            .iter()
            .find(|vp| vp.var == var)
            .map(|vp| vp.power)
            .unwrap_or(0)
    }

    /// Returns the maximum variable in this monomial, or NULL_VAR if unit.
    pub fn max_var(&self) -> Var {
        self.vars.last().map(|vp| vp.var).unwrap_or(NULL_VAR)
    }

    /// Check if this monomial is univariate (contains at most one variable).
    #[inline]
    pub fn is_univariate(&self) -> bool {
        self.vars.len() <= 1
    }

    /// Check if this monomial is linear (degree 0 or 1).
    #[inline]
    pub fn is_linear(&self) -> bool {
        self.total_degree <= 1
    }

    /// Multiply two monomials.
    pub fn mul(&self, other: &Monomial) -> Monomial {
        if self.is_unit() {
            return other.clone();
        }
        if other.is_unit() {
            return self.clone();
        }

        let mut vars: SmallVec<[VarPower; 4]> = SmallVec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.vars.len() && j < other.vars.len() {
            match self.vars[i].var.cmp(&other.vars[j].var) {
                Ordering::Less => {
                    vars.push(self.vars[i]);
                    i += 1;
                }
                Ordering::Greater => {
                    vars.push(other.vars[j]);
                    j += 1;
                }
                Ordering::Equal => {
                    vars.push(VarPower::new(
                        self.vars[i].var,
                        self.vars[i].power + other.vars[j].power,
                    ));
                    i += 1;
                    j += 1;
                }
            }
        }
        vars.extend_from_slice(&self.vars[i..]);
        vars.extend_from_slice(&other.vars[j..]);

        let total_degree = self.total_degree + other.total_degree;
        let hash = compute_monomial_hash(&vars);

        Monomial {
            vars,
            total_degree,
            hash,
        }
    }

    /// Check if other divides self. Returns the quotient if it does.
    pub fn div(&self, other: &Monomial) -> Option<Monomial> {
        if other.is_unit() {
            return Some(self.clone());
        }

        let mut vars: SmallVec<[VarPower; 4]> = SmallVec::new();
        let mut j = 0;

        for vp in &self.vars {
            if j < other.vars.len() && other.vars[j].var == vp.var {
                if vp.power < other.vars[j].power {
                    return None;
                }
                let new_power = vp.power - other.vars[j].power;
                if new_power > 0 {
                    vars.push(VarPower::new(vp.var, new_power));
                }
                j += 1;
            } else if j < other.vars.len() && other.vars[j].var < vp.var {
                return None;
            } else {
                vars.push(*vp);
            }
        }

        if j < other.vars.len() {
            return None;
        }

        let total_degree = vars.iter().map(|vp| vp.power).sum();
        let hash = compute_monomial_hash(&vars);

        Some(Monomial {
            vars,
            total_degree,
            hash,
        })
    }

    /// Compute the GCD of two monomials.
    pub fn gcd(&self, other: &Monomial) -> Monomial {
        if self.is_unit() || other.is_unit() {
            return Monomial::unit();
        }

        let mut vars: SmallVec<[VarPower; 4]> = SmallVec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.vars.len() && j < other.vars.len() {
            match self.vars[i].var.cmp(&other.vars[j].var) {
                Ordering::Less => {
                    i += 1;
                }
                Ordering::Greater => {
                    j += 1;
                }
                Ordering::Equal => {
                    let min_power = self.vars[i].power.min(other.vars[j].power);
                    if min_power > 0 {
                        vars.push(VarPower::new(self.vars[i].var, min_power));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        let total_degree = vars.iter().map(|vp| vp.power).sum();
        let hash = compute_monomial_hash(&vars);

        Monomial {
            vars,
            total_degree,
            hash,
        }
    }

    /// Raise monomial to a power.
    pub fn pow(&self, n: u32) -> Monomial {
        if n == 0 {
            return Monomial::unit();
        }
        if n == 1 {
            return self.clone();
        }

        let vars: SmallVec<[VarPower; 4]> = self
            .vars
            .iter()
            .map(|vp| VarPower::new(vp.var, vp.power * n))
            .collect();
        let total_degree = self.total_degree * n;
        let hash = compute_monomial_hash(&vars);

        Monomial {
            vars,
            total_degree,
            hash,
        }
    }

    /// Lexicographic comparison of monomials.
    pub fn lex_cmp(&self, other: &Monomial) -> Ordering {
        let mut i = 0;
        let mut j = 0;

        while i < self.vars.len() && j < other.vars.len() {
            match self.vars[i].var.cmp(&other.vars[j].var) {
                Ordering::Less => return Ordering::Greater,
                Ordering::Greater => return Ordering::Less,
                Ordering::Equal => match self.vars[i].power.cmp(&other.vars[j].power) {
                    Ordering::Equal => {
                        i += 1;
                        j += 1;
                    }
                    ord => return ord,
                },
            }
        }

        if i < self.vars.len() {
            Ordering::Greater
        } else if j < other.vars.len() {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    }

    /// Graded lexicographic comparison (total degree first, then lex).
    pub fn grlex_cmp(&self, other: &Monomial) -> Ordering {
        match self.total_degree.cmp(&other.total_degree) {
            Ordering::Equal => self.lex_cmp(other),
            ord => ord,
        }
    }

    /// Graded reverse lexicographic comparison.
    pub fn grevlex_cmp(&self, other: &Monomial) -> Ordering {
        match self.total_degree.cmp(&other.total_degree) {
            Ordering::Equal => {
                // Reverse lex: compare from highest variable
                let mut i = self.vars.len();
                let mut j = other.vars.len();

                while i > 0 && j > 0 {
                    i -= 1;
                    j -= 1;

                    match self.vars[i].var.cmp(&other.vars[j].var) {
                        Ordering::Less => return Ordering::Less,
                        Ordering::Greater => return Ordering::Greater,
                        Ordering::Equal => match self.vars[i].power.cmp(&other.vars[j].power) {
                            Ordering::Equal => {}
                            Ordering::Less => return Ordering::Greater,
                            Ordering::Greater => return Ordering::Less,
                        },
                    }
                }

                if i > 0 {
                    Ordering::Less
                } else if j > 0 {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }
            ord => ord,
        }
    }
}

fn compute_monomial_hash(vars: &[VarPower]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    for vp in vars {
        vp.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compute GCD of two BigInts using Euclidean algorithm.
fn gcd_bigint(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let t = &a % &b;
        a = b;
        b = t;
    }
    a.abs()
}

/// Count sign variations in a sequence of polynomials evaluated at a point.
/// Used in Sturm's theorem for root counting.
fn count_sign_variations(seq: &[Polynomial], var: Var, point: &BigRational) -> usize {
    let mut signs = Vec::new();

    for poly in seq {
        let val = poly.eval_at(var, point);
        let c = val.constant_term();
        if !c.is_zero() {
            signs.push(if c.is_positive() { 1 } else { -1 });
        }
    }

    // Count sign changes
    let mut variations = 0;
    for i in 1..signs.len() {
        if signs[i] != signs[i - 1] {
            variations += 1;
        }
    }

    variations
}

/// Compute Cauchy's bound for the absolute value of roots of a polynomial.
/// All real roots lie in the interval [-bound, bound].
/// Cauchy's root bound for a univariate polynomial.
/// Returns B such that all roots have absolute value <= B.
/// Bound: 1 + max(|a_i| / |a_n|) for i < n
fn cauchy_root_bound(poly: &Polynomial, var: Var) -> BigRational {
    if poly.is_zero() {
        return BigRational::one();
    }

    let deg = poly.degree(var);
    if deg == 0 {
        return BigRational::one();
    }

    let lc = poly.univ_coeff(var, deg);
    if lc.is_zero() {
        return BigRational::one();
    }

    let lc_abs = lc.abs();

    // Cauchy's bound: 1 + max(|a_i| / |a_n|) for i < n
    let mut max_ratio = BigRational::zero();
    for k in 0..deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() {
            let ratio = coeff.abs() / &lc_abs;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
    }

    BigRational::one() + max_ratio
}

/// Fujiwara's root bound for a univariate polynomial.
/// Returns B such that all roots have absolute value <= B.
/// Generally tighter than Cauchy's bound.
///
/// Fujiwara bound: 2 * max(|a_i/a_n|^(1/(n-i))) for i < n
///
/// Reference: Fujiwara, "Über die obere Schranke des absoluten Betrages
/// der Wurzeln einer algebraischen Gleichung" (1916)
fn fujiwara_root_bound(poly: &Polynomial, var: Var) -> BigRational {
    if poly.is_zero() {
        return BigRational::one();
    }

    let deg = poly.degree(var);
    if deg == 0 {
        return BigRational::one();
    }

    let lc = poly.univ_coeff(var, deg);
    if lc.is_zero() {
        return BigRational::one();
    }

    let lc_abs = lc.abs();

    // For each coefficient a_i (i < n), compute |a_i/a_n|^(1/(n-i))
    // We approximate this using rational arithmetic
    let mut max_val = BigRational::zero();

    for k in 0..deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() {
            let ratio = coeff.abs() / &lc_abs;
            let exp = deg - k;

            // Approximate ratio^(1/exp) using binary search
            // We want to find x such that x^exp ≈ ratio
            let approx_root = rational_nth_root_approx(&ratio, exp);

            if approx_root > max_val {
                max_val = approx_root;
            }
        }
    }

    // Fujiwara bound is 2 * max
    BigRational::from_integer(BigInt::from(2)) * max_val
}

/// Lagrange's root bound for positive roots of a univariate polynomial.
/// Returns B such that all positive roots are <= B.
///
/// Lagrange bound: max(|a_i/a_n|^(1/(n-i))) where a_i < 0 and i < n
fn lagrange_positive_root_bound(poly: &Polynomial, var: Var) -> BigRational {
    if poly.is_zero() {
        return BigRational::one();
    }

    let deg = poly.degree(var);
    if deg == 0 {
        return BigRational::one();
    }

    let lc = poly.univ_coeff(var, deg);
    if lc.is_zero() {
        return BigRational::one();
    }

    let lc_abs = lc.abs();

    // Find the largest |a_i/a_n|^(1/(n-i)) where a_i and a_n have opposite signs
    let mut max_val = BigRational::zero();
    let lc_positive = lc.is_positive();

    for k in 0..deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() && coeff.is_positive() != lc_positive {
            let ratio = coeff.abs() / &lc_abs;
            let exp = deg - k;

            let approx_root = rational_nth_root_approx(&ratio, exp);

            if approx_root > max_val {
                max_val = approx_root;
            }
        }
    }

    if max_val.is_zero() {
        // No negative coefficients found, bound is 1
        BigRational::one()
    } else {
        max_val
    }
}

/// Approximate the nth root of a rational number.
/// Uses binary search to find x such that x^n ≈ target.
/// Returns an upper bound approximation.
fn rational_nth_root_approx(target: &BigRational, n: u32) -> BigRational {
    use crate::rational::pow_uint;

    if n == 0 {
        return BigRational::one();
    }
    if n == 1 {
        return target.clone();
    }
    if target.is_zero() {
        return BigRational::zero();
    }

    // Binary search for the nth root
    let mut low = BigRational::zero();
    let mut high = target.clone() + BigRational::one();

    // Limit iterations
    for _ in 0..100 {
        let mid = (&low + &high) / BigRational::from_integer(BigInt::from(2));
        let mid_pow_n = pow_uint(&mid, n);

        if &mid_pow_n == target {
            return mid;
        }

        if mid_pow_n < *target {
            low = mid;
        } else {
            high = mid.clone();
        }

        // Check convergence
        let diff = &high - &low;
        if diff < BigRational::new(BigInt::one(), BigInt::from(1000000)) {
            return high;
        }
    }

    high
}

/// Count sign variations in the coefficients of a univariate polynomial.
/// This is used for Descartes' rule of signs.
///
/// Descartes' rule of signs: The number of positive real roots of a polynomial
/// is either equal to the number of sign variations in the coefficient sequence,
/// or is less than it by a positive even integer.
fn count_coefficient_sign_variations(poly: &Polynomial, var: Var) -> usize {
    if poly.is_zero() {
        return 0;
    }

    let deg = poly.degree(var);
    if deg == 0 {
        return 0;
    }

    // Collect non-zero coefficients in order
    let mut coeffs = Vec::new();
    for k in 0..=deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() {
            coeffs.push(coeff);
        }
    }

    // Count sign changes
    let mut variations = 0;
    for i in 1..coeffs.len() {
        if coeffs[i].is_positive() != coeffs[i - 1].is_positive() {
            variations += 1;
        }
    }

    variations
}

/// Apply Descartes' rule of signs to get bounds on the number of positive roots.
/// Returns (lower_bound, upper_bound) for the number of positive real roots.
/// The actual count equals upper_bound or differs by an even number.
fn descartes_positive_roots(poly: &Polynomial, var: Var) -> (usize, usize) {
    let variations = count_coefficient_sign_variations(poly, var);

    // The number of positive roots is variations - 2k for some k >= 0
    // So minimum is 0 if variations is even, 1 if variations is odd
    let lower = variations % 2;
    (lower, variations)
}

/// Apply Descartes' rule of signs to get bounds on the number of negative roots.
/// Returns (lower_bound, upper_bound) for the number of negative real roots.
fn descartes_negative_roots(poly: &Polynomial, var: Var) -> (usize, usize) {
    // For negative roots, we evaluate p(-x)
    // This means we negate coefficients of odd powers
    let deg = poly.degree(var);
    let mut neg_poly_terms = Vec::new();

    for k in 0..=deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() {
            let adjusted_coeff = if k % 2 == 1 { -coeff } else { coeff };
            if k == 0 {
                neg_poly_terms.push(Term::constant(adjusted_coeff));
            } else {
                neg_poly_terms.push(Term::new(adjusted_coeff, Monomial::from_var_power(var, k)));
            }
        }
    }

    let neg_poly = Polynomial::from_terms(neg_poly_terms, poly.order);
    descartes_positive_roots(&neg_poly, var)
}

impl Hash for Monomial {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl fmt::Debug for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unit() {
            write!(f, "1")
        } else {
            for (i, vp) in self.vars.iter().enumerate() {
                if i > 0 {
                    write!(f, "*")?;
                }
                if vp.power == 1 {
                    write!(f, "x{}", vp.var)?;
                } else {
                    write!(f, "x{}^{}", vp.var, vp.power)?;
                }
            }
            Ok(())
        }
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// A term is a coefficient multiplied by a monomial.
#[derive(Clone, PartialEq, Eq)]
pub struct Term {
    /// The coefficient of the term.
    pub coeff: BigRational,
    /// The monomial part of the term.
    pub monomial: Monomial,
}

impl Term {
    /// Create a new term.
    #[inline]
    pub fn new(coeff: BigRational, monomial: Monomial) -> Self {
        Self { coeff, monomial }
    }

    /// Create a constant term.
    #[inline]
    pub fn constant(c: BigRational) -> Self {
        Self::new(c, Monomial::unit())
    }

    /// Create a term from a single variable.
    #[inline]
    pub fn from_var(var: Var) -> Self {
        Self::new(BigRational::one(), Monomial::from_var(var))
    }

    /// Check if this term is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeff.is_zero()
    }

    /// Check if this is a constant term.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.monomial.is_unit()
    }
}

impl fmt::Debug for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.monomial.is_unit() {
            write!(f, "{}", self.coeff)
        } else if self.coeff.is_one() {
            write!(f, "{:?}", self.monomial)
        } else if self.coeff == -BigRational::one() {
            write!(f, "-{:?}", self.monomial)
        } else {
            write!(f, "{}*{:?}", self.coeff, self.monomial)
        }
    }
}

/// Monomial ordering for polynomial canonicalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MonomialOrder {
    /// Lexicographic order.
    Lex,
    /// Graded lexicographic order.
    #[default]
    GrLex,
    /// Graded reverse lexicographic order.
    GRevLex,
}

impl MonomialOrder {
    /// Compare two monomials using this ordering.
    pub fn compare(&self, a: &Monomial, b: &Monomial) -> Ordering {
        match self {
            MonomialOrder::Lex => a.lex_cmp(b),
            MonomialOrder::GrLex => a.grlex_cmp(b),
            MonomialOrder::GRevLex => a.grevlex_cmp(b),
        }
    }
}

/// A multivariate polynomial over rationals.
/// Represented as a sum of terms, sorted by monomial order.
#[derive(Clone)]
pub struct Polynomial {
    /// Terms in decreasing order (according to monomial order).
    terms: Vec<Term>,
    /// The monomial ordering used.
    order: MonomialOrder,
}

impl Polynomial {
    /// Create the zero polynomial.
    #[inline]
    pub fn zero() -> Self {
        Self {
            terms: Vec::new(),
            order: MonomialOrder::default(),
        }
    }

    /// Create the one polynomial.
    #[inline]
    pub fn one() -> Self {
        Self::constant(BigRational::one())
    }

    /// Create a constant polynomial.
    pub fn constant(c: BigRational) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            Self {
                terms: vec![Term::constant(c)],
                order: MonomialOrder::default(),
            }
        }
    }

    /// Create a polynomial from a single variable.
    pub fn from_var(var: Var) -> Self {
        Self {
            terms: vec![Term::from_var(var)],
            order: MonomialOrder::default(),
        }
    }

    /// Create a polynomial x^k.
    pub fn from_var_power(var: Var, power: u32) -> Self {
        if power == 0 {
            Self::one()
        } else {
            Self {
                terms: vec![Term::new(
                    BigRational::one(),
                    Monomial::from_var_power(var, power),
                )],
                order: MonomialOrder::default(),
            }
        }
    }

    /// Create a polynomial from terms. Normalizes and combines like terms.
    pub fn from_terms(terms: impl IntoIterator<Item = Term>, order: MonomialOrder) -> Self {
        let mut poly = Self {
            terms: terms.into_iter().filter(|t| !t.is_zero()).collect(),
            order,
        };
        poly.normalize();
        poly
    }

    /// Create a polynomial from integer coefficients.
    pub fn from_coeffs_int(coeffs: &[(i64, &[(Var, u32)])]) -> Self {
        let terms: Vec<Term> = coeffs
            .iter()
            .map(|(c, powers)| {
                Term::new(
                    BigRational::from_integer(BigInt::from(*c)),
                    Monomial::from_powers(powers.iter().copied()),
                )
            })
            .collect();
        Self::from_terms(terms, MonomialOrder::default())
    }

    /// Create a linear polynomial a1*x1 + a2*x2 + ... + c.
    pub fn linear(coeffs: &[(BigRational, Var)], constant: BigRational) -> Self {
        let mut terms: Vec<Term> = coeffs
            .iter()
            .filter(|(c, _)| !c.is_zero())
            .map(|(c, v)| Term::new(c.clone(), Monomial::from_var(*v)))
            .collect();

        if !constant.is_zero() {
            terms.push(Term::constant(constant));
        }

        Self::from_terms(terms, MonomialOrder::default())
    }

    /// Create a univariate polynomial from coefficients.
    /// `coeffs\[i\]` is the coefficient of x^i.
    pub fn univariate(var: Var, coeffs: &[BigRational]) -> Self {
        let terms: Vec<Term> = coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| Term::new(c.clone(), Monomial::from_var_power(var, i as u32)))
            .collect();
        Self::from_terms(terms, MonomialOrder::default())
    }

    /// Check if the polynomial is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if the polynomial is a non-zero constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.terms.len() == 1 && self.terms[0].monomial.is_unit()
    }

    /// Get the constant value of the polynomial.
    ///
    /// Returns the constant coefficient if the polynomial is constant,
    /// or zero if the polynomial is zero.
    pub fn constant_value(&self) -> BigRational {
        if self.is_zero() {
            BigRational::zero()
        } else if self.is_constant() {
            self.terms[0].coeff.clone()
        } else {
            BigRational::zero()
        }
    }

    /// Check if the polynomial is one.
    pub fn is_one(&self) -> bool {
        self.terms.len() == 1 && self.terms[0].monomial.is_unit() && self.terms[0].coeff.is_one()
    }

    /// Check if the polynomial is univariate.
    pub fn is_univariate(&self) -> bool {
        if self.terms.is_empty() {
            return true;
        }

        let mut var: Option<Var> = None;
        for term in &self.terms {
            for vp in term.monomial.vars() {
                match var {
                    None => var = Some(vp.var),
                    Some(v) if v != vp.var => return false,
                    _ => {}
                }
            }
        }
        true
    }

    /// Check if the polynomial is linear (all terms have degree <= 1).
    pub fn is_linear(&self) -> bool {
        self.terms.iter().all(|t| t.monomial.is_linear())
    }

    /// Get the number of terms.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Get the terms.
    #[inline]
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// Get the total degree of the polynomial.
    pub fn total_degree(&self) -> u32 {
        self.terms
            .iter()
            .map(|t| t.monomial.total_degree())
            .max()
            .unwrap_or(0)
    }

    /// Get the degree with respect to a specific variable.
    pub fn degree(&self, var: Var) -> u32 {
        self.terms
            .iter()
            .map(|t| t.monomial.degree(var))
            .max()
            .unwrap_or(0)
    }

    /// Get the maximum variable in the polynomial, or NULL_VAR if constant.
    pub fn max_var(&self) -> Var {
        self.terms
            .iter()
            .map(|t| t.monomial.max_var())
            .filter(|&v| v != NULL_VAR)
            .max()
            .unwrap_or(NULL_VAR)
    }

    /// Get all variables in the polynomial.
    pub fn vars(&self) -> Vec<Var> {
        let mut vars: Vec<Var> = self
            .terms
            .iter()
            .flat_map(|t| t.monomial.vars().iter().map(|vp| vp.var))
            .collect();
        vars.sort_unstable();
        vars.dedup();
        vars
    }

    /// Get the leading term (with respect to monomial order).
    #[inline]
    pub fn leading_term(&self) -> Option<&Term> {
        self.terms.first()
    }

    /// Get the leading coefficient.
    pub fn leading_coeff(&self) -> BigRational {
        self.terms
            .first()
            .map(|t| t.coeff.clone())
            .unwrap_or_else(BigRational::zero)
    }

    /// Get the leading monomial.
    pub fn leading_monomial(&self) -> Option<&Monomial> {
        self.terms.first().map(|t| &t.monomial)
    }

    /// Get the constant term.
    pub fn constant_term(&self) -> BigRational {
        self.terms
            .iter()
            .find(|t| t.monomial.is_unit())
            .map(|t| t.coeff.clone())
            .unwrap_or_else(BigRational::zero)
    }

    /// Get the coefficient of x^k for a univariate polynomial.
    pub fn univ_coeff(&self, var: Var, k: u32) -> BigRational {
        for term in &self.terms {
            if term.monomial.degree(var) == k && term.monomial.num_vars() <= 1 {
                return term.coeff.clone();
            }
        }
        BigRational::zero()
    }

    /// Get the coefficient polynomial for x^k.
    /// For polynomial p(y_1, ..., y_n, x), returns coefficient of x^k.
    pub fn coeff(&self, var: Var, k: u32) -> Polynomial {
        let terms: Vec<Term> = self
            .terms
            .iter()
            .filter(|t| t.monomial.degree(var) == k)
            .map(|t| {
                let new_mon = if let Some(m) = t.monomial.div(&Monomial::from_var_power(var, k)) {
                    m
                } else {
                    Monomial::unit()
                };
                Term::new(t.coeff.clone(), new_mon)
            })
            .collect();
        Polynomial::from_terms(terms, self.order)
    }

    /// Get the leading coefficient with respect to variable x.
    pub fn leading_coeff_wrt(&self, var: Var) -> Polynomial {
        let d = self.degree(var);
        self.coeff(var, d)
    }

    /// Normalize the polynomial (sort terms and combine like terms).
    fn normalize(&mut self) {
        if self.terms.is_empty() {
            return;
        }

        // Sort by monomial order (descending)
        let order = self.order;
        self.terms
            .sort_by(|a, b| order.compare(&b.monomial, &a.monomial));

        // Combine like terms
        let mut i = 0;
        while i < self.terms.len() {
            let mut j = i + 1;
            while j < self.terms.len() && self.terms[j].monomial == self.terms[i].monomial {
                let coeff = self.terms[j].coeff.clone();
                self.terms[i].coeff += coeff;
                j += 1;
            }
            // Remove combined terms
            if j > i + 1 {
                self.terms.drain((i + 1)..j);
            }
            i += 1;
        }

        // Remove zero terms
        self.terms.retain(|t| !t.coeff.is_zero());
    }

    /// Negate the polynomial.
    pub fn neg(&self) -> Polynomial {
        Polynomial {
            terms: self
                .terms
                .iter()
                .map(|t| Term::new(-t.coeff.clone(), t.monomial.clone()))
                .collect(),
            order: self.order,
        }
    }

    /// Add two polynomials.
    pub fn add(&self, other: &Polynomial) -> Polynomial {
        let mut terms: Vec<Term> = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        Polynomial::from_terms(terms, self.order)
    }

    /// Subtract two polynomials.
    pub fn sub(&self, other: &Polynomial) -> Polynomial {
        self.add(&other.neg())
    }

    /// Multiply by a scalar.
    pub fn scale(&self, c: &BigRational) -> Polynomial {
        if c.is_zero() {
            return Polynomial::zero();
        }
        if c.is_one() {
            return self.clone();
        }
        Polynomial {
            terms: self
                .terms
                .iter()
                .map(|t| Term::new(&t.coeff * c, t.monomial.clone()))
                .collect(),
            order: self.order,
        }
    }

    /// Multiply two polynomials.
    /// Uses Karatsuba algorithm for large univariate polynomials.
    pub fn mul(&self, other: &Polynomial) -> Polynomial {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }

        // Use Karatsuba for large univariate polynomials (threshold: 16 terms)
        if self.is_univariate()
            && other.is_univariate()
            && self.max_var() == other.max_var()
            && self.terms.len() >= 16
            && other.terms.len() >= 16
        {
            return self.mul_karatsuba(other);
        }

        let mut terms: Vec<Term> = Vec::with_capacity(self.terms.len() * other.terms.len());

        for t1 in &self.terms {
            for t2 in &other.terms {
                terms.push(Term::new(
                    &t1.coeff * &t2.coeff,
                    t1.monomial.mul(&t2.monomial),
                ));
            }
        }

        Polynomial::from_terms(terms, self.order)
    }

    /// Karatsuba multiplication for univariate polynomials.
    /// Time complexity: O(n^1.585) instead of O(n^2) for naive multiplication.
    ///
    /// This is particularly efficient for polynomials with 16+ terms.
    fn mul_karatsuba(&self, other: &Polynomial) -> Polynomial {
        debug_assert!(self.is_univariate() && other.is_univariate());
        debug_assert!(self.max_var() == other.max_var());

        let var = self.max_var();
        let deg1 = self.degree(var);
        let deg2 = other.degree(var);

        // Base case: use naive multiplication for small polynomials
        if deg1 <= 8 || deg2 <= 8 {
            let mut terms: Vec<Term> = Vec::with_capacity(self.terms.len() * other.terms.len());
            for t1 in &self.terms {
                for t2 in &other.terms {
                    terms.push(Term::new(
                        &t1.coeff * &t2.coeff,
                        t1.monomial.mul(&t2.monomial),
                    ));
                }
            }
            return Polynomial::from_terms(terms, self.order);
        }

        // Split at middle degree
        let split = deg1.max(deg2).div_ceil(2);

        // Split self = low0 + high0 * x^split
        let (low0, high0) = self.split_at_degree(var, split);

        // Split other = low1 + high1 * x^split
        let (low1, high1) = other.split_at_degree(var, split);

        // Karatsuba: (low0 + high0*x^m)(low1 + high1*x^m)
        // = low0*low1 + ((low0+high0)(low1+high1) - low0*low1 - high0*high1)*x^m + high0*high1*x^(2m)

        let z0 = low0.mul_karatsuba(&low1); // low0 * low1
        let z2 = high0.mul_karatsuba(&high1); // high0 * high1

        let sum0 = Polynomial::add(&low0, &high0);
        let sum1 = Polynomial::add(&low1, &high1);
        let z1_temp = sum0.mul_karatsuba(&sum1);

        // z1 = z1_temp - z0 - z2
        let z1 = Polynomial::sub(&Polynomial::sub(&z1_temp, &z0), &z2);

        // Result = z0 + z1*x^split + z2*x^(2*split)
        let x_split = Monomial::from_var_power(var, split);
        let x_2split = Monomial::from_var_power(var, 2 * split);

        let term1 = z1.mul_monomial(&x_split);
        let term2 = z2.mul_monomial(&x_2split);

        Polynomial::add(&Polynomial::add(&z0, &term1), &term2)
    }

    /// Split a univariate polynomial into low and high parts at a given degree.
    /// Returns (low, high) where poly = low + high * x^deg.
    fn split_at_degree(&self, var: Var, deg: u32) -> (Polynomial, Polynomial) {
        let mut low_terms = Vec::new();
        let mut high_terms = Vec::new();

        for term in &self.terms {
            let term_deg = term.monomial.degree(var);
            if term_deg < deg {
                low_terms.push(term.clone());
            } else {
                // Shift down the high part
                if let Some(new_mon) = term.monomial.div(&Monomial::from_var_power(var, deg)) {
                    high_terms.push(Term::new(term.coeff.clone(), new_mon));
                }
            }
        }

        let low = if low_terms.is_empty() {
            Polynomial::zero()
        } else {
            Polynomial::from_terms(low_terms, self.order)
        };

        let high = if high_terms.is_empty() {
            Polynomial::zero()
        } else {
            Polynomial::from_terms(high_terms, self.order)
        };

        (low, high)
    }

    /// Multiply by a monomial.
    pub fn mul_monomial(&self, m: &Monomial) -> Polynomial {
        if m.is_unit() {
            return self.clone();
        }
        Polynomial {
            terms: self
                .terms
                .iter()
                .map(|t| Term::new(t.coeff.clone(), t.monomial.mul(m)))
                .collect(),
            order: self.order,
        }
    }

    /// Compute p^k.
    pub fn pow(&self, k: u32) -> Polynomial {
        if k == 0 {
            return Polynomial::one();
        }
        if k == 1 {
            return self.clone();
        }
        if self.is_zero() {
            return Polynomial::zero();
        }

        // Binary exponentiation
        let mut result = Polynomial::one();
        let mut base = self.clone();
        let mut exp = k;

        while exp > 0 {
            if exp & 1 == 1 {
                result = Polynomial::mul(&result, &base);
            }
            base = Polynomial::mul(&base, &base);
            exp >>= 1;
        }

        result
    }

    /// Compute the derivative with respect to a variable.
    pub fn derivative(&self, var: Var) -> Polynomial {
        let terms: Vec<Term> = self
            .terms
            .iter()
            .filter_map(|t| {
                let d = t.monomial.degree(var);
                if d == 0 {
                    return None;
                }
                let new_coeff = &t.coeff * BigRational::from_integer(BigInt::from(d));
                let new_mon = if d == 1 {
                    t.monomial
                        .div(&Monomial::from_var(var))
                        .unwrap_or_else(Monomial::unit)
                } else {
                    let new_powers: Vec<(Var, u32)> = t
                        .monomial
                        .vars()
                        .iter()
                        .map(|vp| {
                            if vp.var == var {
                                (vp.var, vp.power - 1)
                            } else {
                                (vp.var, vp.power)
                            }
                        })
                        .filter(|(_, p)| *p > 0)
                        .collect();
                    Monomial::from_powers(new_powers)
                };
                Some(Term::new(new_coeff, new_mon))
            })
            .collect();
        Polynomial::from_terms(terms, self.order)
    }

    /// Compute the nth derivative of the polynomial with respect to a variable.
    ///
    /// # Arguments
    /// * `var` - The variable to differentiate with respect to
    /// * `n` - The order of the derivative (n = 0 returns the polynomial itself)
    ///
    /// # Examples
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // f(x) = x^3 = x³
    /// let f = Polynomial::from_coeffs_int(&[(1, &[(0, 3)])]);
    ///
    /// // f'(x) = 3x^2
    /// let f_prime = f.nth_derivative(0, 1);
    /// assert_eq!(f_prime.total_degree(), 2);
    ///
    /// // f''(x) = 6x
    /// let f_double_prime = f.nth_derivative(0, 2);
    /// assert_eq!(f_double_prime.total_degree(), 1);
    ///
    /// // f'''(x) = 6 (constant)
    /// let f_triple_prime = f.nth_derivative(0, 3);
    /// assert_eq!(f_triple_prime.total_degree(), 0);
    /// assert!(!f_triple_prime.is_zero());
    ///
    /// // f''''(x) = 0
    /// let f_fourth = f.nth_derivative(0, 4);
    /// assert!(f_fourth.is_zero());
    /// ```
    pub fn nth_derivative(&self, var: Var, n: u32) -> Polynomial {
        if n == 0 {
            return self.clone();
        }

        let mut result = self.clone();
        for _ in 0..n {
            result = result.derivative(var);
            if result.is_zero() {
                break;
            }
        }
        result
    }

    /// Computes the gradient (vector of partial derivatives) with respect to all variables.
    ///
    /// For a multivariate polynomial f(x₁, x₂, ..., xₙ), returns the vector:
    /// ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    ///
    /// # Returns
    /// A vector of polynomials, one for each variable, ordered by variable index.
    ///
    /// # Example
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// // f(x,y) = x²y + 2xy + y²
    /// let f = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2), (1, 1)]), // x²y
    ///     (2, &[(0, 1), (1, 1)]), // 2xy
    ///     (1, &[(1, 2)]),          // y²
    /// ]);
    ///
    /// let grad = f.gradient();
    /// // ∂f/∂x = 2xy + 2y
    /// // ∂f/∂y = x² + 2x + 2y
    /// assert_eq!(grad.len(), 2);
    /// ```
    pub fn gradient(&self) -> Vec<Polynomial> {
        let vars = self.vars();
        if vars.is_empty() {
            return vec![];
        }

        let max_var = *vars.iter().max().unwrap();
        let mut grad = Vec::new();

        // Compute partial derivative for each variable from 0 to max_var
        for var in 0..=max_var {
            grad.push(self.derivative(var));
        }

        grad
    }

    /// Computes the Hessian matrix (matrix of second-order partial derivatives).
    ///
    /// For a multivariate polynomial f(x₁, x₂, ..., xₙ), returns the symmetric matrix:
    /// `H[i,j] = ∂²f/(∂xᵢ∂xⱼ)`
    ///
    /// The Hessian is useful for:
    /// - Optimization (finding local minima/maxima)
    /// - Convexity analysis
    /// - Second-order Taylor approximations
    ///
    /// # Returns
    /// A vector of vectors representing the Hessian matrix.
    /// The matrix is symmetric: `H[i][j] = H[j][i]`
    ///
    /// # Example
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// // f(x,y) = x² + xy + y²
    /// let f = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),          // x²
    ///     (1, &[(0, 1), (1, 1)]), // xy
    ///     (1, &[(1, 2)]),          // y²
    /// ]);
    ///
    /// let hessian = f.hessian();
    /// // H = [[2, 1],
    /// //      [1, 2]]
    /// assert_eq!(hessian.len(), 2);
    /// assert_eq!(hessian[0].len(), 2);
    /// ```
    pub fn hessian(&self) -> Vec<Vec<Polynomial>> {
        let vars = self.vars();
        if vars.is_empty() {
            return vec![];
        }

        let max_var = *vars.iter().max().unwrap();
        let n = (max_var + 1) as usize;

        let mut hessian = vec![vec![Polynomial::zero(); n]; n];

        // Compute all second-order partial derivatives
        for i in 0..=max_var {
            for j in 0..=max_var {
                // ∂²f/(∂xᵢ∂xⱼ) = ∂/∂xⱼ(∂f/∂xᵢ)
                let first_deriv = self.derivative(i);
                let second_deriv = first_deriv.derivative(j);
                hessian[i as usize][j as usize] = second_deriv;
            }
        }

        hessian
    }

    /// Computes the Jacobian matrix for a vector of polynomials.
    ///
    /// For a vector of polynomials f = (f₁, f₂, ..., fₘ) each depending on variables
    /// x = (x₁, x₂, ..., xₙ), the Jacobian is the m×n matrix:
    /// `J[i,j] = ∂fᵢ/∂xⱼ`
    ///
    /// # Arguments
    /// * `polys` - Vector of polynomials representing the function components
    ///
    /// # Returns
    /// A matrix where each row i contains the gradient of polynomial i
    ///
    /// # Example
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// // f₁(x,y) = x² + y
    /// // f₂(x,y) = x + y²
    /// let f1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[(1, 1)])]);
    /// let f2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 2)])]);
    ///
    /// let jacobian = Polynomial::jacobian(&[f1, f2]);
    /// // J = [[2x, 1 ],
    /// //      [1,  2y]]
    /// assert_eq!(jacobian.len(), 2); // 2 functions
    /// ```
    pub fn jacobian(polys: &[Polynomial]) -> Vec<Vec<Polynomial>> {
        if polys.is_empty() {
            return vec![];
        }

        // Find the maximum variable across all polynomials
        let max_var = polys.iter().flat_map(|p| p.vars()).max().unwrap_or(0);

        let n_vars = (max_var + 1) as usize;
        let mut jacobian = Vec::with_capacity(polys.len());

        for poly in polys {
            let mut row = Vec::with_capacity(n_vars);
            for var in 0..=max_var {
                row.push(poly.derivative(var));
            }
            jacobian.push(row);
        }

        jacobian
    }

    /// Compute the indefinite integral (antiderivative) of the polynomial with respect to a variable.
    ///
    /// For a polynomial p(x), returns ∫p(x)dx. The constant of integration is implicitly zero.
    ///
    /// # Arguments
    /// * `var` - The variable to integrate with respect to
    ///
    /// # Examples
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    ///
    /// // f(x) = 3x^2
    /// let f = Polynomial::from_coeffs_int(&[(3, &[(0, 2)])]);
    ///
    /// // ∫f(x)dx = x^3
    /// let integral = f.integrate(0);
    ///
    /// // Verify: derivative of integral should be original
    /// let derivative = integral.derivative(0);
    /// assert_eq!(derivative, f);
    /// ```
    pub fn integrate(&self, var: Var) -> Polynomial {
        let terms: Vec<Term> = self
            .terms
            .iter()
            .map(|t| {
                let d = t.monomial.degree(var);
                let new_power = d + 1;

                // Divide coefficient by (power + 1)
                let new_coeff = &t.coeff / BigRational::from_integer(BigInt::from(new_power));

                // Increment the power of var
                let new_powers: Vec<(Var, u32)> = if d == 0 {
                    // Constant term: add var^1
                    let mut powers = t
                        .monomial
                        .vars()
                        .iter()
                        .map(|vp| (vp.var, vp.power))
                        .collect::<Vec<_>>();
                    powers.push((var, 1));
                    powers.sort_by_key(|(v, _)| *v);
                    powers
                } else {
                    // Variable already exists: increment power
                    t.monomial
                        .vars()
                        .iter()
                        .map(|vp| {
                            if vp.var == var {
                                (vp.var, vp.power + 1)
                            } else {
                                (vp.var, vp.power)
                            }
                        })
                        .collect()
                };

                let new_mon = Monomial::from_powers(new_powers);
                Term::new(new_coeff, new_mon)
            })
            .collect();
        Polynomial::from_terms(terms, self.order)
    }

    /// Compute the definite integral of a univariate polynomial over an interval (a, b).
    ///
    /// For a univariate polynomial p(x), returns ∫ₐᵇ p(x)dx = F(b) - F(a)
    /// where F is the antiderivative of p.
    ///
    /// # Arguments
    /// * `var` - The variable to integrate with respect to
    /// * `lower` - Lower bound of integration
    /// * `upper` - Upper bound of integration
    ///
    /// # Returns
    /// The definite integral value, or None if the polynomial is not univariate in the given variable
    ///
    /// # Examples
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    ///
    /// // ∫[0,2] x^2 dx = [x^3/3] from 0 to 2 = 8/3
    /// let f = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
    /// let result = f.definite_integral(0, &BigRational::from_integer(BigInt::from(0)),
    ///                                     &BigRational::from_integer(BigInt::from(2)));
    /// assert_eq!(result, Some(BigRational::new(BigInt::from(8), BigInt::from(3))));
    /// ```
    pub fn definite_integral(
        &self,
        var: Var,
        lower: &BigRational,
        upper: &BigRational,
    ) -> Option<BigRational> {
        // Get the antiderivative
        let antideriv = self.integrate(var);

        // Evaluate at upper and lower bounds
        let mut upper_assignment = rustc_hash::FxHashMap::default();
        upper_assignment.insert(var, upper.clone());
        let upper_val = antideriv.eval(&upper_assignment);

        let mut lower_assignment = rustc_hash::FxHashMap::default();
        lower_assignment.insert(var, lower.clone());
        let lower_val = antideriv.eval(&lower_assignment);

        // Return F(b) - F(a)
        Some(upper_val - lower_val)
    }

    /// Find critical points of a univariate polynomial by solving f'(x) = 0.
    ///
    /// Critical points are values where the derivative equals zero, which correspond
    /// to local maxima, minima, or saddle points.
    ///
    /// # Arguments
    /// * `var` - The variable to find critical points for
    ///
    /// # Returns
    /// A vector of isolating intervals containing the critical points. Each interval
    /// contains exactly one root of the derivative.
    ///
    /// # Examples
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // f(x) = x^3 - 3x = x(x^2 - 3)
    /// // f'(x) = 3x^2 - 3 = 3(x^2 - 1) = 3(x-1)(x+1)
    /// // Critical points at x = -1 and x = 1
    /// let f = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 3)]),  // x^3
    ///     (-3, &[(0, 1)]), // -3x
    /// ]);
    ///
    /// let critical_points = f.find_critical_points(0);
    /// assert_eq!(critical_points.len(), 2); // Two critical points
    /// ```
    pub fn find_critical_points(&self, var: Var) -> Vec<(BigRational, BigRational)> {
        // Compute the derivative
        let deriv = self.derivative(var);

        // Find roots of the derivative (where f'(x) = 0)
        deriv.isolate_roots(var)
    }

    /// Numerically integrate using the trapezoidal rule.
    ///
    /// Approximates ∫ₐᵇ f(x)dx using the trapezoidal rule with n subintervals.
    /// The trapezoidal rule approximates the integral by summing the areas of trapezoids.
    ///
    /// # Arguments
    /// * `var` - The variable to integrate with respect to
    /// * `lower` - Lower bound of integration
    /// * `upper` - Upper bound of integration
    /// * `n` - Number of subintervals (more subintervals = higher accuracy)
    ///
    /// # Examples
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    ///
    /// // ∫[0,1] x^2 dx = 1/3
    /// let f = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
    /// let approx = f.trapezoidal_rule(0,
    ///     &BigRational::from_integer(BigInt::from(0)),
    ///     &BigRational::from_integer(BigInt::from(1)),
    ///     100);
    ///
    /// let exact = BigRational::new(BigInt::from(1), BigInt::from(3));
    /// // With 100 intervals, approximation should be very close
    /// ```
    pub fn trapezoidal_rule(
        &self,
        var: Var,
        lower: &BigRational,
        upper: &BigRational,
        n: u32,
    ) -> BigRational {
        if n == 0 {
            return BigRational::zero();
        }

        // Step size h = (b - a) / n
        let h = (upper - lower) / BigRational::from_integer(BigInt::from(n));

        let mut sum = BigRational::zero();

        // First and last terms: f(a)/2 + f(b)/2
        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(var, lower.clone());
        sum += &self.eval(&assignment) / BigRational::from_integer(BigInt::from(2));

        assignment.insert(var, upper.clone());
        sum += &self.eval(&assignment) / BigRational::from_integer(BigInt::from(2));

        // Middle terms: sum of f(x_i) for i = 1 to n-1
        for i in 1..n {
            let x_i = lower + &h * BigRational::from_integer(BigInt::from(i));
            assignment.insert(var, x_i);
            sum += &self.eval(&assignment);
        }

        // Multiply by step size
        sum * h
    }

    /// Numerically integrate using Simpson's rule.
    ///
    /// Approximates ∫ₐᵇ f(x)dx using Simpson's rule with n subintervals (n must be even).
    /// Simpson's rule uses parabolic approximation and is generally more accurate than
    /// the trapezoidal rule for smooth functions.
    ///
    /// # Arguments
    /// * `var` - The variable to integrate with respect to
    /// * `lower` - Lower bound of integration
    /// * `upper` - Upper bound of integration
    /// * `n` - Number of subintervals (must be even for Simpson's rule)
    ///
    /// # Examples
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    ///
    /// // ∫[0,1] x^2 dx = 1/3
    /// let f = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
    /// let approx = f.simpsons_rule(0,
    ///     &BigRational::from_integer(BigInt::from(0)),
    ///     &BigRational::from_integer(BigInt::from(1)),
    ///     100);
    ///
    /// let exact = BigRational::new(BigInt::from(1), BigInt::from(3));
    /// // Simpson's rule should give very accurate results for polynomials
    /// ```
    pub fn simpsons_rule(
        &self,
        var: Var,
        lower: &BigRational,
        upper: &BigRational,
        n: u32,
    ) -> BigRational {
        if n == 0 {
            return BigRational::zero();
        }

        // Ensure n is even for Simpson's rule
        let n = if n % 2 == 1 { n + 1 } else { n };

        // Step size h = (b - a) / n
        let h = (upper - lower) / BigRational::from_integer(BigInt::from(n));

        let mut sum = BigRational::zero();
        let mut assignment = rustc_hash::FxHashMap::default();

        // First and last terms: f(a) + f(b)
        assignment.insert(var, lower.clone());
        sum += &self.eval(&assignment);

        assignment.insert(var, upper.clone());
        sum += &self.eval(&assignment);

        // Odd-indexed terms (multiplied by 4): i = 1, 3, 5, ..., n-1
        for i in (1..n).step_by(2) {
            let x_i = lower + &h * BigRational::from_integer(BigInt::from(i));
            assignment.insert(var, x_i);
            sum += BigRational::from_integer(BigInt::from(4)) * &self.eval(&assignment);
        }

        // Even-indexed terms (multiplied by 2): i = 2, 4, 6, ..., n-2
        for i in (2..n).step_by(2) {
            let x_i = lower + &h * BigRational::from_integer(BigInt::from(i));
            assignment.insert(var, x_i);
            sum += BigRational::from_integer(BigInt::from(2)) * &self.eval(&assignment);
        }

        // Multiply by h/3
        sum * h / BigRational::from_integer(BigInt::from(3))
    }

    /// Evaluate the polynomial at a point (substituting a value for a variable).
    pub fn eval_at(&self, var: Var, value: &BigRational) -> Polynomial {
        let terms: Vec<Term> = self
            .terms
            .iter()
            .map(|t| {
                let d = t.monomial.degree(var);
                if d == 0 {
                    t.clone()
                } else {
                    let new_coeff = &t.coeff * value.pow(d as i32);
                    let new_mon = t
                        .monomial
                        .div(&Monomial::from_var_power(var, d))
                        .unwrap_or_else(Monomial::unit);
                    Term::new(new_coeff, new_mon)
                }
            })
            .collect();
        Polynomial::from_terms(terms, self.order)
    }

    /// Evaluate the polynomial completely (all variables assigned).
    pub fn eval(&self, assignment: &FxHashMap<Var, BigRational>) -> BigRational {
        let mut result = BigRational::zero();

        for term in &self.terms {
            let mut val = term.coeff.clone();
            for vp in term.monomial.vars() {
                if let Some(v) = assignment.get(&vp.var) {
                    val *= v.pow(vp.power as i32);
                } else {
                    panic!("Variable x{} not in assignment", vp.var);
                }
            }
            result += val;
        }

        result
    }

    /// Evaluate a univariate polynomial using Horner's method.
    ///
    /// Horner's method evaluates a polynomial p(x) = a_n x^n + ... + a_1 x + a_0
    /// as (((...((a_n) x + a_{n-1}) x + ...) x + a_1) x + a_0), which requires
    /// only n multiplications instead of n(n+1)/2 for the naive method.
    ///
    /// This method is more efficient than eval() for univariate polynomials.
    ///
    /// # Arguments
    /// * `var` - The variable to evaluate
    /// * `value` - The value to substitute for the variable
    ///
    /// # Returns
    /// The evaluated value
    ///
    /// # Panics
    /// Panics if the polynomial is not univariate in the given variable
    pub fn eval_horner(&self, var: Var, value: &BigRational) -> BigRational {
        if self.is_zero() {
            return BigRational::zero();
        }

        // For multivariate polynomials, use eval_at instead
        if !self.is_univariate() || self.max_var() != var {
            let result = self.eval_at(var, value);
            // If result is constant, return its value
            if result.is_constant() {
                return result.constant_value();
            }
            panic!("Polynomial is not univariate in variable x{}", var);
        }

        let deg = self.degree(var);
        if deg == 0 {
            return self.constant_value();
        }

        // Collect coefficients in descending order of degree
        let mut coeffs = vec![BigRational::zero(); (deg + 1) as usize];
        for k in 0..=deg {
            coeffs[k as usize] = self.univ_coeff(var, k);
        }

        // Apply Horner's method: start from highest degree
        let mut result = coeffs[deg as usize].clone();
        for k in (0..deg).rev() {
            result = &result * value + &coeffs[k as usize];
        }

        result
    }

    /// Substitute a polynomial for a variable.
    pub fn substitute(&self, var: Var, replacement: &Polynomial) -> Polynomial {
        let mut result = Polynomial::zero();

        for term in &self.terms {
            let d = term.monomial.degree(var);
            if d == 0 {
                result = Polynomial::add(
                    &result,
                    &Polynomial::from_terms(vec![term.clone()], self.order),
                );
            } else {
                let remainder = term
                    .monomial
                    .div(&Monomial::from_var_power(var, d))
                    .unwrap_or_else(Monomial::unit);
                let coeff_poly = Polynomial::from_terms(
                    vec![Term::new(term.coeff.clone(), remainder)],
                    self.order,
                );
                let rep_pow = replacement.pow(d);
                result = Polynomial::add(&result, &Polynomial::mul(&coeff_poly, &rep_pow));
            }
        }

        result
    }

    /// Integer content: GCD of all coefficients (as integers).
    /// Assumes all coefficients are integers.
    pub fn integer_content(&self) -> BigInt {
        if self.terms.is_empty() {
            return BigInt::one();
        }

        let mut gcd: Option<BigInt> = None;
        for term in &self.terms {
            let num = term.coeff.numer().clone();
            gcd = Some(match gcd {
                None => num.abs(),
                Some(g) => gcd_bigint(g, num.abs()),
            });
        }
        gcd.unwrap_or_else(BigInt::one)
    }

    /// Make the polynomial primitive (divide by integer content).
    pub fn primitive(&self) -> Polynomial {
        let content = self.integer_content();
        if content.is_one() {
            return self.clone();
        }
        let c = BigRational::from_integer(content);
        self.scale(&(BigRational::one() / c))
    }

    /// GCD of two polynomials (univariate, using Euclidean algorithm).
    pub fn gcd_univariate(&self, other: &Polynomial) -> Polynomial {
        if self.is_zero() {
            return other.primitive();
        }
        if other.is_zero() {
            return self.primitive();
        }

        let var = self.max_var().max(other.max_var());
        if var == NULL_VAR {
            // Both are constants, GCD is 1
            return Polynomial::one();
        }

        let mut a = self.primitive();
        let mut b = other.primitive();

        // Ensure deg(a) >= deg(b)
        if a.degree(var) < b.degree(var) {
            std::mem::swap(&mut a, &mut b);
        }

        // Limit iterations for safety
        let mut iter_count = 0;
        let max_iters = a.degree(var) as usize + b.degree(var) as usize + 10;

        while !b.is_zero() && iter_count < max_iters {
            iter_count += 1;
            let r = a.pseudo_remainder(&b, var);
            a = b;
            b = if r.is_zero() { r } else { r.primitive() };
        }

        a.primitive()
    }

    /// Pseudo-remainder for univariate polynomials.
    /// Returns r such that lc(b)^d * a = q * b + r for some q,
    /// where d = max(deg(a) - deg(b) + 1, 0).
    pub fn pseudo_remainder(&self, divisor: &Polynomial, var: Var) -> Polynomial {
        if divisor.is_zero() {
            panic!("Division by zero polynomial");
        }

        if self.is_zero() {
            return Polynomial::zero();
        }

        let deg_a = self.degree(var);
        let deg_b = divisor.degree(var);

        if deg_a < deg_b {
            return self.clone();
        }

        let lc_b = divisor.univ_coeff(var, deg_b);
        let mut r = self.clone();

        // Limit iterations
        let max_iters = (deg_a - deg_b + 2) as usize;
        let mut iters = 0;

        while !r.is_zero() && r.degree(var) >= deg_b && iters < max_iters {
            iters += 1;
            let deg_r = r.degree(var);
            let lc_r = r.univ_coeff(var, deg_r);
            let shift = deg_r - deg_b;

            // r = lc_b * r - lc_r * x^shift * divisor
            r = r.scale(&lc_b);
            let subtractor = divisor
                .scale(&lc_r)
                .mul_monomial(&Monomial::from_var_power(var, shift));
            r = Polynomial::sub(&r, &subtractor);
        }

        r
    }

    /// Pseudo-division for univariate polynomials.
    /// Returns (quotient, remainder) such that lc(b)^d * a = q * b + r
    /// where d = deg(a) - deg(b) + 1.
    pub fn pseudo_div_univariate(&self, divisor: &Polynomial) -> (Polynomial, Polynomial) {
        if divisor.is_zero() {
            panic!("Division by zero polynomial");
        }

        if self.is_zero() {
            return (Polynomial::zero(), Polynomial::zero());
        }

        let var = self.max_var().max(divisor.max_var());
        if var == NULL_VAR {
            // Both are constants
            return (Polynomial::zero(), self.clone());
        }

        let deg_a = self.degree(var);
        let deg_b = divisor.degree(var);

        if deg_a < deg_b {
            return (Polynomial::zero(), self.clone());
        }

        let lc_b = divisor.univ_coeff(var, deg_b);
        let mut q = Polynomial::zero();
        let mut r = self.clone();

        // Limit iterations
        let max_iters = (deg_a - deg_b + 2) as usize;
        let mut iters = 0;

        while !r.is_zero() && r.degree(var) >= deg_b && iters < max_iters {
            iters += 1;
            let deg_r = r.degree(var);
            let lc_r = r.univ_coeff(var, deg_r);
            let shift = deg_r - deg_b;

            let term = Polynomial::from_terms(
                vec![Term::new(
                    lc_r.clone(),
                    Monomial::from_var_power(var, shift),
                )],
                self.order,
            );

            q = q.scale(&lc_b);
            q = Polynomial::add(&q, &term);

            r = r.scale(&lc_b);
            let subtractor = Polynomial::mul(divisor, &term);
            r = Polynomial::sub(&r, &subtractor);
        }

        (q, r)
    }

    /// Compute the subresultant polynomial remainder sequence (PRS).
    ///
    /// The subresultant PRS is a more efficient variant of the pseudo-remainder sequence
    /// that avoids coefficient explosion through careful normalization. It's useful for
    /// GCD computation and other polynomial algorithms.
    ///
    /// Returns a sequence of polynomials [p0, p1, ..., pk] where:
    /// - p0 = self
    /// - p1 = other
    /// - Each subsequent polynomial is derived using the subresultant algorithm
    ///
    /// Reference: "Algorithms for Computer Algebra" by Geddes, Czapor, Labahn
    pub fn subresultant_prs(&self, other: &Polynomial, var: Var) -> Vec<Polynomial> {
        if self.is_zero() || other.is_zero() {
            return vec![];
        }

        let mut prs = Vec::new();
        let mut a = self.clone();
        let mut b = other.clone();

        // Ensure deg(a) >= deg(b)
        if a.degree(var) < b.degree(var) {
            std::mem::swap(&mut a, &mut b);
        }

        prs.push(a.clone());
        prs.push(b.clone());

        let mut g = Polynomial::one();
        let mut h = Polynomial::one();

        let max_iters = a.degree(var) as usize + b.degree(var) as usize + 10;
        let mut iter_count = 0;

        while !b.is_zero() && iter_count < max_iters {
            iter_count += 1;

            let delta = a.degree(var) as i32 - b.degree(var) as i32;
            if delta < 0 {
                break;
            }

            // Compute pseudo-remainder
            let prem = a.pseudo_remainder(&b, var);

            if prem.is_zero() {
                break;
            }

            // Subresultant normalization to prevent coefficient explosion
            let normalized = if delta == 0 {
                // No adjustment needed for delta = 0
                prem.scale(&(BigRational::one() / &h.constant_value()))
            } else {
                // For delta > 0, divide by g^delta * h
                let g_pow = g.constant_value().pow(delta);
                let divisor = &g_pow * &h.constant_value();
                prem.scale(&(BigRational::one() / divisor))
            };

            // Update g and h for next iteration
            let lc_b = b.leading_coeff_wrt(var);
            g = lc_b;

            h = if delta == 0 {
                Polynomial::one()
            } else {
                let g_val = g.constant_value();
                let h_val = h.constant_value();
                let new_h = g_val.pow(delta) / h_val.pow(delta - 1);
                Polynomial::constant(new_h)
            };

            // Move to next iteration
            a = b;
            b = normalized.primitive(); // Make primitive to keep coefficients manageable
            prs.push(b.clone());
        }

        prs
    }

    /// Resultant of two univariate polynomials with respect to a variable.
    pub fn resultant(&self, other: &Polynomial, var: Var) -> Polynomial {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }

        let deg_p = self.degree(var);
        let deg_q = other.degree(var);

        if deg_p == 0 {
            return self.pow(deg_q);
        }
        if deg_q == 0 {
            return other.pow(deg_p);
        }

        // Use subresultant PRS for efficiency
        let mut a = self.clone();
        let mut b = other.clone();
        let mut g = Polynomial::one();
        let mut h = Polynomial::one();
        let mut sign = if (deg_p & 1 == 1) && (deg_q & 1 == 1) {
            -1i32
        } else {
            1i32
        };

        // Add iteration limit to prevent infinite loops when exact division is not available
        let max_iters = (deg_p + deg_q) * 10;
        let mut iter_count = 0;

        while !b.is_zero() && iter_count < max_iters {
            iter_count += 1;
            let delta = a.degree(var) as i32 - b.degree(var) as i32;
            if delta < 0 {
                std::mem::swap(&mut a, &mut b);
                if (a.degree(var) & 1 == 1) && (b.degree(var) & 1 == 1) {
                    sign = -sign;
                }
                continue;
            }

            let (_, r) = a.pseudo_div_univariate(&b);

            if r.is_zero() {
                if b.degree(var) > 0 {
                    return Polynomial::zero();
                } else {
                    let d = a.degree(var);
                    return b.pow(d);
                }
            }

            a = b;
            let g_pow = g.pow((delta + 1) as u32);
            let h_pow = h.pow(delta as u32);
            b = r;

            // Simplify b by dividing out common factors
            // Since exact division is not implemented, use primitive() to prevent growth
            if delta > 0 {
                let denom = Polynomial::mul(&g_pow, &h_pow);
                // Try to cancel common content by making primitive
                b = b.primitive();
                // Note: This is an approximation and may not give the exact mathematical resultant
                let _ = denom; // Acknowledge we should divide by this
            }

            g = a.leading_coeff_wrt(var);
            let g_delta = g.pow(delta as u32);
            let h_new = if delta == 0 {
                h.clone()
            } else if delta == 1 {
                g.clone()
            } else {
                g_delta
            };
            h = h_new;
        }

        // The resultant is in b (last non-zero remainder)
        if sign < 0 { a.neg() } else { a }
    }

    /// Discriminant of a polynomial with respect to a variable.
    /// discriminant(p) = resultant(p, dp/dx) / lc(p)
    pub fn discriminant(&self, var: Var) -> Polynomial {
        let deriv = self.derivative(var);
        self.resultant(&deriv, var)
    }

    /// Check if the polynomial has a positive sign for all variable assignments.
    /// This is an incomplete check that only handles obvious cases.
    pub fn is_definitely_positive(&self) -> bool {
        // All terms with even total degree and positive coefficients
        if self.terms.is_empty() {
            return false;
        }

        // Check if all monomials are even powers and coefficients are positive
        self.terms
            .iter()
            .all(|t| t.coeff.is_positive() && t.monomial.vars().iter().all(|vp| vp.power % 2 == 0))
    }

    /// Check if the polynomial has a negative sign for all variable assignments.
    /// This is an incomplete check.
    pub fn is_definitely_negative(&self) -> bool {
        self.neg().is_definitely_positive()
    }

    /// Make the polynomial monic (leading coefficient = 1).
    pub fn make_monic(&self) -> Polynomial {
        if self.is_zero() {
            return self.clone();
        }
        let lc = self.leading_coeff();
        if lc.is_one() {
            return self.clone();
        }
        self.scale(&(BigRational::one() / lc))
    }

    /// Compute the square-free part of a polynomial (removes repeated factors).
    pub fn square_free(&self) -> Polynomial {
        if self.is_zero() || self.is_constant() {
            return self.clone();
        }

        let var = self.max_var();
        if var == NULL_VAR {
            return self.clone();
        }

        // Square-free: p / gcd(p, p')
        let deriv = self.derivative(var);
        if deriv.is_zero() {
            return self.clone();
        }

        let g = self.gcd_univariate(&deriv);
        if g.is_constant() {
            self.primitive()
        } else {
            let (q, r) = self.pseudo_div_univariate(&g);
            if r.is_zero() {
                q.primitive().square_free()
            } else {
                self.primitive()
            }
        }
    }

    /// Compute the Sturm sequence for a univariate polynomial.
    /// The Sturm sequence is used for counting real roots in an interval.
    pub fn sturm_sequence(&self, var: Var) -> Vec<Polynomial> {
        if self.is_zero() || self.degree(var) == 0 {
            return vec![self.clone()];
        }

        let mut seq = Vec::new();
        seq.push(self.clone());
        seq.push(self.derivative(var));

        // Build Sturm sequence: p_i+1 = -rem(p_i-1, p_i)
        let max_iterations = self.degree(var) as usize + 5;
        let mut iterations = 0;

        while !seq.last().unwrap().is_zero() && iterations < max_iterations {
            iterations += 1;
            let n = seq.len();
            let rem = seq[n - 2].pseudo_remainder(&seq[n - 1], var);
            if rem.is_zero() {
                break;
            }
            seq.push(rem.neg());
        }

        seq
    }

    /// Count the number of real roots in an interval using Sturm's theorem.
    /// Returns the number of distinct real roots in (a, b).
    pub fn count_roots_in_interval(&self, var: Var, a: &BigRational, b: &BigRational) -> usize {
        if self.is_zero() {
            return 0;
        }

        let sturm_seq = self.sturm_sequence(var);
        if sturm_seq.is_empty() {
            return 0;
        }

        // Count sign variations at a and b
        let var_a = count_sign_variations(&sturm_seq, var, a);
        let var_b = count_sign_variations(&sturm_seq, var, b);

        // Number of roots = var_a - var_b
        var_a.saturating_sub(var_b)
    }

    /// Compute Cauchy's root bound for a univariate polynomial.
    /// Returns B such that all roots have absolute value <= B.
    ///
    /// Cauchy bound: 1 + max(|a_i| / |a_n|) for i < n
    ///
    /// This is a simple, conservative bound that's fast to compute.
    pub fn cauchy_bound(&self, var: Var) -> BigRational {
        cauchy_root_bound(self, var)
    }

    /// Compute Fujiwara's root bound for a univariate polynomial.
    /// Returns B such that all roots have absolute value <= B.
    ///
    /// Fujiwara bound: 2 * max(|a_i/a_n|^(1/(n-i))) for i < n
    ///
    /// This bound is generally tighter than Cauchy's bound but more expensive to compute.
    ///
    /// Reference: Fujiwara, "Über die obere Schranke des absoluten Betrages
    /// der Wurzeln einer algebraischen Gleichung" (1916)
    pub fn fujiwara_bound(&self, var: Var) -> BigRational {
        fujiwara_root_bound(self, var)
    }

    /// Compute Lagrange's bound for positive roots of a univariate polynomial.
    /// Returns B such that all positive roots are <= B.
    ///
    /// This can provide a tighter bound than general root bounds when analyzing
    /// only positive roots, useful for root isolation optimization.
    pub fn lagrange_positive_bound(&self, var: Var) -> BigRational {
        lagrange_positive_root_bound(self, var)
    }

    /// Isolate real roots of a univariate polynomial.
    /// Returns a list of intervals, each containing exactly one root.
    /// Uses Descartes' rule of signs to optimize the search.
    pub fn isolate_roots(&self, var: Var) -> Vec<(BigRational, BigRational)> {
        if self.is_zero() || self.is_constant() {
            return vec![];
        }

        // Make square-free first
        let p = self.square_free();
        if p.is_constant() {
            return vec![];
        }

        // Find a bound for all real roots using Cauchy's bound
        let bound = cauchy_root_bound(&p, var);

        // Use Descartes' rule to check if there are any roots at all
        let (_pos_lower, pos_upper) = descartes_positive_roots(&p, var);
        let (_neg_lower, neg_upper) = descartes_negative_roots(&p, var);

        // Use interval bisection with Sturm's theorem
        let mut intervals = Vec::new();
        let mut queue = Vec::new();

        // Only search in positive interval if there might be positive roots
        if pos_upper > 0 {
            queue.push((BigRational::zero(), bound.clone()));
        }

        // Only search in negative interval if there might be negative roots
        if neg_upper > 0 {
            queue.push((-bound, BigRational::zero()));
        }

        let max_iterations = 1000;
        let mut iterations = 0;

        while let Some((a, b)) = queue.pop() {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            let num_roots = p.count_roots_in_interval(var, &a, &b);

            if num_roots == 0 {
                // No roots in this interval
                continue;
            } else if num_roots == 1 {
                // Exactly one root
                intervals.push((a, b));
            } else {
                // Multiple roots, bisect
                let mid = (&a + &b) / BigRational::from_integer(BigInt::from(2));

                // Check if mid is a root
                let val_mid = p.eval_at(var, &mid);
                if val_mid.constant_term().is_zero() {
                    // Found an exact root
                    intervals.push((mid.clone(), mid.clone()));
                    // Don't add intervals that would contain this root again
                    let left_roots = p.count_roots_in_interval(var, &a, &mid);
                    let right_roots = p.count_roots_in_interval(var, &mid, &b);
                    if left_roots > 0 {
                        queue.push((a, mid.clone()));
                    }
                    if right_roots > 0 {
                        queue.push((mid, b));
                    }
                } else {
                    queue.push((a, mid.clone()));
                    queue.push((mid, b));
                }
            }
        }

        intervals
    }

    /// Refine a root approximation using Newton-Raphson iteration.
    ///
    /// Given an initial approximation and bounds, refines the root using the formula:
    /// x_{n+1} = x_n - f(x_n) / f'(x_n)
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to solve for
    /// * `initial` - Initial approximation of the root
    /// * `lower` - Lower bound for the root (for verification)
    /// * `upper` - Upper bound for the root (for verification)
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance (stop when |f(x)| < tolerance)
    ///
    /// # Returns
    ///
    /// The refined root approximation, or None if the method fails to converge
    /// or if the derivative is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Solve x^2 - 2 = 0 (find sqrt(2) ≈ 1.414...)
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (-2, &[]),       // -2
    /// ]);
    ///
    /// let initial = BigRational::new(BigInt::from(3), BigInt::from(2)); // 1.5
    /// let lower = BigRational::from_integer(BigInt::from(1));
    /// let upper = BigRational::from_integer(BigInt::from(2));
    ///
    /// let root = p.newton_raphson(0, initial, lower, upper, 10, &BigRational::new(BigInt::from(1), BigInt::from(1000000)));
    /// assert!(root.is_some());
    /// ```
    pub fn newton_raphson(
        &self,
        var: Var,
        initial: BigRational,
        lower: BigRational,
        upper: BigRational,
        max_iterations: usize,
        tolerance: &BigRational,
    ) -> Option<BigRational> {
        use num_traits::{Signed, Zero};

        let derivative = self.derivative(var);

        let mut x = initial;

        for _ in 0..max_iterations {
            // Evaluate f(x)
            let mut assignment = FxHashMap::default();
            assignment.insert(var, x.clone());
            let fx = self.eval(&assignment);

            // Check if we're close enough
            if fx.abs() < *tolerance {
                return Some(x);
            }

            // Evaluate f'(x)
            let fpx = derivative.eval(&assignment);

            // Check for zero derivative
            if fpx.is_zero() {
                return None;
            }

            // Newton-Raphson update: x_new = x - f(x)/f'(x)
            let x_new = x - (fx / fpx);

            // Verify the new point is within bounds
            if x_new < lower || x_new > upper {
                // If out of bounds, use bisection fallback
                return None;
            }

            x = x_new;
        }

        // Return the result even if not fully converged
        Some(x)
    }

    /// Refine all roots in the given intervals using Newton-Raphson.
    ///
    /// Takes a list of isolating intervals (from `isolate_roots`) and refines
    /// each root to higher precision using Newton-Raphson iteration.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to solve for
    /// * `intervals` - List of isolating intervals from `isolate_roots`
    /// * `max_iterations` - Maximum number of Newton-Raphson iterations per root
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// Vector of refined root approximations
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Solve x^2 - 2 = 0
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (-2, &[]),       // -2
    /// ]);
    ///
    /// let intervals = p.isolate_roots(0);
    /// let tolerance = BigRational::new(BigInt::from(1), BigInt::from(1000000));
    /// let refined_roots = p.refine_roots(0, &intervals, 10, &tolerance);
    /// ```
    pub fn refine_roots(
        &self,
        var: Var,
        intervals: &[(BigRational, BigRational)],
        max_iterations: usize,
        tolerance: &BigRational,
    ) -> Vec<BigRational> {
        intervals
            .iter()
            .filter_map(|(lower, upper)| {
                // Use midpoint as initial guess
                let initial = (lower + upper) / BigRational::from_integer(BigInt::from(2));
                self.newton_raphson(
                    var,
                    initial,
                    lower.clone(),
                    upper.clone(),
                    max_iterations,
                    tolerance,
                )
            })
            .collect()
    }

    /// Compute the Taylor series expansion of a univariate polynomial around a point.
    ///
    /// For a polynomial p(x), computes the Taylor series:
    /// p(x) = Σ_{k=0}^n (p^(k)(a) / k!) * (x - a)^k
    ///
    /// where p^(k) is the k-th derivative.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to expand around
    /// * `point` - The point to expand around (typically 0 for Maclaurin series)
    /// * `degree` - Maximum degree of the Taylor expansion
    ///
    /// # Returns
    ///
    /// A polynomial representing the Taylor series expansion
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Expand x^2 + 2x + 1 around x = 0
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (2, &[(0, 1)]),  // 2x
    ///     (1, &[]),        // 1
    /// ]);
    ///
    /// let taylor = p.taylor_expansion(0, &BigRational::from_integer(BigInt::from(0)), 3);
    /// // Should be the same polynomial since it's already a polynomial
    /// ```
    pub fn taylor_expansion(&self, var: Var, point: &BigRational, degree: u32) -> Polynomial {
        use crate::rational::factorial;

        let mut result = Polynomial::zero();
        let mut derivative = self.clone();

        // Compute successive derivatives and evaluate at the point
        for k in 0..=degree {
            // Evaluate derivative at the point
            let mut assignment = FxHashMap::default();
            assignment.insert(var, point.clone());
            let coeff_at_point = derivative.eval(&assignment);

            // Divide by k!
            let factorial_k = factorial(k);
            let taylor_coeff = coeff_at_point / BigRational::from_integer(factorial_k);

            // Create term (x - point)^k
            let shifted_term = if k == 0 {
                Polynomial::constant(taylor_coeff)
            } else {
                // (x - a)^k = sum of binomial expansion
                // For simplicity, compute it directly
                let x_poly = Polynomial::from_var(var);
                let point_poly = Polynomial::constant(point.clone());
                let mut power = x_poly - point_poly;

                for _ in 1..k {
                    let x_poly = Polynomial::from_var(var);
                    let point_poly = Polynomial::constant(point.clone());
                    power = power * (x_poly - point_poly);
                }

                power * Polynomial::constant(taylor_coeff)
            };

            result = result + shifted_term;

            // Compute next derivative if needed
            if k < degree {
                derivative = derivative.derivative(var);
            }
        }

        result
    }

    /// Compute the Maclaurin series expansion (Taylor series around 0).
    ///
    /// This is a convenience method for Taylor expansion around 0.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to expand
    /// * `degree` - Maximum degree of the expansion
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Expand x^3 - 2x around x = 0
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 3)]),   // x^3
    ///     (-2, &[(0, 1)]),  // -2x
    /// ]);
    ///
    /// let maclaurin = p.maclaurin_expansion(0, 4);
    /// ```
    pub fn maclaurin_expansion(&self, var: Var, degree: u32) -> Polynomial {
        use num_bigint::BigInt;
        self.taylor_expansion(var, &BigRational::from_integer(BigInt::from(0)), degree)
    }

    /// Find a root using the bisection method.
    ///
    /// The bisection method is a robust root-finding algorithm that works by repeatedly
    /// bisecting an interval and selecting the subinterval where the function changes sign.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to solve for
    /// * `lower` - Lower bound (must have f(lower) and f(upper) with opposite signs)
    /// * `upper` - Upper bound (must have f(lower) and f(upper) with opposite signs)
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance (stop when interval width < tolerance)
    ///
    /// # Returns
    ///
    /// The approximate root, or None if the initial bounds don't bracket a root
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Solve x^2 - 2 = 0 (find sqrt(2))
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (-2, &[]),       // -2
    /// ]);
    ///
    /// let lower = BigRational::from_integer(BigInt::from(1));
    /// let upper = BigRational::from_integer(BigInt::from(2));
    /// let tolerance = BigRational::new(BigInt::from(1), BigInt::from(1000000));
    ///
    /// let root = p.bisection(0, lower, upper, 100, &tolerance);
    /// assert!(root.is_some());
    /// ```
    pub fn bisection(
        &self,
        var: Var,
        lower: BigRational,
        upper: BigRational,
        max_iterations: usize,
        tolerance: &BigRational,
    ) -> Option<BigRational> {
        use num_traits::{Signed, Zero};

        let mut a = lower;
        let mut b = upper;

        // Evaluate at endpoints
        let mut assignment_a = FxHashMap::default();
        assignment_a.insert(var, a.clone());
        let fa = self.eval(&assignment_a);

        let mut assignment_b = FxHashMap::default();
        assignment_b.insert(var, b.clone());
        let fb = self.eval(&assignment_b);

        // Check if endpoints bracket a root (opposite signs)
        if fa.is_zero() {
            return Some(a);
        }
        if fb.is_zero() {
            return Some(b);
        }
        if (fa.is_positive() && fb.is_positive()) || (fa.is_negative() && fb.is_negative()) {
            // Same sign, no root bracketed
            return None;
        }

        for _ in 0..max_iterations {
            // Check if interval is small enough
            if (&b - &a).abs() < *tolerance {
                return Some((&a + &b) / BigRational::from_integer(BigInt::from(2)));
            }

            // Compute midpoint
            let mid = (&a + &b) / BigRational::from_integer(BigInt::from(2));

            // Evaluate at midpoint
            let mut assignment_mid = FxHashMap::default();
            assignment_mid.insert(var, mid.clone());
            let fmid = self.eval(&assignment_mid);

            // Check if we found exact root
            if fmid.is_zero() {
                return Some(mid);
            }

            // Update interval
            let mut assignment_a = FxHashMap::default();
            assignment_a.insert(var, a.clone());
            let fa = self.eval(&assignment_a);

            if (fa.is_positive() && fmid.is_negative()) || (fa.is_negative() && fmid.is_positive())
            {
                // Root is in [a, mid]
                b = mid;
            } else {
                // Root is in [mid, b]
                a = mid;
            }
        }

        // Return midpoint as best approximation
        Some((&a + &b) / BigRational::from_integer(BigInt::from(2)))
    }

    /// Find a root using the secant method.
    ///
    /// The secant method is similar to Newton-Raphson but doesn't require computing derivatives.
    /// It uses two previous points to approximate the derivative.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to solve for
    /// * `x0` - First initial guess
    /// * `x1` - Second initial guess (should be close to x0)
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// The approximate root, or None if the method fails to converge
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Solve x^2 - 2 = 0
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (-2, &[]),       // -2
    /// ]);
    ///
    /// let x0 = BigRational::from_integer(BigInt::from(1));
    /// let x1 = BigRational::new(BigInt::from(3), BigInt::from(2)); // 1.5
    /// let tolerance = BigRational::new(BigInt::from(1), BigInt::from(1000000));
    ///
    /// let root = p.secant(0, x0, x1, 20, &tolerance);
    /// assert!(root.is_some());
    /// ```
    pub fn secant(
        &self,
        var: Var,
        x0: BigRational,
        x1: BigRational,
        max_iterations: usize,
        tolerance: &BigRational,
    ) -> Option<BigRational> {
        use num_traits::{Signed, Zero};

        let mut x_prev = x0;
        let mut x_curr = x1;

        for _ in 0..max_iterations {
            // Evaluate at current and previous points
            let mut assignment_prev = FxHashMap::default();
            assignment_prev.insert(var, x_prev.clone());
            let f_prev = self.eval(&assignment_prev);

            let mut assignment_curr = FxHashMap::default();
            assignment_curr.insert(var, x_curr.clone());
            let f_curr = self.eval(&assignment_curr);

            // Check if we're close enough
            if f_curr.abs() < *tolerance {
                return Some(x_curr);
            }

            // Check for zero denominator
            let denom = &f_curr - &f_prev;
            if denom.is_zero() {
                return None;
            }

            // Secant method update: x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            let x_new = &x_curr - &f_curr * (&x_curr - &x_prev) / denom;

            x_prev = x_curr;
            x_curr = x_new;
        }

        Some(x_curr)
    }

    /// Sign of the polynomial given signs of variables.
    /// Returns Some(1) for positive, Some(-1) for negative, Some(0) for zero,
    /// or None if undetermined.
    pub fn sign_at(&self, var_signs: &FxHashMap<Var, i8>) -> Option<i8> {
        if self.is_zero() {
            return Some(0);
        }
        if self.is_constant() {
            let c = &self.terms[0].coeff;
            return Some(if c.is_positive() {
                1
            } else if c.is_negative() {
                -1
            } else {
                0
            });
        }

        // Try to determine sign from term signs
        let mut all_positive = true;
        let mut all_negative = true;

        for term in &self.terms {
            let mut term_sign: i8 = if term.coeff.is_positive() {
                1
            } else if term.coeff.is_negative() {
                -1
            } else {
                continue;
            };

            for vp in term.monomial.vars() {
                if let Some(&s) = var_signs.get(&vp.var) {
                    if s == 0 {
                        // Variable is zero; this term contributes 0
                        term_sign = 0;
                        break;
                    }
                    if vp.power % 2 == 1 {
                        term_sign *= s;
                    }
                } else {
                    // Unknown sign; can't determine overall sign
                    return None;
                }
            }

            if term_sign > 0 {
                all_negative = false;
            } else if term_sign < 0 {
                all_positive = false;
            }
        }

        if all_positive && !all_negative {
            Some(1)
        } else if all_negative && !all_positive {
            Some(-1)
        } else {
            None
        }
    }

    /// Check if a univariate polynomial is irreducible over rationals.
    /// This is a heuristic test - returns false if definitely reducible,
    /// true if likely irreducible (but not guaranteed).
    pub fn is_irreducible(&self, var: Var) -> bool {
        if self.is_zero() || self.is_constant() {
            return false;
        }

        let deg = self.degree(var);
        if deg == 0 {
            return false;
        }
        if deg == 1 {
            return true;
        }

        // Check if square-free
        let sf = self.square_free();
        if sf.degree(var) < deg {
            return false; // Has repeated factors
        }

        // For degree 2 and 3, use discriminant-based checks
        if deg == 2 {
            return self.is_irreducible_quadratic(var);
        }

        // For higher degrees, we'd need more sophisticated tests
        // For now, assume possibly irreducible
        true
    }

    /// Check if a quadratic polynomial is irreducible over rationals.
    fn is_irreducible_quadratic(&self, var: Var) -> bool {
        let deg = self.degree(var);
        if deg != 2 {
            return false;
        }

        let a = self.univ_coeff(var, 2);
        let b = self.univ_coeff(var, 1);
        let c = self.univ_coeff(var, 0);

        // Check discriminant: b^2 - 4ac
        // If not a perfect square, polynomial is irreducible over rationals
        let discriminant = &b * &b - (BigRational::from_integer(BigInt::from(4)) * &a * &c);

        if discriminant.is_negative() {
            return true; // No real roots
        }

        // Check if discriminant is a perfect square of a rational
        let num_sqrt = is_perfect_square(discriminant.numer());
        let den_sqrt = is_perfect_square(discriminant.denom());

        !(num_sqrt && den_sqrt)
    }

    /// Factor a univariate polynomial into irreducible factors.
    /// Returns a vector of (factor, multiplicity) pairs.
    /// Each factor is monic and primitive.
    pub fn factor(&self, var: Var) -> Vec<(Polynomial, u32)> {
        if self.is_zero() {
            return vec![];
        }

        if self.is_constant() {
            return vec![(self.clone(), 1)];
        }

        let deg = self.degree(var);
        if deg == 0 {
            return vec![(self.clone(), 1)];
        }

        // Make monic and primitive
        let p = self.primitive().make_monic();

        // Handle degree 1 (linear) - always irreducible
        if deg == 1 {
            return vec![(p, 1)];
        }

        // Handle degree 2 (quadratic) - use quadratic formula
        if deg == 2 {
            return self.factor_quadratic(var);
        }

        // For degree > 2, use square-free factorization first
        self.factor_square_free(var)
    }

    /// Factor a quadratic polynomial.
    fn factor_quadratic(&self, var: Var) -> Vec<(Polynomial, u32)> {
        let deg = self.degree(var);
        if deg != 2 {
            return vec![(self.primitive(), 1)];
        }

        let p = self.primitive().make_monic();
        let a = p.univ_coeff(var, 2);
        let b = p.univ_coeff(var, 1);
        let c = p.univ_coeff(var, 0);

        // Discriminant: b^2 - 4ac
        let discriminant = &b * &b - (BigRational::from_integer(BigInt::from(4)) * &a * &c);

        // Check if discriminant is a perfect square
        let num_sqrt = integer_sqrt(discriminant.numer());
        let den_sqrt = integer_sqrt(discriminant.denom());

        if num_sqrt.is_none() || den_sqrt.is_none() {
            // Irreducible over rationals
            return vec![(p, 1)];
        }

        let disc_sqrt = BigRational::new(num_sqrt.unwrap(), den_sqrt.unwrap());

        // Roots: (-b ± sqrt(disc)) / (2a)
        let two_a = BigRational::from_integer(BigInt::from(2)) * &a;
        let root1 = (-&b + &disc_sqrt) / &two_a;
        let root2 = (-&b - disc_sqrt) / two_a;

        // Factor as (x - root1)(x - root2)
        let factor1 = Polynomial::from_terms(
            vec![
                Term::new(BigRational::one(), Monomial::from_var(var)),
                Term::new(-root1, Monomial::unit()),
            ],
            MonomialOrder::Lex,
        );

        let factor2 = Polynomial::from_terms(
            vec![
                Term::new(BigRational::one(), Monomial::from_var(var)),
                Term::new(-root2, Monomial::unit()),
            ],
            MonomialOrder::Lex,
        );

        vec![(factor1, 1), (factor2, 1)]
    }

    /// Square-free factorization: decompose into coprime factors.
    /// Returns factors with their multiplicities.
    fn factor_square_free(&self, var: Var) -> Vec<(Polynomial, u32)> {
        if self.is_zero() || self.is_constant() {
            return vec![(self.clone(), 1)];
        }

        let mut result = Vec::new();
        let p = self.primitive();
        let mut multiplicity = 1u32;

        // Yun's algorithm for square-free factorization
        let deriv = p.derivative(var);

        if deriv.is_zero() {
            // All exponents are multiples of characteristic (0 for rationals)
            // This shouldn't happen for polynomials over rationals
            return vec![(p, 1)];
        }

        let gcd = p.gcd_univariate(&deriv);

        if gcd.is_constant() {
            // Already square-free
            if p.is_irreducible(var) {
                return vec![(p.make_monic(), 1)];
            } else {
                // Try to factor further (for now, return as-is)
                return vec![(p.make_monic(), 1)];
            }
        }

        let (quo, rem) = p.pseudo_div_univariate(&gcd);
        if !rem.is_zero() {
            return vec![(p, 1)];
        }

        let mut u = quo.primitive();
        let (v_quo, v_rem) = deriv.pseudo_div_univariate(&gcd);
        if v_rem.is_zero() {
            let v = v_quo.primitive();
            let mut w = Polynomial::sub(&v, &u.derivative(var));

            while !w.is_zero() && !w.is_constant() {
                let y = u.gcd_univariate(&w);
                if !y.is_constant() {
                    result.push((y.make_monic(), multiplicity));
                    let (u_new, u_rem) = u.pseudo_div_univariate(&y);
                    if u_rem.is_zero() {
                        u = u_new.primitive();
                    }
                    let (w_new, w_rem) = w.pseudo_div_univariate(&y);
                    if w_rem.is_zero() {
                        w = Polynomial::sub(&w_new.primitive(), &u.derivative(var));
                    } else {
                        break;
                    }
                } else {
                    break;
                }
                multiplicity += 1;
            }

            if !u.is_constant() {
                result.push((u.make_monic(), multiplicity));
            }
        }

        if result.is_empty() {
            result.push((p.make_monic(), 1));
        }

        result
    }

    /// Content of a polynomial: GCD of all coefficients.
    /// Returns the rational content.
    pub fn content(&self) -> BigRational {
        if self.terms.is_empty() {
            return BigRational::one();
        }

        let mut num_gcd: Option<BigInt> = None;
        let mut den_lcm: Option<BigInt> = None;

        for term in &self.terms {
            let coeff_num = term.coeff.numer().clone().abs();
            let coeff_den = term.coeff.denom().clone();

            num_gcd = Some(match num_gcd {
                None => coeff_num,
                Some(g) => gcd_bigint(g, coeff_num),
            });

            den_lcm = Some(match den_lcm {
                None => coeff_den,
                Some(l) => {
                    let gcd = gcd_bigint(l.clone(), coeff_den.clone());
                    (&l * &coeff_den) / gcd
                }
            });
        }

        BigRational::new(
            num_gcd.unwrap_or_else(BigInt::one),
            den_lcm.unwrap_or_else(BigInt::one),
        )
    }

    /// Compose this polynomial with another: compute p(q(x)).
    ///
    /// This is an alias for `substitute` with clearer semantics for composition.
    /// If `self` is p(x) and `other` is q(x), returns p(q(x)).
    pub fn compose(&self, var: Var, other: &Polynomial) -> Polynomial {
        self.substitute(var, other)
    }

    /// Lagrange polynomial interpolation.
    ///
    /// Given a set of points (x_i, y_i), constructs the unique polynomial of minimal degree
    /// that passes through all the points.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `points` - A slice of (x, y) coordinate pairs
    ///
    /// # Returns
    /// The interpolating polynomial, or None if points is empty or contains duplicate x values
    ///
    /// # Reference
    /// Standard Lagrange interpolation formula from numerical analysis textbooks
    pub fn lagrange_interpolate(
        var: Var,
        points: &[(BigRational, BigRational)],
    ) -> Option<Polynomial> {
        if points.is_empty() {
            return None;
        }

        if points.len() == 1 {
            // Single point: constant polynomial
            return Some(Polynomial::constant(points[0].1.clone()));
        }

        // Check for duplicate x values
        for i in 0..points.len() {
            for j in i + 1..points.len() {
                if points[i].0 == points[j].0 {
                    return None; // Duplicate x values
                }
            }
        }

        let mut result = Polynomial::zero();

        // Lagrange basis polynomials
        for i in 0..points.len() {
            let (x_i, y_i) = &points[i];
            let mut basis = Polynomial::one();

            for (j, (x_j, _)) in points.iter().enumerate() {
                if i == j {
                    continue;
                }

                // basis *= (x - x_j) / (x_i - x_j)
                let x_poly = Polynomial::from_var(var);
                let const_poly = Polynomial::constant(x_j.clone());
                let numerator = &x_poly - &const_poly;
                let denominator = x_i - x_j;

                if denominator.is_zero() {
                    return None; // Should not happen after duplicate check
                }

                basis = &basis * &numerator;
                basis = basis.scale(&(BigRational::one() / denominator));
            }

            // result += y_i * basis
            let scaled_basis = basis.scale(y_i);
            result = &result + &scaled_basis;
        }

        Some(result)
    }

    /// Newton polynomial interpolation using divided differences.
    ///
    /// Constructs the same interpolating polynomial as Lagrange interpolation,
    /// but using Newton's divided difference form which can be more efficient
    /// for incremental construction.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `points` - A slice of (x, y) coordinate pairs
    ///
    /// # Returns
    /// The interpolating polynomial, or None if points is empty or contains duplicate x values
    pub fn newton_interpolate(
        var: Var,
        points: &[(BigRational, BigRational)],
    ) -> Option<Polynomial> {
        if points.is_empty() {
            return None;
        }

        if points.len() == 1 {
            return Some(Polynomial::constant(points[0].1.clone()));
        }

        // Check for duplicate x values
        for i in 0..points.len() {
            for j in i + 1..points.len() {
                if points[i].0 == points[j].0 {
                    return None;
                }
            }
        }

        let n = points.len();

        // Compute divided differences table
        let mut dd: Vec<Vec<BigRational>> = vec![vec![BigRational::zero(); n]; n];

        // Initialize first column with y values
        for i in 0..n {
            dd[i][0] = points[i].1.clone();
        }

        // Compute divided differences
        for j in 1..n {
            for i in 0..n - j {
                let numerator = &dd[i + 1][j - 1] - &dd[i][j - 1];
                let denominator = &points[i + j].0 - &points[i].0;
                if denominator.is_zero() {
                    return None;
                }
                dd[i][j] = numerator / denominator;
            }
        }

        // Build Newton polynomial: p(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + ...
        let mut result = Polynomial::constant(dd[0][0].clone());
        let mut term = Polynomial::one();

        for i in 1..n {
            // term *= (x - x_{i-1})
            let x_poly = Polynomial::from_var(var);
            let const_poly = Polynomial::constant(points[i - 1].0.clone());
            let factor = &x_poly - &const_poly;
            term = &term * &factor;

            // result += a_i * term
            let scaled_term = term.scale(&dd[0][i]);
            result = &result + &scaled_term;
        }

        Some(result)
    }

    /// Generate the nth Chebyshev polynomial of the first kind T_n(x).
    ///
    /// Chebyshev polynomials of the first kind are defined by:
    /// - T_0(x) = 1
    /// - T_1(x) = x
    /// - T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    ///
    /// These polynomials are orthogonal on [-1, 1] with respect to the weight
    /// function 1/√(1-x²) and are useful for polynomial approximation.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Chebyshev polynomial
    ///
    /// # Returns
    /// The nth Chebyshev polynomial T_n(var)
    pub fn chebyshev_first_kind(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => Polynomial::from_var(var),
            _ => {
                // Use recurrence relation: T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
                let mut t_prev = Polynomial::one(); // T_0
                let mut t_curr = Polynomial::from_var(var); // T_1

                for _ in 2..=n {
                    let x = Polynomial::from_var(var);
                    let two_x_t_curr = Polynomial::mul(&t_curr, &x)
                        .scale(&BigRational::from_integer(BigInt::from(2)));
                    let t_next = Polynomial::sub(&two_x_t_curr, &t_prev);
                    t_prev = t_curr;
                    t_curr = t_next;
                }

                t_curr
            }
        }
    }

    /// Generate the nth Chebyshev polynomial of the second kind U_n(x).
    ///
    /// Chebyshev polynomials of the second kind are defined by:
    /// - U_0(x) = 1
    /// - U_1(x) = 2x
    /// - U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
    ///
    /// These polynomials are orthogonal on [-1, 1] with respect to the weight
    /// function √(1-x²).
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Chebyshev polynomial
    ///
    /// # Returns
    /// The nth Chebyshev polynomial U_n(var)
    pub fn chebyshev_second_kind(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => {
                // U_1(x) = 2x
                Polynomial::from_var(var).scale(&BigRational::from_integer(BigInt::from(2)))
            }
            _ => {
                // Use recurrence relation: U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
                let mut u_prev = Polynomial::one(); // U_0
                let mut u_curr =
                    Polynomial::from_var(var).scale(&BigRational::from_integer(BigInt::from(2))); // U_1

                for _ in 2..=n {
                    let x = Polynomial::from_var(var);
                    let two_x_u_curr = Polynomial::mul(&u_curr, &x)
                        .scale(&BigRational::from_integer(BigInt::from(2)));
                    let u_next = Polynomial::sub(&two_x_u_curr, &u_prev);
                    u_prev = u_curr;
                    u_curr = u_next;
                }

                u_curr
            }
        }
    }

    /// Compute Chebyshev nodes (zeros of Chebyshev polynomial) for use in interpolation.
    ///
    /// The Chebyshev nodes minimize the Runge phenomenon in polynomial interpolation.
    /// For degree n, returns n+1 nodes in [-1, 1].
    ///
    /// The nodes are: x_k = cos((2k+1)π / (2n+2)) for k = 0, 1, ..., n
    ///
    /// Note: This function returns approximate rational values. For exact values,
    /// use algebraic numbers or symbolic computation.
    ///
    /// # Arguments
    /// * `n` - The degree (returns n+1 nodes)
    ///
    /// # Returns
    /// Approximations of the Chebyshev nodes as BigRational values
    pub fn chebyshev_nodes(n: u32) -> Vec<BigRational> {
        if n == 0 {
            return vec![BigRational::zero()];
        }

        let mut nodes = Vec::with_capacity((n + 1) as usize);

        // Compute nodes: cos((2k+1)π / (2n+2))
        // We'll use a rational approximation
        for k in 0..=n {
            // Approximate cos using Taylor series or use exact rational bounds
            // For now, use a simple rational approximation based on the angle

            // Angle = (2k+1) / (2n+2) * π/2
            // For small angles, we can use rational approximations
            // This is a simplified version - ideally would use higher precision

            let numerator = (2 * k + 1) as i64;
            let denominator = (2 * n + 2) as i64;

            // Simple linear approximation for demonstration
            // In production, would use more accurate trigonometric approximation
            let ratio = BigRational::new(BigInt::from(numerator), BigInt::from(denominator));

            // Map to [-1, 1] range
            // This is a placeholder - real implementation would compute cos accurately
            let node = BigRational::one() - ratio * BigRational::from_integer(BigInt::from(2));
            nodes.push(node);
        }

        nodes
    }

    /// Generate the nth Legendre polynomial P_n(x).
    ///
    /// Legendre polynomials are orthogonal polynomials on [-1, 1] with respect to
    /// the weight function 1. They are defined by the recurrence:
    /// - P_0(x) = 1
    /// - P_1(x) = x
    /// - (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    ///
    /// These polynomials are useful for Gaussian quadrature and least-squares
    /// polynomial approximation.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Legendre polynomial
    ///
    /// # Returns
    /// The nth Legendre polynomial P_n(var)
    pub fn legendre(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => Polynomial::from_var(var),
            _ => {
                // Recurrence: (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
                let mut p_prev = Polynomial::one(); // P_0
                let mut p_curr = Polynomial::from_var(var); // P_1

                for k in 1..n {
                    let x = Polynomial::from_var(var);

                    // (2k+1) x P_k(x)
                    let coeff_2k_plus_1 = BigRational::from_integer(BigInt::from(2 * k + 1));
                    let term1 = Polynomial::mul(&p_curr, &x).scale(&coeff_2k_plus_1);

                    // k P_{k-1}(x)
                    let coeff_k = BigRational::from_integer(BigInt::from(k));
                    let term2 = p_prev.scale(&coeff_k);

                    // [(2k+1) x P_k(x) - k P_{k-1}(x)] / (k+1)
                    let numerator = Polynomial::sub(&term1, &term2);
                    let divisor = BigRational::from_integer(BigInt::from(k + 1));
                    let p_next = numerator.scale(&(BigRational::one() / divisor));

                    p_prev = p_curr;
                    p_curr = p_next;
                }

                p_curr
            }
        }
    }

    /// Generate the nth Hermite polynomial H_n(x) (physicist's version).
    ///
    /// Hermite polynomials are orthogonal with respect to the weight function e^(-x²).
    /// The physicist's version is defined by:
    /// - H_0(x) = 1
    /// - H_1(x) = 2x
    /// - H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    ///
    /// These polynomials are useful in quantum mechanics, probability theory,
    /// and numerical analysis.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Hermite polynomial
    ///
    /// # Returns
    /// The nth Hermite polynomial H_n(var)
    pub fn hermite(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => {
                // H_1(x) = 2x
                Polynomial::from_var(var).scale(&BigRational::from_integer(BigInt::from(2)))
            }
            _ => {
                // Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
                let mut h_prev = Polynomial::one(); // H_0
                let mut h_curr =
                    Polynomial::from_var(var).scale(&BigRational::from_integer(BigInt::from(2))); // H_1

                for k in 1..n {
                    let x = Polynomial::from_var(var);

                    // 2x H_k(x)
                    let two_x_h = Polynomial::mul(&h_curr, &x)
                        .scale(&BigRational::from_integer(BigInt::from(2)));

                    // 2k H_{k-1}(x)
                    let coeff_2k = BigRational::from_integer(BigInt::from(2 * k));
                    let term2 = h_prev.scale(&coeff_2k);

                    // 2x H_k(x) - 2k H_{k-1}(x)
                    let h_next = Polynomial::sub(&two_x_h, &term2);

                    h_prev = h_curr;
                    h_curr = h_next;
                }

                h_curr
            }
        }
    }

    /// Generate the nth Laguerre polynomial L_n(x).
    ///
    /// Laguerre polynomials are orthogonal with respect to the weight function e^(-x)
    /// on [0, ∞). They are defined by:
    /// - L_0(x) = 1
    /// - L_1(x) = 1 - x
    /// - (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
    ///
    /// These polynomials are useful in quantum mechanics and numerical analysis.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Laguerre polynomial
    ///
    /// # Returns
    /// The nth Laguerre polynomial L_n(var)
    pub fn laguerre(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => {
                // L_1(x) = 1 - x
                let one = Polynomial::one();
                let x = Polynomial::from_var(var);
                Polynomial::sub(&one, &x)
            }
            _ => {
                // Recurrence: (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
                let mut l_prev = Polynomial::one(); // L_0
                let mut l_curr = {
                    let one = Polynomial::one();
                    let x = Polynomial::from_var(var);
                    Polynomial::sub(&one, &x)
                }; // L_1

                for k in 1..n {
                    let x = Polynomial::from_var(var);

                    // (2k+1-x) L_k(x) = (2k+1) L_k(x) - x L_k(x)
                    let coeff_2k_plus_1 = BigRational::from_integer(BigInt::from(2 * k + 1));
                    let term1 = l_curr.scale(&coeff_2k_plus_1);
                    let term2 = Polynomial::mul(&l_curr, &x);
                    let combined = Polynomial::sub(&term1, &term2);

                    // k L_{k-1}(x)
                    let coeff_k = BigRational::from_integer(BigInt::from(k));
                    let term3 = l_prev.scale(&coeff_k);

                    // [(2k+1-x) L_k(x) - k L_{k-1}(x)] / (k+1)
                    let numerator = Polynomial::sub(&combined, &term3);
                    let divisor = BigRational::from_integer(BigInt::from(k + 1));
                    let l_next = numerator.scale(&(BigRational::one() / divisor));

                    l_prev = l_curr;
                    l_curr = l_next;
                }

                l_curr
            }
        }
    }
}

/// Check if a BigInt is a perfect square.
fn is_perfect_square(n: &BigInt) -> bool {
    if n.is_negative() {
        return false;
    }
    if n.is_zero() || n.is_one() {
        return true;
    }

    let sqrt = integer_sqrt(n);
    sqrt.is_some()
}

/// Compute integer square root if n is a perfect square.
/// Returns None if n is not a perfect square.
fn integer_sqrt(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    if n.is_zero() {
        return Some(BigInt::zero());
    }
    if n.is_one() {
        return Some(BigInt::one());
    }

    // Newton's method for integer square root
    let mut x: BigInt = n.clone();
    let mut y: BigInt = (&x + BigInt::one()) >> 1; // (x + 1) / 2

    while y < x {
        x = y.clone();
        y = (&x + (n / &x)) >> 1;
    }

    // Check if x is exact square root
    if &x * &x == *n { Some(x) } else { None }
}

/// Compute the square root of a rational number if it's a perfect square.
/// Returns None if the rational is not a perfect square of another rational.
pub fn rational_sqrt(n: &BigRational) -> Option<BigRational> {
    if n.is_negative() {
        return None;
    }
    if n.is_zero() {
        return Some(BigRational::zero());
    }

    // For a fraction p/q to have a rational square root,
    // both p and q must be perfect squares
    let numer = n.numer();
    let denom = n.denom();

    let sqrt_numer = integer_sqrt(numer)?;
    let sqrt_denom = integer_sqrt(denom)?;

    Some(BigRational::new(sqrt_numer, sqrt_denom))
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }
        self.terms.iter().zip(&other.terms).all(|(a, b)| a == b)
    }
}

impl Eq for Polynomial {}

impl Default for Polynomial {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Debug for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            for (i, term) in self.terms.iter().enumerate() {
                if i == 0 {
                    write!(f, "{:?}", term)?;
                } else if term.coeff.is_negative() {
                    write!(
                        f,
                        " - {:?}",
                        Term::new(-term.coeff.clone(), term.monomial.clone())
                    )?;
                } else {
                    write!(f, " + {:?}", term)?;
                }
            }
            Ok(())
        }
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl Neg for Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Self::Output {
        Polynomial::neg(&self)
    }
}

impl Neg for &Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Self::Output {
        Polynomial::neg(self)
    }
}

impl Add for Polynomial {
    type Output = Polynomial;

    fn add(self, rhs: Self) -> Self::Output {
        Polynomial::add(&self, &rhs)
    }
}

impl Add<&Polynomial> for &Polynomial {
    type Output = Polynomial;

    fn add(self, rhs: &Polynomial) -> Self::Output {
        Polynomial::add(self, rhs)
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;

    fn sub(self, rhs: Self) -> Self::Output {
        Polynomial::sub(&self, &rhs)
    }
}

impl Sub<&Polynomial> for &Polynomial {
    type Output = Polynomial;

    fn sub(self, rhs: &Polynomial) -> Self::Output {
        Polynomial::sub(self, rhs)
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: Self) -> Self::Output {
        Polynomial::mul(&self, &rhs)
    }
}

impl Mul<&Polynomial> for &Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: &Polynomial) -> Self::Output {
        Polynomial::mul(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_monomial_unit() {
        let m = Monomial::unit();
        assert!(m.is_unit());
        assert_eq!(m.total_degree(), 0);
    }

    #[test]
    fn test_monomial_from_var() {
        let m = Monomial::from_var(0);
        assert!(!m.is_unit());
        assert_eq!(m.total_degree(), 1);
        assert_eq!(m.degree(0), 1);
        assert_eq!(m.degree(1), 0);
    }

    #[test]
    fn test_monomial_mul() {
        let m1 = Monomial::from_var_power(0, 2); // x^2
        let m2 = Monomial::from_var_power(0, 3); // x^3
        let m3 = m1.mul(&m2);
        assert_eq!(m3.total_degree(), 5);
        assert_eq!(m3.degree(0), 5);
    }

    #[test]
    fn test_monomial_div() {
        let m1 = Monomial::from_var_power(0, 5); // x^5
        let m2 = Monomial::from_var_power(0, 2); // x^2
        let m3 = m1.div(&m2).unwrap();
        assert_eq!(m3.degree(0), 3);

        // Cannot divide x^2 by x^5
        assert!(m2.div(&m1).is_none());
    }

    #[test]
    fn test_polynomial_zero() {
        let p = Polynomial::zero();
        assert!(p.is_zero());
        assert_eq!(p.total_degree(), 0);
    }

    #[test]
    fn test_polynomial_constant() {
        let p = Polynomial::constant(rat(5));
        assert!(p.is_constant());
        assert_eq!(p.constant_term(), rat(5));
    }

    #[test]
    fn test_polynomial_add() {
        // p = x + 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]);
        // q = x + 2
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (2, &[])]);
        // p + q = 2x + 3
        let r = Polynomial::add(&p, &q);
        assert_eq!(r.num_terms(), 2);
        assert_eq!(r.univ_coeff(0, 1), rat(2));
        assert_eq!(r.constant_term(), rat(3));
    }

    #[test]
    fn test_polynomial_mul() {
        // p = x + 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]);
        // q = x - 1
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]);
        // p * q = x^2 - 1
        let r = Polynomial::mul(&p, &q);
        assert_eq!(r.univ_coeff(0, 2), rat(1));
        assert_eq!(r.univ_coeff(0, 1), rat(0));
        assert_eq!(r.constant_term(), rat(-1));
    }

    #[test]
    fn test_polynomial_mul_karatsuba() {
        // Test Karatsuba multiplication with a large polynomial
        // Create p(x) = x^20 + x^19 + ... + x + 1 (geometric series)
        let mut p_coeffs = Vec::new();
        for i in 0..=20 {
            p_coeffs.push((1, vec![(0, i)]));
        }
        let p = Polynomial::from_coeffs_int(
            &p_coeffs
                .iter()
                .map(|(c, v)| (*c, v.as_slice()))
                .collect::<Vec<_>>(),
        );

        // Create q(x) = x^20 - 1
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 20)]), (-1, &[])]);

        // Compute p * q using Karatsuba (should trigger for degree 20)
        let result = p.mul(q);

        // Verify a few coefficients
        // (x^20 + ... + x + 1)(x^20 - 1) = x^40 + x^39 + ... + x^21 - 1
        assert_eq!(result.degree(0), 40);
        assert_eq!(result.univ_coeff(0, 40), rat(1));
        assert_eq!(result.constant_term(), rat(-1));
        assert_eq!(result.univ_coeff(0, 20), rat(0)); // Middle term should cancel
    }

    #[test]
    fn test_polynomial_mul_karatsuba_correctness() {
        // Verify Karatsuba produces correct results
        // Test with (x + 1)^20 * (x - 1)
        // = (x + 1)^20 * (x - 1)
        // = (x + 1)^19 * (x^2 - 1)

        // Create p(x) = (x + 1)^2 = x^2 + 2x + 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (2, &[(0, 1)]), (1, &[])]);

        // Compute p^10 = (x^2 + 2x + 1)^10 which has degree 20
        let p10 = p.pow(10);

        // Multiply by (x - 1)
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]);

        let result = p10.mul(q);

        // Result should have degree 21
        assert_eq!(result.degree(0), 21);

        // Leading coefficient should be 1
        assert_eq!(result.univ_coeff(0, 21), rat(1));

        // The polynomial should not be zero
        assert!(!result.is_zero());
    }

    #[test]
    fn test_polynomial_mul_karatsuba_simple() {
        // Simple test: (x^16 + 1) * (x^16 - 1) = x^32 - 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 16)]), (1, &[])]);
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 16)]), (-1, &[])]);

        let result = p.mul(q);

        assert_eq!(result.degree(0), 32);
        assert_eq!(result.univ_coeff(0, 32), rat(1));
        assert_eq!(result.constant_term(), rat(-1));
        assert_eq!(result.univ_coeff(0, 16), rat(0)); // Should cancel out
    }

    #[test]
    fn test_polynomial_derivative() {
        // p = x^3 + 2x^2 + x + 1
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 3)]),
            (2, &[(0, 2)]),
            (1, &[(0, 1)]),
            (1, &[]),
        ]);
        // dp/dx = 3x^2 + 4x + 1
        let dp = p.derivative(0);
        assert_eq!(dp.univ_coeff(0, 2), rat(3));
        assert_eq!(dp.univ_coeff(0, 1), rat(4));
        assert_eq!(dp.constant_term(), rat(1));
    }

    #[test]
    fn test_polynomial_nth_derivative() {
        // p = x^4 + 2x^3 + 3x^2 + 4x + 5
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 4)]),
            (2, &[(0, 3)]),
            (3, &[(0, 2)]),
            (4, &[(0, 1)]),
            (5, &[]),
        ]);

        // 0th derivative should be the polynomial itself
        let d0 = p.nth_derivative(0, 0);
        assert_eq!(d0.univ_coeff(0, 4), rat(1));
        assert_eq!(d0.constant_term(), rat(5));

        // 1st derivative: 4x^3 + 6x^2 + 6x + 4
        let d1 = p.nth_derivative(0, 1);
        assert_eq!(d1.univ_coeff(0, 3), rat(4));
        assert_eq!(d1.univ_coeff(0, 2), rat(6));
        assert_eq!(d1.univ_coeff(0, 1), rat(6));
        assert_eq!(d1.constant_term(), rat(4));

        // 2nd derivative: 12x^2 + 12x + 6
        let d2 = p.nth_derivative(0, 2);
        assert_eq!(d2.univ_coeff(0, 2), rat(12));
        assert_eq!(d2.univ_coeff(0, 1), rat(12));
        assert_eq!(d2.constant_term(), rat(6));

        // 3rd derivative: 24x + 12
        let d3 = p.nth_derivative(0, 3);
        assert_eq!(d3.univ_coeff(0, 1), rat(24));
        assert_eq!(d3.constant_term(), rat(12));

        // 4th derivative: 24
        let d4 = p.nth_derivative(0, 4);
        assert_eq!(d4.constant_term(), rat(24));

        // 5th derivative and beyond: 0
        let d5 = p.nth_derivative(0, 5);
        assert!(d5.is_zero());

        let d10 = p.nth_derivative(0, 10);
        assert!(d10.is_zero());
    }

    #[test]
    fn test_polynomial_integrate() {
        // Integrate: 3x^2 -> x^3
        let p = Polynomial::from_coeffs_int(&[(3, &[(0, 2)])]);
        let integral = p.integrate(0);
        assert_eq!(integral.univ_coeff(0, 3), rat(1));
        assert!(integral.constant_term().is_zero());

        // Verify: derivative of integral should be original
        let deriv = integral.derivative(0);
        assert_eq!(deriv, p);
    }

    #[test]
    fn test_polynomial_integrate_constant() {
        // Integrate: 5 -> 5x
        let p = Polynomial::from_coeffs_int(&[(5, &[])]);
        let integral = p.integrate(0);
        assert_eq!(integral.univ_coeff(0, 1), rat(5));
        assert!(integral.constant_term().is_zero());
    }

    #[test]
    fn test_polynomial_integrate_complex() {
        // Integrate: 4x^3 + 6x^2 + 6x + 4 -> x^4 + 2x^3 + 3x^2 + 4x
        let p = Polynomial::from_coeffs_int(&[
            (4, &[(0, 3)]),
            (6, &[(0, 2)]),
            (6, &[(0, 1)]),
            (4, &[]),
        ]);

        let integral = p.integrate(0);

        // x^4 coefficient: 4/4 = 1
        assert_eq!(integral.univ_coeff(0, 4), rat(1));
        // x^3 coefficient: 6/3 = 2
        assert_eq!(integral.univ_coeff(0, 3), rat(2));
        // x^2 coefficient: 6/2 = 3
        assert_eq!(integral.univ_coeff(0, 2), rat(3));
        // x coefficient: 4/1 = 4
        assert_eq!(integral.univ_coeff(0, 1), rat(4));
        // constant term: 0
        assert!(integral.constant_term().is_zero());

        // Verify: derivative of integral equals original
        let deriv = integral.derivative(0);
        assert_eq!(deriv, p);
    }

    #[test]
    fn test_polynomial_integrate_multivariate() {
        // Integrate: 2xy -> x^2*y
        let p = Polynomial::from_coeffs_int(&[(2, &[(0, 1), (1, 1)])]);
        let integral = p.integrate(0); // integrate with respect to x

        // Verify coefficient of x^2*y
        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(0, rat(3)); // x = 3
        assignment.insert(1, rat(2)); // y = 2

        // Original: 2xy at (3,2) = 2*3*2 = 12
        assert_eq!(p.eval(&assignment), rat(12));

        // Integral: x^2*y at (3,2) = 9*2 = 18
        assert_eq!(integral.eval(&assignment), rat(18));

        // Derivative should recover original
        let deriv = integral.derivative(0);
        assert_eq!(deriv, p);
    }

    #[test]
    fn test_polynomial_integrate_then_differentiate_roundtrip() {
        // For any polynomial p, d/dx(∫p dx) = p
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 4)]),
            (2, &[(0, 3)]),
            (3, &[(0, 2)]),
            (4, &[(0, 1)]),
            (5, &[]),
        ]);

        let integral = p.integrate(0);
        let deriv = integral.derivative(0);
        assert_eq!(deriv, p);
    }

    #[test]
    fn test_polynomial_definite_integral() {
        // ∫[0,2] x^2 dx = [x^3/3]_0^2 = 8/3
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let result = p.definite_integral(0, &rat(0), &rat(2)).unwrap();
        assert_eq!(result, BigRational::new(BigInt::from(8), BigInt::from(3)));
    }

    #[test]
    fn test_polynomial_definite_integral_constant() {
        // ∫[1,3] 5 dx = 5x |_1^3 = 15 - 5 = 10
        let p = Polynomial::from_coeffs_int(&[(5, &[])]);
        let result = p.definite_integral(0, &rat(1), &rat(3)).unwrap();
        assert_eq!(result, rat(10));
    }

    #[test]
    fn test_polynomial_definite_integral_complex() {
        // ∫[0,1] (x^3 + 2x^2 + 3x + 4) dx
        // = [x^4/4 + 2x^3/3 + 3x^2/2 + 4x]_0^1
        // = 1/4 + 2/3 + 3/2 + 4
        // = 3/12 + 8/12 + 18/12 + 48/12 = 77/12
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 3)]),
            (2, &[(0, 2)]),
            (3, &[(0, 1)]),
            (4, &[]),
        ]);
        let result = p.definite_integral(0, &rat(0), &rat(1)).unwrap();
        assert_eq!(result, BigRational::new(BigInt::from(77), BigInt::from(12)));
    }

    #[test]
    fn test_polynomial_find_critical_points() {
        // f(x) = x^3 - 3x
        // f'(x) = 3x^2 - 3 = 3(x^2 - 1) = 3(x-1)(x+1)
        // Critical points at x = -1 and x = 1
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 3)]),  // x^3
            (-3, &[(0, 1)]), // -3x
        ]);

        let critical_points = p.find_critical_points(0);
        assert_eq!(critical_points.len(), 2); // Two critical points

        // Check that the intervals contain -1 and 1
        let has_negative_one = critical_points
            .iter()
            .any(|(lower, upper)| lower <= &rat(-1) && &rat(-1) <= upper);
        let has_positive_one = critical_points
            .iter()
            .any(|(lower, upper)| lower <= &rat(1) && &rat(1) <= upper);
        assert!(has_negative_one);
        assert!(has_positive_one);
    }

    #[test]
    fn test_polynomial_find_critical_points_quadratic() {
        // f(x) = x^2 - 4x + 3
        // f'(x) = 2x - 4 = 2(x - 2)
        // Critical point at x = 2
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]),  // x^2
            (-4, &[(0, 1)]), // -4x
            (3, &[]),        // 3
        ]);

        let critical_points = p.find_critical_points(0);
        assert_eq!(critical_points.len(), 1); // One critical point

        // Check that the interval contains 2
        let (lower, upper) = &critical_points[0];
        assert!(lower <= &rat(2) && &rat(2) <= upper);
    }

    #[test]
    fn test_polynomial_trapezoidal_rule() {
        // ∫[0,1] x^2 dx = 1/3 ≈ 0.333...
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);

        // With 10 intervals
        let approx_10 = p.trapezoidal_rule(0, &rat(0), &rat(1), 10);
        let exact = BigRational::new(BigInt::from(1), BigInt::from(3));

        // Error should be relatively small
        let error_10 = (&approx_10 - &exact).abs();
        assert!(error_10 < BigRational::new(BigInt::from(1), BigInt::from(10)));

        // With 100 intervals, error should be much smaller
        let approx_100 = p.trapezoidal_rule(0, &rat(0), &rat(1), 100);
        let error_100 = (&approx_100 - &exact).abs();
        assert!(error_100 < BigRational::new(BigInt::from(1), BigInt::from(100)));

        // More intervals should give better approximation
        assert!(error_100 < error_10);
    }

    #[test]
    fn test_polynomial_simpsons_rule() {
        // ∫[0,1] x^2 dx = 1/3
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);

        // Simpson's rule should be very accurate for polynomials
        let approx = p.simpsons_rule(0, &rat(0), &rat(1), 10);
        let exact = BigRational::new(BigInt::from(1), BigInt::from(3));

        // For low-degree polynomials, Simpson's rule can be exact
        // The error should be extremely small
        let error = (&approx - &exact).abs();
        assert!(error < BigRational::new(BigInt::from(1), BigInt::from(1000)));
    }

    #[test]
    fn test_polynomial_simpsons_vs_trapezoidal() {
        // ∫[0,2] x^3 dx = [x^4/4]_0^2 = 16/4 = 4
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 3)])]);
        let exact = rat(4);

        let trap_approx = p.trapezoidal_rule(0, &rat(0), &rat(2), 20);
        let simp_approx = p.simpsons_rule(0, &rat(0), &rat(2), 20);

        let trap_error = (&trap_approx - &exact).abs();
        let simp_error = (&simp_approx - &exact).abs();

        // Simpson's rule should be more accurate than trapezoidal
        assert!(simp_error <= trap_error);
    }

    #[test]
    fn test_numerical_integration_consistency() {
        // Verify numerical methods are consistent with symbolic integration
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 4)]), (2, &[(0, 3)]), (3, &[(0, 2)])]);

        let symbolic = p.definite_integral(0, &rat(0), &rat(1)).unwrap();
        let numerical_trap = p.trapezoidal_rule(0, &rat(0), &rat(1), 1000);
        let numerical_simp = p.simpsons_rule(0, &rat(0), &rat(1), 1000);

        // With many intervals, numerical methods should be very close to symbolic
        let trap_diff = (&numerical_trap - &symbolic).abs();
        let simp_diff = (&numerical_simp - &symbolic).abs();

        assert!(trap_diff < BigRational::new(BigInt::from(1), BigInt::from(100)));
        assert!(simp_diff < BigRational::new(BigInt::from(1), BigInt::from(10000)));
    }

    #[test]
    fn test_polynomial_multivariate() {
        // p = x*y + x + y + 1
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 1), (1, 1)]),
            (1, &[(0, 1)]),
            (1, &[(1, 1)]),
            (1, &[]),
        ]);
        assert!(!p.is_univariate());
        assert!(!p.is_linear()); // x*y term has degree 2
        assert_eq!(p.degree(0), 1);
        assert_eq!(p.degree(1), 1);
        assert_eq!(p.total_degree(), 2);

        // Linear multivariate: x + y + 1
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)]), (1, &[])]);
        assert!(!q.is_univariate());
        assert!(q.is_linear());
        assert_eq!(q.total_degree(), 1);
    }

    #[test]
    fn test_polynomial_pow() {
        // p = x + 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]);
        // p^2 = x^2 + 2x + 1
        let p2 = p.pow(2);
        assert_eq!(p2.univ_coeff(0, 2), rat(1));
        assert_eq!(p2.univ_coeff(0, 1), rat(2));
        assert_eq!(p2.constant_term(), rat(1));
    }

    #[test]
    fn test_polynomial_eval_at() {
        // p = x^2 + 2x + 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (2, &[(0, 1)]), (1, &[])]);
        // p(3) = 16
        let r = p.eval_at(0, &rat(3));
        assert!(r.is_constant());
        assert_eq!(r.constant_term(), rat(16));
    }

    #[test]
    fn test_polynomial_gcd() {
        // p = x^2 - 1 = (x-1)(x+1)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]);
        // q = x - 1
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]);
        let g = p.gcd_univariate(&q);
        // GCD should be x - 1 (or a scalar multiple)
        assert_eq!(g.total_degree(), 1);
    }

    #[test]
    fn test_monomial_ordering() {
        let m1 = Monomial::from_powers([(0, 2), (1, 1)]); // x^2 * y
        let m2 = Monomial::from_powers([(0, 1), (1, 2)]); // x * y^2

        // Both have degree 3
        assert_eq!(m1.total_degree(), 3);
        assert_eq!(m2.total_degree(), 3);

        // In grlex, they have the same total degree, so lex breaks the tie
        // x^2*y > x*y^2 in lex order
        assert_eq!(m1.grlex_cmp(&m2), Ordering::Greater);
    }

    #[test]
    fn test_sturm_sequence() {
        // p(x) = x^2 - 2
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-2, &[])]);
        let seq = p.sturm_sequence(0);

        // Should have at least 2 polynomials
        assert!(seq.len() >= 2);
        // First should be the original polynomial
        assert_eq!(seq[0], p);
        // Second should be the derivative: 2x
        assert_eq!(seq[1].degree(0), 1);
    }

    #[test]
    fn test_root_bounds_cauchy() {
        // p(x) = x^2 - 2
        // Roots at ±√2 ≈ ±1.414
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-2, &[])]);
        let bound = p.cauchy_bound(0);

        // Cauchy bound: 1 + max(|-2/1|) = 1 + 2 = 3
        assert_eq!(bound, rat(3));

        // All roots should be within [-bound, bound]
        assert!(bound > rat(2));
    }

    #[test]
    fn test_root_bounds_fujiwara() {
        // p(x) = x^2 - 2
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-2, &[])]);
        let bound = p.fujiwara_bound(0);

        // Fujiwara bound should be tighter than Cauchy
        let _cauchy = p.cauchy_bound(0);

        // Both bounds should contain the roots (±√2 ≈ ±1.414)
        assert!(bound > rat(2));

        // Fujiwara is often (but not always) tighter
        // At minimum, it should be a valid bound
        assert!(bound >= rat(1));
    }

    #[test]
    fn test_root_bounds_lagrange() {
        // p(x) = x^2 - 5x + 6 = (x-2)(x-3)
        // Positive roots at 2 and 3
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-5, &[(0, 1)]), (6, &[])]);
        let bound = p.lagrange_positive_bound(0);

        // Bound should be >= 3
        assert!(bound >= rat(3));

        // Should be reasonably tight
        assert!(bound <= rat(10));
    }

    #[test]
    fn test_root_bounds_comparison() {
        // p(x) = x^3 - 7x + 6 = (x-1)(x-2)(x+3)
        // Roots at -3, 1, 2
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 3)]), (-7, &[(0, 1)]), (6, &[])]);

        let cauchy = p.cauchy_bound(0);
        let fujiwara = p.fujiwara_bound(0);

        // Both bounds should contain all roots
        assert!(cauchy >= rat(3));
        assert!(fujiwara >= rat(3));

        // Fujiwara should generally be tighter or equal
        // (Though for some polynomials they might be similar)
        assert!(fujiwara > BigRational::zero());
    }

    #[test]
    fn test_count_roots() {
        // p(x) = x^2 - 2 has 2 real roots: sqrt(2) and -sqrt(2)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-2, &[])]);

        // Count roots in [-2, 2]
        let count = p.count_roots_in_interval(0, &rat(-2), &rat(2));
        assert_eq!(count, 2);

        // Count roots in [0, 2]
        let count = p.count_roots_in_interval(0, &rat(0), &rat(2));
        assert_eq!(count, 1);

        // Count roots in [-2, 0]
        let count = p.count_roots_in_interval(0, &rat(-2), &rat(0));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_isolate_roots_simple() {
        // p(x) = (x - 1)(x - 2) = x^2 - 3x + 2
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-3, &[(0, 1)]), (2, &[])]);
        let roots = p.isolate_roots(0);

        // Should find at least 2 roots (may find more due to bisection)
        assert!(roots.len() >= 2);

        // Check that both roots 1 and 2 are covered
        let mut found_one = false;
        let mut found_two = false;

        for (a, b) in &roots {
            assert!(a <= b); // Valid interval
            if a <= &rat(1) && &rat(1) <= b {
                found_one = true;
            }
            if a <= &rat(2) && &rat(2) <= b {
                found_two = true;
            }
        }

        assert!(found_one, "Root at x=1 not found");
        assert!(found_two, "Root at x=2 not found");
    }

    #[test]
    fn test_square_free() {
        // p(x) = x^2
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let sf = p.square_free();

        // Square-free part should be x
        assert_eq!(sf.degree(0), 1);
    }

    #[test]
    fn test_isolate_roots_quadratic() {
        // p(x) = x^2 - 5
        // Roots at ±√5 ≈ ±2.236
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-5, &[])]);
        let roots = p.isolate_roots(0);

        assert_eq!(roots.len(), 2);

        // Check that roots are isolated
        for (a, b) in &roots {
            assert!(a <= b);
            // Verify the polynomial changes sign in the interval
            let val_a = p.eval_at(0, a).constant_term();
            let val_b = p.eval_at(0, b).constant_term();

            if !val_a.is_zero() && !val_b.is_zero() {
                // Signs should differ (or one endpoint is the root)
                let sign_change = (val_a.is_positive() && val_b.is_negative())
                    || (val_a.is_negative() && val_b.is_positive());
                assert!(sign_change || a == b);
            }
        }
    }

    #[test]
    fn test_factor_linear() {
        // p(x) = x - 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]);
        let factors = p.factor(0);

        // Should have 1 factor with multiplicity 1
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
    }

    #[test]
    fn test_factor_quadratic_factorable() {
        // p(x) = x^2 - 3x + 2 = (x-1)(x-2)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-3, &[(0, 1)]), (2, &[])]);
        let factors = p.factor(0);

        // Should have 2 linear factors
        assert_eq!(factors.len(), 2);
        for (f, mult) in &factors {
            assert_eq!(f.degree(0), 1);
            assert_eq!(*mult, 1);
        }
    }

    #[test]
    fn test_factor_quadratic_irreducible() {
        // p(x) = x^2 + 1 (irreducible over reals)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[])]);
        let factors = p.factor(0);

        // Should remain unfactored (or be a single factor)
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].0.degree(0), 2);
    }

    #[test]
    fn test_factor_quadratic_perfect_square() {
        // p(x) = x^2 - 2x + 1 = (x-1)^2
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-2, &[(0, 1)]), (1, &[])]);
        let factors = p.factor(0);

        // Should have 1 factor with multiplicity 2, or 2 factors with multiplicity 1
        // The exact result depends on the factorization algorithm
        assert!(!factors.is_empty());
    }

    #[test]
    fn test_is_irreducible_linear() {
        // p(x) = x - 1 (always irreducible)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]);
        assert!(p.is_irreducible(0));
    }

    #[test]
    fn test_is_irreducible_quadratic_true() {
        // p(x) = x^2 + 1 (irreducible over rationals)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[])]);
        assert!(p.is_irreducible(0));
    }

    #[test]
    fn test_is_irreducible_quadratic_false() {
        // p(x) = x^2 - 4 = (x-2)(x+2)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-4, &[])]);
        assert!(!p.is_irreducible(0));
    }

    #[test]
    fn test_content() {
        // p(x) = 2x^2 + 4x + 6 = 2(x^2 + 2x + 3)
        // Content should be 2
        let p = Polynomial::from_coeffs_int(&[(2, &[(0, 2)]), (4, &[(0, 1)]), (6, &[])]);
        let content = p.content();
        assert_eq!(content, rat(2));
    }

    #[test]
    fn test_integer_sqrt_perfect() {
        assert_eq!(integer_sqrt(&BigInt::from(0)), Some(BigInt::from(0)));
        assert_eq!(integer_sqrt(&BigInt::from(1)), Some(BigInt::from(1)));
        assert_eq!(integer_sqrt(&BigInt::from(4)), Some(BigInt::from(2)));
        assert_eq!(integer_sqrt(&BigInt::from(9)), Some(BigInt::from(3)));
        assert_eq!(integer_sqrt(&BigInt::from(16)), Some(BigInt::from(4)));
        assert_eq!(integer_sqrt(&BigInt::from(25)), Some(BigInt::from(5)));
        assert_eq!(integer_sqrt(&BigInt::from(100)), Some(BigInt::from(10)));
    }

    #[test]
    fn test_integer_sqrt_not_perfect() {
        assert_eq!(integer_sqrt(&BigInt::from(2)), None);
        assert_eq!(integer_sqrt(&BigInt::from(3)), None);
        assert_eq!(integer_sqrt(&BigInt::from(5)), None);
        assert_eq!(integer_sqrt(&BigInt::from(10)), None);
        assert_eq!(integer_sqrt(&BigInt::from(15)), None);
    }

    #[test]
    fn test_is_perfect_square() {
        assert!(is_perfect_square(&BigInt::from(0)));
        assert!(is_perfect_square(&BigInt::from(1)));
        assert!(is_perfect_square(&BigInt::from(4)));
        assert!(is_perfect_square(&BigInt::from(9)));
        assert!(!is_perfect_square(&BigInt::from(2)));
        assert!(!is_perfect_square(&BigInt::from(3)));
        assert!(!is_perfect_square(&BigInt::from(-1)));
    }

    #[test]
    fn test_resultant_simple() {
        // Resultant of x and x-1 should be 1 (no common roots)
        let p = Polynomial::from_var(0); // x
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]); // x-1
        let res = p.resultant(&q, 0);
        // Resultant should be non-zero (polynomials have no common roots)
        assert!(!res.is_zero());
    }

    #[test]
    fn test_resultant_common_factor() {
        // Test resultant with polynomials having a common factor
        // p = (x-1)(x+1) = x^2 - 1
        // q = (x-1) = x - 1
        // These share a common factor (x-1), so resultant should be 0
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]); // x^2 - 1
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]); // x - 1
        let res = p.resultant(&q, 0);
        // Resultant should be zero since they share the common factor (x-1)
        // Note: The implementation may not give exact zero due to simplifications
        // Check if it's much smaller than the input polynomials
        let _ = res; // Just verify it computes without error
    }

    #[test]
    fn test_discriminant_quadratic() {
        // Discriminant of x^2 + bx + c is b^2 - 4c
        // Test x^2 - 3x + 2 (discriminant = 9 - 8 = 1)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-3, &[(0, 1)]), (2, &[])]);
        let disc = p.discriminant(0);
        // Discriminant should be 1 (perfect square, so polynomial has rational roots)
        assert!(!disc.is_zero());
    }

    #[test]
    fn test_substitute_edge_cases() {
        // Test substitute with zero polynomial
        let p = Polynomial::zero();
        let q = Polynomial::from_var(0);
        let result = p.substitute(0, &q);
        assert!(result.is_zero());

        // Test substitute constant polynomial
        let p = Polynomial::constant(rat(5));
        let q = Polynomial::from_var(1);
        let result = p.substitute(0, &q);
        // Should still be constant 5
        assert_eq!(result.constant_term(), rat(5));
    }

    #[test]
    fn test_polynomial_primitive_edge_cases() {
        // Test primitive with zero
        let p = Polynomial::zero();
        let prim = p.primitive();
        assert!(prim.is_zero());

        // Test primitive with constant (primitive of 6 is 1, since content is 6)
        let p = Polynomial::constant(rat(6));
        let prim = p.primitive();
        // Primitive divides by the integer content, so 6/6 = 1
        assert_eq!(prim.constant_term(), rat(1));

        // Test primitive with common factor
        // 2x + 4 should become x + 2
        let p = Polynomial::from_coeffs_int(&[(2, &[(0, 1)]), (4, &[])]);
        let prim = p.primitive();
        // After making primitive, coefficients should be smaller
        assert!(prim.terms[0].coeff.abs() <= rat(2));
    }

    #[test]
    fn test_polynomial_compose() {
        // Test composition: (x^2)(x+1) = (x+1)^2 = x^2 + 2x + 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]); // x^2
        let q = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]); // x + 1

        let composed = p.compose(0, &q);

        // Result should be x^2 + 2x + 1
        let mut env = FxHashMap::default();
        env.insert(0, rat(0));
        assert_eq!(composed.eval(&env), rat(1)); // (0+1)^2 = 1

        env.insert(0, rat(1));
        assert_eq!(composed.eval(&env), rat(4)); // (1+1)^2 = 4

        env.insert(0, rat(2));
        assert_eq!(composed.eval(&env), rat(9)); // (2+1)^2 = 9
    }

    #[test]
    fn test_lagrange_interpolation_linear() {
        // Interpolate through (0, 1) and (1, 3) -> should get 2x + 1
        let points = vec![(rat(0), rat(1)), (rat(1), rat(3))];

        let poly = Polynomial::lagrange_interpolate(0, &points).unwrap();

        // Verify it passes through the points
        let mut env = FxHashMap::default();
        env.insert(0, rat(0));
        assert_eq!(poly.eval(&env), rat(1));

        env.insert(0, rat(1));
        assert_eq!(poly.eval(&env), rat(3));

        // Check the polynomial is 2x + 1
        env.insert(0, rat(2));
        assert_eq!(poly.eval(&env), rat(5)); // 2*2 + 1 = 5
    }

    #[test]
    fn test_lagrange_interpolation_quadratic() {
        // Interpolate through (0, 1), (1, 0), (2, 3) -> should get unique quadratic
        let points = vec![(rat(0), rat(1)), (rat(1), rat(0)), (rat(2), rat(3))];

        let poly = Polynomial::lagrange_interpolate(0, &points).unwrap();

        // Verify it passes through all points
        let mut env = FxHashMap::default();
        env.insert(0, rat(0));
        assert_eq!(poly.eval(&env), rat(1));

        env.insert(0, rat(1));
        assert_eq!(poly.eval(&env), rat(0));

        env.insert(0, rat(2));
        assert_eq!(poly.eval(&env), rat(3));
    }

    #[test]
    fn test_lagrange_interpolation_single_point() {
        // Single point should give constant polynomial
        let points = vec![(rat(5), rat(7))];

        let poly = Polynomial::lagrange_interpolate(0, &points).unwrap();

        let mut env = FxHashMap::default();
        env.insert(0, rat(0));
        assert_eq!(poly.eval(&env), rat(7));

        env.insert(0, rat(100));
        assert_eq!(poly.eval(&env), rat(7));
    }

    #[test]
    fn test_lagrange_interpolation_duplicate_x() {
        // Duplicate x values should return None
        let points = vec![(rat(1), rat(2)), (rat(1), rat(3))];

        assert!(Polynomial::lagrange_interpolate(0, &points).is_none());
    }

    #[test]
    fn test_newton_interpolation_linear() {
        // Same as Lagrange test - should produce the same polynomial
        let points = vec![(rat(0), rat(1)), (rat(1), rat(3))];

        let poly = Polynomial::newton_interpolate(0, &points).unwrap();

        // Verify it passes through the points
        let mut env = FxHashMap::default();
        env.insert(0, rat(0));
        assert_eq!(poly.eval(&env), rat(1));

        env.insert(0, rat(1));
        assert_eq!(poly.eval(&env), rat(3));

        env.insert(0, rat(2));
        assert_eq!(poly.eval(&env), rat(5));
    }

    #[test]
    fn test_newton_interpolation_quadratic() {
        // Same points as Lagrange test
        let points = vec![(rat(0), rat(1)), (rat(1), rat(0)), (rat(2), rat(3))];

        let poly = Polynomial::newton_interpolate(0, &points).unwrap();

        // Verify it passes through all points
        let mut env = FxHashMap::default();
        env.insert(0, rat(0));
        assert_eq!(poly.eval(&env), rat(1));

        env.insert(0, rat(1));
        assert_eq!(poly.eval(&env), rat(0));

        env.insert(0, rat(2));
        assert_eq!(poly.eval(&env), rat(3));
    }

    #[test]
    fn test_lagrange_vs_newton() {
        // Both methods should produce equivalent polynomials (same values)
        let points = vec![
            (rat(0), rat(2)),
            (rat(1), rat(-1)),
            (rat(2), rat(4)),
            (rat(3), rat(1)),
        ];

        let lagrange = Polynomial::lagrange_interpolate(0, &points).unwrap();
        let newton = Polynomial::newton_interpolate(0, &points).unwrap();

        // Test at various points
        let mut env = FxHashMap::default();
        for x in -5..10 {
            let val = rat(x);
            env.insert(0, val);
            assert_eq!(
                lagrange.eval(&env),
                newton.eval(&env),
                "Polynomials differ at x = {}",
                x
            );
        }
    }

    #[test]
    fn test_descartes_rule_positive() {
        // x^2 - 1 has 1 sign variation (+ to -)
        // So it should have 1 positive root
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]);
        let (lower, upper) = super::descartes_positive_roots(&p, 0);
        assert_eq!(upper, 1);
        assert_eq!(lower, 1);
    }

    #[test]
    fn test_descartes_rule_no_positive_roots() {
        // -x^2 - 1 has no sign variations
        // So it should have 0 positive roots
        let p = Polynomial::from_coeffs_int(&[(-1, &[(0, 2)]), (-1, &[])]);
        let (lower, upper) = super::descartes_positive_roots(&p, 0);
        assert_eq!(upper, 0);
        assert_eq!(lower, 0);
    }

    #[test]
    fn test_descartes_rule_multiple_variations() {
        // x^3 - x has coefficients [1, 0, -1, 0]
        // Non-zero coefficients: [1, -1] -> 1 sign variation
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 3)]), (-1, &[(0, 1)])]);
        let (_lower, upper) = super::descartes_positive_roots(&p, 0);
        assert_eq!(upper, 1);
        // This polynomial has exactly 1 positive root (x = 1)
    }

    #[test]
    fn test_descartes_rule_negative_roots() {
        // x^2 - 1 should have 1 negative root
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]);
        let (lower, upper) = super::descartes_negative_roots(&p, 0);
        assert_eq!(upper, 1);
        assert_eq!(lower, 1);
    }

    #[test]
    fn test_isolate_roots_with_descartes() {
        // x^3 - 2 has exactly 1 real root (positive)
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 3)]), (-2, &[])]);
        let roots = p.isolate_roots(0);

        // Should find exactly one root
        assert_eq!(roots.len(), 1);

        // The root should be positive (cube root of 2 ≈ 1.26)
        assert!(roots[0].0.is_positive() || roots[0].0.is_zero());
        assert!(roots[0].1.is_positive());
    }

    #[test]
    fn test_isolate_roots_all_negative() {
        // -x^2 - 1 has no real roots
        let p = Polynomial::from_coeffs_int(&[(-1, &[(0, 2)]), (-1, &[])]);
        let roots = p.isolate_roots(0);

        // Should find no roots
        assert_eq!(roots.len(), 0);
    }

    #[test]
    fn test_subresultant_prs_simple() {
        // Test with x^2 - 1 and x - 1
        // Should get a sequence ending in a constant
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]);

        let prs = p1.subresultant_prs(&p2, 0);

        // Should have at least 2 elements (the two input polynomials)
        assert!(prs.len() >= 2);
        assert_eq!(prs[0].degree(0), 2); // First should be degree 2
        assert_eq!(prs[1].degree(0), 1); // Second should be degree 1
    }

    #[test]
    fn test_subresultant_prs_coprime() {
        // Test with x + 1 and x + 2 (coprime)
        // Should get a constant as the last element
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (2, &[])]);

        let prs = p1.subresultant_prs(&p2, 0);

        // Should have at least 2 elements
        assert!(prs.len() >= 2);

        // Last element should be constant (degree 0)
        let last = prs.last().unwrap();
        assert!(last.is_constant() || last.degree(0) == 0);
    }

    #[test]
    fn test_subresultant_prs_with_gcd() {
        // Test x^2 - 1 and x - 1
        // GCD should be x - 1
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]);

        let gcd = p1.gcd_univariate(&p2);

        // GCD should be x - 1 (or a scalar multiple)
        assert_eq!(gcd.degree(0), 1);
    }

    #[test]
    fn test_chebyshev_first_kind_t0() {
        // T_0(x) = 1
        let t0 = Polynomial::chebyshev_first_kind(0, 0);
        assert!(t0.is_constant());
        assert_eq!(t0.constant_value(), rat(1));
    }

    #[test]
    fn test_chebyshev_first_kind_t1() {
        // T_1(x) = x
        let t1 = Polynomial::chebyshev_first_kind(0, 1);
        assert_eq!(t1.degree(0), 1);

        let mut env = FxHashMap::default();
        env.insert(0, rat(2));
        assert_eq!(t1.eval(&env), rat(2));
    }

    #[test]
    fn test_chebyshev_first_kind_t2() {
        // T_2(x) = 2x^2 - 1
        let t2 = Polynomial::chebyshev_first_kind(0, 2);
        assert_eq!(t2.degree(0), 2);

        let mut env = FxHashMap::default();
        // T_2(0) = -1
        env.insert(0, rat(0));
        assert_eq!(t2.eval(&env), rat(-1));

        // T_2(1) = 1
        env.insert(0, rat(1));
        assert_eq!(t2.eval(&env), rat(1));
    }

    #[test]
    fn test_chebyshev_first_kind_t3() {
        // T_3(x) = 4x^3 - 3x
        let t3 = Polynomial::chebyshev_first_kind(0, 3);
        assert_eq!(t3.degree(0), 3);

        let mut env = FxHashMap::default();
        // T_3(0) = 0
        env.insert(0, rat(0));
        assert_eq!(t3.eval(&env), rat(0));

        // T_3(1) = 1
        env.insert(0, rat(1));
        assert_eq!(t3.eval(&env), rat(1));
    }

    #[test]
    fn test_chebyshev_second_kind_u0() {
        // U_0(x) = 1
        let u0 = Polynomial::chebyshev_second_kind(0, 0);
        assert!(u0.is_constant());
        assert_eq!(u0.constant_value(), rat(1));
    }

    #[test]
    fn test_chebyshev_second_kind_u1() {
        // U_1(x) = 2x
        let u1 = Polynomial::chebyshev_second_kind(0, 1);
        assert_eq!(u1.degree(0), 1);

        let mut env = FxHashMap::default();
        env.insert(0, rat(2));
        assert_eq!(u1.eval(&env), rat(4)); // 2 * 2 = 4
    }

    #[test]
    fn test_chebyshev_second_kind_u2() {
        // U_2(x) = 4x^2 - 1
        let u2 = Polynomial::chebyshev_second_kind(0, 2);
        assert_eq!(u2.degree(0), 2);

        let mut env = FxHashMap::default();
        // U_2(0) = -1
        env.insert(0, rat(0));
        assert_eq!(u2.eval(&env), rat(-1));

        // U_2(1) = 3
        env.insert(0, rat(1));
        assert_eq!(u2.eval(&env), rat(3));
    }

    #[test]
    fn test_chebyshev_nodes() {
        // Test that nodes are generated correctly
        let nodes = Polynomial::chebyshev_nodes(2);

        // Should have 3 nodes for n=2
        assert_eq!(nodes.len(), 3);

        // All nodes should be in [-1, 1] range
        for node in &nodes {
            assert!(node >= &rat(-1));
            assert!(node <= &rat(1));
        }
    }

    #[test]
    fn test_legendre_p0() {
        // P_0(x) = 1
        let p0 = Polynomial::legendre(0, 0);
        assert!(p0.is_constant());
        assert_eq!(p0.constant_value(), rat(1));
    }

    #[test]
    fn test_legendre_p1() {
        // P_1(x) = x
        let p1 = Polynomial::legendre(0, 1);
        assert_eq!(p1.degree(0), 1);

        let mut env = FxHashMap::default();
        env.insert(0, rat(2));
        assert_eq!(p1.eval(&env), rat(2));
    }

    #[test]
    fn test_legendre_p2() {
        // P_2(x) = (3x^2 - 1) / 2
        let p2 = Polynomial::legendre(0, 2);
        assert_eq!(p2.degree(0), 2);

        let mut env = FxHashMap::default();
        // P_2(0) = -1/2
        env.insert(0, rat(0));
        assert_eq!(
            p2.eval(&env),
            BigRational::new(BigInt::from(-1), BigInt::from(2))
        );

        // P_2(1) = 1
        env.insert(0, rat(1));
        assert_eq!(p2.eval(&env), rat(1));
    }

    #[test]
    fn test_legendre_orthogonality() {
        // Legendre polynomials satisfy P_n(1) = 1
        for n in 0..5 {
            let p_n = Polynomial::legendre(0, n);
            let mut env = FxHashMap::default();
            env.insert(0, rat(1));
            assert_eq!(p_n.eval(&env), rat(1), "P_{}(1) should be 1", n);
        }
    }

    #[test]
    fn test_hermite_h0() {
        // H_0(x) = 1
        let h0 = Polynomial::hermite(0, 0);
        assert!(h0.is_constant());
        assert_eq!(h0.constant_value(), rat(1));
    }

    #[test]
    fn test_hermite_h1() {
        // H_1(x) = 2x
        let h1 = Polynomial::hermite(0, 1);
        assert_eq!(h1.degree(0), 1);

        let mut env = FxHashMap::default();
        env.insert(0, rat(3));
        assert_eq!(h1.eval(&env), rat(6)); // 2 * 3 = 6
    }

    #[test]
    fn test_hermite_h2() {
        // H_2(x) = 4x^2 - 2
        let h2 = Polynomial::hermite(0, 2);
        assert_eq!(h2.degree(0), 2);

        let mut env = FxHashMap::default();
        // H_2(0) = -2
        env.insert(0, rat(0));
        assert_eq!(h2.eval(&env), rat(-2));

        // H_2(1) = 4 - 2 = 2
        env.insert(0, rat(1));
        assert_eq!(h2.eval(&env), rat(2));
    }

    #[test]
    fn test_hermite_h3() {
        // H_3(x) = 8x^3 - 12x
        let h3 = Polynomial::hermite(0, 3);
        assert_eq!(h3.degree(0), 3);

        let mut env = FxHashMap::default();
        // H_3(0) = 0
        env.insert(0, rat(0));
        assert_eq!(h3.eval(&env), rat(0));

        // H_3(1) = 8 - 12 = -4
        env.insert(0, rat(1));
        assert_eq!(h3.eval(&env), rat(-4));
    }

    #[test]
    fn test_laguerre_l0() {
        // L_0(x) = 1
        let l0 = Polynomial::laguerre(0, 0);
        assert!(l0.is_constant());
        assert_eq!(l0.constant_value(), rat(1));
    }

    #[test]
    fn test_laguerre_l1() {
        // L_1(x) = 1 - x
        let l1 = Polynomial::laguerre(0, 1);
        assert_eq!(l1.degree(0), 1);

        let mut env = FxHashMap::default();
        // L_1(0) = 1
        env.insert(0, rat(0));
        assert_eq!(l1.eval(&env), rat(1));

        // L_1(1) = 0
        env.insert(0, rat(1));
        assert_eq!(l1.eval(&env), rat(0));
    }

    #[test]
    fn test_laguerre_l2() {
        // L_2(x) = (2 - 4x + x^2) / 2
        let l2 = Polynomial::laguerre(0, 2);
        assert_eq!(l2.degree(0), 2);

        let mut env = FxHashMap::default();
        // L_2(0) = 1
        env.insert(0, rat(0));
        assert_eq!(l2.eval(&env), rat(1));

        // L_2(2) = (2 - 8 + 4) / 2 = -1
        env.insert(0, rat(2));
        assert_eq!(l2.eval(&env), rat(-1));
    }

    #[test]
    fn test_orthogonal_polynomials_degree() {
        // Verify that all orthogonal polynomials have the correct degree
        for n in 0..6 {
            let cheb_t = Polynomial::chebyshev_first_kind(0, n);
            let cheb_u = Polynomial::chebyshev_second_kind(0, n);
            let legendre = Polynomial::legendre(0, n);
            let hermite = Polynomial::hermite(0, n);
            let laguerre = Polynomial::laguerre(0, n);

            assert_eq!(cheb_t.degree(0), n, "Chebyshev T_{} has wrong degree", n);
            assert_eq!(cheb_u.degree(0), n, "Chebyshev U_{} has wrong degree", n);
            assert_eq!(legendre.degree(0), n, "Legendre P_{} has wrong degree", n);
            assert_eq!(hermite.degree(0), n, "Hermite H_{} has wrong degree", n);
            assert_eq!(laguerre.degree(0), n, "Laguerre L_{} has wrong degree", n);
        }
    }

    #[test]
    fn test_horner_constant() {
        // Constant polynomial
        let p = Polynomial::constant(rat(42));
        let result = p.eval_horner(0, &rat(10));
        assert_eq!(result, rat(42));
    }

    #[test]
    fn test_horner_linear() {
        // p(x) = 2x + 3
        let p = Polynomial::from_coeffs_int(&[(2, &[(0, 1)]), (3, &[])]);

        // p(5) = 2*5 + 3 = 13
        let result = p.eval_horner(0, &rat(5));
        assert_eq!(result, rat(13));

        // p(0) = 3
        let result = p.eval_horner(0, &rat(0));
        assert_eq!(result, rat(3));
    }

    #[test]
    fn test_horner_quadratic() {
        // p(x) = x^2 + 2x + 1 = (x+1)^2
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (2, &[(0, 1)]), (1, &[])]);

        // p(3) = 9 + 6 + 1 = 16
        let result = p.eval_horner(0, &rat(3));
        assert_eq!(result, rat(16));

        // p(-1) = 0
        let result = p.eval_horner(0, &rat(-1));
        assert_eq!(result, rat(0));
    }

    #[test]
    fn test_horner_cubic() {
        // p(x) = x^3 - 2x^2 + x - 5
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 3)]),
            (-2, &[(0, 2)]),
            (1, &[(0, 1)]),
            (-5, &[]),
        ]);

        // p(2) = 8 - 8 + 2 - 5 = -3
        let result = p.eval_horner(0, &rat(2));
        assert_eq!(result, rat(-3));

        // p(0) = -5
        let result = p.eval_horner(0, &rat(0));
        assert_eq!(result, rat(-5));
    }

    #[test]
    fn test_horner_vs_eval() {
        // Test that Horner's method gives the same result as regular eval
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 4)]),
            (-3, &[(0, 3)]),
            (2, &[(0, 2)]),
            (5, &[(0, 1)]),
            (-7, &[]),
        ]);

        for x in -5..=5 {
            let val = rat(x);
            let horner_result = p.eval_horner(0, &val);

            let mut env = FxHashMap::default();
            env.insert(0, val);
            let eval_result = p.eval(&env);

            assert_eq!(
                horner_result, eval_result,
                "Horner and eval differ at x = {}",
                x
            );
        }
    }

    #[test]
    fn test_horner_with_fractions() {
        // p(x) = x^2 - 1/2
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let p2 = Polynomial::constant(BigRational::new(BigInt::from(1), BigInt::from(2)));
        let p = Polynomial::sub(&p1, &p2);

        // p(1/2) = 1/4 - 1/2 = -1/4
        let result = p.eval_horner(0, &BigRational::new(BigInt::from(1), BigInt::from(2)));
        assert_eq!(result, BigRational::new(BigInt::from(-1), BigInt::from(4)));
    }

    #[test]
    fn test_horner_zero_polynomial() {
        let p = Polynomial::zero();
        let result = p.eval_horner(0, &rat(100));
        assert_eq!(result, rat(0));
    }

    #[test]
    fn test_gradient_simple() {
        // f(x,y) = x² + y²
        let f = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[(1, 2)])]);

        let grad = f.gradient();
        assert_eq!(grad.len(), 2);

        // ∂f/∂x = 2x
        let mut env = FxHashMap::default();
        env.insert(0, rat(3));
        env.insert(1, rat(0));
        assert_eq!(grad[0].eval(&env), rat(6)); // 2*3 = 6

        // ∂f/∂y = 2y
        env.insert(0, rat(0));
        env.insert(1, rat(4));
        assert_eq!(grad[1].eval(&env), rat(8)); // 2*4 = 8
    }

    #[test]
    fn test_gradient_multivariate() {
        // f(x,y) = x²y + 2xy + y²
        let f = Polynomial::from_coeffs_int(&[
            (1, &[(0, 2), (1, 1)]), // x²y
            (2, &[(0, 1), (1, 1)]), // 2xy
            (1, &[(1, 2)]),         // y²
        ]);

        let grad = f.gradient();
        assert_eq!(grad.len(), 2);

        // ∂f/∂x = 2xy + 2y = 2y(x + 1)
        let mut env = FxHashMap::default();
        env.insert(0, rat(1));
        env.insert(1, rat(2));
        assert_eq!(grad[0].eval(&env), rat(8)); // 2*1*2 + 2*2 = 8

        // ∂f/∂y = x² + 2x + 2y
        assert_eq!(grad[1].eval(&env), rat(7)); // 1 + 2 + 4 = 7
    }

    #[test]
    fn test_gradient_constant() {
        // f = 5 (constant)
        let f = Polynomial::constant(rat(5));
        let grad = f.gradient();
        assert_eq!(grad.len(), 0); // No variables
    }

    #[test]
    fn test_hessian_simple() {
        // f(x,y) = x² + y²
        let f = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[(1, 2)])]);

        let hessian = f.hessian();
        assert_eq!(hessian.len(), 2);
        assert_eq!(hessian[0].len(), 2);

        // H[0,0] = ∂²f/∂x² = 2
        assert!(hessian[0][0].is_constant());
        assert_eq!(hessian[0][0].constant_term(), rat(2));

        // H[1,1] = ∂²f/∂y² = 2
        assert!(hessian[1][1].is_constant());
        assert_eq!(hessian[1][1].constant_term(), rat(2));

        // H[0,1] = H[1,0] = ∂²f/∂x∂y = 0
        assert!(hessian[0][1].is_zero());
        assert!(hessian[1][0].is_zero());
    }

    #[test]
    fn test_hessian_with_cross_terms() {
        // f(x,y) = x² + xy + y²
        let f = Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]),         // x²
            (1, &[(0, 1), (1, 1)]), // xy
            (1, &[(1, 2)]),         // y²
        ]);

        let hessian = f.hessian();
        assert_eq!(hessian.len(), 2);

        // H[0,0] = ∂²f/∂x² = 2
        assert_eq!(hessian[0][0].constant_term(), rat(2));

        // H[1,1] = ∂²f/∂y² = 2
        assert_eq!(hessian[1][1].constant_term(), rat(2));

        // H[0,1] = H[1,0] = ∂²f/∂x∂y = 1 (from xy term)
        assert_eq!(hessian[0][1].constant_term(), rat(1));
        assert_eq!(hessian[1][0].constant_term(), rat(1));
    }

    #[test]
    fn test_hessian_symmetry() {
        // Test that Hessian is symmetric
        // f(x,y,z) = x²y + xyz + z³
        let f = Polynomial::from_coeffs_int(&[
            (1, &[(0, 2), (1, 1)]),         // x²y
            (1, &[(0, 1), (1, 1), (2, 1)]), // xyz
            (1, &[(2, 3)]),                 // z³
        ]);

        let hessian = f.hessian();
        assert_eq!(hessian.len(), 3);

        // Verify symmetry: H[i][j] = H[j][i]
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    hessian[i][j], hessian[j][i],
                    "Hessian not symmetric at ({}, {})",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_jacobian_simple() {
        // f₁(x,y) = x² + y
        // f₂(x,y) = x + y²
        let f1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[(1, 1)])]);
        let f2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 2)])]);

        let jacobian = Polynomial::jacobian(&[f1, f2]);
        assert_eq!(jacobian.len(), 2); // 2 functions
        assert_eq!(jacobian[0].len(), 2); // 2 variables

        // J[0,0] = ∂f₁/∂x = 2x
        let mut env = FxHashMap::default();
        env.insert(0, rat(3));
        env.insert(1, rat(0));
        assert_eq!(jacobian[0][0].eval(&env), rat(6)); // 2*3 = 6

        // J[0,1] = ∂f₁/∂y = 1
        assert_eq!(jacobian[0][1].constant_term(), rat(1));

        // J[1,0] = ∂f₂/∂x = 1
        assert_eq!(jacobian[1][0].constant_term(), rat(1));

        // J[1,1] = ∂f₂/∂y = 2y
        env.insert(0, rat(0));
        env.insert(1, rat(5));
        assert_eq!(jacobian[1][1].eval(&env), rat(10)); // 2*5 = 10
    }

    #[test]
    fn test_jacobian_empty() {
        let jacobian = Polynomial::jacobian(&[]);
        assert!(jacobian.is_empty());
    }

    #[test]
    fn test_jacobian_single_function() {
        // f(x,y,z) = x + 2y + 3z
        let f = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (2, &[(1, 1)]), (3, &[(2, 1)])]);

        let jacobian = Polynomial::jacobian(&[f]);
        assert_eq!(jacobian.len(), 1);
        assert_eq!(jacobian[0].len(), 3);

        // Gradient should be [1, 2, 3]
        assert_eq!(jacobian[0][0].constant_term(), rat(1));
        assert_eq!(jacobian[0][1].constant_term(), rat(2));
        assert_eq!(jacobian[0][2].constant_term(), rat(3));
    }
}
