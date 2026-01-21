//! Type definitions for polynomial arithmetic.
//!
//! This module contains the core types used in polynomial representation:
//! - VarPower: Variable with exponent
//! - Monomial: Product of variables with exponents
//! - Term: Coefficient multiplied by a monomial
//! - MonomialOrder: Ordering for polynomial canonicalization

use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

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
    pub(crate) vars: SmallVec<[VarPower; 4]>,
    /// Cached total degree.
    pub(crate) total_degree: u32,
    /// Cached hash value.
    pub(crate) hash: u64,
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
