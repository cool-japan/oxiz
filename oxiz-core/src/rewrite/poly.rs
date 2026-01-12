//! Polynomial Representation and Normalization
//!
//! This module provides polynomial representation for arithmetic expressions:
//! - Polynomial normalization (canonical form)
//! - GCD computation for coefficient simplification
//! - Polynomial arithmetic (add, sub, mul)
//! - Monomial ordering for AC normalization

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Type alias for rational numbers
pub type Rational = BigRational;

/// Create a rational from an integer
fn rat(n: i64) -> Rational {
    Rational::from_integer(BigInt::from(n))
}

/// Compute power of a rational number
fn pow_rational(base: &Rational, exp: u32) -> Rational {
    let mut result = Rational::one();
    for _ in 0..exp {
        result *= base.clone();
    }
    result
}

/// A monomial: coefficient * product of variables with powers
#[derive(Clone)]
pub struct Monomial {
    /// Coefficient (rational number)
    pub coeff: Rational,
    /// Variables with their powers: var_id -> power
    pub vars: FxHashMap<u32, u32>,
}

impl fmt::Debug for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}*", self.coeff)?;
        let mut vars: Vec<_> = self.vars.iter().collect();
        vars.sort_by_key(|(v, _)| *v);
        for (i, (var, pow)) in vars.iter().enumerate() {
            if i > 0 {
                write!(f, "*")?;
            }
            if **pow == 1 {
                write!(f, "x{}", var)?;
            } else {
                write!(f, "x{}^{}", var, pow)?;
            }
        }
        Ok(())
    }
}

impl PartialEq for Monomial {
    fn eq(&self, other: &Self) -> bool {
        // Compare variables only (for polynomial combination)
        self.vars == other.vars
    }
}

impl Eq for Monomial {}

impl Hash for Monomial {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash only the variables structure, not the coefficient
        let mut vars: Vec<_> = self.vars.iter().collect();
        vars.sort_by_key(|(v, _)| *v);
        for (var, pow) in vars {
            var.hash(state);
            pow.hash(state);
        }
    }
}

impl Monomial {
    /// Create a constant monomial
    pub fn constant(c: Rational) -> Self {
        Self {
            coeff: c,
            vars: FxHashMap::default(),
        }
    }

    /// Create a monomial for a single variable
    pub fn var(var_id: u32) -> Self {
        let mut vars = FxHashMap::default();
        vars.insert(var_id, 1);
        Self {
            coeff: Rational::one(),
            vars,
        }
    }

    /// Create a monomial with coefficient and variables
    pub fn new(coeff: Rational, vars: FxHashMap<u32, u32>) -> Self {
        Self { coeff, vars }
    }

    /// Check if this is a constant (no variables)
    pub fn is_constant(&self) -> bool {
        self.vars.is_empty() || self.vars.values().all(|&p| p == 0)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.coeff.is_zero()
    }

    /// Get the total degree
    pub fn degree(&self) -> u32 {
        self.vars.values().sum()
    }

    /// Multiply two monomials
    pub fn mul(&self, other: &Monomial) -> Monomial {
        let mut vars = self.vars.clone();
        for (&var, &pow) in &other.vars {
            *vars.entry(var).or_default() += pow;
        }
        Monomial {
            coeff: self.coeff.clone() * other.coeff.clone(),
            vars,
        }
    }

    /// Scale by a constant
    pub fn scale(&self, c: &Rational) -> Monomial {
        Monomial {
            coeff: self.coeff.clone() * c.clone(),
            vars: self.vars.clone(),
        }
    }

    /// Negate
    pub fn neg(&self) -> Monomial {
        Monomial {
            coeff: -self.coeff.clone(),
            vars: self.vars.clone(),
        }
    }

    /// Compare monomials for ordering (graded lexicographic)
    pub fn cmp_glex(&self, other: &Monomial) -> Ordering {
        // First compare by total degree
        let deg1 = self.degree();
        let deg2 = other.degree();
        if deg1 != deg2 {
            return deg1.cmp(&deg2);
        }

        // Then lexicographically by variables
        let mut vars1: Vec<_> = self.vars.iter().filter(|(_, p)| **p > 0).collect();
        let mut vars2: Vec<_> = other.vars.iter().filter(|(_, p)| **p > 0).collect();
        vars1.sort_by_key(|(v, _)| *v);
        vars2.sort_by_key(|(v, _)| *v);

        for (v1, v2) in vars1.iter().zip(vars2.iter()) {
            match v1.0.cmp(v2.0) {
                Ordering::Equal => match v1.1.cmp(v2.1) {
                    Ordering::Equal => continue,
                    other => return other,
                },
                other => return other,
            }
        }

        vars1.len().cmp(&vars2.len())
    }

    /// Get variables signature (for combining like terms)
    pub fn vars_key(&self) -> Vec<(u32, u32)> {
        let mut vars: Vec<_> = self
            .vars
            .iter()
            .filter(|(_, p)| **p > 0)
            .map(|(v, p)| (*v, *p))
            .collect();
        vars.sort_by_key(|(v, _)| *v);
        vars
    }
}

/// A polynomial: sum of monomials
#[derive(Clone)]
pub struct Polynomial {
    /// Terms in the polynomial
    pub terms: Vec<Monomial>,
}

impl fmt::Debug for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 {
                if term.coeff.is_negative() {
                    write!(f, " - ")?;
                } else {
                    write!(f, " + ")?;
                }
            }
            if term.is_constant() {
                if i > 0 && term.coeff.is_negative() {
                    write!(f, "{}", -term.coeff.clone())?;
                } else {
                    write!(f, "{}", term.coeff)?;
                }
            } else if i > 0 && term.coeff.is_negative() {
                write!(f, "{:?}", term.neg())?;
            } else {
                write!(f, "{:?}", term)?;
            }
        }
        Ok(())
    }
}

impl Default for Polynomial {
    fn default() -> Self {
        Self::zero()
    }
}

impl Polynomial {
    /// Create zero polynomial
    pub fn zero() -> Self {
        Self { terms: Vec::new() }
    }

    /// Create constant polynomial
    pub fn constant(c: Rational) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            Self {
                terms: vec![Monomial::constant(c)],
            }
        }
    }

    /// Create polynomial for a single variable
    pub fn var(var_id: u32) -> Self {
        Self {
            terms: vec![Monomial::var(var_id)],
        }
    }

    /// Create from monomials
    pub fn from_monomials(terms: Vec<Monomial>) -> Self {
        let mut p = Self { terms };
        p.normalize();
        p
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().all(|t| t.is_zero())
    }

    /// Check if constant
    pub fn is_constant(&self) -> bool {
        self.terms.len() <= 1 && self.terms.iter().all(|t| t.is_constant())
    }

    /// Get constant value if constant
    pub fn get_constant(&self) -> Option<Rational> {
        if self.is_zero() {
            Some(Rational::zero())
        } else if self.is_constant() {
            Some(
                self.terms
                    .first()
                    .map_or(Rational::zero(), |t| t.coeff.clone()),
            )
        } else {
            None
        }
    }

    /// Get the total degree
    pub fn degree(&self) -> u32 {
        self.terms.iter().map(|t| t.degree()).max().unwrap_or(0)
    }

    /// Normalize the polynomial (combine like terms, sort, remove zeros)
    pub fn normalize(&mut self) {
        // Combine like terms
        let mut combined: FxHashMap<Vec<(u32, u32)>, Rational> = FxHashMap::default();
        for term in &self.terms {
            if !term.is_zero() {
                let key = term.vars_key();
                let entry = combined.entry(key).or_insert_with(Rational::zero);
                *entry = entry.clone() + term.coeff.clone();
            }
        }

        // Rebuild terms - use explicit type annotations
        self.terms = combined
            .into_iter()
            .filter(|(_vars, c): &(Vec<(u32, u32)>, Rational)| !c.is_zero())
            .map(|(vars, coeff): (Vec<(u32, u32)>, Rational)| {
                let vars_map: FxHashMap<u32, u32> = vars.into_iter().collect();
                Monomial::new(coeff, vars_map)
            })
            .collect();

        // Sort by graded lexicographic order
        self.terms.sort_by(|a, b| a.cmp_glex(b).reverse());
    }

    /// Add two polynomials
    pub fn add(&self, other: &Polynomial) -> Polynomial {
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        Polynomial::from_monomials(terms)
    }

    /// Subtract two polynomials
    pub fn sub(&self, other: &Polynomial) -> Polynomial {
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().map(|t| t.neg()));
        Polynomial::from_monomials(terms)
    }

    /// Multiply two polynomials
    pub fn mul(&self, other: &Polynomial) -> Polynomial {
        let mut terms = Vec::new();
        for t1 in &self.terms {
            for t2 in &other.terms {
                terms.push(t1.mul(t2));
            }
        }
        Polynomial::from_monomials(terms)
    }

    /// Scale by a constant
    pub fn scale(&self, c: &Rational) -> Polynomial {
        if c.is_zero() {
            Polynomial::zero()
        } else {
            Polynomial::from_monomials(self.terms.iter().map(|t| t.scale(c)).collect())
        }
    }

    /// Negate
    pub fn neg(&self) -> Polynomial {
        Polynomial::from_monomials(self.terms.iter().map(|t| t.neg()).collect())
    }

    /// Get the leading coefficient
    pub fn leading_coeff(&self) -> Option<&Rational> {
        self.terms.first().map(|t| &t.coeff)
    }

    /// Get the leading monomial
    #[allow(dead_code)]
    pub fn leading_monomial(&self) -> Option<&Monomial> {
        self.terms.first()
    }

    /// GCD of all coefficients
    pub fn content(&self) -> Rational {
        if self.terms.is_empty() {
            return Rational::one();
        }

        // For integer coefficients, compute GCD
        // For rationals, return 1 (could be more sophisticated)
        let all_integer = self.terms.iter().all(|t| t.coeff.is_integer());

        if all_integer && !self.terms.is_empty() {
            // Try to convert to i64, fall back to 1 if too large
            let first_numer = self.terms[0].coeff.numer();
            let mut gcd = match first_numer.try_into() {
                Ok(v) => {
                    let v: i64 = v;
                    v.unsigned_abs()
                }
                Err(_) => return Rational::one(),
            };

            for term in &self.terms[1..] {
                let numer = term.coeff.numer();
                let val: u64 = match numer.try_into() {
                    Ok(v) => {
                        let v: i64 = v;
                        v.unsigned_abs()
                    }
                    Err(_) => return Rational::one(),
                };
                gcd = gcd_u64(gcd, val);
                if gcd == 1 {
                    break;
                }
            }
            rat(gcd as i64)
        } else {
            Rational::one()
        }
    }

    /// Divide by content (make primitive)
    #[allow(dead_code)]
    pub fn primitive_part(&self) -> Polynomial {
        let c = self.content();
        if c == Rational::one() || c.is_zero() {
            self.clone()
        } else {
            self.scale(&(Rational::one() / c))
        }
    }

    /// Make all leading coefficients positive
    #[allow(dead_code)]
    pub fn make_monic(&self) -> Polynomial {
        if let Some(lc) = self.leading_coeff()
            && lc.is_negative()
        {
            return self.neg();
        }
        self.clone()
    }

    /// Get all variable IDs in the polynomial
    pub fn variables(&self) -> Vec<u32> {
        let mut vars: Vec<u32> = self
            .terms
            .iter()
            .flat_map(|t| t.vars.keys().copied())
            .collect();
        vars.sort();
        vars.dedup();
        vars
    }

    /// Evaluate polynomial at given variable assignments
    pub fn evaluate(&self, assignments: &FxHashMap<u32, Rational>) -> Rational {
        let mut result = Rational::zero();
        for term in &self.terms {
            let mut term_val = term.coeff.clone();
            for (&var, &pow) in &term.vars {
                if let Some(val) = assignments.get(&var) {
                    term_val *= pow_rational(val, pow);
                }
                // Variable not assigned - treat as unresolved (skip)
            }
            result += term_val;
        }
        result
    }

    /// Substitute a variable with a polynomial
    pub fn substitute(&self, var: u32, replacement: &Polynomial) -> Polynomial {
        let mut result = Polynomial::zero();
        for term in &self.terms {
            if let Some(&pow) = term.vars.get(&var) {
                // x^pow -> replacement^pow
                let mut new_term_vars = term.vars.clone();
                new_term_vars.remove(&var);

                let base = Polynomial::from_monomials(vec![Monomial::new(
                    term.coeff.clone(),
                    new_term_vars,
                )]);

                let mut repl_pow = Polynomial::constant(Rational::one());
                for _ in 0..pow {
                    repl_pow = repl_pow.mul(replacement);
                }

                result = result.add(&base.mul(&repl_pow));
            } else {
                // Variable not present in this term
                result = result.add(&Polynomial::from_monomials(vec![term.clone()]));
            }
        }
        result
    }
}

/// GCD for unsigned integers
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monomial_constant() {
        let m = Monomial::constant(rat(5));
        assert!(m.is_constant());
        assert!(!m.is_zero());
        assert_eq!(m.degree(), 0);
    }

    #[test]
    fn test_monomial_var() {
        let m = Monomial::var(0);
        assert!(!m.is_constant());
        assert_eq!(m.degree(), 1);
    }

    #[test]
    fn test_monomial_mul() {
        let m1 = Monomial::var(0); // x
        let m2 = Monomial::var(0); // x
        let prod = m1.mul(&m2); // x^2

        assert_eq!(prod.degree(), 2);
        assert_eq!(prod.vars.get(&0), Some(&2));
    }

    #[test]
    fn test_polynomial_add() {
        let p1 = Polynomial::var(0); // x
        let p2 = Polynomial::var(1); // y
        let sum = p1.add(&p2); // x + y

        assert_eq!(sum.terms.len(), 2);
    }

    #[test]
    fn test_polynomial_sub() {
        let p1 = Polynomial::var(0); // x
        let p2 = Polynomial::var(0); // x
        let diff = p1.sub(&p2); // 0

        assert!(diff.is_zero());
    }

    #[test]
    fn test_polynomial_mul() {
        let p1 = Polynomial::var(0); // x
        let p2 = Polynomial::var(0); // x
        let prod = p1.mul(&p2); // x^2

        assert_eq!(prod.terms.len(), 1);
        assert_eq!(prod.degree(), 2);
    }

    #[test]
    fn test_polynomial_constant() {
        let p = Polynomial::constant(rat(42));
        assert!(p.is_constant());
        assert_eq!(p.get_constant(), Some(rat(42)));
    }

    #[test]
    fn test_polynomial_normalize() {
        // x + x = 2x
        let p1 = Polynomial::var(0);
        let p2 = Polynomial::var(0);
        let sum = p1.add(&p2);

        assert_eq!(sum.terms.len(), 1);
        assert_eq!(sum.leading_coeff(), Some(&rat(2)));
    }

    #[test]
    fn test_polynomial_content() {
        // 6x + 4y has content 2
        let mut p = Polynomial::var(0).scale(&rat(6));
        p = p.add(&Polynomial::var(1).scale(&rat(4)));

        assert_eq!(p.content(), rat(2));
    }

    #[test]
    fn test_polynomial_evaluate() {
        // p = 2x + 3y
        let mut p = Polynomial::var(0).scale(&rat(2));
        p = p.add(&Polynomial::var(1).scale(&rat(3)));

        let mut assignments = FxHashMap::default();
        assignments.insert(0, rat(5)); // x = 5
        assignments.insert(1, rat(7)); // y = 7

        // 2*5 + 3*7 = 10 + 21 = 31
        assert_eq!(p.evaluate(&assignments), rat(31));
    }

    #[test]
    fn test_polynomial_substitute() {
        // p = x^2
        let p = Polynomial::var(0).mul(&Polynomial::var(0));

        // Substitute x -> (y + 1)
        let repl = Polynomial::var(1).add(&Polynomial::constant(rat(1)));
        let result = p.substitute(0, &repl);

        // Should be y^2 + 2y + 1
        assert_eq!(result.degree(), 2);
        assert_eq!(result.terms.len(), 3);
    }

    #[test]
    fn test_polynomial_variables() {
        // p = x + 2y + 3z
        let mut p = Polynomial::var(0);
        p = p.add(&Polynomial::var(1).scale(&rat(2)));
        p = p.add(&Polynomial::var(2).scale(&rat(3)));

        let vars = p.variables();
        assert_eq!(vars, vec![0, 1, 2]);
    }

    #[test]
    fn test_polynomial_neg() {
        let p = Polynomial::var(0).add(&Polynomial::constant(rat(1)));
        let neg = p.neg();

        // Should be -x - 1
        assert_eq!(neg.terms.len(), 2);
    }
}
