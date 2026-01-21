//! Arithmetic operations for polynomials.

use super::types::*;
use num_rational::BigRational;
use num_traits::{One, Zero};

impl super::Polynomial {
    /// Negate the polynomial.
    pub fn neg(&self) -> super::Polynomial {
        super::Polynomial {
            terms: self
                .terms
                .iter()
                .map(|t| Term::new(-t.coeff.clone(), t.monomial.clone()))
                .collect(),
            order: self.order,
        }
    }

    /// Add two polynomials.
    pub fn add(&self, other: &super::Polynomial) -> super::Polynomial {
        let mut terms: Vec<Term> = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        super::Polynomial::from_terms(terms, self.order)
    }

    /// Subtract two polynomials.
    pub fn sub(&self, other: &super::Polynomial) -> super::Polynomial {
        self.add(&other.neg())
    }

    /// Multiply by a scalar.
    pub fn scale(&self, c: &BigRational) -> super::Polynomial {
        if c.is_zero() {
            return super::Polynomial::zero();
        }
        if c.is_one() {
            return self.clone();
        }
        super::Polynomial {
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
    pub fn mul(&self, other: &super::Polynomial) -> super::Polynomial {
        if self.is_zero() || other.is_zero() {
            return super::Polynomial::zero();
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

        super::Polynomial::from_terms(terms, self.order)
    }

    /// Karatsuba multiplication for univariate polynomials.
    /// Time complexity: O(n^1.585) instead of O(n^2) for naive multiplication.
    ///
    /// This is particularly efficient for polynomials with 16+ terms.
    fn mul_karatsuba(&self, other: &super::Polynomial) -> super::Polynomial {
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
            return super::Polynomial::from_terms(terms, self.order);
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

        let sum0 = super::Polynomial::add(&low0, &high0);
        let sum1 = super::Polynomial::add(&low1, &high1);
        let z1_temp = sum0.mul_karatsuba(&sum1);

        // z1 = z1_temp - z0 - z2
        let z1 = super::Polynomial::sub(&super::Polynomial::sub(&z1_temp, &z0), &z2);

        // Result = z0 + z1*x^split + z2*x^(2*split)
        let x_split = Monomial::from_var_power(var, split);
        let x_2split = Monomial::from_var_power(var, 2 * split);

        let term1 = z1.mul_monomial(&x_split);
        let term2 = z2.mul_monomial(&x_2split);

        super::Polynomial::add(&super::Polynomial::add(&z0, &term1), &term2)
    }

    /// Split a univariate polynomial into low and high parts at a given degree.
    /// Returns (low, high) where poly = low + high * x^deg.
    fn split_at_degree(&self, var: Var, deg: u32) -> (super::Polynomial, super::Polynomial) {
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
            super::Polynomial::zero()
        } else {
            super::Polynomial::from_terms(low_terms, self.order)
        };

        let high = if high_terms.is_empty() {
            super::Polynomial::zero()
        } else {
            super::Polynomial::from_terms(high_terms, self.order)
        };

        (low, high)
    }

    /// Multiply by a monomial.
    pub fn mul_monomial(&self, m: &Monomial) -> super::Polynomial {
        if m.is_unit() {
            return self.clone();
        }
        super::Polynomial {
            terms: self
                .terms
                .iter()
                .map(|t| Term::new(t.coeff.clone(), t.monomial.mul(m)))
                .collect(),
            order: self.order,
        }
    }

    /// Compute p^k.
    pub fn pow(&self, k: u32) -> super::Polynomial {
        if k == 0 {
            return super::Polynomial::one();
        }
        if k == 1 {
            return self.clone();
        }
        if self.is_zero() {
            return super::Polynomial::zero();
        }

        // Binary exponentiation
        let mut result = super::Polynomial::one();
        let mut base = self.clone();
        let mut exp = k;

        while exp > 0 {
            if exp & 1 == 1 {
                result = super::Polynomial::mul(&result, &base);
            }
            base = super::Polynomial::mul(&base, &base);
            exp >>= 1;
        }

        result
    }
}
