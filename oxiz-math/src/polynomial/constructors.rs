//! Polynomial constructors and query methods.

use super::types::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

impl super::Polynomial {
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
    /// `coeffs[i]` is the coefficient of x^i.
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
    pub fn coeff(&self, var: Var, k: u32) -> super::Polynomial {
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
        super::Polynomial::from_terms(terms, self.order)
    }

    /// Get the leading coefficient with respect to variable x.
    pub fn leading_coeff_wrt(&self, var: Var) -> super::Polynomial {
        let d = self.degree(var);
        self.coeff(var, d)
    }

    /// Normalize the polynomial (sort terms and combine like terms).
    pub(crate) fn normalize(&mut self) {
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
}
