//! Polynomial arithmetic for non-linear theories.
//!
//! This module provides multivariate polynomial representation and operations
//! for SMT solving, particularly for non-linear real arithmetic (NRA).
//!
//! Reference: Z3's `math/polynomial/` directory.

mod types;
mod constructors;
mod arithmetic;
mod extended_ops;
mod advanced_ops;
mod helpers;

// Re-export all public types
pub use types::{Monomial, MonomialOrder, Term, Var, VarPower, NULL_VAR};

// Re-export public utility functions
pub use helpers::rational_sqrt;

use num_traits::Signed;
use std::ops::{Add, Mul, Neg, Sub};

/// A multivariate polynomial over rationals.
/// Represented as a sum of terms, sorted by monomial order.
#[derive(Clone)]
pub struct Polynomial {
    /// Terms in decreasing order (according to monomial order).
    pub(crate) terms: Vec<Term>,
    /// The monomial ordering used.
    pub(crate) order: MonomialOrder,
}

// Operator trait implementations
impl Neg for Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Polynomial {
        Polynomial::neg(&self)
    }
}

impl Neg for &Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Polynomial {
        Polynomial::neg(self)
    }
}

impl Add for Polynomial {
    type Output = Polynomial;

    fn add(self, other: Polynomial) -> Polynomial {
        Polynomial::add(&self, &other)
    }
}

impl Add<&Polynomial> for &Polynomial {
    type Output = Polynomial;

    fn add(self, other: &Polynomial) -> Polynomial {
        Polynomial::add(self, other)
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: Polynomial) -> Polynomial {
        Polynomial::sub(&self, &other)
    }
}

impl Sub<&Polynomial> for &Polynomial {
    type Output = Polynomial;

    fn sub(self, other: &Polynomial) -> Polynomial {
        Polynomial::sub(self, other)
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: Polynomial) -> Polynomial {
        Polynomial::mul(&self, &other)
    }
}

impl Mul<&Polynomial> for &Polynomial {
    type Output = Polynomial;

    fn mul(self, other: &Polynomial) -> Polynomial {
        Polynomial::mul(self, other)
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }

        for i in 0..self.terms.len() {
            if self.terms[i] != other.terms[i] {
                return false;
            }
        }

        true
    }
}

impl Eq for Polynomial {}

impl std::fmt::Debug for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            for (i, term) in self.terms.iter().enumerate() {
                if i > 0 {
                    if term.coeff.is_positive() {
                        write!(f, " + ")?;
                    } else {
                        write!(f, " - ")?;
                    }
                    let mut t = term.clone();
                    t.coeff = t.coeff.abs();
                    write!(f, "{:?}", t)?;
                } else {
                    write!(f, "{:?}", term)?;
                }
            }
            Ok(())
        }
    }
}

impl std::fmt::Display for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}
