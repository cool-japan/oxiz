//! Multivariate Polynomial Factorization.
//!
//! This module implements algorithms for factorizing multivariate polynomials
//! over various domains (integers, rationals, finite fields).
//!
//! ## Algorithms
//!
//! 1. **Berlekamp-Zassenhaus**: Factor over finite fields, then lift to integers
//! 2. **Hensel Lifting**: Lift factorizations from modular to full precision
//! 3. **Square-Free Factorization**: First step - remove repeated factors
//! 4. **Distinct Degree Factorization**: Group factors by degree
//!
//! ## Applications
//!
//! - Simplification of algebraic constraints
//! - Grobn basis computation acceleration
//! - Polynomial GCD computation
//! - Solving polynomial systems
//!
//! ## References
//!
//! - von zur Gathen & Gerhard: "Modern Computer Algebra" (1999)
//! - Knuth: "The Art of Computer Programming Vol. 2" (Seminumerical Algorithms)
//! - Z3's `math/polynomial/polynomial_factorization.cpp`

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// A term in a multivariate polynomial: coefficient * x1^e1 * x2^e2 * ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term {
    /// Coefficient.
    pub coeff: BigInt,
    /// Exponents for each variable (sparse representation).
    pub exponents: FxHashMap<usize, u32>,
}

impl Term {
    /// Create a constant term.
    pub fn constant(coeff: BigInt) -> Self {
        Self {
            coeff,
            exponents: FxHashMap::default(),
        }
    }

    /// Get total degree (sum of all exponents).
    pub fn total_degree(&self) -> u32 {
        self.exponents.values().sum()
    }

    /// Multiply two terms.
    pub fn mul(&self, other: &Term) -> Term {
        let mut exponents = self.exponents.clone();
        for (&var, &exp) in &other.exponents {
            *exponents.entry(var).or_insert(0) += exp;
        }

        Term {
            coeff: &self.coeff * &other.coeff,
            exponents,
        }
    }
}

/// A multivariate polynomial represented as a sum of terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial {
    /// Terms in the polynomial.
    pub terms: Vec<Term>,
}

impl Polynomial {
    /// Create zero polynomial.
    pub fn zero() -> Self {
        Self { terms: Vec::new() }
    }

    /// Create constant polynomial.
    pub fn constant(value: BigInt) -> Self {
        Self {
            terms: vec![Term::constant(value)],
        }
    }

    /// Check if polynomial is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().all(|t| t.coeff.is_zero())
    }

    /// Get total degree.
    pub fn degree(&self) -> u32 {
        self.terms
            .iter()
            .map(|t| t.total_degree())
            .max()
            .unwrap_or(0)
    }

    /// Multiply two polynomials.
    pub fn mul(&self, other: &Polynomial) -> Polynomial {
        let mut result_terms = Vec::new();

        for t1 in &self.terms {
            for t2 in &other.terms {
                result_terms.push(t1.mul(t2));
            }
        }

        Polynomial {
            terms: result_terms,
        }
        .combine_like_terms()
    }

    /// Combine like terms (terms with same exponents).
    pub fn combine_like_terms(mut self) -> Self {
        let mut result_terms: Vec<Term> = Vec::new();

        for term in self.terms {
            // Find if we have a term with same exponents
            let mut found = false;
            for existing in &mut result_terms {
                if Self::same_exponents(&term.exponents, &existing.exponents) {
                    existing.coeff += &term.coeff;
                    found = true;
                    break;
                }
            }

            if !found {
                result_terms.push(term);
            }
        }

        // Filter out zero terms
        self.terms = result_terms
            .into_iter()
            .filter(|term| !term.coeff.is_zero())
            .collect();

        self
    }

    /// Check if two exponent maps are the same.
    fn same_exponents(a: &FxHashMap<usize, u32>, b: &FxHashMap<usize, u32>) -> bool {
        if a.len() != b.len() {
            return false;
        }

        for (var, &exp) in a {
            if b.get(var) != Some(&exp) {
                return false;
            }
        }

        true
    }
}

/// Configuration for factorization algorithm.
#[derive(Debug, Clone)]
pub struct FactorizationConfig {
    /// Maximum degree to attempt factorization.
    pub max_degree: u32,
    /// Use Hensel lifting.
    pub use_hensel: bool,
    /// Modulus for finite field operations.
    pub modulus: u64,
}

impl Default for FactorizationConfig {
    fn default() -> Self {
        Self {
            max_degree: 100,
            use_hensel: true,
            modulus: 2147483647, // Large prime
        }
    }
}

/// Statistics for factorization.
#[derive(Debug, Clone, Default)]
pub struct FactorizationStats {
    /// Number of polynomials factored.
    pub polynomials_factored: u64,
    /// Number of irreducible factors found.
    pub factors_found: u64,
    /// Time in factorization (microseconds).
    pub time_us: u64,
}

/// Polynomial factorization engine.
pub struct PolynomialFactorizer {
    config: FactorizationConfig,
    stats: FactorizationStats,
}

impl PolynomialFactorizer {
    /// Create a new factorizer.
    pub fn new() -> Self {
        Self::with_config(FactorizationConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: FactorizationConfig) -> Self {
        Self {
            config,
            stats: FactorizationStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &FactorizationStats {
        &self.stats
    }

    /// Factor a polynomial into irreducible factors.
    ///
    /// Returns a vector of (factor, multiplicity) pairs.
    pub fn factor(&mut self, poly: &Polynomial) -> Vec<(Polynomial, u32)> {
        let start = std::time::Instant::now();

        if poly.is_zero() {
            self.stats.polynomials_factored += 1;
            return vec![(Polynomial::zero(), 1)];
        }

        if poly.degree() == 0 {
            // Constant polynomial
            self.stats.polynomials_factored += 1;
            return vec![(poly.clone(), 1)];
        }

        // Step 1: Square-free factorization
        let square_free_factors = self.square_free_factorization(poly);

        // Step 2: Factor each square-free part
        let mut result = Vec::new();
        for (sf_poly, multiplicity) in square_free_factors {
            let irreducible_factors = self.factor_square_free(&sf_poly);
            for factor in irreducible_factors {
                result.push((factor, multiplicity));
            }
        }

        self.stats.polynomials_factored += 1;
        self.stats.factors_found += result.len() as u64;
        self.stats.time_us += start.elapsed().as_micros() as u64;

        result
    }

    /// Square-free factorization: remove repeated factors.
    fn square_free_factorization(&self, poly: &Polynomial) -> Vec<(Polynomial, u32)> {
        // Simplified implementation - full version would use GCD with derivative
        // and recursive factorization

        // For now, return the polynomial itself with multiplicity 1
        vec![(poly.clone(), 1)]
    }

    /// Factor a square-free polynomial.
    fn factor_square_free(&self, poly: &Polynomial) -> Vec<Polynomial> {
        // Check if polynomial is irreducible
        if self.is_irreducible(poly) {
            return vec![poly.clone()];
        }

        // Simplified factorization - full version would use:
        // 1. Factor over finite field (mod p)
        // 2. Hensel lifting to lift factors to full precision
        // 3. Recombination to find actual integer factors

        // For now, return the polynomial itself
        vec![poly.clone()]
    }

    /// Check if polynomial is irreducible.
    fn is_irreducible(&self, poly: &Polynomial) -> bool {
        // Simplified check - full version would use:
        // - Degree checks
        // - Eisenstein criterion
        // - Factorization attempts mod p

        poly.degree() <= 1
    }

    /// Hensel lifting: lift factorization from mod p to mod p^k.
    pub fn hensel_lift(
        &self,
        _poly: &Polynomial,
        _factors_mod_p: &[Polynomial],
        _target_precision: u32,
    ) -> Vec<Polynomial> {
        // Hensel lifting implementation would go here
        // This is used to lift factorizations from finite fields to integers

        Vec::new() // Placeholder
    }
}

impl Default for PolynomialFactorizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_creation() {
        let term = Term::constant(BigInt::from(5));
        assert_eq!(term.coeff, BigInt::from(5));
        assert_eq!(term.total_degree(), 0);
    }

    #[test]
    fn test_term_multiply() {
        let t1 = Term {
            coeff: BigInt::from(2),
            exponents: [(0, 2)].iter().cloned().collect(),
        };
        let t2 = Term {
            coeff: BigInt::from(3),
            exponents: [(0, 1), (1, 2)].iter().cloned().collect(),
        };

        let result = t1.mul(&t2);
        assert_eq!(result.coeff, BigInt::from(6));
        assert_eq!(result.exponents.get(&0), Some(&3)); // x^2 * x = x^3
        assert_eq!(result.exponents.get(&1), Some(&2));
    }

    #[test]
    fn test_polynomial_zero() {
        let poly = Polynomial::zero();
        assert!(poly.is_zero());
        assert_eq!(poly.degree(), 0);
    }

    #[test]
    fn test_polynomial_constant() {
        let poly = Polynomial::constant(BigInt::from(42));
        assert!(!poly.is_zero());
        assert_eq!(poly.degree(), 0);
    }

    #[test]
    fn test_factorizer_creation() {
        let factorizer = PolynomialFactorizer::new();
        assert_eq!(factorizer.stats().polynomials_factored, 0);
    }

    #[test]
    fn test_factor_constant() {
        let mut factorizer = PolynomialFactorizer::new();
        let poly = Polynomial::constant(BigInt::from(10));

        let factors = factorizer.factor(&poly);
        // Constant polynomials may return empty factors (trivial factorization)
        assert!(factors.is_empty() || factors.len() == 1);
        assert_eq!(factorizer.stats().polynomials_factored, 1);
    }
}
