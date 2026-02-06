//! Sparse Polynomial Operations.
//!
//! Optimized operations for sparse multivariate polynomials where most
//! coefficients are zero. Uses efficient data structures and algorithms
//! tailored for sparsity.
//!
//! ## Optimizations
//!
//! - **Sparse storage**: Only store non-zero terms
//! - **Fast lookups**: Hash-based term access
//! - **Efficient multiplication**: Skip zero terms
//! - **Memory efficiency**: Minimal allocation for sparse inputs
//!
//! ## References
//!
//! - Monagan & Pearce: "Sparse Polynomial Division" (2011)
//! - Z3's `math/polynomial/polynomial.cpp` (sparse mode)

use super::{Monomial, MonomialOrder, Polynomial, Term, Var};
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// Configuration for sparse operations.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Threshold for considering a polynomial sparse (ratio of non-zero to total possible terms).
    pub sparsity_threshold: f64,
    /// Enable fast multiplication for very sparse polynomials.
    pub enable_fast_mul: bool,
    /// Maximum terms before switching to dense representation.
    pub max_sparse_terms: usize,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.1,
            enable_fast_mul: true,
            max_sparse_terms: 10000,
        }
    }
}

/// Statistics for sparse operations.
#[derive(Debug, Clone, Default)]
pub struct SparseStats {
    /// Number of zero terms skipped.
    pub zero_terms_skipped: u64,
    /// Memory saved (estimated bytes).
    pub memory_saved: u64,
    /// Fast multiplications performed.
    pub fast_muls: u64,
}

/// Sparse polynomial operations engine.
pub struct SparseOps {
    /// Configuration.
    config: SparseConfig,
    /// Statistics.
    stats: SparseStats,
}

impl SparseOps {
    /// Create a new sparse operations engine.
    pub fn new(config: SparseConfig) -> Self {
        Self {
            config,
            stats: SparseStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SparseConfig::default())
    }

    /// Check if a polynomial is sparse according to configuration.
    pub fn is_sparse(&self, p: &Polynomial) -> bool {
        if p.num_terms() > self.config.max_sparse_terms {
            return false;
        }

        // Estimate total possible terms based on max degree and variables
        let num_vars = p.vars().len();
        let max_degree = p.total_degree() as usize;

        if num_vars == 0 {
            return false;
        }

        // For multivariate, approximate: (degree + vars choose vars)
        // Simplified: if fewer than threshold * degree^vars terms
        let approx_dense_size = if num_vars <= 3 {
            max_degree.pow(num_vars as u32)
        } else {
            // For many variables, just use linear estimate
            max_degree * num_vars * 100
        };

        let sparsity = p.num_terms() as f64 / approx_dense_size as f64;
        sparsity < self.config.sparsity_threshold
    }

    /// Sparse multiplication optimized for very sparse inputs.
    pub fn sparse_mul(&mut self, p: &Polynomial, q: &Polynomial) -> Polynomial {
        if !self.config.enable_fast_mul {
            return p * q; // Use standard multiplication
        }

        self.stats.fast_muls += 1;

        // Use hash map for efficient term collection
        let mut term_map: FxHashMap<Monomial, BigRational> = FxHashMap::default();

        for term_p in p.terms() {
            for term_q in q.terms() {
                // Multiply monomials
                let mono = term_p.monomial.mul(&term_q.monomial);

                // Multiply coefficients
                let coeff = &term_p.coeff * &term_q.coeff;

                if !coeff.is_zero() {
                    // Add to existing term or insert new
                    term_map
                        .entry(mono)
                        .and_modify(|c| *c = c.clone() + &coeff)
                        .or_insert(coeff);
                } else {
                    self.stats.zero_terms_skipped += 1;
                }
            }
        }

        // Convert map to term vector, filtering zeros
        let mut terms: Vec<Term> = term_map
            .iter()
            .filter_map(|(mono, coeff)| {
                if !coeff.is_zero() {
                    Some(Term::new(coeff.clone(), mono.clone()))
                } else {
                    self.stats.zero_terms_skipped += 1;
                    None
                }
            })
            .collect();

        // Sort by monomial order
        terms.sort_by(|a, b| MonomialOrder::GRevLex.compare(&a.monomial, &b.monomial));

        Polynomial::from_terms(terms, MonomialOrder::GRevLex)
    }

    /// Sparse addition that skips zero terms.
    pub fn sparse_add(&mut self, p: &Polynomial, q: &Polynomial) -> Polynomial {
        let mut term_map: FxHashMap<Monomial, BigRational> = FxHashMap::default();

        // Add terms from p
        for term in p.terms() {
            term_map.insert(term.monomial.clone(), term.coeff.clone());
        }

        // Add terms from q
        for term in q.terms() {
            term_map
                .entry(term.monomial.clone())
                .and_modify(|c| *c = c.clone() + &term.coeff)
                .or_insert(term.coeff.clone());
        }

        // Filter zeros and convert
        let terms: Vec<Term> = term_map
            .iter()
            .filter_map(|(mono, coeff)| {
                if !coeff.is_zero() {
                    Some(Term::new(coeff.clone(), mono.clone()))
                } else {
                    self.stats.zero_terms_skipped += 1;
                    None
                }
            })
            .collect();

        Polynomial::from_terms(terms, MonomialOrder::GRevLex)
    }

    /// Evaluate sparse polynomial at given point (hash-based).
    pub fn sparse_eval(
        &mut self,
        p: &Polynomial,
        point: &FxHashMap<Var, BigRational>,
    ) -> BigRational {
        let mut result = BigRational::zero();

        for term in p.terms() {
            // Evaluate monomial
            let mut mono_val = BigRational::one();

            for vp in term.monomial.vars() {
                if let Some(val) = point.get(&vp.var) {
                    // Compute val^power
                    let powered = self.power_rational(val, vp.power);
                    mono_val *= powered;
                } else {
                    // Variable not in point - use 0 (makes monomial zero)
                    self.stats.zero_terms_skipped += 1;
                    mono_val = BigRational::zero();
                    break;
                }
            }

            result += &term.coeff * &mono_val;
        }

        result
    }

    /// Raise rational to integer power.
    fn power_rational(&self, base: &BigRational, exp: u32) -> BigRational {
        if exp == 0 {
            BigRational::one()
        } else if exp == 1 {
            base.clone()
        } else {
            let mut result = BigRational::one();
            let mut b = base.clone();
            let mut e = exp;

            // Fast exponentiation
            while e > 0 {
                if e % 2 == 1 {
                    result *= &b;
                }
                b = &b * &b;
                e /= 2;
            }

            result
        }
    }

    /// Estimate memory usage of polynomial.
    pub fn estimate_memory(&self, p: &Polynomial) -> usize {
        // Rough estimate: each term is ~100 bytes (monomial + coeff + overhead)
        p.num_terms() * 100
    }

    /// Estimate memory savings from sparse representation.
    pub fn estimate_savings(&mut self, p: &Polynomial) -> usize {
        let num_vars = p.vars().len();
        let max_degree = p.total_degree() as usize;

        // Dense representation would have degree^vars terms
        let dense_terms = if num_vars <= 3 {
            max_degree.pow(num_vars as u32)
        } else {
            max_degree * num_vars * 100
        };

        let sparse_memory = self.estimate_memory(p);
        let dense_memory = dense_terms * 100;

        let savings = dense_memory.saturating_sub(sparse_memory);
        self.stats.memory_saved += savings as u64;
        savings
    }

    /// Get statistics.
    pub fn stats(&self) -> &SparseStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SparseStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_sparse_ops_creation() {
        let ops = SparseOps::default_config();
        assert_eq!(ops.stats().fast_muls, 0);
    }

    #[test]
    fn test_is_sparse() {
        let ops = SparseOps::default_config();

        // Sparse: x^5 + y^5 (2 terms in 2 variables, degree 5)
        // With degree 5 and 2 vars, approx_dense = 5^2 = 25, sparsity = 2/25 = 0.08 < 0.1
        let sparse = Polynomial::from_coeffs_int(&[(1, &[(0, 5)]), (1, &[(1, 5)])]);

        assert!(ops.is_sparse(&sparse));

        // Constant is not considered sparse (num_vars = 0)
        let constant = Polynomial::constant(BigRational::from_integer(BigInt::from(5)));
        assert!(!ops.is_sparse(&constant));
    }

    #[test]
    fn test_sparse_mul() {
        let mut ops = SparseOps::default_config();

        // x * y
        let p = Polynomial::from_var(0);
        let q = Polynomial::from_var(1);

        let result = ops.sparse_mul(&p, &q);

        assert_eq!(result.total_degree(), 2);
        assert_eq!(ops.stats().fast_muls, 1);
    }

    #[test]
    fn test_sparse_add() {
        let mut ops = SparseOps::default_config();

        // x + y
        let p = Polynomial::from_var(0);
        let q = Polynomial::from_var(1);

        let result = ops.sparse_add(&p, &q);

        assert_eq!(result.num_terms(), 2);
    }

    #[test]
    fn test_sparse_eval() {
        let mut ops = SparseOps::default_config();

        // 2x + 3y
        let p = Polynomial::from_coeffs_int(&[(2, &[(0, 1)]), (3, &[(1, 1)])]);

        let mut point = FxHashMap::default();
        point.insert(0, rat(5)); // x = 5
        point.insert(1, rat(2)); // y = 2

        let result = ops.sparse_eval(&p, &point);

        // 2*5 + 3*2 = 16
        assert_eq!(result, rat(16));
    }

    #[test]
    fn test_power_rational() {
        let ops = SparseOps::default_config();

        assert_eq!(ops.power_rational(&rat(2), 0), rat(1));
        assert_eq!(ops.power_rational(&rat(2), 1), rat(2));
        assert_eq!(ops.power_rational(&rat(2), 3), rat(8));
    }

    #[test]
    fn test_estimate_memory() {
        let ops = SparseOps::default_config();

        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)])]);

        let memory = ops.estimate_memory(&p);
        assert!(memory > 0);
    }

    #[test]
    fn test_estimate_savings() {
        let mut ops = SparseOps::default_config();

        // Very sparse polynomial
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 10)]), // x^10
            (1, &[]),        // constant
        ]);

        let savings = ops.estimate_savings(&p);
        assert!(savings > 0);
    }
}
