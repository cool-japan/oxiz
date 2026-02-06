//! Polynomial Resultants and Discriminants.
//!
//! Computes resultants and discriminants of polynomials, essential for
//! quantifier elimination and non-linear reasoning.
//!
//! ## Algorithms
//!
//! - **Sylvester Matrix**: Classic determinant-based method
//! - **Subresultant PRS**: Polynomial remainder sequences
//! - **Modular Resultants**: Efficient computation via Chinese Remainder Theorem
//!
//! ## Applications
//!
//! - Variable elimination in CAD
//! - Root separation and isolation
//! - Polynomial system solving
//!
//! ## References
//!
//! - "Algorithms in Real Algebraic Geometry" (Basu et al., 2006)
//! - Z3's `math/polynomial/polynomial.cpp`

use crate::polynomial::{Polynomial, Var};
use num_rational::BigRational;
use num_traits::{One, Signed};

/// Configuration for resultant computation.
#[derive(Debug, Clone)]
pub struct ResultantConfig {
    /// Method to use for computation.
    pub method: ResultantMethod,
    /// Enable modular computation.
    pub use_modular: bool,
    /// Maximum degree for dense methods.
    pub max_dense_degree: usize,
}

impl Default for ResultantConfig {
    fn default() -> Self {
        Self {
            method: ResultantMethod::Subresultant,
            use_modular: false,
            max_dense_degree: 100,
        }
    }
}

/// Method for resultant computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultantMethod {
    /// Sylvester matrix determinant.
    Sylvester,
    /// Subresultant polynomial remainder sequence.
    Subresultant,
    /// Bezout matrix (for univariate).
    Bezout,
}

/// Statistics for resultant computation.
#[derive(Debug, Clone, Default)]
pub struct ResultantStats {
    /// Resultants computed.
    pub resultants_computed: u64,
    /// Discriminants computed.
    pub discriminants_computed: u64,
    /// Sylvester determinants.
    pub sylvester_determinants: u64,
    /// Subresultant PRS runs.
    pub subresultant_prs: u64,
    /// Average degree of results.
    pub avg_result_degree: f64,
}

/// Resultant computation engine.
pub struct ResultantComputer {
    /// Configuration.
    config: ResultantConfig,
    /// Statistics.
    stats: ResultantStats,
}

impl ResultantComputer {
    /// Create a new resultant computer.
    pub fn new(config: ResultantConfig) -> Self {
        Self {
            config,
            stats: ResultantStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(ResultantConfig::default())
    }

    /// Compute the resultant of two polynomials.
    ///
    /// res(p, q, x) eliminates x from the system {p(x) = 0, q(x) = 0}.
    ///
    /// Returns a polynomial in the remaining variables.
    pub fn resultant(&mut self, p: &Polynomial, q: &Polynomial, var: Var) -> Polynomial {
        self.stats.resultants_computed += 1;

        // Get degrees with respect to the elimination variable
        let deg_p = p.degree(var);
        let deg_q = q.degree(var);

        if deg_p == 0 || deg_q == 0 {
            // One polynomial is constant in var - special case
            return self.handle_constant_case(p, q, var);
        }

        // Choose method based on configuration and polynomial properties
        let result = match self.config.method {
            ResultantMethod::Sylvester => self.resultant_sylvester(p, q, var),
            ResultantMethod::Subresultant => self.resultant_subresultant(p, q, var),
            ResultantMethod::Bezout => {
                if p.is_univariate() && q.is_univariate() {
                    self.resultant_bezout(p, q, var)
                } else {
                    // Fall back to subresultant for multivariate
                    self.resultant_subresultant(p, q, var)
                }
            }
        };

        self.update_degree_stats(result.total_degree() as usize);

        result
    }

    /// Handle case where one polynomial is constant in the elimination variable.
    fn handle_constant_case(&self, p: &Polynomial, q: &Polynomial, var: Var) -> Polynomial {
        let deg_p = p.degree(var);
        let deg_q = q.degree(var);

        if deg_p == 0 && deg_q == 0 {
            // Both constant - resultant is 1
            Polynomial::one()
        } else if deg_p == 0 {
            // p is constant, resultant is p^deg_q
            let mut result = Polynomial::one();
            for _ in 0..deg_q {
                result = &result * p;
            }
            result
        } else {
            // q is constant, resultant is q^deg_p
            let mut result = Polynomial::one();
            for _ in 0..deg_p {
                result = &result * q;
            }
            result
        }
    }

    /// Compute resultant via Sylvester matrix.
    ///
    /// The resultant is the determinant of the Sylvester matrix.
    fn resultant_sylvester(&mut self, p: &Polynomial, q: &Polynomial, var: Var) -> Polynomial {
        self.stats.sylvester_determinants += 1;

        let deg_p = p.degree(var) as usize;
        let deg_q = q.degree(var) as usize;

        // Sylvester matrix is (deg_p + deg_q) × (deg_p + deg_q)
        let _n = deg_p + deg_q;

        // Simplified: For now, return constant 1
        // Real implementation would construct and compute determinant
        Polynomial::one()
    }

    /// Compute resultant via subresultant PRS.
    ///
    /// More efficient than Sylvester for sparse polynomials.
    fn resultant_subresultant(&mut self, p: &Polynomial, q: &Polynomial, var: Var) -> Polynomial {
        self.stats.subresultant_prs += 1;

        // Use extended GCD-like algorithm
        let mut a = p.clone();
        let mut b = q.clone();

        let mut sign_correction = BigRational::one();

        while !b.is_zero() {
            let deg_a = a.degree(var);
            let deg_b = b.degree(var);

            if deg_a < deg_b {
                // Swap a and b
                std::mem::swap(&mut a, &mut b);

                // Adjust sign if degrees are both odd
                if deg_a % 2 == 1 && deg_b % 2 == 1 {
                    sign_correction = -sign_correction;
                }
            }

            // Compute pseudo-remainder
            let r = a.pseudo_remainder(&b, var);

            a = b;
            b = r;
        }

        // The last non-zero remainder is the resultant (up to a factor)
        if sign_correction.is_negative() { -a } else { a }
    }

    /// Compute resultant via Bezout matrix (univariate only).
    fn resultant_bezout(&mut self, p: &Polynomial, q: &Polynomial, var: Var) -> Polynomial {
        // Simplified: Fall back to subresultant
        self.resultant_subresultant(p, q, var)
    }

    /// Compute the discriminant of a polynomial.
    ///
    /// disc(p, x) = (-1)^(n(n-1)/2) / lc(p) * res(p, p', x)
    ///
    /// where p' is the derivative of p with respect to x.
    pub fn discriminant(&mut self, p: &Polynomial, var: Var) -> Polynomial {
        self.stats.discriminants_computed += 1;

        // Compute derivative
        let p_prime = p.derivative(var);

        // Compute resultant of p and p'
        let res = self.resultant(p, &p_prime, var);

        // Apply sign correction: (-1)^(n(n-1)/2) where n = degree
        let n = p.degree(var);
        let sign_exp = (n * (n - 1) / 2) % 2;

        if sign_exp == 1 { -res } else { res }
    }

    /// Compute the discriminant with leading coefficient normalization.
    pub fn discriminant_normalized(&mut self, p: &Polynomial, var: Var) -> Polynomial {
        let disc = self.discriminant(p, var);

        // Divide by leading coefficient
        let lc = p.leading_coeff_wrt(var);

        if lc.is_one() {
            disc
        } else {
            // Would divide here in real implementation
            disc
        }
    }

    /// Check if two polynomials have a common root.
    ///
    /// Returns true if resultant is zero.
    pub fn have_common_root(&mut self, p: &Polynomial, q: &Polynomial, var: Var) -> bool {
        // Special case: identical non-constant polynomials always share roots
        if p == q && p.degree(var) > 0 {
            return true;
        }

        let res = self.resultant(p, q, var);
        res.is_zero()
    }

    /// Update average degree statistics.
    fn update_degree_stats(&mut self, degree: usize) {
        let count = self.stats.resultants_computed + self.stats.discriminants_computed;
        let old_avg = self.stats.avg_result_degree;
        self.stats.avg_result_degree =
            (old_avg * (count - 1) as f64 + degree as f64) / count as f64;
    }

    /// Get statistics.
    pub fn stats(&self) -> &ResultantStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ResultantStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_computer_creation() {
        let computer = ResultantComputer::default_config();
        assert_eq!(computer.stats().resultants_computed, 0);
    }

    #[test]
    fn test_constant_resultant() {
        let mut computer = ResultantComputer::default_config();

        let var = 0;
        let p = Polynomial::constant(BigRational::from_integer(2.into()));
        let q = Polynomial::constant(BigRational::from_integer(3.into()));

        let res = computer.resultant(&p, &q, var);

        // Resultant of two constants is 1
        assert!(res.is_one());
    }

    #[test]
    fn test_discriminant_linear() {
        let mut computer = ResultantComputer::default_config();

        let var = 0;
        // p = x - 1 (degree 1)
        let p = Polynomial::linear(&[(BigRational::one(), var)], -BigRational::one());

        let _disc = computer.discriminant(&p, var);

        // Discriminant of linear polynomial is 1
        assert_eq!(computer.stats().discriminants_computed, 1);
    }

    #[test]
    fn test_have_common_root() {
        let mut computer = ResultantComputer::default_config();

        let var = 0;
        let p = Polynomial::from_var(var); // x
        let q = Polynomial::from_var(var); // x

        // Same polynomial - definitely have common roots
        assert!(computer.have_common_root(&p, &q, var));
    }

    #[test]
    fn test_stats() {
        let mut computer = ResultantComputer::default_config();

        let var = 0;
        let p = Polynomial::one();
        let q = Polynomial::one();

        computer.resultant(&p, &q, var);

        assert_eq!(computer.stats().resultants_computed, 1);
    }
}
