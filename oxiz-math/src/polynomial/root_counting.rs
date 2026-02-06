//! Polynomial Root Counting.
//!
//! Efficient algorithms for counting real roots of polynomials in intervals,
//! using Descartes' rule of signs, Sturm sequences, and Budan's theorem.
//!
//! ## Algorithms
//!
//! - **Descartes' Rule**: Count sign variations in coefficients
//! - **Sturm Sequence**: Exact root counting with polynomial remainder sequence
//! - **Budan's Theorem**: Efficient root counting via derivative evaluations
//!
//! ## References
//!
//! - "Algorithms in Real Algebraic Geometry" (Basu, Pollack, Roy, 2006)
//! - Z3's `math/polynomial/polynomial.cpp`

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// A univariate polynomial over Q.
#[derive(Debug, Clone)]
pub struct Polynomial {
    /// Coefficients in increasing degree order: a_0 + a_1*x + a_2*x^2 + ...
    pub coeffs: Vec<BigRational>,
}

impl Polynomial {
    /// Create a polynomial from coefficients.
    pub fn new(coeffs: Vec<BigRational>) -> Self {
        let mut poly = Self { coeffs };
        poly.normalize();
        poly
    }

    /// Remove trailing zeros.
    fn normalize(&mut self) {
        // Safety: len() > 1 ensures last() is Some, but use map_or for no-unwrap policy
        while self.coeffs.len() > 1 && self.coeffs.last().is_some_and(|c| c.is_zero()) {
            self.coeffs.pop();
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(BigRational::zero());
        }
    }

    /// Get the degree of the polynomial.
    pub fn degree(&self) -> usize {
        if self.coeffs.len() == 1 && self.coeffs[0].is_zero() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    /// Evaluate the polynomial at a point.
    pub fn eval(&self, x: &BigRational) -> BigRational {
        // Horner's method
        let mut result = BigRational::zero();
        for coeff in self.coeffs.iter().rev() {
            result = result * x + coeff;
        }
        result
    }

    /// Compute the derivative.
    pub fn derivative(&self) -> Polynomial {
        if self.degree() == 0 {
            return Polynomial::new(vec![BigRational::zero()]);
        }

        let coeffs: Vec<_> = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| c * BigRational::from_integer(BigInt::from(i)))
            .collect();

        Polynomial::new(coeffs)
    }

    /// Compute polynomial remainder: self mod other.
    pub fn remainder(&self, other: &Polynomial) -> Polynomial {
        let mut rem = self.clone();
        let divisor_deg = other.degree();
        let divisor_lead = &other.coeffs[divisor_deg];

        while rem.degree() >= divisor_deg && !rem.coeffs.is_empty() {
            let rem_deg = rem.degree();
            let rem_lead = &rem.coeffs[rem_deg];
            let factor = rem_lead / divisor_lead;

            // Subtract factor * other from rem
            for i in 0..=divisor_deg {
                rem.coeffs[rem_deg - divisor_deg + i] =
                    &rem.coeffs[rem_deg - divisor_deg + i] - &factor * &other.coeffs[i];
            }

            rem.normalize();
        }

        rem
    }
}

/// Configuration for root counting.
#[derive(Debug, Clone)]
pub struct RootCountConfig {
    /// Use Sturm sequences (exact but slower).
    pub use_sturm: bool,
    /// Use Descartes' rule (fast but gives upper bound).
    pub use_descartes: bool,
    /// Use Budan's theorem.
    pub use_budan: bool,
}

impl Default for RootCountConfig {
    fn default() -> Self {
        Self {
            use_sturm: true,
            use_descartes: true,
            use_budan: false,
        }
    }
}

/// Statistics for root counting.
#[derive(Debug, Clone, Default)]
pub struct RootCountStats {
    /// Roots counted.
    pub roots_counted: u64,
    /// Sturm sequences computed.
    pub sturm_sequences: u64,
    /// Descartes sign variations computed.
    pub descartes_variations: u64,
}

/// Root counting engine.
#[derive(Debug)]
pub struct RootCounter {
    /// Configuration.
    config: RootCountConfig,
    /// Statistics.
    stats: RootCountStats,
}

impl RootCounter {
    /// Create a new root counter.
    pub fn new(config: RootCountConfig) -> Self {
        Self {
            config,
            stats: RootCountStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(RootCountConfig::default())
    }

    /// Count real roots in interval (a, b) using Sturm's theorem.
    pub fn count_roots_sturm(
        &mut self,
        poly: &Polynomial,
        a: &BigRational,
        b: &BigRational,
    ) -> usize {
        if !self.config.use_sturm {
            return 0;
        }

        self.stats.sturm_sequences += 1;

        let sturm_seq = self.sturm_sequence(poly);

        let sign_changes_a = self.count_sign_changes(&sturm_seq, a);
        let sign_changes_b = self.count_sign_changes(&sturm_seq, b);

        (sign_changes_a as isize - sign_changes_b as isize).unsigned_abs()
    }

    /// Compute Sturm sequence for a polynomial.
    fn sturm_sequence(&self, poly: &Polynomial) -> Vec<Polynomial> {
        let mut seq = vec![poly.clone(), poly.derivative()];

        loop {
            let n = seq.len();
            let remainder = seq[n - 2].remainder(&seq[n - 1]);

            // Negate remainder (Sturm sequence convention)
            let negated = Polynomial::new(remainder.coeffs.iter().map(|c| -c).collect());

            if negated.degree() == 0 && negated.coeffs[0].is_zero() {
                break;
            }

            seq.push(negated);

            // Prevent infinite loops
            if seq.len() > 1000 {
                break;
            }
        }

        seq
    }

    /// Count sign changes in sequence at point x.
    fn count_sign_changes(&self, seq: &[Polynomial], x: &BigRational) -> usize {
        let mut last_sign = 0i8;
        let mut changes = 0;

        for poly in seq {
            let val = poly.eval(x);
            let sign = if val.is_positive() {
                1
            } else if val.is_negative() {
                -1
            } else {
                0
            };

            if sign != 0 && last_sign != 0 && sign != last_sign {
                changes += 1;
            }

            if sign != 0 {
                last_sign = sign;
            }
        }

        changes
    }

    /// Count positive roots using Descartes' rule of signs.
    pub fn count_positive_roots_descartes(&mut self, poly: &Polynomial) -> usize {
        if !self.config.use_descartes {
            return 0;
        }

        self.stats.descartes_variations += 1;

        let mut last_sign = 0i8;
        let mut variations = 0;

        for coeff in &poly.coeffs {
            let sign = if coeff.is_positive() {
                1
            } else if coeff.is_negative() {
                -1
            } else {
                0
            };

            if sign != 0 && last_sign != 0 && sign != last_sign {
                variations += 1;
            }

            if sign != 0 {
                last_sign = sign;
            }
        }

        variations
    }

    /// Get statistics.
    pub fn stats(&self) -> &RootCountStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = RootCountStats::default();
    }
}

impl Default for RootCounter {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn test_polynomial_creation() {
        let poly = Polynomial::new(vec![
            BigRational::from_integer(1.into()),
            BigRational::from_integer(2.into()),
            BigRational::from_integer(3.into()),
        ]);
        assert_eq!(poly.degree(), 2);
    }

    #[test]
    fn test_polynomial_eval() {
        // p(x) = 1 + 2x + 3x^2
        let poly = Polynomial::new(vec![
            BigRational::from_integer(1.into()),
            BigRational::from_integer(2.into()),
            BigRational::from_integer(3.into()),
        ]);

        // p(0) = 1
        let val = poly.eval(&BigRational::zero());
        assert_eq!(val, BigRational::from_integer(1.into()));

        // p(1) = 1 + 2 + 3 = 6
        let val = poly.eval(&BigRational::one());
        assert_eq!(val, BigRational::from_integer(6.into()));
    }

    #[test]
    fn test_polynomial_derivative() {
        // p(x) = 1 + 2x + 3x^2
        let poly = Polynomial::new(vec![
            BigRational::from_integer(1.into()),
            BigRational::from_integer(2.into()),
            BigRational::from_integer(3.into()),
        ]);

        // p'(x) = 2 + 6x
        let deriv = poly.derivative();
        assert_eq!(deriv.degree(), 1);
        assert_eq!(deriv.coeffs[0], BigRational::from_integer(2.into()));
        assert_eq!(deriv.coeffs[1], BigRational::from_integer(6.into()));
    }

    #[test]
    fn test_descartes_rule() {
        let mut counter = RootCounter::default_config();

        // p(x) = x^2 - 1 (roots at ±1, one positive root)
        let poly = Polynomial::new(vec![
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::from_integer(1.into()),
        ]);

        let count = counter.count_positive_roots_descartes(&poly);
        assert!(count >= 1); // Upper bound
    }

    #[test]
    fn test_counter_creation() {
        let counter = RootCounter::default_config();
        assert_eq!(counter.stats().sturm_sequences, 0);
    }

    #[test]
    fn test_stats() {
        let mut counter = RootCounter::default_config();
        counter.stats.sturm_sequences = 5;

        assert_eq!(counter.stats().sturm_sequences, 5);

        counter.reset_stats();
        assert_eq!(counter.stats().sturm_sequences, 0);
    }
}
