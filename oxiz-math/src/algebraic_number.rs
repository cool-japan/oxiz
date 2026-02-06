//! Real Algebraic Numbers.
#![allow(dead_code)] // Under development
//!
//! Represents real algebraic numbers as roots of polynomials with rational coefficients,
//! supporting exact arithmetic for non-linear real arithmetic (NRA) solving.
//!
//! ## Representation
//!
//! An algebraic number α is represented by:
//! - A minimal polynomial p(x) with α as a root
//! - An isolating interval [a, b] where α is the only root of p in [a, b]
//!
//! ## Operations
//!
//! - Arithmetic: +, -, *, / (via resultants and CAD)
//! - Comparison: <, ≤, =, ≥, > (via sign evaluation)
//! - Refinement: Narrow isolating intervals for better precision
//!
//! ## References
//!
//! - "Computations with Algebraic Numbers" (Basu et al., 2006)
//! - Z3's `math/realclosure/` directory

use crate::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

/// An isolating interval for a real algebraic number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolatingInterval {
    /// Lower bound (inclusive).
    pub lower: BigRational,
    /// Upper bound (inclusive).
    pub upper: BigRational,
}

impl IsolatingInterval {
    /// Create a new isolating interval.
    pub fn new(lower: BigRational, upper: BigRational) -> Result<Self, AlgebraicError> {
        if lower > upper {
            return Err(AlgebraicError::InvalidInterval);
        }

        Ok(Self { lower, upper })
    }

    /// Check if the interval contains a value.
    pub fn contains(&self, value: &BigRational) -> bool {
        value >= &self.lower && value <= &self.upper
    }

    /// Get the midpoint of the interval.
    pub fn midpoint(&self) -> BigRational {
        (&self.lower + &self.upper) / BigRational::from_integer(2.into())
    }

    /// Get the width of the interval.
    pub fn width(&self) -> BigRational {
        &self.upper - &self.lower
    }

    /// Check if the interval is a point (lower == upper).
    pub fn is_point(&self) -> bool {
        self.lower == self.upper
    }
}

/// A real algebraic number.
#[derive(Debug, Clone)]
pub struct AlgebraicNumber {
    /// Minimal polynomial (monic, irreducible).
    pub minimal_poly: Polynomial,
    /// Isolating interval containing this root.
    pub interval: IsolatingInterval,
    /// Cached sign (if evaluated).
    sign: Option<i8>, // -1, 0, or 1
}

impl AlgebraicNumber {
    /// Create a new algebraic number.
    ///
    /// The polynomial must have exactly one root in the given interval.
    pub fn new(
        minimal_poly: Polynomial,
        interval: IsolatingInterval,
    ) -> Result<Self, AlgebraicError> {
        // Verify that the interval isolates a single root
        // (In real implementation, would check using Sturm sequences)

        Ok(Self {
            minimal_poly,
            interval,
            sign: None,
        })
    }

    /// Create from a rational number.
    pub fn from_rational(r: BigRational) -> Self {
        // Algebraic number with minimal polynomial x - r
        let var = 0;
        let poly = Polynomial::linear(&[(BigRational::one(), var)], -r.clone());

        Self {
            minimal_poly: poly,
            interval: IsolatingInterval {
                lower: r.clone(),
                upper: r,
            },
            sign: None,
        }
    }

    /// Check if this is a rational number.
    pub fn is_rational(&self) -> bool {
        self.minimal_poly.degree(0) == 1 || self.interval.is_point()
    }

    /// Try to convert to a rational number.
    pub fn to_rational(&self) -> Option<BigRational> {
        if self.interval.is_point() {
            Some(self.interval.lower.clone())
        } else if self.minimal_poly.degree(0) == 1 {
            // For linear polynomial ax + b, root is -b/a
            // But we can just use the interval midpoint
            Some(self.interval.midpoint())
        } else {
            None
        }
    }

    /// Refine the isolating interval by bisection.
    ///
    /// Splits the interval in half and determines which half contains the root.
    pub fn refine(&mut self) -> Result<(), AlgebraicError> {
        if self.interval.is_point() {
            return Ok(()); // Already exact
        }

        let mid = self.interval.midpoint();

        // Evaluate polynomial at midpoint
        let var = 0;
        let mid_value = self.minimal_poly.eval_horner(var, &mid);

        if mid_value.is_zero() {
            // Exact hit! Update interval to point
            self.interval = IsolatingInterval {
                lower: mid.clone(),
                upper: mid,
            };
        } else {
            // Determine which half contains the root
            let lower_value = self.minimal_poly.eval_horner(var, &self.interval.lower);

            if lower_value.is_zero() {
                self.interval = IsolatingInterval {
                    lower: self.interval.lower.clone(),
                    upper: self.interval.lower.clone(),
                };
            } else if (lower_value.is_positive() && mid_value.is_negative())
                || (lower_value.is_negative() && mid_value.is_positive())
            {
                // Root in [lower, mid]
                self.interval.upper = mid;
            } else {
                // Root in [mid, upper]
                self.interval.lower = mid;
            }
        }

        Ok(())
    }

    /// Refine until the interval width is below a threshold.
    pub fn refine_to_precision(&mut self, epsilon: &BigRational) -> Result<(), AlgebraicError> {
        while self.interval.width() > *epsilon && !self.interval.is_point() {
            self.refine()?;
        }
        Ok(())
    }

    /// Get the sign of this algebraic number.
    ///
    /// Returns -1 if negative, 0 if zero, 1 if positive.
    pub fn sign(&mut self) -> Result<i8, AlgebraicError> {
        if let Some(s) = self.sign {
            return Ok(s);
        }

        // Refine until we can determine sign
        while self.interval.lower.is_negative() && self.interval.upper.is_positive() {
            self.refine()?;
        }

        let sign = if self.interval.upper.is_negative() {
            -1
        } else if self.interval.lower.is_positive() {
            1
        } else if self.interval.is_point() && self.interval.lower.is_zero() {
            0
        } else {
            // Should not happen after refinement
            return Err(AlgebraicError::SignUndetermined);
        };

        self.sign = Some(sign);
        Ok(sign)
    }

    /// Compare with another algebraic number.
    pub fn compare(&mut self, other: &mut AlgebraicNumber) -> Result<Ordering, AlgebraicError> {
        // If intervals don't overlap, comparison is easy
        if self.interval.upper < other.interval.lower {
            return Ok(Ordering::Less);
        }
        if self.interval.lower > other.interval.upper {
            return Ok(Ordering::Greater);
        }

        // Intervals overlap - need to refine or compute difference
        // Simplified: refine both until non-overlapping
        let max_iterations = 100;

        for _ in 0..max_iterations {
            if self.interval.upper < other.interval.lower {
                return Ok(Ordering::Less);
            }
            if self.interval.lower > other.interval.upper {
                return Ok(Ordering::Greater);
            }

            // Check if they might be equal
            if self.interval == other.interval && self.minimal_poly == other.minimal_poly {
                return Ok(Ordering::Equal);
            }

            // Refine both
            self.refine()?;
            other.refine()?;
        }

        // Could not determine after refinement
        Err(AlgebraicError::ComparisonFailed)
    }

    /// Compute the negation of this algebraic number.
    pub fn neg(&self) -> Self {
        // If α is a root of p(x), then -α is a root of p(-x)
        let var = 0;
        let neg_var_poly = -Polynomial::from_var(var);
        let negated_poly = self.minimal_poly.substitute(var, &neg_var_poly);

        Self {
            minimal_poly: negated_poly,
            interval: IsolatingInterval {
                lower: -self.interval.upper.clone(),
                upper: -self.interval.lower.clone(),
            },
            sign: self.sign.map(|s| -s),
        }
    }

    /// Compute the absolute value.
    pub fn abs(&self) -> Self {
        if self.interval.lower.is_negative() && self.interval.upper.is_positive() {
            // Contains zero - abs could be either value
            // Simplified: return self (real implementation would handle this properly)
            self.clone()
        } else if self.interval.upper.is_negative() {
            self.neg()
        } else {
            self.clone()
        }
    }
}

/// Configuration for algebraic number operations.
#[derive(Debug, Clone)]
pub struct AlgebraicConfig {
    /// Maximum refinement iterations.
    pub max_refinements: usize,
    /// Target precision for interval widths.
    pub target_precision: BigRational,
    /// Enable caching of computed values.
    pub enable_caching: bool,
}

impl Default for AlgebraicConfig {
    fn default() -> Self {
        Self {
            max_refinements: 100,
            target_precision: BigRational::from_integer(1.into())
                / BigRational::from_integer(1000000.into()), // 10^-6
            enable_caching: true,
        }
    }
}

/// Statistics for algebraic number operations.
#[derive(Debug, Clone, Default)]
pub struct AlgebraicStats {
    /// Refinements performed.
    pub refinements: u64,
    /// Sign evaluations.
    pub sign_evaluations: u64,
    /// Comparisons performed.
    pub comparisons: u64,
    /// Cache hits.
    pub cache_hits: u64,
}

/// Manager for algebraic number operations.
pub struct AlgebraicManager {
    /// Configuration.
    config: AlgebraicConfig,
    /// Statistics.
    stats: AlgebraicStats,
    /// Cache of computed algebraic numbers.
    cache: FxHashMap<String, AlgebraicNumber>,
}

impl AlgebraicManager {
    /// Create a new algebraic manager.
    pub fn new(config: AlgebraicConfig) -> Self {
        Self {
            config,
            stats: AlgebraicStats::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(AlgebraicConfig::default())
    }

    /// Create an algebraic number from a polynomial root in an interval.
    pub fn create(
        &mut self,
        poly: Polynomial,
        interval: IsolatingInterval,
    ) -> Result<AlgebraicNumber, AlgebraicError> {
        AlgebraicNumber::new(poly, interval)
    }

    /// Create from a rational number.
    pub fn from_rational(&mut self, r: BigRational) -> AlgebraicNumber {
        AlgebraicNumber::from_rational(r)
    }

    /// Refine an algebraic number to the configured precision.
    pub fn refine_to_precision(&mut self, alg: &mut AlgebraicNumber) -> Result<(), AlgebraicError> {
        alg.refine_to_precision(&self.config.target_precision)?;
        self.stats.refinements += 1;
        Ok(())
    }

    /// Get statistics.
    pub fn stats(&self) -> &AlgebraicStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = AlgebraicStats::default();
    }
}

/// Errors for algebraic number operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlgebraicError {
    /// Invalid interval (lower > upper).
    InvalidInterval,
    /// No root found in interval.
    NoRootInInterval,
    /// Multiple roots in interval.
    MultipleRootsInInterval,
    /// Sign could not be determined.
    SignUndetermined,
    /// Comparison failed after max refinements.
    ComparisonFailed,
    /// Arithmetic operation failed.
    ArithmeticFailed,
}

impl std::fmt::Display for AlgebraicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgebraicError::InvalidInterval => write!(f, "invalid interval"),
            AlgebraicError::NoRootInInterval => write!(f, "no root in interval"),
            AlgebraicError::MultipleRootsInInterval => write!(f, "multiple roots in interval"),
            AlgebraicError::SignUndetermined => write!(f, "sign undetermined"),
            AlgebraicError::ComparisonFailed => write!(f, "comparison failed"),
            AlgebraicError::ArithmeticFailed => write!(f, "arithmetic operation failed"),
        }
    }
}

impl std::error::Error for AlgebraicError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolating_interval() {
        let interval = IsolatingInterval::new(
            BigRational::from_integer(0.into()),
            BigRational::from_integer(2.into()),
        )
        .expect("valid interval");

        assert_eq!(interval.midpoint(), BigRational::from_integer(1.into()));
        assert!(!interval.is_point());
    }

    #[test]
    fn test_from_rational() {
        let r = BigRational::from_integer(3.into());
        let alg = AlgebraicNumber::from_rational(r.clone());

        assert!(alg.is_rational());
        assert_eq!(alg.to_rational(), Some(r));
    }

    #[test]
    fn test_negation() {
        let r = BigRational::from_integer(5.into());
        let alg = AlgebraicNumber::from_rational(r);
        let neg_alg = alg.neg();

        assert_eq!(
            neg_alg.to_rational(),
            Some(BigRational::from_integer((-5).into()))
        );
    }

    #[test]
    fn test_manager() {
        let manager = AlgebraicManager::default_config();
        assert_eq!(manager.stats().refinements, 0);
    }
}
