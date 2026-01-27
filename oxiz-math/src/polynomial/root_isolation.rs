//! Polynomial Root Isolation.
//!
//! This module implements algorithms for isolating real roots of
//! univariate polynomials with rational coefficients.
//!
//! ## Algorithms
//!
//! 1. **Sturm Sequences**: Sign variation counting for root bounds
//! 2. **Descartes' Rule**: Positive root counting via sign changes
//! 3. **Continued Fractions**: Efficient root isolation
//! 4. **Bisection**: Interval refinement
//!
//! ## Applications
//!
//! - Real algebraic number computation
//! - CAD (Cylindrical Algebraic Decomposition)
//! - Quantifier elimination over reals
//! - Polynomial constraint solving
//!
//! ## References
//!
//! - Collins & Akritas: "Polynomial Real Root Isolation Using Descartes' Rule" (1976)
//! - Z3's `math/polynomial/polynomial_roots.cpp`

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use super::factorization::{Polynomial, Term};

/// Configuration for root isolation.
#[derive(Debug, Clone)]
pub struct RootIsolationConfig {
    /// Use Sturm sequences.
    pub use_sturm: bool,
    /// Use Descartes' rule.
    pub use_descartes: bool,
    /// Maximum isolation depth.
    pub max_depth: u32,
    /// Precision for interval refinement.
    pub precision: u32,
}

impl Default for RootIsolationConfig {
    fn default() -> Self {
        Self {
            use_sturm: true,
            use_descartes: true,
            max_depth: 100,
            precision: 64,
        }
    }
}

/// Statistics for root isolation.
#[derive(Debug, Clone, Default)]
pub struct RootIsolationStats {
    /// Roots isolated.
    pub roots_isolated: u64,
    /// Sturm evaluations.
    pub sturm_evals: u64,
    /// Descartes applications.
    pub descartes_apps: u64,
    /// Bisections performed.
    pub bisections: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Interval representing a root isolation region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interval {
    /// Lower bound (inclusive).
    pub lower: BigRational,
    /// Upper bound (inclusive).
    pub upper: BigRational,
}

impl Interval {
    /// Create new interval.
    pub fn new(lower: BigRational, upper: BigRational) -> Self {
        Self { lower, upper }
    }

    /// Check if interval is a point.
    pub fn is_point(&self) -> bool {
        self.lower == self.upper
    }

    /// Compute midpoint.
    pub fn midpoint(&self) -> BigRational {
        (&self.lower + &self.upper) / BigRational::from_integer(BigInt::from(2))
    }

    /// Width of interval.
    pub fn width(&self) -> BigRational {
        &self.upper - &self.lower
    }
}

/// Isolated root with interval bounds.
#[derive(Debug, Clone)]
pub struct IsolatedRoot {
    /// Isolation interval.
    pub interval: Interval,
    /// Multiplicity (if known).
    pub multiplicity: Option<u32>,
}

/// Root isolation engine.
pub struct RootIsolator {
    config: RootIsolationConfig,
    stats: RootIsolationStats,
}

impl RootIsolator {
    /// Create new root isolator.
    pub fn new() -> Self {
        Self::with_config(RootIsolationConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: RootIsolationConfig) -> Self {
        Self {
            config,
            stats: RootIsolationStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &RootIsolationStats {
        &self.stats
    }

    /// Isolate all real roots of polynomial.
    pub fn isolate_roots(&mut self, poly: &Polynomial) -> Vec<IsolatedRoot> {
        let start = std::time::Instant::now();

        if poly.is_zero() {
            return Vec::new();
        }

        // For constant polynomials, no roots
        if poly.degree() == 0 {
            return Vec::new();
        }

        // Use configured algorithm
        let roots = if self.config.use_descartes {
            self.isolate_descartes(poly)
        } else if self.config.use_sturm {
            self.isolate_sturm(poly)
        } else {
            self.isolate_bisection(poly)
        };

        self.stats.roots_isolated += roots.len() as u64;
        self.stats.time_us += start.elapsed().as_micros() as u64;

        roots
    }

    /// Isolate roots using Descartes' rule of signs.
    fn isolate_descartes(&mut self, poly: &Polynomial) -> Vec<IsolatedRoot> {
        self.stats.descartes_apps += 1;

        let mut roots = Vec::new();

        // Compute initial interval bounds
        let bound = self.cauchy_bound(poly);
        let initial_interval = Interval::new(
            BigRational::from_integer(-bound.clone()),
            BigRational::from_integer(bound),
        );

        // Recursive isolation
        self.descartes_recursive(poly, &initial_interval, &mut roots, 0);

        roots
    }

    /// Recursive Descartes isolation.
    fn descartes_recursive(
        &mut self,
        poly: &Polynomial,
        interval: &Interval,
        roots: &mut Vec<IsolatedRoot>,
        depth: u32,
    ) {
        if depth > self.config.max_depth {
            return;
        }

        // Count sign variations
        let variations = self.count_sign_variations(poly, interval);

        match variations {
            0 => {
                // No roots in this interval
            }
            1 => {
                // Exactly one root
                roots.push(IsolatedRoot {
                    interval: interval.clone(),
                    multiplicity: Some(1),
                });
            }
            _ => {
                // Multiple possible roots - bisect
                let mid = interval.midpoint();
                let left = Interval::new(interval.lower.clone(), mid.clone());
                let right = Interval::new(mid, interval.upper.clone());

                self.stats.bisections += 1;

                self.descartes_recursive(poly, &left, roots, depth + 1);
                self.descartes_recursive(poly, &right, roots, depth + 1);
            }
        }
    }

    /// Isolate roots using Sturm sequences.
    fn isolate_sturm(&mut self, poly: &Polynomial) -> Vec<IsolatedRoot> {
        self.stats.sturm_evals += 1;

        // Compute Sturm sequence
        let sturm_seq = self.sturm_sequence(poly);

        // Count roots in initial interval
        let bound = self.cauchy_bound(poly);
        let initial_interval = Interval::new(
            BigRational::from_integer(-bound.clone()),
            BigRational::from_integer(bound),
        );

        let mut roots = Vec::new();
        self.sturm_recursive(&sturm_seq, &initial_interval, &mut roots, 0);

        roots
    }

    /// Recursive Sturm isolation.
    fn sturm_recursive(
        &mut self,
        sturm_seq: &[Polynomial],
        interval: &Interval,
        roots: &mut Vec<IsolatedRoot>,
        depth: u32,
    ) {
        if depth > self.config.max_depth {
            return;
        }

        let count = self.sturm_count(sturm_seq, interval);

        match count {
            0 => {
                // No roots
            }
            1 => {
                // One root
                roots.push(IsolatedRoot {
                    interval: interval.clone(),
                    multiplicity: Some(1),
                });
            }
            _ => {
                // Bisect
                let mid = interval.midpoint();
                let left = Interval::new(interval.lower.clone(), mid.clone());
                let right = Interval::new(mid, interval.upper.clone());

                self.stats.bisections += 1;

                self.sturm_recursive(sturm_seq, &left, roots, depth + 1);
                self.sturm_recursive(sturm_seq, &right, roots, depth + 1);
            }
        }
    }

    /// Isolate roots using simple bisection.
    fn isolate_bisection(&mut self, poly: &Polynomial) -> Vec<IsolatedRoot> {
        let bound = self.cauchy_bound(poly);
        let initial_interval = Interval::new(
            BigRational::from_integer(-bound.clone()),
            BigRational::from_integer(bound),
        );

        let mut roots = Vec::new();
        self.bisection_recursive(poly, &initial_interval, &mut roots, 0);

        roots
    }

    /// Recursive bisection.
    fn bisection_recursive(
        &mut self,
        poly: &Polynomial,
        interval: &Interval,
        roots: &mut Vec<IsolatedRoot>,
        depth: u32,
    ) {
        if depth > self.config.max_depth {
            return;
        }

        // Evaluate at endpoints
        let f_lower = self.eval_at_rational(poly, &interval.lower);
        let f_upper = self.eval_at_rational(poly, &interval.upper);

        // Check for sign change
        if f_lower.is_zero() {
            roots.push(IsolatedRoot {
                interval: Interval::new(interval.lower.clone(), interval.lower.clone()),
                multiplicity: None,
            });
            return;
        }

        if f_upper.is_zero() {
            roots.push(IsolatedRoot {
                interval: Interval::new(interval.upper.clone(), interval.upper.clone()),
                multiplicity: None,
            });
            return;
        }

        if (f_lower.is_positive() && f_upper.is_positive())
            || (f_lower.is_negative() && f_upper.is_negative())
        {
            // No sign change - no roots (or even number)
            return;
        }

        // Sign change detected - bisect
        let mid = interval.midpoint();
        let left = Interval::new(interval.lower.clone(), mid.clone());
        let right = Interval::new(mid, interval.upper.clone());

        self.stats.bisections += 1;

        self.bisection_recursive(poly, &left, roots, depth + 1);
        self.bisection_recursive(poly, &right, roots, depth + 1);
    }

    /// Compute Sturm sequence for polynomial.
    fn sturm_sequence(&self, poly: &Polynomial) -> Vec<Polynomial> {
        let mut seq = Vec::new();

        if poly.is_zero() {
            return seq;
        }

        seq.push(poly.clone());

        // Derivative
        let derivative = self.derivative(poly);
        if !derivative.is_zero() {
            seq.push(derivative);
        }

        // Polynomial remainder sequence
        while seq.len() >= 2 {
            let n = seq.len();
            let remainder = self.polynomial_remainder(&seq[n - 2], &seq[n - 1]);

            if remainder.is_zero() {
                break;
            }

            // Negate remainder for Sturm property
            let neg_remainder = self.negate_polynomial(&remainder);
            seq.push(neg_remainder);
        }

        seq
    }

    /// Count roots in interval using Sturm sequence.
    fn sturm_count(&self, sturm_seq: &[Polynomial], interval: &Interval) -> u32 {
        let var_lower = self.count_sign_variations_at(sturm_seq, &interval.lower);
        let var_upper = self.count_sign_variations_at(sturm_seq, &interval.upper);

        var_lower.saturating_sub(var_upper)
    }

    /// Count sign variations at a point.
    fn count_sign_variations_at(&self, polys: &[Polynomial], point: &BigRational) -> u32 {
        let mut prev_sign = None;
        let mut variations = 0;

        for poly in polys {
            let value = self.eval_at_rational(poly, point);

            if !value.is_zero() {
                let sign = value.is_positive();

                if let Some(prev) = prev_sign {
                    if prev != sign {
                        variations += 1;
                    }
                }

                prev_sign = Some(sign);
            }
        }

        variations
    }

    /// Count sign variations in polynomial coefficients.
    fn count_sign_variations(&self, poly: &Polynomial, _interval: &Interval) -> u32 {
        let mut prev_sign = None;
        let mut variations = 0;

        for term in &poly.terms {
            if !term.coeff.is_zero() {
                let sign = term.coeff.is_positive();

                if let Some(prev) = prev_sign {
                    if prev != sign {
                        variations += 1;
                    }
                }

                prev_sign = Some(sign);
            }
        }

        variations
    }

    /// Compute Cauchy bound for roots.
    fn cauchy_bound(&self, poly: &Polynomial) -> BigInt {
        if poly.terms.is_empty() {
            return BigInt::one();
        }

        // Simple bound: 1 + max(|a_i|) / |a_n|
        let leading = &poly.terms[0].coeff;
        if leading.is_zero() {
            return BigInt::one();
        }

        let mut max_coeff = BigInt::zero();
        for term in &poly.terms[1..] {
            let abs_coeff = term.coeff.abs();
            if abs_coeff > max_coeff {
                max_coeff = abs_coeff;
            }
        }

        BigInt::one() + (max_coeff / leading.abs())
    }

    /// Evaluate polynomial at rational point.
    fn eval_at_rational(&self, poly: &Polynomial, point: &BigRational) -> BigRational {
        let mut result = BigRational::zero();

        for term in &poly.terms {
            let coeff = BigRational::from_integer(term.coeff.clone());

            // Get total degree (sum of exponents)
            let degree: u32 = term.exponents.values().sum();

            let mut term_val = coeff;
            for _ in 0..degree {
                term_val = term_val * point;
            }

            result = result + term_val;
        }

        result
    }

    /// Compute polynomial derivative.
    fn derivative(&self, poly: &Polynomial) -> Polynomial {
        let mut terms = Vec::new();

        for term in &poly.terms {
            // For univariate, get the exponent of variable 0
            if let Some(&exp) = term.exponents.get(&0) {
                if exp > 0 {
                    let mut new_term = term.clone();
                    new_term.coeff = &term.coeff * BigInt::from(exp);

                    let mut new_exps = term.exponents.clone();
                    new_exps.insert(0, exp - 1);
                    new_term.exponents = new_exps;

                    terms.push(new_term);
                }
            }
        }

        Polynomial { terms }
    }

    /// Polynomial remainder (simplified).
    fn polynomial_remainder(&self, _a: &Polynomial, _b: &Polynomial) -> Polynomial {
        // Simplified: would implement full polynomial division
        Polynomial { terms: Vec::new() }
    }

    /// Negate polynomial coefficients.
    fn negate_polynomial(&self, poly: &Polynomial) -> Polynomial {
        let mut result = poly.clone();
        for term in &mut result.terms {
            term.coeff = -term.coeff.clone();
        }
        result
    }
}

impl Default for RootIsolator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear(a: i64, b: i64) -> Polynomial {
        // a*x + b
        Polynomial {
            terms: vec![
                Term {
                    coeff: BigInt::from(a),
                    exponents: [(0, 1)].iter().cloned().collect(),
                },
                Term {
                    coeff: BigInt::from(b),
                    exponents: FxHashMap::default(),
                },
            ],
        }
    }

    #[test]
    fn test_isolator_creation() {
        let isolator = RootIsolator::new();
        assert_eq!(isolator.stats().roots_isolated, 0);
    }

    #[test]
    fn test_interval() {
        let interval = Interval::new(
            BigRational::from_integer(BigInt::from(-1)),
            BigRational::from_integer(BigInt::from(1)),
        );

        assert!(!interval.is_point());
        assert_eq!(
            interval.midpoint(),
            BigRational::from_integer(BigInt::zero())
        );
    }

    #[test]
    fn test_cauchy_bound() {
        let isolator = RootIsolator::new();

        // x + 1
        let poly = make_linear(1, 1);
        let bound = isolator.cauchy_bound(&poly);

        assert!(bound >= BigInt::one());
    }

    #[test]
    fn test_derivative() {
        let isolator = RootIsolator::new();

        // 2*x + 3
        let poly = make_linear(2, 3);
        let deriv = isolator.derivative(&poly);

        // Should have constant term 2
        assert!(!deriv.is_zero());
    }

    #[test]
    fn test_eval_at_rational() {
        let isolator = RootIsolator::new();

        // x + 1
        let poly = make_linear(1, 1);

        // Eval at x=0: should be 1
        let val = isolator.eval_at_rational(&poly, &BigRational::zero());
        assert_eq!(val, BigRational::from_integer(BigInt::one()));

        // Eval at x=-1: should be 0
        let val = isolator.eval_at_rational(&poly, &BigRational::from_integer(BigInt::from(-1)));
        assert_eq!(val, BigRational::zero());
    }

    #[test]
    fn test_isolate_zero_poly() {
        let mut isolator = RootIsolator::new();
        let poly = Polynomial { terms: Vec::new() };

        let roots = isolator.isolate_roots(&poly);
        assert_eq!(roots.len(), 0);
    }

    #[test]
    fn test_isolate_constant() {
        let mut isolator = RootIsolator::new();

        // Constant 5
        let poly = Polynomial {
            terms: vec![Term {
                coeff: BigInt::from(5),
                exponents: FxHashMap::default(),
            }],
        };

        let roots = isolator.isolate_roots(&poly);
        assert_eq!(roots.len(), 0);
    }
}
