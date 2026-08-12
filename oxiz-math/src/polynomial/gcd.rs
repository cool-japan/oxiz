//! Polynomial GCD Algorithms.
//!
//! This module implements efficient algorithms for computing the greatest
//! common divisor (GCD) of multivariate polynomials.
//!
//! ## Algorithms
//!
//! 1. **Euclidean Algorithm**: Classic algorithm for univariate polynomials
//! 2. **Subresultant PRS**: Polynomial remainder sequences avoiding coefficient growth
//! 3. **Modular GCD**: Compute GCD mod p, then lift to full precision
//! 4. **Sparse GCD**: Optimized for sparse polynomials
//!
//! ## Applications
//!
//! - Simplification of rational functions
//! - Factorization preprocessing
//! - Solving polynomial systems
//! - Gröbner basis computation
//!
//! ## References
//!
//! - Knuth: "The Art of Computer Programming Vol. 2" (GCD algorithms)
//! - Geddes et al.: "Algorithms for Computer Algebra" (1992)
//! - Z3's `math/polynomial/polynomial_gcd.cpp`

use super::{NULL_VAR, Polynomial, Var};
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::BigRational;
use num_traits::{One, Zero};

/// Combine two "maximum variable" values (as returned by
/// [`Polynomial::max_var`], where [`NULL_VAR`] means "no variables /
/// constant") the way a remainder/division computation needs: the shared
/// reduction variable is the higher-indexed *actual* variable appearing in
/// either operand, and is `NULL_VAR` only when *both* operands are
/// constant.
///
/// A naive `a.max(b)` is wrong here: `NULL_VAR == u32::MAX`, so it would
/// always "win" over any real variable index, e.g. `combined_var(0,
/// NULL_VAR)` would incorrectly report "no shared variable" even though one
/// operand has variable 0. (`Polynomial::gcd_univariate`'s use of a plain
/// `.max()` is safe only because *its* early-return for `NULL_VAR` --
/// "either operand is a nonzero constant, so GCD is 1" -- happens to be
/// correct regardless of which operand was constant; a remainder/quotient
/// computation is not symmetric like that, so it needs this precise
/// version.)
fn combined_var(a: Var, b: Var) -> Var {
    match (a, b) {
        (NULL_VAR, NULL_VAR) => NULL_VAR,
        (NULL_VAR, v) | (v, NULL_VAR) => v,
        (va, vb) => va.max(vb),
    }
}

/// Configuration for GCD computation.
#[derive(Debug, Clone)]
pub struct GcdConfig {
    /// Use modular GCD algorithm.
    pub use_modular: bool,
    /// Use subresultant PRS.
    pub use_subresultant: bool,
    /// Modulus for modular algorithm.
    pub modulus: u64,
    /// Maximum degree to attempt.
    pub max_degree: u32,
}

impl Default for GcdConfig {
    fn default() -> Self {
        Self {
            use_modular: true,
            use_subresultant: true,
            modulus: 2147483647, // Large prime
            max_degree: 1000,
        }
    }
}

/// Statistics for GCD computation.
#[derive(Debug, Clone, Default)]
pub struct GcdStats {
    /// GCDs computed.
    pub gcds_computed: u64,
    /// Euclidean steps performed.
    pub euclidean_steps: u64,
    /// Modular reductions performed.
    pub modular_reductions: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Polynomial GCD engine.
pub struct PolynomialGcd {
    config: GcdConfig,
    stats: GcdStats,
}

impl PolynomialGcd {
    /// Create new GCD engine.
    pub fn new() -> Self {
        Self::with_config(GcdConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: GcdConfig) -> Self {
        Self {
            config,
            stats: GcdStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &GcdStats {
        &self.stats
    }

    /// Compute GCD of two polynomials.
    pub fn gcd(&mut self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        #[cfg(feature = "std")]
        let start = oxiz_time::Instant::now();

        // Handle special cases
        if a.is_zero() {
            self.stats.gcds_computed += 1;
            return b.clone();
        }
        if b.is_zero() {
            self.stats.gcds_computed += 1;
            return a.clone();
        }

        // Check degrees
        if a.total_degree() > self.config.max_degree || b.total_degree() > self.config.max_degree {
            // Too large - return trivial GCD
            self.stats.gcds_computed += 1;
            return Polynomial::constant(BigRational::one());
        }

        // Use Euclidean algorithm for simple cases
        let result = if a.total_degree() < 10 && b.total_degree() < 10 {
            self.euclidean_gcd(a, b)
        } else if self.config.use_modular {
            self.modular_gcd(a, b)
        } else {
            self.euclidean_gcd(a, b)
        };

        self.stats.gcds_computed += 1;
        #[cfg(feature = "std")]
        {
            self.stats.time_us += start.elapsed().as_micros() as u64;
        }

        result
    }

    /// Euclidean algorithm for polynomial GCD.
    ///
    /// Based on repeated polynomial division until remainder is zero.
    fn euclidean_gcd(&mut self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        let mut r0 = a.clone();
        let mut r1 = b.clone();

        // Bound iterations for safety. For genuinely univariate inputs each
        // step strictly reduces the degree in the shared variable, so this
        // bound is never the limiting factor there. For inputs that mix
        // multiple distinct variables -- where `polynomial_remainder`'s
        // "univariate in one designated variable" approach (see its docs)
        // cannot always make progress -- the sequence can settle into a
        // fixed 2-cycle instead of shrinking toward zero; this bound
        // guarantees the loop still terminates rather than hanging.
        let max_iters = a.total_degree() as usize + b.total_degree() as usize + 16;
        let mut iters = 0;

        while !r1.is_zero() && iters < max_iters {
            iters += 1;
            self.stats.euclidean_steps += 1;

            // Compute r0 mod r1 (polynomial remainder)
            let remainder = self.polynomial_remainder(&r0, &r1);

            r0 = r1;
            r1 = remainder;
        }

        // Normalize by making leading coefficient positive
        self.normalize_polynomial(r0)
    }

    /// Modular GCD algorithm.
    ///
    /// 1. Compute GCD mod p for a prime p
    /// 2. Lift to full precision
    fn modular_gcd(&mut self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        self.stats.modular_reductions += 1;

        // Simplified implementation - full version would:
        // 1. Reduce polynomials modulo p
        // 2. Compute GCD in Z_p[x]
        // 3. Lift using Hensel lifting or Chinese remainder theorem

        // For now, fall back to Euclidean
        self.euclidean_gcd(a, b)
    }

    /// Compute polynomial remainder: a mod b (exact rational polynomial long
    /// division, not the placeholder that used to always return zero).
    ///
    /// Division by the zero polynomial is mathematically undefined; rather
    /// than panicking on a crafted degenerate `b` (see finding R3), this
    /// treats it as "no reduction possible" and returns `a` unchanged.
    ///
    /// # Multivariate limitation
    ///
    /// This performs *univariate* long division in the highest-indexed
    /// variable appearing in either operand, reusing
    /// [`Polynomial::exact_div_rem_univariate`]. It is exact when `a` and
    /// `b` are both univariate in that variable. For genuinely multivariate
    /// polynomials (terms mixing two or more distinct variables), the
    /// result is not guaranteed to be the true multivariate remainder --
    /// see [`Polynomial::exact_div_rem_univariate`]'s docs for why. Callers
    /// that need an exact n-variate remainder should use
    /// [`super::gcd_multivariate`] / [`super::gcd_multivariate_advanced`]
    /// instead.
    fn polynomial_remainder(&self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        if b.is_zero() {
            return a.clone();
        }
        if a.is_zero() {
            return Polynomial::zero();
        }

        let var = combined_var(a.max_var(), b.max_var());
        if var == NULL_VAR {
            // Both operands are non-zero constants: over a field, a mod b
            // is always exactly 0.
            return Polynomial::zero();
        }

        a.exact_div_rem_univariate(b, var).1
    }

    /// Normalize polynomial (make leading coefficient = 1, i.e. monic).
    fn normalize_polynomial(&self, poly: Polynomial) -> Polynomial {
        if poly.is_zero() {
            return poly;
        }

        let leading_coeff = poly.leading_coeff();
        if leading_coeff.is_one() || leading_coeff.is_zero() {
            return poly;
        }

        // Divide every coefficient by the leading coefficient (exact over
        // BigRational) so the result is monic, the standard GCD
        // normalization convention.
        poly.scale(&leading_coeff.recip())
    }

    /// Compute GCD of multiple polynomials.
    pub fn gcd_multiple(&mut self, polys: &[Polynomial]) -> Polynomial {
        if polys.is_empty() {
            return Polynomial::zero();
        }

        let mut result = polys[0].clone();

        for poly in &polys[1..] {
            result = self.gcd(&result, poly);

            // Early termination if GCD becomes 1
            if result.total_degree() == 0 && !result.is_zero() {
                break;
            }
        }

        result
    }

    /// Compute LCM (least common multiple) of two polynomials.
    ///
    /// LCM(a,b) = (a * b) / GCD(a,b), computed via exact polynomial
    /// division (previously this omitted the division entirely and
    /// returned the plain product `a * b`).
    pub fn lcm(&mut self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        if a.is_zero() || b.is_zero() {
            return Polynomial::zero();
        }

        let gcd = self.gcd(a, b);

        if gcd.is_zero() {
            return Polynomial::zero();
        }

        let product = a.mul(b);
        let var = combined_var(product.max_var(), gcd.max_var());
        if var == NULL_VAR {
            // Both product and gcd are constants: LCM of nonzero constants
            // is conventionally 1 (any nonzero constant divides any other).
            return Polynomial::one();
        }

        let (quotient, _remainder) = product.exact_div_rem_univariate(&gcd, var);
        quotient
    }
}

impl Default for PolynomialGcd {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    #[test]
    fn test_gcd_creation() {
        let gcd_engine = PolynomialGcd::new();
        assert_eq!(gcd_engine.stats().gcds_computed, 0);
    }

    #[test]
    fn test_gcd_zero() {
        let mut gcd_engine = PolynomialGcd::new();

        let a = Polynomial::constant(BigRational::from(BigInt::from(42)));
        let b = Polynomial::zero();

        let result = gcd_engine.gcd(&a, &b);

        // GCD of a polynomial with zero is the non-zero polynomial
        assert!(!result.is_zero());
        assert_eq!(gcd_engine.stats().gcds_computed, 1);
    }

    #[test]
    fn test_gcd_constants() {
        let mut gcd_engine = PolynomialGcd::new();

        let a = Polynomial::constant(BigRational::from(BigInt::from(12)));
        let b = Polynomial::constant(BigRational::from(BigInt::from(18)));

        let result = gcd_engine.gcd(&a, &b);

        assert!(!result.is_zero());
    }

    #[test]
    fn test_gcd_config() {
        let config = GcdConfig {
            use_modular: false,
            use_subresultant: false,
            ..Default::default()
        };

        let gcd_engine = PolynomialGcd::with_config(config);
        assert!(!gcd_engine.config.use_modular);
    }

    #[test]
    fn test_gcd_multiple() {
        let mut gcd_engine = PolynomialGcd::new();

        let polys = vec![
            Polynomial::constant(BigRational::from(BigInt::from(12))),
            Polynomial::constant(BigRational::from(BigInt::from(18))),
            Polynomial::constant(BigRational::from(BigInt::from(24))),
        ];

        let result = gcd_engine.gcd_multiple(&polys);

        assert!(!result.is_zero());
    }

    #[test]
    fn test_lcm() {
        let mut gcd_engine = PolynomialGcd::new();

        let a = Polynomial::constant(BigRational::from(BigInt::from(4)));
        let b = Polynomial::constant(BigRational::from(BigInt::from(6)));

        let result = gcd_engine.lcm(&a, &b);

        assert!(!result.is_zero());
    }

    // -- Regression tests for MATH-2 (PolynomialGcd was a broken stub) --

    /// Build the univariate polynomial `x^2` (variable 0).
    fn poly_x_squared() -> Polynomial {
        Polynomial::from_coeffs_int(&[(1, &[(0, 2)])])
    }

    /// Build the univariate polynomial `x + 1` (variable 0).
    fn poly_x_plus_1() -> Polynomial {
        Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])])
    }

    /// Build the univariate polynomial `x - 1` (variable 0).
    fn poly_x_minus_1() -> Polynomial {
        Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])])
    }

    #[test]
    fn test_polynomial_remainder_is_real_division_not_stub() {
        // Regression test for MATH-2: `polynomial_remainder` used to
        // unconditionally return `Polynomial::zero()` whenever
        // `deg(a) >= deg(b)`. x^2 mod (x+1) is the constant 1 (long
        // division: x^2 = (x-1)(x+1) + 1), never 0.
        let gcd_engine = PolynomialGcd::new();
        let a = poly_x_squared();
        let b = poly_x_plus_1();

        let remainder = gcd_engine.polynomial_remainder(&a, &b);

        assert!(!remainder.is_zero(), "stub would wrongly return zero here");
        assert!(remainder.is_constant());
        assert_eq!(remainder.constant_value(), BigRational::one());
    }

    #[test]
    fn test_gcd_coprime_polynomials_is_a_unit() {
        // Regression test for MATH-2: gcd(x^2, x+1) must be a nonzero
        // constant (they are coprime). The old stub's broken
        // `polynomial_remainder` made `euclidean_gcd` terminate after one
        // step and return `x+1` -- a non-unit -- as if it were the GCD.
        let mut gcd_engine = PolynomialGcd::new();
        let a = poly_x_squared();
        let b = poly_x_plus_1();

        let result = gcd_engine.gcd(&a, &b);

        assert!(!result.is_zero());
        assert_eq!(
            result.total_degree(),
            0,
            "gcd of coprime polynomials must be a nonzero constant, got {result:?}"
        );
    }

    #[test]
    fn test_gcd_shared_linear_factor() {
        // gcd((x-1)(x+1), (x-1)(x+2)) == (a unit multiple of) x-1.
        let mut gcd_engine = PolynomialGcd::new();
        let a = poly_x_minus_1().mul(&poly_x_plus_1()); // x^2 - 1
        let b = poly_x_minus_1().mul(&Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (2, &[])])); // x^2 + x - 2

        let result = gcd_engine.gcd(&a, &b);

        assert_eq!(result.total_degree(), 1, "expected a linear common factor");
        // The result must exactly divide both inputs (remainder zero).
        assert!(gcd_engine.polynomial_remainder(&a, &result).is_zero());
        assert!(gcd_engine.polynomial_remainder(&b, &result).is_zero());
        // Normalization makes it monic: leading coefficient exactly 1.
        assert_eq!(result.leading_coeff(), BigRational::one());
    }

    #[test]
    fn test_normalize_polynomial_makes_monic() {
        // Regression test for MATH-2: `normalize_polynomial`'s coefficient
        // loop used to be an empty no-op, so `2x + 4` stayed `2x + 4`
        // instead of becoming monic `x + 2`.
        let gcd_engine = PolynomialGcd::new();
        let poly = Polynomial::from_coeffs_int(&[(2, &[(0, 1)]), (4, &[])]); // 2x + 4

        let normalized = gcd_engine.normalize_polynomial(poly);

        assert_eq!(normalized.leading_coeff(), BigRational::one());
        assert_eq!(
            normalized,
            Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (2, &[])]) // x + 2
        );
    }

    #[test]
    fn test_normalize_polynomial_zero_and_already_monic_unchanged() {
        let gcd_engine = PolynomialGcd::new();

        let zero = Polynomial::zero();
        assert!(gcd_engine.normalize_polynomial(zero).is_zero());

        let monic = poly_x_plus_1();
        assert_eq!(gcd_engine.normalize_polynomial(monic.clone()), monic);
    }

    #[test]
    fn test_lcm_of_coprime_polynomials_equals_product() {
        // Regression test for MATH-2: `lcm` used to just return `a * b`
        // unconditionally (never dividing by the GCD). For coprime a, b
        // this happens to coincide with the correct answer, so exercise it
        // to confirm the *new* division-based implementation still gets
        // the right answer here too.
        let mut gcd_engine = PolynomialGcd::new();
        let a = poly_x_minus_1(); // x - 1
        let b = poly_x_plus_1(); // x + 1

        let result = gcd_engine.lcm(&a, &b);

        assert_eq!(
            result,
            Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])])
        ); // x^2 - 1
    }

    #[test]
    fn test_lcm_divides_product_by_true_gcd() {
        // lcm(a, b) * gcd(a, b) must equal a * b up to a unit factor; check
        // this via exact divisibility instead of a hard-coded polynomial,
        // to catch a regression to the old "just multiply" stub for an
        // input with a nontrivial common factor.
        let mut gcd_engine = PolynomialGcd::new();
        let a = poly_x_minus_1().mul(&poly_x_plus_1()); // x^2 - 1
        let b = poly_x_minus_1().mul(&Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (2, &[])])); // x^2 + x - 2

        let lcm = gcd_engine.lcm(&a, &b);
        let product = a.mul(&b);

        // lcm must be strictly smaller-degree than the naive product
        // whenever a, b share a nontrivial factor (product has degree 4,
        // lcm should have degree 3: (x-1)(x+1)(x+2)).
        assert_eq!(lcm.total_degree(), 3);
        assert!(gcd_engine.polynomial_remainder(&product, &lcm).is_zero());
    }

    #[test]
    fn test_polynomial_remainder_zero_divisor_no_panic() {
        // Regression test for R3: division by the zero polynomial must not
        // panic; `polynomial_remainder` treats it as "no reduction
        // possible" and returns the dividend unchanged.
        let gcd_engine = PolynomialGcd::new();
        let a = poly_x_plus_1();
        let zero = Polynomial::zero();

        let remainder = gcd_engine.polynomial_remainder(&a, &zero);
        assert_eq!(remainder, a);
    }

    #[test]
    fn test_gcd_disjoint_variables_terminates() {
        // Regression test: `euclidean_gcd` must terminate (not hang) even
        // for polynomials in entirely different variables, where the
        // univariate-in-one-variable remainder computation cannot always
        // reduce the degree (see `polynomial_remainder`'s documented
        // multivariate limitation). This is guarded by an iteration bound;
        // this test simply confirms the call returns instead of looping
        // forever.
        let mut gcd_engine = PolynomialGcd::new();
        let a = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]); // x0 + 1
        let b = Polynomial::from_coeffs_int(&[(1, &[(1, 1)]), (1, &[])]); // x1 + 1

        let _ = gcd_engine.gcd(&a, &b); // must return, not hang
    }

    #[test]
    fn test_gcd_constant_dividend_nonconstant_divisor_remainder() {
        // Regression test for the `combined_var` fix: when the *dividend*
        // is a nonzero constant and the divisor is non-constant, the
        // remainder must be the dividend itself (deg(dividend) <
        // deg(divisor)), not incorrectly forced to zero by treating a
        // one-sided constant as "both operands are constant".
        let gcd_engine = PolynomialGcd::new();
        let five = Polynomial::constant(BigRational::from(BigInt::from(5)));
        let b = poly_x_plus_1();

        let remainder = gcd_engine.polynomial_remainder(&five, &b);
        assert_eq!(remainder, five);
    }
}
