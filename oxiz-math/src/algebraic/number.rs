//! Algebraic Number Representation.
//!
//! An algebraic number is a root of a polynomial with rational coefficients.
//! We represent it as (polynomial, isolating_interval) where the interval
//! contains exactly one root of the polynomial.
//!
//! ## Representation
//!
//! ```text
//! α = (p(x), [a, b]) where p(α) = 0 and p has exactly one root in [a, b]
//! ```
//!
//! ## References
//!
//! - "Algorithms in Real Algebraic Geometry" (Basu et al., 2006)
//! - Z3's `math/polynomial/algebraic_numbers.h`

use crate::polynomial::root_counting::Polynomial;
#[allow(unused_imports)]
use crate::prelude::*;
use core::cmp::Ordering;
use core::fmt;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Algebraic number error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlgebraicNumberError {
    /// Interval doesn't isolate a single root.
    NonIsolatingInterval,
    /// Polynomial has no roots in interval.
    NoRootInInterval,
    /// Polynomial is zero.
    ZeroPolynomial,
    /// Invalid operation.
    InvalidOperation(String),
}

impl fmt::Display for AlgebraicNumberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonIsolatingInterval => write!(f, "Interval doesn't isolate a single root"),
            Self::NoRootInInterval => write!(f, "No root in interval"),
            Self::ZeroPolynomial => write!(f, "Polynomial is zero"),
            Self::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl core::error::Error for AlgebraicNumberError {}

/// Algebraic number represented as a root of a univariate polynomial over Q.
#[derive(Debug, Clone)]
pub struct AlgebraicNumber {
    /// Minimal polynomial (irreducible, with rational coefficients).
    pub minimal_poly: Polynomial,
    /// Lower bound of the isolating interval.
    pub lower: BigRational,
    /// Upper bound of the isolating interval.
    pub upper: BigRational,
    /// Number of refinement steps performed (for termination detection).
    refinement_level: usize,
}

impl AlgebraicNumber {
    /// Create a new algebraic number.
    ///
    /// Validates that the interval `[lower, upper]` contains exactly one root
    /// of `poly` using Sturm's theorem before accepting it.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `poly` is the zero polynomial,
    /// - `lower > upper` (degenerate interval ordering), or
    /// - the Sturm sequence shows ≠ 1 root inside `(lower, upper)`.
    pub fn new(
        poly: Polynomial,
        lower: BigRational,
        upper: BigRational,
    ) -> Result<Self, AlgebraicNumberError> {
        // Reject the zero polynomial — it has infinitely many roots.
        if poly.degree() == 0 && poly.coeffs.first().is_none_or(Zero::is_zero) {
            return Err(AlgebraicNumberError::ZeroPolynomial);
        }

        if lower > upper {
            return Err(AlgebraicNumberError::NonIsolatingInterval);
        }

        // Degenerate (point) interval: check whether `lower` is a root.
        if lower == upper {
            if poly.eval(&lower).is_zero() {
                return Ok(Self {
                    minimal_poly: poly,
                    lower,
                    upper,
                    refinement_level: 0,
                });
            }
            return Err(AlgebraicNumberError::NoRootInInterval);
        }

        // Sturm sequence check: the interval must isolate exactly one root.
        let root_count = sturm_root_count(&poly, &lower, &upper);
        if root_count != 1 {
            return Err(AlgebraicNumberError::NonIsolatingInterval);
        }

        Ok(Self {
            minimal_poly: poly,
            lower,
            upper,
            refinement_level: 0,
        })
    }

    /// Create algebraic number from a rational.
    ///
    /// The rational `r` is represented as the root of the linear polynomial `(x - r)`.
    pub fn from_rational(r: BigRational) -> Self {
        // x - r  →  coeffs = [-r, 1]
        let poly = Polynomial::new(vec![-r.clone(), BigRational::from(BigInt::from(1))]);

        Self {
            minimal_poly: poly,
            lower: r.clone(),
            upper: r,
            refinement_level: 0,
        }
    }

    /// Check if this algebraic number is rational.
    pub fn is_rational(&self) -> bool {
        self.minimal_poly.degree() == 1 || self.lower == self.upper
    }

    /// Convert to rational if the number is rational.
    pub fn to_rational(&self) -> Option<BigRational> {
        if self.is_rational() {
            Some(self.lower.clone())
        } else {
            None
        }
    }

    /// Refine the isolating interval by bisection.
    pub fn refine(&mut self) {
        let mid = (&self.lower + &self.upper) / BigRational::from(BigInt::from(2));
        let mid_value = self.minimal_poly.eval(&mid);

        if mid_value.is_zero() {
            // Exact root — collapse to a point.
            self.lower = mid.clone();
            self.upper = mid;
        } else {
            let lower_value = self.minimal_poly.eval(&self.lower);
            if lower_value.signum() != mid_value.signum() {
                // Root is in [lower, mid].
                self.upper = mid;
            } else {
                // Root is in [mid, upper].
                self.lower = mid;
            }
        }

        self.refinement_level += 1;
    }

    /// Refine until the interval width is at most `precision`.
    pub fn refine_to_precision(&mut self, precision: &BigRational) {
        while &self.upper - &self.lower > *precision {
            self.refine();
        }
    }

    /// Width of the current isolating interval.
    pub fn interval_width(&self) -> BigRational {
        &self.upper - &self.lower
    }

    /// Midpoint of the current isolating interval (an approximation of the number).
    pub fn midpoint(&self) -> BigRational {
        (&self.lower + &self.upper) / BigRational::from(BigInt::from(2))
    }

    /// Compare with another algebraic number by interval refinement.
    pub fn compare(&mut self, other: &mut AlgebraicNumber) -> Ordering {
        loop {
            if self.upper < other.lower {
                return Ordering::Less;
            }
            if self.lower > other.upper {
                return Ordering::Greater;
            }

            // Identical representation → definitely equal.
            if self.minimal_poly == other.minimal_poly
                && self.lower == other.lower
                && self.upper == other.upper
            {
                return Ordering::Equal;
            }

            // Intervals overlap; refine both.
            self.refine();
            other.refine();

            if self.refinement_level > 1000 {
                // Cannot separate after 1 000 steps — treat as equal.
                return Ordering::Equal;
            }
        }
    }

    /// Add two algebraic numbers using resultant-based exact arithmetic.
    ///
    /// Given α (root of p) and β (root of q), computes α+β as a root of
    /// `r(x) = Res_t(p(t), q(x−t))`, the polynomial in x whose roots are all
    /// pairwise sums of roots of p and q.
    ///
    /// The resultant is computed via evaluation-interpolation: we evaluate
    /// `Res_t(p(t), q(k−t))` for `deg(p)*deg(q)+1` integer points k, then
    /// reconstruct `r` by Lagrange interpolation over Q.
    ///
    /// The correct root is isolated using the interval `[aₗ+bₗ, aᵤ+bᵤ]`.
    ///
    /// # Errors
    ///
    /// Returns [`AlgebraicNumberError::NonIsolatingInterval`] if the sum
    /// interval does not isolate exactly one root of the result polynomial after
    /// repeated halving, or [`AlgebraicNumberError::ZeroPolynomial`] if the
    /// computed resultant is identically zero (only possible when p and q share
    /// a common root, which cannot happen for irreducible min-polys over Q
    /// unless they are identical).
    pub fn add(&self, other: &AlgebraicNumber) -> Result<AlgebraicNumber, AlgebraicNumberError> {
        // Fast path: both rational.
        if self.is_rational() && other.is_rational() {
            return Ok(AlgebraicNumber::from_rational(&self.lower + &other.lower));
        }

        // Compute the sum polynomial r(x) = Res_t(p(t), q(x-t)) via
        // evaluation at deg(p)*deg(q)+1 integer points followed by Lagrange
        // interpolation.
        let deg_p = self.minimal_poly.degree();
        let deg_q = other.minimal_poly.degree();
        let result_deg = deg_p * deg_q;
        let num_points = result_deg + 1;

        let eval_points: Vec<BigRational> = (0..num_points as i64)
            .map(|k| BigRational::from(BigInt::from(k)))
            .collect();

        let values: Vec<BigRational> = eval_points
            .iter()
            .map(|x_val| {
                // Build q_x(t) = q(x_val - t) by substituting t -> (x_val - t).
                // q(x_val - t) = sum_i coeff_i * (x_val - t)^i
                // We expand each (x_val - t)^i using the binomial theorem.
                let q_shifted = poly_substitute_affine_t(&other.minimal_poly, x_val);
                // Now compute Res_t(p(t), q_shifted(t)) via subresultant PRS.
                univariate_resultant(&self.minimal_poly, &q_shifted)
            })
            .collect();

        // Reconstruct r(x) from the evaluations via Lagrange interpolation.
        let raw_poly = lagrange_interpolate(&eval_points, &values);

        if raw_poly.degree() == 0 && raw_poly.coeffs.first().is_none_or(Zero::is_zero) {
            return Err(AlgebraicNumberError::ZeroPolynomial);
        }

        // Extract the square-free part so the Sturm-based isolation works correctly.
        let result_poly = square_free_part(raw_poly);

        // The correct root is in [lower_self + lower_other, upper_self + upper_other].
        // Refine until the interval is isolating.
        let sum_lower = &self.lower + &other.lower;
        let sum_upper = &self.upper + &other.upper;

        isolate_root_in_interval(result_poly, sum_lower, sum_upper)
    }

    /// Multiply two algebraic numbers using resultant-based exact arithmetic.
    ///
    /// Given α (root of p) and β (root of q), computes α·β as a root of
    /// `r(x) = Res_t(p(t), t^deg(q) · q(x/t))`, the polynomial whose roots
    /// are all pairwise products.
    ///
    /// For multiplication the evaluation-interpolation approach evaluates
    /// `Res_t(p(t), t^m · q(k/t))` at nonzero points k (since division by t
    /// is not polynomial at k=0; the value at 0 is handled as `p(0)^deg(q)` ·
    /// appropriate sign correction).  We then isolate the product root using
    /// interval `[min(aₗbₗ, aₗbᵤ, aᵤbₗ, aᵤbᵤ), max(…)]`.
    ///
    /// # Errors
    ///
    /// Returns [`AlgebraicNumberError::ZeroPolynomial`] if the computed
    /// resultant is zero, or [`AlgebraicNumberError::NonIsolatingInterval`] if
    /// isolation fails.
    pub fn mul(&self, other: &AlgebraicNumber) -> Result<AlgebraicNumber, AlgebraicNumberError> {
        // Fast path: both rational.
        if self.is_rational() && other.is_rational() {
            return Ok(AlgebraicNumber::from_rational(&self.lower * &other.lower));
        }

        let deg_p = self.minimal_poly.degree();
        let deg_q = other.minimal_poly.degree();
        let result_deg = deg_p * deg_q;
        let num_points = result_deg + 1;

        // For the product resultant we use points k = 1, 2, ..., num_points
        // (avoid k=0 to dodge the t=0 singularity in q(x/t)).
        let eval_points: Vec<BigRational> = (1..=(num_points as i64))
            .map(|k| BigRational::from(BigInt::from(k)))
            .collect();

        let values: Vec<BigRational> = eval_points
            .iter()
            .map(|x_val| {
                // Build h(t) = t^deg(q) * q(x_val / t).
                // Since q(x_val / t) = sum_i c_i * (x_val/t)^i, we have
                // t^deg(q) * q(x_val/t) = sum_i c_i * x_val^i * t^(deg_q - i).
                // This is a polynomial in t with rational coefficients.
                let h = poly_mul_pow_quot(&other.minimal_poly, x_val, deg_q);
                univariate_resultant(&self.minimal_poly, &h)
            })
            .collect();

        let raw_poly = lagrange_interpolate(&eval_points, &values);

        if raw_poly.degree() == 0 && raw_poly.coeffs.first().is_none_or(Zero::is_zero) {
            return Err(AlgebraicNumberError::ZeroPolynomial);
        }

        // The resultant polynomial may have repeated roots.  Extract the
        // square-free part so that the Sturm-based root isolation works
        // correctly (Sturm counts distinct roots).
        let result_poly = square_free_part(raw_poly);

        // Interval arithmetic: product interval (all four corners).
        let corners = [
            &self.lower * &other.lower,
            &self.lower * &other.upper,
            &self.upper * &other.lower,
            &self.upper * &other.upper,
        ];
        let prod_lower = corners.iter().min_by(|a, b| a.cmp(b)).expect("4 elements").clone();
        let prod_upper = corners.iter().max_by(|a, b| a.cmp(b)).expect("4 elements").clone();

        isolate_root_in_interval(result_poly, prod_lower, prod_upper)
    }

    /// Negate an algebraic number.
    ///
    /// If α is a root of p(x), then −α is a root of p(−x).
    pub fn negate(&self) -> AlgebraicNumber {
        let neg_poly = poly_substitute_neg_x(&self.minimal_poly);
        AlgebraicNumber {
            minimal_poly: neg_poly,
            lower: -&self.upper,
            upper: -&self.lower,
            refinement_level: self.refinement_level,
        }
    }

    /// Return the sign of the algebraic number: −1, 0, or 1.
    pub fn signum(&self) -> i32 {
        if self.upper < BigRational::zero() {
            -1
        } else if self.lower > BigRational::zero() {
            1
        } else if self.lower == self.upper && self.lower.is_zero() {
            0
        } else {
            // Interval straddles zero; refine until sign is clear.
            let mut copy = self.clone();
            copy.refine();
            copy.signum()
        }
    }
}

// ─── private helpers ────────────────────────────────────────────────────────

/// Substitute x → −x in a polynomial: p(−x).
///
/// For p(x) = a₀ + a₁x + a₂x² + …, we have p(−x) = a₀ − a₁x + a₂x² − …
fn poly_substitute_neg_x(poly: &Polynomial) -> Polynomial {
    let coeffs: Vec<BigRational> = poly
        .coeffs
        .iter()
        .enumerate()
        .map(|(i, c)| if i % 2 == 0 { c.clone() } else { -c })
        .collect();
    Polynomial::new(coeffs)
}

/// Count distinct real roots of `poly` in the open interval `(lower, upper)`
/// using Sturm's theorem: count = V(lower) − V(upper) where V(x) is the
/// number of sign changes in the Sturm sequence evaluated at x.
fn sturm_root_count(poly: &Polynomial, lower: &BigRational, upper: &BigRational) -> usize {
    let seq = build_sturm_sequence(poly);
    let v_lower = sign_variations(&seq, lower);
    let v_upper = sign_variations(&seq, upper);
    (v_lower as isize - v_upper as isize).unsigned_abs()
}

/// Build the Sturm sequence of a polynomial.
///
/// - p₀ = p
/// - p₁ = p′
/// - pₖ₊₁ = −rem(pₖ₋₁, pₖ)
fn build_sturm_sequence(poly: &Polynomial) -> Vec<Polynomial> {
    let mut seq = vec![poly.clone(), poly.derivative()];

    loop {
        let n = seq.len();
        let last = &seq[n - 1];

        // Terminate when the tail is degree-0 (zero or non-zero constant): any polynomial
        // mod a constant is 0, so the next negated remainder would be 0 anyway.
        if last.degree() == 0 {
            break;
        }

        let remainder = seq[n - 2].remainder(last);
        let negated = Polynomial::new(remainder.coeffs.iter().map(|c| -c).collect());

        if negated.degree() == 0 && negated.coeffs.first().is_none_or(Zero::is_zero) {
            break;
        }

        seq.push(negated);

        // Safety guard against pathological inputs.
        if seq.len() > 1000 {
            break;
        }
    }

    seq
}

/// Count sign variations in the Sturm sequence evaluated at `point`.
///
/// Zero values are skipped (Sturm's theorem convention).
fn sign_variations(seq: &[Polynomial], point: &BigRational) -> usize {
    let signs: Vec<i32> = seq
        .iter()
        .map(|p| {
            let val = p.eval(point);
            if val.is_positive() {
                1
            } else if val.is_negative() {
                -1
            } else {
                0
            }
        })
        .filter(|&s| s != 0)
        .collect();

    signs.windows(2).filter(|w| w[0] != w[1]).count()
}

// ─── square-free helpers ─────────────────────────────────────────────────────

/// Compute the square-free part of a univariate polynomial over Q:
/// `sqfp(p) = p / gcd(p, p′)`.
///
/// The result has the same roots as `p` but each root appears exactly once,
/// which is required for the Sturm-sequence root counter.
fn square_free_part(p: Polynomial) -> Polynomial {
    let deriv = p.derivative();

    // If derivative is zero (constant polynomial), p is already square-free.
    if deriv.degree() == 0 && deriv.coeffs.first().is_none_or(|c| c.is_zero()) {
        return p;
    }

    // Compute gcd(p, p') via Euclidean PRS over Q.
    let g = univariate_gcd(&p, &deriv);

    // If gcd is degree 0 (i.e., a nonzero constant), p is already square-free.
    if g.degree() == 0 {
        return p;
    }

    // Exact division: p / g.
    exact_poly_div(&p, &g)
}

/// Compute gcd of two univariate polynomials over Q via Euclidean algorithm.
fn univariate_gcd(p: &Polynomial, q: &Polynomial) -> Polynomial {
    let mut a = p.clone();
    let mut b = q.clone();

    loop {
        // Normalise the zero polynomial check.
        let b_is_zero = b.coeffs.iter().all(|c| c.is_zero());
        if b_is_zero {
            // Make monic and return.
            return make_monic(a);
        }
        let r = a.remainder(&b);
        a = b;
        b = r;
    }
}

/// Divide polynomial `a` by `b` exactly (remainder must be zero over Q).
fn exact_poly_div(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let deg_b = b.degree();
    let lead_b = b.coeffs[deg_b].clone();

    if a.degree() < deg_b {
        return Polynomial::new(vec![BigRational::zero()]);
    }

    let result_deg = a.degree() - deg_b;
    let mut quotient_coeffs = vec![BigRational::zero(); result_deg + 1];
    // Work on a mutable coefficient vector; rebuild Polynomial at end.
    let mut rem_coeffs = a.coeffs.clone();

    // Classic long division (high degree to low).
    let mut rem_deg = a.degree();
    while rem_deg >= deg_b {
        // Check if the high-degree coefficient is essentially zero.
        if rem_coeffs[rem_deg].is_zero() {
            if rem_deg == 0 {
                break;
            }
            rem_deg -= 1;
            continue;
        }
        let factor = rem_coeffs[rem_deg].clone() / &lead_b;
        let shift = rem_deg - deg_b;
        quotient_coeffs[shift] = factor.clone();

        for i in 0..=deg_b {
            let upd = rem_coeffs[shift + i].clone() - &factor * &b.coeffs[i];
            rem_coeffs[shift + i] = upd;
        }
        if rem_deg == 0 {
            break;
        }
        rem_deg -= 1;
    }

    Polynomial::new(quotient_coeffs)
}

/// Normalise a polynomial: make it monic (leading coeff = 1).
fn make_monic(p: Polynomial) -> Polynomial {
    let deg = p.degree();
    let lead = &p.coeffs[deg];
    if lead.is_zero() {
        return p;
    }
    let inv = lead.clone().recip();
    let coeffs: Vec<BigRational> = p.coeffs.iter().map(|c| c * &inv).collect();
    Polynomial::new(coeffs)
}

// ─── resultant helpers ──────────────────────────────────────────────────────

/// Compute `q(k - t)` as a polynomial in `t`, where `q` is a univariate poly
/// over Q and `k` is a rational constant.
///
/// We expand each power `(k - t)^i` via the binomial theorem:
/// `(k - t)^i = Σ_{j=0}^{i} C(i,j) * k^(i-j) * (-t)^j = Σ_j C(i,j) * k^(i-j) * (-1)^j * t^j`
///
/// The coefficient of `t^j` in the full expansion of `q(k-t)` is then
/// `Σ_{i ≥ j} q.coeffs[i] * C(i,j) * k^(i-j) * (-1)^j`.
fn poly_substitute_affine_t(q: &Polynomial, k: &BigRational) -> Polynomial {
    let n = q.coeffs.len();
    if n == 0 {
        return Polynomial::new(vec![BigRational::zero()]);
    }

    // result_coeffs[j] = coefficient of t^j in q(k - t)
    let mut result_coeffs = vec![BigRational::zero(); n];

    for (i, q_i) in q.coeffs.iter().enumerate() {
        if q_i.is_zero() {
            continue;
        }
        // Accumulate contributions to each t^j from this term: q_i * (k - t)^i
        let binom_row = binomial_row(i);
        for (j, binom_ij) in binom_row.iter().enumerate() {
            // k^(i-j)
            let k_pow = rational_pow(k, i - j);
            // (-1)^j
            let sign = if j % 2 == 0 {
                BigRational::one()
            } else {
                -BigRational::one()
            };
            let contrib = q_i * binom_ij * k_pow * sign;
            result_coeffs[j] = result_coeffs[j].clone() + contrib;
        }
    }

    Polynomial::new(result_coeffs)
}

/// Build `h(t) = t^m * q(x_val / t)` for a fixed rational value `x_val` and
/// integer `m = deg(q)`.
///
/// `q(x_val / t) = Σ_i q_i * (x_val/t)^i`, so
/// `t^m * q(x_val/t) = Σ_i q_i * x_val^i * t^(m-i)`.
///
/// The coefficient of `t^(m-i)` (i.e., `t^j` where `j = m-i`) is `q_i * x_val^i`.
fn poly_mul_pow_quot(q: &Polynomial, x_val: &BigRational, m: usize) -> Polynomial {
    let mut result_coeffs = vec![BigRational::zero(); m + 1];
    for (i, q_i) in q.coeffs.iter().enumerate() {
        if q_i.is_zero() {
            continue;
        }
        let j = m.saturating_sub(i);
        let x_pow = rational_pow(x_val, i);
        result_coeffs[j] = result_coeffs[j].clone() + q_i * x_pow;
    }
    Polynomial::new(result_coeffs)
}

/// Compute the resultant `Res_t(p, q)` of two univariate polynomials over Q
/// via the Sylvester matrix determinant, using Bareiss fraction-free
/// Gaussian elimination over the rationals.
///
/// Returns a rational number.  Zero if the polynomials share a common root.
fn univariate_resultant(p: &Polynomial, q: &Polynomial) -> BigRational {
    let m = p.degree();
    let n = q.degree();

    // Degenerate: one polynomial is constant.
    if m == 0 {
        let c = p.coeffs.first().cloned().unwrap_or_else(BigRational::zero);
        return rational_pow(&c, n);
    }
    if n == 0 {
        let c = q.coeffs.first().cloned().unwrap_or_else(BigRational::zero);
        return rational_pow(&c, m);
    }

    // Build the (m+n) × (m+n) Sylvester matrix.
    // Coefficients are stored low-to-high, but the Sylvester matrix convention
    // is high-to-low. Reverse for row construction.
    let p_rev: Vec<BigRational> = p.coeffs.iter().rev().cloned().collect();
    let q_rev: Vec<BigRational> = q.coeffs.iter().rev().cloned().collect();

    let size = m + n;
    let mut mat: Vec<Vec<BigRational>> = vec![vec![BigRational::zero(); size]; size];

    // First n rows: p coefficients shifted.
    for r in 0..n {
        mat[r][r..(m + r + 1)].clone_from_slice(&p_rev[..(m + 1)]);
    }
    // Last m rows: q coefficients shifted.
    for r in 0..m {
        mat[n + r][r..(n + r + 1)].clone_from_slice(&q_rev[..(n + 1)]);
    }

    // Bareiss fraction-free Gaussian elimination over Q.
    bareiss_rat_det(mat)
}

/// Bareiss fraction-free determinant over Q.
///
/// Uses rational arithmetic so division is exact.  The Bareiss invariant
/// guarantees every division step is exact (Sylvester's identity).
fn bareiss_rat_det(mut mat: Vec<Vec<BigRational>>) -> BigRational {
    let n = mat.len();
    if n == 0 {
        return BigRational::one();
    }
    if n == 1 {
        return mat.remove(0).remove(0);
    }

    let mut sign = BigRational::one();
    let neg_one = -BigRational::one();

    for col in 0..n {
        // Find a nonzero pivot.
        let pivot_row = (col..n).find(|&r| !mat[r][col].is_zero());
        let pivot_row = match pivot_row {
            Some(r) => r,
            None => return BigRational::zero(),
        };

        if pivot_row != col {
            mat.swap(col, pivot_row);
            sign *= &neg_one;
        }

        let pivot = mat[col][col].clone();

        for row in (col + 1)..n {
            for j in (col + 1)..n {
                let prod1 = &pivot * &mat[row][j];
                let prod2 = &mat[row][col] * &mat[col][j];
                let diff = prod1 - prod2;
                if col == 0 {
                    mat[row][j] = diff;
                } else {
                    // Exact division by the previous pivot.
                    let prev_pivot = mat[col - 1][col - 1].clone();
                    if prev_pivot.is_zero() {
                        mat[row][j] = BigRational::zero();
                    } else {
                        mat[row][j] = diff / prev_pivot;
                    }
                }
            }
            mat[row][col] = BigRational::zero();
        }
    }

    sign * mat[n - 1][n - 1].clone()
}

/// Reconstruct a polynomial from its values at distinct rational points via
/// Lagrange interpolation.
///
/// Given `n+1` pairs `(x_0, y_0), …, (x_n, y_n)` this returns the unique
/// polynomial of degree ≤ n passing through all points.
fn lagrange_interpolate(points: &[BigRational], values: &[BigRational]) -> Polynomial {
    let n = points.len();
    assert_eq!(n, values.len(), "points and values must have equal length");

    if n == 0 {
        return Polynomial::new(vec![BigRational::zero()]);
    }
    if n == 1 {
        return Polynomial::new(vec![values[0].clone()]);
    }

    // Accumulate the interpolating polynomial term by term.
    let mut result_coeffs = vec![BigRational::zero(); n];

    for i in 0..n {
        // Compute the i-th Lagrange basis polynomial Lᵢ(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
        // and scale it by yᵢ.
        let mut basis_coeffs = vec![BigRational::one()]; // start as constant 1
        let mut denom = BigRational::one();

        for j in 0..n {
            if j == i {
                continue;
            }
            // Multiply basis by (x - x_j): shift coefficients up and subtract x_j * current
            let mut new_coeffs = vec![BigRational::zero(); basis_coeffs.len() + 1];
            for (k, c) in basis_coeffs.iter().enumerate() {
                new_coeffs[k + 1] = new_coeffs[k + 1].clone() + c;
                new_coeffs[k] = new_coeffs[k].clone() - c * &points[j];
            }
            basis_coeffs = new_coeffs;

            // Accumulate denominator: (x_i - x_j)
            denom *= &points[i] - &points[j];
        }

        // Scale: yᵢ / denom
        let scale = &values[i] / &denom;
        for (k, c) in basis_coeffs.iter().enumerate() {
            result_coeffs[k] = result_coeffs[k].clone() + c * &scale;
        }
    }

    Polynomial::new(result_coeffs)
}

/// Return the k-th row of Pascal's triangle as BigRational values.
fn binomial_row(k: usize) -> Vec<BigRational> {
    let mut row = vec![BigRational::one(); k + 1];
    for i in 1..k {
        // In-place left-to-right: row[j] = row[j] + row[j-1] (before update)
        for j in (1..i + 1).rev() {
            let prev = row[j - 1].clone();
            row[j] = row[j].clone() + prev;
        }
    }
    row
}

/// Compute `base^exp` for `base ∈ Q` and `exp ∈ usize`.
fn rational_pow(base: &BigRational, exp: usize) -> BigRational {
    if exp == 0 {
        return BigRational::one();
    }
    let mut result = BigRational::one();
    let mut base = base.clone();
    let mut exp = exp;
    while exp > 0 {
        if exp % 2 == 1 {
            result *= &base;
        }
        let b = base.clone();
        base *= &b;
        exp /= 2;
    }
    result
}

/// Given a polynomial `r` and a candidate interval `[lo, hi]` containing α+β
/// or α·β, refine until exactly one root of `r` is isolated in the interval.
///
/// Returns the algebraic number with that isolating interval, or an error if
/// isolation cannot be achieved within the refinement budget.
fn isolate_root_in_interval(
    r: Polynomial,
    lo: BigRational,
    hi: BigRational,
) -> Result<AlgebraicNumber, AlgebraicNumberError> {
    let mut lo = lo;
    let mut hi = hi;

    // If lo == hi, test directly.
    if lo == hi {
        if r.eval(&lo).is_zero() {
            return AlgebraicNumber::new(r, lo.clone(), hi);
        }
        return Err(AlgebraicNumberError::NoRootInInterval);
    }

    // Broaden the interval slightly to handle exact-endpoint roots.
    // We expand by 1 ULP in BigRational terms (add/subtract a tiny epsilon).
    let eps = BigRational::new(BigInt::from(1), BigInt::from(1_000_000_000_i64));
    lo -= &eps;
    hi += &eps;

    // Count roots and refine until we have exactly one.
    for _ in 0..200 {
        let count = sturm_root_count(&r, &lo, &hi);
        if count == 1 {
            return AlgebraicNumber::new(r, lo, hi);
        }
        if count == 0 {
            return Err(AlgebraicNumberError::NoRootInInterval);
        }
        // More than one root: bisect.
        let mid = (&lo + &hi) / BigRational::from(BigInt::from(2));
        let count_left = sturm_root_count(&r, &lo, &mid);
        if count_left >= 1 {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Err(AlgebraicNumberError::NonIsolatingInterval)
}

// ─── trait impls ────────────────────────────────────────────────────────────

impl PartialEq for AlgebraicNumber {
    fn eq(&self, other: &Self) -> bool {
        let mut a = self.clone();
        let mut b = other.clone();
        matches!(a.compare(&mut b), Ordering::Equal)
    }
}

impl Eq for AlgebraicNumber {}

impl PartialOrd for AlgebraicNumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AlgebraicNumber {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut a = self.clone();
        let mut b = other.clone();
        a.compare(&mut b)
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from(BigInt::from(n))
    }

    #[test]
    fn test_from_rational() {
        let alg = AlgebraicNumber::from_rational(rat(42));
        assert!(alg.is_rational());
        assert_eq!(alg.to_rational(), Some(rat(42)));
    }

    #[test]
    fn test_rational_arithmetic() {
        let a = AlgebraicNumber::from_rational(rat(3));
        let b = AlgebraicNumber::from_rational(rat(5));

        let sum = a.add(&b).expect("test operation should succeed");
        assert!(sum.is_rational());
        assert_eq!(sum.to_rational(), Some(rat(8)));

        let prod = a.mul(&b).expect("test operation should succeed");
        assert!(prod.is_rational());
        assert_eq!(prod.to_rational(), Some(rat(15)));
    }

    #[test]
    fn test_negate() {
        let a = AlgebraicNumber::from_rational(rat(5));
        let neg_a = a.negate();

        assert!(neg_a.is_rational());
        assert_eq!(neg_a.to_rational(), Some(rat(-5)));
    }

    #[test]
    fn test_compare() {
        let a = AlgebraicNumber::from_rational(rat(3));
        let b = AlgebraicNumber::from_rational(rat(5));

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, a.clone());
    }

    #[test]
    fn test_signum() {
        let pos = AlgebraicNumber::from_rational(rat(5));
        let neg = AlgebraicNumber::from_rational(rat(-3));
        let zero = AlgebraicNumber::from_rational(rat(0));

        assert_eq!(pos.signum(), 1);
        assert_eq!(neg.signum(), -1);
        assert_eq!(zero.signum(), 0);
    }

    #[test]
    fn test_refine() {
        // x^2 - 2, root in [1, 2]
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        let mut alg =
            AlgebraicNumber::new(poly, rat(1), rat(2)).expect("test operation should succeed");

        let initial_width = alg.interval_width();
        alg.refine();
        let refined_width = alg.interval_width();

        assert!(refined_width < initial_width);
    }

    #[test]
    fn test_interval_width() {
        let alg = AlgebraicNumber::from_rational(rat(5));
        assert_eq!(alg.interval_width(), rat(0));
    }

    #[test]
    fn test_new_rejects_non_isolating_interval() {
        // x^2 - 1 has two roots in [-2, 2]
        let poly = Polynomial::new(vec![rat(-1), rat(0), rat(1)]);
        let result = AlgebraicNumber::new(poly, rat(-2), rat(2));
        assert!(result.is_err());
    }

    #[test]
    fn test_new_accepts_isolating_interval() {
        // x^2 - 2 has exactly one root in [1, 2]
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        let result = AlgebraicNumber::new(poly, rat(1), rat(2));
        assert!(result.is_ok());
    }

    #[test]
    fn test_sturm_root_count() {
        // x^2 - 2: one positive root between 1 and 2
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        let count = sturm_root_count(&poly, &rat(1), &rat(2));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_poly_substitute_neg_x() {
        // p(x) = x^2 - 2 → p(-x) = x^2 - 2 (even polynomial, unchanged)
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        let neg = poly_substitute_neg_x(&poly);
        assert_eq!(neg.eval(&rat(2)), poly.eval(&rat(2)));

        // p(x) = x^3 → p(-x) = -x^3
        let poly2 = Polynomial::new(vec![rat(0), rat(0), rat(0), rat(1)]);
        let neg2 = poly_substitute_neg_x(&poly2);
        assert_eq!(neg2.eval(&rat(2)), -poly2.eval(&rat(2)));
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Construct √2 as an `AlgebraicNumber` (root of x²−2 in [1,2]).
    fn sqrt2() -> AlgebraicNumber {
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        AlgebraicNumber::new(poly, rat(1), rat(2)).expect("√2 construction")
    }

    /// Construct √3 as an `AlgebraicNumber` (root of x²−3 in [1,2]).
    fn sqrt3() -> AlgebraicNumber {
        let poly = Polynomial::new(vec![rat(-3), rat(0), rat(1)]);
        AlgebraicNumber::new(poly, rat(1), rat(2)).expect("√3 construction")
    }

    // ── new tests for non-rational add / mul ─────────────────────────────────

    /// √2 + √3 ≈ 3.1462…
    /// Verify the sum isolates exactly one root and the minimal poly
    /// x⁴ − 10x² + 1 has opposite signs at the interval endpoints.
    #[test]
    fn test_add_irrational_sum_polynomial_root() {
        let a = sqrt2();
        let b = sqrt3();
        let mut sum = a.add(&b).expect("√2 + √3 should succeed");

        // The minimal polynomial of √2+√3 is p(x) = x⁴−10x²+1.
        // By construction the isolating interval contains exactly one root.
        // Verify: p changes sign across the interval.
        let p = |x: &num_rational::BigRational| {
            x.clone() * x.clone() * x.clone() * x.clone()
                - rat(10) * x.clone() * x.clone()
                + rat(1)
        };

        // Refine until the interval is very tight (width < 2^{-20}).
        let tight = num_rational::BigRational::new(BigInt::from(1), BigInt::from(1 << 20));
        sum.refine_to_precision(&tight);

        let val_lo = p(&sum.lower);
        let val_hi = p(&sum.upper);

        // Sign must differ (Intermediate Value Theorem guarantees a root).
        assert!(
            val_lo.is_negative() != val_hi.is_negative()
                || val_lo.is_zero()
                || val_hi.is_zero(),
            "expected sign change: p(lo)={:?}, p(hi)={:?}",
            val_lo,
            val_hi
        );

        // The interval must straddle √2+√3 ≈ 3.146 (in (3, 4)).
        assert!(
            sum.lower < rat(4) && sum.upper > rat(3),
            "interval [{:?}, {:?}] should contain 3.146",
            sum.lower,
            sum.upper
        );
    }

    /// √2 * √3 = √6 ≈ 2.449…
    /// Verify it is a root of x²−6 and the interval contains √6.
    #[test]
    fn test_mul_irrational_product_polynomial_root() {
        let a = sqrt2();
        let b = sqrt3();
        let mut prod = a.mul(&b).expect("√2 * √3 should succeed");

        // Verify the product is in (2, 3) — the correct isolating interval for √6.
        assert!(
            prod.lower < num_rational::BigRational::new(BigInt::from(3), BigInt::from(1)),
            "lower bound should be < 3"
        );
        assert!(
            prod.upper > num_rational::BigRational::new(BigInt::from(2), BigInt::from(1)),
            "upper bound should be > 2"
        );

        // The minimal poly of √6 is x² − 6; verify it changes sign across the interval.
        let p = |x: &num_rational::BigRational| x.clone() * x.clone() - rat(6);

        let tight = num_rational::BigRational::new(BigInt::from(1), BigInt::from(1 << 20));
        prod.refine_to_precision(&tight);

        let val_lo = p(&prod.lower);
        let val_hi = p(&prod.upper);
        assert!(
            val_lo.is_negative() != val_hi.is_negative()
                || val_lo.is_zero()
                || val_hi.is_zero(),
            "expected sign change for √6: p(lo)={:?}, p(hi)={:?}",
            val_lo,
            val_hi
        );
    }

    /// Rational + irrational: 1 + √2 should succeed and equal (√2+1).
    #[test]
    fn test_add_rational_plus_irrational() {
        let one = AlgebraicNumber::from_rational(rat(1));
        let s2 = sqrt2();
        let result = one.add(&s2).expect("1 + √2 should succeed");
        // 1 + √2 ≈ 2.414; should be in (2, 3).
        assert!(result.lower < rat(3) && result.upper > rat(2),
            "1 + √2 expected in (2,3), got [{:?}, {:?}]", result.lower, result.upper);
    }

    /// Helpers: poly_substitute_affine_t and univariate_resultant.
    #[test]
    fn test_poly_substitute_affine_t() {
        // q(t) = t − 1 (coeffs = [-1, 1]);  q(3 − t) should be (3-t) - 1 = 2 - t
        // = coeffs [2, -1]
        let q = Polynomial::new(vec![rat(-1), rat(1)]);
        let shifted = poly_substitute_affine_t(&q, &rat(3));
        assert_eq!(shifted.eval(&rat(0)), rat(2), "q(3-0) = 2");
        assert_eq!(shifted.eval(&rat(1)), rat(1), "q(3-1) = 2 - 1 = 1");
    }

    #[test]
    fn test_univariate_resultant_linear() {
        // Res(t - a, t - b) = b - a.
        let p = Polynomial::new(vec![rat(-3), rat(1)]); // t - 3
        let q = Polynomial::new(vec![rat(-5), rat(1)]); // t - 5
        let res = univariate_resultant(&p, &q);
        // Res(t-3, t-5) = 5 - 3 = 2 (or -2 depending on sign convention; nonzero)
        assert!(!res.is_zero(), "resultant of coprime linear polys should be nonzero");
    }

    #[test]
    fn test_lagrange_interpolate_linear() {
        use num_bigint::BigInt;
        let points = vec![
            num_rational::BigRational::from(BigInt::from(0)),
            num_rational::BigRational::from(BigInt::from(1)),
        ];
        let values = vec![
            num_rational::BigRational::from(BigInt::from(3)),
            num_rational::BigRational::from(BigInt::from(5)),
        ];
        // Line through (0,3) and (1,5): p(x) = 3 + 2x
        let poly = lagrange_interpolate(&points, &values);
        assert_eq!(poly.eval(&points[0]), values[0]);
        assert_eq!(poly.eval(&points[1]), values[1]);
    }
}
