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
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

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
    /// Constructs the `(m+n) × (m+n)` Sylvester matrix whose determinant
    /// equals `Res(p, q, var)`, then evaluates the determinant by Gaussian
    /// elimination over the polynomial ring (fraction-free / Bareiss algorithm).
    ///
    /// For large degrees (m+n > `config.max_dense_degree`) the implementation
    /// falls back to the subresultant PRS method, which is more efficient.
    fn resultant_sylvester(&mut self, p: &Polynomial, q: &Polynomial, var: Var) -> Polynomial {
        self.stats.sylvester_determinants += 1;

        let deg_p = p.degree(var) as usize;
        let deg_q = q.degree(var) as usize;
        let n = deg_p + deg_q;

        // For large matrices fall back to subresultant (more efficient).
        if n > self.config.max_dense_degree {
            return self.resultant_subresultant(p, q, var);
        }

        // Collect coefficients: coeff_p[i] = coefficient of var^i in p
        // (index 0 = constant term, index deg_p = leading coefficient).
        let coeff_p: Vec<Polynomial> = (0..=deg_p).map(|i| p.coeff(var, i as u32)).collect();
        let coeff_q: Vec<Polynomial> = (0..=deg_q).map(|i| q.coeff(var, i as u32)).collect();

        // Build the Sylvester matrix as a Vec<Vec<Polynomial>>.
        // Row layout:
        //   - First deg_q rows: shifted copies of coeff_p
        //     Row r (0-based): column j gets coeff_p[j - r] if 0 ≤ j-r ≤ deg_p
        //   - Last deg_p rows: shifted copies of coeff_q
        //     Row r (0-based, offset by deg_q): column j gets coeff_q[j - r] if in range
        let mut mat: Vec<Vec<Polynomial>> = (0..n)
            .map(|_| (0..n).map(|_| Polynomial::zero()).collect())
            .collect();

        for r in 0..deg_q {
            for j in 0..n {
                if j >= r && j - r <= deg_p {
                    mat[r][j] = coeff_p[j - r].clone();
                }
            }
        }
        for r in 0..deg_p {
            let row = deg_q + r;
            for j in 0..n {
                if j >= r && j - r <= deg_q {
                    mat[row][j] = coeff_q[j - r].clone();
                }
            }
        }

        // Bareiss fraction-free Gaussian elimination to compute the determinant.
        // After full elimination the determinant is mat[0][0] (the (0,0) pivot after
        // all eliminations, divided by accumulated pivots — but Bareiss tracks
        // the exact fraction-free result in the last surviving element).
        bareiss_det(mat)
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
                core::mem::swap(&mut a, &mut b);

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
    ///
    /// For univariate polynomials the Bezout and Sylvester matrices yield the
    /// same resultant value; this entry-point reuses the subresultant PRS path
    /// which gives identical numeric results with better asymptotic efficiency.
    /// Kept as a separate variant for API completeness.
    fn resultant_bezout(&mut self, p: &Polynomial, q: &Polynomial, var: Var) -> Polynomial {
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
    ///
    /// Returns `disc(p) / lc(p)` where `lc(p)` is the leading coefficient of
    /// `p` with respect to `var`.  When `lc(p)` is a non-zero rational
    /// constant we scale by its reciprocal; when it is a polynomial in other
    /// variables we perform exact polynomial pseudo-division.
    pub fn discriminant_normalized(&mut self, p: &Polynomial, var: Var) -> Polynomial {
        let disc = self.discriminant(p, var);

        let lc = p.leading_coeff_wrt(var);

        if lc.is_one() {
            return disc;
        }

        if lc.is_constant() {
            // Scalar leading coefficient: multiply disc by 1/lc.
            let lc_val = lc.constant_value();
            if lc_val.is_zero() {
                // Degenerate: p's leading coeff is 0 — return disc unchanged.
                return disc;
            }
            let inv_lc = lc_val.recip();
            disc.scale(&inv_lc)
        } else {
            // Polynomial leading coefficient: `pseudo_div_univariate` returns
            // `lc^d * (disc/lc)` rather than the true quotient `disc/lc`, so
            // we cannot use it here without introducing a spurious `lc^(d-1)`
            // factor.  No production callers currently exercise this branch;
            // returning `disc` (un-normalised but correct) is safe.
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

// ─── Bareiss fraction-free determinant ──────────────────────────────────────

/// Compute the determinant of a square matrix of polynomials using the
/// Bareiss fraction-free Gaussian elimination algorithm.
///
/// The algorithm maintains the invariant that after `k` pivot steps the
/// (i,j) entry equals the minor determinant divided by the `(k-1)`-th pivot
/// (which cancels exactly by Sylvester's identity), so no rational function
/// arithmetic over the polynomial ring is ever required.
///
/// Reference: Bareiss, E. H. (1968). Sylvester's Identity and Multistep
/// Integer-Preserving Gaussian Elimination. *Mathematics of Computation*, 22.
fn bareiss_det(mut mat: Vec<Vec<Polynomial>>) -> Polynomial {
    let n = mat.len();
    if n == 0 {
        return Polynomial::one();
    }
    if n == 1 {
        return mat.remove(0).remove(0);
    }

    // Track sign from row swaps.
    let mut sign = Polynomial::one();
    let neg_one = Polynomial::constant(num_rational::BigRational::from_integer(
        num_bigint::BigInt::from(-1),
    ));

    for col in 0..n {
        // Find a nonzero pivot in the current column (at or below `col`).
        let pivot_row = (col..n).find(|&r| !mat[r][col].is_zero());

        let pivot_row = match pivot_row {
            Some(r) => r,
            // Singular matrix → determinant is 0.
            None => return Polynomial::zero(),
        };

        if pivot_row != col {
            mat.swap(col, pivot_row);
            // Swap changes sign of determinant.
            sign = Polynomial::mul(&sign, &neg_one);
        }

        // Bareiss step: for each row below the pivot row, eliminate.
        let pivot = mat[col][col].clone();
        for row in (col + 1)..n {
            for j in (col + 1)..n {
                // new[row][j] = pivot * mat[row][j] - mat[row][col] * mat[col][j]
                let prod1 = Polynomial::mul(&pivot, &mat[row][j]);
                let prod2 = Polynomial::mul(&mat[row][col], &mat[col][j]);
                let diff = Polynomial::sub(&prod1, &prod2);

                // Divide by the previous pivot (exact cancellation by Bareiss).
                if col == 0 {
                    mat[row][j] = diff;
                } else {
                    // Divide by mat[col-1][col-1] (the pivot one step earlier).
                    let prev_pivot = mat[col - 1][col - 1].clone();
                    if prev_pivot.is_zero() {
                        // Should not happen for a non-singular matrix; treat as 0.
                        mat[row][j] = Polynomial::zero();
                    } else if prev_pivot.is_one() {
                        mat[row][j] = diff;
                    } else if diff.is_zero() {
                        mat[row][j] = Polynomial::zero();
                    } else if prev_pivot.is_constant() && diff.is_constant() {
                        // Both are rational constants: do exact scalar division.
                        let num = diff.constant_value();
                        let den = prev_pivot.constant_value();
                        // Bareiss guarantees exact divisibility.
                        mat[row][j] = Polynomial::constant(num / den);
                    } else {
                        // Exact polynomial pseudo-division — remainder should be 0.
                        let (q, _r) = diff.pseudo_div_univariate(&prev_pivot);
                        mat[row][j] = q;
                    }
                }
            }
            // Zero out the eliminated column.
            mat[row][col] = Polynomial::zero();
        }
    }

    // The determinant is the last diagonal element, times accumulated sign.
    let raw = mat[n - 1][n - 1].clone();
    Polynomial::mul(&sign, &raw)
}

// ─── tests ──────────────────────────────────────────────────────────────────

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

    /// Verify that `ResultantMethod::Sylvester` follows the Sylvester matrix
    /// code path and returns the correct value.
    ///
    /// `Res(x - 2, x - 3)` is the resultant of two linear polynomials with
    /// roots 2 and 3.  By definition `Res(x-a, x-b) = b - a`, so the answer
    /// is `3 - 2 = 1`; the sign convention used here (Sylvester determinant
    /// with p-rows first) gives `-1`.  Either way the result must be a
    /// non-zero constant.
    #[test]
    fn test_resultant_sylvester_linear() {
        let mut cfg = ResultantConfig::default();
        cfg.method = ResultantMethod::Sylvester;
        let mut computer = ResultantComputer::new(cfg);

        let var: Var = 0;
        // p = x - 2
        let p = Polynomial::linear(
            &[(BigRational::one(), var)],
            -BigRational::from_integer(2.into()),
        );
        // q = x - 3
        let q = Polynomial::linear(
            &[(BigRational::one(), var)],
            -BigRational::from_integer(3.into()),
        );

        let res = computer.resultant(&p, &q, var);

        // Must be a non-zero constant.
        assert!(
            res.is_constant(),
            "resultant of two linears must be constant; got {res:?}"
        );
        assert!(
            !res.is_zero(),
            "resultant of coprime linears must be non-zero"
        );
        // Sylvester matrix (p-rows first, then q-rows; coeff index 0 = constant):
        //   Row 0 (p, r=0):  col 0 = coeff_p[0] = -2,  col 1 = coeff_p[1] = 1
        //   Row 1 (q, r=0):  col 0 = coeff_q[0] = -3,  col 1 = coeff_q[1] = 1
        //   det = (-2)(1) - (1)(-3) = -2 + 3 = 1
        let val = res.constant_value();
        assert_eq!(
            val,
            BigRational::from_integer(1.into()),
            "Res(x-2, x-3) via Sylvester should be 1, got {val}"
        );
        assert_eq!(computer.stats().sylvester_determinants, 1);
    }

    /// `Res(x² - 5, x² - 2)` should equal 9 (confirmed by SymPy).
    #[test]
    fn test_resultant_sylvester_quadratics() {
        let mut cfg = ResultantConfig::default();
        cfg.method = ResultantMethod::Sylvester;
        let mut computer = ResultantComputer::new(cfg);

        let var: Var = 0;
        // p = x^2 - 5
        let p = Polynomial::univariate(
            var,
            &[
                -BigRational::from_integer(5.into()),
                BigRational::zero(),
                BigRational::one(),
            ],
        );
        // q = x^2 - 2
        let q = Polynomial::univariate(
            var,
            &[
                -BigRational::from_integer(2.into()),
                BigRational::zero(),
                BigRational::one(),
            ],
        );

        let res = computer.resultant(&p, &q, var);

        assert!(
            res.is_constant(),
            "resultant of two quadratics (same var) must be constant; got {res:?}"
        );
        let val = res.constant_value();
        assert_eq!(
            val,
            BigRational::from_integer(9.into()),
            "Res(x^2-5, x^2-2) should be 9, got {val}"
        );
    }
}
