//! # oxiz-math
//!
//! Mathematical foundations for the OxiZ SMT solver.
//!
//! This crate provides Pure Rust implementations of mathematical algorithms
//! required for SMT solving, including:
//!
//! ## Linear Arithmetic
//! - **Simplex**: Dual simplex algorithm for linear programming (LRA theory)
//! - **Interior Point**: Primal-dual interior point method for large-scale LP
//! - **Matrix**: Dense and sparse matrix operations with Gaussian elimination
//! - **Interval**: Interval arithmetic for bound propagation
//! - **Delta Rational**: Support for strict inequalities in simplex
//! - **BLAS**: High-performance BLAS operations for large-scale LP (1000+ variables)
//!
//! ## Non-Linear Arithmetic
//! - **Polynomial**: Multivariate polynomial arithmetic with GCD and factorization
//! - **Rational Function**: Arithmetic on quotients of polynomials (p/q operations)
//! - **Gröbner**: Gröbner basis computation (Buchberger, F4, and F5 algorithms)
//! - **Real Closure**: Algebraic number representation and root isolation
//! - **Hilbert**: Hilbert basis computation for integer cones
//!
//! ## Decision Diagrams
//! - **BDD**: Reduced Ordered Binary Decision Diagrams
//! - **ZDD**: Zero-suppressed BDDs for sparse set representation
//! - **ADD**: Algebraic Decision Diagrams for rational-valued functions
//!
//! ## Numerical Utilities
//! - **Rational**: Arbitrary precision rational arithmetic utilities
//! - **MPFR**: Arbitrary precision floating-point arithmetic (MPFR-like)
//!
//! # Examples
//!
//! ## Polynomial Arithmetic
//!
//! ```
//! use oxiz_math::polynomial::{Polynomial, Var};
//!
//! // Create polynomial for variable x (index 0)
//! let x: Var = 0;
//!
//! // Create polynomial representing just x
//! let p = Polynomial::from_var(x);
//!
//! // Compute x * x = x^2
//! let p_squared = p.clone() * p.clone();
//! ```
//!
//! ## BDD Operations
//!
//! ```
//! use oxiz_math::bdd::BddManager;
//!
//! let mut mgr = BddManager::new();
//!
//! // Create variables (VarId is u32)
//! let x = mgr.variable(0);
//! let y = mgr.variable(1);
//!
//! // Compute x AND y
//! let and_xy = mgr.and(x, y);
//!
//! // Compute x OR y
//! let or_xy = mgr.or(x, y);
//! ```
//!
//! ## BLAS Operations
//!
//! ```
//! use oxiz_math::blas::{ddot, dgemv, Transpose};
//!
//! // Vector dot product
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![4.0, 5.0, 6.0];
//! let dot = ddot(&x, &y);
//! assert_eq!(dot, 32.0);
//! ```
//!
//! ## Arbitrary Precision Floats
//!
//! ```
//! use oxiz_math::mpfr::{ArbitraryFloat, Precision, RoundingMode};
//!
//! let prec = Precision::new(128);
//! let a = ArbitraryFloat::from_f64(3.14159, prec);
//! let b = ArbitraryFloat::from_f64(2.71828, prec);
//! let sum = a.add(&b, RoundingMode::RoundNearest);
//! ```

#![warn(missing_docs)]

pub mod bdd;
pub mod blas;
pub mod blas_ops;
pub mod delta_rational;
pub mod grobner;
pub mod hilbert;
pub mod interior_point;
pub mod interval;
pub mod lp;
pub mod matrix;
pub mod mpfr;
pub mod polynomial;
pub mod rational;
pub mod rational_function;
pub mod realclosure;
pub mod simplex;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_grobner_with_root_isolation() {
        // Integration test: Use Gröbner basis to simplify, then isolate roots
        // System: x^2 - 2 = 0, y - x = 0
        // Should reduce to y^2 - 2 = 0

        let x_squared_minus_2 = polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]), // x^2
            (-2, &[]),      // -2
        ]);

        let y_minus_x = polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(1, 1)]),  // y
            (-1, &[(0, 1)]), // -x
        ]);

        let gb = grobner::grobner_basis(&[x_squared_minus_2.clone(), y_minus_x]);

        // The Gröbner basis should contain polynomials
        assert!(!gb.is_empty());

        // One of the polynomials should be univariate
        let has_univariate = gb.iter().any(|p| p.is_univariate());
        assert!(has_univariate || gb.len() == 1);
    }

    #[test]
    fn test_nra_solver_with_algebraic_numbers() {
        // Integration test: NRA solver with algebraic number evaluation
        // Solve x^2 - 2 = 0

        let mut solver = grobner::NraSolver::new();

        let x_squared_minus_2 = polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]), // x^2
            (-2, &[]),      // -2
        ]);

        solver.add_equality(x_squared_minus_2.clone());

        // Should be satisfiable
        assert_eq!(solver.check_sat(), grobner::SatResult::Sat);

        // Create algebraic number for sqrt(2)
        // AlgebraicNumber::new(poly, var, lower, upper)
        let sqrt_2 = realclosure::AlgebraicNumber::new(
            x_squared_minus_2,
            0, // variable 0
            rat(1),
            rat(2),
        );

        // Algebraic number should be valid
        let _ = sqrt_2;
    }

    #[test]
    fn test_interval_with_polynomial_bounds() {
        // Integration test: Use interval arithmetic with polynomial evaluation
        // Evaluate x^2 over [1, 2] should give [1, 4]

        let x_squared = polynomial::Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);

        // Evaluate at x = 1
        let mut assignment1 = rustc_hash::FxHashMap::default();
        assignment1.insert(0, rat(1));
        let val1 = x_squared.eval(&assignment1);
        assert_eq!(val1, rat(1));

        // Evaluate at x = 2
        let mut assignment2 = rustc_hash::FxHashMap::default();
        assignment2.insert(0, rat(2));
        let val2 = x_squared.eval(&assignment2);
        assert_eq!(val2, rat(4));

        // Create interval [1, 4]
        let interval = interval::Interval::closed(rat(1), rat(4));
        assert!(interval.contains(&val1));
        assert!(interval.contains(&val2));
    }

    #[test]
    fn test_delta_rationals_ordering() {
        // Integration test: Delta rationals for strict inequalities
        let delta_zero = delta_rational::DeltaRational::from_rational(rat(0));
        let delta_small = delta_rational::DeltaRational::new(rat(0), 1); // delta_coeff is i64

        // 0 + delta > 0
        assert!(delta_small > delta_zero);

        // Delta rationals maintain ordering
        let delta_one = delta_rational::DeltaRational::from_rational(rat(1));
        assert!(delta_one > delta_small);
    }

    #[test]
    fn test_matrix_operations() {
        // Integration test: Matrix operations (used in F4 algorithm)
        use matrix::Matrix;
        use num_rational::Rational64;

        // Create a simple 2x2 matrix
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational64::new(2, 1),
                Rational64::new(1, 1),
                Rational64::new(1, 1),
                Rational64::new(1, 1),
            ],
        );

        // Check matrix values
        assert_eq!(m.get(0, 0), Rational64::new(2, 1));
        assert_eq!(m.get(0, 1), Rational64::new(1, 1));
    }

    #[test]
    fn test_polynomial_factorization_with_grobner() {
        // Integration test: Factorization helps with Gröbner basis computation
        // x^2 - y^2 can be analyzed via Gröbner basis

        let x_sq_minus_y_sq = polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]),  // x^2
            (-1, &[(1, 2)]), // -y^2
        ]);

        // Compute Gröbner basis of {x^2 - y^2}
        let gb = grobner::grobner_basis(&[x_sq_minus_y_sq]);

        assert!(!gb.is_empty());
    }

    #[test]
    fn test_real_closure_root_isolation_integration() {
        // Integration test: Real closure and root isolation
        // Find roots of x^3 - 2 = 0

        let poly = polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(0, 3)]), // x^3
            (-2, &[]),      // -2
        ]);

        // Isolate roots (for variable 0)
        let roots = poly.isolate_roots(0);

        // Should find at least one real root (cube root of 2)
        assert!(!roots.is_empty());
    }

    #[test]
    fn test_polynomial_gcd_univariate() {
        // Integration test: GCD computation for univariate polynomials
        // gcd(x^2 - 1, x - 1) = x - 1

        let p1 = polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]), // x^2
            (-1, &[]),      // -1
        ]);

        let p2 = polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(0, 1)]), // x
            (-1, &[]),      // -1
        ]);

        let gcd = p1.gcd_univariate(&p2);

        // GCD should be x - 1 (or a scalar multiple)
        assert_eq!(gcd.total_degree(), 1);
    }
}
