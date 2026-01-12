//! Rational function arithmetic.
//!
//! This module provides operations on rational functions, which are quotients
//! of polynomials: f(x) = p(x) / q(x).
//!
//! Rational functions are useful for:
//! - Symbolic computation
//! - Partial fraction decomposition
//! - Control theory and signal processing
//! - Approximation theory

use crate::polynomial::{Polynomial, Var};
use num_rational::BigRational;
use num_traits::Zero;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A rational function represented as numerator / denominator.
///
/// Invariants:
/// - The denominator is never the zero polynomial
/// - The representation is kept in reduced form (gcd of numerator and denominator is 1)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalFunction {
    numerator: Polynomial,
    denominator: Polynomial,
}

impl RationalFunction {
    /// Create a new rational function from numerator and denominator.
    ///
    /// # Arguments
    ///
    /// * `numerator` - The numerator polynomial
    /// * `denominator` - The denominator polynomial
    ///
    /// # Panics
    ///
    /// Panics if the denominator is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use oxiz_math::rational_function::RationalFunction;
    ///
    /// let num = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]); // x
    /// let den = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]); // x + 1
    /// let rf = RationalFunction::new(num, den);
    /// ```
    pub fn new(numerator: Polynomial, denominator: Polynomial) -> Self {
        assert!(!denominator.is_zero(), "Denominator cannot be zero");

        let mut rf = RationalFunction {
            numerator,
            denominator,
        };
        rf.reduce();
        rf
    }

    /// Create a rational function from a polynomial (denominator = 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use oxiz_math::rational_function::RationalFunction;
    ///
    /// let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]); // x^2
    /// let rf = RationalFunction::from_polynomial(p);
    /// ```
    pub fn from_polynomial(p: Polynomial) -> Self {
        RationalFunction {
            numerator: p,
            denominator: Polynomial::one(),
        }
    }

    /// Create a rational function from a constant.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::rational_function::RationalFunction;
    ///
    /// let rf = RationalFunction::from_constant(BigRational::from_integer(BigInt::from(5)));
    /// ```
    pub fn from_constant(c: BigRational) -> Self {
        RationalFunction {
            numerator: Polynomial::constant(c),
            denominator: Polynomial::one(),
        }
    }

    /// Get the numerator.
    pub fn numerator(&self) -> &Polynomial {
        &self.numerator
    }

    /// Get the denominator.
    pub fn denominator(&self) -> &Polynomial {
        &self.denominator
    }

    /// Check if the rational function is zero.
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Check if the rational function is constant.
    pub fn is_constant(&self) -> bool {
        self.numerator.is_constant() && self.denominator.is_constant()
    }

    /// Reduce the rational function to lowest terms.
    ///
    /// Divides both numerator and denominator by their GCD.
    fn reduce(&mut self) {
        if self.numerator.is_zero() {
            self.denominator = Polynomial::one();
            return;
        }

        // For univariate case, compute GCD and use pseudo_div to reduce
        if self.numerator.is_univariate() && self.denominator.is_univariate() {
            let gcd = self.numerator.gcd_univariate(&self.denominator);
            if !gcd.is_constant() {
                // Use pseudo_div_univariate to divide both by GCD
                let (q1, r1) = self.numerator.pseudo_div_univariate(&gcd);
                let (q2, r2) = self.denominator.pseudo_div_univariate(&gcd);

                // If remainders are zero, we can reduce
                if r1.is_zero() && r2.is_zero() {
                    self.numerator = q1;
                    self.denominator = q2;
                }
            }
        }
        // For multivariate or when GCD reduction fails, leave as is
    }

    /// Evaluate the rational function at a point.
    ///
    /// # Arguments
    ///
    /// * `assignment` - Variable assignments
    ///
    /// # Returns
    ///
    /// The value of the rational function, or None if the denominator evaluates to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    /// use oxiz_math::rational_function::RationalFunction;
    /// use rustc_hash::FxHashMap;
    ///
    /// let num = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]); // x
    /// let den = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]); // x + 1
    /// let rf = RationalFunction::new(num, den);
    ///
    /// let mut assignment = FxHashMap::default();
    /// assignment.insert(0, BigRational::from_integer(BigInt::from(2)));
    /// let val = rf.eval(&assignment).unwrap();
    /// assert_eq!(val, BigRational::new(BigInt::from(2), BigInt::from(3))); // 2/3
    /// ```
    pub fn eval(
        &self,
        assignment: &rustc_hash::FxHashMap<Var, BigRational>,
    ) -> Option<BigRational> {
        let num_val = self.numerator.eval(assignment);
        let den_val = self.denominator.eval(assignment);

        if den_val.is_zero() {
            None
        } else {
            Some(num_val / den_val)
        }
    }

    /// Compute the derivative of the rational function.
    ///
    /// Uses the quotient rule: (p/q)' = (p'q - pq') / q^2
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use oxiz_math::rational_function::RationalFunction;
    ///
    /// let num = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]); // x
    /// let den = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]); // x + 1
    /// let rf = RationalFunction::new(num, den);
    /// let derivative = rf.derivative(0);
    /// ```
    pub fn derivative(&self, var: Var) -> RationalFunction {
        // (p/q)' = (p'q - pq') / q^2
        let p_prime = self.numerator.derivative(var);
        let q_prime = self.denominator.derivative(var);

        let numerator = p_prime * self.denominator.clone() - self.numerator.clone() * q_prime;
        let denominator = self.denominator.clone() * self.denominator.clone();

        RationalFunction::new(numerator, denominator)
    }

    /// Simplify the rational function.
    ///
    /// This is an alias for reducing to lowest terms.
    pub fn simplify(&mut self) {
        self.reduce();
    }

    /// Check if this is a proper rational function (degree of numerator < degree of denominator).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use oxiz_math::rational_function::RationalFunction;
    ///
    /// let num = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]); // x
    /// let den = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]); // x^2
    /// let rf = RationalFunction::new(num, den);
    /// assert!(rf.is_proper());
    /// ```
    pub fn is_proper(&self) -> bool {
        self.numerator.total_degree() < self.denominator.total_degree()
    }

    /// Perform polynomial long division to separate improper rational functions.
    ///
    /// For an improper rational function N(x)/D(x) where deg(N) >= deg(D),
    /// returns (Q(x), R(x)/D(x)) where N(x)/D(x) = Q(x) + R(x)/D(x)
    /// and R(x)/D(x) is proper.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable for univariate division
    ///
    /// # Returns
    ///
    /// * `(quotient_polynomial, proper_remainder)` if the function is improper and univariate
    /// * `None` if the function is proper or not univariate
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use oxiz_math::rational_function::RationalFunction;
    ///
    /// // (x^2 + 1) / x = x + 1/x
    /// let num = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[])]);
    /// let den = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]);
    /// let rf = RationalFunction::new(num, den);
    ///
    /// let (quotient, remainder) = rf.polynomial_division(0).unwrap();
    /// // quotient should be x, remainder should be 1/x
    /// ```
    pub fn polynomial_division(&self, _var: Var) -> Option<(Polynomial, RationalFunction)> {
        // Only works for univariate and improper fractions
        if !self.numerator.is_univariate() || !self.denominator.is_univariate() {
            return None;
        }

        if self.is_proper() {
            return None;
        }

        // Perform polynomial division
        let (quotient, remainder) = self.numerator.pseudo_div_univariate(&self.denominator);

        // Create the proper fraction from the remainder
        let proper_fraction = RationalFunction::new(remainder, self.denominator.clone());

        Some((quotient, proper_fraction))
    }

    /// Compute partial fraction decomposition for simple cases.
    ///
    /// For a proper rational function with a factored denominator of distinct linear factors,
    /// decomposes into a sum of simpler fractions.
    ///
    /// **Note:** This is a simplified implementation that works for:
    /// - Proper rational functions (deg(numerator) < deg(denominator))
    /// - Univariate polynomials
    /// - Denominators that are products of distinct linear factors
    ///
    /// For more complex cases (repeated factors, irreducible quadratics),
    /// use specialized computer algebra systems.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable
    ///
    /// # Returns
    ///
    /// Vector of simpler rational functions that sum to the original,
    /// or None if decomposition is not applicable
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    /// use oxiz_math::rational_function::RationalFunction;
    ///
    /// // Example: 1 / (x(x-1)) = A/x + B/(x-1)
    /// // where A = -1, B = 1
    /// let num = Polynomial::from_coeffs_int(&[(1, &[])]);  // 1
    /// let den = Polynomial::from_coeffs_int(&[             // x^2 - x
    ///     (1, &[(0, 2)]),   // x^2
    ///     (-1, &[(0, 1)]),  // -x
    /// ]);
    /// let rf = RationalFunction::new(num, den);
    ///
    /// // Partial fraction decomposition (if factors are available)
    /// // This is a placeholder - full implementation would require factorization
    /// ```
    pub fn partial_fraction_decomposition(&self, _var: Var) -> Option<Vec<RationalFunction>> {
        // Check if proper
        if !self.is_proper() {
            return None;
        }

        // Check if univariate
        if !self.numerator.is_univariate() || !self.denominator.is_univariate() {
            return None;
        }

        // For now, return None as a full implementation would require:
        // 1. Factorization of the denominator into irreducible factors
        // 2. Solving systems of equations for unknown coefficients
        // 3. Handling repeated factors and irreducible quadratics
        //
        // This is left as a future enhancement when polynomial factorization
        // over rationals is more complete in the polynomial module.
        None
    }
}

impl Add for RationalFunction {
    type Output = RationalFunction;

    /// Add two rational functions: p/q + r/s = (ps + qr) / (qs)
    fn add(self, other: RationalFunction) -> RationalFunction {
        let numerator = self.numerator.clone() * other.denominator.clone()
            + other.numerator.clone() * self.denominator.clone();
        let denominator = self.denominator * other.denominator;
        RationalFunction::new(numerator, denominator)
    }
}

impl Add for &RationalFunction {
    type Output = RationalFunction;

    fn add(self, other: &RationalFunction) -> RationalFunction {
        let numerator = self.numerator.clone() * other.denominator.clone()
            + other.numerator.clone() * self.denominator.clone();
        let denominator = self.denominator.clone() * other.denominator.clone();
        RationalFunction::new(numerator, denominator)
    }
}

impl Sub for RationalFunction {
    type Output = RationalFunction;

    /// Subtract two rational functions: p/q - r/s = (ps - qr) / (qs)
    fn sub(self, other: RationalFunction) -> RationalFunction {
        let numerator = self.numerator.clone() * other.denominator.clone()
            - other.numerator.clone() * self.denominator.clone();
        let denominator = self.denominator * other.denominator;
        RationalFunction::new(numerator, denominator)
    }
}

impl Sub for &RationalFunction {
    type Output = RationalFunction;

    fn sub(self, other: &RationalFunction) -> RationalFunction {
        let numerator = self.numerator.clone() * other.denominator.clone()
            - other.numerator.clone() * self.denominator.clone();
        let denominator = self.denominator.clone() * other.denominator.clone();
        RationalFunction::new(numerator, denominator)
    }
}

impl Mul for RationalFunction {
    type Output = RationalFunction;

    /// Multiply two rational functions: (p/q) * (r/s) = (pr) / (qs)
    fn mul(self, other: RationalFunction) -> RationalFunction {
        let numerator = self.numerator * other.numerator;
        let denominator = self.denominator * other.denominator;
        RationalFunction::new(numerator, denominator)
    }
}

impl Mul for &RationalFunction {
    type Output = RationalFunction;

    fn mul(self, other: &RationalFunction) -> RationalFunction {
        let numerator = self.numerator.clone() * other.numerator.clone();
        let denominator = self.denominator.clone() * other.denominator.clone();
        RationalFunction::new(numerator, denominator)
    }
}

impl Div for RationalFunction {
    type Output = RationalFunction;

    /// Divide two rational functions: (p/q) / (r/s) = (ps) / (qr)
    fn div(self, other: RationalFunction) -> RationalFunction {
        let numerator = self.numerator * other.denominator;
        let denominator = self.denominator * other.numerator;
        RationalFunction::new(numerator, denominator)
    }
}

impl Div for &RationalFunction {
    type Output = RationalFunction;

    fn div(self, other: &RationalFunction) -> RationalFunction {
        let numerator = self.numerator.clone() * other.denominator.clone();
        let denominator = self.denominator.clone() * other.numerator.clone();
        RationalFunction::new(numerator, denominator)
    }
}

impl Neg for RationalFunction {
    type Output = RationalFunction;

    /// Negate a rational function: -(p/q) = (-p)/q
    fn neg(self) -> RationalFunction {
        RationalFunction {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl Neg for &RationalFunction {
    type Output = RationalFunction;

    fn neg(self) -> RationalFunction {
        RationalFunction {
            numerator: -self.numerator.clone(),
            denominator: self.denominator.clone(),
        }
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
    fn test_rational_function_creation() {
        let num = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]); // x
        let den = Polynomial::from_coeffs_int(&[(1, &[])]); // 1
        let rf = RationalFunction::new(num, den);
        assert!(!rf.is_zero());
    }

    #[test]
    #[should_panic(expected = "Denominator cannot be zero")]
    fn test_rational_function_zero_denominator() {
        let num = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]); // x
        let den = Polynomial::zero();
        let _rf = RationalFunction::new(num, den);
    }

    #[test]
    fn test_rational_function_addition() {
        // x/1 + 1/1 = (x + 1)/1
        let rf1 = RationalFunction::from_polynomial(
            Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]), // x
        );
        let rf2 = RationalFunction::from_constant(rat(1));

        let sum = rf1 + rf2;
        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(0, rat(2));
        assert_eq!(sum.eval(&assignment).unwrap(), rat(3)); // 2 + 1 = 3
    }

    #[test]
    fn test_rational_function_multiplication() {
        // (x/1) * (2/1) = (2x)/1
        let rf1 = RationalFunction::from_polynomial(
            Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]), // x
        );
        let rf2 = RationalFunction::from_constant(rat(2));

        let prod = rf1 * rf2;
        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(0, rat(3));
        assert_eq!(prod.eval(&assignment).unwrap(), rat(6)); // 2 * 3 = 6
    }

    #[test]
    fn test_rational_function_division() {
        // (x) / (x+1)
        let num = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]); // x
        let den = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]); // x + 1

        let rf = RationalFunction::new(num, den.clone());

        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(0, rat(2));
        assert_eq!(
            rf.eval(&assignment).unwrap(),
            BigRational::new(BigInt::from(2), BigInt::from(3))
        );
    }

    #[test]
    fn test_rational_function_derivative() {
        // d/dx(x/(x+1)) = 1/(x+1)^2
        let num = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]); // x
        let den = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[])]); // x + 1

        let rf = RationalFunction::new(num, den);
        let deriv = rf.derivative(0);

        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(0, rat(1));
        // At x=1: derivative = 1/(1+1)^2 = 1/4
        assert_eq!(
            deriv.eval(&assignment).unwrap(),
            BigRational::new(BigInt::from(1), BigInt::from(4))
        );
    }

    #[test]
    fn test_rational_function_reduction() {
        // (x^2 - 1) / (x - 1) should reduce to (x + 1) / 1
        let num = Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]), // x^2
            (-1, &[]),      // -1
        ]);
        let den = Polynomial::from_coeffs_int(&[
            (1, &[(0, 1)]), // x
            (-1, &[]),      // -1
        ]);

        let rf = RationalFunction::new(num, den);

        // Should be reduced
        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(0, rat(2));
        assert_eq!(rf.eval(&assignment).unwrap(), rat(3)); // (x + 1) at x=2 is 3
    }

    #[test]
    fn test_rational_function_is_constant() {
        let rf = RationalFunction::from_constant(rat(5));
        assert!(rf.is_constant());

        let rf2 = RationalFunction::from_polynomial(
            Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]), // x
        );
        assert!(!rf2.is_constant());
    }

    #[test]
    fn test_rational_function_negation() {
        let rf = RationalFunction::from_constant(rat(5));
        let neg_rf = -rf;

        let assignment = rustc_hash::FxHashMap::default();
        assert_eq!(neg_rf.eval(&assignment).unwrap(), rat(-5));
    }
}
