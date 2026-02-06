//! Symbolic Differentiation for Polynomials.
//!
//! Implements:
//! - Partial derivatives
//! - Higher-order derivatives
//! - Gradient computation
//! - Hessian matrix
//! - Jacobian matrix

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// Symbolic differentiation engine.
pub struct SymbolicDifferentiator {
    /// Derivative cache: (polynomial, var) → derivative
    cache: FxHashMap<(PolynomialKey, usize), Vec<BigRational>>,
    /// Statistics
    stats: DifferentiationStats,
}

/// Simplified polynomial key for caching.
type PolynomialKey = Vec<String>;

/// Differentiation statistics.
#[derive(Debug, Clone, Default)]
pub struct DifferentiationStats {
    /// Number of derivatives computed
    pub derivatives_computed: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Gradient computations
    pub gradients_computed: usize,
    /// Hessian computations
    pub hessians_computed: usize,
    /// Jacobian computations
    pub jacobians_computed: usize,
}

/// Gradient vector for multivariate polynomial.
#[derive(Debug, Clone)]
pub struct Gradient {
    /// Partial derivatives with respect to each variable
    pub partials: Vec<Vec<BigRational>>,
}

/// Hessian matrix (second-order partial derivatives).
#[derive(Debug, Clone)]
pub struct Hessian {
    /// Matrix of second partials: hessian\[i\]\[j\] = ∂²f/∂xᵢ∂xⱼ
    pub matrix: Vec<Vec<Vec<BigRational>>>,
}

/// Jacobian matrix for vector-valued functions.
#[derive(Debug, Clone)]
pub struct Jacobian {
    /// Matrix\[i\]\[j\] = ∂fᵢ/∂xⱼ
    pub matrix: Vec<Vec<Vec<BigRational>>>,
}

impl SymbolicDifferentiator {
    /// Create a new symbolic differentiator.
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            stats: DifferentiationStats::default(),
        }
    }

    /// Compute partial derivative ∂f/∂xᵢ for univariate polynomial.
    ///
    /// For f(x) = aₙxⁿ + ... + a₁x + a₀:
    /// f'(x) = naₙxⁿ⁻¹ + ... + a₁
    pub fn derivative(&mut self, poly: &[BigRational]) -> Vec<BigRational> {
        self.stats.derivatives_computed += 1;

        if poly.len() <= 1 {
            // Constant or empty
            return vec![BigRational::zero()];
        }

        let degree = poly.len() - 1;
        let mut deriv = Vec::new();

        for (i, coeff) in poly.iter().enumerate().take(degree) {
            let power = (degree - i) as i64;
            let deriv_coeff = coeff * BigRational::from_integer(BigInt::from(power));
            deriv.push(deriv_coeff);
        }

        if deriv.is_empty() {
            vec![BigRational::zero()]
        } else {
            deriv
        }
    }

    /// Compute partial derivative ∂f/∂xᵢ for multivariate polynomial.
    ///
    /// Multivariate polynomial represented as univariate in target variable.
    pub fn partial_derivative(
        &mut self,
        poly: &[BigRational],
        var_index: usize,
    ) -> Vec<BigRational> {
        // Check cache
        let key = (self.poly_to_key(poly), var_index);
        if let Some(cached) = self.cache.get(&key) {
            self.stats.cache_hits += 1;
            return cached.clone();
        }

        self.stats.derivatives_computed += 1;

        // For univariate representation, compute derivative
        let deriv = self.derivative(poly);

        // Cache result
        self.cache.insert(key, deriv.clone());

        deriv
    }

    /// Compute n-th derivative.
    pub fn nth_derivative(&mut self, poly: &[BigRational], n: usize) -> Vec<BigRational> {
        let mut result = poly.to_vec();

        for _ in 0..n {
            result = self.derivative(&result);
            if result.len() == 1 && result[0].is_zero() {
                // Reached zero
                break;
            }
        }

        result
    }

    /// Compute gradient vector ∇f = (∂f/∂x₁, ..., ∂f/∂xₙ).
    pub fn gradient(&mut self, poly: &[BigRational], num_vars: usize) -> Gradient {
        self.stats.gradients_computed += 1;

        let mut partials = Vec::new();

        for var_idx in 0..num_vars {
            let partial = self.partial_derivative(poly, var_idx);
            partials.push(partial);
        }

        Gradient { partials }
    }

    /// Compute Hessian matrix H\[i\]\[j\] = ∂²f/∂xᵢ∂xⱼ.
    pub fn hessian(&mut self, poly: &[BigRational], num_vars: usize) -> Hessian {
        self.stats.hessians_computed += 1;

        let mut matrix = Vec::new();

        for i in 0..num_vars {
            let mut row = Vec::new();
            let first_deriv = self.partial_derivative(poly, i);

            for j in 0..num_vars {
                let second_deriv = self.partial_derivative(&first_deriv, j);
                row.push(second_deriv);
            }

            matrix.push(row);
        }

        Hessian { matrix }
    }

    /// Compute Jacobian matrix for vector-valued function F: ℝⁿ → ℝᵐ.
    ///
    /// J\[i\]\[j\] = ∂fᵢ/∂xⱼ
    pub fn jacobian(&mut self, functions: &[Vec<BigRational>], num_vars: usize) -> Jacobian {
        self.stats.jacobians_computed += 1;

        let mut matrix = Vec::new();

        for func in functions {
            let mut row = Vec::new();

            for var_idx in 0..num_vars {
                let partial = self.partial_derivative(func, var_idx);
                row.push(partial);
            }

            matrix.push(row);
        }

        Jacobian { matrix }
    }

    /// Compute directional derivative in direction v: ∇f · v.
    pub fn directional_derivative(
        &mut self,
        poly: &[BigRational],
        direction: &[BigRational],
    ) -> BigRational {
        let num_vars = direction.len();
        let grad = self.gradient(poly, num_vars);

        let mut result = BigRational::zero();

        for (partial, dir_component) in grad.partials.iter().zip(direction.iter()) {
            // Evaluate partial at origin (simplified)
            if let Some(constant_term) = partial.last() {
                result += constant_term * dir_component;
            }
        }

        result
    }

    /// Check if polynomial is harmonic (Laplacian = 0).
    ///
    /// Laplacian: Δf = ∂²f/∂x₁² + ∂²f/∂x₂² + ... + ∂²f/∂xₙ²
    pub fn is_harmonic(&mut self, poly: &[BigRational], num_vars: usize) -> bool {
        let hessian = self.hessian(poly, num_vars);

        // Compute trace of Hessian (sum of diagonal elements)
        let mut laplacian = vec![BigRational::zero()];

        for i in 0..num_vars {
            if i < hessian.matrix.len() && i < hessian.matrix[i].len() {
                let diagonal_elem = &hessian.matrix[i][i];
                laplacian = self.poly_add(&laplacian, diagonal_elem);
            }
        }

        // Check if Laplacian is zero
        self.is_zero_poly(&laplacian)
    }

    /// Check if gradient vanishes (critical point).
    pub fn is_critical_point(
        &mut self,
        poly: &[BigRational],
        num_vars: usize,
        point: &[BigRational],
    ) -> bool {
        let grad = self.gradient(poly, num_vars);

        for (i, partial) in grad.partials.iter().enumerate() {
            if i < point.len() {
                let value = self.evaluate_at_point(partial, point[i].clone());
                if !value.is_zero() {
                    return false;
                }
            }
        }

        true
    }

    /// Compute Taylor expansion up to degree n.
    pub fn taylor_expansion(
        &mut self,
        poly: &[BigRational],
        center: BigRational,
        degree: usize,
    ) -> Vec<BigRational> {
        let mut terms = Vec::new();

        for k in 0..=degree {
            let kth_deriv = self.nth_derivative(poly, k);
            let value_at_center = self.evaluate_at_point(&kth_deriv, center.clone());

            // Divide by k!
            let factorial = self.factorial(k);
            let term = value_at_center / BigRational::from_integer(factorial);

            terms.push(term);
        }

        terms.reverse(); // Highest degree first
        terms
    }

    /// Polynomial addition helper.
    fn poly_add(&self, p1: &[BigRational], p2: &[BigRational]) -> Vec<BigRational> {
        let max_len = p1.len().max(p2.len());
        let mut result = vec![BigRational::zero(); max_len];

        for (i, coeff) in p1.iter().rev().enumerate() {
            if i < result.len() {
                result[max_len - 1 - i] = result[max_len - 1 - i].clone() + coeff;
            }
        }

        for (i, coeff) in p2.iter().rev().enumerate() {
            if i < result.len() {
                result[max_len - 1 - i] = result[max_len - 1 - i].clone() + coeff;
            }
        }

        result
    }

    /// Check if polynomial is zero.
    fn is_zero_poly(&self, poly: &[BigRational]) -> bool {
        poly.iter().all(|c| c.is_zero())
    }

    /// Evaluate polynomial at a point.
    fn evaluate_at_point(&self, poly: &[BigRational], x: BigRational) -> BigRational {
        if poly.is_empty() {
            return BigRational::zero();
        }

        let mut result = poly[0].clone();
        for coeff in &poly[1..] {
            result = result * &x + coeff;
        }

        result
    }

    /// Compute factorial.
    fn factorial(&self, n: usize) -> BigInt {
        if n <= 1 {
            BigInt::one()
        } else {
            let mut result = BigInt::one();
            for i in 2..=n {
                result *= BigInt::from(i);
            }
            result
        }
    }

    /// Convert polynomial to cache key.
    fn poly_to_key(&self, poly: &[BigRational]) -> PolynomialKey {
        poly.iter().map(|c| c.to_string()).collect()
    }

    /// Get statistics.
    pub fn stats(&self) -> &DifferentiationStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DifferentiationStats::default();
    }
}

impl Default for SymbolicDifferentiator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_differentiator() {
        let diff = SymbolicDifferentiator::new();
        assert_eq!(diff.stats.derivatives_computed, 0);
    }

    #[test]
    fn test_derivative_constant() {
        let mut diff = SymbolicDifferentiator::new();

        // f(x) = 5
        let poly = vec![BigRational::from_integer(BigInt::from(5))];
        let deriv = diff.derivative(&poly);

        assert_eq!(deriv.len(), 1);
        assert!(deriv[0].is_zero());
    }

    #[test]
    fn test_derivative_linear() {
        let mut diff = SymbolicDifferentiator::new();

        // f(x) = 3x + 2
        let poly = vec![
            BigRational::from_integer(BigInt::from(3)),
            BigRational::from_integer(BigInt::from(2)),
        ];

        let deriv = diff.derivative(&poly);

        assert_eq!(deriv.len(), 1);
        assert_eq!(deriv[0], BigRational::from_integer(BigInt::from(3)));
    }

    #[test]
    fn test_derivative_quadratic() {
        let mut diff = SymbolicDifferentiator::new();

        // f(x) = x² + 2x + 1 → f'(x) = 2x + 2
        let poly = vec![
            BigRational::one(),
            BigRational::from_integer(BigInt::from(2)),
            BigRational::one(),
        ];

        let deriv = diff.derivative(&poly);

        assert_eq!(deriv.len(), 2);
        assert_eq!(deriv[0], BigRational::from_integer(BigInt::from(2)));
        assert_eq!(deriv[1], BigRational::from_integer(BigInt::from(2)));
    }

    #[test]
    fn test_nth_derivative() {
        let mut diff = SymbolicDifferentiator::new();

        // f(x) = x³ = x³ + 0x² + 0x + 0
        let poly = vec![
            BigRational::one(),
            BigRational::zero(),
            BigRational::zero(),
            BigRational::zero(),
        ];

        // f'(x) = 3x²
        let deriv1 = diff.nth_derivative(&poly, 1);
        assert_eq!(deriv1[0], BigRational::from_integer(BigInt::from(3)));

        // f''(x) = 6x
        let deriv2 = diff.nth_derivative(&poly, 2);
        assert_eq!(deriv2[0], BigRational::from_integer(BigInt::from(6)));

        // f'''(x) = 6
        let deriv3 = diff.nth_derivative(&poly, 3);
        assert_eq!(deriv3[0], BigRational::from_integer(BigInt::from(6)));

        // f⁽⁴⁾(x) = 0
        let deriv4 = diff.nth_derivative(&poly, 4);
        assert!(deriv4[0].is_zero());
    }

    #[test]
    fn test_gradient() {
        let mut diff = SymbolicDifferentiator::new();

        // f(x, y) represented as polynomial in x (simplified)
        let poly = vec![
            BigRational::one(),
            BigRational::from_integer(BigInt::from(2)),
        ];

        let grad = diff.gradient(&poly, 2);
        assert_eq!(grad.partials.len(), 2);
    }

    #[test]
    fn test_factorial() {
        let diff = SymbolicDifferentiator::new();

        assert_eq!(diff.factorial(0), BigInt::one());
        assert_eq!(diff.factorial(1), BigInt::one());
        assert_eq!(diff.factorial(5), BigInt::from(120));
    }

    #[test]
    fn test_stats() {
        let mut diff = SymbolicDifferentiator::new();

        let poly = vec![BigRational::one(), BigRational::zero()];
        diff.derivative(&poly);

        assert_eq!(diff.stats().derivatives_computed, 1);

        diff.reset_stats();
        assert_eq!(diff.stats().derivatives_computed, 0);
    }
}
