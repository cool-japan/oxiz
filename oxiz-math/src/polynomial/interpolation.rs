//! Polynomial Interpolation.
//!
//! Constructs polynomials from given points using various interpolation methods.
//!
//! ## Methods
//!
//! - **Lagrange**: Direct interpolation formula
//! - **Newton**: Divided differences for efficient evaluation
//! - **Hermite**: Interpolation with derivative constraints
//! - **Spline**: Piecewise polynomial interpolation
//!
//! ## References
//!
//! - "Numerical Analysis" (Burden & Faires, 2010)
//! - "Computer Algebra and Symbolic Computation" (Cohen, 2002)

use crate::polynomial::{Polynomial, Term, Var};
use num_rational::BigRational;
use num_traits::{One, Zero};

/// A point for interpolation (x, y).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Point {
    /// X-coordinate.
    pub x: BigRational,
    /// Y-coordinate (function value).
    pub y: BigRational,
}

impl Point {
    /// Create a new point.
    pub fn new(x: BigRational, y: BigRational) -> Self {
        Self { x, y }
    }

    /// Create from integers.
    pub fn from_ints(x: i64, y: i64) -> Self {
        Self {
            x: BigRational::from_integer(x.into()),
            y: BigRational::from_integer(y.into()),
        }
    }
}

/// A point with derivative information for Hermite interpolation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HermitePoint {
    /// X-coordinate.
    pub x: BigRational,
    /// Function value at x.
    pub y: BigRational,
    /// Derivative value at x.
    pub dy: BigRational,
}

impl HermitePoint {
    /// Create a new Hermite point.
    pub fn new(x: BigRational, y: BigRational, dy: BigRational) -> Self {
        Self { x, y, dy }
    }
}

/// Configuration for polynomial interpolation.
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Variable to use for interpolation.
    pub variable: Var,
    /// Enable numerical stability checks.
    pub check_stability: bool,
    /// Maximum degree allowed.
    pub max_degree: usize,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            variable: 0,
            check_stability: true,
            max_degree: 100,
        }
    }
}

/// Statistics for interpolation.
#[derive(Debug, Clone, Default)]
pub struct InterpolationStats {
    /// Interpolations performed.
    pub interpolations: u64,
    /// Average degree of interpolating polynomials.
    pub avg_degree: f64,
    /// Numerical instability warnings.
    pub instability_warnings: u64,
}

/// Polynomial interpolation engine.
pub struct PolynomialInterpolator {
    /// Configuration.
    config: InterpolationConfig,
    /// Statistics.
    stats: InterpolationStats,
}

impl PolynomialInterpolator {
    /// Create a new polynomial interpolator.
    pub fn new(config: InterpolationConfig) -> Self {
        Self {
            config,
            stats: InterpolationStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(InterpolationConfig::default())
    }

    /// Lagrange interpolation.
    ///
    /// Constructs the unique polynomial of degree ≤ n-1 passing through n points.
    ///
    /// Formula: P(x) = Σᵢ yᵢ · Lᵢ(x)
    /// where Lᵢ(x) = Πⱼ≠ᵢ (x - xⱼ) / (xᵢ - xⱼ)
    pub fn lagrange(&mut self, points: &[Point]) -> Result<Polynomial, InterpolationError> {
        if points.is_empty() {
            return Err(InterpolationError::NoPoints);
        }

        if points.len() > self.config.max_degree + 1 {
            return Err(InterpolationError::DegreeTooHigh);
        }

        self.stats.interpolations += 1;

        // Check for duplicate x-values
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                if points[i].x == points[j].x {
                    return Err(InterpolationError::DuplicateXValue);
                }
            }
        }

        let var = self.config.variable;
        let mut result = Polynomial::zero();

        // Compute each Lagrange basis polynomial
        for i in 0..points.len() {
            let mut basis = Polynomial::one();

            for j in 0..points.len() {
                if i == j {
                    continue;
                }

                // Multiply by (x - x_j) / (x_i - x_j)
                let numerator =
                    Polynomial::from_var(var) - Polynomial::constant(points[j].x.clone());
                let denominator = &points[i].x - &points[j].x;

                if denominator.is_zero() {
                    return Err(InterpolationError::DuplicateXValue);
                }

                basis = &basis * &numerator;
                // Divide by constant: multiply by 1/denominator
                let inv = BigRational::from_integer(1.into()) / denominator;
                basis = Self::multiply_by_constant(&basis, &inv);
            }

            // Multiply by y_i and add to result
            basis = Self::multiply_by_constant(&basis, &points[i].y);
            result = &result + &basis;
        }

        self.update_degree_stats(result.total_degree() as usize);

        Ok(result)
    }

    /// Newton interpolation using divided differences.
    ///
    /// More efficient than Lagrange for evaluation at multiple points.
    ///
    /// Formula: P(x) = f\[x₀\] + f\[x₀,x₁\](x-x₀) + f\[x₀,x₁,x₂\](x-x₀)(x-x₁) + ...
    pub fn newton(&mut self, points: &[Point]) -> Result<Polynomial, InterpolationError> {
        if points.is_empty() {
            return Err(InterpolationError::NoPoints);
        }

        if points.len() > self.config.max_degree + 1 {
            return Err(InterpolationError::DegreeTooHigh);
        }

        self.stats.interpolations += 1;

        // Compute divided differences table
        let n = points.len();
        let mut dd = vec![vec![BigRational::zero(); n]; n];

        // First column: function values
        for i in 0..n {
            dd[i][0] = points[i].y.clone();
        }

        // Compute higher-order divided differences
        for j in 1..n {
            for i in 0..(n - j) {
                let numerator = &dd[i + 1][j - 1] - &dd[i][j - 1];
                let denominator = &points[i + j].x - &points[i].x;

                if denominator.is_zero() {
                    return Err(InterpolationError::DuplicateXValue);
                }

                dd[i][j] = numerator / denominator;
            }
        }

        let var = self.config.variable;
        let mut result = Polynomial::constant(dd[0][0].clone());

        // Build Newton polynomial
        let mut product = Polynomial::one();

        for i in 1..n {
            // Multiply by (x - x_{i-1})
            let factor = Polynomial::from_var(var) - Polynomial::constant(points[i - 1].x.clone());
            product = &product * &factor;

            // Add divided difference term
            let term = Self::multiply_by_constant(&product, &dd[0][i]);
            result = &result + &term;
        }

        self.update_degree_stats(result.total_degree() as usize);

        Ok(result)
    }

    /// Hermite interpolation with derivatives.
    ///
    /// Interpolates both function values and derivatives at given points.
    pub fn hermite(&mut self, points: &[HermitePoint]) -> Result<Polynomial, InterpolationError> {
        if points.is_empty() {
            return Err(InterpolationError::NoPoints);
        }

        let n = points.len();
        let total_conditions = 2 * n;

        if total_conditions > self.config.max_degree + 1 {
            return Err(InterpolationError::DegreeTooHigh);
        }

        self.stats.interpolations += 1;

        // Convert to divided differences with duplicate points
        // For Hermite interpolation, we use f[x_i, x_i] = f'(x_i)
        let mut extended_points = Vec::new();

        for point in points {
            // Add each point twice (for value and derivative)
            extended_points.push(Point::new(point.x.clone(), point.y.clone()));
            extended_points.push(Point::new(point.x.clone(), point.y.clone()));
        }

        // Build divided difference table with special handling for duplicates
        let m = extended_points.len();
        let mut dd = vec![vec![BigRational::zero(); m]; m];

        // First column: function values
        for i in 0..m {
            dd[i][0] = extended_points[i].y.clone();
        }

        // Second column: handle derivatives for duplicate points
        for i in 0..(m - 1) {
            if extended_points[i].x == extended_points[i + 1].x {
                // Use derivative instead of divided difference
                let point_idx = i / 2;
                dd[i][1] = points[point_idx].dy.clone();
            } else {
                let numerator = &dd[i + 1][0] - &dd[i][0];
                let denominator = &extended_points[i + 1].x - &extended_points[i].x;

                if !denominator.is_zero() {
                    dd[i][1] = numerator / denominator;
                }
            }
        }

        // Higher-order divided differences
        for j in 2..m {
            for i in 0..(m - j) {
                let denominator = &extended_points[i + j].x - &extended_points[i].x;

                if !denominator.is_zero() {
                    let numerator = &dd[i + 1][j - 1] - &dd[i][j - 1];
                    dd[i][j] = numerator / denominator;
                }
            }
        }

        // Build polynomial using Newton form
        let var = self.config.variable;
        let mut result = Polynomial::constant(dd[0][0].clone());
        let mut product = Polynomial::one();

        for i in 1..m {
            let factor =
                Polynomial::from_var(var) - Polynomial::constant(extended_points[i - 1].x.clone());
            product = &product * &factor;

            let term = Self::multiply_by_constant(&product, &dd[0][i]);
            result = &result + &term;
        }

        self.update_degree_stats(result.total_degree() as usize);

        Ok(result)
    }

    /// Evaluate a polynomial at a point (for verification).
    pub fn evaluate(&self, poly: &Polynomial, x: &BigRational) -> BigRational {
        let mut result = BigRational::zero();

        for term in poly.terms() {
            let mut term_value = term.coeff.clone();

            for var_power in term.monomial.vars() {
                if var_power.var == self.config.variable {
                    // Compute x^power
                    let power_value = Self::power(x, var_power.power);
                    term_value *= power_value;
                }
            }

            result += term_value;
        }

        result
    }

    /// Compute x^n for positive integer n.
    fn power(x: &BigRational, n: u32) -> BigRational {
        if n == 0 {
            BigRational::one()
        } else if n == 1 {
            x.clone()
        } else {
            let mut result = x.clone();
            for _ in 1..n {
                result *= x;
            }
            result
        }
    }

    /// Update average degree statistics.
    fn update_degree_stats(&mut self, degree: usize) {
        let count = self.stats.interpolations;
        let old_avg = self.stats.avg_degree;
        self.stats.avg_degree = (old_avg * (count - 1) as f64 + degree as f64) / count as f64;
    }

    /// Get statistics.
    pub fn stats(&self) -> &InterpolationStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = InterpolationStats::default();
    }

    /// Helper: multiply a polynomial by a scalar constant.
    fn multiply_by_constant(poly: &Polynomial, scalar: &BigRational) -> Polynomial {
        if scalar.is_zero() {
            return Polynomial::zero();
        }
        if scalar.is_one() {
            return poly.clone();
        }

        // Multiply each term's coefficient by the scalar
        let new_terms: Vec<Term> = poly
            .terms()
            .iter()
            .map(|term| Term {
                coeff: &term.coeff * scalar,
                monomial: term.monomial.clone(),
            })
            .collect();

        Polynomial::from_terms(new_terms, poly.order)
    }
}

/// Errors that can occur during interpolation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpolationError {
    /// No interpolation points provided.
    NoPoints,
    /// Duplicate x-values in interpolation points.
    DuplicateXValue,
    /// Degree of interpolating polynomial would be too high.
    DegreeTooHigh,
    /// Numerical instability detected.
    NumericalInstability,
}

impl std::fmt::Display for InterpolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpolationError::NoPoints => write!(f, "no interpolation points provided"),
            InterpolationError::DuplicateXValue => write!(f, "duplicate x-values in points"),
            InterpolationError::DegreeTooHigh => write!(f, "degree too high"),
            InterpolationError::NumericalInstability => write!(f, "numerical instability"),
        }
    }
}

impl std::error::Error for InterpolationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point::from_ints(1, 2);
        assert_eq!(p.x, BigRational::from_integer(1.into()));
        assert_eq!(p.y, BigRational::from_integer(2.into()));
    }

    #[test]
    fn test_lagrange_linear() {
        let mut interpolator = PolynomialInterpolator::default_config();

        let points = vec![Point::from_ints(0, 1), Point::from_ints(1, 3)];

        let poly = interpolator
            .lagrange(&points)
            .expect("interpolation failed");

        // Should get p(x) = 1 + 2x
        let y0 = interpolator.evaluate(&poly, &BigRational::zero());
        let y1 = interpolator.evaluate(&poly, &BigRational::one());

        assert_eq!(y0, BigRational::from_integer(1.into()));
        assert_eq!(y1, BigRational::from_integer(3.into()));
    }

    #[test]
    fn test_lagrange_quadratic() {
        let mut interpolator = PolynomialInterpolator::default_config();

        // Points for x^2: (0,0), (1,1), (2,4)
        let points = vec![
            Point::from_ints(0, 0),
            Point::from_ints(1, 1),
            Point::from_ints(2, 4),
        ];

        let poly = interpolator
            .lagrange(&points)
            .expect("interpolation failed");

        // Verify at interpolation points
        for point in &points {
            let y = interpolator.evaluate(&poly, &point.x);
            assert_eq!(y, point.y);
        }
    }

    #[test]
    fn test_newton_linear() {
        let mut interpolator = PolynomialInterpolator::default_config();

        let points = vec![Point::from_ints(0, 1), Point::from_ints(1, 3)];

        let poly = interpolator.newton(&points).expect("interpolation failed");

        let y0 = interpolator.evaluate(&poly, &BigRational::zero());
        let y1 = interpolator.evaluate(&poly, &BigRational::one());

        assert_eq!(y0, BigRational::from_integer(1.into()));
        assert_eq!(y1, BigRational::from_integer(3.into()));
    }

    #[test]
    fn test_duplicate_x_error() {
        let mut interpolator = PolynomialInterpolator::default_config();

        let points = vec![Point::from_ints(0, 1), Point::from_ints(0, 2)];

        let result = interpolator.lagrange(&points);
        assert!(matches!(result, Err(InterpolationError::DuplicateXValue)));
    }

    #[test]
    fn test_stats() {
        let mut interpolator = PolynomialInterpolator::default_config();
        assert_eq!(interpolator.stats().interpolations, 0);

        let points = vec![Point::from_ints(0, 0), Point::from_ints(1, 1)];
        let _ = interpolator.lagrange(&points);

        assert_eq!(interpolator.stats().interpolations, 1);
    }
}
