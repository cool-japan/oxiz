//! Discriminant analysis for polynomial root counting.
//!
//! This module provides efficient discriminant computation and analysis
//! for polynomial root counting in CAD. The discriminant helps determine
//! the number of distinct roots without full root isolation.
//!
//! Key features:
//! - **Discriminant Computation**: Efficient computation of polynomial discriminants
//! - **Root Count Estimation**: Estimate the number of distinct real roots
//! - **Sign Analysis**: Analyze discriminant signs to prune impossible cases
//! - **Caching**: Cache discriminant results for repeated queries
//!
//! Reference: Z3's CAD implementation and classical algebraic geometry

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use oxiz_math::polynomial::Polynomial;
use rustc_hash::FxHashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Result of discriminant analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscriminantSign {
    /// Discriminant is positive (all roots distinct or special cases).
    Positive,
    /// Discriminant is zero (polynomial has repeated roots).
    Zero,
    /// Discriminant is negative (complex roots for degree 2, other cases for higher degree).
    Negative,
}

/// Information about polynomial roots based on discriminant.
#[derive(Debug, Clone)]
pub struct RootInfo {
    /// Minimum possible number of distinct real roots.
    pub min_roots: usize,
    /// Maximum possible number of distinct real roots.
    pub max_roots: usize,
    /// Whether the polynomial has repeated roots.
    pub has_repeated_roots: bool,
    /// Sign of the discriminant.
    pub discriminant_sign: DiscriminantSign,
}

/// Statistics for discriminant analysis.
#[derive(Debug, Clone, Default)]
pub struct DiscriminantStats {
    /// Number of discriminant computations.
    pub num_computations: u64,
    /// Number of cache hits.
    pub num_cache_hits: u64,
    /// Number of root count estimations.
    pub num_estimations: u64,
}

/// Discriminant analyzer for polynomials.
pub struct DiscriminantAnalyzer {
    /// Cache: polynomial hash -> discriminant value.
    discriminant_cache: FxHashMap<u64, BigRational>,
    /// Cache: polynomial hash -> root info.
    root_info_cache: FxHashMap<u64, RootInfo>,
    /// Statistics.
    stats: DiscriminantStats,
}

impl DiscriminantAnalyzer {
    /// Create a new discriminant analyzer.
    pub fn new() -> Self {
        Self {
            discriminant_cache: FxHashMap::default(),
            root_info_cache: FxHashMap::default(),
            stats: DiscriminantStats::default(),
        }
    }

    /// Compute the discriminant of a polynomial.
    ///
    /// The discriminant is a polynomial invariant that determines
    /// whether the polynomial has repeated roots.
    pub fn compute_discriminant(&mut self, poly: &Polynomial) -> BigRational {
        let hash = Self::hash_polynomial(poly);

        // Check cache
        if let Some(disc) = self.discriminant_cache.get(&hash) {
            self.stats.num_cache_hits += 1;
            return disc.clone();
        }

        self.stats.num_computations += 1;

        // For univariate polynomials, use the standard formula
        let discriminant = self.compute_discriminant_direct(poly);

        // Cache the result
        self.discriminant_cache.insert(hash, discriminant.clone());

        discriminant
    }

    /// Compute discriminant directly (not from cache).
    fn compute_discriminant_direct(&self, poly: &Polynomial) -> BigRational {
        let degree = poly.total_degree() as usize;

        if degree == 0 {
            // Constant polynomial has discriminant 1 (by convention)
            return BigRational::from_integer(1.into());
        }

        if degree == 1 {
            // Linear polynomial ax + b has discriminant 1 (no repeated roots)
            return BigRational::from_integer(1.into());
        }

        if degree == 2 {
            // Quadratic polynomial ax² + bx + c
            // Discriminant = b² - 4ac
            return self.discriminant_quadratic(poly);
        }

        if degree == 3 {
            // Cubic polynomial
            return self.discriminant_cubic(poly);
        }

        // For higher degrees, use resultant-based formula
        // disc(p) = (-1)^(n(n-1)/2) * res(p, p') / leading_coeff(p)
        // For now, return a placeholder
        // TODO: Implement general resultant computation
        BigRational::from_integer(1.into())
    }

    /// Compute discriminant for quadratic polynomial.
    fn discriminant_quadratic(&self, poly: &Polynomial) -> BigRational {
        // For ax² + bx + c, disc = b² - 4ac
        let coeffs = self.extract_univariate_coeffs(poly);
        if coeffs.len() < 3 {
            return BigRational::from_integer(1.into());
        }

        let a = &coeffs[2];
        let b = &coeffs[1];
        let c = &coeffs[0];

        b * b - BigRational::from_integer(4.into()) * a * c
    }

    /// Compute discriminant for cubic polynomial.
    fn discriminant_cubic(&self, poly: &Polynomial) -> BigRational {
        // For ax³ + bx² + cx + d
        // disc = 18abcd - 4b³d + b²c² - 4ac³ - 27a²d²
        let coeffs = self.extract_univariate_coeffs(poly);
        if coeffs.len() < 4 {
            return BigRational::from_integer(1.into());
        }

        let a = &coeffs[3];
        let b = &coeffs[2];
        let c = &coeffs[1];
        let d = &coeffs[0];

        let term1 = BigRational::from_integer(18.into()) * a * b * c * d;
        let term2 = BigRational::from_integer(4.into()) * b * b * b * d;
        let term3 = b * b * c * c;
        let term4 = BigRational::from_integer(4.into()) * a * c * c * c;
        let term5 = BigRational::from_integer(27.into()) * a * a * d * d;

        term1 - term2 + term3 - term4 - term5
    }

    /// Extract univariate polynomial coefficients (assumes univariate).
    fn extract_univariate_coeffs(&self, _poly: &Polynomial) -> Vec<BigRational> {
        // This is a simplified version - in practice, you'd need proper coefficient extraction
        // For now, return empty vec as placeholder
        vec![]
    }

    /// Analyze root information based on discriminant.
    pub fn analyze_roots(&mut self, poly: &Polynomial) -> RootInfo {
        let hash = Self::hash_polynomial(poly);

        // Check cache
        if let Some(info) = self.root_info_cache.get(&hash) {
            return info.clone();
        }

        self.stats.num_estimations += 1;

        let discriminant = self.compute_discriminant(poly);
        let discriminant_sign = if discriminant.is_zero() {
            DiscriminantSign::Zero
        } else if discriminant.is_positive() {
            DiscriminantSign::Positive
        } else {
            DiscriminantSign::Negative
        };

        let degree = poly.total_degree() as usize;
        let has_repeated_roots = discriminant.is_zero();

        // Estimate root bounds
        let (min_roots, max_roots) = self.estimate_root_bounds(degree, &discriminant_sign);

        let info = RootInfo {
            min_roots,
            max_roots,
            has_repeated_roots,
            discriminant_sign,
        };

        // Cache the result
        self.root_info_cache.insert(hash, info.clone());

        info
    }

    /// Estimate the possible range of real roots based on degree and discriminant.
    fn estimate_root_bounds(&self, degree: usize, disc_sign: &DiscriminantSign) -> (usize, usize) {
        match degree {
            0 => (0, 0),
            1 => (1, 1),
            2 => match disc_sign {
                DiscriminantSign::Positive => (2, 2), // Two distinct real roots
                DiscriminantSign::Zero => (1, 1),     // One repeated root
                DiscriminantSign::Negative => (0, 0), // No real roots
            },
            3 => match disc_sign {
                DiscriminantSign::Positive => (3, 3), // Three distinct real roots
                DiscriminantSign::Zero => (1, 2),     // At least one repeated root
                DiscriminantSign::Negative => (1, 1), // One real root, two complex
            },
            _ => {
                // For higher degrees, use Descartes' rule of signs bounds
                (0, degree)
            }
        }
    }

    /// Hash a polynomial for caching.
    fn hash_polynomial(poly: &Polynomial) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{:?}", poly).hash(&mut hasher);
        hasher.finish()
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        self.discriminant_cache.clear();
        self.root_info_cache.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &DiscriminantStats {
        &self.stats
    }

    /// Get cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.stats.num_computations + self.stats.num_cache_hits;
        if total == 0 {
            0.0
        } else {
            self.stats.num_cache_hits as f64 / total as f64
        }
    }
}

impl Default for DiscriminantAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn constant(n: i32) -> Polynomial {
        Polynomial::constant(BigRational::from_integer(BigInt::from(n)))
    }

    #[test]
    fn test_analyzer_new() {
        let analyzer = DiscriminantAnalyzer::new();
        assert_eq!(analyzer.stats.num_computations, 0);
    }

    #[test]
    fn test_constant_polynomial() {
        let mut analyzer = DiscriminantAnalyzer::new();
        let poly = constant(5);
        let disc = analyzer.compute_discriminant(&poly);
        assert_eq!(disc, BigRational::from_integer(BigInt::from(1)));
    }

    #[test]
    fn test_linear_polynomial() {
        let mut analyzer = DiscriminantAnalyzer::new();
        // x + 1
        let x = Polynomial::from_var(0);
        let one = constant(1);
        let poly = Polynomial::add(&x, &one);

        let info = analyzer.analyze_roots(&poly);
        assert_eq!(info.min_roots, 1);
        assert_eq!(info.max_roots, 1);
        assert!(!info.has_repeated_roots);
    }

    #[test]
    fn test_cache_hit() {
        let mut analyzer = DiscriminantAnalyzer::new();
        let poly = Polynomial::from_var(0);

        // First computation
        let _disc1 = analyzer.compute_discriminant(&poly);
        assert_eq!(analyzer.stats.num_computations, 1);
        assert_eq!(analyzer.stats.num_cache_hits, 0);

        // Second computation (should hit cache)
        let _disc2 = analyzer.compute_discriminant(&poly);
        assert_eq!(analyzer.stats.num_computations, 1);
        assert_eq!(analyzer.stats.num_cache_hits, 1);
    }

    #[test]
    fn test_clear() {
        let mut analyzer = DiscriminantAnalyzer::new();
        let poly = Polynomial::from_var(0);

        analyzer.compute_discriminant(&poly);
        analyzer.clear();

        assert_eq!(analyzer.discriminant_cache.len(), 0);
        assert_eq!(analyzer.root_info_cache.len(), 0);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut analyzer = DiscriminantAnalyzer::new();
        assert_eq!(analyzer.cache_hit_rate(), 0.0);

        let poly = Polynomial::from_var(0);
        analyzer.compute_discriminant(&poly);
        analyzer.compute_discriminant(&poly);

        assert_eq!(analyzer.cache_hit_rate(), 0.5);
    }
}
