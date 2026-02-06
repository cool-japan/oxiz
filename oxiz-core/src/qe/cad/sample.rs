//! Sample Point Selection for CAD.
//!
//! Selects representative sample points within CAD cells for theory reasoning.
//!
//! ## Strategies
//!
//! - **Rational**: Use rational sample points when possible
//! - **Algebraic**: Use algebraic numbers for better precision
//! - **Floating-point**: Use approximate samples for efficiency
//! - **Adaptive**: Choose strategy based on cell properties
//!
//! ## References
//!
//! - Collins: \"Quantifier Elimination for Real Closed Fields\" (1975)
//! - Z3's `nlsat/nlsat_solver.cpp` - sample point selection

use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

/// A sample point in a CAD cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplePoint {
    /// Coordinates (one per dimension).
    pub coords: Vec<BigRational>,
    /// Whether this is an exact sample or approximation.
    pub exact: bool,
}

impl SamplePoint {
    /// Create a new sample point.
    pub fn new(coords: Vec<BigRational>) -> Self {
        Self {
            coords,
            exact: true,
        }
    }

    /// Create an approximate sample point.
    pub fn approximate(coords: Vec<BigRational>) -> Self {
        Self {
            coords,
            exact: false,
        }
    }

    /// Get coordinate at dimension.
    pub fn coord(&self, dim: usize) -> Option<&BigRational> {
        self.coords.get(dim)
    }

    /// Number of dimensions.
    pub fn dimension(&self) -> usize {
        self.coords.len()
    }
}

/// Sample selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStrategy {
    /// Use rational samples.
    Rational,
    /// Use algebraic numbers.
    Algebraic,
    /// Use floating-point approximations.
    Approximate,
    /// Choose adaptively based on cell.
    Adaptive,
}

/// Cell type for sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellType {
    /// Open interval (sector).
    Sector,
    /// Point (section).
    Point,
    /// Full line (unbounded).
    Full,
}

/// Configuration for sample selection.
#[derive(Debug, Clone)]
pub struct SampleConfig {
    /// Sampling strategy.
    pub strategy: SampleStrategy,
    /// Prefer simple samples (e.g., 0, 1, -1).
    pub prefer_simple: bool,
    /// Maximum sample magnitude.
    pub max_magnitude: Option<i64>,
    /// Use midpoints for sectors.
    pub use_midpoints: bool,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            strategy: SampleStrategy::Adaptive,
            prefer_simple: true,
            max_magnitude: Some(1000000),
            use_midpoints: true,
        }
    }
}

/// Statistics for sample selection.
#[derive(Debug, Clone, Default)]
pub struct SampleStats {
    /// Samples selected.
    pub samples_selected: u64,
    /// Rational samples used.
    pub rational_samples: u64,
    /// Algebraic samples used.
    pub algebraic_samples: u64,
    /// Approximate samples used.
    pub approximate_samples: u64,
    /// Simple samples used (0, ±1).
    pub simple_samples: u64,
}

/// Sample point selector for CAD.
pub struct SampleSelector {
    /// Configuration.
    config: SampleConfig,
    /// Statistics.
    stats: SampleStats,
    /// Cache of previously selected samples.
    cache: FxHashMap<String, SamplePoint>,
}

impl SampleSelector {
    /// Create a new sample selector.
    pub fn new(config: SampleConfig) -> Self {
        Self {
            config,
            stats: SampleStats::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SampleConfig::default())
    }

    /// Select a sample point in an interval.
    ///
    /// For sectors (open intervals), selects an interior point.
    /// For sections (points), returns the point itself.
    pub fn select_in_interval(
        &mut self,
        lower: Option<&BigRational>,
        upper: Option<&BigRational>,
        cell_type: CellType,
    ) -> SamplePoint {
        self.stats.samples_selected += 1;

        match cell_type {
            CellType::Point => {
                // For point cells, return the point itself
                if let Some(pt) = lower {
                    self.stats.rational_samples += 1;
                    SamplePoint::new(vec![pt.clone()])
                } else if let Some(pt) = upper {
                    self.stats.rational_samples += 1;
                    SamplePoint::new(vec![pt.clone()])
                } else {
                    // Should not happen
                    self.stats.rational_samples += 1;
                    SamplePoint::new(vec![BigRational::zero()])
                }
            }
            CellType::Sector => {
                // For sectors, select an interior point
                self.select_sector_sample(lower, upper)
            }
            CellType::Full => {
                // For full line, select a simple sample
                self.select_simple_sample()
            }
        }
    }

    /// Select a sample in a sector (open interval).
    fn select_sector_sample(
        &mut self,
        lower: Option<&BigRational>,
        upper: Option<&BigRational>,
    ) -> SamplePoint {
        match (lower, upper) {
            (Some(l), Some(u)) => {
                // Bounded interval (l, u)
                if self.config.prefer_simple {
                    // Try simple values first
                    if let Some(simple) = self.try_simple_in_interval(l, u) {
                        return simple;
                    }
                }

                // Use midpoint
                if self.config.use_midpoints {
                    self.stats.rational_samples += 1;
                    let mid = (l + u) / BigRational::from_integer(2.into());
                    SamplePoint::new(vec![mid])
                } else {
                    // Use l + 1/2 or similar
                    self.stats.rational_samples += 1;
                    let offset =
                        BigRational::from_integer(1.into()) / BigRational::from_integer(2.into());
                    SamplePoint::new(vec![l + offset])
                }
            }
            (Some(l), None) => {
                // Unbounded above: (l, +∞)
                if self.config.prefer_simple {
                    // Try l + 1, l + 2, etc.
                    for i in 1..=10 {
                        let candidate = l + BigRational::from_integer(i.into());
                        if self.check_magnitude(&candidate) {
                            self.stats.rational_samples += 1;
                            self.stats.simple_samples += 1;
                            return SamplePoint::new(vec![candidate]);
                        }
                    }
                }

                // Default: l + 1
                self.stats.rational_samples += 1;
                SamplePoint::new(vec![l + BigRational::one()])
            }
            (None, Some(u)) => {
                // Unbounded below: (-∞, u)
                if self.config.prefer_simple {
                    // Try u - 1, u - 2, etc.
                    for i in 1..=10 {
                        let candidate = u - BigRational::from_integer(i.into());
                        if self.check_magnitude(&candidate) {
                            self.stats.rational_samples += 1;
                            self.stats.simple_samples += 1;
                            return SamplePoint::new(vec![candidate]);
                        }
                    }
                }

                // Default: u - 1
                self.stats.rational_samples += 1;
                SamplePoint::new(vec![u - BigRational::one()])
            }
            (None, None) => {
                // Full line (-∞, +∞)
                self.select_simple_sample()
            }
        }
    }

    /// Try to find a simple sample (0, ±1) in an interval.
    fn try_simple_in_interval(
        &mut self,
        lower: &BigRational,
        upper: &BigRational,
    ) -> Option<SamplePoint> {
        let candidates = [BigRational::zero(), BigRational::one(), -BigRational::one()];

        for candidate in &candidates {
            if candidate > lower && candidate < upper {
                self.stats.rational_samples += 1;
                self.stats.simple_samples += 1;
                return Some(SamplePoint::new(vec![candidate.clone()]));
            }
        }

        None
    }

    /// Select a simple sample (prefer 0, then 1, then -1).
    fn select_simple_sample(&mut self) -> SamplePoint {
        self.stats.rational_samples += 1;
        self.stats.simple_samples += 1;
        SamplePoint::new(vec![BigRational::zero()])
    }

    /// Check if a value is within magnitude bounds.
    fn check_magnitude(&self, value: &BigRational) -> bool {
        if let Some(max_mag) = self.config.max_magnitude {
            // Simplified: check if numerator and denominator are reasonable
            let num = value.numer().abs();
            let denom = value.denom().abs();

            // Convert to i64 if possible for comparison
            if let (Some(n), Some(d)) = (num.to_i64(), denom.to_i64()) {
                let n_abs: i64 = n.abs();
                n_abs <= max_mag && d <= max_mag
            } else {
                false
            }
        } else {
            true
        }
    }

    /// Refine a sample point for better precision.
    ///
    /// Used when an approximate sample needs to be made more precise.
    pub fn refine_sample(&mut self, sample: &SamplePoint) -> SamplePoint {
        if sample.exact {
            // Already exact, no refinement needed
            sample.clone()
        } else {
            // Would implement refinement logic here
            // For now, just return as-is
            self.stats.samples_selected += 1;
            sample.clone()
        }
    }

    /// Clear the sample cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &SampleStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SampleStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_point_creation() {
        let coords = vec![BigRational::zero(), BigRational::one()];
        let sample = SamplePoint::new(coords);

        assert_eq!(sample.dimension(), 2);
        assert!(sample.exact);
        assert_eq!(sample.coord(0), Some(&BigRational::zero()));
    }

    #[test]
    fn test_selector_creation() {
        let selector = SampleSelector::default_config();
        assert_eq!(selector.stats().samples_selected, 0);
    }

    #[test]
    fn test_point_cell_sampling() {
        let mut selector = SampleSelector::default_config();

        let pt = BigRational::from_integer(5.into());
        let sample = selector.select_in_interval(Some(&pt), None, CellType::Point);

        assert_eq!(sample.coords[0], pt);
        assert_eq!(selector.stats().samples_selected, 1);
    }

    #[test]
    fn test_bounded_sector_sampling() {
        let mut selector = SampleSelector::default_config();

        let lower = BigRational::from_integer(0.into());
        let upper = BigRational::from_integer(10.into());

        let sample = selector.select_in_interval(Some(&lower), Some(&upper), CellType::Sector);

        // Sample should be in (0, 10)
        assert!(sample.coords[0] > lower);
        assert!(sample.coords[0] < upper);
    }

    #[test]
    fn test_simple_sample_preference() {
        let mut selector = SampleSelector::new(SampleConfig {
            prefer_simple: true,
            ..Default::default()
        });

        let lower = BigRational::from_integer((-2).into());
        let upper = BigRational::from_integer(2.into());

        let sample = selector.select_in_interval(Some(&lower), Some(&upper), CellType::Sector);

        // Should prefer 0, 1, or -1
        let simple_values = [BigRational::zero(), BigRational::one(), -BigRational::one()];
        assert!(simple_values.contains(&sample.coords[0]));
        assert!(selector.stats().simple_samples > 0);
    }

    #[test]
    fn test_unbounded_above_sampling() {
        let mut selector = SampleSelector::default_config();

        let lower = BigRational::from_integer(5.into());
        let sample = selector.select_in_interval(Some(&lower), None, CellType::Sector);

        // Should be > 5
        assert!(sample.coords[0] > lower);
    }

    #[test]
    fn test_unbounded_below_sampling() {
        let mut selector = SampleSelector::default_config();

        let upper = BigRational::from_integer(5.into());
        let sample = selector.select_in_interval(None, Some(&upper), CellType::Sector);

        // Should be < 5
        assert!(sample.coords[0] < upper);
    }

    #[test]
    fn test_full_line_sampling() {
        let mut selector = SampleSelector::default_config();

        let sample = selector.select_in_interval(None, None, CellType::Full);

        // Should return a simple sample (default: 0)
        assert_eq!(sample.coords[0], BigRational::zero());
        assert!(selector.stats().simple_samples > 0);
    }

    #[test]
    fn test_stats() {
        let mut selector = SampleSelector::default_config();

        let lower = BigRational::zero();
        let upper = BigRational::one();

        selector.select_in_interval(Some(&lower), Some(&upper), CellType::Sector);

        assert_eq!(selector.stats().samples_selected, 1);
        assert!(selector.stats().rational_samples > 0);
    }
}
