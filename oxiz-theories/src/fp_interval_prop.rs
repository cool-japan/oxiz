//! Floating-Point Interval Propagation.
//!
//! This module implements IEEE 754 interval arithmetic for propagating bounds
//! and constraints in floating-point theory solving.
//!
//! ## Interval Arithmetic
//!
//! Track intervals [low, high] for floating-point values, considering:
//! - IEEE 754 rounding modes
//! - Special values (±∞, NaN, ±0)
//! - Precision limitations
//!
//! ## Propagation Rules
//!
//! - `x + y ∈ [x.low + y.low, x.high + y.high]` (with rounding)
//! - `x * y` intervals depend on signs and rounding
//! - Comparisons reduce intervals: `x < 5.0` implies `x.high < 5.0`
//!
//! ## Rounding Mode Tracking
//!
//! Different rounding modes affect interval bounds:
//! - RNE (round to nearest, ties to even)
//! - RTP (round toward +∞)
//! - RTN (round toward -∞)
//! - RTZ (round toward zero)
//!
//! ## References
//!
//! - IEEE 754-2008 standard
//! - Moore: "Interval Analysis" (1966)
//! - Z3's `fpa/fpa2bv_converter.cpp` and interval reasoning
//! - Brain et al.: "An Automatable Formal Semantics for IEEE-754 Floating-Point Arithmetic" (2015)

use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;

/// IEEE 754 rounding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RoundingMode {
    /// Round to nearest, ties to even.
    #[default]
    RNE,
    /// Round toward negative infinity.
    RTN,
    /// Round toward positive infinity.
    RTP,
    /// Round toward zero.
    RTZ,
    /// Round to nearest, ties away from zero.
    RNA,
}

/// Floating-point interval (bounds with rounding mode).
#[derive(Debug, Clone, PartialEq)]
pub struct FpInterval {
    /// Lower bound.
    pub low: f64,
    /// Upper bound.
    pub high: f64,
    /// Is the interval known to contain NaN?
    pub may_be_nan: bool,
    /// Rounding mode for operations on this interval.
    pub rounding: RoundingMode,
}

impl FpInterval {
    /// Create a point interval [v, v].
    pub fn point(v: f64) -> Self {
        Self {
            low: v,
            high: v,
            may_be_nan: v.is_nan(),
            rounding: RoundingMode::RNE,
        }
    }

    /// Create an interval [low, high].
    pub fn new(low: f64, high: f64) -> Self {
        Self {
            low,
            high,
            may_be_nan: false,
            rounding: RoundingMode::RNE,
        }
    }

    /// Create an interval with rounding mode.
    pub fn with_rounding(low: f64, high: f64, rounding: RoundingMode) -> Self {
        Self {
            low,
            high,
            may_be_nan: false,
            rounding,
        }
    }

    /// Create the full real interval [-∞, +∞].
    pub fn full() -> Self {
        Self {
            low: f64::NEG_INFINITY,
            high: f64::INFINITY,
            may_be_nan: true,
            rounding: RoundingMode::RNE,
        }
    }

    /// Check if interval is empty (inconsistent).
    pub fn is_empty(&self) -> bool {
        self.low > self.high || (self.low.is_nan() && self.high.is_nan())
    }

    /// Check if interval is a single point.
    pub fn is_point(&self) -> bool {
        self.low == self.high && !self.may_be_nan
    }

    /// Intersect two intervals.
    pub fn intersect(&self, other: &FpInterval) -> FpInterval {
        FpInterval {
            low: self.low.max(other.low),
            high: self.high.min(other.high),
            may_be_nan: self.may_be_nan && other.may_be_nan,
            rounding: self.rounding,
        }
    }

    /// Add two intervals (with rounding).
    pub fn add(&self, other: &FpInterval) -> FpInterval {
        // Simplified: real implementation would handle rounding carefully
        let low = self.low + other.low;
        let high = self.high + other.high;

        FpInterval {
            low,
            high,
            may_be_nan: self.may_be_nan || other.may_be_nan,
            rounding: self.rounding,
        }
    }

    /// Multiply two intervals (with rounding).
    pub fn mul(&self, other: &FpInterval) -> FpInterval {
        // Compute all four products and take min/max
        let products = [
            self.low * other.low,
            self.low * other.high,
            self.high * other.low,
            self.high * other.high,
        ];

        let low = products
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| a.min(b));
        let high = products
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        FpInterval {
            low,
            high,
            may_be_nan: self.may_be_nan || other.may_be_nan,
            rounding: self.rounding,
        }
    }

    /// Negate an interval.
    pub fn neg(&self) -> FpInterval {
        FpInterval {
            low: -self.high,
            high: -self.low,
            may_be_nan: self.may_be_nan,
            rounding: self.rounding,
        }
    }
}

/// Statistics for FP interval propagation.
#[derive(Debug, Clone, Default)]
pub struct FpIntervalStats {
    /// Number of interval refinements.
    pub refinements: u64,
    /// Number of FP conflicts detected.
    pub conflicts: u64,
    /// Number of propagations performed.
    pub propagations: u64,
    /// Time in interval operations (microseconds).
    pub interval_time_us: u64,
}

/// Floating-point interval propagator.
pub struct FpIntervalPropagator {
    /// Interval domains for FP terms.
    intervals: FxHashMap<TermId, FpInterval>,
    /// Current rounding mode.
    current_rounding: RoundingMode,
    /// Statistics.
    stats: FpIntervalStats,
}

impl FpIntervalPropagator {
    /// Create a new FP interval propagator.
    pub fn new() -> Self {
        Self {
            intervals: FxHashMap::default(),
            current_rounding: RoundingMode::RNE,
            stats: FpIntervalStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &FpIntervalStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = FpIntervalStats::default();
    }

    /// Set current rounding mode.
    pub fn set_rounding_mode(&mut self, mode: RoundingMode) {
        self.current_rounding = mode;
    }

    /// Get interval for a term.
    pub fn get_interval(&self, term: TermId) -> FpInterval {
        self.intervals
            .get(&term)
            .cloned()
            .unwrap_or_else(FpInterval::full)
    }

    /// Set interval for a term.
    pub fn set_interval(&mut self, term: TermId, interval: FpInterval) {
        if interval.is_empty() {
            self.stats.conflicts += 1;
        }

        self.intervals.insert(term, interval);
    }

    /// Refine interval (intersect with new bounds).
    pub fn refine_interval(&mut self, term: TermId, new_interval: FpInterval) {
        let current = self.get_interval(term);
        let refined = current.intersect(&new_interval);

        if refined != current {
            self.stats.refinements += 1;
        }

        self.set_interval(term, refined);
    }

    /// Propagate intervals through addition: z = x + y.
    pub fn propagate_add(&mut self, x: TermId, y: TermId, z: TermId) {
        let start = std::time::Instant::now();

        let interval_x = self.get_interval(x);
        let interval_y = self.get_interval(y);

        let interval_z = interval_x.add(&interval_y);
        self.refine_interval(z, interval_z);

        self.stats.propagations += 1;
        self.stats.interval_time_us += start.elapsed().as_micros() as u64;
    }

    /// Propagate intervals through multiplication: z = x * y.
    pub fn propagate_mul(&mut self, x: TermId, y: TermId, z: TermId) {
        let start = std::time::Instant::now();

        let interval_x = self.get_interval(x);
        let interval_y = self.get_interval(y);

        let interval_z = interval_x.mul(&interval_y);
        self.refine_interval(z, interval_z);

        self.stats.propagations += 1;
        self.stats.interval_time_us += start.elapsed().as_micros() as u64;
    }

    /// Check if any interval is empty (conflict).
    pub fn has_conflict(&self) -> bool {
        self.intervals.values().any(|i| i.is_empty())
    }

    /// Clear all intervals.
    pub fn clear(&mut self) {
        self.intervals.clear();
    }
}

impl Default for FpIntervalPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp_interval_point() {
        let interval = FpInterval::point(2.75);
        assert_eq!(interval.low, 2.75);
        assert_eq!(interval.high, 2.75);
        assert!(interval.is_point());
        assert!(!interval.is_empty());
    }

    #[test]
    fn test_fp_interval_new() {
        let interval = FpInterval::new(1.0, 5.0);
        assert_eq!(interval.low, 1.0);
        assert_eq!(interval.high, 5.0);
        assert!(!interval.is_point());
    }

    #[test]
    fn test_fp_interval_full() {
        let interval = FpInterval::full();
        assert!(interval.low.is_infinite() && interval.low.is_sign_negative());
        assert!(interval.high.is_infinite() && interval.high.is_sign_positive());
        assert!(interval.may_be_nan);
    }

    #[test]
    fn test_fp_interval_empty() {
        let interval = FpInterval::new(5.0, 1.0); // low > high
        assert!(interval.is_empty());
    }

    #[test]
    fn test_fp_interval_intersect() {
        let i1 = FpInterval::new(0.0, 10.0);
        let i2 = FpInterval::new(5.0, 15.0);

        let intersection = i1.intersect(&i2);
        assert_eq!(intersection.low, 5.0);
        assert_eq!(intersection.high, 10.0);
    }

    #[test]
    fn test_fp_interval_add() {
        let i1 = FpInterval::new(1.0, 2.0);
        let i2 = FpInterval::new(3.0, 4.0);

        let sum = i1.add(&i2);
        assert_eq!(sum.low, 4.0);
        assert_eq!(sum.high, 6.0);
    }

    #[test]
    fn test_fp_interval_mul() {
        let i1 = FpInterval::new(2.0, 3.0);
        let i2 = FpInterval::new(4.0, 5.0);

        let product = i1.mul(&i2);
        assert!(product.low >= 8.0);
        assert!(product.high <= 15.0);
    }

    #[test]
    fn test_fp_interval_neg() {
        let interval = FpInterval::new(1.0, 5.0);
        let negated = interval.neg();

        assert_eq!(negated.low, -5.0);
        assert_eq!(negated.high, -1.0);
    }

    #[test]
    fn test_fp_interval_propagator_creation() {
        let prop = FpIntervalPropagator::new();
        assert_eq!(prop.stats().propagations, 0);
        assert_eq!(prop.current_rounding, RoundingMode::RNE);
    }

    #[test]
    fn test_fp_propagator_set_interval() {
        let mut prop = FpIntervalPropagator::new();

        let term = TermId::new(1);
        let interval = FpInterval::new(1.0, 5.0);

        prop.set_interval(term, interval.clone());

        assert_eq!(prop.get_interval(term), interval);
    }

    #[test]
    fn test_fp_propagator_refine() {
        let mut prop = FpIntervalPropagator::new();

        let term = TermId::new(1);
        prop.set_interval(term, FpInterval::new(0.0, 10.0));

        // Refine to [2.0, 5.0]
        prop.refine_interval(term, FpInterval::new(2.0, 5.0));

        let refined = prop.get_interval(term);
        assert_eq!(refined.low, 2.0);
        assert_eq!(refined.high, 5.0);
        assert_eq!(prop.stats().refinements, 1);
    }

    #[test]
    fn test_fp_propagator_add() {
        let mut prop = FpIntervalPropagator::new();

        let x = TermId::new(1);
        let y = TermId::new(2);
        let z = TermId::new(3);

        prop.set_interval(x, FpInterval::new(1.0, 2.0));
        prop.set_interval(y, FpInterval::new(3.0, 4.0));

        prop.propagate_add(x, y, z);

        let interval_z = prop.get_interval(z);
        assert_eq!(interval_z.low, 4.0);
        assert_eq!(interval_z.high, 6.0);
        assert_eq!(prop.stats().propagations, 1);
    }

    #[test]
    fn test_fp_propagator_mul() {
        let mut prop = FpIntervalPropagator::new();

        let x = TermId::new(1);
        let y = TermId::new(2);
        let z = TermId::new(3);

        prop.set_interval(x, FpInterval::new(2.0, 3.0));
        prop.set_interval(y, FpInterval::new(4.0, 5.0));

        prop.propagate_mul(x, y, z);

        let interval_z = prop.get_interval(z);
        assert!(interval_z.low >= 8.0);
        assert!(interval_z.high <= 15.0);
    }

    #[test]
    fn test_fp_conflict_detection() {
        let mut prop = FpIntervalPropagator::new();

        let term = TermId::new(1);
        prop.set_interval(term, FpInterval::new(5.0, 1.0)); // Empty

        assert!(prop.has_conflict());
        assert_eq!(prop.stats().conflicts, 1);
    }

    #[test]
    fn test_rounding_mode_setting() {
        let mut prop = FpIntervalPropagator::new();

        prop.set_rounding_mode(RoundingMode::RTP);
        assert_eq!(prop.current_rounding, RoundingMode::RTP);

        prop.set_rounding_mode(RoundingMode::RTN);
        assert_eq!(prop.current_rounding, RoundingMode::RTN);
    }

    #[test]
    fn test_fp_propagator_clear() {
        let mut prop = FpIntervalPropagator::new();

        prop.set_interval(TermId::new(1), FpInterval::point(std::f64::consts::PI));
        assert!(!prop.intervals.is_empty());

        prop.clear();
        assert!(prop.intervals.is_empty());
    }

    #[test]
    fn test_stats_reset() {
        let mut prop = FpIntervalPropagator::new();

        prop.stats.propagations = 100;
        prop.stats.refinements = 50;

        prop.reset_stats();

        assert_eq!(prop.stats().propagations, 0);
        assert_eq!(prop.stats().refinements, 0);
    }
}
