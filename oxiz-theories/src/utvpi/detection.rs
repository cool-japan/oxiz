//! UTVPI Constraint Detection
//!
//! Detects whether linear constraints are UTVPI (Unit Two-Variable Per Inequality).
//! A constraint is UTVPI if it has the form: ax + by ≤ c where a, b ∈ {-1, 0, 1}

use num_rational::Rational64;
use std::collections::HashMap;

/// Kind of UTVPI constraint detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtConstraintKind {
    /// x - y ≤ c (difference constraint)
    Difference,
    /// x + y ≤ c (sum constraint)
    Sum,
    /// -x - y ≤ c (negative sum)
    NegativeSum,
    /// -x + y ≤ c (equivalent to y - x ≤ c)
    NegativeDifference,
    /// x ≤ c (upper bound)
    UpperBound,
    /// -x ≤ c (lower bound, equivalent to x ≥ -c)
    LowerBound,
    /// 0 ≤ c (tautology or contradiction)
    Constant,
}

/// Result of UTVPI detection on a linear constraint
#[derive(Debug, Clone)]
pub struct DetectedConstraint {
    /// The kind of UTVPI constraint
    pub kind: UtConstraintKind,
    /// First variable (if any)
    pub var1: Option<u32>,
    /// Coefficient of first variable (-1, 0, or 1)
    pub coef1: i8,
    /// Second variable (if any)
    pub var2: Option<u32>,
    /// Coefficient of second variable (-1, 0, or 1)
    pub coef2: i8,
    /// Constant bound
    pub bound: Rational64,
    /// Is this a strict inequality?
    pub strict: bool,
}

impl DetectedConstraint {
    /// Create a new detected constraint
    pub fn new(
        kind: UtConstraintKind,
        var1: Option<u32>,
        coef1: i8,
        var2: Option<u32>,
        coef2: i8,
        bound: Rational64,
        strict: bool,
    ) -> Self {
        Self {
            kind,
            var1,
            coef1,
            var2,
            coef2,
            bound,
            strict,
        }
    }
}

/// UTVPI constraint detector
#[derive(Debug, Default)]
pub struct UtvpiDetector {
    /// Statistics
    total_checked: u64,
    utvpi_detected: u64,
    non_utvpi: u64,
}

impl UtvpiDetector {
    /// Create a new detector
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a linear constraint is UTVPI
    ///
    /// Takes a map of variable -> coefficient and a constant bound.
    /// Returns Some(DetectedConstraint) if UTVPI, None otherwise.
    pub fn detect(
        &mut self,
        coefficients: &HashMap<u32, Rational64>,
        bound: Rational64,
        strict: bool,
    ) -> Option<DetectedConstraint> {
        self.total_checked += 1;

        // Filter out zero coefficients
        let non_zero: Vec<_> = coefficients
            .iter()
            .filter(|(_, c)| **c != Rational64::from_integer(0))
            .collect();

        // UTVPI constraints have at most 2 variables
        if non_zero.len() > 2 {
            self.non_utvpi += 1;
            return None;
        }

        // Check if all coefficients are unit (-1 or 1)
        for (_, coef) in &non_zero {
            if **coef != Rational64::from_integer(1) && **coef != Rational64::from_integer(-1) {
                self.non_utvpi += 1;
                return None;
            }
        }

        self.utvpi_detected += 1;

        // Classify the constraint
        match non_zero.len() {
            0 => {
                // 0 ≤ c (tautology if c >= 0, contradiction otherwise)
                Some(DetectedConstraint::new(
                    UtConstraintKind::Constant,
                    None,
                    0,
                    None,
                    0,
                    bound,
                    strict,
                ))
            }
            1 => {
                let (&var, &coef) = non_zero[0];
                let coef_i8 = if coef == Rational64::from_integer(1) {
                    1
                } else {
                    -1
                };
                let kind = if coef_i8 > 0 {
                    UtConstraintKind::UpperBound
                } else {
                    UtConstraintKind::LowerBound
                };
                Some(DetectedConstraint::new(
                    kind,
                    Some(var),
                    coef_i8,
                    None,
                    0,
                    bound,
                    strict,
                ))
            }
            2 => {
                // Sort by variable ID for deterministic ordering
                let mut sorted: Vec<_> = non_zero.iter().collect();
                sorted.sort_by_key(|(var, _)| **var);

                let (&var1, &coef1) = *sorted[0];
                let (&var2, &coef2) = *sorted[1];

                let c1 = if coef1 == Rational64::from_integer(1) {
                    1i8
                } else {
                    -1i8
                };
                let c2 = if coef2 == Rational64::from_integer(1) {
                    1i8
                } else {
                    -1i8
                };

                let kind = match (c1, c2) {
                    (1, -1) => UtConstraintKind::Difference,
                    (-1, 1) => UtConstraintKind::NegativeDifference,
                    (1, 1) => UtConstraintKind::Sum,
                    (-1, -1) => UtConstraintKind::NegativeSum,
                    _ => unreachable!("Invalid coefficient combination"),
                };

                Some(DetectedConstraint::new(
                    kind,
                    Some(var1),
                    c1,
                    Some(var2),
                    c2,
                    bound,
                    strict,
                ))
            }
            _ => None,
        }
    }

    /// Check if a constraint is UTVPI (simplified version)
    ///
    /// Returns true if at most 2 variables with unit coefficients.
    pub fn is_utvpi(&self, coefficients: &HashMap<u32, Rational64>) -> bool {
        let non_zero: Vec<_> = coefficients
            .iter()
            .filter(|(_, c)| **c != Rational64::from_integer(0))
            .collect();

        if non_zero.len() > 2 {
            return false;
        }

        for (_, coef) in &non_zero {
            if **coef != Rational64::from_integer(1) && **coef != Rational64::from_integer(-1) {
                return false;
            }
        }

        true
    }

    /// Check if a problem is entirely in UTVPI fragment
    pub fn is_problem_utvpi<'a>(
        &self,
        constraints: impl Iterator<Item = &'a HashMap<u32, Rational64>>,
    ) -> bool {
        for coeffs in constraints {
            if !self.is_utvpi(coeffs) {
                return false;
            }
        }
        true
    }

    /// Get statistics
    pub fn stats(&self) -> UtvpiDetectorStats {
        UtvpiDetectorStats {
            total_checked: self.total_checked,
            utvpi_detected: self.utvpi_detected,
            non_utvpi: self.non_utvpi,
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_checked = 0;
        self.utvpi_detected = 0;
        self.non_utvpi = 0;
    }
}

/// Statistics for UTVPI detection
#[derive(Debug, Clone, Default)]
pub struct UtvpiDetectorStats {
    /// Total constraints checked
    pub total_checked: u64,
    /// Constraints detected as UTVPI
    pub utvpi_detected: u64,
    /// Constraints that are not UTVPI
    pub non_utvpi: u64,
}

impl UtvpiDetectorStats {
    /// Percentage of constraints that are UTVPI
    pub fn utvpi_percentage(&self) -> f64 {
        if self.total_checked == 0 {
            0.0
        } else {
            (self.utvpi_detected as f64 / self.total_checked as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coeffs(pairs: &[(u32, i64)]) -> HashMap<u32, Rational64> {
        pairs
            .iter()
            .map(|&(var, coef)| (var, Rational64::from_integer(coef)))
            .collect()
    }

    #[test]
    fn test_constant_constraint() {
        let mut detector = UtvpiDetector::new();
        let coeffs = make_coeffs(&[]);
        let result = detector.detect(&coeffs, Rational64::from_integer(5), false);

        assert!(result.is_some());
        let detected = result.unwrap();
        assert_eq!(detected.kind, UtConstraintKind::Constant);
    }

    #[test]
    fn test_upper_bound() {
        let mut detector = UtvpiDetector::new();
        let coeffs = make_coeffs(&[(1, 1)]);
        let result = detector.detect(&coeffs, Rational64::from_integer(10), false);

        assert!(result.is_some());
        let detected = result.unwrap();
        assert_eq!(detected.kind, UtConstraintKind::UpperBound);
        assert_eq!(detected.var1, Some(1));
        assert_eq!(detected.coef1, 1);
    }

    #[test]
    fn test_lower_bound() {
        let mut detector = UtvpiDetector::new();
        let coeffs = make_coeffs(&[(1, -1)]);
        let result = detector.detect(&coeffs, Rational64::from_integer(10), false);

        assert!(result.is_some());
        let detected = result.unwrap();
        assert_eq!(detected.kind, UtConstraintKind::LowerBound);
        assert_eq!(detected.var1, Some(1));
        assert_eq!(detected.coef1, -1);
    }

    #[test]
    fn test_difference_constraint() {
        let mut detector = UtvpiDetector::new();
        // x - y ≤ 5
        let coeffs = make_coeffs(&[(1, 1), (2, -1)]);
        let result = detector.detect(&coeffs, Rational64::from_integer(5), false);

        assert!(result.is_some());
        let detected = result.unwrap();
        assert_eq!(detected.kind, UtConstraintKind::Difference);
    }

    #[test]
    fn test_sum_constraint() {
        let mut detector = UtvpiDetector::new();
        // x + y ≤ 5
        let coeffs = make_coeffs(&[(1, 1), (2, 1)]);
        let result = detector.detect(&coeffs, Rational64::from_integer(5), false);

        assert!(result.is_some());
        let detected = result.unwrap();
        assert_eq!(detected.kind, UtConstraintKind::Sum);
    }

    #[test]
    fn test_negative_sum() {
        let mut detector = UtvpiDetector::new();
        // -x - y ≤ 5
        let coeffs = make_coeffs(&[(1, -1), (2, -1)]);
        let result = detector.detect(&coeffs, Rational64::from_integer(5), false);

        assert!(result.is_some());
        let detected = result.unwrap();
        assert_eq!(detected.kind, UtConstraintKind::NegativeSum);
    }

    #[test]
    fn test_non_utvpi_three_vars() {
        let mut detector = UtvpiDetector::new();
        // x + y + z ≤ 5 (not UTVPI - 3 variables)
        let coeffs = make_coeffs(&[(1, 1), (2, 1), (3, 1)]);
        let result = detector.detect(&coeffs, Rational64::from_integer(5), false);

        assert!(result.is_none());
    }

    #[test]
    fn test_non_utvpi_non_unit_coef() {
        let mut detector = UtvpiDetector::new();
        // 2x + y ≤ 5 (not UTVPI - coefficient is 2)
        let coeffs = make_coeffs(&[(1, 2), (2, 1)]);
        let result = detector.detect(&coeffs, Rational64::from_integer(5), false);

        assert!(result.is_none());
    }

    #[test]
    fn test_is_utvpi() {
        let detector = UtvpiDetector::new();

        assert!(detector.is_utvpi(&make_coeffs(&[])));
        assert!(detector.is_utvpi(&make_coeffs(&[(1, 1)])));
        assert!(detector.is_utvpi(&make_coeffs(&[(1, 1), (2, -1)])));
        assert!(!detector.is_utvpi(&make_coeffs(&[(1, 1), (2, 1), (3, 1)])));
        assert!(!detector.is_utvpi(&make_coeffs(&[(1, 2), (2, 1)])));
    }

    #[test]
    fn test_stats() {
        let mut detector = UtvpiDetector::new();

        detector.detect(&make_coeffs(&[(1, 1)]), Rational64::from_integer(5), false);
        detector.detect(&make_coeffs(&[(1, 2)]), Rational64::from_integer(5), false);
        detector.detect(
            &make_coeffs(&[(1, 1), (2, 1)]),
            Rational64::from_integer(5),
            false,
        );

        let stats = detector.stats();
        assert_eq!(stats.total_checked, 3);
        assert_eq!(stats.utvpi_detected, 2);
        assert_eq!(stats.non_utvpi, 1);
    }
}
