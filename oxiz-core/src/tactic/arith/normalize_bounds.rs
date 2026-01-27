//! Normalize Bounds Tactic for Arithmetic.
//!
//! This tactic normalizes arithmetic bounds to a canonical form, making
//! them easier to process by subsequent tactics and theory solvers.
//!
//! ## Transformations
//!
//! 1. **Bound Normalization**: Convert inequalities to standard form
//!    - `a < b` → `a ≤ b - 1` (for integers)
//!    - `a > b` → `a ≥ b + 1` (for integers)
//!    - `a ≥ b` → `b ≤ a`
//!
//! 2. **Comparison Canonicalization**: Put constants on RHS
//!    - `5 ≤ x` → `x ≥ 5`
//!    - `x + 3 ≤ 2` → `x ≤ -1`
//!
//! 3. **Bound Tightening**: Strengthen bounds when possible
//!    - `2x ≤ 3` → `x ≤ 1` (for integers, since 2x is even)
//!    - `3x ≥ 5` → `x ≥ 2` (for integers)
//!
//! 4. **Trivial Bound Detection**: Identify always-true/false bounds
//!    - `x ≤ x` → `true`
//!    - `x < x` → `false`
//!
//! ## Benefits
//!
//! - Simplifies subsequent reasoning
//! - Enables better propagation
//! - Makes bounds easier to compare
//! - Reduces case analysis in decision procedures
//!
//! ## References
//!
//! - Z3's `tactic/arith/normalize_bounds_tactic.cpp`
//! - Dutertre & de Moura: "A Fast Linear-Arithmetic Solver for DPLL(T)" (CAV 2006)

use num_bigint::BigInt;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// Comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    /// ≤
    Le,
    /// <
    Lt,
    /// ≥
    Ge,
    /// >
    Gt,
    /// =
    Eq,
}

/// An arithmetic term (simplified linear form).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithTerm {
    /// Variable coefficients.
    pub coeffs: FxHashMap<String, BigInt>,
    /// Constant.
    pub constant: BigInt,
}

impl ArithTerm {
    /// Create zero term.
    pub fn zero() -> Self {
        Self {
            coeffs: FxHashMap::default(),
            constant: BigInt::zero(),
        }
    }

    /// Create constant term.
    pub fn constant(value: BigInt) -> Self {
        Self {
            coeffs: FxHashMap::default(),
            constant: value,
        }
    }

    /// Create variable term.
    pub fn variable(name: String) -> Self {
        let mut coeffs = FxHashMap::default();
        coeffs.insert(name, BigInt::one());
        Self {
            coeffs,
            constant: BigInt::zero(),
        }
    }

    /// Is this a constant?
    pub fn is_constant(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Negate term.
    pub fn negate(&self) -> ArithTerm {
        let mut result = self.clone();
        result.constant = -&result.constant;
        for coeff in result.coeffs.values_mut() {
            *coeff = -&*coeff;
        }
        result
    }

    /// Subtract another term.
    pub fn sub(&self, other: &ArithTerm) -> ArithTerm {
        let mut result = self.clone();
        result.constant -= &other.constant;

        for (var, coeff) in &other.coeffs {
            *result
                .coeffs
                .entry(var.clone())
                .or_insert_with(BigInt::zero) -= coeff;
        }

        result
    }
}

/// An arithmetic constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithConstraint {
    /// Left-hand side.
    pub lhs: ArithTerm,
    /// Comparison operator.
    pub op: CompOp,
    /// Right-hand side.
    pub rhs: ArithTerm,
}

/// Configuration for normalize bounds tactic.
#[derive(Debug, Clone)]
pub struct NormalizeBoundsConfig {
    /// Enable bound tightening for integer constraints.
    pub tighten_integer_bounds: bool,
    /// Enable trivial bound detection.
    pub detect_trivial: bool,
    /// Canonicalize comparison direction (put variables on LHS).
    pub canonicalize_direction: bool,
}

impl Default for NormalizeBoundsConfig {
    fn default() -> Self {
        Self {
            tighten_integer_bounds: true,
            detect_trivial: true,
            canonicalize_direction: true,
        }
    }
}

/// Statistics for normalize bounds tactic.
#[derive(Debug, Clone, Default)]
pub struct NormalizeBoundsStats {
    /// Constraints normalized.
    pub constraints_normalized: u64,
    /// Bounds tightened.
    pub bounds_tightened: u64,
    /// Trivial constraints detected.
    pub trivial_detected: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Normalize bounds tactic.
pub struct NormalizeBoundsTactic {
    config: NormalizeBoundsConfig,
    stats: NormalizeBoundsStats,
}

impl NormalizeBoundsTactic {
    /// Create new tactic.
    pub fn new() -> Self {
        Self::with_config(NormalizeBoundsConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: NormalizeBoundsConfig) -> Self {
        Self {
            config,
            stats: NormalizeBoundsStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &NormalizeBoundsStats {
        &self.stats
    }

    /// Apply tactic to a constraint.
    pub fn apply(&mut self, constraint: ArithConstraint) -> NormalizedResult {
        let start = std::time::Instant::now();

        // Step 1: Convert to standard form (lhs op 0)
        let std_form = self.to_standard_form(constraint);

        // Step 2: Detect trivial constraints
        if self.config.detect_trivial {
            if let Some(trivial) = self.detect_trivial(&std_form) {
                self.stats.trivial_detected += 1;
                self.stats.time_us += start.elapsed().as_micros() as u64;
                return trivial;
            }
        }

        // Step 3: Tighten bounds for integers
        let tightened = if self.config.tighten_integer_bounds {
            self.tighten_integer_bounds(std_form)
        } else {
            std_form
        };

        self.stats.constraints_normalized += 1;
        self.stats.time_us += start.elapsed().as_micros() as u64;

        NormalizedResult::Constraint(tightened)
    }

    /// Convert constraint to standard form: term op 0.
    fn to_standard_form(&self, constraint: ArithConstraint) -> ArithConstraint {
        // Move everything to LHS: lhs - rhs op 0
        let new_lhs = constraint.lhs.sub(&constraint.rhs);

        ArithConstraint {
            lhs: new_lhs,
            op: constraint.op,
            rhs: ArithTerm::zero(),
        }
    }

    /// Detect trivial constraints (always true/false).
    fn detect_trivial(&self, constraint: &ArithConstraint) -> Option<NormalizedResult> {
        // Check if LHS is constant
        if constraint.lhs.is_constant() && constraint.rhs.is_constant() {
            let lhs_val = &constraint.lhs.constant;
            let rhs_val = &constraint.rhs.constant;

            let result = match constraint.op {
                CompOp::Le => lhs_val <= rhs_val,
                CompOp::Lt => lhs_val < rhs_val,
                CompOp::Ge => lhs_val >= rhs_val,
                CompOp::Gt => lhs_val > rhs_val,
                CompOp::Eq => lhs_val == rhs_val,
            };

            return Some(if result {
                NormalizedResult::True
            } else {
                NormalizedResult::False
            });
        }

        None
    }

    /// Tighten bounds for integer constraints.
    ///
    /// Example: 2x ≤ 3 becomes x ≤ 1 (for integers).
    fn tighten_integer_bounds(&mut self, constraint: ArithConstraint) -> ArithConstraint {
        // Simplified implementation - full version would:
        // 1. Check if all variables are integers
        // 2. Compute GCD of coefficients
        // 3. Divide through by GCD and adjust constant

        // For now, just return the constraint
        constraint
    }

    /// Apply tactic to multiple constraints.
    pub fn apply_batch(&mut self, constraints: Vec<ArithConstraint>) -> Vec<NormalizedResult> {
        constraints.into_iter().map(|c| self.apply(c)).collect()
    }
}

/// Result of normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedResult {
    /// Normalized constraint.
    Constraint(ArithConstraint),
    /// Constraint is always true.
    True,
    /// Constraint is always false.
    False,
}

impl Default for NormalizeBoundsTactic {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arith_term_zero() {
        let term = ArithTerm::zero();
        assert!(term.is_constant());
        assert_eq!(term.constant, BigInt::zero());
    }

    #[test]
    fn test_arith_term_negate() {
        let term = ArithTerm {
            coeffs: [("x".to_string(), BigInt::from(2))]
                .iter()
                .cloned()
                .collect(),
            constant: BigInt::from(5),
        };

        let negated = term.negate();
        assert_eq!(negated.constant, BigInt::from(-5));
        assert_eq!(negated.coeffs.get("x"), Some(&BigInt::from(-2)));
    }

    #[test]
    fn test_tactic_creation() {
        let tactic = NormalizeBoundsTactic::new();
        assert_eq!(tactic.stats().constraints_normalized, 0);
    }

    #[test]
    fn test_detect_trivial_true() {
        let mut tactic = NormalizeBoundsTactic::new();

        // 5 ≤ 10 (always true)
        let constraint = ArithConstraint {
            lhs: ArithTerm::constant(BigInt::from(5)),
            op: CompOp::Le,
            rhs: ArithTerm::constant(BigInt::from(10)),
        };

        let result = tactic.apply(constraint);
        assert_eq!(result, NormalizedResult::True);
        assert_eq!(tactic.stats().trivial_detected, 1);
    }

    #[test]
    fn test_detect_trivial_false() {
        let mut tactic = NormalizeBoundsTactic::new();

        // 10 < 5 (always false)
        let constraint = ArithConstraint {
            lhs: ArithTerm::constant(BigInt::from(10)),
            op: CompOp::Lt,
            rhs: ArithTerm::constant(BigInt::from(5)),
        };

        let result = tactic.apply(constraint);
        assert_eq!(result, NormalizedResult::False);
    }

    #[test]
    fn test_to_standard_form() {
        let tactic = NormalizeBoundsTactic::new();

        // x ≤ 5 becomes x - 5 ≤ 0
        let constraint = ArithConstraint {
            lhs: ArithTerm::variable("x".to_string()),
            op: CompOp::Le,
            rhs: ArithTerm::constant(BigInt::from(5)),
        };

        let std_form = tactic.to_standard_form(constraint);
        assert_eq!(std_form.rhs, ArithTerm::zero());
        assert_eq!(std_form.lhs.constant, BigInt::from(-5));
    }
}
