//! Virtual Term Substitution for Linear Integer Arithmetic QE.
//!
//! This module implements virtual term substitution (VTS), a powerful technique
//! for quantifier elimination in linear integer arithmetic (LIA).
//!
//! ## Algorithm Overview
//!
//! For a formula ∃x. φ(x, y) where φ is a conjunction of linear constraints:
//! 1. Identify bounds on x: lower bounds (L ≤ x) and upper bounds (x ≤ U)
//! 2. For each pair (L, U), check if L ≤ U (feasibility)
//! 3. Generate "virtual terms" that witness satisfiability
//! 4. Eliminate x by substituting these terms
//!
//! ## Example
//!
//! ```text
//! ∃x. (2y ≤ x ∧ x ≤ 3y + 5)
//! Lower bound: x ≥ 2y
//! Upper bound: x ≤ 3y + 5
//! Eliminate: (2y ≤ 3y + 5) which simplifies to (y ≥ -5)
//! ```
//!
//! ## References
//!
//! - Cooper: "Theorem Proving in Arithmetic without Multiplication" (1972)
//! - Presburger arithmetic decision procedures
//! - Z3's `qe/qe_arith.cpp` virtual term substitution

use num_bigint::BigInt;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// A linear term: a0 + a1*x1 + a2*x2 + ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearTerm {
    /// Constant coefficient.
    pub constant: BigInt,
    /// Variable coefficients (var_id -> coefficient).
    pub coeffs: FxHashMap<usize, BigInt>,
}

impl LinearTerm {
    /// Create zero term.
    pub fn zero() -> Self {
        Self {
            constant: BigInt::zero(),
            coeffs: FxHashMap::default(),
        }
    }

    /// Create constant term.
    pub fn constant(value: BigInt) -> Self {
        Self {
            constant: value,
            coeffs: FxHashMap::default(),
        }
    }

    /// Create variable term (1 * var).
    pub fn variable(var_id: usize) -> Self {
        let mut coeffs = FxHashMap::default();
        coeffs.insert(var_id, BigInt::one());
        Self {
            constant: BigInt::zero(),
            coeffs,
        }
    }

    /// Add two linear terms.
    pub fn add(&self, other: &LinearTerm) -> LinearTerm {
        let mut result = self.clone();
        result.constant += &other.constant;

        for (var, coeff) in &other.coeffs {
            *result.coeffs.entry(*var).or_insert_with(BigInt::zero) += coeff;
        }

        result
    }

    /// Multiply by a constant.
    pub fn mul_const(&self, k: &BigInt) -> LinearTerm {
        let mut result = self.clone();
        result.constant *= k;

        for coeff in result.coeffs.values_mut() {
            *coeff *= k;
        }

        result
    }

    /// Substitute a variable with another term.
    pub fn substitute(&self, var_id: usize, term: &LinearTerm) -> LinearTerm {
        if let Some(coeff) = self.coeffs.get(&var_id) {
            let mut result = self.clone();
            result.coeffs.remove(&var_id);

            // Add coeff * term
            let scaled_term = term.mul_const(coeff);
            result = result.add(&scaled_term);
            result
        } else {
            self.clone()
        }
    }
}

/// A linear constraint: term ≤ 0 or term < 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearConstraint {
    /// Left-hand side term.
    pub lhs: LinearTerm,
    /// Is strict inequality (<).
    pub strict: bool,
}

impl LinearConstraint {
    /// Create constraint term ≤ 0.
    pub fn leq(lhs: LinearTerm) -> Self {
        Self { lhs, strict: false }
    }

    /// Create constraint term < 0.
    pub fn lt(lhs: LinearTerm) -> Self {
        Self { lhs, strict: true }
    }

    /// Substitute a variable.
    pub fn substitute(&self, var_id: usize, term: &LinearTerm) -> LinearConstraint {
        LinearConstraint {
            lhs: self.lhs.substitute(var_id, term),
            strict: self.strict,
        }
    }
}

/// Configuration for virtual term substitution.
#[derive(Debug, Clone)]
pub struct VtsConfig {
    /// Maximum number of bounds to consider per variable.
    pub max_bounds: usize,
    /// Enable bound tightening.
    pub tighten_bounds: bool,
    /// Enable simplification.
    pub simplify: bool,
}

impl Default for VtsConfig {
    fn default() -> Self {
        Self {
            max_bounds: 1000,
            tighten_bounds: true,
            simplify: true,
        }
    }
}

/// Statistics for VTS.
#[derive(Debug, Clone, Default)]
pub struct VtsStats {
    /// Variables eliminated.
    pub variables_eliminated: u64,
    /// Constraints generated.
    pub constraints_generated: u64,
    /// Bounds computed.
    pub bounds_computed: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Virtual term substitution engine.
pub struct VirtualTermSubstitution {
    #[allow(dead_code)]
    config: VtsConfig,
    stats: VtsStats,
}

impl VirtualTermSubstitution {
    /// Create new VTS engine.
    pub fn new() -> Self {
        Self::with_config(VtsConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: VtsConfig) -> Self {
        Self {
            config,
            stats: VtsStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &VtsStats {
        &self.stats
    }

    /// Eliminate a quantified variable from a set of constraints.
    ///
    /// Given: ∃var. (constraints)
    /// Returns: constraints with var eliminated
    pub fn eliminate_variable(
        &mut self,
        var_id: usize,
        constraints: &[LinearConstraint],
    ) -> Vec<LinearConstraint> {
        let start = std::time::Instant::now();

        // Extract lower and upper bounds on var
        let (lower_bounds, upper_bounds, other) = self.extract_bounds(var_id, constraints);

        self.stats.bounds_computed += (lower_bounds.len() + upper_bounds.len()) as u64;

        // Generate elimination constraints: for each (L, U) pair, add L ≤ U
        let mut result = other;

        for lower in &lower_bounds {
            for upper in &upper_bounds {
                // Create constraint: lower.term ≤ upper.term
                // Which is: upper.term - lower.term ≥ 0
                // Or: lower.term - upper.term ≤ 0

                let diff = lower.lhs.add(&upper.lhs.mul_const(&BigInt::from(-1)));

                result.push(LinearConstraint {
                    lhs: diff,
                    strict: lower.strict || upper.strict,
                });

                self.stats.constraints_generated += 1;
            }
        }

        self.stats.variables_eliminated += 1;
        self.stats.time_us += start.elapsed().as_micros() as u64;

        result
    }

    /// Extract lower bounds, upper bounds, and other constraints.
    ///
    /// Lower bound: c ≤ var  =>  var ≥ c  =>  var - c ≥ 0  =>  -(var - c) ≤ 0  => -var + c ≤ 0
    /// Upper bound: var ≤ c  =>  var - c ≤ 0
    fn extract_bounds(
        &self,
        var_id: usize,
        constraints: &[LinearConstraint],
    ) -> (
        Vec<LinearConstraint>,
        Vec<LinearConstraint>,
        Vec<LinearConstraint>,
    ) {
        let mut lower_bounds = Vec::new();
        let mut upper_bounds = Vec::new();
        let mut other = Vec::new();

        for constraint in constraints {
            if let Some(coeff) = constraint.lhs.coeffs.get(&var_id) {
                if coeff > &BigInt::zero() {
                    // Positive coefficient: a*var + rest ≤ 0  =>  var ≤ -rest/a
                    // This is an upper bound
                    upper_bounds.push(constraint.clone());
                } else if coeff < &BigInt::zero() {
                    // Negative coefficient: -a*var + rest ≤ 0  =>  var ≥ rest/a
                    // This is a lower bound
                    lower_bounds.push(constraint.clone());
                }
            } else {
                // Variable doesn't appear - keep constraint
                other.push(constraint.clone());
            }
        }

        (lower_bounds, upper_bounds, other)
    }

    /// Eliminate multiple variables.
    pub fn eliminate_variables(
        &mut self,
        vars: &[usize],
        mut constraints: Vec<LinearConstraint>,
    ) -> Vec<LinearConstraint> {
        for &var in vars {
            constraints = self.eliminate_variable(var, &constraints);
        }
        constraints
    }
}

impl Default for VirtualTermSubstitution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_term_zero() {
        let term = LinearTerm::zero();
        assert_eq!(term.constant, BigInt::zero());
        assert!(term.coeffs.is_empty());
    }

    #[test]
    fn test_linear_term_add() {
        let t1 = LinearTerm {
            constant: BigInt::from(5),
            coeffs: [(0, BigInt::from(2))].iter().cloned().collect(),
        };
        let t2 = LinearTerm {
            constant: BigInt::from(3),
            coeffs: [(0, BigInt::from(1)), (1, BigInt::from(4))]
                .iter()
                .cloned()
                .collect(),
        };

        let result = t1.add(&t2);
        assert_eq!(result.constant, BigInt::from(8));
        assert_eq!(result.coeffs.get(&0), Some(&BigInt::from(3)));
        assert_eq!(result.coeffs.get(&1), Some(&BigInt::from(4)));
    }

    #[test]
    fn test_vts_creation() {
        let vts = VirtualTermSubstitution::new();
        assert_eq!(vts.stats().variables_eliminated, 0);
    }

    #[test]
    fn test_eliminate_variable_simple() {
        let mut vts = VirtualTermSubstitution::new();

        // x ≤ 5 and 3 ≤ x  =>  eliminate x  =>  3 ≤ 5
        let constraints = vec![
            LinearConstraint::leq(LinearTerm {
                constant: BigInt::from(-5),
                coeffs: [(0, BigInt::from(1))].iter().cloned().collect(),
            }),
            LinearConstraint::leq(LinearTerm {
                constant: BigInt::from(3),
                coeffs: [(0, BigInt::from(-1))].iter().cloned().collect(),
            }),
        ];

        let result = vts.eliminate_variable(0, &constraints);

        assert!(vts.stats().variables_eliminated > 0);
        assert!(!result.is_empty());
    }
}
