//! Advanced Bound Tightening for Linear Real Arithmetic.
//!
//! Implements sophisticated bound propagation and tightening techniques
//! for LRA theory to improve constraint solving efficiency.

use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Variable identifier for LRA.
pub type VarId = usize;

/// Bound information for a variable.
#[derive(Clone, Debug)]
pub struct Bound {
    /// Lower bound (if any)
    pub lower: Option<BigRational>,
    /// Upper bound (if any)
    pub upper: Option<BigRational>,
    /// Is lower bound strict? (x > bound vs x >= bound)
    pub lower_strict: bool,
    /// Is upper bound strict? (x < bound vs x <= bound)
    pub upper_strict: bool,
    /// Reason for lower bound (clause/constraint ID)
    pub lower_reason: Option<usize>,
    /// Reason for upper bound
    pub upper_reason: Option<usize>,
}

impl Bound {
    /// Create a new unconstrained bound.
    pub fn new() -> Self {
        Self {
            lower: None,
            upper: None,
            lower_strict: false,
            upper_strict: false,
            lower_reason: None,
            upper_reason: None,
        }
    }

    /// Check if bounds are consistent (lower <= upper).
    pub fn is_consistent(&self) -> bool {
        match (&self.lower, &self.upper) {
            (Some(l), Some(u)) => {
                if self.lower_strict || self.upper_strict {
                    l < u
                } else {
                    l <= u
                }
            }
            _ => true,
        }
    }

    /// Get the width of the bound interval.
    pub fn width(&self) -> Option<BigRational> {
        match (&self.lower, &self.upper) {
            (Some(l), Some(u)) => Some(u - l),
            _ => None,
        }
    }

    /// Check if the bound is a point (lower == upper).
    pub fn is_point(&self) -> bool {
        match (&self.lower, &self.upper) {
            (Some(l), Some(u)) => l == u && !self.lower_strict && !self.upper_strict,
            _ => false,
        }
    }
}

impl Default for Bound {
    fn default() -> Self {
        Self::new()
    }
}

/// Bound tightening engine for LRA.
pub struct BoundTightener {
    /// Current bounds for each variable
    bounds: FxHashMap<VarId, Bound>,
    /// Propagation queue
    queue: VecDeque<VarId>,
    /// Variables that have been enqueued
    in_queue: FxHashSet<VarId>,
    /// Statistics
    stats: TighteningStats,
}

/// Statistics about bound tightening.
#[derive(Clone, Debug, Default)]
pub struct TighteningStats {
    /// Number of bounds tightened
    pub bounds_tightened: usize,
    /// Number of propagations performed
    pub propagations: usize,
    /// Number of conflicts detected
    pub conflicts_detected: usize,
    /// Number of fixed variables
    pub fixed_variables: usize,
}

impl BoundTightener {
    /// Create a new bound tightener.
    pub fn new() -> Self {
        Self {
            bounds: FxHashMap::default(),
            queue: VecDeque::new(),
            in_queue: FxHashSet::default(),
            stats: TighteningStats::default(),
        }
    }

    /// Get the bound for a variable.
    pub fn get_bound(&self, var: VarId) -> Bound {
        self.bounds.get(&var).cloned().unwrap_or_default()
    }

    /// Try to tighten the lower bound of a variable.
    pub fn tighten_lower(
        &mut self,
        var: VarId,
        new_lower: BigRational,
        strict: bool,
        reason: usize,
    ) -> Result<bool, ()> {
        let bound = self.bounds.entry(var).or_default();

        // Check if new bound is better than current
        let is_better = match &bound.lower {
            None => true,
            Some(current) => {
                if new_lower > *current {
                    true
                } else if new_lower == *current && strict && !bound.lower_strict {
                    true
                } else {
                    false
                }
            }
        };

        if is_better {
            bound.lower = Some(new_lower.clone());
            bound.lower_strict = strict;
            bound.lower_reason = Some(reason);
            self.stats.bounds_tightened += 1;

            // Check consistency
            if !bound.is_consistent() {
                self.stats.conflicts_detected += 1;
                return Err(());
            }

            // Enqueue for propagation
            if !self.in_queue.contains(&var) {
                self.queue.push_back(var);
                self.in_queue.insert(var);
            }

            // Check if variable became fixed
            if bound.is_point() {
                self.stats.fixed_variables += 1;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Try to tighten the upper bound of a variable.
    pub fn tighten_upper(
        &mut self,
        var: VarId,
        new_upper: BigRational,
        strict: bool,
        reason: usize,
    ) -> Result<bool, ()> {
        let bound = self.bounds.entry(var).or_default();

        let is_better = match &bound.upper {
            None => true,
            Some(current) => {
                if new_upper < *current {
                    true
                } else if new_upper == *current && strict && !bound.upper_strict {
                    true
                } else {
                    false
                }
            }
        };

        if is_better {
            bound.upper = Some(new_upper.clone());
            bound.upper_strict = strict;
            bound.upper_reason = Some(reason);
            self.stats.bounds_tightened += 1;

            if !bound.is_consistent() {
                self.stats.conflicts_detected += 1;
                return Err(());
            }

            if !self.in_queue.contains(&var) {
                self.queue.push_back(var);
                self.in_queue.insert(var);
            }

            if bound.is_point() {
                self.stats.fixed_variables += 1;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Propagate bounds through linear constraints.
    ///
    /// For constraint: a₁x₁ + a₂x₂ + ... + aₙxₙ ≤ b,
    /// we can derive: xᵢ ≤ (b - Σⱼ≠ᵢ aⱼxⱼ) / aᵢ
    pub fn propagate_constraint(
        &mut self,
        constraint: &LinearConstraint,
    ) -> Result<(), ()> {
        self.stats.propagations += 1;

        // For each variable in the constraint, try to derive bounds
        for i in 0..constraint.coeffs.len() {
            let (var, coeff) = constraint.coeffs[i];

            if coeff.is_zero() {
                continue;
            }

            // Compute bound contribution from other variables
            let mut lower_contrib = constraint.rhs.clone();
            let mut upper_contrib = constraint.rhs.clone();

            for j in 0..constraint.coeffs.len() {
                if i == j {
                    continue;
                }

                let (other_var, other_coeff) = constraint.coeffs[j];
                let other_bound = self.get_bound(other_var);

                // Subtract contribution of other variables
                if other_coeff > BigRational::zero() {
                    // Positive coefficient
                    if let Some(other_lower) = &other_bound.lower {
                        upper_contrib = &upper_contrib - &other_coeff * other_lower;
                    }
                    if let Some(other_upper) = &other_bound.upper {
                        lower_contrib = &lower_contrib - &other_coeff * other_upper;
                    }
                } else {
                    // Negative coefficient
                    if let Some(other_upper) = &other_bound.upper {
                        upper_contrib = &upper_contrib - &other_coeff * other_upper;
                    }
                    if let Some(other_lower) = &other_bound.lower {
                        lower_contrib = &lower_contrib - &other_coeff * other_lower;
                    }
                }
            }

            // Derive bound for xᵢ
            if coeff > BigRational::zero() {
                // aᵢ > 0: xᵢ ≤ upper_contrib / aᵢ
                let new_upper = upper_contrib / &coeff;
                let strict = constraint.strict;
                self.tighten_upper(var, new_upper, strict, constraint.id)?;
            } else {
                // aᵢ < 0: xᵢ ≥ upper_contrib / aᵢ (flips inequality)
                let new_lower = upper_contrib / &coeff;
                let strict = constraint.strict;
                self.tighten_lower(var, new_lower, strict, constraint.id)?;
            }
        }

        Ok(())
    }

    /// Run bound propagation to fixed point.
    pub fn propagate_to_fixpoint(
        &mut self,
        constraints: &[LinearConstraint],
    ) -> Result<(), ()> {
        // Initial propagation
        for constraint in constraints {
            self.propagate_constraint(constraint)?;
        }

        // Iterative propagation
        while let Some(var) = self.queue.pop_front() {
            self.in_queue.remove(&var);

            // Propagate all constraints involving this variable
            for constraint in constraints {
                if constraint.involves(var) {
                    self.propagate_constraint(constraint)?;
                }
            }
        }

        Ok(())
    }

    /// Get all fixed variables (bounds are equal).
    pub fn get_fixed_variables(&self) -> Vec<(VarId, BigRational)> {
        self.bounds
            .iter()
            .filter_map(|(&var, bound)| {
                if bound.is_point() {
                    bound.lower.clone().map(|val| (var, val))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get statistics.
    pub fn stats(&self) -> &TighteningStats {
        &self.stats
    }

    /// Reset all bounds.
    pub fn reset(&mut self) {
        self.bounds.clear();
        self.queue.clear();
        self.in_queue.clear();
        self.stats = TighteningStats::default();
    }
}

impl Default for BoundTightener {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a linear constraint: Σ aᵢxᵢ ≤ b (or < b if strict).
#[derive(Clone, Debug)]
pub struct LinearConstraint {
    /// Constraint ID
    pub id: usize,
    /// Coefficients and variables: (var, coeff)
    pub coeffs: Vec<(VarId, BigRational)>,
    /// Right-hand side
    pub rhs: BigRational,
    /// Is the inequality strict? (< vs ≤)
    pub strict: bool,
}

impl LinearConstraint {
    /// Check if this constraint involves a variable.
    pub fn involves(&self, var: VarId) -> bool {
        self.coeffs.iter().any(|(v, _)| *v == var)
    }

    /// Normalize the constraint (make leading coefficient positive).
    pub fn normalize(&mut self) {
        if let Some((_, lead_coeff)) = self.coeffs.first() {
            if *lead_coeff < BigRational::zero() {
                // Flip signs
                for (_, coeff) in &mut self.coeffs {
                    *coeff = -coeff.clone();
                }
                self.rhs = -self.rhs.clone();
            }
        }
    }

    /// Simplify the constraint by removing zero coefficients.
    pub fn simplify(&mut self) {
        self.coeffs.retain(|(_, coeff)| !coeff.is_zero());
    }
}

/// Interval arithmetic for bound propagation.
pub struct IntervalAnalyzer;

impl IntervalAnalyzer {
    /// Compute the interval for a linear expression.
    pub fn evaluate_expression(
        expr: &[(VarId, BigRational)],
        bounds: &FxHashMap<VarId, Bound>,
    ) -> (Option<BigRational>, Option<BigRational>) {
        let mut lower = Some(BigRational::zero());
        let mut upper = Some(BigRational::zero());

        for &(var, ref coeff) in expr {
            let var_bound = bounds.get(&var).cloned().unwrap_or_default();

            if coeff > &BigRational::zero() {
                // Positive coefficient
                if let Some(l) = &var_bound.lower {
                    if let Some(curr_lower) = &mut lower {
                        *curr_lower = curr_lower.clone() + coeff * l;
                    }
                } else {
                    lower = None;
                }

                if let Some(u) = &var_bound.upper {
                    if let Some(curr_upper) = &mut upper {
                        *curr_upper = curr_upper.clone() + coeff * u;
                    }
                } else {
                    upper = None;
                }
            } else {
                // Negative coefficient
                if let Some(u) = &var_bound.upper {
                    if let Some(curr_lower) = &mut lower {
                        *curr_lower = curr_lower.clone() + coeff * u;
                    }
                } else {
                    lower = None;
                }

                if let Some(l) = &var_bound.lower {
                    if let Some(curr_upper) = &mut upper {
                        *curr_upper = curr_upper.clone() + coeff * l;
                    }
                } else {
                    upper = None;
                }
            }
        }

        (lower, upper)
    }

    /// Check if an expression is definitely positive/negative/zero.
    pub fn sign_analysis(
        expr: &[(VarId, BigRational)],
        bounds: &FxHashMap<VarId, Bound>,
    ) -> SignResult {
        let (lower, upper) = Self::evaluate_expression(expr, bounds);

        match (lower, upper) {
            (Some(l), Some(u)) => {
                if l > BigRational::zero() {
                    SignResult::Positive
                } else if u < BigRational::zero() {
                    SignResult::Negative
                } else if l == BigRational::zero() && u == BigRational::zero() {
                    SignResult::Zero
                } else {
                    SignResult::Unknown
                }
            }
            _ => SignResult::Unknown,
        }
    }
}

/// Result of sign analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignResult {
    /// Expression is definitely positive
    Positive,
    /// Expression is definitely negative
    Negative,
    /// Expression is definitely zero
    Zero,
    /// Sign is unknown
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_bound_creation() {
        let bound = Bound::new();
        assert!(bound.lower.is_none());
        assert!(bound.upper.is_none());
        assert!(bound.is_consistent());
    }

    #[test]
    fn test_bound_consistency() {
        let mut bound = Bound::new();
        bound.lower = Some(rat(5));
        bound.upper = Some(rat(10));
        assert!(bound.is_consistent());

        bound.upper = Some(rat(3));
        assert!(!bound.is_consistent());
    }

    #[test]
    fn test_bound_width() {
        let mut bound = Bound::new();
        bound.lower = Some(rat(5));
        bound.upper = Some(rat(10));

        assert_eq!(bound.width(), Some(rat(5)));
    }

    #[test]
    fn test_tighten_lower() {
        let mut tightener = BoundTightener::new();

        let result = tightener.tighten_lower(0, rat(5), false, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);

        let bound = tightener.get_bound(0);
        assert_eq!(bound.lower, Some(rat(5)));
    }

    #[test]
    fn test_tighten_conflict() {
        let mut tightener = BoundTightener::new();

        tightener.tighten_lower(0, rat(10), false, 0).unwrap();
        let result = tightener.tighten_upper(0, rat(5), false, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_fixed_variable() {
        let mut tightener = BoundTightener::new();

        tightener.tighten_lower(0, rat(5), false, 0).unwrap();
        tightener.tighten_upper(0, rat(5), false, 1).unwrap();

        let fixed = tightener.get_fixed_variables();
        assert_eq!(fixed.len(), 1);
        assert_eq!(fixed[0], (0, rat(5)));
    }

    #[test]
    fn test_interval_evaluation() {
        let mut bounds = FxHashMap::default();

        let mut bound0 = Bound::new();
        bound0.lower = Some(rat(1));
        bound0.upper = Some(rat(3));
        bounds.insert(0, bound0);

        let mut bound1 = Bound::new();
        bound1.lower = Some(rat(2));
        bound1.upper = Some(rat(4));
        bounds.insert(1, bound1);

        // Expression: 2*x0 + x1
        let expr = vec![(0, rat(2)), (1, rat(1))];

        let (lower, upper) = IntervalAnalyzer::evaluate_expression(&expr, &bounds);

        assert_eq!(lower, Some(rat(4)));  // 2*1 + 2 = 4
        assert_eq!(upper, Some(rat(10))); // 2*3 + 4 = 10
    }

    #[test]
    fn test_sign_analysis() {
        let mut bounds = FxHashMap::default();

        let mut bound = Bound::new();
        bound.lower = Some(rat(5));
        bound.upper = Some(rat(10));
        bounds.insert(0, bound);

        let expr = vec![(0, rat(1))];

        let sign = IntervalAnalyzer::sign_analysis(&expr, &bounds);
        assert_eq!(sign, SignResult::Positive);
    }
}
