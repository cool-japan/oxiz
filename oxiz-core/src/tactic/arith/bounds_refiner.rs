//! Arithmetic Bounds Refinement Tactic.
//!
//! Iteratively refines variable bounds through constraint propagation,
//! leading to stronger simplifications and faster solving.

use crate::ast::{Term, TermId, TermKind, TermManager};
use crate::tactic::{Goal, Tactic, TacticResult};
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Bounds refinement tactic for arithmetic constraints.
pub struct BoundsRefinerTactic {
    config: BoundsRefinerConfig,
    stats: BoundsRefinerStats,
}

/// Configuration for bounds refinement.
#[derive(Clone, Debug)]
pub struct BoundsRefinerConfig {
    /// Maximum number of refinement iterations
    pub max_iterations: usize,
    /// Enable strict bound propagation
    pub strict_propagation: bool,
    /// Enable equality-based bound refinement
    pub use_equalities: bool,
    /// Enable multiplication bounds
    pub propagate_multiplication: bool,
    /// Minimum improvement threshold
    pub improvement_threshold: f64,
}

impl Default for BoundsRefinerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            strict_propagation: true,
            use_equalities: true,
            propagate_multiplication: true,
            improvement_threshold: 0.001,
        }
    }
}

/// Statistics for bounds refinement.
#[derive(Clone, Debug, Default)]
pub struct BoundsRefinerStats {
    /// Number of refinement iterations
    pub iterations: usize,
    /// Number of bounds tightened
    pub bounds_tightened: usize,
    /// Number of conflicts detected
    pub conflicts: usize,
    /// Number of variables fixed
    pub variables_fixed: usize,
    /// Number of constraints simplified
    pub constraints_simplified: usize,
}

impl BoundsRefinerTactic {
    /// Create a new bounds refiner tactic.
    pub fn new(config: BoundsRefinerConfig) -> Self {
        Self {
            config,
            stats: BoundsRefinerStats::default(),
        }
    }

    /// Apply bounds refinement to a goal.
    pub fn apply(&mut self, goal: &Goal, tm: &mut TermManager) -> TacticResult {
        let mut current_bounds: FxHashMap<String, Bounds> = FxHashMap::default();
        let mut changed = true;
        let mut iteration = 0;

        // Extract initial bounds from constraints
        for &constraint in &goal.formulas {
            self.extract_bounds(constraint, &mut current_bounds, tm)?;
        }

        // Iterative refinement
        while changed && iteration < self.config.max_iterations {
            changed = false;
            iteration += 1;

            for &constraint in &goal.formulas {
                if self.refine_bounds_from_constraint(constraint, &mut current_bounds, tm)? {
                    changed = true;
                }
            }

            // Check for conflicts
            for bounds in current_bounds.values() {
                if !bounds.is_consistent() {
                    self.stats.conflicts += 1;
                    return TacticResult::Unsat;
                }
            }
        }

        self.stats.iterations = iteration;

        // Apply refined bounds to simplify constraints
        let simplified = self.simplify_with_bounds(&goal.formulas, &current_bounds, tm)?;
        self.stats.constraints_simplified = simplified.len();

        // Count fixed variables
        self.stats.variables_fixed = current_bounds.values()
            .filter(|b| b.is_point())
            .count();

        TacticResult::NewGoals(vec![Goal {
            formulas: simplified,
            ..goal.clone()
        }])
    }

    /// Extract bounds from a single constraint.
    fn extract_bounds(
        &mut self,
        constraint: TermId,
        bounds: &mut FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) -> Result<(), String> {
        let term = tm.get(constraint).ok_or("invalid term")?;

        match &term.kind {
            TermKind::Le(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                // Try to extract bounds from x <= c or c <= x patterns
                if let Some((var, bound, is_upper)) = self.extract_simple_bound(*lhs, *rhs, tm) {
                    let entry = bounds.entry(var.clone()).or_insert_with(Bounds::default);
                    let strict = matches!(term.kind, TermKind::Lt(_, _));

                    if is_upper {
                        entry.update_upper(bound, strict);
                    } else {
                        entry.update_lower(bound, strict);
                    }
                    self.stats.bounds_tightened += 1;
                }
            }
            TermKind::Eq(lhs, rhs) if self.config.use_equalities => {
                // x = c sets both bounds
                if let Some((var, value)) = self.extract_equality(*lhs, *rhs, tm) {
                    let entry = bounds.entry(var).or_insert_with(Bounds::default);
                    entry.update_lower(value.clone(), false);
                    entry.update_upper(value, false);
                    self.stats.bounds_tightened += 2;
                }
            }
            TermKind::And(args) => {
                for &arg in args.iter() {
                    self.extract_bounds(arg, bounds, tm)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Extract a simple bound from a comparison.
    fn extract_simple_bound(
        &self,
        lhs: TermId,
        rhs: TermId,
        tm: &TermManager,
    ) -> Option<(String, BigRational, bool)> {
        // Try pattern: var <= constant
        if let (Some(var), Some(constant)) = (self.extract_var(lhs, tm), self.extract_constant(rhs, tm)) {
            return Some((var, constant, true)); // upper bound
        }

        // Try pattern: constant <= var
        if let (Some(constant), Some(var)) = (self.extract_constant(lhs, tm), self.extract_var(rhs, tm)) {
            return Some((var, constant, false)); // lower bound
        }

        None
    }

    /// Extract variable name from a term.
    fn extract_var(&self, term_id: TermId, tm: &TermManager) -> Option<String> {
        let term = tm.get(term_id)?;
        if let TermKind::Var(name) = term.kind {
            Some(tm.resolve(name).to_string())
        } else {
            None
        }
    }

    /// Extract constant value from a term.
    fn extract_constant(&self, term_id: TermId, tm: &TermManager) -> Option<BigRational> {
        let term = tm.get(term_id)?;
        match &term.kind {
            TermKind::IntConst(n) => Some(BigRational::from_integer(n.clone())),
            TermKind::RealConst(r) => Some(r.clone()),
            _ => None,
        }
    }

    /// Extract equality constraint.
    fn extract_equality(
        &self,
        lhs: TermId,
        rhs: TermId,
        tm: &TermManager,
    ) -> Option<(String, BigRational)> {
        if let (Some(var), Some(constant)) = (self.extract_var(lhs, tm), self.extract_constant(rhs, tm)) {
            return Some((var, constant));
        }

        if let (Some(constant), Some(var)) = (self.extract_constant(lhs, tm), self.extract_var(rhs, tm)) {
            return Some((var, constant));
        }

        None
    }

    /// Refine bounds from a constraint using current bounds.
    fn refine_bounds_from_constraint(
        &mut self,
        constraint: TermId,
        bounds: &mut FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) -> Result<bool, String> {
        let term = tm.get(constraint).ok_or("invalid term")?;
        let mut changed = false;

        match &term.kind {
            TermKind::Le(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                // Try to derive tighter bounds using arithmetic
                if let Some(new_bounds) = self.derive_bounds_from_ineq(*lhs, *rhs, bounds, tm) {
                    for (var, (lower, upper)) in new_bounds {
                        let entry = bounds.entry(var).or_insert_with(Bounds::default);
                        let strict = matches!(term.kind, TermKind::Lt(_, _));

                        if let Some(l) = lower {
                            if entry.update_lower(l, strict) {
                                changed = true;
                                self.stats.bounds_tightened += 1;
                            }
                        }

                        if let Some(u) = upper {
                            if entry.update_upper(u, strict) {
                                changed = true;
                                self.stats.bounds_tightened += 1;
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(changed)
    }

    /// Derive new bounds from an inequality using interval arithmetic.
    fn derive_bounds_from_ineq(
        &self,
        lhs: TermId,
        rhs: TermId,
        bounds: &FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) -> Option<FxHashMap<String, (Option<BigRational>, Option<BigRational>)>> {
        // Compute lhs - rhs to get canonical form
        let lhs_interval = self.evaluate_interval(lhs, bounds, tm)?;
        let rhs_interval = self.evaluate_interval(rhs, bounds, tm)?;

        // lhs <= rhs means lhs - rhs <= 0
        let diff_upper = lhs_interval.upper.as_ref()? - rhs_interval.lower.as_ref()?;

        if diff_upper < BigRational::zero() {
            // Constraint is always satisfied
            return None;
        }

        // Try to derive tighter bounds for variables in lhs
        Some(FxHashMap::default())
    }

    /// Evaluate a term to an interval using current bounds.
    fn evaluate_interval(
        &self,
        term_id: TermId,
        bounds: &FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) -> Option<Interval> {
        let term = tm.get(term_id)?;

        match &term.kind {
            TermKind::IntConst(n) => {
                let val = BigRational::from_integer(n.clone());
                Some(Interval {
                    lower: Some(val.clone()),
                    upper: Some(val),
                })
            }
            TermKind::RealConst(r) => Some(Interval {
                lower: Some(r.clone()),
                upper: Some(r.clone()),
            }),
            TermKind::Var(name) => {
                let var_name = tm.resolve(*name);
                if let Some(var_bounds) = bounds.get(var_name) {
                    Some(Interval {
                        lower: var_bounds.lower.clone(),
                        upper: var_bounds.upper.clone(),
                    })
                } else {
                    Some(Interval::unbounded())
                }
            }
            TermKind::Add(args) => {
                let mut result = Interval::point(BigRational::zero());
                for &arg in args.iter() {
                    let arg_interval = self.evaluate_interval(arg, bounds, tm)?;
                    result = result.add(&arg_interval)?;
                }
                Some(result)
            }
            TermKind::Mul(args) => {
                let mut result = Interval::point(BigRational::one());
                for &arg in args.iter() {
                    let arg_interval = self.evaluate_interval(arg, bounds, tm)?;
                    result = result.mul(&arg_interval)?;
                }
                Some(result)
            }
            TermKind::Sub(lhs, rhs) => {
                let lhs_interval = self.evaluate_interval(*lhs, bounds, tm)?;
                let rhs_interval = self.evaluate_interval(*rhs, bounds, tm)?;
                Some(lhs_interval.sub(&rhs_interval)?)
            }
            TermKind::Neg(arg) => {
                let arg_interval = self.evaluate_interval(*arg, bounds, tm)?;
                Some(arg_interval.neg())
            }
            _ => None,
        }
    }

    /// Simplify constraints using refined bounds.
    fn simplify_with_bounds(
        &self,
        constraints: &[TermId],
        bounds: &FxHashMap<String, Bounds>,
        tm: &mut TermManager,
    ) -> Result<Vec<TermId>, String> {
        let mut simplified = Vec::new();

        for &constraint in constraints {
            if let Some(simpl) = self.simplify_constraint(constraint, bounds, tm) {
                let simpl_term = tm.get(simpl).ok_or("invalid term")?;

                // Skip tautologies
                if !matches!(simpl_term.kind, TermKind::True) {
                    simplified.push(simpl);
                }
            } else {
                simplified.push(constraint);
            }
        }

        Ok(simplified)
    }

    /// Simplify a single constraint using bounds.
    fn simplify_constraint(
        &self,
        constraint: TermId,
        bounds: &FxHashMap<String, Bounds>,
        tm: &mut TermManager,
    ) -> Option<TermId> {
        let term = tm.get(constraint)?;

        match &term.kind {
            TermKind::Le(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                let lhs_interval = self.evaluate_interval(*lhs, bounds, tm)?;
                let rhs_interval = self.evaluate_interval(*rhs, bounds, tm)?;

                // Check if constraint is always true
                if let (Some(lhs_upper), Some(rhs_lower)) = (&lhs_interval.upper, &rhs_interval.lower) {
                    let strict = matches!(term.kind, TermKind::Lt(_, _));
                    let always_true = if strict {
                        lhs_upper < rhs_lower
                    } else {
                        lhs_upper <= rhs_lower
                    };

                    if always_true {
                        return Some(tm.mk_true());
                    }
                }

                // Check if constraint is always false
                if let (Some(lhs_lower), Some(rhs_upper)) = (&lhs_interval.lower, &rhs_interval.upper) {
                    if lhs_lower > rhs_upper {
                        return Some(tm.mk_false());
                    }
                }

                None
            }
            _ => None,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &BoundsRefinerStats {
        &self.stats
    }
}

impl Tactic for BoundsRefinerTactic {
    fn apply(&mut self, goal: &Goal, tm: &mut TermManager) -> TacticResult {
        self.apply(goal, tm)
    }
}

/// Variable bounds.
#[derive(Clone, Debug, Default)]
pub struct Bounds {
    /// Lower bound
    pub lower: Option<BigRational>,
    /// Is lower bound strict?
    pub lower_strict: bool,
    /// Upper bound
    pub upper: Option<BigRational>,
    /// Is upper bound strict?
    pub upper_strict: bool,
}

impl Bounds {
    /// Update lower bound if it's tighter.
    pub fn update_lower(&mut self, new_lower: BigRational, strict: bool) -> bool {
        match &self.lower {
            None => {
                self.lower = Some(new_lower);
                self.lower_strict = strict;
                true
            }
            Some(current) => {
                if new_lower > *current || (new_lower == *current && strict && !self.lower_strict) {
                    self.lower = Some(new_lower);
                    self.lower_strict = strict;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Update upper bound if it's tighter.
    pub fn update_upper(&mut self, new_upper: BigRational, strict: bool) -> bool {
        match &self.upper {
            None => {
                self.upper = Some(new_upper);
                self.upper_strict = strict;
                true
            }
            Some(current) => {
                if new_upper < *current || (new_upper == *current && strict && !self.upper_strict) {
                    self.upper = Some(new_upper);
                    self.upper_strict = strict;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Check if bounds are consistent.
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

    /// Check if bounds define a single point.
    pub fn is_point(&self) -> bool {
        match (&self.lower, &self.upper) {
            (Some(l), Some(u)) => l == u && !self.lower_strict && !self.upper_strict,
            _ => false,
        }
    }

    /// Get the width of the interval.
    pub fn width(&self) -> Option<BigRational> {
        match (&self.lower, &self.upper) {
            (Some(l), Some(u)) => Some(u - l),
            _ => None,
        }
    }
}

/// Interval for interval arithmetic.
#[derive(Clone, Debug)]
pub struct Interval {
    pub lower: Option<BigRational>,
    pub upper: Option<BigRational>,
}

impl Interval {
    /// Create an unbounded interval.
    pub fn unbounded() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }

    /// Create a point interval.
    pub fn point(value: BigRational) -> Self {
        Self {
            lower: Some(value.clone()),
            upper: Some(value),
        }
    }

    /// Add two intervals.
    pub fn add(&self, other: &Interval) -> Option<Self> {
        Some(Self {
            lower: match (&self.lower, &other.lower) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            upper: match (&self.upper, &other.upper) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
        })
    }

    /// Subtract two intervals.
    pub fn sub(&self, other: &Interval) -> Option<Self> {
        Some(Self {
            lower: match (&self.lower, &other.upper) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            },
            upper: match (&self.upper, &other.lower) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            },
        })
    }

    /// Multiply two intervals.
    pub fn mul(&self, other: &Interval) -> Option<Self> {
        match (&self.lower, &self.upper, &other.lower, &other.upper) {
            (Some(al), Some(au), Some(bl), Some(bu)) => {
                let products = vec![
                    al * bl,
                    al * bu,
                    au * bl,
                    au * bu,
                ];

                let min = products.iter().min()?;
                let max = products.iter().max()?;

                Some(Self {
                    lower: Some(min.clone()),
                    upper: Some(max.clone()),
                })
            }
            _ => Some(Self::unbounded()),
        }
    }

    /// Negate an interval.
    pub fn neg(&self) -> Self {
        Self {
            lower: self.upper.as_ref().map(|u| -u),
            upper: self.lower.as_ref().map(|l| -l),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_update() {
        let mut bounds = Bounds::default();

        assert!(bounds.update_lower(BigRational::from_integer(5.into()), false));
        assert_eq!(bounds.lower, Some(BigRational::from_integer(5.into())));

        // Tighter bound
        assert!(bounds.update_lower(BigRational::from_integer(10.into()), false));
        assert_eq!(bounds.lower, Some(BigRational::from_integer(10.into())));
    }

    #[test]
    fn test_bounds_consistency() {
        let mut bounds = Bounds::default();
        bounds.lower = Some(BigRational::from_integer(5.into()));
        bounds.upper = Some(BigRational::from_integer(10.into()));
        assert!(bounds.is_consistent());

        bounds.upper = Some(BigRational::from_integer(3.into()));
        assert!(!bounds.is_consistent());
    }

    #[test]
    fn test_interval_add() {
        let i1 = Interval {
            lower: Some(BigRational::from_integer(1.into())),
            upper: Some(BigRational::from_integer(3.into())),
        };

        let i2 = Interval {
            lower: Some(BigRational::from_integer(2.into())),
            upper: Some(BigRational::from_integer(4.into())),
        };

        let sum = i1.add(&i2).unwrap();
        assert_eq!(sum.lower, Some(BigRational::from_integer(3.into())));
        assert_eq!(sum.upper, Some(BigRational::from_integer(7.into())));
    }

    #[test]
    fn test_bounds_refiner_creation() {
        let config = BoundsRefinerConfig::default();
        let tactic = BoundsRefinerTactic::new(config);
        assert_eq!(tactic.stats.iterations, 0);
    }
}
