//! Arithmetic Bounds Refinement Tactic.
//!
//! Iteratively refines variable bounds through interval propagation, using
//! the refined bounds to discharge constraints that are provably true and to
//! detect an early UNSAT from contradictory bounds.
//!
//! Compared with [`super::arith_bounds::ArithBoundsTactic`], which only reads
//! *literal* `variable`-versus-constant bounds, this tactic evaluates whole
//! arithmetic expressions (`+`, `*`, `-`, unary minus) to intervals, so it can
//! tighten a variable's bound against a compound side and decide comparisons
//! between compound terms.
//!
//! ## References
//!
//! - Z3's `tactic/arith/propagate_ineqs_tactic.cpp`

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::tactic::core::{Goal, SolveResult, Tactic, TacticResult};
use core::fmt;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

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
    /// Enable equality-based bound refinement
    pub use_equalities: bool,
}

impl Default for BoundsRefinerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            use_equalities: true,
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
    /// Number of variables pinned to a single value
    pub variables_fixed: usize,
    /// Number of constraints discharged as provably true
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

    /// Refine the bounds implied by `goal`'s assertions and use them to
    /// discharge provably-true constraints (or report UNSAT).
    ///
    /// This is the real, term-aware entry point: it needs `TermManager`
    /// access that the registry-dispatched [`Tactic::apply`] — whose signature
    /// is `fn apply(&self, goal: &Goal) -> Result<TacticResult>`, with no
    /// manager parameter — structurally cannot provide. `Tactic::apply` on
    /// this type therefore honestly reports [`TacticResult::NotApplicable`]
    /// rather than guessing.
    pub fn refine(&mut self, goal: &Goal, manager: &mut TermManager) -> Result<TacticResult> {
        let all_bounds = self.bounds_for(&goal.assertions, manager)?;

        for bounds in all_bounds.values() {
            if !bounds.is_consistent() {
                self.stats.conflicts += 1;
                return Ok(TacticResult::Solved(SolveResult::Unsat));
            }
        }

        self.stats.variables_fixed = all_bounds.values().filter(|b| b.is_point()).count();

        // Dropping an assertion re-derives the bounds from the *surviving*
        // assertions only. Deciding an assertion against bounds it helped
        // derive would let it justify its own removal — two equivalent
        // constraints would then discharge each other and the goal would lose
        // the bound entirely, which is unsound. This is quadratic in the
        // number of assertions, hence the cap.
        if goal.assertions.len() > self.config.max_iterations {
            return Ok(TacticResult::NotApplicable);
        }

        let mut dropped = vec![false; goal.assertions.len()];
        let mut any_dropped = false;

        for index in 0..goal.assertions.len() {
            let others: Vec<TermId> = goal
                .assertions
                .iter()
                .enumerate()
                .filter(|&(other, _)| other != index && !dropped[other])
                .map(|(_, &assertion)| assertion)
                .collect();

            let other_bounds = self.bounds_for(&others, manager)?;

            match self.decide_constraint(goal.assertions[index], &other_bounds, manager) {
                Some(true) => {
                    // Provably implied by the surviving assertions: drop it.
                    dropped[index] = true;
                    any_dropped = true;
                    self.stats.constraints_simplified += 1;
                }
                Some(false) => {
                    // The surviving assertions force this one to fail, so the
                    // conjunction is unsatisfiable.
                    self.stats.conflicts += 1;
                    return Ok(TacticResult::Solved(SolveResult::Unsat));
                }
                None => {}
            }
        }

        if !any_dropped {
            return Ok(TacticResult::NotApplicable);
        }

        let assertions = goal
            .assertions
            .iter()
            .enumerate()
            .filter(|&(index, _)| !dropped[index])
            .map(|(_, &assertion)| assertion)
            .collect();

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions,
            precision: goal.precision,
        }]))
    }

    /// Derive the bounds implied by `assertions`, refined to a fixed point.
    fn bounds_for(
        &mut self,
        assertions: &[TermId],
        tm: &TermManager,
    ) -> Result<FxHashMap<String, Bounds>> {
        let mut bounds: FxHashMap<String, Bounds> = FxHashMap::default();

        for &constraint in assertions {
            self.extract_bounds(constraint, &mut bounds, tm)?;
        }

        let mut changed = true;
        let mut iteration = 0;
        while changed && iteration < self.config.max_iterations {
            changed = false;
            iteration += 1;

            for &constraint in assertions {
                if self.refine_bounds_from_constraint(constraint, &mut bounds, tm)? {
                    changed = true;
                }
            }
        }

        self.stats.iterations += iteration;
        Ok(bounds)
    }

    /// Extract bounds from a single constraint.
    ///
    /// Iterative (explicit heap stack) over top-level conjunctions: an
    /// `And` nest's depth follows the input formula, so recursing through it
    /// could overflow the native stack on a deeply nested assertion.
    fn extract_bounds(
        &mut self,
        constraint: TermId,
        bounds: &mut FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) -> Result<()> {
        let mut stack = vec![constraint];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }

            let term = tm.get(current).ok_or(OxizError::InvalidTermId(current.0))?;

            match &term.kind {
                TermKind::Le(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                    let strict = matches!(term.kind, TermKind::Lt(_, _));
                    self.apply_simple_bound(*lhs, *rhs, strict, bounds, tm);
                }
                TermKind::Ge(lhs, rhs) | TermKind::Gt(lhs, rhs) => {
                    // `a >= b` is `b <= a`; likewise for the strict form.
                    let strict = matches!(term.kind, TermKind::Gt(_, _));
                    self.apply_simple_bound(*rhs, *lhs, strict, bounds, tm);
                }
                TermKind::Eq(lhs, rhs) if self.config.use_equalities => {
                    // x = c sets both bounds.
                    if let Some((var, value)) = self.extract_equality(*lhs, *rhs, tm) {
                        let entry = bounds.entry(var).or_default();
                        if entry.update_lower(value.clone(), false) {
                            self.stats.bounds_tightened += 1;
                        }
                        if entry.update_upper(value, false) {
                            self.stats.bounds_tightened += 1;
                        }
                    }
                }
                TermKind::And(args) => stack.extend(args.iter().copied()),
                // Any other connective (disjunction, negation, quantifier, a
                // non-arithmetic atom, ...) carries no unconditional bound.
                _ => {}
            }
        }

        Ok(())
    }

    /// Record the bound implied by `lhs <= rhs` (or `lhs < rhs`), if either
    /// side is a bare variable compared against a constant.
    fn apply_simple_bound(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        strict: bool,
        bounds: &mut FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) {
        if let Some((var, bound, is_upper)) = self.extract_simple_bound(lhs, rhs, tm) {
            let entry = bounds.entry(var).or_default();

            let tightened = if is_upper {
                entry.update_upper(bound, strict)
            } else {
                entry.update_lower(bound, strict)
            };

            if tightened {
                self.stats.bounds_tightened += 1;
            }
        }
    }

    /// Extract a simple bound from a comparison `lhs <= rhs`.
    fn extract_simple_bound(
        &self,
        lhs: TermId,
        rhs: TermId,
        tm: &TermManager,
    ) -> Option<(String, BigRational, bool)> {
        // Pattern: var <= constant.
        if let (Some(var), Some(constant)) =
            (self.extract_var(lhs, tm), self.extract_constant(rhs, tm))
        {
            return Some((var, constant, true)); // upper bound
        }

        // Pattern: constant <= var.
        if let (Some(constant), Some(var)) =
            (self.extract_constant(lhs, tm), self.extract_var(rhs, tm))
        {
            return Some((var, constant, false)); // lower bound
        }

        None
    }

    /// Extract variable name from a term.
    fn extract_var(&self, term_id: TermId, tm: &TermManager) -> Option<String> {
        let term = tm.get(term_id)?;
        if let TermKind::Var(name) = term.kind {
            Some(tm.resolve_str(name).to_string())
        } else {
            None
        }
    }

    /// Extract constant value from a term.
    fn extract_constant(&self, term_id: TermId, tm: &TermManager) -> Option<BigRational> {
        let term = tm.get(term_id)?;
        match &term.kind {
            TermKind::IntConst(n) => Some(BigRational::from_integer(n.clone())),
            TermKind::RealConst(r) => Some(BigRational::new(
                BigInt::from(*r.numer()),
                BigInt::from(*r.denom()),
            )),
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
        if let (Some(var), Some(constant)) =
            (self.extract_var(lhs, tm), self.extract_constant(rhs, tm))
        {
            return Some((var, constant));
        }

        if let (Some(constant), Some(var)) =
            (self.extract_constant(lhs, tm), self.extract_var(rhs, tm))
        {
            return Some((var, constant));
        }

        None
    }

    /// Refine bounds from a constraint using the bounds derived so far.
    ///
    /// For `lhs <= rhs` (or `lhs < rhs`): if one side is a bare variable and
    /// the other side evaluates to an interval under the current bounds, the
    /// variable's bound is tightened against that interval's endpoint. This is
    /// what makes the pass iterative — a tightened bound can make the next
    /// interval evaluation sharper.
    fn refine_bounds_from_constraint(
        &mut self,
        constraint: TermId,
        bounds: &mut FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) -> Result<bool> {
        let term = tm
            .get(constraint)
            .ok_or(OxizError::InvalidTermId(constraint.0))?;

        let (lhs, rhs, strict) = match &term.kind {
            TermKind::Le(lhs, rhs) => (*lhs, *rhs, false),
            TermKind::Lt(lhs, rhs) => (*lhs, *rhs, true),
            TermKind::Ge(lhs, rhs) => (*rhs, *lhs, false),
            TermKind::Gt(lhs, rhs) => (*rhs, *lhs, true),
            _ => return Ok(false),
        };

        let mut changed = false;

        // `var <= rhs`: the variable is bounded above by rhs's upper endpoint.
        if let Some(var) = self.extract_var(lhs, tm)
            && let Some(rhs_interval) = self.evaluate_interval(rhs, bounds, tm)
            && let Some(upper) = rhs_interval.upper
        {
            let entry = bounds.entry(var).or_default();
            if entry.update_upper(upper, strict) {
                changed = true;
                self.stats.bounds_tightened += 1;
            }
        }

        // `lhs <= var`: the variable is bounded below by lhs's lower endpoint.
        if let Some(var) = self.extract_var(rhs, tm)
            && let Some(lhs_interval) = self.evaluate_interval(lhs, bounds, tm)
            && let Some(lower) = lhs_interval.lower
        {
            let entry = bounds.entry(var).or_default();
            if entry.update_lower(lower, strict) {
                changed = true;
                self.stats.bounds_tightened += 1;
            }
        }

        Ok(changed)
    }

    /// Evaluate a term to an interval using current bounds.
    ///
    /// Iterative (explicit heap stack): an arithmetic expression's depth
    /// follows the input formula, and `-> Option<Interval>` cannot distinguish
    /// "too deep" from "not evaluable", so recursing here could only fail by
    /// overflowing the native stack. `None` keeps its single, honest meaning:
    /// the term is outside the supported linear/multiplicative fragment.
    fn evaluate_interval(
        &self,
        term_id: TermId,
        bounds: &FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) -> Option<Interval> {
        /// Work item for the iterative interval evaluation.
        enum EvalTask {
            /// Evaluate this subterm.
            Enter(TermId),
            /// Sum the top `n` results.
            FoldAdd(usize),
            /// Multiply the top `n` results.
            FoldMul(usize),
            /// Subtract the top result from the one below it.
            FoldSub,
            /// Negate the top result.
            FoldNeg,
        }

        /// Detach the top `n` results, preserving their original order.
        fn take(results: &mut Vec<Interval>, n: usize) -> Option<Vec<Interval>> {
            if results.len() < n {
                return None;
            }
            let start = results.len() - n;
            Some(results.split_off(start))
        }

        let mut tasks = vec![EvalTask::Enter(term_id)];
        let mut results: Vec<Interval> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                EvalTask::Enter(current) => {
                    let term = tm.get(current)?;
                    match &term.kind {
                        TermKind::IntConst(n) => {
                            results.push(Interval::point(BigRational::from_integer(n.clone())));
                        }
                        TermKind::RealConst(r) => results.push(Interval::point(BigRational::new(
                            BigInt::from(*r.numer()),
                            BigInt::from(*r.denom()),
                        ))),
                        TermKind::Var(name) => {
                            let var_name = tm.resolve_str(*name);
                            match bounds.get(var_name) {
                                Some(var_bounds) => results.push(Interval {
                                    lower: var_bounds.lower.clone(),
                                    upper: var_bounds.upper.clone(),
                                }),
                                None => results.push(Interval::unbounded()),
                            }
                        }
                        TermKind::Add(args) => {
                            tasks.push(EvalTask::FoldAdd(args.len()));
                            tasks.extend(args.iter().rev().map(|&a| EvalTask::Enter(a)));
                        }
                        TermKind::Mul(args) => {
                            tasks.push(EvalTask::FoldMul(args.len()));
                            tasks.extend(args.iter().rev().map(|&a| EvalTask::Enter(a)));
                        }
                        TermKind::Sub(lhs, rhs) => {
                            let (lhs, rhs) = (*lhs, *rhs);
                            tasks.push(EvalTask::FoldSub);
                            tasks.push(EvalTask::Enter(rhs));
                            tasks.push(EvalTask::Enter(lhs));
                        }
                        TermKind::Neg(arg) => {
                            let arg = *arg;
                            tasks.push(EvalTask::FoldNeg);
                            tasks.push(EvalTask::Enter(arg));
                        }
                        // Outside the supported fragment.
                        _ => return None,
                    }
                }
                EvalTask::FoldAdd(arity) => {
                    let operands = take(&mut results, arity)?;
                    let mut acc = Interval::point(BigRational::zero());
                    for operand in &operands {
                        acc = acc.add(operand)?;
                    }
                    results.push(acc);
                }
                EvalTask::FoldMul(arity) => {
                    let operands = take(&mut results, arity)?;
                    let mut acc = Interval::point(BigRational::one());
                    for operand in &operands {
                        acc = acc.mul(operand)?;
                    }
                    results.push(acc);
                }
                EvalTask::FoldSub => {
                    let operands = take(&mut results, 2)?;
                    let (lhs, rhs) = (operands.first()?, operands.get(1)?);
                    results.push(lhs.sub(rhs)?);
                }
                EvalTask::FoldNeg => {
                    let operand = take(&mut results, 1)?;
                    results.push(operand.first()?.neg());
                }
            }
        }

        results.pop()
    }

    /// Decide a single comparison against the refined bounds.
    ///
    /// `Some(true)` / `Some(false)` mean the constraint holds / fails for
    /// *every* assignment inside the bounds; `None` means undecided.
    fn decide_constraint(
        &self,
        constraint: TermId,
        bounds: &FxHashMap<String, Bounds>,
        tm: &TermManager,
    ) -> Option<bool> {
        let term = tm.get(constraint)?;

        let (lhs, rhs, strict) = match &term.kind {
            TermKind::Le(lhs, rhs) => (*lhs, *rhs, false),
            TermKind::Lt(lhs, rhs) => (*lhs, *rhs, true),
            TermKind::Ge(lhs, rhs) => (*rhs, *lhs, false),
            TermKind::Gt(lhs, rhs) => (*rhs, *lhs, true),
            _ => return None,
        };

        let lhs_interval = self.evaluate_interval(lhs, bounds, tm)?;
        let rhs_interval = self.evaluate_interval(rhs, bounds, tm)?;

        // Always true: every lhs value is below every rhs value.
        if let (Some(lhs_upper), Some(rhs_lower)) = (&lhs_interval.upper, &rhs_interval.lower) {
            let always_true = if strict {
                lhs_upper < rhs_lower
            } else {
                lhs_upper <= rhs_lower
            };
            if always_true {
                return Some(true);
            }
        }

        // Always false: every lhs value is strictly above every rhs value.
        if let (Some(lhs_lower), Some(rhs_upper)) = (&lhs_interval.lower, &rhs_interval.upper) {
            let always_false = if strict {
                lhs_lower >= rhs_upper
            } else {
                lhs_lower > rhs_upper
            };
            if always_false {
                return Some(false);
            }
        }

        None
    }

    /// Get statistics.
    pub fn stats(&self) -> &BoundsRefinerStats {
        &self.stats
    }
}

impl Default for BoundsRefinerTactic {
    fn default() -> Self {
        Self::new(BoundsRefinerConfig::default())
    }
}

impl Tactic for BoundsRefinerTactic {
    fn name(&self) -> &str {
        "bounds-refiner"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // See the doc comment on `BoundsRefinerTactic::refine`: this dispatch
        // path has no `TermManager` access and therefore honestly reports
        // NotApplicable rather than fabricating a result.
        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        "Refine arithmetic bounds by interval propagation (see BoundsRefinerTactic::refine for the real, TermManager-aware entry point)"
    }
}

impl fmt::Debug for BoundsRefinerTactic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoundsRefinerTactic")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    /// Lower endpoint (inclusive), or `None` for unbounded below.
    pub lower: Option<BigRational>,
    /// Upper endpoint (inclusive), or `None` for unbounded above.
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
                let products = [al * bl, al * bu, au * bl, au * bu];

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

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    #[test]
    fn test_bounds_update() {
        let mut bounds = Bounds::default();

        assert!(bounds.update_lower(rat(5), false));
        assert_eq!(bounds.lower, Some(rat(5)));

        // Tighter bound
        assert!(bounds.update_lower(rat(10), false));
        assert_eq!(bounds.lower, Some(rat(10)));

        // Looser bound is rejected
        assert!(!bounds.update_lower(rat(1), false));
        assert_eq!(bounds.lower, Some(rat(10)));
    }

    #[test]
    fn test_bounds_consistency() {
        let mut bounds = Bounds {
            lower: Some(rat(5)),
            upper: Some(rat(10)),
            ..Bounds::default()
        };
        assert!(bounds.is_consistent());

        bounds.upper = Some(rat(3));
        assert!(!bounds.is_consistent());
    }

    #[test]
    fn test_interval_add() {
        let i1 = Interval {
            lower: Some(rat(1)),
            upper: Some(rat(3)),
        };

        let i2 = Interval {
            lower: Some(rat(2)),
            upper: Some(rat(4)),
        };

        let sum = i1.add(&i2).expect("test operation should succeed");
        assert_eq!(sum.lower, Some(rat(3)));
        assert_eq!(sum.upper, Some(rat(7)));
    }

    #[test]
    fn test_bounds_refiner_creation() {
        let config = BoundsRefinerConfig::default();
        let tactic = BoundsRefinerTactic::new(config);
        assert_eq!(tactic.stats().iterations, 0);
    }

    #[test]
    fn test_manager_free_dispatch_is_not_applicable() {
        let tactic = BoundsRefinerTactic::default();
        let result = tactic
            .apply(&Goal::empty())
            .expect("test operation should succeed");
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_refine_drops_provably_true_constraint() {
        // 0 <= x, x <= 5  ==>  x < 10 is provably true and is dropped.
        let mut manager = TermManager::default();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let lower = manager.mk_le(zero, x);
        let upper = manager.mk_le(x, five);
        let implied = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![lower, upper, implied]);
        let mut tactic = BoundsRefinerTactic::default();
        let result = tactic
            .refine(&goal, &mut manager)
            .expect("test operation should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert!(!goals[0].assertions.contains(&implied));
                assert!(goals[0].assertions.contains(&lower));
                assert!(goals[0].assertions.contains(&upper));
            }
            other => panic!("expected SubGoals, got {other:?}"),
        }
    }

    #[test]
    fn test_refine_detects_contradictory_bounds() {
        // 10 <= x and x <= 5 is UNSAT.
        let mut manager = TermManager::default();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let lower = manager.mk_le(ten, x);
        let upper = manager.mk_le(x, five);

        let goal = Goal::new(vec![lower, upper]);
        let mut tactic = BoundsRefinerTactic::default();
        let result = tactic
            .refine(&goal, &mut manager)
            .expect("test operation should succeed");

        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    /// Run `body` on a worker thread with a deliberately small (128 KiB) stack,
    /// so a recursive walk over a deep term would abort instead of getting
    /// away with the main thread's much larger stack.
    fn run_with_small_stack<F>(body: F)
    where
        F: FnOnce() + Send + 'static,
    {
        std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(body)
            .expect("thread spawn should succeed")
            .join()
            .expect("deep-nesting walk must not overflow the stack");
    }

    #[test]
    fn test_extract_bounds_handles_deeply_nested_conjunctions() {
        run_with_small_stack(|| {
            const DEPTH: usize = 6_250;

            let mut manager = TermManager::default();
            let bool_sort = manager.sorts.bool_sort;
            let int_sort = manager.sorts.int_sort;
            let x = manager.mk_var("x", int_sort);
            let zero = manager.mk_int(0);
            let five = manager.mk_int(5);

            // `And(And(...And(0 <= x, x <= 5)...), x <= 5)`, nested 50k deep.
            // `intern_term` bypasses `mk_and`'s flattening so the nesting is
            // real.
            let lower = manager.mk_le(zero, x);
            let upper = manager.mk_le(x, five);
            let mut current = lower;
            for _ in 0..DEPTH {
                current =
                    manager.intern_term(TermKind::And(vec![current, upper].into()), bool_sort);
            }

            let mut tactic = BoundsRefinerTactic::default();
            let mut bounds = FxHashMap::default();
            tactic
                .extract_bounds(current, &mut bounds, &manager)
                .expect("deep extraction should succeed");

            let x_bounds = bounds.get("x").expect("x must have bounds");
            assert_eq!(x_bounds.lower, Some(rat(0)));
            assert_eq!(x_bounds.upper, Some(rat(5)));
        });
    }

    #[test]
    fn test_evaluate_interval_handles_deeply_nested_arithmetic() {
        run_with_small_stack(|| {
            const DEPTH: usize = 6_250;

            let mut manager = TermManager::default();
            let int_sort = manager.sorts.int_sort;
            let one = manager.mk_int(1);

            // `((...(1 + 1) + 1...) + 1)`, nested 50k deep. `intern_term`
            // bypasses `mk_add`'s constant folding and flattening.
            let mut current = one;
            for _ in 0..DEPTH {
                current = manager.intern_term(TermKind::Add(vec![current, one].into()), int_sort);
            }

            let tactic = BoundsRefinerTactic::default();
            let bounds = FxHashMap::default();
            let interval = tactic
                .evaluate_interval(current, &bounds, &manager)
                .expect("deep interval evaluation should succeed");

            let expected = rat(DEPTH as i64 + 1);
            assert_eq!(interval.lower, Some(expected.clone()));
            assert_eq!(interval.upper, Some(expected));
        });
    }

    #[test]
    fn test_evaluate_interval_reports_unsupported_terms_as_none() {
        let mut manager = TermManager::default();
        let bool_sort = manager.sorts.bool_sort;
        let flag = manager.mk_var("p", bool_sort);
        let negated = manager.mk_not(flag);

        let tactic = BoundsRefinerTactic::default();
        let bounds = FxHashMap::default();
        assert!(
            tactic
                .evaluate_interval(negated, &bounds, &manager)
                .is_none()
        );
    }

    #[test]
    fn test_refine_keeps_undecided_constraint() {
        // 0 <= x alone says nothing about x <= 5.
        let mut manager = TermManager::default();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let five = manager.mk_int(5);

        let lower = manager.mk_le(zero, x);
        let undecided = manager.mk_le(x, five);

        let goal = Goal::new(vec![lower, undecided]);
        let mut tactic = BoundsRefinerTactic::default();
        let result = tactic
            .refine(&goal, &mut manager)
            .expect("test operation should succeed");

        assert!(matches!(result, TacticResult::NotApplicable));
    }
}
