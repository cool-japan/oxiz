// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

use super::delta::DeltaRational;
use crate::config::{PivotingRule, SimplexConfig};
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::{One, Signed, Zero};
#[cfg(feature = "profiling")]
use oxiz_core::profiling::{ProfilingCategory, ScopedTimer};
use smallvec::SmallVec;
/// Variable index
pub type VarId = u32;
/// GCD of two `i128` values (used by the checked-rational helpers below to
/// reduce results computed via `i128` intermediates before narrowing back
/// to `i64`).
fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}
/// Build a fully-reduced `Rational64` from an `i128` numerator/denominator
/// pair, returning `None` if the reduced value does not fit back into
/// `i64`. All of the checked-rational helpers below route through this so
/// that a value which cannot be represented as a `Rational64` is reported
/// as `None` (overflow) rather than silently truncated.
pub(crate) fn checked_ratio_i128(numer: i128, denom: i128) -> Option<Rational64> {
    if denom == 0 {
        return None;
    }
    let g = gcd_i128(numer, denom);
    let g = if g == 0 { 1 } else { g };
    let mut n = numer / g;
    let mut d = denom / g;
    if d < 0 {
        n = -n;
        d = -d;
    }
    if !(i64::MIN as i128..=i64::MAX as i128).contains(&n) || d > i64::MAX as i128 {
        return None;
    }
    // `new_raw`, not `new`: `n / g` and `d / g` above are already coprime (a
    // division by the gcd always is) and `d` was just sign-normalized
    // non-negative, so this pair is already in `Ratio`'s canonical form —
    // `new`'s own gcd-reduction pass would be redundant work repeating what
    // this function just did.
    Some(Rational64::new_raw(n as i64, d as i64))
}
/// Checked rational multiplication: `a * b`, via `i128` intermediates.
/// Returns `None` on overflow instead of silently wrapping (the `i64`-based
/// `Rational64` multiplication used by `num-rational`'s `Mul` impl does not
/// check for overflow: it panics in debug builds and silently wraps to a
/// wrong coefficient in release builds).
pub(crate) fn checked_mul_r64(a: Rational64, b: Rational64) -> Option<Rational64> {
    let numer = (*a.numer() as i128).checked_mul(*b.numer() as i128)?;
    let denom = (*a.denom() as i128).checked_mul(*b.denom() as i128)?;
    checked_ratio_i128(numer, denom)
}
/// Checked rational division: `a / b`. Returns `None` if `b` is zero or the
/// result overflows `i64` after reduction.
pub(crate) fn checked_div_r64(a: Rational64, b: Rational64) -> Option<Rational64> {
    if b.numer() == &0 {
        return None;
    }
    let numer = (*a.numer() as i128).checked_mul(*b.denom() as i128)?;
    let denom = (*a.denom() as i128).checked_mul(*b.numer() as i128)?;
    checked_ratio_i128(numer, denom)
}
/// Checked rational addition: `a + b`. Returns `None` on overflow.
pub(crate) fn checked_add_r64(a: Rational64, b: Rational64) -> Option<Rational64> {
    let ad = (*a.numer() as i128).checked_mul(*b.denom() as i128)?;
    let cb = (*b.numer() as i128).checked_mul(*a.denom() as i128)?;
    let numer = ad.checked_add(cb)?;
    let denom = (*a.denom() as i128).checked_mul(*b.denom() as i128)?;
    checked_ratio_i128(numer, denom)
}
/// Checked rational negation: `-a`. Only fails for the `i64::MIN` edge
/// case, whose absolute value has no positive `i64` representation.
fn checked_neg_r64(a: Rational64) -> Option<Rational64> {
    let n = (*a.numer() as i128).checked_neg()?;
    if !(i64::MIN as i128..=i64::MAX as i128).contains(&n) {
        return None;
    }
    // `new_raw`: `a` is already a valid, reduced `Rational64` (the
    // invariant every `Ratio` value carries), and negating only the
    // numerator changes neither `gcd(|numer|, denom)` nor the
    // denominator's sign, so the result is still in canonical form.
    Some(Rational64::new_raw(n as i64, *a.denom()))
}
/// Checked rational reciprocal: `1 / a`. Returns `None` if `a` is zero.
fn checked_recip_r64(a: Rational64) -> Option<Rational64> {
    if a.numer() == &0 {
        return None;
    }
    checked_ratio_i128(*a.denom() as i128, *a.numer() as i128)
}
/// Split a full reason list into `(primary, auxiliary)`, deduplicating so a
/// reason never appears twice. Returns `None` for an empty list (a derived
/// bound with no recorded antecedent is not applied rather than fabricating a
/// reason).
fn split_reasons(reasons: SmallVec<[u32; 4]>) -> Option<(u32, SmallVec<[u32; 4]>)> {
    let mut iter = reasons.into_iter();
    let primary = iter.next()?;
    let mut aux: SmallVec<[u32; 4]> = SmallVec::new();
    for r in iter {
        if r != primary && !aux.contains(&r) {
            aux.push(r);
        }
    }
    Some((primary, aux))
}
/// A linear expression: sum of (coefficient, variable) pairs + constant
#[derive(Debug, Clone, Default)]
pub struct LinExpr {
    /// Terms: (variable, coefficient)
    pub terms: SmallVec<[(VarId, Rational64); 4]>,
    /// Constant term
    pub constant: Rational64,
}
impl LinExpr {
    /// Create a new linear expression
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Create a constant expression
    #[must_use]
    pub fn constant(c: Rational64) -> Self {
        Self {
            terms: SmallVec::new(),
            constant: c,
        }
    }
    /// Create a variable expression
    #[must_use]
    pub fn var(v: VarId) -> Self {
        Self {
            terms: smallvec::smallvec![(v, Rational64::one())],
            constant: Rational64::zero(),
        }
    }
    /// Add a term
    pub fn add_term(&mut self, var: VarId, coef: Rational64) {
        if !coef.is_zero() {
            for (v, c) in &mut self.terms {
                if *v == var {
                    *c += coef;
                    if c.is_zero() {
                        self.terms.retain(|(v, _)| *v != var);
                    }
                    return;
                }
            }
            self.terms.push((var, coef));
        }
    }
    /// Add a constant
    pub fn add_constant(&mut self, c: Rational64) {
        self.constant += c;
    }
    /// Overflow-checked variant of [`Self::add_term`]: merges `coef` into
    /// the existing coefficient of `var` (or inserts a new term) exactly
    /// like `add_term`, but via `i64`-checked rational addition. Returns
    /// `false` (leaving `self` unmodified) if the merged coefficient would
    /// not fit back into a `Rational64`, instead of silently wrapping.
    #[must_use]
    fn try_add_term(&mut self, var: VarId, coef: Rational64) -> bool {
        if coef.is_zero() {
            return true;
        }
        for (v, c) in &mut self.terms {
            if *v == var {
                let Some(sum) = checked_add_r64(*c, coef) else {
                    return false;
                };
                *c = sum;
                if c.is_zero() {
                    self.terms.retain(|(v, _)| *v != var);
                }
                return true;
            }
        }
        self.terms.push((var, coef));
        true
    }
    /// Negate the expression
    pub fn negate(&mut self) {
        for (_, c) in &mut self.terms {
            *c = -*c;
        }
        self.constant = -self.constant;
    }
    /// Multiply by a constant
    pub fn scale(&mut self, factor: Rational64) {
        for (_, c) in &mut self.terms {
            *c *= factor;
        }
        self.constant *= factor;
    }
    /// Check if this expression subsumes another (i.e., this is weaker or equal)
    ///
    /// For example, x + y <= 10 subsumes x + y <= 5 (the latter is stronger)
    /// Returns true if adding the other constraint is redundant given this one
    #[must_use]
    pub fn subsumes(&self, other: &LinExpr, self_is_le: bool, other_is_le: bool) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }
        for (i, (v1, c1)) in self.terms.iter().enumerate() {
            if let Some((v2, c2)) = other.terms.get(i) {
                if v1 != v2 || c1 != c2 {
                    return false;
                }
            } else {
                return false;
            }
        }
        match (self_is_le, other_is_le) {
            (true, true) => self.constant >= other.constant,
            (false, false) => self.constant <= other.constant,
            _ => false,
        }
    }
}
/// Bound type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BoundType {
    /// No bound
    None,
    /// Lower bound (x >= b)
    Lower,
    /// Upper bound (x <= b)
    Upper,
    /// Equality (x = b)
    Equal,
}
/// A bound on a variable
#[derive(Debug, Clone)]
pub struct Bound {
    /// Bound type
    pub kind: BoundType,
    /// Bound value (supports strict bounds via delta)
    pub value: DeltaRational,
    /// Primary reason (assertion that caused this bound).
    pub reason: u32,
    /// Additional contributing reasons beyond `reason`. Populated when this
    /// bound was *derived* by propagation from several non-basic-variable
    /// bounds (see [`Simplex::propagate_bounds`] / [`Simplex::tighten_bounds`]):
    /// such a derived bound is implied by ALL of the bounds that fed the
    /// derivation, not just one. Conflict explanations
    /// ([`Simplex::explain_conflict`] and the bound-crossing check in
    /// [`Simplex::check`]) must emit `reason` together with every entry here,
    /// otherwise the Farkas/conflict clause is incomplete -- an unsound
    /// explanation that omits genuine antecedents.
    pub aux_reasons: SmallVec<[u32; 4]>,
}
impl Bound {
    /// Iterate over every reason (primary + auxiliary) backing this bound.
    fn all_reasons(&self) -> impl Iterator<Item = u32> + '_ {
        core::iter::once(self.reason).chain(self.aux_reasons.iter().copied())
    }
}
/// A propagated bound derived from constraint analysis
#[derive(Debug, Clone)]
pub struct PropagatedBound {
    /// The variable that got a new bound
    pub var: VarId,
    /// Whether it's a lower bound (true) or upper bound (false)
    pub is_lower: bool,
    /// The bound value
    pub value: DeltaRational,
    /// The reasons (assertion IDs) that imply this bound
    pub reasons: SmallVec<[u32; 4]>,
}
/// An undo entry for reverting a bound change
#[derive(Debug, Clone)]
enum BoundUndo {
    /// Lower bound was None, now has a value
    LowerWasNone(VarId),
    /// Lower bound was Some, save old value
    LowerWasSome(VarId, Bound),
    /// Upper bound was None, now has a value
    UpperWasNone(VarId),
    /// Upper bound was Some, save old value
    UpperWasSome(VarId, Bound),
    /// A new variable was added
    NewVar,
    /// A new slack variable was added
    NewSlack(VarId),
}
/// Simplex tableau state
#[derive(Debug)]
pub struct Simplex {
    /// Number of original variables
    num_vars: usize,
    /// Number of slack variables
    num_slack: usize,
    /// Current assignment (using delta-rationals for strict bounds)
    assignment: Vec<DeltaRational>,
    /// Lower bounds
    lower: Vec<Option<Bound>>,
    /// Upper bounds
    upper: Vec<Option<Bound>>,
    /// Tableau rows: basic variable -> linear combination of non-basic
    tableau: FxHashMap<VarId, LinExpr>,
    /// Basic variables
    basic: Vec<bool>,
    /// Infeasible basic variable (if any)
    infeasible: Option<VarId>,
    /// Pending propagated bounds
    propagated: Vec<PropagatedBound>,
    /// Trail of undo operations
    trail: Vec<BoundUndo>,
    /// Trail size at each decision level
    trail_limits: Vec<usize>,
    /// Cached assignments for warm-starting (basis caching)
    /// Saves assignment state at each decision level for faster incremental solving
    cached_assignments: Vec<Vec<DeltaRational>>,
    /// Saved tableau snapshots for correct restoration on pop.
    /// Pivoting during check() modifies the tableau rows in-place; without saving
    /// the full tableau at push time, pop() cannot restore the correct basis.
    saved_tableaux: Vec<(FxHashMap<VarId, LinExpr>, Vec<bool>)>,
    /// Pivoting rule to use
    pivoting_rule: PivotingRule,
    /// Maximum number of pivot operations before giving up
    max_pivots: usize,
    /// Set to `true` when the most recent `check()`/`dual_simplex()` aborted
    /// because it hit `max_pivots` without proving feasibility or infeasibility.
    ///
    /// When this flag is set, an `Ok(())` result from `check()` MUST NOT be
    /// interpreted as "satisfiable" — the LP state is unresolved (an incomplete
    /// resource-limited run), and callers deciding satisfiability have to report
    /// `Unknown` rather than `Sat`.  See [`Simplex::resource_limit_reached`].
    resource_limit: bool,
}
impl Default for Simplex {
    fn default() -> Self {
        Self::new()
    }
}
impl Simplex {
    /// Create a new Simplex instance
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SimplexConfig::default())
    }
    /// Create a new Simplex instance with custom configuration
    #[must_use]
    pub fn with_config(config: SimplexConfig) -> Self {
        Self {
            num_vars: 0,
            num_slack: 0,
            assignment: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
            tableau: FxHashMap::default(),
            basic: Vec::new(),
            infeasible: None,
            propagated: Vec::new(),
            trail: Vec::new(),
            trail_limits: vec![0],
            cached_assignments: Vec::new(),
            saved_tableaux: Vec::new(),
            pivoting_rule: config.pivoting_rule,
            max_pivots: config.max_pivots,
            resource_limit: false,
        }
    }
    /// Whether the most recent feasibility run (`check` / `dual_simplex`) gave up
    /// after exhausting the pivot budget without a definitive answer.
    ///
    /// If this returns `true`, the last `Ok(())` is a *resource limit*, not a
    /// proof of feasibility, and any satisfiability decision built on top of the
    /// simplex must be reported as `Unknown`.
    #[inline]
    #[must_use]
    pub fn resource_limit_reached(&self) -> bool {
        self.resource_limit
    }
    /// Set the pivoting rule
    pub fn set_pivoting_rule(&mut self, rule: PivotingRule) {
        self.pivoting_rule = rule;
    }
    /// Get the current pivoting rule
    #[must_use]
    pub fn pivoting_rule(&self) -> PivotingRule {
        self.pivoting_rule
    }
    /// Grow every per-variable parallel array by exactly one slot, in
    /// lockstep, and return the new (non-basic) variable's id.
    ///
    /// This is the *single* choke point through which `assignment`, `lower`,
    /// `upper` and `basic` gain a slot for an ordinary variable, so the four
    /// arrays can never drift out of length relative to one another. A
    /// matching `NewVar` undo record is pushed so that [`Self::pop`] shrinks
    /// all four together.
    fn register_var(&mut self) -> VarId {
        let id = self.assignment.len() as VarId;
        self.num_vars += 1;
        self.assignment.push(DeltaRational::zero());
        self.lower.push(None);
        self.upper.push(None);
        self.basic.push(false);
        self.trail.push(BoundUndo::NewVar);
        id
    }
    /// Ensure every per-variable array covers index `idx`, materializing any
    /// missing slots (contiguously, including gaps) as fresh unconstrained,
    /// non-basic variables via [`Self::register_var`].
    ///
    /// Every code path that can hand a variable index to the tableau or the
    /// bounds arrays routes through this, so a variable index the caller
    /// cached and replayed across a backtrack (which shrank the arrays) — or
    /// any other stale/out-of-range index — can never index past the parallel
    /// arrays and panic. The replayed index is simply reinstated as a fresh
    /// variable, and the `NewVar` undo records pushed here keep `pop` correct.
    fn ensure_var(&mut self, idx: usize) {
        while self.assignment.len() <= idx {
            let _ = self.register_var();
        }
    }
    /// Add a new variable
    pub fn new_var(&mut self) -> VarId {
        self.register_var()
    }
    /// Add a slack variable for a constraint
    fn new_slack(&mut self) -> VarId {
        let id = self.assignment.len() as VarId;
        self.num_slack += 1;
        self.assignment.push(DeltaRational::zero());
        self.lower.push(None);
        self.upper.push(None);
        self.basic.push(true);
        self.trail.push(BoundUndo::NewSlack(id));
        id
    }
    /// Get the current value of a variable (returns the real part)
    #[inline]
    #[must_use]
    pub fn value(&self, var: VarId) -> Rational64 {
        self.assignment
            .get(var as usize)
            .map(|d| d.real)
            .unwrap_or_default()
    }
    /// Get the current delta-rational value of a variable
    #[inline]
    #[must_use]
    pub fn delta_value(&self, var: VarId) -> DeltaRational {
        self.assignment
            .get(var as usize)
            .copied()
            .unwrap_or_default()
    }
    /// Concrete positive rational to substitute for the infinitesimal `δ` when
    /// turning the delta-rational assignment into an ordinary rational model.
    ///
    /// A strict bound such as `x > 0` is stored as the delta-rational lower
    /// bound `(0, 1)` and the assignment then sits at `0 + δ`.  Reading back
    /// only the real part reports `x = 0`, which *violates* the very constraint
    /// that produced it.  The fix is the standard δ-instantiation of
    /// Dutertre & de Moura's "Simplex for DPLL(T)": pick the largest `δ₀ ∈ (0,1]`
    /// for which every bound still holds after substituting `δ := δ₀`.
    ///
    /// Each bound contributes a constraint of the form `dr + dd·δ ≥ 0` where
    /// `dr`/`dd` are the real/delta gaps between the assignment and the bound.
    /// Only `dd < 0` can be violated by a large δ, and feasibility of the
    /// delta-rational assignment guarantees `dr > 0` in that case, so the
    /// binding limit is `δ ≤ dr / (-dd)`.  Tableau rows are linear in δ and are
    /// preserved by any substitution, so bounds are the only source of
    /// constraints.
    ///
    /// Reference: Z3's `lp::lar_solver::get_model` delta adjustment.
    #[must_use]
    pub fn delta_instantiation(&self) -> Rational64 {
        // Smallest representable positive rational, used as a conservative
        // fallback when an exact ratio overflows `Rational64`.
        let tiny = Rational64::new(1, i64::MAX);
        let mut delta = Rational64::one();
        let mut tighten = |dr: Rational64, dd: Rational64| {
            // Constraint `dr + dd·δ >= 0`.  Non-negative `dd` can never be
            // violated by a positive δ, and a non-positive `dr` means the
            // delta-rational assignment already violates this bound (the state
            // is infeasible) — nothing to instantiate.
            if !dd.is_negative() || !dr.is_positive() {
                return;
            }
            let limit = checked_neg_r64(dd).and_then(|neg_dd| checked_div_r64(dr, neg_dd));
            match limit {
                Some(cand) => {
                    if cand < delta {
                        delta = cand;
                    }
                }
                // Ratio not representable: clamp to the smallest positive value
                // rather than risk keeping a δ that breaks the bound.
                None => {
                    if tiny < delta {
                        delta = tiny;
                    }
                }
            }
        };
        for (idx, assigned) in self.assignment.iter().enumerate() {
            if let Some(bound) = self.lower.get(idx).and_then(Option::as_ref) {
                // assignment >= lower  =>  (a.real - l.real) + (a.delta - l.delta)·δ >= 0
                if let (Some(dr), Some(dd)) = (
                    checked_neg_r64(bound.value.real)
                        .and_then(|n| checked_add_r64(assigned.real, n)),
                    checked_neg_r64(bound.value.delta)
                        .and_then(|n| checked_add_r64(assigned.delta, n)),
                ) {
                    tighten(dr, dd);
                }
            }
            if let Some(bound) = self.upper.get(idx).and_then(Option::as_ref) {
                // assignment <= upper  =>  (u.real - a.real) + (u.delta - a.delta)·δ >= 0
                if let (Some(dr), Some(dd)) = (
                    checked_neg_r64(assigned.real)
                        .and_then(|n| checked_add_r64(bound.value.real, n)),
                    checked_neg_r64(assigned.delta)
                        .and_then(|n| checked_add_r64(bound.value.delta, n)),
                ) {
                    tighten(dr, dd);
                }
            }
        }
        delta
    }
    /// Set a lower bound (x >= value)
    pub fn set_lower(&mut self, var: VarId, value: Rational64, reason: u32) {
        let idx = var as usize;
        self.ensure_var(idx);
        match &self.lower[idx] {
            None => self.trail.push(BoundUndo::LowerWasNone(var)),
            Some(old) => {
                let old = old.clone();
                self.trail.push(BoundUndo::LowerWasSome(var, old));
            }
        }
        self.lower[idx] = Some(Bound {
            kind: BoundType::Lower,
            value: DeltaRational::from_rational(value),
            reason,
            aux_reasons: SmallVec::new(),
        });
    }
    /// Set a lower bound directly from a `DeltaRational` (supports strict
    /// bounds carrying an infinitesimal `δ` component), pushing an undo
    /// record onto `self.trail` exactly like [`Self::set_lower`]. Used by
    /// [`Self::propagate_bounds`], whose derived bound values are already
    /// `DeltaRational` (propagation chains through strict inequalities).
    ///
    /// Takes the FULL set of contributing reasons: the first becomes the
    /// bound's primary `reason`, the remainder its `aux_reasons`, so that a
    /// propagated bound records every antecedent for later conflict
    /// explanation (see [`Bound::aux_reasons`]).
    fn set_lower_delta(&mut self, var: VarId, value: DeltaRational, reasons: SmallVec<[u32; 4]>) {
        let idx = var as usize;
        let Some((reason, aux_reasons)) = split_reasons(reasons) else {
            return;
        };
        self.ensure_var(idx);
        match &self.lower[idx] {
            None => self.trail.push(BoundUndo::LowerWasNone(var)),
            Some(old) => {
                let old = old.clone();
                self.trail.push(BoundUndo::LowerWasSome(var, old));
            }
        }
        self.lower[idx] = Some(Bound {
            kind: BoundType::Lower,
            value,
            reason,
            aux_reasons,
        });
    }
    /// Set an upper bound directly from a `DeltaRational`; see
    /// [`Self::set_lower_delta`].
    fn set_upper_delta(&mut self, var: VarId, value: DeltaRational, reasons: SmallVec<[u32; 4]>) {
        let idx = var as usize;
        let Some((reason, aux_reasons)) = split_reasons(reasons) else {
            return;
        };
        self.ensure_var(idx);
        match &self.upper[idx] {
            None => self.trail.push(BoundUndo::UpperWasNone(var)),
            Some(old) => {
                let old = old.clone();
                self.trail.push(BoundUndo::UpperWasSome(var, old));
            }
        }
        self.upper[idx] = Some(Bound {
            kind: BoundType::Upper,
            value,
            reason,
            aux_reasons,
        });
    }
    /// Set a strict lower bound (x > value), represented as x >= value + δ
    pub fn set_strict_lower(&mut self, var: VarId, value: Rational64, reason: u32) {
        let idx = var as usize;
        self.ensure_var(idx);
        match &self.lower[idx] {
            None => self.trail.push(BoundUndo::LowerWasNone(var)),
            Some(old) => {
                let old = old.clone();
                self.trail.push(BoundUndo::LowerWasSome(var, old));
            }
        }
        self.lower[idx] = Some(Bound {
            kind: BoundType::Lower,
            value: DeltaRational::new(value, Rational64::one()),
            reason,
            aux_reasons: SmallVec::new(),
        });
    }
    /// Set an upper bound (x <= value)
    pub fn set_upper(&mut self, var: VarId, value: Rational64, reason: u32) {
        let idx = var as usize;
        self.ensure_var(idx);
        match &self.upper[idx] {
            None => self.trail.push(BoundUndo::UpperWasNone(var)),
            Some(old) => {
                let old = old.clone();
                self.trail.push(BoundUndo::UpperWasSome(var, old));
            }
        }
        self.upper[idx] = Some(Bound {
            kind: BoundType::Upper,
            value: DeltaRational::from_rational(value),
            reason,
            aux_reasons: SmallVec::new(),
        });
    }
    /// Set a strict upper bound (x < value), represented as x <= value - δ
    pub fn set_strict_upper(&mut self, var: VarId, value: Rational64, reason: u32) {
        let idx = var as usize;
        self.ensure_var(idx);
        match &self.upper[idx] {
            None => self.trail.push(BoundUndo::UpperWasNone(var)),
            Some(old) => {
                let old = old.clone();
                self.trail.push(BoundUndo::UpperWasSome(var, old));
            }
        }
        self.upper[idx] = Some(Bound {
            kind: BoundType::Upper,
            value: DeltaRational::new(value, -Rational64::one()),
            reason,
            aux_reasons: SmallVec::new(),
        });
    }
    /// Add a constraint: expr <= 0
    pub fn add_le(&mut self, mut expr: LinExpr, reason: u32) {
        let mut substituted_expr = LinExpr::constant(expr.constant);
        for (var, coef) in &expr.terms {
            if let Some(basic_expr) = self.tableau.get(var).cloned() {
                substituted_expr.add_constant(coef * basic_expr.constant);
                for (inner_var, inner_coef) in &basic_expr.terms {
                    substituted_expr.add_term(*inner_var, coef * inner_coef);
                }
            } else {
                substituted_expr.add_term(*var, *coef);
            }
        }
        expr = substituted_expr;
        // Register every variable the (substituted) expression references
        // BEFORE allocating the slack, so (a) no tableau row can reference an
        // index past the bounds arrays and (b) the slack's id is guaranteed
        // fresh rather than colliding with an as-yet-unregistered variable.
        if let Some(max_var) = expr.terms.iter().map(|(v, _)| *v).max() {
            self.ensure_var(max_var as usize);
        }
        let slack = self.new_slack();
        expr.add_term(slack, Rational64::one());
        let mut slack_expr = LinExpr::constant(-expr.constant);
        for (var, coef) in &expr.terms {
            if *var != slack {
                slack_expr.add_term(*var, -*coef);
            }
        }
        self.tableau.insert(slack, slack_expr);
        if slack as usize >= self.basic.len() {
            self.basic.resize(slack as usize + 1, false);
        }
        self.basic[slack as usize] = true;
        self.set_lower(slack, Rational64::zero(), reason);
    }
    /// Add a constraint: expr >= 0
    pub fn add_ge(&mut self, mut expr: LinExpr, reason: u32) {
        expr.negate();
        self.add_le(expr, reason);
    }
    /// Add a constraint: expr = 0
    pub fn add_eq(&mut self, expr: LinExpr, reason: u32) {
        self.add_le(expr.clone(), reason);
        self.add_ge(expr, reason);
    }
    /// Add a strict constraint: expr < 0
    /// Uses infinitesimals: expr + s = 0 with s > 0
    pub fn add_strict_lt(&mut self, mut expr: LinExpr, reason: u32) {
        let mut substituted_expr = LinExpr::constant(expr.constant);
        for (var, coef) in &expr.terms {
            if let Some(basic_expr) = self.tableau.get(var).cloned() {
                substituted_expr.add_constant(coef * basic_expr.constant);
                for (inner_var, inner_coef) in &basic_expr.terms {
                    substituted_expr.add_term(*inner_var, coef * inner_coef);
                }
            } else {
                substituted_expr.add_term(*var, *coef);
            }
        }
        expr = substituted_expr;
        // See `add_le`: register the expression's variables before allocating
        // the slack so no tableau row can reference an index past the bounds
        // arrays and the slack id cannot collide with an unregistered variable.
        if let Some(max_var) = expr.terms.iter().map(|(v, _)| *v).max() {
            self.ensure_var(max_var as usize);
        }
        let slack = self.new_slack();
        expr.add_term(slack, Rational64::one());
        let mut slack_expr = LinExpr::constant(-expr.constant);
        for (var, coef) in &expr.terms {
            if *var != slack {
                slack_expr.add_term(*var, -*coef);
            }
        }
        self.tableau.insert(slack, slack_expr);
        self.set_strict_lower(slack, Rational64::zero(), reason);
    }
    /// Add a strict constraint: expr > 0
    /// Uses infinitesimals: -expr < 0
    pub fn add_strict_gt(&mut self, mut expr: LinExpr, reason: u32) {
        expr.negate();
        self.add_strict_lt(expr, reason);
    }
    /// Check if bounds are consistent
    pub fn check(&mut self) -> Result<(), Vec<u32>> {
        self.resource_limit = false;
        for i in 0..self.assignment.len() {
            if let (Some(lo), Some(hi)) = (&self.lower[i], &self.upper[i])
                && lo.value > hi.value
            {
                // Emit ALL antecedents of both crossing bounds, not just their
                // primary reasons: a propagated bound is implied by every
                // reason that fed its derivation, and dropping them yields an
                // incomplete (unsound) conflict explanation.
                let mut conflict: Vec<u32> = Vec::new();
                for r in lo.all_reasons().chain(hi.all_reasons()) {
                    if !conflict.contains(&r) {
                        conflict.push(r);
                    }
                }
                return Err(conflict);
            }
        }
        self.crash_basis();
        self.make_feasible()
    }
    /// Crash basis initialization for faster convergence
    ///
    /// This heuristic initializes the basis to a "good" starting point instead of
    /// starting with all slack variables. It assigns variables to their bounds
    /// based on a heuristic that tries to minimize infeasibilities.
    ///
    /// Benefits:
    /// - Reduces number of pivots needed in Phase I
    /// - Speeds up incremental solving
    /// - Particularly effective when many variables have tight bounds
    ///
    /// Reference: Koberstein's crash procedure for MIP solvers
    fn crash_basis(&mut self) {
        for i in 0..self.assignment.len() {
            if i < self.basic.len() && self.basic[i] {
                continue;
            }
            if let Some(lo) = &self.lower[i] {
                self.assignment[i] = lo.value;
            } else if let Some(hi) = &self.upper[i] {
                self.assignment[i] = hi.value;
            } else {
                self.assignment[i] = DeltaRational::zero();
            }
        }
        self.update_assignment();
    }
    /// Pivot to make the solution feasible
    ///
    /// Precondition: the assignment is already consistent with the current
    /// basis and bounds. `make_feasible` is private with its only caller
    /// being [`Self::check`], which always runs [`Self::crash_basis`] (whose
    /// last step is `update_assignment`) immediately before — so
    /// recomputing the assignment again here was a redundant full pass on
    /// every theory check.
    fn make_feasible(&mut self) -> Result<(), Vec<u32>> {
        for _ in 0..self.max_pivots {
            let violating = self.find_violating();
            if violating.is_none() {
                return Ok(());
            }
            let (basic_var, bound) =
                violating.expect("violating basic variable must exist after is_none check");
            let pivot_col = self.find_pivot_col(basic_var, &bound);
            match pivot_col {
                Some(nonbasic_var) => {
                    if !self.pivot(basic_var, nonbasic_var) {
                        return Ok(());
                    }
                }
                None => {
                    return Err(self.explain_conflict(basic_var, &bound));
                }
            }
        }
        self.resource_limit = true;
        Ok(())
    }
    /// Dual Simplex: Restore primal feasibility while maintaining dual feasibility
    ///
    /// The dual simplex algorithm is particularly efficient when:
    /// - After adding cuts in branch-and-bound (cuts make primal infeasible but dual stays feasible)
    /// - When resolving from a previously optimal basis after bound changes
    /// - For incremental solving where the problem structure changes slightly
    ///
    /// Unlike primal simplex which maintains primal feasibility and seeks optimality,
    /// dual simplex maintains dual feasibility (optimal reduced costs) and seeks primal feasibility.
    ///
    /// This is often faster than primal simplex after adding cutting planes because:
    /// - The dual remains feasible after most cuts
    /// - Only a few pivots are needed to restore primal feasibility
    /// - Warm-starting from the previous optimal basis is very effective
    ///
    /// Reference:
    /// - Dantzig, "Linear Programming and Extensions" (1963), Chapter 7
    /// - Bixby, "Implementing the Simplex Method" (2002)
    /// - Modern MIP solvers (CPLEX, Gurobi) use dual simplex as the primary LP solver
    pub fn dual_simplex(&mut self) -> Result<(), Vec<u32>> {
        self.resource_limit = false;
        self.update_assignment();
        for _ in 0..self.max_pivots {
            let violating = self.find_violating();
            if violating.is_none() {
                return Ok(());
            }
            let (leaving_var, bound) =
                violating.expect("violating basic variable must exist after is_none check");
            let entering = self.find_dual_pivot_col(leaving_var, &bound);
            match entering {
                Some(entering_var) => {
                    if !self.pivot(leaving_var, entering_var) {
                        return Ok(());
                    }
                }
                None => {
                    return Err(self.explain_conflict(leaving_var, &bound));
                }
            }
        }
        self.resource_limit = true;
        Ok(())
    }
    /// Find entering variable for dual simplex (maintains dual feasibility)
    ///
    /// Given a leaving variable (basic var violating bounds), find a non-basic variable
    /// to enter the basis such that:
    /// 1. The pivot reduces the bound violation
    /// 2. Dual feasibility is maintained (reduced costs stay optimal)
    ///
    /// For leaving variable x_i with row: x_i = c + sum(a_j * x_j)
    ///
    /// If x_i < lower_i (too small):
    /// - Need to increase x_i
    /// - Choose x_j with a_j > 0 (increases x_i) and can increase
    /// - Or x_j with a_j < 0 (decreases moves x_i up) and can decrease
    ///
    /// If x_i > upper_i (too large):
    /// - Need to decrease x_i
    /// - Choose x_j with a_j < 0 (increases x_j decreases x_i) and can increase
    /// - Or x_j with a_j > 0 (decreases x_j decreases x_i) and can decrease
    ///
    /// Among eligible variables, choose the one that maintains dual feasibility.
    /// This typically means choosing the variable with the smallest ratio of:
    /// (change in objective) / (change in constraint violation)
    ///
    /// For now, we use a simple rule: choose the first eligible variable (Bland's rule for dual)
    #[allow(dead_code)]
    fn find_dual_pivot_col(&self, leaving_var: VarId, bound: &Bound) -> Option<VarId> {
        let expr = self.tableau.get(&leaving_var)?;
        let mut best_var = None;
        for (var, coef) in &expr.terms {
            let can_increase = self.can_increase(*var);
            let can_decrease = self.can_decrease(*var);
            let is_eligible = match bound.kind {
                BoundType::Lower => {
                    (*coef > Rational64::zero() && can_increase)
                        || (*coef < Rational64::zero() && can_decrease)
                }
                BoundType::Upper => {
                    (*coef < Rational64::zero() && can_increase)
                        || (*coef > Rational64::zero() && can_decrease)
                }
                _ => false,
            };
            if is_eligible {
                best_var = match best_var {
                    None => Some(*var),
                    Some(current) if *var < current => Some(*var),
                    Some(current) => Some(current),
                };
            }
        }
        best_var
    }
    /// Find a basic variable that violates its bounds
    fn find_violating(&self) -> Option<(VarId, Bound)> {
        for var in self.tableau.keys() {
            let idx = *var as usize;
            let val = self.assignment[idx];
            if let Some(lo) = &self.lower[idx]
                && val < lo.value
            {
                return Some((*var, lo.clone()));
            }
            if let Some(hi) = &self.upper[idx]
                && val > hi.value
            {
                return Some((*var, hi.clone()));
            }
        }
        None
    }
    /// Find a non-basic variable to pivot with using the configured pivoting rule
    fn find_pivot_col(&self, basic_var: VarId, bound: &Bound) -> Option<VarId> {
        let expr = self.tableau.get(&basic_var)?;
        match self.pivoting_rule {
            PivotingRule::Bland => {
                let mut best_var = None;
                for (var, coef) in &expr.terms {
                    let can_increase = self.can_increase(*var);
                    let can_decrease = self.can_decrease(*var);
                    let is_eligible = match bound.kind {
                        BoundType::Lower => {
                            (*coef > Rational64::zero() && can_increase)
                                || (*coef < Rational64::zero() && can_decrease)
                        }
                        BoundType::Upper => {
                            (*coef < Rational64::zero() && can_increase)
                                || (*coef > Rational64::zero() && can_decrease)
                        }
                        _ => false,
                    };
                    if is_eligible {
                        best_var = match best_var {
                            None => Some(*var),
                            Some(current) if *var < current => Some(*var),
                            Some(current) => Some(current),
                        };
                    }
                }
                best_var
            }
            PivotingRule::Dantzig => {
                let mut best_var = None;
                let mut best_improvement = Rational64::zero();
                for (var, coef) in &expr.terms {
                    let can_increase = self.can_increase(*var);
                    let can_decrease = self.can_decrease(*var);
                    let improvement = match bound.kind {
                        BoundType::Lower if *coef > Rational64::zero() && can_increase => {
                            coef.abs()
                        }
                        BoundType::Lower if *coef < Rational64::zero() && can_decrease => {
                            coef.abs()
                        }
                        BoundType::Upper if *coef < Rational64::zero() && can_increase => {
                            coef.abs()
                        }
                        BoundType::Upper if *coef > Rational64::zero() && can_decrease => {
                            coef.abs()
                        }
                        _ => Rational64::zero(),
                    };
                    if improvement > best_improvement {
                        best_improvement = improvement;
                        best_var = Some(*var);
                    }
                }
                best_var
            }
            PivotingRule::SteepestEdge => {
                let mut best_var = None;
                let mut best_score = Rational64::zero();
                for (var, coef) in &expr.terms {
                    let can_increase = self.can_increase(*var);
                    let can_decrease = self.can_decrease(*var);
                    let score = match bound.kind {
                        BoundType::Lower if *coef > Rational64::zero() && can_increase => {
                            coef.abs()
                        }
                        BoundType::Lower if *coef < Rational64::zero() && can_decrease => {
                            coef.abs()
                        }
                        BoundType::Upper if *coef < Rational64::zero() && can_increase => {
                            coef.abs()
                        }
                        BoundType::Upper if *coef > Rational64::zero() && can_decrease => {
                            coef.abs()
                        }
                        _ => Rational64::zero(),
                    };
                    if score > best_score {
                        best_score = score;
                        best_var = Some(*var);
                    }
                }
                best_var
            }
            PivotingRule::PartialPricing => {
                const SAMPLE_RATE: usize = 4;
                let mut best_var = None;
                let mut best_improvement = Rational64::zero();
                let mut count = 0;
                for (var, coef) in &expr.terms {
                    count += 1;
                    if count % SAMPLE_RATE != 0 {
                        continue;
                    }
                    let can_increase = self.can_increase(*var);
                    let can_decrease = self.can_decrease(*var);
                    let improvement = match bound.kind {
                        BoundType::Lower if *coef > Rational64::zero() && can_increase => {
                            coef.abs()
                        }
                        BoundType::Lower if *coef < Rational64::zero() && can_decrease => {
                            coef.abs()
                        }
                        BoundType::Upper if *coef < Rational64::zero() && can_increase => {
                            coef.abs()
                        }
                        BoundType::Upper if *coef > Rational64::zero() && can_decrease => {
                            coef.abs()
                        }
                        _ => Rational64::zero(),
                    };
                    if improvement > best_improvement {
                        best_improvement = improvement;
                        best_var = Some(*var);
                    }
                }
                if best_var.is_none() {
                    for (var, coef) in &expr.terms {
                        let can_increase = self.can_increase(*var);
                        let can_decrease = self.can_decrease(*var);
                        let is_eligible = match bound.kind {
                            BoundType::Lower => {
                                (*coef > Rational64::zero() && can_increase)
                                    || (*coef < Rational64::zero() && can_decrease)
                            }
                            BoundType::Upper => {
                                (*coef < Rational64::zero() && can_increase)
                                    || (*coef > Rational64::zero() && can_decrease)
                            }
                            _ => false,
                        };
                        if is_eligible {
                            return Some(*var);
                        }
                    }
                }
                best_var
            }
        }
    }
    /// Check if a variable can be increased
    #[inline]
    pub(super) fn can_increase(&self, var: VarId) -> bool {
        let idx = var as usize;
        match &self.upper[idx] {
            Some(hi) => self.assignment[idx] < hi.value,
            None => true,
        }
    }
    /// Check if a variable can be decreased
    #[inline]
    pub(super) fn can_decrease(&self, var: VarId) -> bool {
        let idx = var as usize;
        match &self.lower[idx] {
            Some(lo) => self.assignment[idx] > lo.value,
            None => true,
        }
    }
    /// Perform a pivot operation.
    ///
    /// `Rational64` is `i64`-backed: repeated pivoting can grow numerators
    /// and denominators without bound (the classic fraction-free-elimination
    /// blowup), and `num-rational`'s arithmetic operators do not check for
    /// overflow -- they panic in debug builds and silently wrap to a wrong
    /// coefficient in release builds. To avoid both, every coefficient
    /// computed here goes through the `checked_*_r64` helpers, and the pivot
    /// is fully validated (via a `i128`-checked dry run) BEFORE any tableau
    /// state is mutated: an overflow anywhere aborts the pivot with no
    /// partial mutation, matching the pre-existing `resource_limit` "give up
    /// honestly" contract used for pivot-budget exhaustion. Returns `false`
    /// iff the pivot could not be completed (overflow, or a broken tableau
    /// invariant), in which case `resource_limit` is set so callers report
    /// `Unknown` rather than trusting a fabricated/partial result.
    ///
    /// Not `#[must_use]`: `simplex_opt.rs`'s optimization-direction pivot
    /// loop currently ignores the outcome (pre-existing behavior, out of
    /// this module's scope to change) and relies on the subsequent
    /// pivot-budget/optimality bookkeeping to notice a stalled search.
    pub(super) fn pivot(&mut self, basic_var: VarId, nonbasic_var: VarId) -> bool {
        #[cfg(feature = "profiling")]
        let _timer = ScopedTimer::new(ProfilingCategory::SimplexPivot);
        let Some(expr) = self.tableau.get(&basic_var) else {
            self.resource_limit = true;
            return false;
        };
        let Some(coef) = expr
            .terms
            .iter()
            .find(|(v, _)| *v == nonbasic_var)
            .map(|(_, c)| *c)
        else {
            self.resource_limit = true;
            return false;
        };
        let Some(inv_coef) = checked_recip_r64(coef) else {
            self.resource_limit = true;
            return false;
        };
        let Some(new_constant) =
            checked_neg_r64(expr.constant).and_then(|n| checked_div_r64(n, coef))
        else {
            self.resource_limit = true;
            return false;
        };
        let mut new_expr = LinExpr::new();
        new_expr.terms.push((basic_var, inv_coef));
        new_expr.constant = new_constant;
        for (var, c) in &expr.terms {
            if *var != nonbasic_var {
                let Some(neg_c) = checked_neg_r64(*c) else {
                    self.resource_limit = true;
                    return false;
                };
                let Some(val) = checked_div_r64(neg_c, coef) else {
                    self.resource_limit = true;
                    return false;
                };
                if !new_expr.try_add_term(*var, val) {
                    self.resource_limit = true;
                    return false;
                }
            }
        }
        let mut row_updates: Vec<(VarId, LinExpr)> = Vec::new();
        for (var, row) in &self.tableau {
            if *var == basic_var {
                continue;
            }
            let sub_coef = row
                .terms
                .iter()
                .find(|(v, _)| *v == nonbasic_var)
                .map(|(_, c)| *c);
            let Some(sc) = sub_coef else { continue };
            let mut new_row = row.clone();
            new_row.terms.retain(|(v, _)| *v != nonbasic_var);
            let Some(delta_c) = checked_mul_r64(sc, new_expr.constant) else {
                self.resource_limit = true;
                return false;
            };
            let Some(sum) = checked_add_r64(new_row.constant, delta_c) else {
                self.resource_limit = true;
                return false;
            };
            new_row.constant = sum;
            for (v, c) in &new_expr.terms {
                let Some(term_c) = checked_mul_r64(sc, *c) else {
                    self.resource_limit = true;
                    return false;
                };
                if !new_row.try_add_term(*v, term_c) {
                    self.resource_limit = true;
                    return false;
                }
            }
            row_updates.push((*var, new_row));
        }
        self.tableau.remove(&basic_var);
        for (var, new_row) in row_updates {
            self.tableau.insert(var, new_row);
        }
        self.tableau.insert(nonbasic_var, new_expr);
        self.basic[basic_var as usize] = false;
        self.basic[nonbasic_var as usize] = true;
        self.update_assignment();
        true
    }
    /// Update variable assignments after pivot
    pub(super) fn update_assignment(&mut self) {
        let num_vars = self.assignment.len();
        for i in 0..num_vars {
            if !self.basic[i] {
                if let Some(lo) = &self.lower[i] {
                    self.assignment[i] = lo.value;
                } else if let Some(hi) = &self.upper[i] {
                    self.assignment[i] = hi.value;
                }
            }
        }
        for (var, expr) in &self.tableau {
            let var_idx = *var as usize;
            if var_idx >= num_vars {
                continue;
            }
            let mut val = DeltaRational::from_rational(expr.constant);
            let mut has_stale_ref = false;
            for (v, c) in &expr.terms {
                let v_idx = *v as usize;
                if v_idx >= num_vars {
                    has_stale_ref = true;
                    break;
                }
                val += self.assignment[v_idx] * *c;
            }
            if !has_stale_ref {
                self.assignment[var_idx] = val;
            }
        }
    }
    /// Explain why a conflict occurred using Farkas lemma
    ///
    /// When a basic variable x_i violates its bounds and no pivot is possible,
    /// we can derive a conflict clause from the bounds of all involved variables.
    ///
    /// For x_i = c + sum(a_j * x_j):
    /// - If x_i < lower(x_i), we need to explain why x_i can't reach its lower bound
    /// - If x_i > upper(x_i), we need to explain why x_i can't decrease to its upper bound
    ///
    /// The conflict clause contains the reasons for all the bounds that prevent a pivot.
    fn explain_conflict(&self, basic_var: VarId, bound: &Bound) -> Vec<u32> {
        let mut reasons: Vec<u32> = Vec::new();
        // Every antecedent of the violated bound (primary + auxiliary), so a
        // propagated bound contributes all of the reasons that derived it.
        let push_all = |b: &Bound, reasons: &mut Vec<u32>| {
            for r in b.all_reasons() {
                if !reasons.contains(&r) {
                    reasons.push(r);
                }
            }
        };
        push_all(bound, &mut reasons);
        let expr = match self.tableau.get(&basic_var) {
            Some(e) => e,
            None => return reasons,
        };
        for (var, coef) in &expr.terms {
            let var_idx = *var as usize;
            match bound.kind {
                BoundType::Lower => {
                    if *coef > Rational64::zero()
                        && let Some(hi) = &self.upper[var_idx]
                    {
                        push_all(hi, &mut reasons);
                    } else if *coef < Rational64::zero()
                        && let Some(lo) = &self.lower[var_idx]
                    {
                        push_all(lo, &mut reasons);
                    }
                }
                BoundType::Upper => {
                    if *coef > Rational64::zero()
                        && let Some(lo) = &self.lower[var_idx]
                    {
                        push_all(lo, &mut reasons);
                    } else if *coef < Rational64::zero()
                        && let Some(hi) = &self.upper[var_idx]
                    {
                        push_all(hi, &mut reasons);
                    }
                }
                _ => {}
            }
        }
        reasons
    }
    /// Perform bound propagation through the tableau
    ///
    /// For each basic variable x_i = c + sum(a_j * x_j), we can derive bounds:
    /// - If all x_j have bounds, we can compute bounds for x_i
    /// - If x_i has a bound, we may derive bounds for x_j
    pub fn propagate_bounds(&mut self) {
        self.propagated.clear();
        for (basic_var, expr) in &self.tableau {
            if let Some(bound) = self.derive_basic_bound(*basic_var, expr) {
                self.propagated.push(bound);
            }
        }
        let props = self.propagated.clone();
        for prop in &props {
            let idx = prop.var as usize;
            if idx >= self.lower.len() {
                continue;
            }
            if prop.reasons.is_empty() {
                continue;
            }
            if prop.is_lower {
                let should_update = match &self.lower[idx] {
                    None => true,
                    Some(existing) => prop.value > existing.value,
                };
                if should_update {
                    self.set_lower_delta(prop.var, prop.value, prop.reasons.clone());
                }
            } else {
                let should_update = match &self.upper[idx] {
                    None => true,
                    Some(existing) => prop.value < existing.value,
                };
                if should_update {
                    self.set_upper_delta(prop.var, prop.value, prop.reasons.clone());
                }
            }
        }
    }
    /// Derive bounds for a basic variable from bounds on non-basic variables
    ///
    /// For basic variable x_i = c + sum(a_j * x_j):
    /// - Lower bound: sum of (a_j * lower(x_j) if a_j > 0, a_j * upper(x_j) if a_j < 0)
    /// - Upper bound: sum of (a_j * upper(x_j) if a_j > 0, a_j * lower(x_j) if a_j < 0)
    fn derive_basic_bound(&self, basic_var: VarId, expr: &LinExpr) -> Option<PropagatedBound> {
        let idx = basic_var as usize;
        let mut lower_sum = DeltaRational::from_rational(expr.constant);
        let mut lower_reasons: SmallVec<[u32; 4]> = SmallVec::new();
        let mut can_derive_lower = true;
        for (var, coef) in &expr.terms {
            let var_idx = *var as usize;
            if *coef > Rational64::zero() {
                if let Some(lo) = &self.lower[var_idx] {
                    lower_sum += lo.value * *coef;
                    // Carry EVERY antecedent of this bound (primary + auxiliary),
                    // not just its primary reason: when `lo` is itself a
                    // propagated bound derived from several reasons, dropping its
                    // `aux_reasons` here would yield an incomplete conflict
                    // explanation one derivation step later. `split_reasons`
                    // deduplicates downstream.
                    lower_reasons.extend(lo.all_reasons());
                } else {
                    can_derive_lower = false;
                    break;
                }
            } else {
                if let Some(hi) = &self.upper[var_idx] {
                    lower_sum += hi.value * *coef;
                    lower_reasons.extend(hi.all_reasons());
                } else {
                    can_derive_lower = false;
                    break;
                }
            }
        }
        if can_derive_lower {
            let is_tighter = match &self.lower[idx] {
                None => true,
                Some(existing) => lower_sum > existing.value,
            };
            if is_tighter {
                return Some(PropagatedBound {
                    var: basic_var,
                    is_lower: true,
                    value: lower_sum,
                    reasons: lower_reasons,
                });
            }
        }
        let mut upper_sum = DeltaRational::from_rational(expr.constant);
        let mut upper_reasons: SmallVec<[u32; 4]> = SmallVec::new();
        let mut can_derive_upper = true;
        for (var, coef) in &expr.terms {
            let var_idx = *var as usize;
            if *coef > Rational64::zero() {
                if let Some(hi) = &self.upper[var_idx] {
                    upper_sum += hi.value * *coef;
                    upper_reasons.extend(hi.all_reasons());
                } else {
                    can_derive_upper = false;
                    break;
                }
            } else {
                if let Some(lo) = &self.lower[var_idx] {
                    upper_sum += lo.value * *coef;
                    upper_reasons.extend(lo.all_reasons());
                } else {
                    can_derive_upper = false;
                    break;
                }
            }
        }
        if can_derive_upper {
            let is_tighter = match &self.upper[idx] {
                None => true,
                Some(existing) => upper_sum < existing.value,
            };
            if is_tighter {
                return Some(PropagatedBound {
                    var: basic_var,
                    is_lower: false,
                    value: upper_sum,
                    reasons: upper_reasons,
                });
            }
        }
        None
    }
    /// Get pending propagated bounds
    #[must_use]
    pub fn get_propagated(&self) -> &[PropagatedBound] {
        &self.propagated
    }
    /// Clear propagated bounds
    pub fn clear_propagated(&mut self) {
        self.propagated.clear();
    }
    /// Tighten bounds on a variable if possible
    /// Returns true if bounds were tightened
    ///
    /// Like [`Self::propagate_bounds`] (see its doc comment for the full
    /// rationale), this routes writes through the undo trail via
    /// `set_lower_delta`/`set_upper_delta` rather than writing
    /// `self.lower`/`self.upper` directly, and skips applying a derived
    /// bound with no recorded reason rather than fabricating one.
    pub fn tighten_bounds(&mut self, var: VarId) -> bool {
        let idx = var as usize;
        let mut changed = false;
        if let Some(expr) = self.tableau.get(&var).cloned()
            && let Some(prop) = self.derive_basic_bound(var, &expr)
            && !prop.reasons.is_empty()
        {
            if prop.is_lower {
                let should_update = match &self.lower[idx] {
                    None => true,
                    Some(existing) => prop.value > existing.value,
                };
                if should_update {
                    self.set_lower_delta(var, prop.value, prop.reasons.clone());
                    changed = true;
                }
            } else {
                let should_update = match &self.upper[idx] {
                    None => true,
                    Some(existing) => prop.value < existing.value,
                };
                if should_update {
                    self.set_upper_delta(var, prop.value, prop.reasons.clone());
                    changed = true;
                }
            }
        }
        changed
    }
    /// Get the number of original (non-slack) variables
    #[must_use]
    pub fn num_original_vars(&self) -> usize {
        self.num_vars
    }
    /// Get lower bound of a variable (if any)
    #[must_use]
    pub fn get_lower(&self, var: VarId) -> Option<&Bound> {
        self.lower.get(var as usize).and_then(|b| b.as_ref())
    }
    /// Get upper bound of a variable (if any)
    #[must_use]
    pub fn get_upper(&self, var: VarId) -> Option<&Bound> {
        self.upper.get(var as usize).and_then(|b| b.as_ref())
    }
    /// Reset the solver
    pub fn reset(&mut self) {
        self.num_vars = 0;
        self.num_slack = 0;
        self.assignment.clear();
        self.lower.clear();
        self.upper.clear();
        self.tableau.clear();
        self.basic.clear();
        self.infeasible = None;
        self.propagated.clear();
        self.trail.clear();
        self.trail_limits.clear();
        self.trail_limits.push(0);
        self.cached_assignments.clear();
        self.saved_tableaux.clear();
        self.resource_limit = false;
    }
    /// Push a new decision level
    pub fn push(&mut self) {
        self.trail_limits.push(self.trail.len());
        self.cached_assignments.push(self.assignment.clone());
        self.saved_tableaux
            .push((self.tableau.clone(), self.basic.clone()));
    }
    /// Pop to previous decision level
    pub fn pop(&mut self) {
        if let Some(limit) = self.trail_limits.pop() {
            while self.trail.len() > limit {
                if let Some(undo) = self.trail.pop() {
                    match undo {
                        BoundUndo::LowerWasNone(var) => {
                            self.lower[var as usize] = None;
                        }
                        BoundUndo::LowerWasSome(var, old) => {
                            self.lower[var as usize] = Some(old);
                        }
                        BoundUndo::UpperWasNone(var) => {
                            self.upper[var as usize] = None;
                        }
                        BoundUndo::UpperWasSome(var, old) => {
                            self.upper[var as usize] = Some(old);
                        }
                        BoundUndo::NewVar => {
                            self.num_vars -= 1;
                            self.assignment.pop();
                            self.lower.pop();
                            self.upper.pop();
                            self.basic.pop();
                        }
                        BoundUndo::NewSlack(id) => {
                            self.num_slack -= 1;
                            self.assignment.pop();
                            self.lower.pop();
                            self.upper.pop();
                            self.basic.pop();
                            let _ = id;
                        }
                    }
                }
            }
            if let Some((saved_tableau, saved_basic)) = self.saved_tableaux.pop() {
                self.tableau = saved_tableau;
                let cur_len = self.basic.len();
                let restore_len = saved_basic.len().min(cur_len);
                self.basic[..restore_len].copy_from_slice(&saved_basic[..restore_len]);
                for item in self.basic.iter_mut().skip(restore_len) {
                    *item = false;
                }
            } else {
                let num_vars = self.assignment.len();
                self.tableau.retain(|&var, expr| {
                    if (var as usize) >= num_vars {
                        return false;
                    }
                    for (v, _) in &expr.terms {
                        if (*v as usize) >= num_vars {
                            return false;
                        }
                    }
                    true
                });
                for i in 0..num_vars {
                    let var_id = i as VarId;
                    if self.basic[i] && !self.tableau.contains_key(&var_id) {
                        self.basic[i] = false;
                    }
                }
            }
            if let Some(cached) = self.cached_assignments.pop() {
                let restore_len = cached.len().min(self.assignment.len());
                self.assignment[..restore_len].copy_from_slice(&cached[..restore_len]);
                for item in self.assignment.iter_mut().skip(restore_len) {
                    *item = DeltaRational::zero();
                }
            } else {
                for item in self.assignment.iter_mut() {
                    *item = DeltaRational::zero();
                }
            }
            self.infeasible = None;
        }
    }
    /// Get the current decision level
    #[must_use]
    pub fn decision_level(&self) -> usize {
        self.trail_limits.len().saturating_sub(1)
    }
    /// Current backtracking scope depth: the number of [`Simplex::push`] calls
    /// that have not yet been matched by a [`Simplex::pop`].
    ///
    /// This is the crate-internal name used by the scope-balance assertions in
    /// the theory solvers built on top of the simplex (e.g.
    /// `LiaSolver::check_balanced`), and is exactly [`Simplex::decision_level`]
    /// — a solver that pushes a search scope per decision has the two notions
    /// coincide. Kept as a separate name so a scope-balance assertion reads as
    /// what it checks rather than borrowing the CDCL vocabulary.
    #[inline]
    #[must_use]
    pub(crate) fn scope_depth(&self) -> usize {
        self.decision_level()
    }
    /// Number of allocated variable slots (original + slack).
    #[inline]
    pub(super) fn assignment_len(&self) -> usize {
        self.assignment.len()
    }
    /// Real-part of the assignment at index `idx`.
    #[inline]
    pub(super) fn assignment_real_at(&self, idx: usize) -> Rational64 {
        self.assignment[idx].real
    }
    /// Full `DeltaRational` assignment at index `idx`.
    #[inline]
    pub(super) fn assignment_at(&self, idx: usize) -> Rational64 {
        self.assignment[idx].real
    }
    /// Whether variable at `idx` is currently basic.
    #[inline]
    pub(super) fn is_basic(&self, idx: usize) -> bool {
        idx < self.basic.len() && self.basic[idx]
    }
    /// Iterate over `(basic_var, row)` pairs in the tableau.
    pub(super) fn tableau_iter(&self) -> impl Iterator<Item = (&VarId, &LinExpr)> {
        self.tableau.iter()
    }
    /// Iterate over basic variable IDs in the tableau.
    pub(super) fn tableau_keys(&self) -> impl Iterator<Item = VarId> + '_ {
        self.tableau.keys().copied()
    }
    /// Return the coefficient of `nonbasic` in the row of `basic`, or `None`.
    pub(super) fn tableau_coef_of(&self, basic: VarId, nonbasic: VarId) -> Option<Rational64> {
        self.tableau.get(&basic).and_then(|row| {
            row.terms
                .iter()
                .find(|(v, _)| *v == nonbasic)
                .map(|(_, c)| *c)
        })
    }
    /// Real part of the upper bound for variable at `idx`, if any.
    #[inline]
    pub(super) fn upper_real_at(&self, idx: usize) -> Option<Rational64> {
        self.upper
            .get(idx)
            .and_then(|b| b.as_ref().map(|b| b.value.real))
    }
    /// Real part of the lower bound for variable at `idx`, if any.
    #[inline]
    pub(super) fn lower_real_at(&self, idx: usize) -> Option<Rational64> {
        self.lower
            .get(idx)
            .and_then(|b| b.as_ref().map(|b| b.value.real))
    }
    /// Full `DeltaRational` upper bound for variable at `idx`, if any.
    #[inline]
    pub(super) fn upper_delta_at(&self, idx: usize) -> Option<DeltaRational> {
        self.upper
            .get(idx)
            .and_then(|b| b.as_ref().map(|b| b.value))
    }
    /// Full `DeltaRational` lower bound for variable at `idx`, if any.
    #[inline]
    pub(super) fn lower_delta_at(&self, idx: usize) -> Option<DeltaRational> {
        self.lower
            .get(idx)
            .and_then(|b| b.as_ref().map(|b| b.value))
    }
    /// Overwrite the assignment at `idx` with `val`.
    #[inline]
    pub(super) fn set_assignment_at(&mut self, idx: usize, val: DeltaRational) {
        self.assignment[idx] = val;
    }
    /// Maximum pivot count configured for this instance.
    #[inline]
    pub(super) fn max_pivots(&self) -> usize {
        self.max_pivots
    }
}
pub use super::simplex_opt::SimplexOptStatus;

#[cfg(test)]
mod tests;
