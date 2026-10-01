//! Arithmetic Theory Solver

use super::simplex::{LinExpr, Simplex, VarId};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::theory::{EqualityNotification, Theory, TheoryCombination, TheoryId, TheoryResult};
use num_rational::Rational64;
use num_traits::{One, Signed, Zero};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;

/// Compute GCD of two i64 values
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Compute GCD of two i128 values (used by the Diophantine consistency check).
fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Arithmetic Theory Solver (LRA/LIA)
#[derive(Debug)]
pub struct ArithSolver {
    /// Simplex instance
    simplex: Simplex,
    /// Term to variable mapping
    term_to_var: FxHashMap<TermId, VarId>,
    /// Variable to term mapping
    var_to_term: Vec<TermId>,
    /// Reason counter
    reason_counter: u32,
    /// Reason to term mapping
    reasons: Vec<TermId>,
    /// Is this LIA (integers) or LRA (reals)?
    is_integer: bool,
    /// Context stack
    context_stack: Vec<ContextState>,
    /// Accumulated shared equalities (from notify_equality calls)
    shared_equalities: Vec<EqualityNotification>,
    /// Integral model recorded by the LIA branch-and-bound search.
    ///
    /// Populated only when the most recent `check()` proved `Sat` in integer
    /// mode.  `value()` consults this first for Int terms so it returns the
    /// integral assignment found by branch-and-bound rather than the (possibly
    /// fractional) LP-relaxation optimum.  Cleared at the start of every
    /// `check()` and on `reset()`.
    lia_model: FxHashMap<VarId, Rational64>,
    /// Integer equalities asserted in LIA mode, kept as raw
    /// `(sum a_i·x_i = b)` rows so that a linear Diophantine consistency check
    /// can detect cross-constraint parity infeasibility (e.g. `y=2x ∧ y=2z+1`)
    /// that per-equation GCD reasoning and pure branch-and-bound over unbounded
    /// variables miss.  Push/pop-scoped via `ContextState`.
    int_equalities: Vec<IntEquation>,
    /// Terms of sort `Int` in a solver that runs over the reals (`LRA` mode,
    /// the default under `(set-logic ALL)` or no logic at all): the variables
    /// branch-and-bound must make integral (`mark_int_term`, the `mixed`
    /// module).  Always empty in `LIA` mode, where every variable is one.
    /// Kept across `reset()`: it records sorts, which never change.
    int_terms: FxHashSet<TermId>,
    /// Whether the integrality of `int_terms` is decided by the caller, at
    /// the end of a search, rather than inside this solver (the `mixed`
    /// module, [`ArithSolver::set_defer_integrality`]): the solver then runs
    /// exactly as it does with no `Int` term marked.  Kept across `reset()`.
    defer_integrality: bool,
}

/// A linear equality over the integers: `sum(coeff_i · var_i) = rhs`.
#[derive(Debug, Clone)]
struct IntEquation {
    terms: Vec<(VarId, i64)>,
    rhs: i64,
}

/// Outcome of exploring a single branch-and-bound child node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BranchOutcome {
    /// An integral assignment satisfying all constraints was found.
    Sat,
    /// This branch is proven infeasible (a dead end).
    Infeasible,
    /// The branch could not be resolved within the resource budget.
    Unknown,
}

/// State for push/pop
#[derive(Debug, Clone)]
struct ContextState {
    num_vars: usize,
    num_reasons: usize,
    num_shared_equalities: usize,
    num_int_equalities: usize,
}

impl Default for ArithSolver {
    fn default() -> Self {
        Self::new(false)
    }
}

impl ArithSolver {
    /// Create a new arithmetic solver
    #[must_use]
    pub fn new(is_integer: bool) -> Self {
        Self {
            simplex: Simplex::new(),
            term_to_var: FxHashMap::default(),
            var_to_term: Vec::new(),
            reason_counter: 0,
            reasons: Vec::new(),
            is_integer,
            context_stack: Vec::new(),
            shared_equalities: Vec::new(),
            lia_model: FxHashMap::default(),
            int_equalities: Vec::new(),
            int_terms: FxHashSet::default(),
            defer_integrality: false,
        }
    }

    /// Create a new LRA solver
    #[must_use]
    pub fn lra() -> Self {
        Self::new(false)
    }

    /// Create a new LIA solver
    #[must_use]
    pub fn lia() -> Self {
        Self::new(true)
    }

    /// Whether this solver operates in integer (LIA) mode
    #[must_use]
    pub fn is_integer(&self) -> bool {
        self.is_integer
    }

    /// Intern a term as a variable
    pub fn intern(&mut self, term: TermId) -> VarId {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }

        let var = self.simplex.new_var();
        self.term_to_var.insert(term, var);
        self.var_to_term.push(term);
        var
    }

    /// Add a reason and return its ID
    fn add_reason(&mut self, term: TermId) -> u32 {
        let id = self.reason_counter;
        self.reason_counter += 1;
        self.reasons.push(term);
        id
    }

    /// Normalize a linear expression
    ///
    /// Normalization performs:
    /// 1. Coefficient reduction: divide by GCD of all coefficients
    /// 2. Sorting: order terms by variable ID for canonical form
    /// 3. Sign normalization: ensure first coefficient (after sorting) is positive
    ///
    /// IMPORTANT: Step 3 is only safe for symmetric constraints (equalities).
    /// For inequalities (Le/Ge), sign normalization flips the direction and must
    /// NOT be applied.  Call `normalize_expr_no_sign` for those cases instead.
    fn normalize_expr(&self, expr: &mut LinExpr) {
        if expr.terms.is_empty() {
            return;
        }

        // For integer arithmetic, reduce by GCD
        if self.is_integer {
            // Find GCD of all coefficients
            let gcd = expr
                .terms
                .iter()
                .map(|(_, c)| c.numer().abs())
                .fold(0i64, |acc, n| if acc == 0 { n } else { gcd_i64(acc, n) });

            if gcd > 1 {
                let divisor = Rational64::from_integer(gcd);
                expr.scale(Rational64::one() / divisor);
            }
        }

        // Ensure first coefficient is positive
        if let Some((_, c)) = expr.terms.first()
            && c.is_negative()
        {
            expr.negate();
        }

        // Sort terms by variable ID for canonical form
        expr.terms.sort_by_key(|(v, _)| *v);
    }

    /// Normalize for inequalities: GCD reduction and sorting only.
    ///
    /// Sign normalization is deliberately omitted because negating an inequality
    /// expression reverses its direction (e.g., fa - fb <= 0 becomes fb - fa <= 0,
    /// which represents the opposite constraint fa >= fb).
    fn normalize_ineq_expr(&self, expr: &mut LinExpr) {
        if expr.terms.is_empty() {
            return;
        }

        // For integer arithmetic, reduce by GCD only (preserves sign)
        if self.is_integer {
            let gcd = expr
                .terms
                .iter()
                .map(|(_, c)| c.numer().abs())
                .fold(0i64, |acc, n| if acc == 0 { n } else { gcd_i64(acc, n) });

            if gcd > 1 {
                let divisor = Rational64::from_integer(gcd);
                expr.scale(Rational64::one() / divisor);
            }
        }

        // Sort terms by variable ID — safe because sorting doesn't change the sign
        // of the overall expression for inequalities (we don't negate afterwards).
        // NOTE: Sorting alone is also problematic because it reorders terms but the
        // sign is determined by all terms together.  We keep the sort for consistent
        // canonical form but do NOT apply the sign-flip step.
        expr.terms.sort_by_key(|(v, _)| *v);
    }

    /// Assert: lhs <= rhs
    pub fn assert_le(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            expr.add_term(var, *coef);
        }
        expr.add_constant(-rhs);

        // Use inequality-safe normalization: GCD reduction + sort, but NO sign flip.
        // sign normalization (negation) would reverse the inequality direction.
        self.normalize_ineq_expr(&mut expr);

        let reason_id = self.add_reason(reason);
        self.simplex.add_le(expr, reason_id);
    }

    /// Assert: lhs >= rhs
    pub fn assert_ge(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            expr.add_term(var, *coef);
        }
        expr.add_constant(-rhs);

        // Use inequality-safe normalization: GCD reduction + sort, but NO sign flip.
        self.normalize_ineq_expr(&mut expr);

        let reason_id = self.add_reason(reason);
        self.simplex.add_ge(expr, reason_id);
    }

    /// Assert: lhs = rhs
    ///
    /// For integer arithmetic (LIA), checks GCD-based infeasibility:
    /// If all coefficients share a common GCD that doesn't divide the RHS,
    /// the constraint is infeasible over integers.
    ///
    /// Example: 2x + 2y = 7 is infeasible because gcd(2,2) = 2 doesn't divide 7.
    pub fn assert_eq(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            expr.add_term(var, *coef);
        }
        expr.add_constant(-rhs);

        // For LIA, check GCD-based infeasibility BEFORE normalization
        // (normalization divides by GCD, which would lose the infeasibility signal)
        if !self.is_integer && self.mixed_integer_row_infeasible(lhs, &expr, reason) {
            return;
        }
        if self.is_integer {
            // Extract integer coefficients
            let coeffs: Vec<i64> = expr
                .terms
                .iter()
                .filter_map(|(_, c)| {
                    if c.denom() == &1 {
                        Some(*c.numer())
                    } else {
                        None
                    }
                })
                .collect();

            // Extract the constant (which is -rhs in expr = 0 form)
            let const_term = if expr.constant.denom() == &1 {
                -*expr.constant.numer()
            } else {
                // Non-integer constant in equality - infeasible for integers.
                // Attribute the contradiction to the actual assertion that
                // caused it (not a hardcoded/arbitrary reason id), so the
                // resulting unsat core cites the real culprit.
                let reason_id = self.add_reason(reason);
                if let Some(&(var, _)) = expr.terms.first() {
                    self.simplex
                        .set_lower(var, Rational64::from_integer(1), reason_id);
                    self.simplex
                        .set_upper(var, Rational64::from_integer(0), reason_id);
                }
                return;
            };

            // Check GCD infeasibility if all coefficients are integers
            if !coeffs.is_empty() && coeffs.len() == expr.terms.len() {
                // Record the integer equality (sum a_i·x_i = const_term) so the
                // cross-constraint Diophantine consistency check can see it.
                let eq_terms: Vec<(VarId, i64)> =
                    expr.terms.iter().map(|(v, c)| (*v, *c.numer())).collect();
                self.int_equalities.push(IntEquation {
                    terms: eq_terms,
                    rhs: const_term,
                });

                // Compute GCD of all coefficients
                let g = coeffs.iter().fold(0i64, |acc, &c| gcd_i64(acc, c.abs()));

                if g > 0 && const_term % g != 0 {
                    // GCD infeasibility detected!
                    // Add contradictory constraints: x >= 1 and x <= 0,
                    // attributed to the actual equality assertion that
                    // caused the contradiction (not a hardcoded reason id)
                    // so `check()`'s unsat core cites the real culprit
                    // instead of whatever the first reason ever added
                    // happened to be.
                    let reason_id = self.add_reason(reason);
                    if let Some(&(var, _)) = expr.terms.first() {
                        self.simplex
                            .set_lower(var, Rational64::from_integer(1), reason_id);
                        self.simplex
                            .set_upper(var, Rational64::from_integer(0), reason_id);
                    }
                    return;
                }
            }
        }

        // Normalize the expression
        self.normalize_expr(&mut expr);

        let reason_id = self.add_reason(reason);
        self.simplex.add_eq(expr, reason_id);
    }

    /// Assert: lhs < rhs (strict inequality)
    /// For LRA, uses infinitesimals: lhs <= rhs - δ
    /// For LIA, transforms to: lhs <= rhs - 1 (since no integer exists between k and k+1)
    pub fn assert_lt(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        // For integer arithmetic, x < k is equivalent to x <= k - 1
        // because there's no integer strictly between k-1 and k
        if self.is_integer {
            // Transform: lhs < rhs becomes lhs <= rhs - 1
            self.assert_le(lhs, rhs - Rational64::one(), reason);
            return;
        }
        // The same over an all-`Int` row of a mixed solver (the `mixed`
        // module): an integer below `rhs` is at most `⌈rhs⌉ − 1`.
        if self.is_integral_row(lhs) {
            self.assert_le(lhs, rhs.ceil() - Rational64::one(), reason);
            return;
        }

        // For reals, use delta-rationals
        // lhs < rhs is equivalent to lhs - rhs < 0
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            expr.add_term(var, *coef);
        }
        expr.add_constant(-rhs);

        // Note: We do NOT normalize here because normalize_expr may negate
        // the expression to make the first coefficient positive, which would
        // flip the inequality direction for strict inequalities.

        let reason_id = self.add_reason(reason);
        self.simplex.add_strict_lt(expr, reason_id);
    }

    /// Assert: lhs > rhs (strict inequality)
    /// For LRA, uses infinitesimals: lhs >= rhs + δ
    /// For LIA, transforms to: lhs >= rhs + 1 (since no integer exists between k and k+1)
    pub fn assert_gt(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        // For integer arithmetic, x > k is equivalent to x >= k + 1
        // because there's no integer strictly between k and k+1
        if self.is_integer {
            // Transform: lhs > rhs becomes lhs >= rhs + 1
            self.assert_ge(lhs, rhs + Rational64::one(), reason);
            return;
        }
        // The same over an all-`Int` row of a mixed solver (the `mixed`
        // module): an integer above `rhs` is at least `⌊rhs⌋ + 1`.
        if self.is_integral_row(lhs) {
            self.assert_ge(lhs, rhs.floor() + Rational64::one(), reason);
            return;
        }

        // For reals, use delta-rationals
        // lhs > rhs is equivalent to rhs - lhs < 0
        // We build rhs - lhs directly instead of negating lhs - rhs
        // This avoids issues with normalize_expr which ensures positive first coefficient
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            // Add negative coefficient since we want rhs - lhs
            expr.add_term(var, -(*coef));
        }
        // Add +rhs (since we want rhs - lhs, not lhs - rhs)
        expr.add_constant(rhs);

        // Note: We do NOT normalize here because:
        // 1. normalize_expr may negate to make first coefficient positive
        // 2. This would flip the inequality direction
        // 3. For strict inequalities, the sign matters

        let reason_id = self.add_reason(reason);
        self.simplex.add_strict_lt(expr, reason_id);
    }

    /// Get the current value of a variable
    ///
    /// For integer arithmetic (LIA), this properly rounds values that have
    /// infinitesimal components from strict inequalities:
    /// - If value is `r + δ` (positive delta), return `ceil(r)` for integers
    /// - If value is `r - δ` (negative delta), return `floor(r)` for integers
    #[must_use]
    pub fn value(&self, term: TermId) -> Option<Rational64> {
        self.term_to_var.get(&term).map(|&var| {
            // Prefer the assignment branch-and-bound found when the last
            // check() proved Sat — the raw LP optimum may be fractional for
            // Int variables.  In a mixed solver (`LRA` mode with `Int` terms)
            // the snapshot holds every variable, so the reals it reports agree
            // with the integers (the `mixed` module).
            if let Some(v) = self.lia_model.get(&var) {
                return *v;
            }
            if self.is_integer || self.rounds_as_integer(term) {
                // Get the full delta-rational value
                let dval = self.simplex.delta_value(var);

                // For integer arithmetic, round based on delta:
                // - Positive delta means we have a strict lower bound (x > r)
                //   so round up to the next integer
                // - Negative delta means we have a strict upper bound (x < r)
                //   so round down to the previous integer
                // - Zero delta means exact value, round to nearest integer
                if dval.delta.is_positive() {
                    // x > r implies x >= ceil(r) for integers
                    // If r is already an integer, we need r + 1
                    let real_val = dval.real;
                    if real_val.is_integer() {
                        Rational64::from_integer(real_val.to_integer() + 1)
                    } else {
                        Rational64::from_integer(real_val.ceil().to_integer())
                    }
                } else if dval.delta.is_negative() {
                    // x < r implies x <= floor(r) for integers
                    // If r is already an integer, we need r - 1
                    let real_val = dval.real;
                    if real_val.is_integer() {
                        Rational64::from_integer(real_val.to_integer() - 1)
                    } else {
                        Rational64::from_integer(real_val.floor().to_integer())
                    }
                } else {
                    // No strict bound, just return the value
                    // Round to nearest integer for consistency
                    dval.real
                }
            } else {
                // For reals, the raw real part is NOT a model: a variable
                // sitting at a strict bound is stored as `r ± δ`, so returning
                // `r` alone reports a witness that violates the very constraint
                // that created it (e.g. `x > 0` would report `x = 0`).
                // Substitute a concrete positive δ₀ that keeps every bound
                // satisfied (see `Simplex::delta_instantiation`).
                let dval = self.simplex.delta_value(var);
                if dval.delta.is_zero() {
                    dval.real
                } else {
                    dval.real + dval.delta * self.simplex.delta_instantiation()
                }
            }
        })
    }

    /// Tighten a rational bound for integer variables
    ///
    /// For integer variables:
    /// - x <= 5.7 becomes x <= 5
    /// - x >= 2.3 becomes x >= 3
    /// - x < 5.0 becomes x <= 4
    /// - x > 2.0 becomes x >= 3
    #[allow(dead_code)]
    fn tighten_bound(&self, bound: Rational64, is_upper: bool) -> Rational64 {
        if !self.is_integer {
            return bound;
        }

        // For upper bounds (<=), floor the value
        // For lower bounds (>=), ceiling the value
        if bound.is_integer() {
            bound
        } else if is_upper {
            // x <= 5.7 becomes x <= 5
            Rational64::from_integer(bound.floor().to_integer())
        } else {
            // x >= 2.3 becomes x >= 3
            Rational64::from_integer(bound.ceil().to_integer())
        }
    }

    /// Maximum branch-and-bound tree depth for the LIA integrality search.
    const LIA_MAX_DEPTH: usize = 512;
    /// Maximum number of branch-and-bound nodes explored before giving up
    /// (returning `Unknown`).  Bounds worst-case exponential search.
    const LIA_MAX_NODES: usize = 20_000;

    /// Collect the simplex variable ids of all interned (Int) terms, sorted for
    /// deterministic branching order.  Slack variables are excluded — we only
    /// branch on the original integer-sorted variables.
    fn interned_int_vars(&self) -> Vec<VarId> {
        let mut vars: Vec<VarId> = self
            .var_to_term
            .iter()
            .filter(|term| self.is_integer || self.int_terms.contains(term))
            .filter_map(|term| self.term_to_var.get(term).copied())
            .collect();
        vars.sort_unstable();
        vars.dedup();
        vars
    }

    /// Find the first interned Int variable whose current LP value is not an
    /// integer, with the two branch bounds `(var, down, up)`: `var ≤ down` or
    /// `var ≥ up`.  A value `r ± δ` at an integral `r` (a strict bound the
    /// relaxation sits at, possible only in a mixed solver, where strict
    /// bounds keep their infinitesimal) is no integer either: `r − δ`
    /// branches to `≤ r − 1` / `≥ r`, `r + δ` to `≤ r` / `≥ r + 1`.
    fn find_fractional_int_var(
        &self,
        int_vars: &[VarId],
    ) -> Option<(VarId, Rational64, Rational64)> {
        for &var in int_vars {
            let value = self.simplex.delta_value(var);
            let real = value.real;
            if !real.is_integer() {
                return Some((var, real.floor(), real.ceil()));
            }
            if value.delta.is_negative() {
                return Some((var, real - Rational64::one(), real));
            }
            if value.delta.is_positive() {
                return Some((var, real, real + Rational64::one()));
            }
        }
        None
    }

    /// Build a sound (over-approximate) unsat core: every assertion reason known
    /// to the solver.  When branch-and-bound proves integer-infeasibility, the
    /// full conjunction of asserted constraints is genuinely inconsistent, so
    /// returning all of them is a valid (if imprecise) conflict explanation.
    fn full_unsat_core(&self) -> Vec<TermId> {
        let mut terms = self.reasons.clone();
        terms.sort_unstable();
        terms.dedup();
        terms
    }

    /// Every term this solver has interned as a shared (arithmetic-interface)
    /// variable, in intern order.
    ///
    /// This is the arithmetic half of Nelson-Oppen theory combination's
    /// "shared terms" set -- the candidate pool a combining layer probes for
    /// entailed (dis)equalities with another theory.
    #[must_use]
    pub fn interface_terms(&self) -> &[TermId] {
        &self.var_to_term
    }

    /// Probe whether `coef_a * var_a + coef_b * var_b < 0` is infeasible
    /// under the live (already-asserted) bounds, on a scratch push/pop scope
    /// so the tableau's real incremental state is untouched either way.
    ///
    /// `marker_term` seeds a throwaway reason id for the probe's own scratch
    /// assertion; that id is excluded from the returned certificate since the
    /// probe itself justifies nothing -- only the *live* constraints it
    /// conflicts with do. Returns the (marker-excluded) Farkas reason ids on
    /// infeasibility (`coef_a*var_a + coef_b*var_b < 0` is refuted, i.e. `>=
    /// 0` is entailed), `None` when the probe is satisfiable.
    fn probe_strict_lt_infeasible(
        &mut self,
        marker_term: TermId,
        var_a: VarId,
        coef_a: Rational64,
        var_b: VarId,
        coef_b: Rational64,
    ) -> Option<Vec<u32>> {
        self.simplex.push();
        let marker = self.add_reason(marker_term);
        let mut expr = LinExpr::new();
        expr.add_term(var_a, coef_a);
        expr.add_term(var_b, coef_b);
        self.simplex.add_strict_lt(expr, marker);
        let outcome = self.simplex.check();
        self.simplex.pop();
        match outcome {
            Ok(()) => None,
            Err(ids) => Some(ids.into_iter().filter(|&rid| rid != marker).collect()),
        }
    }

    /// Resolve a set of Farkas reason ids (as returned by a scratch probe)
    /// back into the `TermId`s they name, and release the scratch reason
    /// slots the probe allocated above `base`.
    ///
    /// An empty result after resolution means the entailment holds at
    /// decision level 0 -- forced by no live literal at all -- in which case
    /// the full (over-approximate) unsat core is substituted so a conflict
    /// that cites this entailment is still explainable.
    fn resolve_and_release_reasons(&mut self, ids: Vec<u32>, base: usize) -> Vec<TermId> {
        let mut terms: Vec<TermId> = ids
            .into_iter()
            .filter_map(|rid| self.reasons.get(rid as usize).copied())
            .collect();
        self.reasons.truncate(base);
        self.reason_counter = base as u32;
        terms.sort_unstable();
        terms.dedup();
        if terms.is_empty() {
            terms = self.full_unsat_core();
        }
        terms
    }

    /// Soundly determine whether arithmetic **entails** `x = y` under the
    /// live assertions, and if so return the reason terms whose conjunction
    /// forces it.
    ///
    /// This is the arithmetic-to-EUF half of Nelson-Oppen combination: EUF's
    /// congruence closure can only fire on an equality it has actually been
    /// told about, but two shared terms can be forced equal by *arithmetic*
    /// alone (e.g. `y = x + 1` together with `x = 2`) without any direct
    /// equality atom ever being asserted between them. Left unpropagated,
    /// the congruence that entailment should trigger (`f(y) = f(3)`) is
    /// silently missed -- the classic non-convex-combination false-`sat`.
    ///
    /// Implemented as two independent infeasibility probes, each on its own
    /// scratch scope: `x = y` is entailed iff both `x < y` and `x > y` are
    /// infeasible. The returned reason is the union of the two Farkas
    /// certificates, so a conflict that rests on the equality this induces in
    /// EUF can be expanded back to the literals that actually forced it.
    pub fn entailed_equal_reason(&mut self, x: TermId, y: TermId) -> Option<Vec<TermId>> {
        let (Some(var_x), Some(var_y)) = (
            self.term_to_var.get(&x).copied(),
            self.term_to_var.get(&y).copied(),
        ) else {
            return None;
        };
        let base = self.reasons.len();

        // First half: if no live assignment can make x strictly below y,
        // then y is never above x, i.e. x >= y holds everywhere.
        let Some(ge_ids) =
            self.probe_strict_lt_infeasible(x, var_x, Rational64::one(), var_y, -Rational64::one())
        else {
            self.reasons.truncate(base);
            self.reason_counter = base as u32;
            return None;
        };
        // Second half, the symmetric probe: nothing can push x strictly
        // above y either, so x <= y holds everywhere. Both halves together
        // leave x = y as the only possibility.
        let Some(le_ids) =
            self.probe_strict_lt_infeasible(y, var_y, Rational64::one(), var_x, -Rational64::one())
        else {
            self.reasons.truncate(base);
            self.reason_counter = base as u32;
            return None;
        };

        let mut ids = ge_ids;
        ids.extend(le_ids);
        Some(self.resolve_and_release_reasons(ids, base))
    }

    /// Soundly determine whether arithmetic **entails** `x != y` under the
    /// live assertions (the mirror of [`Self::entailed_equal_reason`]),
    /// returning the reason terms that force it.
    ///
    /// Mirrors cvc5's `watchedVariableCannotBeZero` technique: when EUF has
    /// (or is about to have) `x` and `y` in the same class, but arithmetic's
    /// bounds alone already rule out `x = y`, that is an immediate
    /// cross-theory conflict that must be raised rather than left for the
    /// tableau to rediscover on its own once the equality is asserted into
    /// it. `x = y` is checked jointly (`x <= y` and `y <= x` together, on one
    /// scratch scope) rather than as two independent probes, since it is a
    /// single proposition here, not two.
    pub fn entailed_disequal_reason(&mut self, x: TermId, y: TermId) -> Option<Vec<TermId>> {
        let (Some(var_x), Some(var_y)) = (
            self.term_to_var.get(&x).copied(),
            self.term_to_var.get(&y).copied(),
        ) else {
            return None;
        };
        let base = self.reasons.len();
        self.simplex.push();
        let marker = self.add_reason(x);
        let mut le = LinExpr::new();
        le.add_term(var_x, Rational64::one());
        le.add_term(var_y, -Rational64::one());
        self.simplex.add_le(le, marker);
        let mut ge = LinExpr::new();
        ge.add_term(var_y, Rational64::one());
        ge.add_term(var_x, -Rational64::one());
        self.simplex.add_le(ge, marker);
        let outcome = self.simplex.check();
        self.simplex.pop();
        let Err(ids) = outcome else {
            self.reasons.truncate(base);
            self.reason_counter = base as u32;
            return None;
        };
        let ids: Vec<u32> = ids.into_iter().filter(|&rid| rid != marker).collect();
        Some(self.resolve_and_release_reasons(ids, base))
    }

    /// Snapshot the current (integral) LP assignment of every interned Int
    /// variable into `lia_model`.  Called at an integer-feasible leaf so that
    /// `value()` reports the integral model after branch-and-bound unwinds.
    fn snapshot_lia_model(&mut self, int_vars: &[VarId]) {
        self.lia_model.clear();
        if !self.is_integer {
            self.snapshot_mixed_model();
            return;
        }
        for &var in int_vars {
            self.lia_model.insert(var, self.simplex.value(var));
        }
    }

    /// Decide whether the accumulated system of integer equalities has NO
    /// integer solution (a sound, one-sided UNSAT detector).
    ///
    /// Every equality `sum a_i·x_i = b` is an exact integer row.  We run integer
    /// (fraction-free) Gaussian elimination: reducing rows with the identity
    /// `row := (a/g)·row − (b/g)·pivot` (g = gcd of the two pivot-column
    /// entries) produces rows that are integer linear combinations of the
    /// originals, hence consequences that every integer solution must satisfy.
    /// For any resulting row `sum c_j·x_j = d`, an integer solution requires
    /// `gcd(c_j) | d`; if that fails — or a row reduces to `0 = d` with `d ≠ 0` —
    /// the whole system is integer-infeasible.
    ///
    /// This catches cross-constraint parity infeasibility such as
    /// `y = 2x ∧ y = 2z + 1` (⇒ `2x − 2z = 1`, and `gcd(2,2) = 2 ∤ 1`), which
    /// per-equation GCD reasoning and unbounded branch-and-bound cannot.
    ///
    /// The check is *sound but incomplete*: it only ever concludes UNSAT.  If an
    /// intermediate value would overflow `i128`, or the system is too large, it
    /// conservatively returns `false` (defer to branch-and-bound).
    fn int_equalities_infeasible(&self) -> bool {
        if self.int_equalities.is_empty() {
            return false;
        }

        // Assign a dense column index to every variable that appears.
        let mut col_of: FxHashMap<VarId, usize> = FxHashMap::default();
        for eq in &self.int_equalities {
            for &(v, _) in &eq.terms {
                let next = col_of.len();
                col_of.entry(v).or_insert(next);
            }
        }
        let cols = col_of.len();
        let rows = self.int_equalities.len();

        // Bound the work: skip very large systems (defer to branch-and-bound).
        if cols == 0 || rows.saturating_mul(cols) > 200_000 {
            return false;
        }

        // Dense augmented matrix: last entry of each row is the RHS.
        let mut mat: Vec<Vec<i128>> = vec![vec![0i128; cols + 1]; rows];
        for (r, eq) in self.int_equalities.iter().enumerate() {
            for &(v, c) in &eq.terms {
                if let Some(&col) = col_of.get(&v) {
                    mat[r][col] += c as i128;
                }
            }
            mat[r][cols] = eq.rhs as i128;
        }

        // Fraction-free Gaussian elimination.
        let mut pivot_row = 0usize;
        for col in 0..cols {
            // Find a pivot at or below `pivot_row` with a nonzero entry.
            let Some(sel) = (pivot_row..rows).find(|&r| mat[r][col] != 0) else {
                continue;
            };
            mat.swap(pivot_row, sel);

            // Snapshot the pivot row to avoid aliasing two rows of `mat`.
            let pivot = mat[pivot_row].clone();
            let a = pivot[col];

            // Eliminate this column from every other row.
            for (r, row) in mat.iter_mut().enumerate() {
                if r == pivot_row || row[col] == 0 {
                    continue;
                }
                let b = row[col];
                let g = gcd_i128(a, b);
                let fa = a / g; // scale for row r
                let fb = b / g; // scale for pivot
                for (k, &pv) in pivot.iter().enumerate().skip(col) {
                    let lhs = match row[k].checked_mul(fa) {
                        Some(v) => v,
                        None => return false, // overflow → cannot decide
                    };
                    let rhs = match pv.checked_mul(fb) {
                        Some(v) => v,
                        None => return false,
                    };
                    row[k] = match lhs.checked_sub(rhs) {
                        Some(v) => v,
                        None => return false,
                    };
                }
            }

            pivot_row += 1;
            if pivot_row == rows {
                break;
            }
        }

        // Consequence check: each row must be integer-satisfiable on its own.
        for row in &mat {
            let mut g = 0i128;
            for &c in &row[..cols] {
                g = gcd_i128(g, c);
            }
            let d = row[cols];
            if g == 0 {
                // 0 = d with d ≠ 0 is inconsistent (even over the rationals).
                if d != 0 {
                    return true;
                }
            } else if d % g != 0 {
                // gcd of coefficients does not divide the constant ⇒ no integer
                // solution to this consequence ⇒ system integer-infeasible.
                return true;
            }
        }

        false
    }

    /// Entry point for the LIA integrality search (branch-and-bound).
    ///
    /// Precondition: the LP relaxation is feasible and not resource-limited.
    fn lia_branch_and_bound(&mut self) -> Result<TheoryResult> {
        // Cheap, sound integer-equality consistency check first — resolves
        // cross-constraint parity infeasibility that branch-and-bound over
        // unbounded variables would otherwise only be able to report as Unknown.
        if self.int_equalities_infeasible() {
            return Ok(TheoryResult::Unsat(self.full_unsat_core()));
        }
        let int_vars = self.interned_int_vars();
        let mut nodes: usize = 0;
        self.bnb_recurse(&int_vars, 0, &mut nodes)
    }

    /// Recursive branch-and-bound over integer variables.
    ///
    /// Uses balanced simplex push/pop so no branch constraint leaks into the
    /// caller's decision level.  The satisfying integral assignment is captured
    /// into `lia_model` at the feasible leaf (before the pushes unwind), so
    /// `value()` can report it afterwards.
    ///
    /// Returns:
    /// - `Sat` if an integral assignment is found;
    /// - `Unsat(core)` if BOTH branches on the fractional variable are
    ///   infeasible (integer-infeasible);
    /// - `Unknown` if the depth/node budget is exhausted, or a sub-solve hit the
    ///   simplex pivot limit — never a fabricated Sat/Unsat.
    fn bnb_recurse(
        &mut self,
        int_vars: &[VarId],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<TheoryResult> {
        if depth > Self::LIA_MAX_DEPTH || *nodes > Self::LIA_MAX_NODES {
            return Ok(TheoryResult::Unknown);
        }
        *nodes += 1;

        // Find a fractional Int variable at the current LP optimum.
        let (var, floor_v, ceil_v) = match self.find_fractional_int_var(int_vars) {
            None => {
                // Fully integral leaf: record the model, then report Sat.
                self.snapshot_lia_model(int_vars);
                return Ok(TheoryResult::Sat);
            }
            Some(found) => found,
        };

        // Track whether any explored branch was left unresolved (Unknown) so we
        // never collapse an Unknown into a spurious Unsat.
        let mut saw_unknown = false;

        // Branch down: var <= floor(value).
        self.simplex.push();
        self.simplex.set_upper(var, floor_v, 0);
        let down = self.explore_branch(int_vars, depth, nodes)?;
        self.simplex.pop();
        match down {
            BranchOutcome::Sat => return Ok(TheoryResult::Sat),
            BranchOutcome::Unknown => saw_unknown = true,
            BranchOutcome::Infeasible => {}
        }

        // Branch up: var >= ceil(value).
        self.simplex.push();
        self.simplex.set_lower(var, ceil_v, 0);
        let up = self.explore_branch(int_vars, depth, nodes)?;
        self.simplex.pop();
        match up {
            BranchOutcome::Sat => return Ok(TheoryResult::Sat),
            BranchOutcome::Unknown => saw_unknown = true,
            BranchOutcome::Infeasible => {}
        }

        // Neither branch produced Sat.  If any branch was left unresolved we must
        // answer Unknown; only when both branches are proven infeasible may we
        // conclude integer-infeasibility (Unsat).
        if saw_unknown {
            Ok(TheoryResult::Unknown)
        } else {
            Ok(TheoryResult::Unsat(self.full_unsat_core()))
        }
    }

    /// Explore the current (already-constrained) branch: re-solve the LP and, if
    /// feasible and not resource-limited, recurse into branch-and-bound.
    ///
    /// The caller is responsible for the surrounding `push`/`pop`.
    fn explore_branch(
        &mut self,
        int_vars: &[VarId],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<BranchOutcome> {
        match self.simplex.check() {
            Ok(()) => {
                if self.simplex.resource_limit_reached() {
                    // LP unresolved within the pivot budget — Unknown, not Sat.
                    Ok(BranchOutcome::Unknown)
                } else {
                    Ok(match self.bnb_recurse(int_vars, depth + 1, nodes)? {
                        TheoryResult::Sat => BranchOutcome::Sat,
                        TheoryResult::Unsat(_) => BranchOutcome::Infeasible,
                        _ => BranchOutcome::Unknown,
                    })
                }
            }
            // LP infeasible on this branch: a proven dead end.
            Err(_) => Ok(BranchOutcome::Infeasible),
        }
    }

    /// Tighten constraints for integer arithmetic
    ///
    /// Returns true if any tightening was performed
    pub fn tighten_constraints(&mut self) -> bool {
        if !self.is_integer {
            return false;
        }

        // In a full implementation, we would:
        // 1. Iterate through all bounds
        // 2. Apply tightening rules
        // 3. Propagate tightened bounds
        //
        // For now, tightening is applied during assertion
        false
    }
}

impl Theory for ArithSolver {
    fn id(&self) -> TheoryId {
        if self.is_integer {
            TheoryId::LIA
        } else {
            TheoryId::LRA
        }
    }

    fn name(&self) -> &str {
        if self.is_integer { "LIA" } else { "LRA" }
    }

    fn can_handle(&self, _term: TermId) -> bool {
        // In a full implementation, check if term is arithmetic
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        // In a full implementation, parse the term and add constraints
        let _ = self.intern(term);
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        let _ = self.intern(term);
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        // Any previously recorded integral model is stale for this fresh check.
        self.lia_model.clear();

        // Step 1: solve the LP (real) relaxation.
        match self.simplex.check() {
            Ok(()) => {
                // The pivot budget may have been exhausted without a definitive
                // answer.  In that case the assignment is NOT a model — report
                // Unknown rather than a fabricated Sat.
                if self.simplex.resource_limit_reached() {
                    return Ok(TheoryResult::Unknown);
                }
            }
            Err(reasons) => {
                // `reasons` and the simplex constraints that carry these ids are
                // pushed and popped together, so every id must resolve.  A miss
                // would silently shrink the core — and a conflict explanation
                // that loses one of its causes is not weaker, it is wrong — so
                // assert it loudly and, in release, fall back to the full set of
                // known reasons rather than a truncated one.
                let mut terms: Vec<TermId> = Vec::with_capacity(reasons.len());
                for &r in &reasons {
                    match self.reasons.get(r as usize).copied() {
                        Some(term) => terms.push(term),
                        None => {
                            debug_assert!(
                                false,
                                "simplex reported reason id {r} with no recorded term \
                                 (only {} known): the conflict core would lose a cause",
                                self.reasons.len()
                            );
                            return Ok(TheoryResult::Unsat(self.full_unsat_core()));
                        }
                    }
                }
                return Ok(TheoryResult::Unsat(terms));
            }
        }

        // Step 2 (LRA): the LP relaxation is exact — feasible LP ⇒ Sat.  An
        // `Int` term among the variables makes it the relaxation of a
        // mixed-integer problem; its integrality is decided once, at the end
        // of a search, by `check_integrality` (the `mixed` module).
        if !self.is_integer {
            return Ok(TheoryResult::Sat);
        }

        // Step 3 (LIA): the LP relaxation being feasible is NOT sufficient — a
        // fractional assignment over Int variables must be resolved by
        // branch-and-bound before we may answer Sat.  Otherwise integer-
        // infeasible-but-LP-feasible systems (e.g. y = 2x ∧ y = 2z+1) would be
        // wrongly reported Sat with fractional values for Int terms.
        self.lia_branch_and_bound()
    }

    fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_vars: self.var_to_term.len(),
            num_reasons: self.reasons.len(),
            num_shared_equalities: self.shared_equalities.len(),
            num_int_equalities: self.int_equalities.len(),
        });
        self.simplex.push();
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // Roll back the term→var interning done since the matching push.
            //
            // `var_to_term` is the intern trail: `intern` appends exactly one
            // entry per fresh variable and the pushed index equals the VarId
            // (`simplex.new_var()` returns the current array length). So the
            // terms interned inside this scope are precisely the tail beyond
            // `state.num_vars`. Draining that tail and removing each term from
            // `term_to_var` keeps the two maps consistent in O(delta).
            //
            // This is load-bearing for correctness: `simplex.pop()` recycles
            // VarIds (it shrinks its per-variable arrays), so a `term_to_var`
            // entry left dangling here would make `intern` replay a stale index
            // that now belongs to a different (or not-yet-created) variable.
            let cut = state.num_vars.min(self.var_to_term.len());
            let removed: Vec<TermId> = self.var_to_term.drain(cut..).collect();
            for term in removed {
                self.term_to_var.remove(&term);
            }
            self.reasons.truncate(state.num_reasons);
            self.reason_counter = state.num_reasons as u32;
            self.shared_equalities.truncate(state.num_shared_equalities);
            self.int_equalities.truncate(state.num_int_equalities);
            // The LIA branch-and-bound model is a snapshot of the *last* check's
            // integral assignment, keyed by VarId. Because VarIds are recycled
            // across this pop, a leftover entry could be misread by `value()`
            // for a freshly interned term that reuses the index before the next
            // `check()` repopulates it. It is only valid immediately after a
            // successful `check()`, so drop it on backtrack.
            self.lia_model.clear();
            self.simplex.pop();
        }
    }

    fn reset(&mut self) {
        self.simplex.reset();
        self.term_to_var.clear();
        self.var_to_term.clear();
        self.reason_counter = 0;
        self.reasons.clear();
        self.context_stack.clear();
        self.shared_equalities.clear();
        self.lia_model.clear();
        self.int_equalities.clear();
        // `int_terms` stays: a term's sort never changes, and the solver's
        // rebases (`Solver::rebase_theory_state`) reset this solver and
        // replay the asserted constraints without registering their terms
        // again.
    }

    fn get_model(&self) -> Vec<(TermId, TermId)> {
        // Return variable -> value pairs
        // In a full implementation, we'd create value terms
        Vec::new()
    }
}

impl TheoryCombination for ArithSolver {
    fn notify_equality(&mut self, eq: EqualityNotification) -> bool {
        // Check if both terms are relevant to arithmetic
        let lhs_var = self.term_to_var.get(&eq.lhs).copied();
        let rhs_var = self.term_to_var.get(&eq.rhs).copied();

        if let (Some(lhs), Some(rhs)) = (lhs_var, rhs_var) {
            // Enforce lhs = rhs in the simplex by asserting lhs - rhs <= 0 and rhs - lhs <= 0.
            // This is equivalent to lhs - rhs = 0, i.e., add_eq(lhs - rhs, 0).
            let reason_id = if let Some(r) = eq.reason {
                self.add_reason(r)
            } else {
                self.add_reason(eq.lhs)
            };

            // Build expression: lhs - rhs
            let mut expr_le = LinExpr::new();
            expr_le.add_term(lhs, Rational64::one());
            expr_le.add_term(rhs, -Rational64::one());
            // lhs - rhs <= 0
            self.simplex.add_le(expr_le, reason_id);

            // Build expression: rhs - lhs
            let mut expr_ge = LinExpr::new();
            expr_ge.add_term(rhs, Rational64::one());
            expr_ge.add_term(lhs, -Rational64::one());
            // rhs - lhs <= 0  (i.e., lhs - rhs >= 0)
            self.simplex.add_le(expr_ge, reason_id);

            // Record so that get_shared_equalities can return it
            self.shared_equalities.push(eq);

            true
        } else {
            // Terms not relevant to this arithmetic solver
            false
        }
    }

    fn get_shared_equalities(&self) -> Vec<EqualityNotification> {
        // Sound Nelson-Oppen propagation (model-based + entailment verification).
        //
        // Algorithm:
        // a) Collect interface variables (those mapped from interned terms).
        // b) Group by current delta_value in the simplex model — same-valued vars
        //    are candidates for equality.
        // c) For each adjacent same-bucket pair (x, y):
        //    i)  Probe: push, add x - y < 0 (strict), check → if UNSAT then
        //        "x < y" is infeasible → entailed_ge holds.
        //    ii) Probe: push, add y - x < 0 (strict), check → if UNSAT then
        //        "x > y" is infeasible → entailed_le holds.
        //    iii) Emit equality only if BOTH probes are UNSAT.
        // d) Also include equalities accumulated via notify_equality.

        // We need a mutable borrow on the simplex for probing, so we collect
        // results in a separate step.  Use an immutable reference for reading
        // variable assignments first, then do mutable probing.

        // Need &mut self for probing; but the trait signature is &self.
        // We work around this by cloning the accumulated `shared_equalities` and
        // returning them — the model-based probing path requires &mut self, so we
        // use an internal helper that takes &mut ArithSolver.
        self.shared_equalities.clone()
    }

    fn is_relevant(&self, term: TermId) -> bool {
        // Check if this term has been interned in the arithmetic solver
        self.term_to_var.contains_key(&term)
    }
}

impl ArithSolver {
    /// Sound Nelson-Oppen equality propagation.
    ///
    /// Returns entailed equalities between interface terms that are shared between
    /// this arithmetic theory and other theories in the Nelson-Oppen combination.
    ///
    /// Only emits `x = y` if BOTH `x < y` and `x > y` are infeasible in the
    /// current simplex state — this guarantees soundness: no false equality is
    /// ever propagated.
    ///
    /// Uses probe-and-pop to avoid permanently modifying the simplex state.
    pub fn derive_shared_equalities(&mut self) -> Vec<EqualityNotification> {
        let num_interface_terms = self.var_to_term.len();
        if num_interface_terms < 2 {
            return self.shared_equalities.clone();
        }

        // Collect (delta_value, VarId, TermId) for all interned variables.
        let mut candidates: Vec<(super::delta::DeltaRational, VarId, TermId)> = self
            .var_to_term
            .iter()
            .enumerate()
            .filter_map(|(idx, &term)| {
                // term_to_var maps TermId → VarId; we stored in var_to_term in order
                let var = self.term_to_var.get(&term).copied()?;
                let _ = idx; // suppress warning
                let dval = self.simplex.delta_value(var);
                Some((dval, var, term))
            })
            .collect();

        if candidates.len() < 2 {
            return self.shared_equalities.clone();
        }

        // Sort by current assignment value so same-valued pairs are adjacent.
        candidates.sort_by_key(|a| a.0);

        let mut result = self.shared_equalities.clone();

        // Check adjacent same-bucket pairs.
        let mut i = 0;
        while i < candidates.len() {
            // Find end of this bucket (same delta_value)
            let bucket_start = i;
            while i < candidates.len() && candidates[i].0 == candidates[bucket_start].0 {
                i += 1;
            }
            let bucket = &candidates[bucket_start..i];

            // For each adjacent pair in the bucket, probe for entailment.
            for pair_idx in 0..bucket.len().saturating_sub(1) {
                let (_, var_x, term_x) = bucket[pair_idx];
                let (_, var_y, term_y) = bucket[pair_idx + 1];

                // Probe 1: Can x < y? (i.e., x - y < 0)
                // If UNSAT → x >= y is entailed (x cannot be strictly less than y).
                let entailed_ge = {
                    self.simplex.push();
                    // Add strict x - y < 0
                    let mut expr = LinExpr::new();
                    expr.add_term(var_x, Rational64::one());
                    expr.add_term(var_y, -Rational64::one());
                    self.simplex.add_strict_lt(expr, 0);
                    let infeasible = self.simplex.check().is_err();
                    self.simplex.pop();
                    infeasible
                };

                // Probe 2: Can x > y? (i.e., y - x < 0)
                // If UNSAT → x <= y is entailed (x cannot be strictly greater than y).
                let entailed_le = {
                    self.simplex.push();
                    // Add strict y - x < 0
                    let mut expr = LinExpr::new();
                    expr.add_term(var_y, Rational64::one());
                    expr.add_term(var_x, -Rational64::one());
                    self.simplex.add_strict_lt(expr, 0);
                    let infeasible = self.simplex.check().is_err();
                    self.simplex.pop();
                    infeasible
                };

                // Both strict directions infeasible → x = y is entailed.
                if entailed_ge && entailed_le {
                    // Avoid duplicates from shared_equalities.
                    let already_known = result.iter().any(|eq| {
                        (eq.lhs == term_x && eq.rhs == term_y)
                            || (eq.lhs == term_y && eq.rhs == term_x)
                    });
                    if !already_known {
                        result.push(EqualityNotification {
                            lhs: term_x,
                            rhs: term_y,
                            reason: None,
                        });
                    }
                }
            }
        }

        result
    }
}

mod mixed;

#[cfg(test)]
mod tests;
