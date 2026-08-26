//! Nonlinear arithmetic (NLSAT/NIA/NRA) constraint checking
//!
//! This module implements early conflict detection for nonlinear arithmetic
//! constraints in QF_NIRA, QF_NIA, and QF_NRA benchmarks. It handles cases
//! where the main CDCL(T) loop with linear arithmetic cannot detect UNSAT
//! because the constraints involve nonlinear terms (e.g., x*x).
//!
//! ## Detected Patterns
//!
//! 1. `x^2 = c` where c < 0 → UNSAT (squares are non-negative)
//! 2. `x^2 = c` (integer x) where c is not a perfect square → UNSAT
//! 3. System contradictions: e.g., `sq > 0 ∧ sq + y = 0 ∧ y >= 0`
//!    (sq > 0 implies sq + y > 0 when y >= 0, contradicting sq + y = 0)

#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::{One, ToPrimitive, Zero};
use oxiz_core::ast::{TermId, TermKind, TermManager};
// The NIA-over-LP relaxation engine. Note the path: `arithmetic::nla` is
// `std`-gated, unlike `oxiz_theories::nlsat` below, so this import needs no
// `cfg` and the engine is present in every build this file compiles in.
use oxiz_theories::arithmetic::nla::{self, NlaConfig, NlaVerdict};
use oxiz_theories::nl_eval::{Interpretation, holds_under};
#[cfg(feature = "nlsat")]
use oxiz_theories::nlsat::{NlDispatchResult, dispatch_nia_constraints, dispatch_nra_constraints};
use smallvec::SmallVec;

use super::Solver;
use super::types::{Model, SolverResult};

/// Narrow an exact rational witness to the fixed-width rational the term
/// language stores, or `None` when it does not fit.
///
/// Refusing is the point: a `Real` value that has no exact `Rational64` form
/// must not be rounded into one, because the rounded value would then be
/// reported by `(get-value ...)` as *the* solution while satisfying none of
/// the constraints the real one did.
fn narrow_rational(value: &num_rational::BigRational) -> Option<num_rational::Rational64> {
    use num_traits::ToPrimitive;
    let numerator = value.numer().to_i64()?;
    let denominator = value.denom().to_i64()?;
    (denominator != 0).then(|| num_rational::Rational64::new(numerator, denominator))
}

/// A polynomial atom extracted from an assertion.
/// Represents: `coeff * square_term OP constant`
/// where `square_term` is a term of the form `x * x` (or product of identical terms).
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum NlAtom {
    /// `sq_term = const` — the square term equals a constant
    SqEq {
        sq_term: TermId,
        val: Rational64,
        is_integer_sort: bool,
    },
    /// `sq_term > 0`
    SqGtZero { sq_term: TermId },
    /// `sq_term >= 0`
    SqGeZero { sq_term: TermId },
    /// `sq_term + linear_coeff * other_var = const`
    /// i.e., `sq + coeff * v = c`
    SqPlusLinearEq {
        sq_term: TermId,
        sq_coeff: Rational64,
        linear_var: TermId,
        linear_coeff: Rational64,
        rhs: Rational64,
    },
    /// `linear_var >= const`
    LinearGe { var: TermId, bound: Rational64 },
    /// `linear_var > const`
    LinearGt { var: TermId, bound: Rational64 },
}

impl Solver {
    /// Dispatch nonlinear arithmetic assertions to the full NIA/NRA polynomial
    /// solver.
    ///
    /// Translates all top-level assertions to polynomial form and runs either
    /// `NiaSolver` (integer) or `NlsatSolver` (real). Returns a definitive
    /// `SolverResult` when the solver is conclusive, or `None` to fall
    /// through to CDCL(T).
    ///
    /// Handles:
    /// - `x * y`, `x * y * z` (products of distinct variables)
    /// - `x * x` (squares / higher powers via repeated multiplication)
    /// - `(x + 1) * (y - 2)` (products of linear expressions)
    ///
    /// # The model
    ///
    /// A `Sat` from these procedures arrives with the witness that justifies
    /// it, and [`Self::adopt_nl_witness`] turns that witness into the
    /// `Solver`'s own model so a following `(get-model)` / `(get-value ...)`
    /// can answer. The verdict does not depend on whether that succeeds: the
    /// procedures' own trust conditions decide `Sat`, and a witness this
    /// solver cannot represent (a rational too wide for the value type)
    /// simply leaves the model unset, which is exactly the behaviour before
    /// witnesses existed. What must never happen — and does not — is
    /// publishing a model that has not been re-checked against the assertions
    /// it claims to satisfy.
    ///
    /// One kind of witness bypasses `adopt_nl_witness` entirely rather than
    /// being declined by it: an **irrational** NRA root. `√2` has no term in
    /// the rational term language a [`Model`] is written in, so the real cell
    /// decomposition reports it through
    /// [`Solver::nl_algebraic_values`](Self::nl_algebraic_values) instead —
    /// the exact-value side-channel `(get-model)` renders as SMT-LIB
    /// `root-obj`. That channel is filled *instead of* the rational witness
    /// and covers the whole problem when it is filled at all, so it does not
    /// weaken the re-check above: there is simply nothing rational to
    /// re-check, and the point it carries was already verified against every
    /// assigned atom by the procedure that produced it.
    ///
    /// # Without the `nlsat` feature
    ///
    /// The cell-decomposition step below is compiled out — it is the only
    /// caller of `oxiz_theories::nlsat`, which is the only route to the
    /// `oxiz-nlsat` crate. Everything else in this function is unchanged. The
    /// three components under it are `std`-gated rather than `nlsat`-gated and
    /// so are present in *both* builds:
    ///
    /// * the **NIA-over-LP relaxation engine**
    ///   (`oxiz_theories::arithmetic::nla`), which answers `unsat` from an LP
    ///   infeasibility proof and `sat` from a witness re-verified here;
    /// * the two **model searches** (`nl_repair_search`, `nl_ground_reduce`),
    ///   which answer `sat` or nothing.
    ///
    /// All three are gated on QF_NIA, so the loss is not symmetric between the
    /// two logics:
    ///
    /// * **QF_NIA** answers identically in both builds on this tree. The
    ///   relaxation engine and the searches still run, and each `sat` is still
    ///   a witness re-checked against the untouched assertions; the `unsat`s
    ///   come from that engine's proofs and from `check_core`'s
    ///   `check_nonlinear_constraints` patterns, neither of which is affected.
    /// * **QF_NRA** loses every nonlinear verdict. There is nothing left below
    ///   to reach — the relaxation engine declines a Real-sorted variable
    ///   outright, since its case splits (`x ≤ -1 ∨ x = 0 ∨ x ≥ 1`) are
    ///   tautologies over `Z` and not over `R` — so a goal that needed a cell
    ///   decomposition, including one that is provably `unsat` such as
    ///   `x*x < 0`, is conceded.
    ///
    /// Whatever this function declines to decide meets `check_core`'s
    /// `arith_atoms_need_theory` gate and is answered `unknown` — never
    /// guessed by the SAT layer. `oxiz-solver/tests/nlsat_feature_gate.rs`
    /// pins each of those claims in the build it applies to.
    pub(super) fn dispatch_nl_solver(&mut self, manager: &mut TermManager) -> Option<SolverResult> {
        let logic = self.logic.as_deref()?;

        let is_nia = logic.contains("NIA") || (logic.contains("NIRA") && !logic.contains("NRA"));
        let is_nra = logic.contains("NRA") && !is_nia;

        if !is_nia && !is_nra {
            return None;
        }

        // The algebraic side-channel belongs to whichever procedure below
        // answers *this* call. `invalidate_results` already drops it whenever
        // the assertion stack moves, but a repeated `check` on an unchanged
        // stack does not go through that hook, and neither does a `check`
        // whose verdict this time comes from the relaxation engine or a
        // search rather than from the cell decomposition. Clearing on entry
        // makes "populated" mean "populated by the procedure that just
        // answered", with no path that inherits.
        self.nl_algebraic_values.clear();

        #[cfg(feature = "nlsat")]
        {
            let dispatched = if is_nia {
                dispatch_nia_constraints(&self.assertions, manager, true)
            } else {
                dispatch_nra_constraints(&self.assertions, manager)
            };

            if let Some(dispatched) = dispatched {
                return match dispatched {
                    NlDispatchResult::Sat { witness, algebraic } => {
                        // The exact-value channel, for a real model that has
                        // no rational form at all (`x² = 2`). It and
                        // `witness` are alternatives — the dispatcher fills
                        // exactly one, and the one it fills is complete — so
                        // taking it wholesale here neither blends two model
                        // sources nor leaves a hole for `(get-model)` to fill
                        // in from sort defaults. Nothing needs re-checking:
                        // the point was verified against every assigned atom
                        // inside `oxiz-nlsat`, and no gate in this crate
                        // reads these values (see the field docs).
                        //
                        // That exclusivity is a contract of
                        // `NlDispatchResult`, and `Context::get_model`
                        // depends on it: it consults the side-channel *before*
                        // the `Model`, so two populated channels disagreeing
                        // about a constant would silently publish the
                        // algebraic one and hide the rational one. Cheap to
                        // check, and a debug build should not let a future
                        // dispatcher quietly break it.
                        debug_assert!(
                            algebraic.is_empty() || witness.num_count() == 0,
                            "the nonlinear dispatcher populated both witness \
                             channels for one Sat; they are alternatives, not \
                             layers (see NlDispatchResult)"
                        );
                        self.nl_algebraic_values = algebraic;
                        // The verdict is the dispatcher's, decided by its own trust
                        // conditions; installing a model is a separate, best-effort
                        // step that may decline a witness it cannot represent.
                        let adopted = self.adopt_nl_witness(&witness, manager);
                        // Declining for representational reasons (an Int-sorted
                        // term with a fractional witness, a real with no exact
                        // narrow form) is ordinary. Declining because the witness
                        // does not *satisfy the assertions* is not: the dispatcher
                        // has just claimed `Sat` on the strength of that very
                        // witness, so a re-check that disagrees means one of the
                        // two is wrong. Release behaviour is unchanged — the
                        // verdict stands, modelless — but a debug build must not
                        // let a signal that strong pass in silence.
                        //
                        // The `num_count() == 0` arm is load-bearing for QF_NRA
                        // real-algebraic models, and is guaranteed rather than
                        // hoped for. A cell decomposition can witness `x² = 2`
                        // only with `x = √2`, which has no rational form to pin
                        // here, so `oxiz_theories::nlsat`'s
                        // `witness_from_real_translator` is all-or-nothing: if
                        // any variable's value cannot be represented, it returns
                        // an *empty* interpretation rather than a partial one.
                        // A partial witness would evaluate to `false` under
                        // `holds_under` — through no fault of the dispatcher —
                        // and trip this assertion on a perfectly correct `sat`.
                        // That is exactly the case the `algebraic` map above
                        // now covers: `witness` stays empty and the model is
                        // reported from the side-channel instead, so this arm
                        // keeps holding and the assertion keeps its teeth for
                        // the rational witnesses it was written for.
                        debug_assert!(
                            adopted
                                || witness.num_count() == 0
                                || holds_under(&self.assertions, manager, &witness),
                            "the cell-decomposition dispatcher reported Sat with a witness \
                             that does not satisfy the assertions"
                        );
                        Some(SolverResult::Sat)
                    }
                    NlDispatchResult::Unsat => Some(SolverResult::Unsat),
                };
            }
        }

        // The cell-decomposition core had no verdict. Next comes the
        // NIA-over-LP relaxation engine, which *can* derive `unsat` -- unlike
        // the two searches below it.
        //
        // # Why it is slotted here and not earlier
        //
        // Placing it *after* the cell-decomposition block means it never sees a
        // goal that block decided: every verdict CAD reaches is returned above
        // without this code running at all. So no CAD *verdict* can move, which
        // is what makes the wiring a completeness gain rather than a parity
        // risk. Running it *before* CAD -- where it would be cheaper on the
        // many QF_NIA goals CAD grinds through -- is a recorded follow-up
        // experiment, and a real one: it would put two procedures' verdicts in
        // contention on goals both can decide, so it has to be measured against
        // the parity suite rather than assumed equivalent.
        //
        // What this ordering does *not* buy is model stability, and the
        // distinction matters. This engine sits above the two searches below,
        // so on a goal CAD declined but a search would have solved, the engine
        // now answers first and reports *its* witness. Both are re-verified and
        // both satisfy the assertions, but they need not be the same
        // assignment -- measured on this tree, `x*y = 6 ∧ x+y = 5` reports
        // (2, 3) in the default build and (3, 2) in the no-`nlsat` build, where
        // CAD is absent and the engine is what answers. A test that pins an
        // exact `(get-value ...)` string for a multi-solution QF_NIA goal is
        // therefore pinning an implementation detail, not a contract; the ones
        // in `tests/qf_nia_relaxation.rs` accept any correct root deliberately.
        //
        // # Why both verdicts are safe to take
        //
        // `Unsat` is proof-backed: `nla` produces it only from an LP
        // infeasibility closure over constraints each of which is a
        // consequence over `Z` of the input, with exhaustive case splits (see
        // that module's soundness contract). A dropped conjunct only ever
        // *weakens* the problem, so an infeasible relaxation still refutes the
        // original.
        //
        // `Sat` is advisory by that same contract, so it is not taken on
        // trust: the witness goes through `adopt_nl_witness`, which re-checks
        // it with `holds_under` against the untouched assertions in exact
        // `BigRational` arithmetic and installs a model only if it really
        // satisfies them. That is strictly stricter than the CAD path above,
        // which reports `Sat` on the dispatcher's own trust conditions and
        // treats the model install as best-effort. A witness that fails the
        // re-check here yields no verdict at all and simply falls through.
        //
        // This engine is `std`-gated, not `nlsat`-gated, so it is present in
        // *both* builds -- which is why the QF_NIA half of
        // `tests/nlsat_feature_gate.rs` stays uncfg'd.
        if is_nia && self.config.nonlinear_relaxation_engine {
            match nla::check_assertions(&self.assertions, manager, &NlaConfig::default()) {
                NlaVerdict::Unsat => return Some(SolverResult::Unsat),
                NlaVerdict::Sat(witness) => {
                    if self.adopt_nl_witness(&witness, manager) {
                        return Some(SolverResult::Sat);
                    }
                }
                NlaVerdict::Unknown => {}
            }
        }

        // Failing that, the search-based procedures. Both of them answer `Sat`
        // or nothing -- neither can derive `unsat` -- so running them here can
        // only turn an `unknown` into a `sat`, never change a verdict either of
        // the two procedures above had already reached. That is why they run
        // last, and why gating them off is a budget decision rather than a
        // soundness one.
        if !(is_nia && self.config.nonlinear_model_search) {
            return None;
        }
        let effort = oxiz_theories::nl_repair_search::Effort::default();
        let assertions = self.assertions.clone();

        // Model repair over the arithmetic as written.
        if let Some(witness) =
            oxiz_theories::nl_repair_search::find_integer_model(&assertions, manager, effort)
            && self.adopt_nl_witness(&witness, manager)
        {
            return Some(SolverResult::Sat);
        }

        // Failing that, the problem may simply have been invisible: an array
        // read or an uninterpreted application in an arithmetic position has
        // no polynomial translation, so the atom containing it was dropped and
        // what the engines solved was a weaker formula. The grammar reduction
        // abstracts those foreign terms into arithmetic unknowns and searches
        // the result. Its rewrite is not trusted -- only a witness verified
        // against the untouched assertions licenses the verdict.
        if let Some(witness) =
            oxiz_theories::nl_ground_reduce::find_ground_model(&assertions, manager, effort)
            && self.adopt_nl_witness(&witness, manager)
        {
            return Some(SolverResult::Sat);
        }
        None
    }

    /// Install `witness` as this solver's model, but only if it really is one.
    ///
    /// The witness is re-evaluated against every current assertion by
    /// [`oxiz_theories::nl_eval::holds_under`] — in exact `BigRational`
    /// arithmetic, from the leaves up, over the *original* assertion terms
    /// rather than any rewritten form. Only a complete positive result
    /// installs anything; a partial witness (one that leaves a leaf the
    /// assertions mention unassigned) fails that check and leaves the model
    /// unset rather than reporting values that do not add up.
    ///
    /// Values are stored as ordinary constant terms, so the existing
    /// `(get-value ...)` path reads them with no special case. A value outside
    /// [`Rational64`]'s range has no such term and is skipped — which makes
    /// the whole install fail the completeness check below, again leaving no
    /// model rather than a partial one.
    ///
    /// Returns whether a model was installed, which a caller may use as its
    /// licence to report `Sat`. Reading `self.model.is_some()` instead would
    /// be wrong: that field can still hold the previous check's model, so a
    /// witness rejected here would inherit someone else's verdict.
    fn adopt_nl_witness(&mut self, witness: &Interpretation, manager: &mut TermManager) -> bool {
        if witness.num_count() == 0 {
            return false;
        }
        if !holds_under(&self.assertions, manager, witness) {
            return false;
        }
        let int_sort = manager.sorts.int_sort;
        let mut model = Model::new();
        for (term, value) in witness.numeric_entries() {
            let is_integer_sorted = manager.get(term).is_some_and(|t| t.sort == int_sort);
            let value_term = if is_integer_sorted {
                if !value.is_integer() {
                    return false;
                }
                manager.mk_int(value.to_integer())
            } else {
                let Some(exact) = narrow_rational(value) else {
                    return false;
                };
                manager.mk_real(exact)
            };
            model.set(term, value_term);
        }
        for (term, value) in witness.truth_entries() {
            let value_term = if value {
                manager.mk_true()
            } else {
                manager.mk_false()
            };
            model.set(term, value_term);
        }
        self.model = Some(model);
        true
    }

    /// Check nonlinear arithmetic constraints for early UNSAT detection.
    ///
    /// Returns `true` if the constraint set is detected as UNSAT.
    pub(super) fn check_nonlinear_constraints(&self, manager: &TermManager) -> bool {
        // Only run for NIA/NRA logics
        let is_nl = self
            .logic
            .as_deref()
            .map(|l| l.contains("NIA") || l.contains("NRA") || l.contains("NIRA"))
            .unwrap_or(false);

        if !is_nl {
            return false;
        }

        // Collect nonlinear atoms from all top-level assertions
        let mut atoms: Vec<NlAtom> = Vec::new();
        for &assertion in &self.assertions {
            self.collect_nl_atoms(assertion, manager, &mut atoms);
        }

        if atoms.is_empty() {
            return false;
        }

        // Check pattern 1: x^2 = c where c < 0 (never has a real solution)
        for atom in &atoms {
            if let NlAtom::SqEq { val, .. } = atom {
                if *val < Rational64::zero() {
                    return true;
                }
            }
        }

        // Check pattern 2: x^2 = c where c is not a perfect square (integer context)
        for atom in &atoms {
            if let NlAtom::SqEq {
                val,
                is_integer_sort,
                ..
            } = atom
            {
                if *is_integer_sort && *val >= Rational64::zero() {
                    if let Some(n) = val.to_i64() {
                        if n >= 0 && !is_perfect_square(n as u64) {
                            return true;
                        }
                    }
                }
            }
        }

        // Check pattern 3: system contradictions involving squares.
        //
        // Look for triples:
        //   (A) sq_term > 0                    [or sq_term >= 1 in integer case]
        //   (B) sq_term * a + var * b = c      [sum constraint]
        //   (C) var >= d                        [lower bound on var]
        //
        // where sq > 0 and b * var = c - a * sq, so var = (c - a*sq) / b.
        // Combined with var >= d: (c - a*sq)/b >= d.
        // If sq > 0 (sq >= 1 for int, sq > 0 for real) and a > 0, then
        // a*sq >= a (int) or a*sq > 0 (real), so c - a*sq < c (for positive a).
        // When d = 0 (y >= 0) and c = 0: c - a*sq = -a*sq <= -a < 0,
        // but we need var >= 0 — contradiction.
        //
        // Concretely, check:
        //   sq > 0  AND  sq + v = 0  AND  v >= 0
        // → v = -sq < 0  contradicts  v >= 0
        if self.check_sq_sum_bound_contradiction(&atoms) {
            return true;
        }

        false
    }

    /// Check for the "sq > 0 AND sq + v = 0 AND v >= 0" type contradiction.
    fn check_sq_sum_bound_contradiction(&self, atoms: &[NlAtom]) -> bool {
        // Build sets for quick lookup
        let sq_gt_zero: Vec<TermId> = atoms
            .iter()
            .filter_map(|a| {
                if let NlAtom::SqGtZero { sq_term } = a {
                    Some(*sq_term)
                } else {
                    None
                }
            })
            .collect();

        // For each "sq + coeff * var = rhs" constraint, check if we have sq > 0
        // and var >= -rhs/coeff is violated
        for atom in atoms {
            let NlAtom::SqPlusLinearEq {
                sq_term,
                sq_coeff,
                linear_var,
                linear_coeff,
                rhs,
            } = atom
            else {
                continue;
            };

            // Only handle the case where both sq_coeff and linear_coeff are non-zero
            if sq_coeff.is_zero() || linear_coeff.is_zero() {
                continue;
            }

            // Check if sq_term is known to be > 0
            let sq_positive = sq_gt_zero.contains(sq_term);
            if !sq_positive {
                continue;
            }

            // From: sq_coeff * sq + linear_coeff * var = rhs
            // → var = (rhs - sq_coeff * sq) / linear_coeff
            // If sq > 0 (at least epsilon > 0):
            // For real: sq > 0, so sq_coeff * sq > 0 when sq_coeff > 0
            //   → rhs - sq_coeff * sq < rhs
            //   → var < rhs / linear_coeff  (when linear_coeff > 0)
            //   OR var > rhs / linear_coeff  (when linear_coeff < 0)

            // The var = (rhs - sq_coeff * sq) / linear_coeff must satisfy
            // any lower bounds we have on var.
            let var_expr_at_sq_zero = *rhs / *linear_coeff; // value of var if sq=0

            // The sign of d(var)/d(sq) = -sq_coeff / linear_coeff
            // If sq increases from 0 (since sq > 0), var moves in direction -sq_coeff/linear_coeff

            // Check against all >= bounds on linear_var
            for bound_atom in atoms {
                let bound = match bound_atom {
                    NlAtom::LinearGe { var, bound } if *var == *linear_var => bound,
                    _ => continue,
                };

                // We need: var >= bound
                // From the sum constraint, as sq→0+, var→var_expr_at_sq_zero
                // If the sum constraint requires var < bound for all sq > 0,
                // that contradicts var >= bound.

                // Direction: d(var)/d(sq) = -sq_coeff / linear_coeff
                let deriv_sign = -(*sq_coeff) / *linear_coeff;

                // If deriv_sign < 0, then as sq increases (sq > 0), var decreases.
                // At sq = 0: var = var_expr_at_sq_zero
                // For all sq > 0: var < var_expr_at_sq_zero
                // If var_expr_at_sq_zero <= bound, then for sq > 0: var < bound — contradiction with var >= bound.

                if deriv_sign < Rational64::zero() && var_expr_at_sq_zero <= *bound {
                    return true;
                }

                // If deriv_sign > 0, then as sq increases (sq > 0), var increases.
                // The infimum is at sq = 0 (var → var_expr_at_sq_zero from above).
                // For all sq > 0: var > var_expr_at_sq_zero.
                // If var_expr_at_sq_zero >= bound, no contradiction from this alone.
                // But if we also have an upper bound on var that forces a contradiction...
                // For now, skip this case.
            }

            // Also check against strict lower bounds (LinearGt)
            for bound_atom in atoms {
                let bound = match bound_atom {
                    NlAtom::LinearGt { var, bound } if *var == *linear_var => bound,
                    _ => continue,
                };

                let deriv_sign = -(*sq_coeff) / *linear_coeff;

                // If deriv_sign < 0, as sq > 0: var < var_expr_at_sq_zero
                // Contradiction if var_expr_at_sq_zero <= bound (need var > bound, but var < bound)
                if deriv_sign < Rational64::zero() && var_expr_at_sq_zero <= *bound {
                    return true;
                }
            }
        }

        false
    }

    /// Collect nonlinear atoms from a term (top-level assertion).
    ///
    /// Iterative over the `and`-nesting (the only structure descended into),
    /// with a visited set so shared conjuncts of the hash-consed DAG
    /// contribute their atoms once; every downstream consumer performs pure
    /// existence checks over the collected atoms, so dropping duplicates
    /// preserves the verdict exactly.
    fn collect_nl_atoms(&self, term_id: TermId, manager: &TermManager, atoms: &mut Vec<NlAtom>) {
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = vec![term_id];
        while let Some(term_id) = stack.pop() {
            if !visited.insert(term_id) {
                continue;
            }
            let Some(term) = manager.get(term_id) else {
                continue;
            };

            match &term.kind {
                TermKind::Eq(lhs, rhs) => {
                    self.extract_nl_eq(*lhs, *rhs, manager, atoms);
                }
                TermKind::Gt(lhs, rhs) => {
                    // lhs > rhs  i.e. lhs - rhs > 0
                    self.extract_nl_comparison(*lhs, *rhs, CompOp::Gt, manager, atoms);
                }
                TermKind::Ge(lhs, rhs) => {
                    self.extract_nl_comparison(*lhs, *rhs, CompOp::Ge, manager, atoms);
                }
                TermKind::Lt(lhs, rhs) => {
                    // lhs < rhs  →  rhs > lhs
                    self.extract_nl_comparison(*rhs, *lhs, CompOp::Gt, manager, atoms);
                }
                TermKind::Le(lhs, rhs) => {
                    // lhs <= rhs  →  rhs >= lhs
                    self.extract_nl_comparison(*rhs, *lhs, CompOp::Ge, manager, atoms);
                }
                TermKind::And(args) => {
                    // Reversed push keeps the original left-to-right visit
                    // order (and with it the order of `atoms`).
                    stack.extend(args.iter().rev().copied());
                }
                _ => {}
            }
        }
    }

    /// Extract atoms from an equality `lhs = rhs`.
    fn extract_nl_eq(
        &self,
        lhs: TermId,
        rhs: TermId,
        manager: &TermManager,
        atoms: &mut Vec<NlAtom>,
    ) {
        // Try: is lhs a pure square (x * x) and rhs a constant?
        if let Some((sq_term, sq_coeff, is_int)) = self.extract_pure_square(lhs, manager) {
            if let Some(rhs_val) = self.extract_rational_const(rhs, manager) {
                // sq_coeff * sq_term = rhs_val  →  sq_term = rhs_val / sq_coeff
                if !sq_coeff.is_zero() {
                    let val = rhs_val / sq_coeff;
                    atoms.push(NlAtom::SqEq {
                        sq_term,
                        val,
                        is_integer_sort: is_int,
                    });
                    return;
                }
            }
        }

        // Try reversed: rhs is pure square, lhs is constant
        if let Some((sq_term, sq_coeff, is_int)) = self.extract_pure_square(rhs, manager) {
            if let Some(lhs_val) = self.extract_rational_const(lhs, manager) {
                if !sq_coeff.is_zero() {
                    let val = lhs_val / sq_coeff;
                    atoms.push(NlAtom::SqEq {
                        sq_term,
                        val,
                        is_integer_sort: is_int,
                    });
                    return;
                }
            }
        }

        // Try: lhs = Add(...) where the Add contains a square term plus a linear var
        // Pattern: (* x x) + y = const  or  y + (* x x) = const
        self.extract_nl_sum_eq(lhs, rhs, manager, atoms);
        self.extract_nl_sum_eq(rhs, lhs, manager, atoms);
    }

    /// Extract "sq_term + linear_var = rhs" from a sum equality.
    fn extract_nl_sum_eq(
        &self,
        sum_side: TermId,
        const_side: TermId,
        manager: &TermManager,
        atoms: &mut Vec<NlAtom>,
    ) {
        let Some(rhs_val) = self.extract_rational_const(const_side, manager) else {
            return;
        };

        let Some(sum_term) = manager.get(sum_side) else {
            return;
        };

        let TermKind::Add(args) = &sum_term.kind else {
            return;
        };

        // Try to identify: one arg is a pure square, the rest are linear vars
        let mut sq_term_opt: Option<(TermId, Rational64)> = None;
        let mut linear_term_opt: Option<(TermId, Rational64)> = None;
        let mut ok = true;

        for &arg in args {
            if let Some((sq_term, sq_coeff, _)) = self.extract_pure_square(arg, manager) {
                if sq_term_opt.is_some() {
                    ok = false;
                    break;
                }
                sq_term_opt = Some((sq_term, sq_coeff));
            } else if let Some((var, coeff)) = self.extract_linear_var(arg, manager) {
                if linear_term_opt.is_some() {
                    ok = false;
                    break;
                }
                linear_term_opt = Some((var, coeff));
            } else {
                ok = false;
                break;
            }
        }

        if !ok {
            return;
        }

        if let (Some((sq_term, sq_coeff)), Some((linear_var, linear_coeff))) =
            (sq_term_opt, linear_term_opt)
        {
            atoms.push(NlAtom::SqPlusLinearEq {
                sq_term,
                sq_coeff,
                linear_var,
                linear_coeff,
                rhs: rhs_val,
            });
        }
    }

    /// Extract atoms from a comparison `lhs OP 0` or `lhs OP rhs`.
    fn extract_nl_comparison(
        &self,
        lhs: TermId,
        rhs: TermId,
        op: CompOp,
        manager: &TermManager,
        atoms: &mut Vec<NlAtom>,
    ) {
        // Check if lhs is a pure square and rhs is a constant.
        // After normalization: sq_term OP (rhs_val / sq_coeff)
        if let Some((sq_term, sq_coeff, _)) = self.extract_pure_square(lhs, manager) {
            if let Some(rhs_val) = self.extract_rational_const(rhs, manager) {
                if !sq_coeff.is_zero() {
                    // sq_coeff * sq_term OP rhs_val
                    // → sq_term OP rhs_val/sq_coeff  (flip op if sq_coeff < 0)
                    let normalized = rhs_val / sq_coeff;
                    let effective_op = if sq_coeff < Rational64::zero() {
                        op.flip()
                    } else {
                        op
                    };
                    match effective_op {
                        CompOp::Gt => {
                            if normalized < Rational64::zero() {
                                // sq > negative → always true, not useful
                            } else if normalized.is_zero() {
                                atoms.push(NlAtom::SqGtZero { sq_term });
                            }
                        }
                        CompOp::Ge => {
                            if normalized <= Rational64::zero() {
                                atoms.push(NlAtom::SqGeZero { sq_term });
                            }
                        }
                    }
                    return;
                }
            }
        }

        // Check if this is a simple linear comparison: var OP const
        if let Some((var, coeff)) = self.extract_linear_var(lhs, manager) {
            if let Some(rhs_val) = self.extract_rational_const(rhs, manager) {
                if !coeff.is_zero() {
                    // coeff * var OP rhs_val
                    // → var OP rhs_val/coeff (flip op if coeff < 0)
                    let bound = rhs_val / coeff;
                    let effective_op = if coeff < Rational64::zero() {
                        op.flip()
                    } else {
                        op
                    };
                    match effective_op {
                        CompOp::Gt => atoms.push(NlAtom::LinearGt { var, bound }),
                        CompOp::Ge => atoms.push(NlAtom::LinearGe { var, bound }),
                    }
                }
                return;
            }
        }

        // Also handle reversed (const OP lhs → lhs OP' const) but skip for now
        // since the benchmark uses canonical form (lhs > 0, var >= 0)
        let _ = (lhs, rhs, op, manager, atoms);
    }

    /// Extract a pure square: a Mul term where all factors are the same variable.
    /// Returns `(representative_var_term, coefficient, is_integer_sort)` or None.
    ///
    /// Handles patterns like:
    /// - `(* x x)` → Some((x_term, 1, is_int))
    /// - `(* 2 x x)` → Some((x_term, 2, is_int))  [if we ever see this]
    fn extract_pure_square(
        &self,
        term_id: TermId,
        manager: &TermManager,
    ) -> Option<(TermId, Rational64, bool)> {
        let term = manager.get(term_id)?;

        match &term.kind {
            TermKind::Mul(args) => {
                let mut const_coeff = Rational64::one();
                let mut var_factors: Vec<TermId> = Vec::new();

                for &arg in args {
                    let arg_term = manager.get(arg)?;
                    match &arg_term.kind {
                        TermKind::IntConst(n) => {
                            let v = n.to_i64()?;
                            const_coeff *= Rational64::from_integer(v);
                        }
                        TermKind::RealConst(r) => {
                            const_coeff *= *r;
                        }
                        TermKind::Var(_) => {
                            var_factors.push(arg);
                        }
                        _ => return None, // nested expressions not handled
                    }
                }

                // Must have exactly 2 variable factors and they must be the same
                if var_factors.len() == 2 && var_factors[0] == var_factors[1] {
                    let v = var_factors[0];
                    let vt = manager.get(v)?;
                    let is_int = vt.sort == manager.sorts.int_sort;
                    Some((v, const_coeff, is_int))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract a simple linear variable term with coefficient.
    /// Returns `(var_term_id, coefficient)` or None.
    ///
    /// Handles:
    /// - `x` → Some((x, 1))
    /// - `(* c x)` → Some((x, c))
    ///
    /// Iterative: the only recursion was through `Neg` nesting, which is a
    /// simple sign-flipping unwrap loop.
    fn extract_linear_var(
        &self,
        term_id: TermId,
        manager: &TermManager,
    ) -> Option<(TermId, Rational64)> {
        let mut sign = Rational64::one();
        let mut current = term_id;
        loop {
            let term = manager.get(current)?;

            match &term.kind {
                TermKind::Neg(inner) => {
                    sign = -sign;
                    current = *inner;
                }
                TermKind::Var(_) => return Some((current, sign)),
                TermKind::Mul(args) => {
                    let mut const_coeff = Rational64::one();
                    let mut var_opt: Option<TermId> = None;

                    for &arg in args {
                        let arg_term = manager.get(arg)?;
                        match &arg_term.kind {
                            TermKind::IntConst(n) => {
                                let v = n.to_i64()?;
                                const_coeff *= Rational64::from_integer(v);
                            }
                            TermKind::RealConst(r) => {
                                const_coeff *= *r;
                            }
                            TermKind::Var(_) => {
                                if var_opt.is_some() {
                                    return None; // multiple vars → nonlinear
                                }
                                var_opt = Some(arg);
                            }
                            _ => return None,
                        }
                    }

                    return var_opt.map(|v| (v, sign * const_coeff));
                }
                _ => return None,
            }
        }
    }

    /// Extract a rational constant from a term.
    ///
    /// Handles:
    /// - `IntConst(n)` → n
    /// - `RealConst(r)` → r
    /// - `Neg(x)` → -extract(x)
    /// - `Sub(0, x)` → -extract(x)  [unary minus is parsed as Sub(0, x)]
    /// - `Sub(x, y)` → extract(x) - extract(y)
    /// - `Add(xs)` → Σ extract(xᵢ)
    ///
    /// Iterative (explicit frame stack), so arbitrarily deep constant
    /// expressions are folded without native recursion; any non-constant
    /// sub-term makes the whole extraction `None`, exactly as before.
    fn extract_rational_const(&self, term_id: TermId, manager: &TermManager) -> Option<Rational64> {
        /// A pending arithmetic operator waiting for operand values.
        enum ConstFrame {
            Neg,
            SubLhs {
                rhs: TermId,
            },
            SubRhs {
                lhs: Rational64,
            },
            Add {
                args: SmallVec<[TermId; 4]>,
                next: usize,
                acc: Rational64,
            },
        }

        let mut frames: Vec<ConstFrame> = Vec::new();
        let mut current = term_id;
        'open: loop {
            // Descend to a constant leaf.
            let mut value: Rational64 = loop {
                let term = manager.get(current)?;
                match &term.kind {
                    TermKind::IntConst(n) => {
                        let v = n.to_i64()?;
                        break Rational64::from_integer(v);
                    }
                    TermKind::RealConst(r) => break *r,
                    TermKind::Neg(inner) => {
                        frames.push(ConstFrame::Neg);
                        current = *inner;
                    }
                    TermKind::Sub(lhs, rhs) => {
                        frames.push(ConstFrame::SubLhs { rhs: *rhs });
                        current = *lhs;
                    }
                    TermKind::Add(args) => match args.first() {
                        Some(&first) => {
                            frames.push(ConstFrame::Add {
                                args: args.clone(),
                                next: 1,
                                acc: Rational64::zero(),
                            });
                            current = first;
                        }
                        None => break Rational64::zero(),
                    },
                    _ => return None,
                }
            };

            // Fold the leaf value into the pending operators.
            loop {
                match frames.pop() {
                    None => return Some(value),
                    Some(ConstFrame::Neg) => value = -value,
                    Some(ConstFrame::SubLhs { rhs }) => {
                        frames.push(ConstFrame::SubRhs { lhs: value });
                        current = rhs;
                        continue 'open;
                    }
                    Some(ConstFrame::SubRhs { lhs }) => value = lhs - value,
                    Some(ConstFrame::Add { args, next, acc }) => {
                        let acc = acc + value;
                        if let Some(&child) = args.get(next) {
                            frames.push(ConstFrame::Add {
                                args,
                                next: next + 1,
                                acc,
                            });
                            current = child;
                            continue 'open;
                        }
                        value = acc;
                    }
                }
            }
        }
    }
}

/// Comparison operator (strict or non-strict greater-than).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompOp {
    Gt,
    Ge,
}

impl CompOp {
    fn flip(self) -> Self {
        match self {
            CompOp::Gt => CompOp::Ge, // flipping strict: -x > c → x < -c → -x >= c (approx)
            CompOp::Ge => CompOp::Gt,
        }
    }
}

/// Check if n is a perfect square (i.e., there exists k such that k*k = n).
fn is_perfect_square(n: u64) -> bool {
    if n == 0 {
        return true;
    }
    let r = (n as f64).sqrt() as u64;
    // Check r and r+1 in case of floating-point rounding
    (r * r == n) || ((r + 1) * (r + 1) == n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_perfect_square() {
        assert!(is_perfect_square(0));
        assert!(is_perfect_square(1));
        assert!(is_perfect_square(4));
        assert!(is_perfect_square(9));
        assert!(is_perfect_square(16));
        assert!(is_perfect_square(25));
        assert!(!is_perfect_square(2));
        assert!(!is_perfect_square(3));
        assert!(!is_perfect_square(5));
        assert!(!is_perfect_square(6));
        assert!(!is_perfect_square(7));
        assert!(!is_perfect_square(8));
    }
}
