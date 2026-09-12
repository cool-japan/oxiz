//! Nelson-Oppen combination for the bit-vector theory: [`BvSolver`]'s
//! [`TheoryCombination`] implementation.
//!
//! Split out of `solver.rs` to keep that file under the workspace's 2000-line
//! ceiling; it is the same `impl TheoryCombination for BvSolver` block, moved
//! verbatim.
//!
//! # Reachability
//!
//! Nothing on the SMT-LIB script path calls [`BvSolver::notify_equality`]
//! today: `oxiz-solver`'s `TheoryManager` forwards equality notifications only
//! to the arithmetic solver, and `oxiz-theories`' own `combination` module
//! holds no `BvSolver` at all. It is reachable through the public
//! `TheoryCombination` trait (re-exported from `oxiz-solver`), which is why it
//! carries the same budget wiring and the same verdict discipline as
//! [`BvSolver::check`] — an unfinished solve must never have a model read out
//! of it.

use super::BvSolver;
use crate::theory::{EqualityNotification, TheoryCombination};
use oxiz_core::ast::TermId;
use oxiz_sat::SolverResult;

impl TheoryCombination for BvSolver {
    fn notify_equality(&mut self, eq: EqualityNotification) -> bool {
        // Check if both terms are relevant to the BV theory
        let lhs_known = self.term_to_bv.contains_key(&eq.lhs);
        let rhs_known = self.term_to_bv.contains_key(&eq.rhs);

        if lhs_known && rhs_known {
            // Both terms are BV variables -- enforce bit-level equality
            // via SAT encoding and check for consistency.
            //
            // `assert_eq` returns `false` when the two bit-vectors have
            // *different* widths, which is not a well-sorted equality and has
            // no bit-level encoding.  Nothing is asserted, so the `solve()`
            // below re-checks only the constraints that were already there and
            // this returns `true`.
            //
            // `true` is the correct answer, not merely the convenient one.  The
            // combination layer reads this method as "`!accepted` **and** both
            // operands `is_relevant` ⇒ a genuine cross-theory conflict"
            // (`combination.rs`, which spells out that a bare `false` is
            // ambiguous).  Both operands *are* in `term_to_bv` on this branch,
            // so `is_relevant` holds for both, and returning `false` here would
            // be read as a refutation this theory never derived — a fabricated
            // `Unsat`.  Reporting "accepted" is incomplete (the pair is not
            // modelled) but sound.
            //
            // Made explicit because the same silent discard at the
            // theory-manager call sites is what let an unasserted atom read as
            // "no theory objection" (U-Z10's sibling family) — there the safe
            // direction is the opposite one, which is why the two are handled
            // differently.
            let _widths_agreed = self.assert_eq(eq.lhs, eq.rhs);
            self.equality_notifications.push(eq);

            // After encoding the equality, check if the SAT solver detects
            // an immediate conflict (e.g., the two BVs were already constrained
            // to different constant values)
            // Budgeted like every other embedded solve (U-Z12): this one is
            // off the SMT-LIB script path today (`TheoryManager` only forwards
            // equality notifications to the arithmetic solver), but
            // `TheoryCombination` is public API and a direct caller reaches it.
            let allowance = self.remaining_conflict_budget();
            self.apply_budget_to_embedded(allowance);
            let conflicts_before = self.sat.stats().conflicts;
            let solve_result = self.sat.solve();
            self.charge_embedded_conflicts(conflicts_before);
            match solve_result {
                SolverResult::Unsat => {
                    // The equality is inconsistent with current BV constraints
                    self.sat.backtrack_to_root();
                    false
                }
                SolverResult::Sat => {
                    // Extract model-based equalities: if two BV terms now have
                    // the same value in the model, propagate that equality
                    self.extract_model_equalities();
                    self.sat.backtrack_to_root();
                    true
                }
                SolverResult::Unknown => {
                    // The solve ran out of budget, so there is no model to read
                    // and `extract_model_equalities` must NOT be called: it
                    // reads `self.sat.model()`, which after an unfinished solve
                    // holds whatever partial assignment the search abandoned,
                    // and would derive shared equalities that nothing entails.
                    // Reporting "accepted, no equalities derived" is the sound
                    // answer: `false` here means "this theory refutes the
                    // equality", which is a claim an exhausted solve cannot
                    // make.  Incompleteness is handled by the enclosing search,
                    // which sees the budget exhaustion through its own poll.
                    self.sat.backtrack_to_root();
                    true
                }
            }
        } else if lhs_known || rhs_known {
            // One term is a BV term, the other is foreign (shared variable).
            // Record the notification for later processing.
            self.equality_notifications.push(eq);
            true
        } else {
            // Neither term is relevant to this theory
            false
        }
    }

    fn get_shared_equalities(&self) -> Vec<EqualityNotification> {
        self.shared_equalities.clone()
    }

    fn is_relevant(&self, term: TermId) -> bool {
        self.term_to_bv.contains_key(&term)
    }
}
