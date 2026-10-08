//! The last step of a final check: the integrality of the `Int` terms of a
//! solver whose arithmetic runs over the reals.
//!
//! Under `(set-logic ALL)`, or no logic, the arithmetic solver runs in `LRA`
//! mode and an `Int` term was solved over the reals — `(assert (= (* 2 x)
//! 1))` over an `Int` `x` answered `sat` on every build (see
//! `oxiz_theories::arithmetic::solver::mixed`).  The arithmetic solver now
//! decides integrality by branch-and-bound over its `Int` terms
//! (`ArithSolver::check_integrality`); it is asked here, once per final
//! check and only after Nelson-Oppen combination has reached its fixpoint
//! without a conflict, because every check inside the search sees a partial
//! assignment and branch-and-bound on each of them costs what the search
//! cannot afford.  Its leaf is the assignment the model is read from, and no
//! arithmetic check follows it in this final check.
//!
//! A refutation comes back as the solver's full reason set — a valid, if
//! coarse, conflict clause; an undecided branch-and-bound (its node budget)
//! flags resource exhaustion, so the solver answers `unknown` rather than a
//! `sat` over a fractional integer.

use super::TheoryManager;
use oxiz_sat::TheoryCheckResult;

impl TheoryManager<'_> {
    /// Nelson-Oppen combination, then — where it ends without a conflict —
    /// the integrality of the `Int` terms (see the module docs).
    pub(super) fn combine_then_integrality(&mut self) -> TheoryCheckResult {
        let combined = self.nelson_oppen_combine();
        if !matches!(combined, TheoryCheckResult::Sat) || self.resource_exhausted {
            return combined;
        }
        match self.arith.check_integrality() {
            Ok(oxiz_theories::TheoryCheckResult::Unsat(conflict_terms)) => {
                self.report_theory_conflict(conflict_terms)
            }
            Ok(oxiz_theories::TheoryCheckResult::Sat)
            | Ok(oxiz_theories::TheoryCheckResult::Propagate(_)) => TheoryCheckResult::Sat,
            Ok(oxiz_theories::TheoryCheckResult::Unknown) | Err(_) => {
                self.resource_exhausted = true;
                TheoryCheckResult::Sat
            }
        }
    }
}
