//! The integrality of the `Int` terms of a quantified goal, decided where the
//! MBQI loop would answer `sat` (re-fix pass 16, recheck 15's `#P2b-79`
//! trajectory losses).
//!
//! Under `(set-logic ALL)`, or no logic, the arithmetic runs over the reals
//! and an `Int` term is marked for integrality (`#P2b-79`,
//! `oxiz_theories::arithmetic::solver::mixed`).  On a quantifier-free goal the
//! arithmetic solver decides it itself, by branch-and-bound at every final
//! check.  On a quantified goal that changed which candidate each MBQI round
//! read — the tightened strict bounds and the branch-and-bound leaf reach the
//! counterexample search through the model — and `c702310`'s refutations were
//! lost: `(forall ((i Int)) (distinct i j))` answered `unknown` after 176
//! conflicts (`unsat` in 14 on `c702310`), and so did 25 checks of recheck
//! 15's `gen_mix.py` corpus, seven of `gen14.py` seed 30093155 and two of
//! `gen_guard.py` seed 30093152, every one restored by switching the marks off.
//!
//! So a quantified goal defers the integrality
//! (`ArithSolver::set_defer_integrality`): every round searches the
//! relaxation exactly as `c702310` did, and the MBQI loop reads the values it
//! read there.  Only where the loop would conclude — a `sat` exit with no
//! instance pending, or the exit that gives up after ten rounds without a
//! verdict — is the integrality looked at: a candidate whose every marked
//! `Int` term is integral is published (or given up on) exactly as before;
//! one that leaves a marked term fractional (or at a strict bound's `δ`) takes
//! the deferral back, and the loop searches again with the arithmetic solver
//! deciding the integrality itself (branch-and-bound at every final check,
//! strict bounds over `Int` rows tightened, the divisibility check on
//! all-`Int` equalities — `#P2b-79`'s machinery) for the rest of the check.
//! Every verdict `c702310` reached without concluding over a fractional
//! integer is therefore reached on the same trajectory, and a `sat` over a
//! fractional integer — the wrong `sat` `#P2b-79` closed — is never
//! published: `(= (* 2 x) 1)` beside a universal is refuted after the switch,
//! and so is `(forall ((i Int)) (= (* 2 (f i)) (+ (* 2 i) 1)))`, whose
//! relaxation satisfied every instance with `f(v) = v + 1/2` until the loop
//! gave up.  The cost, measured: taking the integrality back where the loop
//! gives up turned `c702310`'s quick `unknown` into a search on `gen14.py`
//! seed 30093155 `s00262` (its second check, `unsat` on `c702310`, gets no
//! answer in 130 s); taking back the divisibility check alone there, or
//! keeping it on throughout, lost other `c702310` verdicts (`gen_mix.py`
//! `m00585`, and `s00262` itself), so the full take-back stands and the loss is
//! named in `TODO.md` decision (24a).

use super::Solver;

/// What the integrality check at a quantified exit leaves the loop to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IntegralityAtExit {
    /// Every marked `Int` term is integral in the candidate (or the
    /// arithmetic solver already decides the integrality itself): conclude as
    /// before.
    Integral,
    /// The deferral was taken back: run another round.
    Resume,
}

impl Solver {
    /// Decide the integrality of the candidate at a quantified `sat` exit (see
    /// the module docs): a fractional marked term takes the deferral back in
    /// full.
    pub(super) fn integrality_at_quantified_exit(&mut self) -> IntegralityAtExit {
        if !self.arith.integrality_deferred() || self.arith.first_non_integral_int_term().is_none()
        {
            return IntegralityAtExit::Integral;
        }
        self.arith.set_defer_integrality(false);
        IntegralityAtExit::Resume
    }
}
