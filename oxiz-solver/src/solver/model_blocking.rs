//! Bounded blocking of refuted candidate models (issue #40).
//!
//! [`Solver::model_refutes_assertions`] is the last gate before a `Sat` leaves
//! the ground branch of [`Solver::check_core`].  When it fires, the candidate
//! assignment on the table is not a model of the assertions — but *the search
//! was not finished*, and the formula may well have other models the CDCL(T)
//! core would have reached had it been asked to keep going.  The gate used to
//! discard the model and concede `Unknown` on the spot, which turned every
//! single unlucky first candidate into a lost `sat`.
//!
//! This module supplies the missing step: **exclude that one assignment and
//! re-solve**, up to [`MAX_MODEL_BLOCKING_ROUNDS`] times.
//!
//! # A search restriction, not a lemma — and why that distinction is the
//! whole soundness argument
//!
//! A blocking clause here is emphatically **not** a consequence of the
//! assertions.  The gate fires on two different things (see
//! [`super::model_eval::EvalOutcome`]):
//!
//! * `Value(Bool(false))` — the model genuinely falsifies an assertion.  A
//!   clause excluding it *would* be entailed.
//! * `Unrepresentable` — the *evaluator's* fixed-width `Rational64` arithmetic
//!   could not represent an intermediate result.  That is a statement about
//!   the evaluator, not about the assignment: the assignment may be a perfectly
//!   good model that this particular checker cannot certify.  A clause
//!   excluding it is entailed by **nothing**.
//!
//! Because the second trigger exists, the clauses this module adds are treated
//! as a *restriction of the search space*, not as knowledge:
//!
//! * A `Sat` found afterwards is still a real `Sat` — it is a surviving
//!   assignment, re-verified by the same gate, and a restricted search can only
//!   ever return genuine models of the unrestricted problem.
//! * An `Unsat` found afterwards means "no model outside the excluded region",
//!   which is **not** `unsat`.  While [`Solver::model_blocking_active`] is
//!   nonzero, every SAT-core-derived `Unsat` is therefore surfaced as
//!   [`SolverResult::Unknown`] with no unsat core — see
//!   [`Solver::blocking_clauses_present`].
//!
//! The verdict lattice this produces is `Unknown -> {Sat, Unknown}`: the
//! feature can only ever upgrade the old spurious `Unknown` into a `Sat`, never
//! turn any answer into a worse or a wrong one.
//!
//! Lemma-grade blocking — emitting an entailed clause and keeping `Unsat`
//! answerable — is deliberately **deferred**.  It would require the gate to
//! report *which* trigger fired and to block only on the `Bool(false)` one, and
//! then to justify the clause against the trail rather than against the model
//! projection.  The `Unrepresentable` trigger is the reason that work is not
//! done here.
//!
//! # Two opposite-direction "drop a literal" rules
//!
//! This module drops literals in a place where a sibling in `check_core` must
//! not, and the two rules are worth stating together because they look alike
//! and are exact opposites.
//!
//! 1. **MBQI's all-or-nothing reason clause** (`check_core`, the
//!    `MBQIResult::Conflict` arm) turns a *reason* — a set of terms whose
//!    conjunction the quantifier refutes — into a blocking clause.  Dropping a
//!    reason term that names no SAT literal there does not weaken the clause,
//!    it **strengthens** it into a claim the reason never made: that the
//!    surviving literals alone are contradictory.  So that site collects into
//!    `Option<Vec<Lit>>` and adds nothing at all when any term is unmapped.
//!
//! 2. **Here the clause is a projection, not a reason.**  Its meaning is "not
//!    exactly this assignment (restricted to the mapped variables)", and a
//!    projection onto fewer variables is a *stronger* restriction that excludes
//!    a superset of assignments — which is sound in this direction precisely
//!    because the clause is a search restriction whose `Unsat` is already
//!    downgraded.  What is *not* acceptable is projecting onto **nothing**: an
//!    empty clause is the false clause, which would make every subsequent solve
//!    report `Unsat` (downgraded to `Unknown`) and would have excluded the
//!    entire search space rather than one assignment.  So an empty projection
//!    **declines** (`false`), leaving the caller to concede `Unknown` exactly as
//!    before.
//!
//! # Why `LBool::Undef` on a mapped variable is dropped, not guessed
//!
//! A variable the SAT core left unassigned is one the candidate model does not
//! constrain: *both* polarities are consistent with what the search committed
//! to.  Omitting it from the clause blocks the assignment for both polarities
//! at once, which is what "this candidate" actually means.  Guessing a polarity
//! and adding that literal would be the unsound direction — it would leave the
//! sibling assignment (the same commitments, opposite guess) unblocked and let
//! the very same refuted candidate come straight back, burning the round
//! budget without progress.
//!
//! Note the asymmetry with rule 2 above: dropping an *unassigned* variable is
//! always correct, while dropping an *assigned* literal is what makes the
//! clause stronger and is only tolerable under the downgrade discipline.
//!
//! # Scoping
//!
//! The clauses are added with [`oxiz_sat::Solver::add_clause`], so they live at
//! the SAT core's current scope and are retracted by `Solver::pop`'s
//! `self.sat.pop()` along with everything else that scope added.  The
//! `model_blocking_active` counter is snapshotted in
//! [`super::trail::ContextState`] so it is restored in lockstep.
//!
//! `sat.push()` / `sat.pop()` around the blocking round itself was considered
//! and **rejected**: the pop would also retract the case-split and array-axiom
//! lemmas the sibling repair paths added in the same search (those *are*
//! entailed and must survive), and it would desynchronise the scope counting
//! `Solver::push` / `Solver::pop` rely on.

use oxiz_sat::{LBool, Lit, Var};

use super::Solver;

/// How many refuted candidate models one context scope may exclude before the
/// solver gives up and concedes `Unknown`.
///
/// OxiZ tuning decision. Every round is a full re-solve of the whole problem,
/// so the budget has to be small enough that a pathological formula (one whose
/// models are refuted by the millions) cannot turn a fast `unknown` into an
/// unbounded grind. 64 is comfortably above what the failure this exists to
/// fix needs — a handful of unlucky candidates ahead of a good one — and the
/// wall-clock ceiling the caller applies on top of it (the same
/// `int_case_split::REFINEMENT_TIME_CEILING_MS` the case-split refinement uses)
/// is the real guard on hard instances.
///
/// This is a *lifetime* budget for the scope, not a per-`check` one: the
/// clauses are permanent until the scope is popped, so the count of clauses
/// added and the count of rounds spent are the same number, and one counter
/// serves as both the budget and the "the database is restricted" flag.
pub(super) const MAX_MODEL_BLOCKING_ROUNDS: usize = 64;

impl Solver {
    /// `true` iff at least one model-blocking clause is live in the SAT
    /// database, so a SAT-core-derived `Unsat` must be surfaced as `Unknown`.
    ///
    /// See this module's header for the argument. Every call site that turns an
    /// `oxiz_sat::SatResult::Unsat` into a [`crate::SolverResult::Unsat`] must
    /// consult this first; the syntactic early-conflict detectors
    /// (`check_string_constraints` and friends), the `has_false_assertion`
    /// fast path and the nonlinear dispatch do not, because none of them reads
    /// the SAT clause database.
    pub(super) fn blocking_clauses_present(&self) -> bool {
        self.model_blocking_active > 0
    }

    /// The clause that excludes the assignment currently on the SAT trail:
    /// the negation of that assignment, projected onto the SAT variables that
    /// carry a term.
    ///
    /// An unmapped SAT variable is a Tseitin auxiliary with no meaning outside
    /// the encoding, and a variable the core left `Undef` is one the candidate
    /// does not constrain; both are omitted. See the module header for why
    /// omitting is the correct direction here and the wrong one in MBQI's
    /// reason clause.
    ///
    /// Split out from [`Self::block_refuted_model`] so the projection rule can
    /// be tested literal by literal, independently of the budget and of the
    /// SAT database it would otherwise be written into.
    pub(super) fn refuted_model_projection(&self) -> Vec<Lit> {
        let mut lits: Vec<Lit> = Vec::with_capacity(self.var_to_term.len());
        for idx in 0..self.var_to_term.len() {
            // `var_to_term` is indexed by SAT variable index, which
            // `oxiz_sat::Var` stores as a `u32`; a length past `u32::MAX` is
            // unreachable, and skipping is the sound response either way (a
            // shorter projection is a stronger restriction, never a wrong one).
            let Ok(raw) = u32::try_from(idx) else {
                continue;
            };
            let var = Var::new(raw);
            match self.sat.model_value(var) {
                LBool::True => lits.push(Lit::neg(var)),
                LBool::False => lits.push(Lit::pos(var)),
                // Deliberately dropped, never guessed — see the module header.
                LBool::Undef => {}
            }
        }
        lits
    }

    /// Exclude the candidate assignment currently on the SAT trail.
    ///
    /// Returns `true` when a blocking clause was added (the caller owes a
    /// theory rebase and a re-solve) and `false` when this module declines —
    /// the feature is off, the round budget is spent, or the assignment
    /// projects onto no mapped variable at all. On `false` the solver is left
    /// exactly as it was found.
    ///
    /// # Why `add_clause` returning `false` still counts as blocked
    ///
    /// `false` from [`oxiz_sat::Solver::add_clause`] means the clause was
    /// refused as an unconditional (level-0) conflict — the SAT core is now
    /// trivially unsat. That is still a *successful* restriction of the search
    /// space, and the counter has already been bumped, so the next solve
    /// reports `Unsat` and [`Self::blocking_clauses_present`] downgrades it to
    /// `Unknown`. Reporting `false` here instead would let the caller return
    /// `Sat` for the very model it just refuted.
    pub(super) fn block_refuted_model(&mut self) -> bool {
        if !self.config.enable_model_blocking {
            return false;
        }
        if self.model_blocking_active >= self.config.max_model_blocking_rounds {
            return false;
        }

        let lits = self.refuted_model_projection();
        if lits.is_empty() {
            // An empty clause is the false clause, not "block this one
            // assignment". Decline rather than poison the database.
            return false;
        }

        self.model_blocking_active += 1;
        self.statistics.model_blocking_clauses += 1;
        // The return value is intentionally ignored: see the doc comment.
        let _ = self.sat.add_clause(lits);
        true
    }

    /// [`Self::block_refuted_model`] plus the state repair every re-solve in
    /// `check_core` owes before it continues.
    ///
    /// `add_clause` left the SAT core at the refuted candidate's trail, and the
    /// incremental theory solvers still hold that candidate's facts (only
    /// level-scoped `pop` is available, no surgical undo), so the next round
    /// must be driven from a rebased state — exactly as the case-split and
    /// array-axiom repair paths do. Skipping this reproduces the task-#26 false
    /// `unsat`, which the downgrade above would then mask into `Unknown` and
    /// make very hard to notice.
    ///
    /// `affordable` is the caller's wall-clock gate, passed in rather than
    /// re-measured here so the `std` / `no_std` split stays in one place.
    pub(super) fn block_refuted_model_and_rebase(&mut self, affordable: bool) -> bool {
        if !affordable || !self.block_refuted_model() {
            return false;
        }
        self.rebase_theory_state();
        self.debug_check_invariants("check_core: after model-blocking backtrack");
        true
    }
}

#[cfg(test)]
mod tests;
