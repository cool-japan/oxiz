//! The lazy array-axiom refinement *round*, and the two candidate-model paths
//! that run it.
//!
//! # Why this is a module and not eight lines of `check_core`
//!
//! The refinement used to be written inline, inside `check_core`'s
//! `if !self.has_quantifiers { … }` branch.  That placement was the seam's
//! other half: a script with a single `forall` in it never reached the branch,
//! so `instantiate_array_axioms` was never called for it — not for the array
//! terms an MBQI instance grounds, and not even for the ones the script writes
//! out in its ground assertions.  Every array rule this crate has (read-over-
//! write, constant-array congruence, the `ite` reads, extensionality, the
//! cardinality refutations) was unreachable the moment the input quantified
//! over anything.
//!
//! Hoisting the round into [`Solver::array_refinement_round`] is what lets the
//! quantified path call it too.  `check_core` now calls it from both places:
//! once on the ground candidate model, and once on the partial model the MBQI
//! loop builds, before any of that loop's three `Sat` exits.
//!
//! # What a round is, and what the caller still owns
//!
//! One round asks [`Solver::instantiate_array_axioms`] for every axiom
//! instance the current candidate model violates, asserts them as lemmas, and
//! prepares the solver to search again: it charges the two deterministic
//! budgets (decision (9)), arms the resolve-conflict ceiling on the first
//! round, re-installs the deadline and the bit-vector budget, defines the
//! arithmetic meaning of any `ite` the lemmas introduced, and rebases the
//! theory state off the SAT trail.
//!
//! What it deliberately does **not** do is rebuild the `TheoryManager` or jump
//! to the top of the loop.  The manager holds `&mut` borrows of three of the
//! solver's fields, so it cannot exist across a `&mut self` call; constructing
//! it is the caller's job, at the call site, immediately before `continue`.
//! The return value says which of the three things the caller must do.

use oxiz_core::ast::TermManager;

use super::Solver;
use super::check_core::{ARRAY_REFINEMENT_LEMMA_BUDGET, ARRAY_REFINEMENT_RESOLVE_CONFLICTS};

/// How many refinement rounds one `check` may run.
///
/// Bounded independently of the MBQI budget; deduplication guarantees
/// saturation well within this generous cap for realistic inputs.
pub(super) const MAX_ARRAY_REFINEMENT_ROUNDS: usize = 256;

/// What [`Solver::array_refinement_round`] leaves for the caller to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ArrayRefinementStep {
    /// The candidate model satisfies every applicable axiom instance (or the
    /// problem has no array operations at all).  Nothing was asserted and
    /// nothing was reset; carry on with the candidate.
    NoLemma,
    /// At least one lemma was asserted and the solver has been prepared for a
    /// fresh search.  The caller must rebuild its `TheoryManager` and
    /// `continue` the search loop.
    Resolve,
    /// A deterministic budget is exhausted, so the axiomatisation this search
    /// ran against is a strict subset of the array theory and no verdict may
    /// be reported from it.  `self.model` and `self.unsat_core` have already
    /// been cleared; the caller must return `SolverResult::Unknown`.
    OutOfBudget,
}

impl Solver {
    /// Run one lazy array-axiom refinement round against the current candidate
    /// model.
    ///
    /// The syntactic array pre-checks and EUF congruence do not implement a
    /// complete array decision procedure, so a candidate `Sat` may violate
    /// read-over-write or extensionality.  This watches the array terms in the
    /// candidate model, asserts every axiom instance it does not already
    /// satisfy as a lemma, and prepares a re-solve; only genuine array models
    /// survive it.
    ///
    /// `rounds` and `resolve_conflict_ceiling` are the caller's per-`check`
    /// loop state, threaded through rather than stored on the solver because
    /// they belong to one search and not to the assertion stack.
    pub(super) fn array_refinement_round(
        &mut self,
        manager: &mut TermManager,
        rounds: &mut usize,
        resolve_conflict_ceiling: &mut Option<u64>,
        conflict_budget: Option<u64>,
        conflicts_so_far: u64,
        deadline: Option<oxiz_time::Instant>,
    ) -> ArrayRefinementStep {
        if !self.has_array_ops || !self.instantiate_array_axioms(manager) {
            return ArrayRefinementStep::NoLemma;
        }
        *rounds = rounds.saturating_add(1);
        self.statistics.array_refinement_rounds =
            self.statistics.array_refinement_rounds.saturating_add(1);
        if self.statistics.array_lemma_instances >= ARRAY_REFINEMENT_LEMMA_BUDGET {
            // The second deterministic currency (decision
            // (9)).  The conflict ceiling at the loop head
            // bounds a refinement loop that *searches*;
            // this one bounds a loop that only *builds*,
            // which the conflict counter cannot see
            // because such a loop never conflicts.  See
            // [`ARRAY_REFINEMENT_LEMMA_BUDGET`].
            //
            // The model goes with it for the same reason
            // the round-budget exit below gives.
            self.model = None;
            self.unsat_core = None;
            return ArrayRefinementStep::OutOfBudget;
        }
        if *rounds >= MAX_ARRAY_REFINEMENT_ROUNDS {
            // Could not saturate the array axioms within the
            // round budget: do not fabricate a verdict.
            //
            // The model goes with it (issue #40): since the
            // refutation gate moved *below* this path, the
            // candidate on the table here may be one the
            // gate would have rejected, and a rejected model
            // must not stay readable behind an `Unknown`.
            self.model = None;
            self.unsat_core = None;
            return ArrayRefinementStep::OutOfBudget;
        }
        // Arm the refinement's own budget on the first
        // round.  Without it a single re-solve can run
        // unboundedly: the lemmas this loop asserts enlarge
        // the bit-blasted circuit, the per-assignment
        // `BvSolver::check` grows with it, and a
        // fifteen-line script that the same tree answered
        // in 0.46 s before the lemmas existed ran for more
        // than five minutes with no answer at all.
        // `Unknown` is the honest outcome there, and it is
        // what the round budget above already returns for
        // the same reason.
        //
        // The budget is a *conflict count*, not a clock
        // (decision (9), `#P2b-38` strand (c)).  It was a
        // wall-clock floor, and that made the verdict a
        // property of the machine: the same release binary
        // on the same script answered `sat` at 77.5 s run
        // alone and `unknown` at the 120 s floor with ten
        // copies in flight.  A solver whose answers are
        // consumed as verification evidence cannot do that
        // — the evidence has to reproduce elsewhere — so
        // the bound is now the number of Boolean conflicts
        // the search accrues from the first array lemma
        // onwards: monotone, measurable, and identical on
        // an idle and a loaded machine.  An explicit
        // `:timeout` remains the only wall clock, and it is
        // installed at the top of `check` exactly as
        // before.
        if resolve_conflict_ceiling.is_none() {
            let ceiling = self
                .sat
                .stats()
                .conflicts
                .saturating_add(ARRAY_REFINEMENT_RESOLVE_CONFLICTS);
            *resolve_conflict_ceiling = Some(ceiling);
            // Bound the *inner* search too, so one re-solve
            // cannot run past the ceiling before the round
            // boundary above gets to look at it.  Never
            // above a user `:max-conflicts`, which is the
            // stricter instruction where both are present.
            // A user `:max-conflicts N` is the budget of the
            // *whole check*, not of every refinement round:
            // the entry ceiling is
            // `conflicts_so_far + N` with `conflicts_so_far`
            // read once, at the top of this function, and it
            // is that value the refinement must not exceed.
            // Re-reading `stats().conflicts` here instead
            // re-based the ceiling on the conflicts the
            // first solve had already spent, silently
            // granting the search more than the user asked
            // for.
            let installed = match conflict_budget {
                Some(user) => core::cmp::min(ceiling, conflicts_so_far.saturating_add(user)),
                None => ceiling,
            };
            self.sat.set_max_conflicts(Some(installed));
        }
        self.sat.set_deadline(deadline);
        self.bv.set_budget(conflict_budget, deadline);
        // A read-over-write lemma is an `ite` over the two
        // array values; at Int/Real sort that `ite` is a new
        // opaque arithmetic atom, so define it before the
        // re-solve or the lemma carries no numeric meaning.
        self.instantiate_arith_axioms(manager);
        // Re-solve with the freshly asserted array lemmas from
        // a clean state.  `add_clause` backtracked the SAT core
        // to root for the unit lemmas, but the incremental
        // theory solvers still hold the facts committed by the
        // just-refuted candidate model (e.g. a stale
        // `select = 6`) — including any left in scopes this
        // round's search never unwound.
        self.rebase_theory_state();
        // After backtracking to root and resetting the
        // theory solvers: the SAT-variable <-> term tables
        // and the Tseitin memo are *not* reset here, so
        // they must still describe the same variables the
        // replayed search will re-derive.
        self.debug_check_invariants("check_core: after array-lemma backtrack");
        ArrayRefinementStep::Resolve
    }
}
