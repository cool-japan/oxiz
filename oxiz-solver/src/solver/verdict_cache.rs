//! Recognising a repeated `check-sat` on a goal the caller has not touched.
//!
//! [`Solver::check`](super::Solver::check) answers such a query from the
//! previous verdict instead of running the search again.  This module holds the
//! two halves of that: the [`GoalFingerprint`] that decides when a query is
//! "the same", and the small pair of helpers `check` calls.
//!
//! # Why a repeat must not be re-run (task #28)
//!
//! Re-running is *not* idempotent, and both directions of that are bugs.
//!
//! It costs clauses.  The SAT solver deliberately keeps what it learned, so a
//! second search takes a different route through the same goal and ends on a
//! different model.  MBQI is model-*based*: handed a different model it produces
//! different counterexamples, instantiates at different ground terms, and emits
//! lemmas over SAT variables that did not exist before.  Those clauses are
//! original clauses added with no assertion behind them, and nothing will ever
//! retract them — the assertion stack did not move, so no `pop` covers them.  A
//! caller polling `(check-sat)` in a loop pays for that permanently.  Measured
//! on a UFLIA goal that answers `unknown`: flat for four calls, then +57
//! original clauses at call 5, which is why the growth first read as
//! non-deterministic.
//!
//! It costs work without bound.  An `unknown` goal re-ran the full MBQI round
//! budget on *every* call, so the total work a caller could provoke grew with
//! the number of times it asked the same question.
//!
//! # Why the cache cannot go stale
//!
//! [`GoalFingerprint`] is the guard that decides.  It is re-derived from live
//! state on entry to every `check` and compared against the one stored beside
//! the verdict; a mismatch anywhere runs the search.  A mutation that escaped it
//! would have to add no assertion, allocate no SAT variable, emit no clause,
//! journal no undo entry, register no quantifier or instantiation candidate, and
//! leave every solver setting — the whole [`SolverConfig`], the unsat-core and
//! branching switches, the declared logic and the SAT engine's random seed —
//! exactly as it found them.  That is, to change nothing a `check` can read.
//!
//! Two hooks keep the second half of that true rather than merely hoped for.
//! `Solver::invalidate_results` (called from `assert`, `push`, `pop`, `reset`)
//! drops the cached verdict alongside the cached `model` and `unsat_core`, so a
//! cached `Sat` is exactly as trustworthy as the model `(get-model)` would
//! return beside it.  [`Solver::settings_changed`] does the same for the
//! settings half, and every setter in `solver::config` ends by calling it.
//!
//! Both hooks are defence in depth: emptied out, the fingerprint alone still
//! refuses every stale query, because it carries by value everything they
//! announce (this was measured — see the mutation testing recorded in the ticket
//! for #28).  They earn their place by being *cheap* and by failing safe in the
//! opposite direction: they drop a live cache entry rather than keep a dead one,
//! so a hook added to a new mutator can never introduce a stale read even if the
//! fingerprint has not yet learned about the field it touches.
//!
//! Note what is deliberately *not* invalidated: nothing about the search's own
//! residue.  A cache hit skips a search whose every by-product (learned clauses,
//! heuristic scores, kept lemma clauses) is already in the solver, so the next
//! genuine `check` starts from the state it would have started from anyway.

use super::Solver;
use super::types::{Proof, SolverConfig, SolverResult};

#[cfg(test)]
mod tests;

/// A cheap, structural summary of everything a `check` reads as *input*.
///
/// Every counter here is one the goal-changing commands must move: `assert`
/// appends to `assertions` (and journals a trail op), `push` / `pop` change the
/// scope depth (and `pop` truncates the trail), `declare-const` grows the MBQI
/// candidate pool, and `reset` zeroes all of them.  The settings half is
/// carried whole rather than field by field — see [`Self::settings_epoch`] and
/// [`Self::config`].
///
/// Comparing one of these is a handful of integer comparisons: `SolverConfig`
/// is scalars and field-less enums throughout, so cloning it into the
/// fingerprint is a `memcpy` with no allocation.  Nothing here allocates, which
/// matters because `check` builds a fingerprint on entry (before it knows
/// whether it will hit) and again on exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GoalFingerprint {
    /// `Solver::assertions.len()`
    num_assertions: usize,
    /// `Solver::named_assertions.len()`
    num_named_assertions: usize,
    /// `Solver::context_stack.len()` — the open `push` depth.
    num_scopes: usize,
    /// `Solver::trail.len()` — the undo journal's length.
    trail_len: usize,
    /// Number of SAT variables allocated so far.
    num_sat_vars: usize,
    /// Number of original (asserted) SAT clauses.
    num_original_clauses: usize,
    /// Quantifiers registered with MBQI.
    num_mbqi_quantifiers: usize,
    /// Quantifiers registered with the e-matching engine.
    num_ematch_quantifiers: usize,
    /// Ground terms registered as MBQI instantiation candidates.
    num_mbqi_candidates: usize,
    /// `Solver::has_false_assertion`
    has_false_assertion: bool,
    /// `Solver::settings_epoch` — bumped by every setter in `solver::config`.
    ///
    /// This is what covers the settings a fingerprint cannot hold by value: the
    /// declared logic (a `String`, and copying it per call would put an
    /// allocation on this path for no gain) and the SAT engine's random seed
    /// (which the SAT solver owns and does not read back out).
    settings_epoch: u64,
    /// The whole search configuration, by value.
    ///
    /// Not a hand-picked subset, and deliberately so.  Anything the solve loop
    /// honours — `timeout_ms`, `max_conflicts`, `max_decisions`, `theory_mode`,
    /// `simplify`, `proof`, `finite_expansion_budget`, the preprocessing
    /// switches — changes what a `check` may conclude, and above all *whether*
    /// it concludes: an `Unknown` is a statement about the budget, not about the
    /// goal, so replaying one across a budget change answers a question that was
    /// not asked.  Holding the struct whole means a field added to
    /// `SolverConfig` later is covered the day it is added.
    ///
    /// It also covers the one mutation path that legitimately does *not* bump
    /// `settings_epoch`: [`Solver::check_with_limits`] narrows `max_conflicts` /
    /// `max_decisions` around a single call by writing the fields directly and
    /// restoring them afterwards, so a plain `check()` afterwards sees a
    /// different config here and re-runs, while a repeated `check_with_limits`
    /// under the same limits sees the same one and hits.
    config: SolverConfig,
    /// `Solver::produce_unsat_cores` — a cached `Unsat` computed with core
    /// production off has no core to hand to a caller who has since turned it
    /// on.
    produce_unsat_cores: bool,
    /// `Solver::theory_aware_branching` — changes the decision order, hence
    /// which model a satisfiable goal yields and whether a budgeted search
    /// finishes.
    theory_aware_branching: bool,
}

impl Solver {
    /// Discard the results of the last [`Self::check`].
    ///
    /// # The rule
    ///
    /// A model and an unsat core are statements *about a particular assertion
    /// stack*.  Z3 applies the corresponding rule at the context level — any
    /// change to the assertion stack invalidates the previous verdict, which is
    /// also SMT-LIB 2.6 §4.1.1's mode machine (`assert`, `push`, `pop`,
    /// `reset-assertions` and `reset` all return the solver to `assert` mode,
    /// where `get-model` / `get-unsat-core` / `get-proof` are unavailable).
    /// This is that rule, enforced on the *solver* rather than only on
    /// [`crate::Context`], so an embedder driving [`Solver`] directly cannot
    /// read a result that no longer describes what it has asserted.
    ///
    /// # Why each field is or is not cleared
    ///
    /// * `model` — a model of the *old* stack need not satisfy the new one
    ///   (`assert` strengthens it, `pop` replaces it wholesale).  Cleared.
    /// * `unsat_core` — its `indices` name positions in `assertions`, which
    ///   `pop` truncates.  A survivor is not merely imprecise, it *dangles*:
    ///   `Solver::minimize_unsat_core` indexes `assertions` by them and the
    ///   structural net [`crate::invariants::check_unsat_core`] rejects them.
    ///   Cleared.
    /// * `proof` — the `Option` carries the `:produce-proofs` setting, so it is
    ///   emptied in place rather than taken: dropping it to `None` would
    ///   silently *disable* proof production for the rest of the session.
    /// * `statistics` — cumulative over the solver's whole life, not a verdict
    ///   about one stack; SMT-LIB's `(get-info :all-statistics)` is not gated on
    ///   solver mode either.  Deliberately kept (see [`Self::reset_statistics`]
    ///   for the explicit way to zero it).
    /// * `last_check` — the verdict itself, cached so that asking the same
    ///   question twice does not re-run the search (see this module's docs).  It is a
    ///   verdict about the old stack in exactly the way `model` is, so it is
    ///   cleared through exactly the same hook.
    pub(super) fn invalidate_results(&mut self) {
        self.model = None;
        // The other half of the model — the exact algebraic values `Model`
        // cannot hold (see `Solver::nl_algebraic_values`). It describes the
        // discarded verdict's assignment just as `model` does, so a stale
        // entry surviving here would let a later `(get-model)` report `√2`
        // for a variable the new assertion stack constrains differently.
        self.nl_algebraic_values.clear();
        self.unsat_core = None;
        self.last_check = None;
        // Part of the model, not of the formula: the pure-equality fast path's
        // partition describes the assertion stack the discarded verdict was
        // computed on, and grouping a *different* stack's constants by it
        // would hand `(get-model)` witnesses that share a name for constants
        // nothing equates.
        self.equality_skeleton_classes.clear();
        if let Some(proof) = self.proof.as_mut() {
            *proof = Proof::new();
        }
    }

    /// Structural summary of the current goal — see [`GoalFingerprint`].
    pub(super) fn goal_fingerprint(&self) -> GoalFingerprint {
        GoalFingerprint {
            num_assertions: self.assertions.len(),
            num_named_assertions: self.named_assertions.len(),
            num_scopes: self.context_stack.len(),
            trail_len: self.trail.len(),
            num_sat_vars: self.sat.num_vars(),
            num_original_clauses: self.sat.num_original_clauses(),
            num_mbqi_quantifiers: self.mbqi.num_quantifiers(),
            num_ematch_quantifiers: self.ematch_engine.num_quantifiers(),
            num_mbqi_candidates: self.mbqi.num_candidates(),
            has_false_assertion: self.has_false_assertion,
            settings_epoch: self.settings_epoch,
            config: self.config.clone(),
            produce_unsat_cores: self.produce_unsat_cores,
            theory_aware_branching: self.theory_aware_branching,
        }
    }

    /// The verdict of the previous `check`, if this query is the same one.
    pub(super) fn cached_verdict(&self, fingerprint: &GoalFingerprint) -> Option<SolverResult> {
        match &self.last_check {
            Some((cached, result)) if cached == fingerprint => Some(*result),
            _ => None,
        }
    }

    /// Drop the cached verdict without touching anything else.
    ///
    /// Test-only, and deliberately so: it is the one way to make the *next*
    /// `check` run a real search on a goal nothing has changed, which is what
    /// the task-#28 pins in `solver::scope_rebase_tests` need in order to
    /// observe what the search machinery beneath the cache does when it is run
    /// twice.  Production code has no business asking for that — a repeat is
    /// answered from the cache precisely because re-running is not idempotent.
    #[cfg(test)]
    pub(crate) fn forget_cached_verdict(&mut self) {
        self.last_check = None;
    }

    /// Remember `result` for the goal as it stands *now*.
    ///
    /// Called after the search, so the fingerprint describes the state the
    /// search left behind rather than the one it started from: a search
    /// allocates variables, emits lemma clauses and journals undo entries, all
    /// of which move the fingerprint.  Comparing the next call's entry state
    /// against this one is what makes "nothing happened in between" the
    /// condition for a hit.
    pub(super) fn remember_verdict(&mut self, result: SolverResult) {
        self.last_check = Some((self.goal_fingerprint(), result));
    }
}
