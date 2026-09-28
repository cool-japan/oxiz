//! [`Solver::solve_with_assumptions`]: search under assumption literals, with
//! the failed-assumption core on `Unsat`.
//!
//! # The invariant this module keeps, and the defect it replaced
//!
//! Assumption `assumptions[i]` owns decision level `i + 1`: it is either the
//! decision of that level or — when something below already implied it — the
//! level is opened empty.  Only once every assumption holds does the search
//! branch on anything else.  This is MiniSat's scheme, and its point is that
//! the assumptions are **re-decided** whenever the search drops below them:
//! after a backjump to a level `j < k`, and after a restart to level 0, the
//! decide step finds `decision_level() < assumptions.len()` again and puts the
//! next assumption back before any free choice is made.
//!
//! Until re-fix pass 12 (2026-09-28) the assumptions were decided **once**,
//! before the loop.  A conflict then backjumped to `max(bt, 1)` and a limited
//! restart went to level 0, and nothing put the undone assumptions back:
//! `pick_branch_var` treated them as ordinary variables and a propagation
//! could set one false.  The call still answered `Sat`, with a model that
//! falsified an assumption — so a caller asking "is `F ∧ a₁ ∧ … ∧ aₖ`
//! satisfiable" got `sat` for an unsatisfiable question.  Pinned by
//! `oxiz-sat/tests/assumption_retention.rs`, which compares every answer with
//! a fresh solver given the assumptions as unit clauses.
//!
//! # Why a learned clause may outlive the call
//!
//! An assumption is placed as a *decision*, never as a clause, so conflict
//! analysis treats it like any other decision: every learned clause is a
//! resolvent of clauses in the database and holds under **every** assumption
//! set.  A caller may therefore keep the learned clauses across calls with
//! different assumptions — which is what makes this entry point usable as an
//! incremental theory check (`oxiz-theories`' bit-vector solver does exactly
//! that) and not only as a MaxSAT core extractor.

use super::*;

impl Solver {
    /// Solve with assumptions and return unsat core if UNSAT
    ///
    /// This is the key method for MaxSAT: it solves under assumptions and
    /// if the result is UNSAT, returns the subset of assumptions in the core.
    ///
    /// # Arguments
    /// * `assumptions` - Literals that must be true
    ///
    /// # Returns
    /// * `(SolverResult, Option<Vec<Lit>>)` - Result and unsat core (if UNSAT).
    ///   A `Sat` model satisfies every assumption; an `Unsat` core is a subset
    ///   of `assumptions` that is unsatisfiable together with the clause set
    ///   (empty when the clause set is unsatisfiable on its own).
    ///
    /// The search ends at decision level 0 on every exit, and every clause it
    /// learned stays in the database (see the module documentation for why
    /// that is sound under a later, different assumption set).
    ///
    /// # LRAT tracing is unsupported here
    ///
    /// This entry point's clause-learning goes through `Solver::learn_clause`
    /// (a private method, not part of this crate's public API) rather than
    /// the plain [`Solver::solve`] loop's hint-chain-aware inline
    /// path, and an assumption literal is installed without going through
    /// [`Solver::add_clause`] (so it has no original-clause LRAT id to be
    /// justified by regardless). Rather than emit an LRAT proof this port
    /// cannot back with a real hint chain, LRAT tracing is force-disabled the
    /// instant this entry point runs (DRAT is unaffected — `learn_clause`
    /// already emits it correctly, self-justifying, independent of this gap).
    ///
    /// # Unsat core is expressed in the caller's own literals
    ///
    /// Internally, an assumption may be rewritten before it is decided on
    /// (see `resolve_reintroduced_literal`, a private method not part of
    /// this crate's public API: an equivalent-literal-substituted variable
    /// becomes its class representative). A core drawn from those *resolved*
    /// literals is translated back to the caller's originals (via
    /// `translate_core_to_original`, likewise private) before this method
    /// returns, so `core ⊆ assumptions` — a genuine subset of exactly what
    /// was passed in — always holds, matching what a MaxSAT-style caller
    /// keying relaxations on its own selector literals expects.
    pub fn solve_with_assumptions(
        &mut self,
        assumptions: &[Lit],
    ) -> (SolverResult, Option<Vec<Lit>>) {
        self.disable_lrat_proof();
        // See `Solver::solve`'s identical guard: a prior `add_clause` may
        // have tried to reintroduce a bounded-variable-eliminated variable.
        if self.fatal_error.is_some() {
            return (SolverResult::Unknown, None);
        }
        if self.trivially_unsat {
            return (SolverResult::Unsat, Some(Vec::new()));
        }

        // Ensure all assumption variables exist
        for &lit in assumptions {
            while self.num_vars <= lit.var().index() {
                self.new_var();
            }
        }

        // Resolve each assumption literal exactly like a fresh `add_clause`
        // literal (see `Self::resolve_reintroduced_literal`); any core handed
        // back is translated to `original_assumptions` before it is returned.
        let original_assumptions = assumptions;
        let mut resolved_assumptions: Vec<Lit> = Vec::with_capacity(assumptions.len());
        for &lit in assumptions {
            match self.resolve_reintroduced_literal(lit) {
                Some(resolved) => resolved_assumptions.push(resolved),
                None => return (SolverResult::Unknown, None),
            }
        }
        let assumptions: &[Lit] = &resolved_assumptions;

        // A prior solve() may have returned Sat while leaving its full model on
        // the trail; start from the root, with the conflict-analysis marks
        // clean, so leftover decisions never masquerade as facts.
        self.backtrack_with_phase_saving(0);
        for s in &mut self.seen {
            *s = false;
        }

        // Initial propagation at level 0: a conflict here is the clause set's
        // own, and owes the caller an empty core.
        if self.propagate().is_some() {
            return (SolverResult::Unsat, Some(Vec::new()));
        }

        // Lucky phase, the assumption-aware variant: the resolved assumption
        // literals are seeded into the candidate as frozen values, so a model
        // it reports satisfies them by construction (see `solver/lucky.rs`).
        if self.try_lucky_phase(assumptions).is_some() {
            return (SolverResult::Sat, None);
        }

        loop {
            // Resource budget / interrupt check.
            if self.should_stop_search() {
                self.backtrack(0);
                return (SolverResult::Unknown, None);
            }

            if let Some(conflict) = self.propagate() {
                self.debug_check_conflict_clause(conflict);
                self.stats.conflicts += 1;

                // No decision on the trail: the clause set refutes itself.
                if self.trail.decision_level() == 0 {
                    return (SolverResult::Unsat, Some(Vec::new()));
                }

                let (bt_level, learnt_clause) = self.analyze(conflict);
                if learnt_clause.is_empty() {
                    self.backtrack(0);
                    return (SolverResult::Unsat, Some(Vec::new()));
                }

                // The backjump may land below the assumption levels; the decide
                // step below re-decides whatever it undid (module docs).
                self.backtrack_with_phase_saving(bt_level);
                self.debug_check_invariants("after backtrack (assumptions)");
                self.learn_clause(learnt_clause);

                self.vsids.decay();
                self.clauses.decay_activity(self.config.clause_decay);
                self.handle_clause_deletion_and_restart_limited(0);
                continue;
            }

            // Propagation fixpoint: the next assumption first, one decision
            // level each, and only then a free branch.
            //
            // The debug-only fixpoint net (`debug_check_fixpoint_invariants`,
            // a scan of every clause) runs where the verdict is produced — at
            // the `Sat` exit below — and not at every fixpoint: an incremental
            // caller issues thousands of these calls, each placing hundreds of
            // assumptions, over a clause set that only grows (`oxiz-theories`'
            // bit-vector solver, decision (45)), and a scan per decision made
            // its debug-build tests two orders of magnitude slower.  `solve`
            // keeps the per-fixpoint scan.
            let mut next: Option<Lit> = None;
            while (self.trail.decision_level() as usize) < assumptions.len() {
                let index = self.trail.decision_level() as usize;
                let assumption = assumptions[index];
                let value = self.trail.lit_value(assumption);
                if value.is_true() {
                    // Already implied below: an empty level keeps assumption
                    // `i` at level `i + 1`.
                    self.trail.new_decision_level();
                } else if value.is_false() {
                    // Refuted by the assumptions before it (or by a level-0
                    // fact): the core is the assumptions its falsity rests on,
                    // plus itself.
                    let core = self.extract_assumption_core(assumptions, index);
                    self.backtrack(0);
                    let core =
                        Self::translate_core_to_original(core, assumptions, original_assumptions);
                    return (SolverResult::Unsat, Some(core));
                } else {
                    next = Some(assumption);
                    break;
                }
            }

            let decision = match next {
                Some(assumption) => assumption,
                None => match self.pick_branch_var() {
                    Some(var) => {
                        self.stats.decisions += 1;
                        let polarity = if self.rand_bool(self.config.random_polarity_prob) {
                            self.rand_bool(0.5)
                        } else {
                            self.phase.get(var.index()).copied().unwrap_or(false)
                                ^ self.phase_inverted
                        };
                        if polarity {
                            Lit::pos(var)
                        } else {
                            Lit::neg(var)
                        }
                    }
                    None => {
                        // All variables assigned, every assumption among them.
                        self.debug_check_fixpoint_invariants("at SAT (assumptions)");
                        self.save_model();
                        self.debug_verify_model();
                        self.debug_check_invariants("at SAT (assumptions)");
                        self.backtrack(0);
                        return (SolverResult::Sat, None);
                    }
                },
            };
            self.trail.new_decision_level();
            self.trail.assign_decision(decision);
        }
    }

    /// Translate a core drawn from `resolved` (the literals
    /// [`Self::solve_with_assumptions`] actually decided on) back to the
    /// caller's `original` assumption literals it corresponds to.
    ///
    /// `resolved[i]` is what `original[i]` became after
    /// [`Self::resolve_reintroduced_literal`]; a core literal that does not
    /// match any position in `resolved` (should not happen — every core
    /// literal comes from `resolved` in the first place) is passed through
    /// unchanged rather than dropped, so a coding error here fails toward
    /// "core has an unexpected literal" rather than silently shrinking the
    /// core. First occurrence wins on a duplicate resolved literal, matching
    /// how `analyze_final_core`'s own `assumption_of` map is built from this
    /// same `resolved` list.
    fn translate_core_to_original(core: Vec<Lit>, resolved: &[Lit], original: &[Lit]) -> Vec<Lit> {
        core.into_iter()
            .map(|lit| {
                resolved
                    .iter()
                    .position(|&r| r == lit)
                    .map_or(lit, |i| original[i])
            })
            .collect()
    }
}
