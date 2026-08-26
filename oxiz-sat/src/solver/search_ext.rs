//! Extended search entry points split out of `solver/mod.rs`.
//!
//! Currently hosts the CDCL(T) search loop [`Solver::solve_with_theory`]. It
//! lives in its own file so `solver/mod.rs` stays under the 2000-line limit
//! while keeping all `impl Solver` search variants close together.

use super::*;

impl Solver {
    /// Solve with theory integration via callbacks
    ///
    /// This implements the CDCL(T) loop:
    /// 1. BCP (Boolean Constraint Propagation)
    /// 2. Theory propagation (via callback)
    /// 3. On conflict: analyze and learn
    /// 4. Decision
    /// 5. Final theory check when all vars assigned
    ///
    /// # Propagation fixpoint invariant
    ///
    /// Every conflict handler here ends by rejoining the **outer** `'search`
    /// loop, whose first act is `propagate()`. That is what makes step 4/5 sound:
    /// a decision, and above all the `final_check`/`save_model` step that answers
    /// `Sat`, may only run once Boolean propagation has reached a fixpoint over
    /// the *whole* clause database.
    ///
    /// Handling a conflict inside the theory loop and rejoining that inner loop
    /// instead — which is what the theory-conflict branches used to do — skips
    /// BCP: the learned clause's asserting literal (for a unit lemma, a fresh
    /// level-0 fact) is appended to the trail but never propagated. If the theory
    /// conflict happens to resolve the *last* unassigned variable, `pick_branch_var`
    /// then reports "all assigned", `final_check` sees a theory-consistent atom
    /// assignment and answers `Sat` — over a trail on which an **original** clause
    /// is already falsified by level-0 facts alone. The instance is `Unsat` and
    /// the caller is handed a model that does not satisfy the formula.
    ///
    /// # LRAT tracing is unsupported here
    ///
    /// A theory-propagated trail literal carries [`Reason::Theory`] — no
    /// clause backs it, so no LRAT hint chain can ever cite one no matter
    /// how it is built (see `solver/lrat_trace.rs`'s antecedent-closure walk
    /// — a private module, not part of this crate's public API). Rather
    /// than emit an LRAT proof that is silently
    /// incomplete the moment a theory conflict or propagation touches the
    /// trail, LRAT tracing is force-disabled the instant this entry point
    /// runs (DRAT is unaffected: it stays self-justifying for any *Boolean*
    /// clause this loop learns, independent of what the theory layer did —
    /// the caller is responsible for the theory layer's own proof obligations,
    /// which are outside this crate's boundary).
    pub fn solve_with_theory<T: TheoryCallback>(&mut self, theory: &mut T) -> SolverResult {
        self.disable_lrat_proof();
        // See `Solver::solve`'s identical guard: a prior `add_clause` may
        // have tried to reintroduce a bounded-variable-eliminated variable.
        if self.fatal_error.is_some() {
            return SolverResult::Unknown;
        }
        if self.trivially_unsat {
            return SolverResult::Unsat;
        }

        // Initial propagation
        if self.propagate().is_some() {
            return SolverResult::Unsat;
        }

        // Track how many assignments have been sent to the theory.
        // We only send NEW assignments (not previously processed ones) to avoid
        // duplicate theory constraints that would cause spurious UNSAT.
        let mut theory_processed: usize = 0;

        'search: loop {
            // Resource budget / interrupt check: honor a configured conflict
            // limit or an external interrupt by returning Unknown.
            if self.should_stop_search() {
                return SolverResult::Unknown;
            }

            // Boolean propagation
            if let Some(conflict) = self.propagate() {
                self.stats.conflicts += 1;

                if self.trail.decision_level() == 0 {
                    return SolverResult::Unsat;
                }

                let (backtrack_level, learnt_clause) = self.analyze(conflict);

                // Empty learned clause = genuine root-level (level-0) refutation:
                // the conflict clause is falsified under unconditional facts alone,
                // so the instance is UNSAT. `analyze` returns this even when the
                // trail sits above decision level 0 (an on-the-fly clause added
                // already-falsified at the root).
                if learnt_clause.is_empty() {
                    self.trivially_unsat = true;
                    return SolverResult::Unsat;
                }

                theory.on_backtrack(backtrack_level);
                // Clamp the theory cursor to the rollback boundary, not to the
                // trail length: chronological backtracking re-appends the
                // literals that survive the rollback above that boundary, and
                // the theory — which was just told to unwind to
                // `backtrack_level` — has to see them again.
                let boundary = self.backtrack_with_phase_saving(backtrack_level);
                theory_processed = theory_processed.min(boundary);
                self.learn_clause(learnt_clause);

                self.vsids.decay();
                self.clauses.decay_activity(self.config.clause_decay);
                self.handle_deletion_restart_with_theory(theory, &mut theory_processed);
                continue;
            }

            // Theory propagation check after each assignment
            loop {
                // Only the UNPROCESSED suffix of the trail is new information for
                // the theory: everything before `safe_start` was already sent to
                // `theory.on_assignment` in a prior iteration (or survived a
                // backtrack's clamp below `boundary`, in which case it will be
                // re-sent only once the trail regrows past it) — except on the
                // `Conflict` break below, which marks the *whole* suffix processed
                // even though delivery stopped at the conflicting literal; that
                // relies on `theory_processed.min(boundary)` after the ensuing
                // backtrack to reopen anything left undelivered. Pre-existing
                // behavior (the old `assignments.len()` did the same), unchanged
                // here. Cloning the whole trail here every iteration made this
                // loop O(trail_len^2) over a search (each of the trail's N
                // literals re-copied on each of the ~N later iterations); slicing
                // `[safe_start..]` makes each iteration's work proportional to
                // what is actually new.
                //
                // `TheoryCallback::on_assignment` takes `&mut T` (the theory), not
                // `&mut Solver`, so nothing in this loop body needs a mutable
                // borrow of `self` — an immutable slice borrow of `self.trail` can
                // live for the loop's duration without a clone.
                let trail_len = self.trail.assignments().len();
                // Guard against stale theory_processed after backtracks/restarts.
                let safe_start = theory_processed.min(trail_len);
                let mut theory_conflict = None;
                let mut theory_propagations = Vec::new();

                for &lit in &self.trail.assignments()[safe_start..] {
                    match theory.on_assignment(lit) {
                        TheoryCheckResult::Sat => {}
                        TheoryCheckResult::Conflict(conflict_lits) => {
                            theory_conflict = Some(conflict_lits);
                            break;
                        }
                        TheoryCheckResult::Propagated(props) => {
                            theory_propagations.extend(props);
                        }
                    }
                }
                // Update processed count: everything up to the trail length we
                // read above has now been sent to the theory.
                theory_processed = trail_len;

                // Handle theory conflict
                if let Some(conflict_lits) = theory_conflict {
                    self.stats.conflicts += 1;

                    if self.trail.decision_level() == 0 {
                        return SolverResult::Unsat;
                    }

                    let (backtrack_level, learnt_clause) =
                        self.analyze_theory_conflict(&conflict_lits);

                    // Empty learned clause signals all-level-0 conflict = fundamental UNSAT
                    if learnt_clause.is_empty() {
                        self.trivially_unsat = true;
                        return SolverResult::Unsat;
                    }

                    theory.on_backtrack(backtrack_level);
                    let boundary = self.backtrack_with_phase_saving(backtrack_level);
                    theory_processed = theory_processed.min(boundary);
                    self.learn_clause(learnt_clause);

                    self.vsids.decay();
                    self.clauses.decay_activity(self.config.clause_decay);
                    self.handle_deletion_restart_with_theory(theory, &mut theory_processed);
                    // Rejoin the outer loop, NOT this one: the clause just learned
                    // put its asserting literal on the trail unpropagated, and only
                    // `'search`'s leading `propagate()` closes that gap. See the
                    // propagation-fixpoint invariant on `solve_with_theory`.
                    continue 'search;
                }

                // Handle theory propagations
                let mut made_propagation = false;
                for (lit, reason_lits) in theory_propagations {
                    if !self.trail.is_assigned(lit.var()) {
                        // Add reason clause and propagate
                        let clause_id = self.add_theory_reason_clause(&reason_lits, lit);
                        self.trail.assign_propagation(lit, clause_id);
                        made_propagation = true;
                    }
                }

                if made_propagation {
                    // Re-run Boolean propagation
                    if let Some(conflict) = self.propagate() {
                        self.stats.conflicts += 1;

                        if self.trail.decision_level() == 0 {
                            return SolverResult::Unsat;
                        }

                        let (backtrack_level, learnt_clause) = self.analyze(conflict);

                        // Empty learned clause = genuine root-level (level-0)
                        // refutation → UNSAT (see the companion guard above).
                        if learnt_clause.is_empty() {
                            self.trivially_unsat = true;
                            return SolverResult::Unsat;
                        }

                        theory.on_backtrack(backtrack_level);
                        let boundary = self.backtrack_with_phase_saving(backtrack_level);
                        theory_processed = theory_processed.min(boundary);
                        self.learn_clause(learnt_clause);

                        self.vsids.decay();
                        self.clauses.decay_activity(self.config.clause_decay);
                        self.handle_deletion_restart_with_theory(theory, &mut theory_processed);
                        // Same reason as the theory-conflict branch above: the
                        // learned clause left an unpropagated asserting literal.
                        continue 'search;
                    }
                    continue;
                }

                break;
            }

            // The theory loop is quiescent. Boolean propagation must be at a
            // fixpoint before a decision is taken and, critically, before
            // `final_check` is allowed to answer `Sat` over this trail. Every path
            // that assigns without propagating rejoins `'search` above, so this
            // guard is the belt to that braces — one comparison, and it makes the
            // invariant hold no matter how the branches above are later edited.
            if self.trail.has_pending_propagation() {
                continue 'search;
            }

            // Try to decide
            if let Some(var) = self.pick_branch_var() {
                self.stats.decisions += 1;
                self.trail.new_decision_level();
                let new_level = self.trail.decision_level();
                theory.on_new_level(new_level);

                let polarity = if self.rand_bool(self.config.random_polarity_prob) {
                    self.rand_bool(0.5)
                } else {
                    self.phase[var.index()] ^ self.phase_inverted
                };
                let lit = if polarity {
                    Lit::pos(var)
                } else {
                    Lit::neg(var)
                };
                self.trail.assign_decision(lit);
            } else {
                // All variables assigned - do final theory check
                match theory.final_check() {
                    TheoryCheckResult::Sat => {
                        self.save_model();
                        self.debug_verify_model_input();
                        return SolverResult::Sat;
                    }
                    TheoryCheckResult::Conflict(conflict_lits) => {
                        self.stats.conflicts += 1;

                        if self.trail.decision_level() == 0 {
                            return SolverResult::Unsat;
                        }

                        let (backtrack_level, learnt_clause) =
                            self.analyze_theory_conflict(&conflict_lits);

                        // If all conflict literals are at level 0, analyze_theory_conflict
                        // returns an empty learned clause as a signal of fundamental UNSAT.
                        if learnt_clause.is_empty() {
                            self.trivially_unsat = true;
                            return SolverResult::Unsat;
                        }

                        theory.on_backtrack(backtrack_level);
                        let boundary = self.backtrack_with_phase_saving(backtrack_level);
                        theory_processed = theory_processed.min(boundary);
                        self.learn_clause(learnt_clause);

                        self.vsids.decay();
                        self.clauses.decay_activity(self.config.clause_decay);
                        self.handle_deletion_restart_with_theory(theory, &mut theory_processed);
                    }
                    TheoryCheckResult::Propagated(props) => {
                        // Handle late propagations
                        for (lit, reason_lits) in props {
                            if !self.trail.is_assigned(lit.var()) {
                                let clause_id = self.add_theory_reason_clause(&reason_lits, lit);
                                self.trail.assign_propagation(lit, clause_id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run clause-database reduction and the restart check, keeping the theory's
    /// view of the trail in sync.
    ///
    /// A restart backtracks the trail (to level 0 for the global strategies, or a
    /// local level for `LocalLbd`) purely inside the Boolean core — `restart()`
    /// only holds `&mut self` and cannot reach the theory. Without notifying the
    /// theory, its per-atom polarity bookkeeping keeps the assignments the restart
    /// just discarded and, on the next check, reports a "conflict" whose clause
    /// still lists those now-unassigned literals. That stale clause is not a real
    /// conflict (its open literals are unassigned), and feeding it into
    /// conflict analysis corrupts the trail (see `analyze_theory_conflict`). By
    /// detecting the trail shrinking and forwarding the new level through
    /// `on_backtrack`, the theory unwinds exactly what the Boolean core did, so no
    /// stale literal survives into the next theory check. `theory_processed` is
    /// clamped to the shortened trail so the newly-restored prefix is re-sent to
    /// the theory on the following iteration.
    fn handle_deletion_restart_with_theory<T: TheoryCallback>(
        &mut self,
        theory: &mut T,
        theory_processed: &mut usize,
    ) {
        let level_before = self.trail.decision_level();
        self.handle_clause_deletion_and_restart();
        let level_after = self.trail.decision_level();
        if level_after < level_before {
            theory.on_backtrack(level_after);
            // The restart backtracked inside `handle_clause_deletion_and_restart`,
            // so the rollback boundary is not returned here. The propagation head
            // is rewound to that boundary by every rollback, so it is a safe (never
            // too large) stand-in — important under chronological backtracking,
            // where literals surviving the rollback are re-appended above the
            // boundary and must be re-sent to the theory.
            *theory_processed = (*theory_processed).min(self.trail.propagation_head());
        }
    }
}
