//! Original-clause insertion — [`Solver::add_clause`] and its DIMACS
//! wrapper — split out of `solver/mod.rs` to keep that file under the
//! 2000-line limit while keeping the (fairly involved) pre-attach
//! conflict/effective-unit resolution next to the one caller that needs it.

use super::*;

/// Outcome of [`Solver::pre_check_effective_unit`], resolved *before* any
/// watches are chosen for a new clause in [`Solver::add_clause`].
enum PreAttachOutcome {
    /// Already satisfied by the current assignment, or simply not
    /// effectively unit (2+ literals still undefined). Either way, nothing
    /// special is needed: add and watch the clause normally.
    Ordinary,
    /// Every literal is false and, after resolving level-0-only facts via
    /// [`Solver::backtrack_to_root`] where needed, still is: an
    /// unconditional (level-0) conflict. The caller must set
    /// `trivially_unsat` and return `false` without adding the clause.
    UnconditionalConflict,
    /// The clause is an effective unit (every literal false except this one,
    /// which is undefined) and every false literal is confirmed to be a
    /// permanent level-0 fact. The caller must force this literal via
    /// `Trail::assign_propagation_at(_, clause_id, 0)` once the clause has
    /// been inserted and its `ClauseId` is known (not yet, at the point this
    /// outcome is produced).
    ForceUnitAtLevelZero(Lit),
}

impl Solver {
    /// Scan `clause_lits` against the *current* trail: is any literal true,
    /// what is the highest level among the false literals (0 if there are
    /// none), and which literals are still undefined.
    ///
    /// Read-only. Used by [`Solver::pre_check_effective_unit`] both before
    /// and (when it backtracks) after a `backtrack_to_root()` call, so it
    /// must not itself assume anything about levels.
    pub(super) fn scan_clause_for_attach(
        &self,
        clause_lits: &[Lit],
    ) -> (bool, u32, SmallVec<[Lit; 4]>) {
        let mut has_true = false;
        let mut max_false_level = 0u32;
        let mut undefined: SmallVec<[Lit; 4]> = SmallVec::new();
        for &lit in clause_lits {
            let value = self.trail.lit_value(lit);
            if value.is_true() {
                has_true = true;
                break;
            } else if value.is_false() {
                max_false_level = max_false_level.max(self.trail.level(lit.var()));
            } else {
                undefined.push(lit);
            }
        }
        (has_true, max_false_level, undefined)
    }

    /// Resolve `clause_lits`'s conflict / effective-unit status against the
    /// current trail, performing any necessary backtrack, *before* the
    /// caller chooses which literals to watch.
    ///
    /// # Why this must run before watch selection
    ///
    /// The two-watched-literal ranking (`watch_rank` and its call sites in
    /// `add_clause`) is computed against whatever the trail looks like when
    /// it runs. A `backtrack_to_root()` performed *after* that ranking would
    /// silently invalidate it: literals the ranking saw as false may now be
    /// undefined, so the "watch the two latest-falsified literals" choice it
    /// made is no longer meaningful. Running this check first, and letting
    /// its backtrack (if any) land before ranking ever executes, keeps the
    /// two steps consistent with each other.
    ///
    /// # Why "effectively unit" needs the same treatment as "all false"
    ///
    /// A clause is only safe to attach as-is when every literal that is
    /// currently false is false *permanently* (at level 0). A literal false
    /// above level 0 can be unassigned by a future backtrack while some
    /// *other* disjunct of the clause survives (in particular, an implied
    /// literal this same function forces at the wrong level would -- see the
    /// history of this function for the bug that motivated this rewrite):
    /// the clause is then silently reopened, with no live watcher able to
    /// notice, because watch/graph registration only fires on a literal's
    /// *next* transition, not because of anything a backtrack does. This is
    /// true whether the clause is fully false (a conflict) or has exactly
    /// one literal left undefined (an effective unit) -- both are handled by
    /// the same rule here.
    ///
    /// `backtrack_to_root()` resolves the ambiguity outright: every literal
    /// false above level 0 becomes undefined, so a mandatory re-scan
    /// afterward finds either 2+ undefined literals (ordinary watching is
    /// then correct and sufficient -- the clause is genuinely open again) or
    /// still at most one undefined literal, with every remaining false
    /// literal now unconditionally at level 0 (forced at level 0, which
    /// survives every future backtrack by construction).
    fn pre_check_effective_unit(&mut self, clause_lits: &[Lit]) -> PreAttachOutcome {
        let (has_true, max_false_level, undefined) = self.scan_clause_for_attach(clause_lits);
        if has_true || undefined.len() >= 2 {
            return PreAttachOutcome::Ordinary;
        }

        if max_false_level > 0 {
            self.backtrack_to_root();
            // Mandatory re-scan: the sets computed above are now stale.
            let (has_true, _post_backtrack_max_level, undefined) =
                self.scan_clause_for_attach(clause_lits);
            debug_assert!(
                !has_true,
                "backtrack_to_root() cannot turn a false/undefined literal true"
            );
            return if undefined.is_empty() {
                PreAttachOutcome::UnconditionalConflict
            } else if undefined.len() == 1 {
                PreAttachOutcome::ForceUnitAtLevelZero(undefined[0])
            } else {
                PreAttachOutcome::Ordinary
            };
        }

        if undefined.is_empty() {
            PreAttachOutcome::UnconditionalConflict
        } else {
            PreAttachOutcome::ForceUnitAtLevelZero(undefined[0])
        }
    }

    /// Add a clause.
    ///
    /// Returns `true` when the clause was accepted (it may still have been a
    /// tautology, or already satisfied — either way nothing was left
    /// unsound). Returns `false` in two distinct situations, distinguishable
    /// via [`Solver::error`]:
    ///
    /// - The clause made the instance trivially UNSAT (an empty clause, or a
    ///   unit that contradicts an existing level-0 fact): `Solver::error`
    ///   reports `None`, and [`Solver::solve`] will correctly answer `Unsat`.
    /// - The clause named a variable [`Solver::var_eliminated`] reports as
    ///   bounded-variable-eliminated, with no sound way to honor it (see
    ///   [`SolverError::EliminatedVariableReintroduction`]):
    ///   `Solver::error` reports `Some(..)`, and every `solve*` entry point
    ///   will refuse to answer `Sat`/`Unsat` from here on rather than risk
    ///   either being wrong.
    pub fn add_clause(&mut self, lits: impl IntoIterator<Item = Lit>) -> bool {
        let mut clause_lits: SmallVec<[Lit; 8]> = lits.into_iter().collect();

        // Ensure we have all variables
        for lit in &clause_lits {
            let var_idx = lit.var().index();
            if var_idx >= self.num_vars {
                self.ensure_vars(var_idx + 1);
            }
        }

        // A caller can add a clause mentioning a variable the one-shot
        // inprocessing toolkit already eliminated from an earlier `solve()`
        // on this same (incremental) solver — see
        // `Self::resolve_reintroduced_literal`'s doc comment for the two
        // cases and why they are handled differently. Every literal is
        // checked, not just ones already known to be eliminated: the lookup
        // is cheap and uniform (identity for the ordinary case), and
        // skipping it would mean re-deriving `var_eliminated` here too.
        let mut resolved_lits: SmallVec<[Lit; 8]> = SmallVec::with_capacity(clause_lits.len());
        for &lit in &clause_lits {
            match self.resolve_reintroduced_literal(lit) {
                Some(resolved) => resolved_lits.push(resolved),
                None => return false, // `self.fatal_error` now explains why.
            }
        }
        clause_lits = resolved_lits;

        // Remove duplicates and check for tautology
        clause_lits.sort_by_key(|l| l.code());
        clause_lits.dedup();

        // Register with LRAT tracing (no-op, `None`, when it is off) before
        // any of the special-casing below: this way "one original-clause
        // registration per `add_clause` call, in call order" holds
        // regardless of which branch a particular call falls into, matching
        // how a caller who also writes a DIMACS file alongside these calls
        // would number that file's clause lines.
        let original_lrat_id = self.lrat_register_original();

        // Check for tautology (x and ~x in same clause)
        for i in 0..clause_lits.len() {
            for j in (i + 1)..clause_lits.len() {
                if clause_lits[i] == clause_lits[j].negate() {
                    return true; // Tautology - always satisfied
                }
            }
        }

        // Handle special cases
        match clause_lits.len() {
            0 => {
                self.trivially_unsat = true;
                // A literally empty original clause is, on its own, a
                // complete LRAT proof of UNSAT — no derived empty-clause
                // line is needed on top of it.
                self.lrat_mark_finalized_by_original_empty();
                return false; // Empty clause - unsat
            }
            1 => {
                // Unit clause - enqueue at decision level 0
                // Unit clauses must be assigned at level 0 to survive backtracking.
                // After solve(), current_level may be > 0, so we must backtrack first.
                let lit = clause_lits[0];

                if self.trail.lit_value(lit).is_false() {
                    // The literal conflicts with the current trail.
                    // Check if the conflict is at decision level 0 (permanent constraint)
                    // or from a previous solve (can be retried after backtrack).
                    let var = lit.var();
                    let level = self.trail.level(var);
                    if level == 0 {
                        // Conflict with a level-0 assignment - truly UNSAT.
                        // Crucially, `lit.var()`'s *existing* value (the
                        // opposite polarity) is untouched here — this new
                        // clause never becomes `lit.var()`'s justification,
                        // so `lrat_unit_id` must not be overwritten with it;
                        // doing so would make a later hint chain cite the
                        // very clause that *contradicts* the trail instead
                        // of the one that actually put it there.
                        self.trivially_unsat = true;
                        // The new unit clause and the existing level-0 fact
                        // it contradicts, together, already contain the
                        // empty clause: hint-chain from the new clause's own
                        // literal (it is the one fully falsified) and id.
                        self.lrat_emit_empty_from(&clause_lits, original_lrat_id.unwrap_or(0));
                        return false;
                    } else {
                        // Conflict with higher-level assignment from previous solve.
                        // Backtrack to root and assign the new unit literal at level 0.
                        self.backtrack_to_root();
                        self.trail.assign_decision(lit);
                        // The backtrack just unassigned the old (higher-level)
                        // value; this original clause is the fresh level-0
                        // justification for the value now on the trail.
                        if let Some(id) = original_lrat_id {
                            self.lrat_set_unit_justification(var, id);
                        }
                        return true;
                    }
                }

                if self.trail.lit_value(lit).is_true() {
                    // Already satisfied - check if at level 0
                    let var = lit.var();
                    let level = self.trail.level(var);
                    if level == 0 {
                        // Already assigned at level 0 by an earlier
                        // registration, nothing to do — that earlier
                        // registration's id remains `lit.var()`'s
                        // justification, still valid for this same polarity.
                        return true;
                    }
                    // Assigned at higher level - backtrack and reassign at level 0
                    self.backtrack_to_root();
                    self.trail.assign_decision(lit);
                    // Same reasoning as the higher-level conflict case above:
                    // the old (higher-level) justification is gone, this
                    // clause is the fresh one.
                    if let Some(id) = original_lrat_id {
                        self.lrat_set_unit_justification(var, id);
                    }
                    return true;
                }

                // Variable is unassigned - backtrack to level 0 first to ensure
                // the assignment is at level 0 (survives future backtracks)
                if self.trail.decision_level() > 0 {
                    self.backtrack_to_root();
                }
                self.trail.assign_decision(lit);
                if let Some(id) = original_lrat_id {
                    self.lrat_set_unit_justification(lit.var(), id);
                }
                return true;
            }
            2 => {
                // Binary clause - check if it conflicts with current assignment
                let lit0 = clause_lits[0];
                let lit1 = clause_lits[1];
                let val0 = self.trail.lit_value(lit0);
                let val1 = self.trail.lit_value(lit1);

                // If clause is satisfied, just add it
                if val0.is_true() || val1.is_true() {
                    // Clause already satisfied by current assignment
                    let clause_id = self.clauses.add_original(clause_lits.iter().copied());
                    if let Some(id) = original_lrat_id {
                        self.lrat_set_clause_id(clause_id, id);
                    }
                    if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                        current_level_clauses.push(clause_id);
                    }
                    self.binary_graph.add(lit0.negate(), lit1, clause_id);
                    self.binary_graph.add(lit1.negate(), lit0, clause_id);
                    self.watches
                        .add(lit0.negate(), Watcher::new(clause_id, lit1));
                    self.watches
                        .add(lit1.negate(), Watcher::new(clause_id, lit0));
                    return true;
                }

                // Resolve conflict / effective-unit status *before*
                // attaching the clause -- see `pre_check_effective_unit`'s
                // doc comment for the full reasoning (in particular why an
                // "effectively unit" binary clause, not just an "all false"
                // one, needs its level bookkeeping resolved this way: the
                // watches registered below cannot be trusted to discover it
                // on their own, since they only fire on a literal's *next*
                // transition -- a level-0 fact from earlier in this
                // incremental session was already dequeued long ago and will
                // never be dequeued again).
                let outcome = self.pre_check_effective_unit(&clause_lits);
                if matches!(outcome, PreAttachOutcome::UnconditionalConflict) {
                    self.trivially_unsat = true;
                    // This just-registered clause is itself the one fully
                    // falsified.
                    self.lrat_emit_empty_from(&clause_lits, original_lrat_id.unwrap_or(0));
                    return false;
                }

                let clause_id = self.clauses.add_original(clause_lits.iter().copied());
                if let Some(id) = original_lrat_id {
                    self.lrat_set_clause_id(clause_id, id);
                }
                if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                    current_level_clauses.push(clause_id);
                }
                self.binary_graph.add(lit0.negate(), lit1, clause_id);
                self.binary_graph.add(lit1.negate(), lit0, clause_id);
                self.watches
                    .add(lit0.negate(), Watcher::new(clause_id, lit1));
                self.watches
                    .add(lit1.negate(), Watcher::new(clause_id, lit0));

                if let PreAttachOutcome::ForceUnitAtLevelZero(forced) = outcome {
                    self.trail.assign_propagation_at(forced, clause_id, 0);
                }
                return true;
            }
            _ => {}
        }

        // Add clause (3+ literals)
        // Resolve conflict / effective-unit status *before* choosing watches
        // -- see `pre_check_effective_unit`'s doc comment. Must run before
        // the `watch_rank` selection below: a `backtrack_to_root()` decided
        // on afterward would silently invalidate whatever ranking that
        // selection just computed.
        let outcome = self.pre_check_effective_unit(&clause_lits);
        if matches!(outcome, PreAttachOutcome::UnconditionalConflict) {
            self.trivially_unsat = true;
            // This just-registered clause is itself the one fully
            // falsified.
            self.lrat_emit_empty_from(&clause_lits, original_lrat_id.unwrap_or(0));
            return false;
        }

        // Choose the two watch literals *before* storing the clause, following
        // MiniSat's attachClause invariant: watch the two literals that are the
        // last to become false under the current assignment. Ranking prefers a
        // true literal, then an unassigned one, and only then a false literal at
        // the highest decision level (see `watch_rank`).
        //
        // The previous code unconditionally watched `clause_lits[0..2]`. After a
        // prior `solve()` left a full trail (with `prop_head == len`), a clause
        // whose two lowest-code literals are false-but-already-propagated would
        // have both watches on false literals; those watch events never fire
        // again, so the clause could be silently falsified. A later `solve()`
        // could then return Sat on a model violating the clause, or miss a
        // conflict on an actually-UNSAT formula. Watching the two
        // latest-falsified literals restores the invariant that a watched
        // literal becoming false always re-examines the clause.
        //
        // Safe to run *after* `pre_check_effective_unit` above: any
        // `backtrack_to_root()` it performed has already happened, so this
        // ranking sees the final, post-backtrack trail state rather than one
        // that gets invalidated out from under it.
        let n = clause_lits.len();
        let mut best = 0;
        for i in 1..n {
            if self.watch_rank(clause_lits[i]) > self.watch_rank(clause_lits[best]) {
                best = i;
            }
        }
        clause_lits.swap(0, best);
        let mut second = 1;
        for i in 2..n {
            if self.watch_rank(clause_lits[i]) > self.watch_rank(clause_lits[second]) {
                second = i;
            }
        }
        clause_lits.swap(1, second);

        let clause_id = self.clauses.add_original(clause_lits.iter().copied());
        if let Some(id) = original_lrat_id {
            self.lrat_set_clause_id(clause_id, id);
        }

        // Track clause for incremental solving
        if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
            current_level_clauses.push(clause_id);
        }

        let lit0 = clause_lits[0];
        let lit1 = clause_lits[1];

        self.watches
            .add(lit0.negate(), Watcher::new(clause_id, lit1));
        self.watches
            .add(lit1.negate(), Watcher::new(clause_id, lit0));

        // `pre_check_effective_unit` already determined -- against the exact
        // pre-watch-selection trail state, before anything here could shift
        // it -- whether this clause needs its sole undefined literal forced,
        // and confirmed every false literal is a permanent level-0 fact when
        // it did. Apply that decision now that `clause_id` exists.
        if let PreAttachOutcome::ForceUnitAtLevelZero(forced) = outcome {
            self.trail.assign_propagation_at(forced, clause_id, 0);
        }

        true
    }

    /// Rank a literal for two-watched-literal selection; a higher rank is a
    /// better watch. A true literal is best (the clause is satisfied through it),
    /// then an unassigned literal, and finally a false literal — and among false
    /// literals the one assigned at the highest decision level (falsified latest)
    /// is preferred. Watching the two highest-ranked literals mirrors MiniSat's
    /// attachClause invariant so a watch always fires when a watched literal is
    /// (re)falsified.
    ///
    /// `pub(super)`: also used by `solver/learn.rs`'s vivification and
    /// in-place clause-strengthening rewatch logic, which need the same
    /// ranking when picking new watches for a shortened clause.
    pub(super) fn watch_rank(&self, l: Lit) -> (u8, u32) {
        let v = self.trail.lit_value(l);
        if v.is_true() {
            (2, u32::MAX)
        } else if v.is_false() {
            (0, self.trail.level(l.var()))
        } else {
            (1, u32::MAX)
        }
    }

    /// Add a clause from DIMACS literals
    pub fn add_clause_dimacs(&mut self, lits: &[i32]) -> bool {
        self.add_clause(lits.iter().map(|&l| Lit::from_dimacs(l)))
    }
}
