//! Clause learning, LBD computation, database reduction, inprocessing, and vivification

use super::*;
use smallvec::SmallVec;

impl Solver {
    /// Install the consequence of a freshly learned clause on the trail.
    ///
    /// This is the single place where a learned clause's asserting literal gets
    /// assigned, and it is where chronological backtracking's central invariant
    /// lives: **a propagated literal's decision level is the maximum level over
    /// the other literals of its reason clause, not the level the search happens
    /// to sit at.**
    ///
    /// Without chronological backtracking those two coincide, because a backjump
    /// always lands exactly on the assertion level. With it the solver
    /// deliberately stops above the assertion level and keeps the intervening
    /// decisions, so recording the current level instead would claim the literal
    /// depends on decisions it does not depend on. That is not merely imprecise:
    /// a **unit** learned clause is a consequence of the formula alone and must
    /// be pinned at level 0, and recording it at the post-backtrack level both
    /// loses it on the next rollback and plants a second reason-less literal
    /// inside a decision level, which breaks the termination invariant of the
    /// 1-UIP walk in [`Solver::analyze`] and lets it emit clauses that are
    /// stronger than what resolution derives — a direct route to a false `unsat`.
    ///
    /// The clause is asserted only when it really is unit under the
    /// post-backtrack assignment. A degenerate analysis result (two literals at
    /// the top level, which the theory-propagation path can still produce) is
    /// therefore added to the database but not propagated, rather than
    /// overwriting a live trail entry.
    ///
    /// Reference: Z3's `sat_solver.cpp` (`assign_core` / `propagate_clause`
    /// compute the same `assign_level`).
    pub(super) fn assert_learned_clause(&mut self, lits: &[Lit], clause_id: ClauseId) {
        let Some(&asserting) = lits.first() else {
            return;
        };

        if self.trail.is_assigned(asserting.var()) {
            // Already satisfied is fine — nothing to install. Already *falsified*
            // is not: every caller backtracks to a level strictly below the
            // asserting literal's own before getting here (`analyze` and
            // `analyze_theory_conflict` both assert that invariant, and
            // `analyze_theory_asserting_lemma` picks an unassigned literal for
            // index 0), so the literal must be free. If it ever were false the
            // silent return would drop a refutation on the floor: a unit lemma is
            // stored without watches, and a longer clause gets both of its watches
            // pinned on literals that are already false, whose watch events have
            // been and gone. Either way nothing re-examines the clause and the
            // search can go on to report `Sat` over a trail that falsifies it.
            debug_assert!(
                !self.trail.lit_value(asserting).is_false(),
                "learned clause {lits:?} is already falsified at its asserting literal \
                 (level {}, search at level {}); returning silently would drop the refutation",
                self.trail.level(asserting.var()),
                self.trail.decision_level()
            );
            return;
        }

        if lits.len() == 1 {
            self.trail.assign_unit_fact(asserting);
            return;
        }

        let mut level = 0;
        for &lit in &lits[1..] {
            if !self.trail.lit_value(lit).is_false() {
                // Not actually unit — do not fabricate a propagation.
                return;
            }
            level = level.max(self.trail.level(lit.var()));
        }

        self.trail
            .assign_propagation_at(asserting, clause_id, level);
    }

    /// Compute LBD (Literal Block Distance) of a clause
    /// LBD is the number of distinct decision levels in the clause
    pub(super) fn compute_lbd(&mut self, lits: &[Lit]) -> u32 {
        self.lbd_mark += 1;
        let mark = self.lbd_mark;

        let mut count = 0u32;
        for &lit in lits {
            let level = self.trail.level(lit.var()) as usize;
            if level < self.level_marks.len() && self.level_marks[level] != mark {
                self.level_marks[level] = mark;
                count += 1;
            }
        }

        count
    }

    /// Learn a clause and set up watches
    /// Includes on-the-fly subsumption check
    /// Tracks allocation via memory optimizer for size-class pool accounting
    pub(super) fn learn_clause(&mut self, learnt_clause: SmallVec<[Lit; 16]>) {
        // Track allocation in memory optimizer for pool accounting
        let _pool_buf = self.memory_optimizer.allocate(learnt_clause.len());

        // Record the learned clause in the DRAT proof (no-op unless enabled). It
        // is RUP-derivable from the current database by 1-UIP construction.
        self.drat_add(&learnt_clause);

        if learnt_clause.len() == 1 {
            // Store unit learned clause in database for persistence across backtracks
            let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
            self.stats.learned_clauses += 1;
            self.stats.unit_clauses += 1;
            self.learned_clause_ids.push(clause_id);

            self.assert_learned_clause(&learnt_clause, clause_id);
        } else if learnt_clause.len() == 2 {
            // Binary learned clause - add to binary implication graph
            let lbd = self.compute_lbd(&learnt_clause);
            let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
            self.stats.learned_clauses += 1;
            self.stats.binary_clauses += 1;
            self.stats.total_lbd += lbd as u64;

            if let Some(clause) = self.clauses.get_mut(clause_id) {
                clause.lbd = lbd;
                clause.assign_tier_from_lbd();
            }
            self.debug_check_learned_clause_lbd(clause_id);

            self.learned_clause_ids.push(clause_id);

            let lit0 = learnt_clause[0];
            let lit1 = learnt_clause[1];

            // Add to binary graph
            self.binary_graph.add(lit0.negate(), lit1, clause_id);
            self.binary_graph.add(lit1.negate(), lit0, clause_id);

            self.watches
                .add(lit0.negate(), Watcher::new(clause_id, lit1));
            self.watches
                .add(lit1.negate(), Watcher::new(clause_id, lit0));

            self.assert_learned_clause(&learnt_clause, clause_id);
        } else {
            let lbd = self.compute_lbd(&learnt_clause);
            self.stats.total_lbd += lbd as u64;
            let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
            self.stats.learned_clauses += 1;

            if let Some(clause) = self.clauses.get_mut(clause_id) {
                clause.lbd = lbd;
                clause.assign_tier_from_lbd();
            }
            self.debug_check_learned_clause_lbd(clause_id);

            self.learned_clause_ids.push(clause_id);

            let lit0 = learnt_clause[0];
            let lit1 = learnt_clause[1];
            self.watches
                .add(lit0.negate(), Watcher::new(clause_id, lit1));
            self.watches
                .add(lit1.negate(), Watcher::new(clause_id, lit0));

            self.assert_learned_clause(&learnt_clause, clause_id);

            // On-the-fly subsumption: check if this new clause subsumes existing clauses
            if learnt_clause.len() <= 5 && lbd <= 3 {
                self.check_subsumption(clause_id);
            }
        }
    }

    /// Check if the given clause subsumes any existing clauses
    /// A clause C subsumes C' if all literals of C are in C'
    pub(super) fn check_subsumption(&mut self, new_clause_id: ClauseId) {
        let new_clause = match self.clauses.get(new_clause_id) {
            Some(c) => c.lits.clone(),
            None => return,
        };

        if new_clause.len() > 10 {
            return; // Don't check subsumption for large clauses (too expensive)
        }

        // Check against learned clauses only
        let mut to_remove = Vec::new();
        for &cid in &self.learned_clause_ids {
            if cid == new_clause_id {
                continue;
            }

            if let Some(clause) = self.clauses.get(cid) {
                if clause.deleted || clause.lits.len() < new_clause.len() {
                    continue;
                }

                // Check if new_clause subsumes clause
                if new_clause.iter().all(|&lit| clause.lits.contains(&lit)) {
                    to_remove.push(cid);
                }
            }
        }

        // Remove subsumed clauses
        for cid in to_remove {
            // Purge binary-graph edges and record the DRAT deletion before the
            // clause's literals become inaccessible.
            self.purge_binary_edges(cid);
            self.drat_delete(cid);
            self.lrat_delete(cid);
            self.clauses.remove(cid);
            self.stats.deleted_clauses += 1;
        }
    }

    /// Add a theory reason clause
    /// The clause is: reason_lits[0] OR reason_lits[1] OR ... OR propagated_lit
    pub(super) fn add_theory_reason_clause(
        &mut self,
        reason_lits: &[Lit],
        propagated_lit: Lit,
    ) -> ClauseId {
        let mut clause_lits: SmallVec<[Lit; 8]> = SmallVec::new();
        clause_lits.push(propagated_lit);
        for &lit in reason_lits {
            clause_lits.push(lit.negate());
        }

        // Pick the second watch the way `Solver::add_clause` does, instead of
        // blindly taking `clause_lits[1]`.
        //
        // `propagated_lit` stays at index 0: the caller assigns it immediately,
        // so it is the only literal here that is not already false and is by
        // construction the best watch. Every other literal is the negation of a
        // theory reason literal and is therefore false *right now* — and, being
        // an existing assignment, already behind the propagation head. Watching
        // whichever of them happens to come first can leave the clause watched on
        // a literal falsified long ago, whose watch event has already been and
        // gone; that watch can never fire again, so the clause can be silently
        // falsified and a later `solve()` reports `Sat` on a model violating it.
        //
        // `watch_rank` instead picks the literal that is *last* to become false
        // (a true literal, else an unassigned one, else the false literal at the
        // highest decision level), which is MiniSat's `attachClause` invariant:
        // the clause can then only become fully false through a watched literal,
        // and that always re-examines it. This mirrors the identical fix already
        // documented in `add_clause`.
        if clause_lits.len() >= 2 {
            let mut second = 1;
            for i in 2..clause_lits.len() {
                if self.watch_rank(clause_lits[i]) > self.watch_rank(clause_lits[second]) {
                    second = i;
                }
            }
            clause_lits.swap(1, second);
        }

        let clause_id = self.clauses.add_learned(clause_lits.iter().copied());

        // Set up watches
        if clause_lits.len() >= 2 {
            let lit0 = clause_lits[0];
            let lit1 = clause_lits[1];
            self.watches
                .add(lit0.negate(), Watcher::new(clause_id, lit1));
            self.watches
                .add(lit1.negate(), Watcher::new(clause_id, lit0));

            // Every other `add_learned` call site computes and stores an LBD for
            // its clause (see `Solver::compute_lbd`'s call sites in `solve` and
            // `learn_clause`); this one used to be the odd one out, leaving
            // `clause.lbd` at `Clause::learned`'s default of 0 forever. That is
            // not just a missing statistic: `Clause::record_usage` promotes a
            // clause straight to the rarely-deleted `Core` tier once
            // `lbd <= 2`, so a stuck LBD of 0 gave every theory reason clause an
            // artificially easy path into permanent retention regardless of its
            // actual quality. Computed here, before `propagated_lit` is assigned
            // by the caller, matching the timing every other call site uses
            // (LBD is computed while the asserting/propagated literal is still
            // unassigned).
            let lbd = self.compute_lbd(&clause_lits);
            if let Some(clause) = self.clauses.get_mut(clause_id) {
                clause.lbd = lbd;
            }
            self.debug_check_learned_clause_lbd(clause_id);
        }

        clause_id
    }

    /// Reduce the learned clause database using tier-based deletion strategy
    /// - Core tier (Tier 1): Rarely deleted, only if very inactive
    /// - Mid tier (Tier 2): Delete ~30% based on activity
    /// - Local tier (Tier 3): Delete ~75% based on activity
    pub(super) fn reduce_clause_database(&mut self) {
        use crate::clause::ClauseTier;

        let mut core_candidates: Vec<(ClauseId, f64)> = Vec::new();
        let mut mid_candidates: Vec<(ClauseId, f64)> = Vec::new();
        let mut local_candidates: Vec<(ClauseId, f64)> = Vec::new();

        for &cid in &self.learned_clause_ids {
            if let Some(clause) = self.clauses.get(cid) {
                if clause.deleted {
                    continue;
                }

                // Don't delete binary clauses (very useful)
                if clause.lits.len() <= 2 {
                    continue;
                }

                // Check if clause is currently a reason for any assignment
                // (we can't delete reason clauses).
                //
                // Only the asserting literal (`lits[0]`) can ever be the
                // literal a clause propagated: `propagate` always assigns
                // whichever watched literal remains after the other one is
                // (re)falsified, and that literal is read as `clause.lits[0]`
                // at the moment of assignment (see `Solver::propagate`'s
                // "make sure the false literal is at position 1" swap, which
                // moves the *falsified* literal to index 1, never the true
                // one out of index 0). Once `lits[0]` is the reason, its
                // watcher's cached `blocker` is that same true literal, so
                // every later scan of the *other* watch short-circuits on the
                // blocker check before it could reach — let alone swap away
                // from — index 0. So checking only `lits[0]`'s reason is
                // exactly as precise as scanning every literal, but O(1)
                // instead of O(clause length) — this runs once per learned
                // clause on every database reduction pass.
                let asserting_var = clause.lits[0].var();
                let is_reason = self.trail.is_assigned(asserting_var)
                    && matches!(
                        self.trail.reason(asserting_var),
                        Reason::Propagation(r) if r == cid
                    );

                if !is_reason {
                    match clause.tier {
                        ClauseTier::Core => core_candidates.push((cid, clause.activity)),
                        ClauseTier::Mid => mid_candidates.push((cid, clause.activity)),
                        ClauseTier::Local => local_candidates.push((cid, clause.activity)),
                    }
                }
            }
        }

        // Sort by activity (ascending) - delete low-activity clauses first
        core_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        mid_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        local_candidates
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        // Delete different percentages from each tier
        // Core: Delete bottom 10% (very conservative)
        let num_core_delete = core_candidates.len() / 10;
        // Mid: Delete bottom 30%
        let num_mid_delete = (mid_candidates.len() * 3) / 10;
        // Local: Delete bottom 75% (very aggressive)
        let num_local_delete = (local_candidates.len() * 3) / 4;

        for (cid, _) in core_candidates.iter().take(num_core_delete) {
            // Track clause size for memory pool accounting before removal
            if let Some(clause) = self.clauses.get(*cid) {
                let num_lits = clause.lits.len();
                let buf = self.memory_optimizer.allocate(num_lits);
                self.memory_optimizer.free(buf, num_lits);
            }
            self.drat_delete(*cid);
            self.lrat_delete(*cid);
            self.clauses.remove(*cid);
            self.stats.deleted_clauses += 1;
        }

        for (cid, _) in mid_candidates.iter().take(num_mid_delete) {
            if let Some(clause) = self.clauses.get(*cid) {
                let num_lits = clause.lits.len();
                let buf = self.memory_optimizer.allocate(num_lits);
                self.memory_optimizer.free(buf, num_lits);
            }
            self.drat_delete(*cid);
            self.lrat_delete(*cid);
            self.clauses.remove(*cid);
            self.stats.deleted_clauses += 1;
        }

        for (cid, _) in local_candidates.iter().take(num_local_delete) {
            if let Some(clause) = self.clauses.get(*cid) {
                let num_lits = clause.lits.len();
                let buf = self.memory_optimizer.allocate(num_lits);
                self.memory_optimizer.free(buf, num_lits);
            }
            self.drat_delete(*cid);
            self.lrat_delete(*cid);
            self.clauses.remove(*cid);
            self.stats.deleted_clauses += 1;
        }

        // Clean up learned_clause_ids (remove deleted clauses)
        self.learned_clause_ids
            .retain(|&cid| self.clauses.get(cid).is_some_and(|c| !c.deleted));

        // Apply memory optimizer recommendations after deletion
        match self.memory_optimizer.recommend_action() {
            MemoryAction::Compact => {
                self.memory_optimizer.compact();
                self.clauses.compact();
            }
            MemoryAction::ReduceClauseDatabase => {
                // Already reduced; just compact the pool
                self.memory_optimizer.compact();
            }
            MemoryAction::ExpandPools | MemoryAction::None => {
                // No action needed
            }
        }
    }

    /// Handle clause deletion check and restart check
    pub(super) fn handle_clause_deletion_and_restart(&mut self) {
        self.conflicts_since_deletion += 1;

        if self.conflicts_since_deletion >= self.config.clause_deletion_threshold as u64 {
            self.reduce_clause_database();
            self.debug_check_invariants("after clause database reduction");
            self.conflicts_since_deletion = 0;
        }

        if self.stats.conflicts >= self.restart_threshold {
            self.restart();
            // A reuse-trail restart (the default) deliberately stops above
            // level 0, so the level-0 invariant `debug_check_restart_consistency`
            // checks does not apply to it.
            if !self.config.reuse_trail {
                self.debug_check_restart_consistency();
            }
        }
    }

    /// Handle clause deletion and restart, but don't backtrack past assumptions
    pub(super) fn handle_clause_deletion_and_restart_limited(&mut self, min_level: u32) {
        self.conflicts_since_deletion += 1;

        if self.conflicts_since_deletion >= self.config.clause_deletion_threshold as u64 {
            self.reduce_clause_database();
            self.debug_check_invariants("after clause database reduction (assumptions)");
            self.conflicts_since_deletion = 0;
        }

        if self.stats.conflicts >= self.restart_threshold {
            // Limited restart - don't backtrack past assumptions, so unlike
            // `Solver::restart` this does NOT land at decision level 0;
            // `debug_check_restart_consistency` (which asserts exactly that)
            // does not apply here.
            self.backtrack(min_level);
            self.stats.restarts += 1;
            self.luby_index += 1;
            self.restart_threshold =
                self.stats.conflicts + self.config.restart_interval * Self::luby(self.luby_index);
            self.debug_check_invariants("after limited restart (assumptions)");
        }
    }

    /// Save the model
    pub(super) fn save_model(&mut self) {
        self.model.resize(self.num_vars, LBool::Undef);
        for i in 0..self.num_vars {
            self.model[i] = self.trail.value(Var::new(i as u32));
        }
        self.apply_model_reconstruction();
    }

    /// Fix up `self.model` for every variable the preprocessing/inprocessing
    /// toolkit removed from the live formula.
    ///
    /// Split out of [`Solver::save_model`] so a caller that produces a full
    /// assignment by some route *other* than the trail — the pre-search lucky
    /// phase (see `solver/lucky.rs`) builds one in a scratch buffer — can
    /// install it and still go through the exact same reconstruction the
    /// ordinary Sat exit does. Every step below overwrites unconditionally
    /// and depends only on `self.model`, never on the trail, so the two
    /// callers are interchangeable.
    pub(super) fn apply_model_reconstruction(&mut self) {
        // Reconstruct pure literals eliminated during inprocessing. Their clauses
        // were deleted on the promise that the literal is fixed to its polarity;
        // the search may have assigned the variable the opposite phase, so force
        // it here. This can only satisfy additional clauses: no remaining clause
        // contains the opposite polarity (that is exactly what "pure" means).
        for &lit in &self.pure_literal_reconstruction {
            let idx = lit.var().index();
            if idx < self.model.len() {
                self.model[idx] = if lit.is_pos() {
                    LBool::True
                } else {
                    LBool::False
                };
            }
        }

        // Reconstruct variables removed by the inprocessing toolkit's
        // variable-elimination mechanisms. Both overwrite unconditionally
        // (not only when the caller's assignment left `Undef`): a variable the
        // toolkit eliminated no longer appears in any live clause, so
        // `pick_branch_var` should never hand it out as a decision (see the
        // `var_eliminated` guards in `solver/decide.rs`), but nothing stops
        // it from still carrying a *stale* value from before elimination (or
        // an incidental one from theory/assumption bookkeeping) — the
        // reconstruction rule, not whatever happens to already be sitting in
        // `self.model[idx]`, is the only thing that can be trusted here.
        self.reconstruct_equiv_eliminated_variables();
        self.reconstruct_bve_eliminated_variables();
    }

    /// Safety net for the purely Boolean entry points: check that the model just
    /// saved actually satisfies the clause database.
    ///
    /// A false `Unsat` merely fails to solve; a false `Sat` hands every
    /// downstream consumer an assignment they will trust. Verifying the finished
    /// model is one linear pass, so in debug builds — which covers every test and
    /// CI run — this converts such corruption into a loud, precisely localised
    /// failure instead of a silently wrong answer. It compiles to nothing in
    /// release, so the shipped hot path is unaffected.
    ///
    /// Deliberately **not** called from `solve_with_theory`; that path has its own
    /// narrower guard, [`Solver::debug_verify_model_input`]. There the database
    /// also holds lemmas injected through `TheoryCallback` (see
    /// [`Solver::add_theory_reason_clause`]) whose validity and lifetime this
    /// crate does not control: the theory retracts its context through
    /// `on_backtrack` without the Boolean core retracting the corresponding
    /// lemma, so a final model may legitimately falsify one. Asserting on those
    /// would make `oxiz-sat` fail on behalf of a component it cannot police.
    pub(super) fn debug_verify_model(&self) {
        #[cfg(debug_assertions)]
        if let Some(id) = self.find_model_violation(true) {
            let lits = self.clauses.get(id).map(|c| c.lits.clone());
            panic!(
                "solve() reported Sat with a model that violates clause {id:?} ({lits:?}); \
                 the search accepted an assignment that does not satisfy the database"
            );
        }
    }

    /// Safety net for the CDCL(T) entry point [`Solver::solve_with_theory`].
    ///
    /// The full check above cannot be used there, but the reason it cannot —
    /// `TheoryCallback`-injected lemmas whose validity `oxiz-sat` does not own —
    /// applies only to *learned* clauses: theory reason clauses and theory lemmas
    /// all enter through `ClauseDb::add_learned`, as do the resolvents that 1-UIP
    /// analysis derives over them. **Original** clauses are a different matter
    /// entirely. They arrived through `add_clause` from the caller, they are the
    /// Boolean abstraction the caller asked to be satisfied, and nothing but
    /// `pop` retracts them. A `Sat` answer that falsifies one is a bug in this
    /// crate no matter what the theory did, so restricting the scan to them gives
    /// a guard that is both meaningful and impossible to trip on the theory's
    /// behalf.
    ///
    /// That is precisely the class the propagation-fixpoint bug produced: an
    /// original ternary clause with all three literals falsified by level-0 facts
    /// while `final_check` answered `Sat`.
    pub(super) fn debug_verify_model_input(&self) {
        #[cfg(debug_assertions)]
        if let Some(id) = self.find_model_violation(false) {
            let lits = self.clauses.get(id).map(|c| c.lits.clone());
            panic!(
                "solve_with_theory() reported Sat with a model that violates ORIGINAL clause \
                 {id:?} ({lits:?}); the CDCL(T) search accepted an assignment that does not \
                 satisfy the input formula's Boolean abstraction"
            );
        }
    }

    /// Find a live, *enforced* clause that the saved model does not satisfy.
    ///
    /// The scope is deliberately "everything the two-watched-literal scheme is
    /// responsible for": non-deleted clauses of at least two literals. A model
    /// falsifying one of those means propagation failed to fire on a watch, which
    /// is a soundness bug whether the clause is original or learned. Callers that
    /// cannot vouch for learned clauses pass `include_learned == false`.
    ///
    /// Unit clauses are excluded because the database is not what enforces them.
    /// `add_clause` never stores a unit at all — it assigns the literal at level
    /// 0 — and the copies that learned units leave behind carry no watches; their
    /// force comes solely from that level-0 trail assignment. An incremental
    /// caller that retracts the assignment (`pop`, `restore_to_trail_size`)
    /// without also dropping the record leaves a lemma the formula no longer
    /// entails, which a later model may legitimately falsify.
    ///
    /// Clauses retired by inprocessing are skipped for a similar reason: they are
    /// re-satisfied by model reconstruction rather than by the assignment, and
    /// their literals need not even be in the model's variable range.
    #[cfg(debug_assertions)]
    fn find_model_violation(&self, include_learned: bool) -> Option<ClauseId> {
        self.first_violated_clause(&self.model, include_learned, 2)
    }

    /// Find a live clause of at least `min_len` literals that `assignment`
    /// does not satisfy, restricted to original clauses unless
    /// `include_learned` is set.
    ///
    /// The scan body [`Solver::find_model_violation`] used to own, lifted out
    /// so it can run against a *candidate* assignment held in a scratch
    /// buffer rather than only against `self.model`, and — unlike that
    /// debug-only wrapper — compiled in every build. The pre-search lucky
    /// phase (see `solver/lucky.rs`) is the release-build caller: for it the
    /// scan is not an assertion but the actual acceptance test that decides
    /// whether a guessed assignment may be reported as a model, so it must
    /// exist when `debug_assertions` is off.
    ///
    /// `min_len` exists because the two callers disagree about unit clauses
    /// on purpose. `find_model_violation` passes `2`, excluding them for the
    /// reasons its own doc comment gives (the database is not what enforces a
    /// unit — a level-0 trail assignment is — so a retracted one may be
    /// legitimately falsified by a later model). The lucky phase passes `1`:
    /// it is deciding a verdict rather than checking an invariant, so it
    /// takes the strictly more conservative reading and refuses a candidate
    /// falsifying *any* live original clause, including a length-1 one left
    /// behind by in-place strengthening (`Solver::self_subsuming_resolution`).
    ///
    /// A variable outside `assignment`'s range, or one left `Undef`, counts
    /// as satisfying nothing — so an incomplete assignment can only ever be
    /// *rejected* by this scan, never wrongly accepted.
    pub(super) fn first_violated_clause(
        &self,
        assignment: &[LBool],
        include_learned: bool,
        min_len: usize,
    ) -> Option<ClauseId> {
        self.clauses.iter_ids().find(|&id| {
            self.clauses.get(id).is_some_and(|clause| {
                (include_learned || !clause.learned)
                    && clause.lits.len() >= min_len
                    && !clause
                        .lits
                        .iter()
                        .any(|lit| match assignment.get(lit.var().index()) {
                            Some(LBool::True) => lit.is_pos(),
                            Some(LBool::False) => !lit.is_pos(),
                            _ => false,
                        })
            })
        })
    }

    /// Release build: compiles away entirely.
    #[cfg(not(debug_assertions))]
    #[inline]
    pub(super) fn debug_check_invariants(&self, _context: &str) {}

    /// Debug-only structural/soundness net over the whole CDCL data model:
    /// clause well-formedness, trail/assignment consistency, decision-level
    /// bookkeeping, static learned-clause LBD bounds, live reason clauses, and
    /// implication-graph acyclicity. See `crate::invariants` for exactly what
    /// each check covers and why the checks this sweep deliberately does
    /// *not* run are situational instead (only meaningful right after a
    /// specific event, such as a conflict or a restart). Compiles to nothing
    /// in release builds.
    #[cfg(debug_assertions)]
    pub(super) fn debug_check_invariants(&self, context: &str) {
        if let Err(msg) = crate::invariants::check_all_sat_invariants(self) {
            panic!("SAT solver invariant violated ({context}): {msg}");
        }
    }

    /// Release build: compiles away entirely.
    #[cfg(not(debug_assertions))]
    #[inline]
    pub(super) fn debug_check_fixpoint_invariants(&self, _context: &str) {}

    /// Debug-only net for the two invariants that only hold once
    /// `propagate()` has reached a fixpoint (returned `None`): no live clause
    /// has both watched literals false while unsatisfied
    /// (`crate::invariants::check_watched_literals`), and no live clause is a
    /// hanging unit (`crate::invariants::check_unit_propagation_complete`).
    /// Call this only at a fixpoint — mid-scan both are routinely and
    /// harmlessly violated. Compiles to nothing in release builds.
    #[cfg(debug_assertions)]
    pub(super) fn debug_check_fixpoint_invariants(&self, context: &str) {
        if let Err(msg) = crate::invariants::check_watched_literals(self) {
            panic!("SAT solver invariant violated at a propagation fixpoint ({context}): {msg}");
        }
        if let Err(msg) = crate::invariants::check_unit_propagation_complete(self) {
            panic!("SAT solver invariant violated at a propagation fixpoint ({context}): {msg}");
        }
    }

    /// Release build: compiles away entirely.
    #[cfg(not(debug_assertions))]
    #[inline]
    pub(super) fn debug_check_conflict_clause(&self, _conflict: ClauseId) {}

    /// Debug-only net: the clause `propagate()` just reported as a conflict
    /// is fully assigned and fully falsified. Call this right where
    /// `propagate()` returns `Some(conflict)`, before any backtrack changes
    /// the trail. Compiles to nothing in release builds.
    #[cfg(debug_assertions)]
    pub(super) fn debug_check_conflict_clause(&self, conflict: ClauseId) {
        if let Err(msg) = crate::invariants::check_conflict_clause(self, conflict) {
            panic!("SAT solver invariant violated at conflict detection: {msg}");
        }
    }

    /// Release build: compiles away entirely.
    #[cfg(not(debug_assertions))]
    #[inline]
    pub(super) fn debug_check_restart_consistency(&self) {}

    /// Debug-only net: right after a restart, the decision level is 0 and
    /// every trail entry is a level-0 fact. Compiles to nothing in release
    /// builds.
    #[cfg(debug_assertions)]
    pub(super) fn debug_check_restart_consistency(&self) {
        if let Err(msg) = crate::invariants::check_restart_consistency(self) {
            panic!("SAT solver invariant violated after restart: {msg}");
        }
    }

    /// Release build: compiles away entirely.
    #[cfg(not(debug_assertions))]
    #[inline]
    pub(super) fn debug_check_learned_clause_lbd(&self, _clause_id: ClauseId) {}

    /// Debug-only net: a freshly learned clause's stored LBD matches
    /// recomputing it right now. Only sound to call in the instant right
    /// after `clause_id` was learned and its `lbd` field set — see
    /// `crate::invariants::check_learned_clause_lbd`'s doc comment for why
    /// this cannot be a standing, whole-database invariant. Compiles to
    /// nothing in release builds.
    #[cfg(debug_assertions)]
    pub(super) fn debug_check_learned_clause_lbd(&self, clause_id: ClauseId) {
        if let Err(msg) = crate::invariants::check_learned_clause_lbd(self, clause_id) {
            panic!("SAT solver invariant violated for freshly learned clause: {msg}");
        }
    }

    /// Remove the literal at `idx` from a learned clause and rebuild its two
    /// watches so the two-watched-literal invariant is preserved.
    ///
    /// Vivification and inprocessing strengthening shrink a clause in place. If
    /// the removed literal sat at a watched position (index 0 or 1), the stale
    /// watcher would keep pointing at a literal no longer in the clause, breaking
    /// watch firing — a watched literal becoming false would no longer re-examine
    /// the clause, causing missed unit propagations and, if index 0 were removed
    /// repeatedly, a clause left effectively unwatched (a missed conflict). This
    /// detaches the old watches (always keyed on the pre-removal literals at
    /// positions 0 and 1), removes the literal, re-selects the two best watch
    /// literals (mirroring [`Solver::add_clause`]), and re-attaches them.
    pub(super) fn remove_literal_and_rewatch(&mut self, clause_id: ClauseId, idx: usize) {
        let mut lits: Vec<Lit> = match self.clauses.get(clause_id) {
            Some(c) if !c.deleted && c.lits.len() > 2 && idx < c.lits.len() => {
                c.lits.iter().copied().collect()
            }
            _ => return,
        };

        // Snapshot the pre-strengthening literals so the DRAT proof can retire the
        // original clause once the shorter (still F-entailed) form is recorded.
        let original_lits = lits.clone();

        // Detach the existing watches (keyed on the current positions 0 and 1).
        let old_w0 = lits[0];
        let old_w1 = lits[1];
        self.watches.remove_clause(old_w0.negate(), clause_id);
        self.watches.remove_clause(old_w1.negate(), clause_id);

        // Remove the redundant literal.
        lits.remove(idx);

        // Re-select the two best watch literals: prefer a satisfied literal, then
        // an unassigned one, and finally the latest-falsified. Mirrors
        // `Solver::add_clause` so a watched literal always fires when re-falsified.
        let n = lits.len();
        let mut best = 0;
        for i in 1..n {
            if self.watch_rank(lits[i]) > self.watch_rank(lits[best]) {
                best = i;
            }
        }
        lits.swap(0, best);
        let mut second = 1;
        for i in 2..n {
            if self.watch_rank(lits[i]) > self.watch_rank(lits[second]) {
                second = i;
            }
        }
        lits.swap(1, second);

        let w0 = lits[0];
        let w1 = lits[1];

        // Write the reordered literals back into the clause, and bring its LBD
        // back inside the range the shortened clause can actually justify.
        //
        // LBD counts *distinct decision levels* among a clause's literals, so
        // it can never exceed the clause's length — an invariant
        // `crate::invariants` checks in debug builds, and one that clause-tier
        // promotion (`Clause::assign_tier_from_lbd`) and database reduction
        // both read. Dropping a literal can drop the last representative of a
        // decision level, so a stale stored LBD may now exceed the new length
        // (a ternary strengthened to a binary while its recorded LBD was 3
        // trips the invariant outright).
        //
        // Clamped rather than recomputed, deliberately: `compute_lbd` reads
        // the *current* trail, and every caller of this runs at decision level
        // 0 where each variable is either level-0 or unassigned — recomputing
        // would score every strengthened clause as LBD 1 and silently promote
        // the whole strengthened population into the never-deleted glue tier.
        // Clamping repairs the bound and leaves the recorded quality ordering
        // alone.
        let capped_lbd = lits.len() as u32;
        if let Some(clause) = self.clauses.get_mut(clause_id) {
            clause.lits.clear();
            clause.lits.extend(lits.iter().copied());
            clause.lbd = clause.lbd.min(capped_lbd);
        }

        // Re-attach watches on the new positions 0 and 1.
        self.watches.add(w0.negate(), Watcher::new(clause_id, w1));
        self.watches.add(w1.negate(), Watcher::new(clause_id, w0));

        // Rewind the propagation head so the whole surviving trail is scanned
        // again on the next `propagate()`.
        //
        // Re-attaching watches is not enough on its own: a watch only ever
        // fires when its literal is *newly* falsified, and the literals that
        // falsify a just-shortened clause are typically already on the trail,
        // propagated long ago. A clause that had two unassigned literals and
        // loses one becomes unit under the *current* assignment with no future
        // event left to notice it — a "hanging unit" that unit propagation
        // never fires on, which the debug fixpoint invariants catch and which
        // in a release build simply loses an implication (and, if that
        // implication was the one that closed a branch, can change the search
        // enough to matter). Rewinding is always safe: re-propagating a
        // literal that is still assigned is a no-op, so this costs one extra
        // pass over the watch lists and nothing else.
        self.trail.reset_propagation_head();

        // Record the in-place strengthening in the DRAT proof: add the shorter
        // clause (RUP-derivable — vivification proved it entailed) then delete the
        // original, keeping the proof's clause set consistent with the database.
        if self.drat.is_some() {
            let new_lits: SmallVec<[Lit; 8]> = lits.iter().copied().collect();
            self.drat_add(&new_lits);
            self.drat_delete_lits(&original_lits);
        }
    }

    /// Vivification: try to strengthen clauses by checking if some literals are redundant
    /// This is an inprocessing technique that should be called periodically
    ///
    /// Skipped outright while LRAT tracing is active — see
    /// [`Solver::inprocess`]'s doc comment; `remove_literal_and_rewatch`'s
    /// DRAT-only add+delete pair applies to this entry point too, since it
    /// is called both from here and from `inprocess`.
    pub(super) fn vivify_clauses(&mut self) {
        if self.trail.decision_level() != 0 || self.lrat.is_some() {
            return; // Only vivify at decision level 0
        }

        let mut vivified_count = 0;
        let max_vivifications = 100; // Limit to avoid too much overhead

        // Try to vivify some learned clauses
        let clause_ids: Vec<ClauseId> = self
            .learned_clause_ids
            .iter()
            .copied()
            .take(max_vivifications)
            .collect();

        for clause_id in clause_ids {
            if vivified_count >= max_vivifications {
                break;
            }

            let clause_lits = match self.clauses.get(clause_id) {
                Some(c) if !c.deleted && c.lits.len() > 2 => c.lits.clone(),
                _ => continue,
            };

            // Try to find redundant literals in the clause
            // Assign all literals except one to false and see if we can derive the last one
            for skip_idx in 0..clause_lits.len() {
                // Save current state
                let saved_level = self.trail.decision_level();

                // Assign all literals except skip_idx to false
                self.trail.new_decision_level();
                let mut conflict = false;

                for (i, &lit) in clause_lits.iter().enumerate() {
                    if i == skip_idx {
                        continue;
                    }

                    let value = self.trail.lit_value(lit);
                    if value.is_true() {
                        // Clause is already satisfied
                        conflict = false;
                        break;
                    } else if value.is_false() {
                        // Already false
                        continue;
                    } else {
                        // Assign to false
                        self.trail.assign_decision(lit.negate());

                        // Propagate
                        if self.propagate().is_some() {
                            conflict = true;
                            break;
                        }
                    }
                }

                // Backtrack
                self.backtrack(saved_level);

                if conflict {
                    // The literal at skip_idx is implied by the rest, so it can
                    // be dropped (vivification succeeded). Remove it *and* rebuild
                    // the clause's watches so the two-watched invariant survives.
                    let removable = self
                        .clauses
                        .get(clause_id)
                        .is_some_and(|c| !c.deleted && c.lits.len() > 2 && skip_idx < c.lits.len());
                    if removable {
                        self.remove_literal_and_rewatch(clause_id, skip_idx);
                        vivified_count += 1;
                        break; // Done with this clause
                    }
                }
            }
        }

        // The probe loop above drains any still-pending level-0 propagation
        // queue under a probe decision level and then throws the resulting
        // assignments away on backtrack. Rewind the head so the next
        // `propagate()` re-derives the level-0 consequences that were lost —
        // see the identical note at the end of `Solver::inprocess` for the
        // full reasoning.
        self.trail.reset_propagation_head();
    }

    /// Perform inprocessing (apply preprocessing during search)
    ///
    /// Skipped outright while LRAT tracing is active. DRAT stays fully
    /// supported here (see the pre-pass/post-pass literal snapshot below,
    /// which already emits a correct deletion line for every clause this
    /// retires, and `remove_literal_and_rewatch`'s own DRAT add+delete
    /// pair for strengthening — both self-justifying, no hints needed).
    /// LRAT is different: `strengthen_clauses_inprocessing`'s redundant-
    /// literal check derives its shorter clause via a hypothetical
    /// assign-and-propagate probe this module does not thread a hint chain
    /// through, so rather than emit an LRAT addition this port cannot back
    /// with a real chain, the whole pass — pure-literal elimination,
    /// subsumption, and strengthening alike — steps aside for LRAT.
    pub(super) fn inprocess(&mut self) {
        use crate::Preprocessor;

        // Only inprocess at decision level 0
        if self.trail.decision_level() != 0 || self.lrat.is_some() {
            return;
        }

        // Create preprocessor with current number of variables
        let mut preprocessor = Preprocessor::new(self.num_vars);

        // Snapshot every live clause's literals before the elimination passes
        // below run. `Preprocessor::pure_literal_elimination` and
        // `subsumption_elimination` retire clauses by setting `Clause::deleted`
        // directly on `self.clauses` (they don't go through
        // `ClauseDatabase::remove`) and report only a count, not which ids were
        // touched. `drat_delete(id)` can't be used afterwards either — by
        // design it refuses to read literals off a clause already marked
        // deleted, to avoid ever emitting a deletion line with garbage
        // literals. Without this snapshot the deletions below would never
        // reach the DRAT proof: the checker would still accept the proof (an
        // omitted deletion hint only makes it larger, never invalid), but the
        // proof would keep clauses the live database no longer has, which is
        // exactly the minimality gap this snapshot closes. Skipped entirely
        // when proof logging is off.
        let pre_lits: Vec<(ClauseId, SmallVec<[Lit; 8]>)> = if self.drat.is_some() {
            self.clauses
                .iter_ids()
                .filter_map(|id| {
                    self.clauses
                        .get(id)
                        .map(|c| (id, c.lits.iter().copied().collect()))
                })
                .collect()
        } else {
            Vec::new()
        };

        // Pure-literal elimination deletes original clauses; that is only
        // satisfiability-preserving if the pure literal is fixed to its polarity
        // in the reconstructed model. It is also unsound across incremental
        // scopes, where a later `add_clause` could reintroduce the opposite
        // polarity after the clauses were dropped, so it is only run at the base
        // assertion level (no active `push`).
        if self.assertion_levels.len() <= 1 {
            // Variables already fixed on the level-0 trail must be excluded
            // from the purity scan: unit facts are normally consumed straight
            // onto the trail and never stored as a clause, so
            // `pure_literal_elimination`'s occurrence scan can't see them --
            // without this exclusion it could misjudge a trail-forced
            // variable "pure" in the opposite polarity, drop clauses that are
            // only satisfied by the trail's actual value, and hand back a
            // model that violates them.
            let trail_fixed: Vec<bool> = (0..self.num_vars)
                .map(|i| self.trail.is_assigned(Var::new(i as u32)))
                .collect();
            let _pure_elim = preprocessor.pure_literal_elimination(&mut self.clauses, &trail_fixed);
            // Record each eliminated pure literal so `save_model` can fix it to
            // `true`, keeping the deleted clauses satisfied even if the search
            // later assigns the variable the opposite phase. Keep at most one
            // polarity per variable (the first recorded).
            for &lit in preprocessor.eliminated_pure_literals() {
                let already = self
                    .pure_literal_reconstruction
                    .iter()
                    .any(|existing| existing.var() == lit.var());
                if !already {
                    self.pure_literal_reconstruction.push(lit);
                }
            }
        }

        let _subsumption = preprocessor.subsumption_elimination(&mut self.clauses);

        // Emit a DRAT deletion line for every clause the two passes above
        // retired, identified by diffing against the pre-pass snapshot (any
        // previously-live clause that is now deleted).
        for (id, lits) in &pre_lits {
            if self.clauses.get(*id).is_some_and(|c| c.deleted) {
                self.drat_delete_lits(lits);
            }
        }

        // Self-subsuming resolution: strengthen clauses whose resolvent
        // against another live clause subsumes them. Deliberately sequenced
        // *after* subsumption elimination and before the entailment-probing
        // strengthening below:
        //
        // * after subsumption, because subsumption only ever deletes clauses
        //   (which shrinks the work here) whereas this pass rewrites live
        //   clause contents — running it first would leave
        //   `subsumption_elimination`'s freshly built occurrence lists
        //   describing literals that are no longer there;
        // * before `strengthen_clauses_inprocessing`, because this pass is a
        //   pure syntactic check over two clauses while that one runs an
        //   assign-and-propagate probe per literal, so the cheap pass gets
        //   first crack at the easy removals.
        //
        // Every removal it performs goes through `remove_literal_and_rewatch`,
        // which keeps watches and the DRAT proof in step; nothing extra is
        // needed from the `pre_lits` snapshot above (that snapshot detects
        // clause *deletions*, and this pass deletes none).
        self.self_subsuming_resolution();

        // On-the-fly clause strengthening
        self.strengthen_clauses_inprocessing();

        // Re-arm unit propagation over the whole surviving trail.
        //
        // This is not an optimization guard, it repairs a real completeness
        // hole. `inprocess` is called from the search loop right after a
        // conflict is handled, at which point the level-0 propagation queue is
        // routinely *non-empty* (`prop_head < trail.size()`).
        // `strengthen_clauses_inprocessing` — and `vivify_clauses`, which
        // shares the technique — then open a probe decision level, assign
        // literals speculatively and call `propagate()`. That call drains the
        // still-pending level-0 literals under the probe's assumptions:
        // `prop_head` advances past them, any consequence it derives is filed
        // at the probe level, and the subsequent `backtrack` throws those
        // consequences away. The genuine level-0 implications among them are
        // then never re-derived — leaving a live clause with one unassigned
        // literal and every other literal false that nothing will ever fire on
        // (a "hanging unit"; `crate::invariants::check_unit_propagation_complete`
        // catches exactly this in debug builds, and in a release build it
        // silently loses an implication).
        //
        // Rewinding the head makes the next `propagate()` rescan the whole
        // trail, which re-derives every lost level-0 consequence. Safe by
        // construction: re-propagating an already-assigned literal is a no-op
        // (see `Trail::reset_propagation_head`), so the only cost is one extra
        // pass over the watch lists per inprocessing stop.
        self.trail.reset_propagation_head();

        // Watch lists for clauses these passes *retired* need no repair:
        // `propagate` skips clauses marked deleted, and a strengthened clause
        // is re-watched in place by `remove_literal_and_rewatch`.
    }

    /// On-the-fly clause strengthening during inprocessing
    ///
    /// Try to remove literals from clauses by checking if they're redundant.
    /// A literal is redundant if the clause is satisfied when it's assigned to false.
    pub(super) fn strengthen_clauses_inprocessing(&mut self) {
        if self.trail.decision_level() != 0 {
            return;
        }

        let max_clauses_to_strengthen = 50; // Limit to avoid overhead
        let mut strengthened_count = 0;

        // Collect candidate clauses (learned clauses with LBD > 2)
        let mut candidates: Vec<(ClauseId, u32)> = Vec::new();

        for &clause_id in &self.learned_clause_ids {
            if let Some(clause) = self.clauses.get(clause_id)
                && !clause.deleted
                && clause.lits.len() > 3
                && clause.lbd > 2
            {
                candidates.push((clause_id, clause.lbd));
            }
        }

        // Sort by LBD (prioritize higher LBD clauses for strengthening)
        candidates.sort_by_key(|(_, lbd)| core::cmp::Reverse(*lbd));

        for (clause_id, _) in candidates.iter().take(max_clauses_to_strengthen) {
            if strengthened_count >= max_clauses_to_strengthen {
                break;
            }

            let clause_lits = match self.clauses.get(*clause_id) {
                Some(c) if !c.deleted && c.lits.len() > 3 => c.lits.clone(),
                _ => continue,
            };

            // Correct self-subsumption / vivification.
            //
            // A literal `l_k` of clause C is redundant iff the formula F already
            // entails C \ {l_k}. To prove that, assert the negation of every
            // *other* literal of C and propagate: a conflict means
            //   F ∧ ¬(C \ {l_k}) ⊨ ⊥   ⇔   F ⊨ (C \ {l_k}),
            // so `l_k` can be dropped while the clause stays F-entailed.
            //
            // The previous implementation asserted only ¬l_k and, on conflict,
            // concluded F ⊨ l_k (a backbone) — then removed l_k. That is the
            // wrong direction: F ⊨ l_k does NOT imply F ⊨ C \ {l_k}, so the
            // shrunken clause excluded legitimate models and could flip SAT to
            // UNSAT whenever inprocessing was enabled. This version negates the
            // *other* literals (matching `vivify_clauses`) and never assigns an
            // already-assigned variable.
            let mut removed_idx: Option<usize> = None;

            for skip_idx in 0..clause_lits.len() {
                let saved_level = self.trail.decision_level();
                self.trail.new_decision_level();

                let mut conflict = false;
                let mut already_sat = false;

                for (i, &lit) in clause_lits.iter().enumerate() {
                    if i == skip_idx {
                        continue;
                    }
                    let value = self.trail.lit_value(lit);
                    if value.is_true() {
                        // C \ {skip_idx} is already satisfied under F, so this
                        // probe cannot justify removing skip_idx.
                        already_sat = true;
                        break;
                    } else if value.is_false() {
                        continue;
                    } else {
                        self.trail.assign_decision(lit.negate());
                        if self.propagate().is_some() {
                            conflict = true;
                            break;
                        }
                    }
                }

                self.backtrack(saved_level);

                if conflict && !already_sat && clause_lits.len() > 2 {
                    removed_idx = Some(skip_idx);
                    break; // Only remove one literal at a time
                }
            }

            // Apply strengthening if we found a redundant literal.
            if let Some(idx) = removed_idx {
                // Remove the redundant literal and rebuild the clause's watches
                // (removing a watched literal in place would break the
                // two-watched invariant, causing missed propagations/conflicts).
                let removable = self
                    .clauses
                    .get(*clause_id)
                    .is_some_and(|c| !c.deleted && c.lits.len() > 2 && idx < c.lits.len());
                if removable {
                    self.remove_literal_and_rewatch(*clause_id, idx);
                }

                // Recompute LBD after the removal.
                if let Some(clause) = self.clauses.get(*clause_id) {
                    let lits_clone = clause.lits.clone();
                    let new_lbd = self.compute_lbd(&lits_clone);

                    if let Some(clause) = self.clauses.get_mut(*clause_id) {
                        clause.lbd = new_lbd;
                    }

                    strengthened_count += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the `inprocess()` DRAT-deletion gap: pure-literal
    /// elimination and subsumption elimination retire original clauses
    /// directly on the clause database (setting `Clause::deleted` without
    /// going through `ClauseDatabase::remove`), and every clause they retire
    /// must show up as a `d`-line in the DRAT proof — not just the separate
    /// on-the-fly-strengthening path (`remove_literal_and_rewatch`), which
    /// already logged correctly before this fix.
    #[test]
    fn inprocess_drat_deletion_count_matches_removed_clauses() {
        let path = std::env::temp_dir().join("oxiz_sat_inprocess_drat_deletion_count.drat");

        let mut solver = Solver::new();
        let y = solver.new_var();
        let w1 = solver.new_var();
        let w2 = solver.new_var();
        let p = solver.new_var();
        let q = solver.new_var();
        let r = solver.new_var();

        // Pure-literal family: `y` occurs only positively, across two
        // clauses that must both be deleted by `pure_literal_elimination`.
        solver.add_clause([Lit::pos(y), Lit::pos(w1)]);
        solver.add_clause([Lit::pos(y), Lit::pos(w2)]);

        // Subsumption pair: (p ∨ q) subsumes (p ∨ q ∨ r), so the latter must
        // be deleted by `subsumption_elimination`.
        solver.add_clause([Lit::pos(p), Lit::pos(q)]);
        solver.add_clause([Lit::pos(p), Lit::pos(q), Lit::pos(r)]);

        // Decoy giving w1, w2, p, q, r an opposite-polarity occurrence each,
        // so none of them is independently pure (only `y` is) — this keeps
        // the (p ∨ q ∨ r) deletion attributable to subsumption alone rather
        // than being pre-empted by pure-literal elimination on `r`.
        solver.add_clause([
            Lit::neg(w1),
            Lit::neg(w2),
            Lit::neg(p),
            Lit::neg(q),
            Lit::neg(r),
        ]);

        solver
            .enable_drat_proof(&path)
            .expect("enable DRAT proof logging");

        // `inprocess` only acts at decision level 0, which a freshly built
        // solver already is.
        assert_eq!(solver.trail.decision_level(), 0);
        let live_before: Vec<ClauseId> = solver.clauses.iter_ids().collect();

        solver.inprocess();

        let removed: Vec<ClauseId> = live_before
            .into_iter()
            .filter(|&id| solver.clauses.get(id).is_some_and(|c| c.deleted))
            .collect();
        assert_eq!(
            removed.len(),
            3,
            "expected exactly 3 clauses removed (2 pure-literal + 1 subsumed)"
        );

        solver.disable_drat_proof();

        let contents = std::fs::read_to_string(&path).expect("read DRAT proof file");
        std::fs::remove_file(&path).ok();

        let deletion_lines = contents
            .lines()
            .filter(|line| line.trim_start().starts_with("d "))
            .count();

        assert_eq!(
            deletion_lines,
            removed.len(),
            "DRAT deletion-line count must match the number of clauses inprocess() removed"
        );
    }
}
