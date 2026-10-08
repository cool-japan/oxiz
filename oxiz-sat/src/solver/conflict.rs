//! Conflict analysis, clause minimization, and assumption handling

use super::*;
use smallvec::SmallVec;

/// Compute LBD (Literals per Block Distance / "glue" score) from a set of clause literals.
///
/// LBD = number of distinct decision levels among the literals, excluding level 0.
/// Level-0 literals are excluded because they are consequences of unit propagation at the
/// root level and are always true — they do not contribute to the "block distance" that
/// measures how spread across the search tree a learned clause is.
///
/// This is an O(n) computation with no heap allocation in the common case
/// (`SmallVec<[u32; 32]>` avoids a heap allocation for clauses up to 32 distinct decision
/// levels, which covers the overwhelming majority of real CDCL learned clauses).
///
/// This is the standard Glucose/MiniSat LBD definition applied to the **actual learned
/// (1-UIP) clause literals**, so the value it returns satisfies `lbd <= literals.len()`.
/// It is a pure function (no shared scratch state) so it can be called at sites where a
/// `&mut self` borrow of the solver is unavailable — in particular after `self.learnt`
/// has been finalized but while `&self.trail` is borrowed to fire the external hook.
fn compute_lbd_from_literals(literals: &[Lit], trail: &Trail) -> u32 {
    let mut levels: SmallVec<[u32; 32]> = SmallVec::new();
    for &lit in literals {
        let level = trail.level(lit.var());
        if level > 0 && !levels.contains(&level) {
            levels.push(level);
        }
    }
    levels.len() as u32
}

impl Solver {
    /// Bump every variable in `vars` in the VMTF move-to-front queue,
    /// mirroring the same conflict-involvement signal VSIDS/CHB/LRB already
    /// receive. A no-op unless VMTF decisions are enabled (see
    /// [`SolverConfig::use_vmtf`]/[`SolverConfig::enable_stabilize`]) —
    /// bumping is cheap, but there is no reason to maintain the queue's
    /// order when nothing ever reads from it.
    ///
    /// `vars` is sorted by current queue recency first so that bumping
    /// (a move-to-front) preserves whatever relative order the batch already
    /// had, instead of imposing the batch's original collection order (which
    /// reflects resolution-walk order, not recency) on the queue.
    pub(super) fn vmtf_bump_batch(&mut self, vars: &[Var]) {
        if !self.config.use_vmtf {
            return;
        }
        let mut ordered: SmallVec<[Var; 16]> = vars.iter().copied().collect();
        ordered.sort_by_key(|&v| self.vmtf.activity(v));
        for v in ordered {
            let assigned = self.trail.is_assigned(v);
            self.vmtf.bump(v, assigned);
        }
    }

    /// Analyze conflict and learn clause
    pub(super) fn analyze(&mut self, conflict: ClauseId) -> (u32, SmallVec<[Lit; 16]>) {
        // Debug: print conflict info (only with analyze-debug feature)
        #[cfg(feature = "analyze-debug")]
        if self.num_vars <= 5 {
            eprintln!("[ANALYZE] Conflict clause id={:?}", conflict);
            if let Some(c) = self.clauses.get(conflict) {
                let lits_str: Vec<String> = c
                    .lits
                    .iter()
                    .map(|lit| {
                        let val = self.trail.lit_value(*lit);
                        let level = self.trail.level(lit.var());
                        let sign = if lit.is_pos() { "" } else { "~" };
                        format!("{}v{}@{}={:?}", sign, lit.var().index(), level, val)
                    })
                    .collect();
                eprintln!("[ANALYZE] Conflict clause: ({})", lits_str.join(" | "));
            }
            eprintln!("[ANALYZE] Trail:");
            for &lit in self.trail.assignments() {
                let var = lit.var();
                let level = self.trail.level(var);
                let reason = self.trail.reason(var);
                let sign = if lit.is_pos() { "" } else { "~" };
                eprintln!("  {}v{}@{} reason={:?}", sign, var.index(), level, reason);
            }
        }

        self.learnt.clear();
        self.learnt.push(Lit::from_code(0)); // Placeholder for asserting literal

        let mut counter = 0;
        let mut p = None;
        let mut index = self.trail.assignments().len();

        // The "conflict level" is the highest decision level among the conflict
        // clause's literals. In textbook CDCL this always equals
        // `trail.decision_level()`, because propagation is run to completion at
        // every level before a new decision is taken. However, clauses added
        // *on the fly* — theory reason/lemma clauses in CDCL(T), or clauses
        // encountered after chronological backtracking — can be falsified at a
        // level strictly BELOW the current decision level. Running 1-UIP
        // resolution relative to `decision_level()` in that situation is
        // unsound for backtracking: the conflict clause contributes NO literal
        // at the pivot level, so the current-level counter starts at 0, the
        // trail walk underflows it, and the asserting literal ends up at a level
        // <= the computed backtrack level. Backtracking then fails to unassign
        // that variable and `learn_clause` re-assigns it in place — corrupting
        // the trail (observed as a wrong top-level UNSAT on disjunctive LIA).
        // Anchoring the analysis at the genuine conflict level restores the
        // 1-UIP invariant (asserting literal strictly above the backtrack
        // level) for both the normal and the on-the-fly-clause cases.
        let current_level = {
            let mut lvl = 0;
            if let Some(c) = self.clauses.get(conflict) {
                for &lit in &c.lits {
                    let l = self.trail.level(lit.var());
                    if l > lvl {
                        lvl = l;
                    }
                }
            }
            lvl
        };

        // A conflict whose genuine level is 0 has EVERY literal falsified under
        // unconditional (level-0) assignments — a root-level refutation, so the
        // instance is UNSAT. This can happen above decision level 0 when an
        // on-the-fly clause (a theory reason/lemma clause) is added already
        // fully falsified at the root: the watched-literal scheme only visits it
        // on the next propagation, which may run at a higher decision level.
        // There is no asserting literal to learn, so we return an empty clause
        // (backtrack level 0) — the caller treats an empty learned clause as
        // fundamental UNSAT, exactly as `analyze_theory_conflict` already does.
        // Fabricating a 1-UIP clause here instead would resolve the trail's
        // bottom literal into a spurious unit clause that contradicts a
        // level-0 fact, corrupting the trail (the earlier `decision_level()`
        // fallback did precisely this, tripping the trail-consistency assert).
        if current_level == 0 {
            self.learnt.clear();
            return (0, SmallVec::new());
        }

        // Reset seen flags
        for s in &mut self.seen {
            *s = false;
        }

        // Collect variables to bump in batch (avoids repeated heap sift-ups)
        let mut vars_to_bump: SmallVec<[Var; 32]> = SmallVec::new();

        let mut reason_clause = conflict;

        while let Some(clause) = self.clauses.get(reason_clause) {
            // Process reason clause (must exist, as it's either conflict or a propagation reason)
            let is_learned = clause.learned;

            // Record clause usage for tier promotion and bump activity (if it's a learned clause)
            if is_learned && let Some(clause_mut) = self.clauses.get_mut(reason_clause) {
                clause_mut.record_usage();
                // Promote to Core if LBD ≤ 2 (GLUE clause)
                if clause_mut.lbd <= 2 {
                    clause_mut.promote_to_core();
                }
                // Bump clause activity (MapleSAT-style)
                clause_mut.activity += self.clause_bump_increment;
            }

            let Some(clause) = self.clauses.get(reason_clause) else {
                break;
            };
            for &lit in &clause.lits {
                // When resolving a *reason* clause (`p` is Some), the propagated
                // literal `p` is the one being resolved out: it is TRUE on the trail
                // and must NOT be added to the learned clause. We skip it BY VALUE
                // rather than by a fixed index, because binary-implication-graph
                // propagation (propagate.rs) records the reason without moving the
                // implied literal to index 0 — so the propagated literal may sit at
                // index 1. Skipping index 0 positionally would drop the false
                // antecedent at index 0, producing over-strong (unsound) learned
                // clauses. For the initial conflict clause `p` is None, so every
                // literal is processed.
                if p == Some(lit) {
                    continue;
                }
                let var = lit.var();
                let level = self.trail.level(var);

                if !self.seen[var.index()] && level > 0 {
                    self.seen[var.index()] = true;
                    // Collect variable for batch bumping instead of individual bumps
                    vars_to_bump.push(var);

                    if level == current_level {
                        counter += 1;
                    } else {
                        // Add the literal itself (not negated) to the learned clause.
                        // The conflict clause has all literals FALSE. To prevent this
                        // conflict, we need at least one of these literals to become TRUE.
                        self.learnt.push(lit);
                    }
                }
            }

            // Find next literal to resolve on: the most recently assigned
            // still-unresolved literal AT THE CONFLICT LEVEL.
            //
            // The level check is what makes this walk correct under
            // chronological backtracking. The trail is no longer sorted by
            // decision level — a literal implied at a low level can sit near the
            // top of the trail — so "the last `seen` literal" is not necessarily
            // a conflict-level literal any more. Resolving on a lower-level one
            // would decrement the conflict-level counter for a literal that was
            // never counted in it, terminating the 1-UIP loop early and emitting
            // a clause that is missing literals, i.e. stronger than what
            // resolution actually derives. Reference: Z3's `sat_solver.cpp`,
            // whose 1-UIP loop skips marked literals with
            // `lvl(c_var) != m_conflict_lvl` for exactly this reason.
            let mut current_lit = Lit::from_code(0); // sentinel default
            let mut found_next = false;
            loop {
                if index == 0 {
                    // Guard against underflow: this should not happen in a
                    // well-formed conflict, but theory-conflict injection can
                    // occasionally produce a degenerate state.  Break out to
                    // avoid a usize overflow panic.
                    break;
                }
                index -= 1;
                current_lit = self.trail.assignments()[index];
                let var = current_lit.var();
                if self.seen[var.index()] && self.trail.level(var) == current_level {
                    p = Some(current_lit);
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                break;
            }

            counter -= 1;
            if counter == 0 {
                break;
            }

            let var = current_lit.var();
            match self.trail.reason(var) {
                Reason::Propagation(c) => reason_clause = c,
                _ => break,
            }
        }

        // Batch bump all collected variables at once (single heap rebuild)
        self.vsids.bump_batch(&vars_to_bump);
        self.chb.bump_batch(&vars_to_bump);
        self.lrb.on_reason_batch(&vars_to_bump);
        self.vmtf_bump_batch(&vars_to_bump);

        // Set asserting literal (p is guaranteed to be Some at this point)
        if let Some(lit) = p {
            self.learnt[0] = lit.negate();
        }

        // Repair an early exit from the resolution loop.
        //
        // The loop above stops as soon as it reaches a literal with no clausal
        // reason (a decision, or a theory propagation whose explanation is not a
        // clause in the database). If `counter` has not reached 0 by then, some
        // conflict-level literals were counted but never resolved away, and they
        // are simply missing from `self.learnt` — an over-strong clause, which is
        // unsound: it can drive the solver to a bogus root-level unit and hence
        // to a false `unsat`. Every such literal is still `seen`, and its
        // contribution to the resolvent is the negation of its trail assignment
        // (the resolvent's literals are all false), so adding those recovers a
        // clause that resolution genuinely derives.
        if counter > 0 {
            let uip_var = p.map(|lit| lit.var());
            for &lit in self.trail.assignments() {
                let var = lit.var();
                if self.seen[var.index()]
                    && self.trail.level(var) == current_level
                    && Some(var) != uip_var
                {
                    // Stays `seen`: that flag is how minimization recognises the
                    // literals that are in the learned clause.
                    self.learnt.push(lit.negate());
                }
            }
        }

        // Minimize learnt clause using recursive resolution
        self.minimize_learnt_clause();

        // Compute the real LBD from the FINAL learned (1-UIP) clause literals.
        // This is the standard Glucose definition: the number of distinct decision
        // levels in the learned clause itself (level 0 excluded), not the larger
        // `vars_to_bump` set. It is computed AFTER minimization so it reflects the
        // exact clause that will be stored, and therefore satisfies lbd <= clause len.
        let lbd = compute_lbd_from_literals(&self.learnt, &self.trail);

        // Notify external heuristic of each conflict-involved variable with the
        // learned-clause LBD score.
        if let Some(ref ext) = self.config.external_branching
            && let Ok(mut h) = ext.lock()
        {
            for &var in &vars_to_bump {
                h.on_conflict_var_with_lbd(var, lbd);
            }
        }

        // Order the clause so that its two highest-level literals occupy the
        // watched positions, `learnt[0]` being the highest.
        //
        // For a textbook 1-UIP clause `learnt[0]` is already the unique
        // conflict-level literal, so this only moves the second watch into place.
        // Under chronological backtracking that is no longer guaranteed: the
        // asserting literal is assigned at its true implication level, which may
        // sit *below* another literal of the clause. Leaving such a clause
        // unordered would compute a backtrack level above `learnt[0]`'s level, so
        // backtracking would not unassign it and the learned clause would
        // re-assign an already-assigned variable, corrupting the trail. Z3 does
        // the same swap in `learn_lemma_and_backjump` ("with scope tracking and
        // chronological backtracking, consequent may not be at highest decision
        // level").
        let uip_level = self.reorder_learnt_watches();

        // Level at which the clause becomes unit (the second highest level).
        let assertion_level = if self.learnt.len() <= 1 {
            0
        } else {
            self.trail.level(self.learnt[1].var())
        };

        // Apply chronological backtracking if enabled
        let backtrack_level = self.chrono_backtrack.compute_backtrack_level(
            &self.trail,
            &self.learnt,
            uip_level,
            assertion_level,
        );

        // Track chronological vs non-chronological backtracks
        if backtrack_level != assertion_level {
            self.stats.chrono_backtracks += 1;
        } else {
            self.stats.non_chrono_backtracks += 1;
        }

        // Debug: print learned clause (only with analyze-debug feature)
        #[cfg(feature = "analyze-debug")]
        if self.num_vars <= 5 {
            let lits_str: Vec<String> = self
                .learnt
                .iter()
                .map(|lit| {
                    let sign = if lit.is_pos() { "" } else { "~" };
                    format!("{}v{}", sign, lit.var().index())
                })
                .collect();
            eprintln!(
                "[ANALYZE] Learned clause: ({}), backtrack_level={}",
                lits_str.join(" | "),
                backtrack_level
            );
        }

        // Trail-consistency invariants (debug builds only, so no release-path
        // panic on user input). A well-formed 1-UIP learned clause has its
        // asserting literal (learnt[0]) at the conflict level and every other
        // literal strictly below the backtrack level, so backtracking is
        // guaranteed to unassign the asserting variable before `learn_clause`
        // re-asserts it. If either invariant is violated the trail would be
        // corrupted by an in-place re-assignment.
        debug_assert!(
            self.learnt.is_empty()
                || !self.trail.is_assigned(self.learnt[0].var())
                || self.trail.level(self.learnt[0].var()) > backtrack_level,
            "asserting literal must be above the backtrack level (uip level {}, backtrack {})",
            self.trail.level(self.learnt[0].var()),
            backtrack_level
        );
        debug_assert!(
            self.learnt
                .iter()
                .skip(1)
                .all(|lit| self.trail.level(lit.var()) <= backtrack_level),
            "every non-asserting literal must be at or below the backtrack level"
        );

        (backtrack_level, self.learnt.clone())
    }

    /// Move the two highest-level literals of `self.learnt` into the watched
    /// positions — `learnt[0]` highest, `learnt[1]` second highest — and return
    /// `learnt[0]`'s decision level.
    ///
    /// This is the standard "watch the two literals falsified latest" invariant.
    /// For a textbook 1-UIP clause `learnt[0]` already holds the unique
    /// conflict-level literal, so only the second watch actually moves.
    fn reorder_learnt_watches(&mut self) -> u32 {
        if self.learnt.is_empty() {
            return 0;
        }

        let mut max_idx = 0;
        let mut max_level = self.trail.level(self.learnt[0].var());
        for i in 1..self.learnt.len() {
            let level = self.trail.level(self.learnt[i].var());
            if level > max_level {
                max_level = level;
                max_idx = i;
            }
        }
        self.learnt.swap(0, max_idx);

        if self.learnt.len() > 1 {
            let mut second_idx = 1;
            let mut second_level = self.trail.level(self.learnt[1].var());
            for i in 2..self.learnt.len() {
                let level = self.trail.level(self.learnt[i].var());
                if level > second_level {
                    second_level = level;
                    second_idx = i;
                }
            }
            self.learnt.swap(1, second_idx);
        }

        max_level
    }

    /// Minimize the learned clause by removing redundant literals
    ///
    /// A literal can be removed if it is implied by the remaining literals.
    /// We use a recursive check: a literal l is redundant if its reason clause
    /// contains only literals that are either:
    /// - Already in the learnt clause (marked as seen)
    /// - At decision level 0 (always true in the learned clause context)
    /// - Themselves redundant (recursive check)
    ///
    /// This also performs clause strengthening by checking for stronger implications
    pub(super) fn minimize_learnt_clause(&mut self) {
        if self.learnt.len() <= 2 {
            // Don't minimize very small clauses
            return;
        }

        let original_len = self.learnt.len();

        // Mark all literals in the learned clause as "in clause"
        // We use analyze_stack to track literals to check
        self.analyze_stack.clear();

        // Phase 1: Basic minimization - remove redundant literals
        let mut j = 1; // Write position
        for i in 1..self.learnt.len() {
            let lit = self.learnt[i];
            if self.lit_is_redundant(lit) {
                // Skip this literal (it's redundant)
            } else {
                // Keep this literal
                self.learnt[j] = lit;
                j += 1;
            }
        }
        self.learnt.truncate(j);

        // Phase 2: Clause strengthening - check for self-subsuming resolution
        // If the clause contains both l and ~l' where l' is in a reason clause,
        // we might be able to strengthen the clause
        self.strengthen_learnt_clause();

        // Track minimization statistics
        let final_len = self.learnt.len();
        if final_len < original_len {
            self.stats.minimizations += 1;
            self.stats.literals_removed += (original_len - final_len) as u64;
        }
    }

    /// Strengthen the learned clause using on-the-fly self-subsuming resolution
    pub(super) fn strengthen_learnt_clause(&mut self) {
        if self.learnt.len() <= 2 {
            return;
        }

        // Check each literal to see if we can strengthen by resolution
        let mut j = 1;
        for i in 1..self.learnt.len() {
            let lit = self.learnt[i];
            let var = lit.var();

            // Check if this literal can be strengthened
            if let Reason::Propagation(reason_id) = self.trail.reason(var)
                && let Some(reason_clause) = self.clauses.get(reason_id)
                && reason_clause.lits.len() == 2
            {
                // Binary reason: one literal is lit, the other is the implied literal
                let other_lit = if reason_clause.lits[0] == lit.negate() {
                    reason_clause.lits[1]
                } else if reason_clause.lits[1] == lit.negate() {
                    reason_clause.lits[0]
                } else {
                    // Keep the literal
                    self.learnt[j] = lit;
                    j += 1;
                    continue;
                };

                // If other_lit is already in the learned clause at level 0,
                // we can remove lit
                if self.trail.level(other_lit.var()) == 0 && self.seen[other_lit.var().index()] {
                    // Skip this literal (strengthened)
                    continue;
                }
            }

            // Keep this literal
            self.learnt[j] = lit;
            j += 1;
        }
        self.learnt.truncate(j);
    }

    /// Check if a literal is redundant in the learned clause
    ///
    /// A literal is redundant if its reason clause only contains:
    /// - Literals marked as seen (in the learned clause)
    /// - Literals at decision level 0
    /// - Literals that are themselves redundant (recursive)
    pub(super) fn lit_is_redundant(&mut self, lit: Lit) -> bool {
        let var = lit.var();

        // Decision variables and theory propagations are not redundant
        let reason = match self.trail.reason(var) {
            Reason::Decision => return false,
            Reason::Theory => return false, // Theory propagations can't be minimized
            Reason::Propagation(c) => c,
        };

        let reason_clause = match self.clauses.get(reason) {
            Some(c) => c,
            None => return false,
        };

        // Check all literals in the reason clause
        for &reason_lit in &reason_clause.lits {
            if reason_lit == lit.negate() {
                // Skip the literal we're analyzing
                continue;
            }

            let reason_var = reason_lit.var();

            // Level 0 literals are always OK
            if self.trail.level(reason_var) == 0 {
                continue;
            }

            // If the literal is in the learned clause (seen), it's OK
            if self.seen[reason_var.index()] {
                continue;
            }

            // Otherwise, this literal prevents minimization
            // (A full recursive check would be more powerful but more expensive)
            return false;
        }

        true
    }

    /// Analyze a theory conflict (given as a list of literals that are all false)
    pub(super) fn analyze_theory_conflict(
        &mut self,
        conflict_lits: &[Lit],
    ) -> (u32, SmallVec<[Lit; 16]>) {
        // A well-formed theory conflict clause is fully falsified — every literal
        // is assigned false on the trail — which is what makes the 1-UIP
        // resolution below well-defined. The MBQI / quantifier-instantiation path,
        // however, can hand us a "conflict" clause that still contains an
        // UNASSIGNED literal. The usual cause is a variable that was assigned when
        // the theory recorded the lemma but has since been unassigned by a
        // backtrack: `Trail` leaves `VarInfo.level` stale on unassignment, so
        // `trail.level()` reports a bogus non-zero level for it.
        //
        // Feeding such a clause into the all-false 1-UIP machinery is unsound. The
        // stale level becomes a spurious `current_level`; the pivot counter is
        // incremented for a literal that is not on the trail, so the backward
        // trail walk can never discharge it and instead resolves against an
        // unrelated variable at a lower level; the asserting literal is then
        // duplicated at the computed backtrack level. That produces
        // `backtrack_level == uip_level` (tripping the debug-assert below in debug
        // builds) and, in release builds, corrupts the trail into a wrong
        // top-level UNSAT on quantified UFLIA.
        //
        // A clause with an open literal is not a conflict at all but an
        // *asserting* theory lemma: it is unit under the current assignment (one
        // open literal, the rest false) and must simply propagate that literal.
        // Route it to a dedicated, trail-safe handler; keep the 1-UIP path for
        // genuine all-false conflicts (and for the pre-existing already-satisfied
        // case, which does not corrupt the trail).
        if conflict_lits
            .iter()
            .any(|&l| self.trail.lit_value(l) == LBool::Undef)
        {
            return self.analyze_theory_asserting_lemma(conflict_lits);
        }

        self.learnt.clear();
        self.learnt.push(Lit::from_code(0)); // Placeholder

        let mut counter = 0;

        // Anchor the analysis at the genuine conflict level — the highest
        // decision level among the (all-false) theory conflict literals —
        // rather than `trail.decision_level()`. A theory conflict can be
        // reported while the SAT trail sits at a strictly higher decision level
        // than any literal actually involved in the conflict; running 1-UIP
        // against `decision_level()` would then leave the asserting literal at
        // or below the backtrack level and corrupt the trail via an in-place
        // re-assignment in `learn_clause`. See the companion note in `analyze`.
        let current_level = {
            let mut lvl = 0;
            for &lit in conflict_lits {
                let l = self.trail.level(lit.var());
                if l > lvl {
                    lvl = l;
                }
            }
            lvl
        };

        // Reset seen flags
        for s in &mut self.seen {
            *s = false;
        }

        // Collect variables for batch bumping
        let mut vars_to_bump: SmallVec<[Var; 32]> = SmallVec::new();

        // Process conflict literals
        let mut all_level_zero = true;
        for &lit in conflict_lits {
            let var = lit.var();
            let level = self.trail.level(var);

            if !self.seen[var.index()] && level > 0 {
                all_level_zero = false;
                self.seen[var.index()] = true;
                vars_to_bump.push(var);

                if level == current_level {
                    counter += 1;
                } else {
                    // Add the literal itself (not negated) to the learned clause.
                    // The conflict clause has all literals FALSE. To prevent this
                    // conflict, we need at least one of these literals to become TRUE.
                    // So we add the literal directly to the learned clause.
                    self.learnt.push(lit);
                }
            }
        }

        // If ALL conflict literals are at level 0, this is a fundamental UNSAT
        // that cannot be resolved by backtracking. Return an empty learned clause
        // with backtrack_level=0 as a signal.
        if !conflict_lits.is_empty() && all_level_zero {
            return (0, SmallVec::new());
        }

        // Find UIP by walking back through trail
        let mut index = self.trail.assignments().len();
        let mut p = None;

        while counter > 0 {
            if index == 0 {
                break; // Avoid underflow — no more trail entries
            }
            index -= 1;
            if index >= self.trail.assignments().len() {
                break; // Guard against stale length
            }
            let current_lit = self.trail.assignments()[index];
            p = Some(current_lit);
            let var = current_lit.var();

            if self.seen[var.index()] {
                counter -= 1;

                if counter > 0
                    && let Reason::Propagation(reason_clause) = self.trail.reason(var)
                    && let Some(clause) = self.clauses.get(reason_clause)
                {
                    // Get reason and process its literals.
                    // `current_lit` is the propagated (TRUE) literal being resolved
                    // out; skip it BY VALUE rather than assuming it sits at index 0.
                    // Binary-implication-graph propagation does not move the implied
                    // literal to index 0, so a positional `[1..]` skip would drop the
                    // false antecedent at index 0 and yield unsound learned clauses.
                    for &lit in &clause.lits {
                        if lit == current_lit {
                            continue;
                        }
                        let reason_var = lit.var();
                        let level = self.trail.level(reason_var);

                        if !self.seen[reason_var.index()] && level > 0 {
                            self.seen[reason_var.index()] = true;
                            vars_to_bump.push(reason_var);

                            if level == current_level {
                                counter += 1;
                            } else {
                                // Add the literal itself to the learned clause
                                self.learnt.push(lit);
                            }
                        }
                    }
                }
            }
        }

        // Batch bump all collected variables
        self.vsids.bump_batch(&vars_to_bump);
        self.chb.bump_batch(&vars_to_bump);
        self.lrb.on_reason_batch(&vars_to_bump);
        self.vmtf_bump_batch(&vars_to_bump);

        // Set asserting literal
        if let Some(uip) = p {
            self.learnt[0] = uip.negate();
        }

        // Minimize
        self.minimize_learnt_clause();

        // Compute the real LBD from the FINAL learned clause literals (post-minimization),
        // matching the standard Glucose definition rather than using the larger
        // `vars_to_bump` proxy. For theory conflicts the learned clause shape may differ
        // from Boolean conflicts, but the distinct-decision-level count of the actual
        // learned clause is the correct glue score and never exceeds the clause length.
        let lbd = compute_lbd_from_literals(&self.learnt, &self.trail);

        // Notify external heuristic of each conflict-involved variable with the
        // learned-clause LBD score.
        if let Some(ref ext) = self.config.external_branching
            && let Ok(mut h) = ext.lock()
        {
            for &var in &vars_to_bump {
                h.on_conflict_var_with_lbd(var, lbd);
            }
        }

        // Calculate backtrack level: search `learnt[1..]` for the
        // second-highest level and move it into the watched position 1.
        //
        // Investigated generalizing this to the same two-phase
        // "find-the-true-max-then-the-true-second-max" routine
        // `reorder_learnt_watches` uses for the plain-conflict path
        // (`analyze`), on the theory that the same trail-non-monotonicity
        // chronological backtracking introduces there could also put a
        // higher-level literal below `learnt[0]` here. Concluded that
        // generalization does not apply to this function: unlike `analyze`,
        // this walk's `counter` only decrements for literals at the anchored
        // `current_level` (the theory conflict's own max level, computed
        // above), so the UIP `p` it settles on is by construction a
        // `current_level` literal — `learnt[0]` already holds the clause's
        // true maximum. This function also never calls
        // `chrono_backtrack.compute_backtrack_level` (unlike `analyze`), so
        // it is not part of the chronological-backtracking-aware path that
        // motivated `reorder_learnt_watches`'s extra phase in the first
        // place. Swapping in the fuller routine would additionally risk
        // moving `learnt[0]` off the literal `p` picked above, changing which
        // literal this clause propagates, with no reachable counterexample
        // found to justify that behavior change. Left as the narrower,
        // already-correct search.
        let backtrack_level = if self.learnt.len() == 1 {
            0
        } else {
            let mut max_level = 0;
            let mut max_idx = 1;
            for (i, &lit) in self.learnt.iter().enumerate().skip(1) {
                let level = self.trail.level(lit.var());
                if level > max_level {
                    max_level = level;
                    max_idx = i;
                }
            }
            self.learnt.swap(1, max_idx);
            max_level
        };

        // Trail-consistency invariants (debug builds only): the asserting
        // literal must sit strictly above the backtrack level so backtracking
        // unassigns it before it is re-asserted, and every other learned
        // literal must be at or below the backtrack level. See `analyze`.
        debug_assert!(
            self.learnt.is_empty()
                || !self.trail.is_assigned(self.learnt[0].var())
                || self.trail.level(self.learnt[0].var()) > backtrack_level,
            "theory: asserting literal must be above the backtrack level (uip level {}, backtrack {})",
            self.trail.level(self.learnt[0].var()),
            backtrack_level
        );
        debug_assert!(
            self.learnt
                .iter()
                .skip(1)
                .all(|lit| self.trail.level(lit.var()) <= backtrack_level),
            "theory: every non-asserting literal must be at or below the backtrack level"
        );

        (backtrack_level, self.learnt.clone())
    }

    /// Build a learned clause from a theory lemma that is *asserting* rather than
    /// *conflicting*: at least one of its literals is still unassigned while the
    /// rest are false, so the clause is unit under the current assignment and must
    /// propagate its open literal instead of driving 1-UIP resolution.
    ///
    /// The learned clause is the full, deduplicated theory lemma (dropping a
    /// literal without resolving it would be unsound — the lemma's validity does
    /// not carry over to any strict subset). It is returned with an unassigned
    /// literal at index 0 — the asserting / watch-0 literal that `learn_clause`
    /// will propagate — and the highest-level false literal at index 1 (watch 1).
    ///
    /// The backtrack level is the maximum decision level among the *assigned*
    /// (false) literals only; an unassigned literal's `VarInfo.level` is stale and
    /// must never be consulted. After backtracking to that level every false
    /// literal remains assigned false and index 0 remains unassigned, so the
    /// clause is unit and propagates index 0 — exactly the two-watched-literal
    /// contract `learn_clause` relies on. Computing the level from assigned
    /// literals alone is what keeps it in range with the live trail.
    fn analyze_theory_asserting_lemma(
        &mut self,
        conflict_lits: &[Lit],
    ) -> (u32, SmallVec<[Lit; 16]>) {
        self.learnt.clear();

        // Deduplicate by variable (a lemma may legitimately list a literal twice;
        // it must appear at most once in the learned clause). First occurrence
        // wins, preserving the theory's reported order.
        let mut seen_vars: SmallVec<[u32; 16]> = SmallVec::new();
        let mut asserting_idx: Option<usize> = None;
        let mut vars_to_bump: SmallVec<[Var; 16]> = SmallVec::new();
        for &lit in conflict_lits {
            let vi = lit.var().index() as u32;
            if seen_vars.contains(&vi) {
                continue;
            }
            seen_vars.push(vi);
            let idx = self.learnt.len();
            self.learnt.push(lit);
            if self.trail.lit_value(lit) == LBool::Undef {
                if asserting_idx.is_none() {
                    asserting_idx = Some(idx);
                }
            } else {
                // Assigned (false, or — for the defensive already-satisfied case —
                // true). Bump it so the heuristics still learn from the event.
                vars_to_bump.push(lit.var());
            }
        }

        // A lemma with no literals cannot arise from a real theory conflict; guard
        // by signalling a fundamental refutation (empty clause), matching the
        // all-level-0 convention of `analyze_theory_conflict`.
        if self.learnt.is_empty() {
            return (0, SmallVec::new());
        }

        // Place the (first) unassigned literal at index 0 as the asserting literal.
        // `any(... == Undef)` in the caller guarantees `asserting_idx` is `Some`.
        if let Some(ai) = asserting_idx {
            self.learnt.swap(0, ai);
        }

        // Bump activity for the falsified literals, mirroring 1-UIP conflict
        // analysis so branching heuristics still react to the near-conflict.
        self.vsids.bump_batch(&vars_to_bump);
        self.chb.bump_batch(&vars_to_bump);
        self.lrb.on_reason_batch(&vars_to_bump);
        self.vmtf_bump_batch(&vars_to_bump);

        // Backtrack level = highest decision level among the *assigned* (false)
        // non-asserting literals; unassigned literals' stale levels are ignored.
        // Promote that literal to index 1 to serve as the second watch.
        let mut backtrack_level = 0u32;
        let mut second_idx = 0usize;
        for i in 1..self.learnt.len() {
            let lit = self.learnt[i];
            if self.trail.is_assigned(lit.var()) {
                let level = self.trail.level(lit.var());
                if level >= backtrack_level {
                    backtrack_level = level;
                    second_idx = i;
                }
            }
        }
        if second_idx >= 1 {
            self.learnt.swap(1, second_idx);
        }

        (backtrack_level, self.learnt.clone())
    }

    /// Extract the core of assumptions responsible for a *directly conflicting*
    /// assumption — one whose required polarity is already falsified on the trail
    /// when it is about to be asserted (index `conflict_idx`).
    ///
    /// The failed assumption's variable sits on the trail with the opposite phase,
    /// implied (transitively) by earlier assumptions through unit propagation.
    /// Seeding the analysis from that variable and resolving every antecedent back
    /// to its decision (assumption) roots yields *all* contributing assumptions,
    /// not merely the failed one. The previous implementation only ever returned
    /// the single failed assumption (its `seen`-based guard was never populated
    /// for this path), so a core such as `{a, b}` for
    /// `a ∧ b ∧ (¬a ∨ ¬b)` under `[a, b]` came back as just `{b}` — an incomplete,
    /// and therefore unsound-for-minimisation, core.
    pub(super) fn extract_assumption_core(
        &mut self,
        assumptions: &[Lit],
        conflict_idx: usize,
    ) -> Vec<Lit> {
        let failed = assumptions[conflict_idx];
        // Only assumptions asserted up to (and including) the failure can be on
        // the trail and thus contribute.
        self.analyze_final_core(&[failed], &[failed], &assumptions[..=conflict_idx])
    }

    /// Shared "analyze final" implementation (à la MiniSat `analyzeFinal`).
    ///
    /// Marks the `seed` literals' variables, walks the trail from newest to
    /// oldest resolving each marked propagated literal against its reason clause,
    /// and collects the assumption literals sitting at the decision roots. Any
    /// literals in `include` are unconditionally placed in the resulting core
    /// first (used to force the directly-failed assumption into its own core).
    ///
    /// Uses the solver's shared `seen` scratch buffer and restores it to all-false
    /// before returning, so it composes cleanly with the rest of conflict analysis.
    fn analyze_final_core(
        &mut self,
        seed: &[Lit],
        include: &[Lit],
        assumptions: &[Lit],
    ) -> Vec<Lit> {
        use crate::prelude::HashSet;

        // The assumption literals, by literal code.  An assumption `a` is
        // placed via `assign_decision(a)`, so the decision literal the walk
        // meets on the trail *is* the assumption that put it there.
        //
        // Keyed by the literal, not by its variable: when both `x` and `¬x`
        // are assumed, the failed one is `include`d and the one on the trail
        // is its partner, and a variable-keyed core collapsed the two into
        // the failed literal alone — a "core" `{¬x}` that is satisfiable on
        // its own (found 2026-09-28 by `oxiz-theories`'
        // `bv_root_scoped_definitions` differential test, where `a <u b` and
        // `a >=u b` are both asserted).
        let assumption_codes: HashSet<u32> = assumptions.iter().map(|a| a.code()).collect();

        let mut core: Vec<Lit> = Vec::new();
        let mut in_core: HashSet<u32> = HashSet::new();
        for &lit in include {
            if in_core.insert(lit.code()) {
                core.push(lit);
            }
        }

        // Seed the marks with every above-root seed variable.
        let mut touched: Vec<usize> = Vec::new();
        for &lit in seed {
            let var = lit.var();
            if self.trail.level(var) > 0 {
                let vi = var.index();
                if vi < self.seen.len() && !self.seen[vi] {
                    self.seen[vi] = true;
                    touched.push(vi);
                }
            }
        }

        // Walk the trail newest-to-oldest. `assignments()` is a snapshot copy so
        // the loop can freely borrow `self.clauses` / `self.trail` / `self.seen`.
        let trail_lits: Vec<Lit> = self.trail.assignments().to_vec();
        for &tlit in trail_lits.iter().rev() {
            let var = tlit.var();
            let vi = var.index();
            if vi >= self.seen.len() || !self.seen[vi] {
                continue;
            }
            self.seen[vi] = false;
            if self.trail.level(var) == 0 {
                continue;
            }
            match self.trail.reason(var) {
                Reason::Decision | Reason::Theory => {
                    // A decision root above level 0: if it is one of our
                    // assumptions, it belongs in the core.
                    if assumption_codes.contains(&tlit.code()) && in_core.insert(tlit.code()) {
                        core.push(tlit);
                    }
                }
                Reason::Propagation(cid) => {
                    // Resolve against the reason clause: mark every other literal's
                    // variable so its own antecedents are visited in turn.
                    let antecedents: SmallVec<[Var; 8]> = match self.clauses.get(cid) {
                        Some(clause) => clause
                            .lits
                            .iter()
                            .map(|l| l.var())
                            .filter(|&av| av != var)
                            .collect(),
                        None => SmallVec::new(),
                    };
                    for av in antecedents {
                        if self.trail.level(av) > 0 {
                            let avi = av.index();
                            if avi < self.seen.len() && !self.seen[avi] {
                                self.seen[avi] = true;
                                touched.push(avi);
                            }
                        }
                    }
                }
            }
        }

        // Restore the shared scratch buffer (any marks not cleared during the walk).
        for vi in touched {
            if vi < self.seen.len() {
                self.seen[vi] = false;
            }
        }

        core
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trail::Trail;

    // ---------------------------------------------------------------------------
    // Tests for compute_lbd_from_literals
    // ---------------------------------------------------------------------------

    #[test]
    fn test_compute_lbd_all_same_level() {
        // Three literals whose vars are all assigned at level 3 → LBD = 1.
        let n = 4;
        let mut trail = Trail::new(n);
        // Level 0 is implicit; push 3 levels.
        trail.new_decision_level(); // → level 1
        trail.new_decision_level(); // → level 2
        trail.new_decision_level(); // → level 3

        let v0 = Var::new(0);
        let v1 = Var::new(1);
        let v2 = Var::new(2);
        trail.assign_decision(Lit::pos(v0));
        trail.assign_decision(Lit::pos(v1));
        trail.assign_decision(Lit::pos(v2));

        let lits = [Lit::pos(v0), Lit::neg(v1), Lit::pos(v2)];
        let lbd = compute_lbd_from_literals(&lits, &trail);
        assert_eq!(lbd, 1, "all literals at same level → LBD should be 1");
    }

    #[test]
    fn test_compute_lbd_distinct_levels() {
        // Three literals at levels 1, 2, 3 → LBD = 3.
        let n = 4;
        let mut trail = Trail::new(n);

        let v0 = Var::new(0);
        let v1 = Var::new(1);
        let v2 = Var::new(2);

        trail.new_decision_level(); // → level 1
        trail.assign_decision(Lit::pos(v0));

        trail.new_decision_level(); // → level 2
        trail.assign_decision(Lit::pos(v1));

        trail.new_decision_level(); // → level 3
        trail.assign_decision(Lit::pos(v2));

        let lits = [Lit::pos(v0), Lit::pos(v1), Lit::neg(v2)];
        let lbd = compute_lbd_from_literals(&lits, &trail);
        assert_eq!(lbd, 3, "literals at levels 1, 2, 3 → LBD should be 3");
    }

    #[test]
    fn test_compute_lbd_excludes_level_zero() {
        // Literals: one var at level 0 (unit prop), two at level 2 → LBD = 1.
        // Level-0 variables must not be counted.
        let n = 4;
        let mut trail = Trail::new(n);

        let v0 = Var::new(0); // Will be at level 0
        let v1 = Var::new(1); // Will be at level 2
        let v2 = Var::new(2); // Will be at level 2

        // Assign v0 at level 0 (root decision level, no new_decision_level call).
        trail.assign_decision(Lit::pos(v0));

        trail.new_decision_level(); // → level 1 (unused)
        trail.new_decision_level(); // → level 2
        trail.assign_decision(Lit::pos(v1));
        trail.assign_decision(Lit::pos(v2));

        let lits = [Lit::pos(v0), Lit::pos(v1), Lit::pos(v2)];
        let lbd = compute_lbd_from_literals(&lits, &trail);
        assert_eq!(
            lbd, 1,
            "level-0 var must be excluded; only level-2 vars count → LBD = 1"
        );
    }

    #[test]
    fn test_compute_lbd_mixed_duplicates_and_zero() {
        // v0 @ level 0 (excluded), v1 @ level 2, v2 @ level 4, v3 @ level 2 (duplicate)
        // → distinct non-zero levels: {2, 4} → LBD = 2.
        let n = 5;
        let mut trail = Trail::new(n);

        let v0 = Var::new(0);
        let v1 = Var::new(1);
        let v2 = Var::new(2);
        let v3 = Var::new(3);

        trail.assign_decision(Lit::pos(v0)); // level 0

        trail.new_decision_level(); // → 1
        trail.new_decision_level(); // → 2
        trail.assign_decision(Lit::pos(v1));
        trail.assign_decision(Lit::pos(v3));

        trail.new_decision_level(); // → 3
        trail.new_decision_level(); // → 4
        trail.assign_decision(Lit::pos(v2));

        let lits = [Lit::pos(v0), Lit::pos(v1), Lit::neg(v2), Lit::pos(v3)];
        let lbd = compute_lbd_from_literals(&lits, &trail);
        assert_eq!(lbd, 2, "levels {{2, 4}} → LBD = 2");
    }

    #[test]
    fn test_compute_lbd_empty_literals() {
        // Empty literal set → LBD = 0.
        let trail = Trail::new(0);
        let lits: [Lit; 0] = [];
        let lbd = compute_lbd_from_literals(&lits, &trail);
        assert_eq!(lbd, 0, "empty literal set → LBD = 0");
    }

    // ---------------------------------------------------------------------------
    // Integration test: conflict analysis passes LBD to the external hook
    // ---------------------------------------------------------------------------

    #[test]
    fn test_conflict_analysis_passes_lbd_to_hook() {
        // Solve PHP(3,2) — the same UNSAT formula used in the external_branching tests.
        // A ConflictLbdRecordingHeuristic records all LBD values received via
        // on_conflict_var_with_lbd.  After solving, assert:
        //   1. at least one call was made (conflicts happened)
        //   2. all recorded LBD values are > 0 (no degenerate LBD-0 passed through)
        use crate::solver::heuristic::BranchingHeuristic;
        use crate::{Solver, SolverConfig, SolverResult};
        use std::sync::{Arc, Mutex};

        struct ConflictLbdRecordingHeuristic {
            lbd_values: Arc<Mutex<Vec<u32>>>,
        }

        impl BranchingHeuristic for ConflictLbdRecordingHeuristic {
            fn select(&mut self, _candidates: &[Var], _scores: &[f64]) -> Option<Var> {
                None // always defer — VSIDS drives the solve
            }

            fn on_conflict_var_with_lbd(&mut self, _var: Var, lbd: u32) {
                self.lbd_values
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(lbd);
            }
        }

        let lbd_values: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let heuristic = Arc::new(Mutex::new(ConflictLbdRecordingHeuristic {
            lbd_values: Arc::clone(&lbd_values),
        }));

        let config = SolverConfig {
            external_branching: Some(heuristic),
            ..SolverConfig::default()
        };
        let mut solver = Solver::with_config(config);

        // PHP(3,2): 6 variables
        for _ in 0..6 {
            solver.new_var();
        }
        // Each pigeon must be in at least one hole
        solver.add_clause_dimacs(&[1, 2]);
        solver.add_clause_dimacs(&[3, 4]);
        solver.add_clause_dimacs(&[5, 6]);
        // At most one pigeon per hole
        solver.add_clause_dimacs(&[-1, -3]);
        solver.add_clause_dimacs(&[-1, -5]);
        solver.add_clause_dimacs(&[-3, -5]);
        solver.add_clause_dimacs(&[-2, -4]);
        solver.add_clause_dimacs(&[-2, -6]);
        solver.add_clause_dimacs(&[-4, -6]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Unsat, "PHP(3,2) must be UNSAT");

        let values = lbd_values.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            !values.is_empty(),
            "on_conflict_var_with_lbd must have been called at least once"
        );
        for &lbd in values.iter() {
            assert!(
                lbd > 0,
                "LBD passed to hook must be > 0 (got {lbd}); level-0 vars should be excluded"
            );
        }
    }

    #[test]
    fn test_lbd_matches_learned_clause_glue() {
        // The LBD passed to the hook must be the glue score of the ACTUAL learned
        // (1-UIP) clause — i.e. the distinct decision-level count of `self.learnt` —
        // NOT the distinct-level count of the larger `vars_to_bump` union.
        //
        // We solve a crafted UNSAT instance with clause deletion effectively disabled
        // so every learned clause persists. The hook records the set of LBD values it
        // receives; the solver stores `clause.lbd` (computed independently in
        // learn.rs::compute_lbd from the same final clause). Since a 1-UIP learned
        // clause never contains level-0 literals, the two definitions coincide, so the
        // set of hook LBDs must be a SUBSET of the stored learned-clause LBD set
        // (plus 1 for unit learned clauses, whose single literal sits at the current
        // decision level). The old `vars_to_bump` proxy would routinely report values
        // ABSENT from any stored clause's LBD, so a subset relation is decisive.
        use crate::solver::heuristic::BranchingHeuristic;
        use crate::{Solver, SolverConfig, SolverResult};
        use std::collections::BTreeSet;
        use std::sync::{Arc, Mutex};

        struct SetRecordingHeuristic {
            lbd_set: Arc<Mutex<BTreeSet<u32>>>,
        }

        impl BranchingHeuristic for SetRecordingHeuristic {
            fn select(&mut self, _candidates: &[Var], _scores: &[f64]) -> Option<Var> {
                None
            }

            fn on_conflict_var_with_lbd(&mut self, _var: Var, lbd: u32) {
                self.lbd_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(lbd);
            }
        }

        let lbd_set: Arc<Mutex<BTreeSet<u32>>> = Arc::new(Mutex::new(BTreeSet::new()));
        let heuristic = Arc::new(Mutex::new(SetRecordingHeuristic {
            lbd_set: Arc::clone(&lbd_set),
        }));

        let config = SolverConfig {
            external_branching: Some(heuristic),
            // Disable clause deletion so every learned clause survives for inspection.
            clause_deletion_threshold: usize::MAX,
            ..SolverConfig::default()
        };
        let mut solver = Solver::with_config(config);

        // PHP(3,2): 6 variables, UNSAT, produces multi-level conflicts.
        for _ in 0..6 {
            solver.new_var();
        }
        solver.add_clause_dimacs(&[1, 2]);
        solver.add_clause_dimacs(&[3, 4]);
        solver.add_clause_dimacs(&[5, 6]);
        solver.add_clause_dimacs(&[-1, -3]);
        solver.add_clause_dimacs(&[-1, -5]);
        solver.add_clause_dimacs(&[-3, -5]);
        solver.add_clause_dimacs(&[-2, -4]);
        solver.add_clause_dimacs(&[-2, -6]);
        solver.add_clause_dimacs(&[-4, -6]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Unsat, "PHP(3,2) must be UNSAT");

        // Gather the LBD of every surviving learned clause from the solver's
        // internal database (these fields are crate-visible). Unit learned clauses
        // (len == 1) keep the default lbd 0 but their hook LBD is 1, so we add 1 for
        // every unit clause to the allowed set.
        let mut stored_lbds: BTreeSet<u32> = BTreeSet::new();
        for &cid in &solver.learned_clause_ids {
            if let Some(clause) = solver.clauses.get(cid) {
                if clause.lits.len() == 1 {
                    stored_lbds.insert(1);
                } else {
                    stored_lbds.insert(clause.lbd);
                }
            }
        }

        let hook_lbds = lbd_set.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!hook_lbds.is_empty(), "hook must have received LBD values");

        // Decisive check: every LBD the hook reported is the glue score of a real
        // learned clause (subset relation). With the old vars_to_bump proxy this
        // would fail because vars_to_bump-derived LBDs exceed any stored clause's LBD.
        for &lbd in hook_lbds.iter() {
            assert!(
                stored_lbds.contains(&lbd),
                "hook LBD {lbd} must match the glue score of an actual learned clause; \
                 stored learned-clause LBDs = {stored_lbds:?}"
            );
        }
    }

    #[test]
    fn test_lbd_le_clause_size() {
        // The LBD of a learned clause can never exceed its literal count: each literal
        // contributes at most one distinct decision level. The fix computes LBD from
        // the actual learned clause, so this invariant must hold for the value handed
        // to the hook (unlike the old vars_to_bump proxy, which could exceed the clause
        // length). We verify it on every surviving learned clause and also confirm the
        // hook never reported an LBD larger than the largest learned clause.
        use crate::solver::heuristic::BranchingHeuristic;
        use crate::{Solver, SolverConfig, SolverResult};
        use std::sync::{Arc, Mutex};

        struct MaxRecordingHeuristic {
            max_lbd: Arc<Mutex<u32>>,
        }

        impl BranchingHeuristic for MaxRecordingHeuristic {
            fn select(&mut self, _candidates: &[Var], _scores: &[f64]) -> Option<Var> {
                None
            }

            fn on_conflict_var_with_lbd(&mut self, _var: Var, lbd: u32) {
                let mut m = self.max_lbd.lock().unwrap_or_else(|e| e.into_inner());
                if lbd > *m {
                    *m = lbd;
                }
            }
        }

        let max_lbd: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let heuristic = Arc::new(Mutex::new(MaxRecordingHeuristic {
            max_lbd: Arc::clone(&max_lbd),
        }));

        let config = SolverConfig {
            external_branching: Some(heuristic),
            clause_deletion_threshold: usize::MAX,
            ..SolverConfig::default()
        };
        let mut solver = Solver::with_config(config);

        // PHP(4,3): 12 variables, UNSAT, deeper search → larger learned clauses.
        for _ in 0..12 {
            solver.new_var();
        }
        // Each of 4 pigeons in at least one of 3 holes (pigeon p, hole h → var 3*(p-1)+h).
        solver.add_clause_dimacs(&[1, 2, 3]);
        solver.add_clause_dimacs(&[4, 5, 6]);
        solver.add_clause_dimacs(&[7, 8, 9]);
        solver.add_clause_dimacs(&[10, 11, 12]);
        // At most one pigeon per hole (for each hole h, no two pigeons share it).
        for hole in 0..3 {
            let h = hole + 1;
            let occupants = [h, h + 3, h + 6, h + 9];
            for i in 0..occupants.len() {
                for j in (i + 1)..occupants.len() {
                    solver.add_clause_dimacs(&[-occupants[i], -occupants[j]]);
                }
            }
        }

        let result = solver.solve();
        assert_eq!(result, SolverResult::Unsat, "PHP(4,3) must be UNSAT");

        // Invariant on every surviving learned clause: lbd <= number of literals.
        let mut max_learnt_len: usize = 0;
        for &cid in &solver.learned_clause_ids {
            if let Some(clause) = solver.clauses.get(cid) {
                let len = clause.lits.len();
                max_learnt_len = max_learnt_len.max(len);
                // Unit clauses store the default lbd 0; the invariant lbd <= len is
                // trivially satisfied. For len >= 2 the stored lbd was computed from
                // the clause literals and must not exceed the literal count.
                assert!(
                    clause.lbd as usize <= len,
                    "learned clause LBD {} exceeds its literal count {len}",
                    clause.lbd
                );
            }
        }

        let observed_max = *max_lbd.lock().unwrap_or_else(|e| e.into_inner());
        assert!(observed_max > 0, "hook must have received a positive LBD");
        assert!(
            observed_max as usize <= max_learnt_len,
            "max hook LBD {observed_max} must not exceed the largest learned clause length \
             {max_learnt_len} — proves LBD is computed from the learned clause, not vars_to_bump"
        );
    }

    // ---------------------------------------------------------------------------
    // Regression: conflict clause whose literals all sit BELOW the current
    // decision level (an on-the-fly / theory-lemma-style clause).
    // ---------------------------------------------------------------------------

    /// Root cause of the disjunctive-LIA wrong-UNSAT: `analyze` used to anchor
    /// its 1-UIP resolution at `trail.decision_level()`. When the conflict
    /// clause contains NO literal at that level (its highest literal is at a
    /// strictly lower level — as happens for theory reason/lemma clauses added
    /// mid-search), the pivot-level counter starts at 0, the trail walk
    /// underflows it, and the asserting literal comes out at or below the
    /// computed backtrack level. Backtracking then fails to unassign the
    /// asserting variable and the clause-learning step re-assigns it in place,
    /// corrupting the trail.
    ///
    /// We reconstruct exactly that state by hand and assert the fixed invariant:
    /// the asserting literal `learnt[0]` must live strictly above the backtrack
    /// level, so a subsequent backtrack unassigns it.
    #[test]
    fn test_analyze_conflict_below_current_level_is_asserting() {
        use crate::Solver;

        let mut solver = Solver::new();
        let v0 = solver.new_var();
        let v1 = solver.new_var();
        let v2 = solver.new_var();

        // Level 1: decide v0 = false.
        solver.trail.new_decision_level();
        solver.trail.assign_decision(Lit::neg(v0));

        // Level 1: propagate v1 = false with reason clause (¬v1 ∨ v0).
        // With v0 false, this clause forces ¬v1.
        let r1 = solver.clauses.add_learned([Lit::neg(v1), Lit::pos(v0)]);
        solver.trail.assign_propagation(Lit::neg(v1), r1);

        // Level 2: an UNRELATED decision v2 = true, lifting the trail's
        // decision level to 2 while the impending conflict lives entirely at
        // level 1.
        solver.trail.new_decision_level();
        solver.trail.assign_decision(Lit::pos(v2));

        assert_eq!(solver.trail.decision_level(), 2);

        // Conflict clause (v0 ∨ v1): both literals are FALSE (v0 = false,
        // v1 = false), so the clause is falsified. Its highest literal level is
        // 1 — strictly below the current decision level 2.
        let conflict = solver.clauses.add_learned([Lit::pos(v0), Lit::pos(v1)]);

        let (backtrack_level, learnt) = solver.analyze(conflict);

        assert!(!learnt.is_empty(), "learned clause must not be empty");

        let uip = learnt[0];
        let uip_level = solver.trail.level(uip.var());

        // The genuine conflict level is 1, so the asserting literal must be at
        // level 1 and the backtrack level must be strictly below it (0 here).
        assert_eq!(
            uip_level, 1,
            "asserting literal must sit at the true conflict level (1), not the \
             stale decision level (2)"
        );
        assert!(
            backtrack_level < uip_level,
            "backtrack level {backtrack_level} must be strictly below the asserting \
             literal's level {uip_level}; otherwise backtracking leaves the variable \
             assigned and clause learning corrupts the trail by re-assigning it"
        );

        // Every non-asserting literal must be unassigned or restorable at the
        // backtrack target (i.e. at a level <= backtrack_level).
        for &lit in learnt.iter().skip(1) {
            assert!(
                solver.trail.level(lit.var()) <= backtrack_level,
                "non-asserting literal at level {} exceeds backtrack level {backtrack_level}",
                solver.trail.level(lit.var())
            );
        }
    }

    /// End-to-end guard: a normal (current-level) conflict must be unaffected by
    /// the conflict-level anchoring — the asserting literal still sits at the
    /// current decision level and the backtrack level below it.
    #[test]
    fn test_analyze_normal_current_level_conflict_unaffected() {
        use crate::Solver;

        let mut solver = Solver::new();
        let v0 = solver.new_var();
        let v1 = solver.new_var();
        let v2 = solver.new_var();

        // Level 1: decide v0 = true.
        solver.trail.new_decision_level();
        solver.trail.assign_decision(Lit::pos(v0));

        // Level 2: decide v1 = true.
        solver.trail.new_decision_level();
        solver.trail.assign_decision(Lit::pos(v1));

        // Level 2: propagate v2 = true with reason (¬v1 ∨ v2).
        let r = solver.clauses.add_learned([Lit::neg(v1), Lit::pos(v2)]);
        solver.trail.assign_propagation(Lit::pos(v2), r);

        // Conflict clause (¬v0 ∨ ¬v1 ∨ ¬v2): all three literals false at their
        // levels; the highest is v2/v1 at level 2 == current decision level.
        let conflict = solver
            .clauses
            .add_learned([Lit::neg(v0), Lit::neg(v1), Lit::neg(v2)]);

        let (backtrack_level, learnt) = solver.analyze(conflict);
        assert!(!learnt.is_empty());
        let uip_level = solver.trail.level(learnt[0].var());
        assert_eq!(uip_level, 2, "asserting literal at current level");
        assert!(
            backtrack_level < uip_level,
            "backtrack level {backtrack_level} must be below asserting level {uip_level}"
        );
    }

    // ---------------------------------------------------------------------------
    // Regression: theory conflict clause containing an UNASSIGNED literal.
    //
    // The MBQI / quantifier-instantiation path builds its conflict clause from a
    // per-atom polarity map that is not pruned on every SAT backtrack (notably a
    // restart). It can therefore hand `analyze_theory_conflict` a "conflict" whose
    // clause still lists a variable that has since been unassigned — its
    // `VarInfo.level` left stale at the level it last held. Two OxiZ z3-parity
    // reproducers (`injective_unsat.smt2`, `nested_quantifiers.smt2`) drove
    // exactly this and panicked at the theory trail-consistency `debug_assert`
    // ("asserting literal must be above the backtrack level"): the stale level
    // became a bogus `current_level`, the 1-UIP counter was charged for a literal
    // absent from the trail, and the asserting literal was duplicated at the
    // backtrack level (`backtrack_level == uip_level`). In release the same trail
    // corruption produced a wrong top-level UNSAT on a SAT instance.
    //
    // The fix recognizes such a clause as an *asserting theory lemma* (unit under
    // the current assignment) and propagates its one open literal, keeping the
    // whole (valid) lemma. These tests reconstruct the exact trail shape by hand.
    // ---------------------------------------------------------------------------

    #[test]
    fn test_theory_conflict_stale_unassigned_literal_is_asserting() {
        use crate::Solver;

        let mut solver = Solver::new();
        let v0 = solver.new_var(); // becomes a false literal @ level 3
        let v1 = solver.new_var(); // the stale, now-unassigned literal
        let v2 = solver.new_var(); // becomes a false literal @ level 1

        // Level 1: decide ¬v2, so the positive literal v2 is FALSE at level 1.
        solver.trail.new_decision_level();
        solver.trail.assign_decision(Lit::neg(v2));

        // Levels 2 and 3.
        solver.trail.new_decision_level(); // level 2
        solver.trail.new_decision_level(); // level 3
        // Propagate ¬v0 at level 3, so the positive literal v0 is FALSE at level 3.
        let r = solver.clauses.add_learned([Lit::neg(v0), Lit::pos(v2)]);
        solver.trail.assign_propagation(Lit::neg(v0), r);

        // Levels 4 and 5: assign v1, then backtrack it away so it becomes
        // UNASSIGNED while `VarInfo.level` stays stale at 5.
        solver.trail.new_decision_level(); // level 4
        solver.trail.new_decision_level(); // level 5
        solver.trail.assign_decision(Lit::pos(v1));
        assert_eq!(solver.trail.decision_level(), 5);

        solver.trail.backtrack_to(3);
        assert_eq!(solver.trail.decision_level(), 3);
        assert!(!solver.trail.is_assigned(v1), "v1 must be unassigned");
        assert_eq!(
            solver.trail.level(v1),
            5,
            "v1's level must remain stale at 5 after backtrack"
        );

        // Theory conflict clause: all three are meant to be the (false) clause
        // literals, but v1 is now unassigned. Pre-fix this panicked / corrupted.
        let conflict_lits = [Lit::pos(v0), Lit::pos(v1), Lit::pos(v2)];
        let (backtrack_level, learnt) = solver.analyze_theory_conflict(&conflict_lits);

        assert!(!learnt.is_empty(), "learned clause must not be empty");

        // The asserting literal (index 0) is the unassigned one — the clause is
        // unit and will propagate it — so backtracking never leaves it assigned.
        assert_eq!(
            learnt[0].var(),
            v1,
            "the unassigned literal must be the asserting (index-0) literal"
        );
        assert!(
            !solver.trail.is_assigned(learnt[0].var()),
            "the asserting literal must be unassigned"
        );

        // Backtrack level is the max level among the *assigned* (false) literals —
        // never the unassigned literal's stale level 5.
        assert_eq!(
            backtrack_level, 3,
            "backtrack level must be the max assigned (false) level, not the stale 5"
        );

        // Soundness: the full theory lemma is preserved — every original literal is
        // present exactly once, none dropped (dropping would strengthen the clause
        // and could be unsound).
        let mut got: Vec<Lit> = learnt.to_vec();
        got.sort_by_key(|l| l.code());
        let mut want = vec![Lit::pos(v0), Lit::pos(v1), Lit::pos(v2)];
        want.sort_by_key(|l| l.code());
        assert_eq!(
            got, want,
            "learned clause must be the full deduplicated lemma"
        );

        // Every non-asserting literal is assigned and at a level <= backtrack level,
        // so after backtracking the clause stays unit on the asserting literal.
        for &lit in learnt.iter().skip(1) {
            assert!(
                solver.trail.is_assigned(lit.var()),
                "non-asserting literal {lit:?} must be assigned"
            );
            assert!(
                solver.trail.level(lit.var()) <= backtrack_level,
                "non-asserting literal level {} exceeds backtrack level {backtrack_level}",
                solver.trail.level(lit.var())
            );
        }
    }

    #[test]
    fn test_theory_conflict_two_unassigned_literals_no_panic() {
        // Defensive: a lemma with more than one open literal is not unit, but the
        // handler must still produce a valid, non-corrupting result (an unassigned
        // asserting literal, a backtrack level drawn only from assigned literals).
        use crate::Solver;

        let mut solver = Solver::new();
        let v0 = solver.new_var(); // false @ level 2
        let v1 = solver.new_var(); // unassigned, stale level 4
        let v2 = solver.new_var(); // unassigned, stale level 3

        // Level 1 (unused), level 2: v0 false via a decision on ¬v0.
        solver.trail.new_decision_level(); // 1
        solver.trail.new_decision_level(); // 2
        solver.trail.assign_decision(Lit::neg(v0));

        // Levels 3, 4: assign v2 then v1, then backtrack both away (stale levels).
        solver.trail.new_decision_level(); // 3
        solver.trail.assign_decision(Lit::pos(v2));
        solver.trail.new_decision_level(); // 4
        solver.trail.assign_decision(Lit::pos(v1));

        solver.trail.backtrack_to(2);
        assert_eq!(solver.trail.decision_level(), 2);
        assert!(!solver.trail.is_assigned(v1));
        assert!(!solver.trail.is_assigned(v2));

        let conflict_lits = [Lit::pos(v0), Lit::pos(v1), Lit::pos(v2)];
        let (backtrack_level, learnt) = solver.analyze_theory_conflict(&conflict_lits);

        // No panic; the asserting literal is one of the unassigned vars; the
        // backtrack level is the only assigned literal's level (2), never a stale
        // level (3 or 4).
        assert!(!learnt.is_empty());
        assert!(
            !solver.trail.is_assigned(learnt[0].var()),
            "asserting literal must be unassigned"
        );
        assert_eq!(
            backtrack_level, 2,
            "backtrack level must come from the single assigned literal (level 2)"
        );
        // The full lemma is preserved (all three vars, once each).
        let vars: std::collections::BTreeSet<u32> =
            learnt.iter().map(|l| l.var().index() as u32).collect();
        assert_eq!(vars.len(), 3, "all three distinct vars must be present");
    }

    #[test]
    fn test_theory_conflict_all_false_still_uses_1uip() {
        // Regression guard: a genuine, fully-falsified theory conflict must keep
        // going through the 1-UIP path (asserting literal strictly above the
        // backtrack level), unaffected by the unassigned-literal branch.
        use crate::Solver;

        let mut solver = Solver::new();
        let v0 = solver.new_var();
        let v1 = solver.new_var();
        let v2 = solver.new_var();

        // Level 1: decide v0 = true.
        solver.trail.new_decision_level();
        solver.trail.assign_decision(Lit::pos(v0));
        // Level 2: decide v1 = true.
        solver.trail.new_decision_level();
        solver.trail.assign_decision(Lit::pos(v1));
        // Level 2: propagate v2 = true with reason (¬v1 ∨ v2).
        let r = solver.clauses.add_learned([Lit::neg(v1), Lit::pos(v2)]);
        solver.trail.assign_propagation(Lit::pos(v2), r);

        // Conflict clause (¬v0 ∨ ¬v1 ∨ ¬v2): every literal is FALSE (assigned).
        let conflict_lits = [Lit::neg(v0), Lit::neg(v1), Lit::neg(v2)];
        let (backtrack_level, learnt) = solver.analyze_theory_conflict(&conflict_lits);

        assert!(!learnt.is_empty());
        let uip_level = solver.trail.level(learnt[0].var());
        assert!(
            solver.trail.is_assigned(learnt[0].var()),
            "for an all-false conflict the 1-UIP asserting literal is assigned"
        );
        assert!(
            uip_level > backtrack_level,
            "1-UIP asserting literal level {uip_level} must be strictly above the \
             backtrack level {backtrack_level}"
        );
    }
}
