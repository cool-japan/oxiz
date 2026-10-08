//! [`Solver::add_clause_at_root`]: a clause installed at assertion level 0,
//! whatever assertion levels are open and wherever the search stands.
//!
//! # What "at the root" means, and who needs it
//!
//! [`Solver::add_clause`] registers a clause in the **innermost** open
//! assertion level, so the matching [`Solver::pop`] deletes it.  That is right
//! for an *assertion* and wrong for a *definition*: a Tseitin clause that
//! defines a fresh variable as a function of others — a gate of a bit-blasted
//! circuit, the truth variable of a comparison — is a conservative extension
//! of every clause set (any model of the rest extends to it by evaluating the
//! gate), so it is valid at every assertion level and nothing is gained by
//! retracting it.  Something is lost: a caller that keeps a memo of what it
//! has defined (`oxiz-theories`' bit-vector solver keeps four) must either
//! retract the memo with the clauses — and pay a fresh variable for every
//! re-definition — or keep a memo whose defining clauses are gone, which is
//! the U-Z10 wrong `sat`.  A root clause removes the dilemma: it lives exactly
//! as long as the solver does (until [`Solver::reset`]).
//!
//! # The two places a root clause differs from `add_clause`
//!
//! * **Registration.**  The clause id goes to the level-0 list, which no
//!   `pop` visits.  A root *unit* has no clause to register — `add_clause`
//!   stores a unit only as a level-0 trail assignment, and `pop` truncates the
//!   trail to its `push`-time size — so a root unit is also remembered in
//!   `Solver::root_units` and re-asserted by every `pop`.
//! * **A clause that the current assignment falsifies is still installed.**
//!   `add_clause` answers an all-false clause by latching `trivially_unsat` and
//!   *not adding it*; that is fine when the clause dies with the scope that
//!   falsified it, and unsound for a clause that must outlive that scope.  So
//!   the clause is attached first, and only then is the conflict recorded: as
//!   a **permanent** contradiction (every `push`-time snapshot of the latch is
//!   set too) when every falsifying fact is permanent, and as an ordinary
//!   latch — which the matching `pop` restores — when any of them belongs to
//!   an open scope.
//!
//! # A clause added while the trail is above decision level 0
//!
//! It is handled the way a clause learned mid-search is: if the current
//! assignment makes it unit or conflicting, the search backjumps to the
//! highest level at which it is unit (for a conflicting clause: the second
//! highest level among its literals when one literal sits alone at the top,
//! one below the top when two or more do) and the implied literal is assigned
//! there with the clause as its reason.  Never is a clause attached with both
//! watches on literals that are already false and propagated, which is the
//! silent-falsification hazard `add_clause`'s own watch ranking exists for.

use super::*;

impl Solver {
    /// Add a clause at assertion level 0: it survives every [`Solver::pop`].
    ///
    /// Returns `false` exactly when the clause is falsified by the current
    /// level-0 facts (the instance is then `Unsat` for as long as those facts
    /// hold — permanently if they are permanent, until the `pop` that retracts
    /// one of them otherwise) or names a bounded-variable-eliminated variable
    /// ([`Solver::error`] reports that case, as for `add_clause`).  In every
    /// other case the clause is installed and `true` is returned.
    ///
    /// See the module documentation for when a clause belongs here rather than
    /// in the innermost assertion level.
    pub fn add_clause_at_root(&mut self, lits: impl IntoIterator<Item = Lit>) -> bool {
        let mut clause_lits: SmallVec<[Lit; 8]> = lits.into_iter().collect();
        for lit in &clause_lits {
            let var_idx = lit.var().index();
            if var_idx >= self.num_vars {
                self.ensure_vars(var_idx + 1);
            }
        }
        let mut resolved_lits: SmallVec<[Lit; 8]> = SmallVec::with_capacity(clause_lits.len());
        for &lit in &clause_lits {
            match self.resolve_reintroduced_literal(lit) {
                Some(resolved) => resolved_lits.push(resolved),
                None => return false,
            }
        }
        clause_lits = resolved_lits;
        clause_lits.sort_by_key(|l| l.code());
        clause_lits.dedup();
        let original_lrat_id = self.lrat_register_original();
        for i in 0..clause_lits.len() {
            for j in (i + 1)..clause_lits.len() {
                if clause_lits[i] == clause_lits[j].negate() {
                    return true;
                }
            }
        }
        match clause_lits.len() {
            0 => {
                self.latch_root_contradiction();
                self.lrat_mark_finalized_by_original_empty();
                false
            }
            1 => self.add_root_unit(clause_lits[0], original_lrat_id),
            _ => self.add_root_long(clause_lits, original_lrat_id),
        }
    }

    /// Latch `trivially_unsat` so that no `pop` clears it: the contradiction
    /// rests on permanent facts only.
    fn latch_root_contradiction(&mut self) {
        self.trivially_unsat = true;
        for latch in &mut self.assertion_trivially_unsat {
            *latch = true;
        }
    }

    /// Whether every literal of `lits` is false by a fact no `pop` retracts:
    /// assigned at decision level 0 and, when an assertion level is open,
    /// below the first open level's trail mark.
    fn falsified_permanently(&self, lits: &[Lit]) -> bool {
        let permanent_prefix = self
            .assertion_trail_sizes
            .get(1)
            .copied()
            .unwrap_or_else(|| self.trail.size());
        let prefix = &self.trail.assignments()[..permanent_prefix.min(self.trail.size())];
        lits.iter().all(|&lit| {
            self.trail.lit_value(lit).is_false()
                && self.trail.level(lit.var()) == 0
                && prefix.contains(&lit.negate())
        })
    }

    /// A root unit: assigned at level 0 now, and re-asserted by every `pop`.
    fn add_root_unit(&mut self, lit: Lit, lrat_id: Option<u64>) -> bool {
        // Kept even when the literal already holds: the fact that makes it
        // true now may belong to a scope a later `pop` retracts.
        self.root_units.push((lit, lrat_id));
        if self.trail.decision_level() > 0 {
            self.backtrack_to_root();
        }
        let value = self.trail.lit_value(lit);
        if value.is_true() {
            return true;
        }
        if value.is_false() {
            if self.falsified_permanently(&[lit]) {
                self.latch_root_contradiction();
                self.lrat_emit_empty_from(&[lit], lrat_id.unwrap_or(0));
            } else {
                // Falsified by a scoped fact: the `pop` that retracts it
                // restores the latch and re-asserts this unit.
                self.trivially_unsat = true;
            }
            return false;
        }
        self.trail.assign_decision(lit);
        if let Some(id) = lrat_id {
            self.lrat_set_unit_justification(lit.var(), id);
        }
        true
    }

    /// Move the search to the level at which `lits` — about to be installed —
    /// is unit or satisfiable, the way a clause learned mid-search is asserted
    /// (see the module documentation).  A no-op at decision level 0.
    fn backjump_for_root_clause(&mut self, lits: &[Lit]) {
        let (has_true, _, undefined) = self.scan_clause_for_attach(lits);
        if has_true || undefined.len() >= 2 {
            return;
        }
        let mut levels: Vec<u32> = lits
            .iter()
            .filter(|&&lit| self.trail.lit_value(lit).is_false())
            .map(|&lit| self.trail.level(lit.var()))
            .collect();
        levels.sort_unstable_by(|a, b| b.cmp(a));
        let target = if undefined.len() == 1 {
            levels.first().copied().unwrap_or(0)
        } else {
            let top = levels.first().copied().unwrap_or(0);
            match levels.get(1) {
                Some(&second) if second == top => top.saturating_sub(1),
                Some(&second) => second,
                None => 0,
            }
        };
        if target < self.trail.decision_level() {
            self.backtrack_with_phase_saving(target);
        }
    }

    /// A root clause of two or more literals; see the module documentation.
    fn add_root_long(&mut self, mut clause_lits: SmallVec<[Lit; 8]>, lrat_id: Option<u64>) -> bool {
        if self.trail.decision_level() > 0 {
            self.backjump_for_root_clause(&clause_lits);
        }
        let (has_true, max_false_level, undefined) = self.scan_clause_for_attach(&clause_lits);
        let conflicting = !has_true && undefined.is_empty();
        let permanent_conflict = conflicting && self.falsified_permanently(&clause_lits);

        // Watch the two literals that become false last (MiniSat's
        // attachClause invariant).  For a clause falsified at level 0 by facts
        // an open scope owns, prefer the literals latest on the trail: those
        // are the ones the `pop` unassigns, which leaves both watches on
        // unassigned literals afterwards.
        // The trail position is read only for a conflicting clause: it is the
        // one case where two level-0 literals must be told apart, and the scan
        // is linear in the trail.
        let position = |solver: &Self, lit: Lit| -> usize {
            if !conflicting {
                return 0;
            }
            solver
                .trail
                .assignments()
                .iter()
                .rposition(|&assigned| assigned.var() == lit.var())
                .unwrap_or(usize::MAX)
        };
        let rank = |solver: &Self, lit: Lit| -> (u8, u32, usize) {
            let (class, level) = solver.watch_rank(lit);
            (class, level, position(solver, lit))
        };
        let n = clause_lits.len();
        let mut best = 0;
        for i in 1..n {
            if rank(self, clause_lits[i]) > rank(self, clause_lits[best]) {
                best = i;
            }
        }
        clause_lits.swap(0, best);
        let mut second = 1;
        for i in 2..n {
            if rank(self, clause_lits[i]) > rank(self, clause_lits[second]) {
                second = i;
            }
        }
        clause_lits.swap(1, second);

        let clause_id = self.clauses.add_original(clause_lits.iter().copied());
        if let Some(id) = lrat_id {
            self.lrat_set_clause_id(clause_id, id);
        }
        if let Some(root_level) = self.assertion_clause_ids.first_mut() {
            root_level.push(clause_id);
        }
        let lit0 = clause_lits[0];
        let lit1 = clause_lits[1];
        if n == 2 {
            self.binary_graph.add(lit0.negate(), lit1, clause_id);
            self.binary_graph.add(lit1.negate(), lit0, clause_id);
        }
        self.watches
            .add(lit0.negate(), Watcher::new(clause_id, lit1));
        self.watches
            .add(lit1.negate(), Watcher::new(clause_id, lit0));

        if conflicting {
            if permanent_conflict {
                self.latch_root_contradiction();
                self.lrat_emit_empty_from(&clause_lits, lrat_id.unwrap_or(0));
            } else {
                self.trivially_unsat = true;
            }
            return false;
        }
        if !has_true && undefined.len() == 1 {
            // Unit at the highest level among its false literals, which the
            // backjump above made the current level (or below it).
            self.trail
                .assign_propagation_at(undefined[0], clause_id, max_false_level);
        }
        true
    }

    /// Re-assert every root unit after a `pop` truncated the trail below it.
    ///
    /// Called by [`Solver::pop`] with the trail at decision level 0.  A root
    /// unit the surviving facts falsify is a permanent contradiction.
    ///
    /// Under LRAT tracing the unit's justification is re-installed with the
    /// assignment: `pop` cleared it when the truncation unassigned the
    /// variable, and a hint chain that later reads the replayed fact needs
    /// the original clause id it rests on.
    pub(super) fn replay_root_units(&mut self) {
        let units = self.root_units.clone();
        for (lit, lrat_id) in units {
            let value = self.trail.lit_value(lit);
            if value.is_true() {
                continue;
            }
            if value.is_false() {
                // Permanent only if every fact that falsifies it is: a scope
                // still open around this `pop` may own the falsifying fact.
                if self.falsified_permanently(&[lit]) {
                    self.latch_root_contradiction();
                    if let Some(id) = lrat_id {
                        self.lrat_emit_empty_from(&[lit], id);
                    }
                } else {
                    self.trivially_unsat = true;
                }
                continue;
            }
            self.trail.assign_decision(lit);
            if let Some(id) = lrat_id {
                self.lrat_set_unit_justification(lit.var(), id);
            }
        }
    }
}
