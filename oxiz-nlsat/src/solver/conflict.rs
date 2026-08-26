//! Conflict analysis, backtracking, activity management, and restart/reduction
//! for the NLSAT solver.

use super::NlsatSolver;
use crate::assignment::Justification;
use crate::clause::ClauseId;
use crate::restart::RestartStrategy;
use crate::types::{BoolVar, Literal};
use oxiz_math::polynomial::Var;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{HashMap, HashSet};

/// Outcome of beginning resolution of one variable in
/// [`NlsatSolver::is_redundant_literal`].
enum EntryOutcome {
    /// The variable's redundancy is already known without exploring its
    /// reason clause (no trail entry, a non-propagation justification, a
    /// missing reason clause, or a back-edge).
    Resolved(bool),
    /// The variable is propagated by a live reason clause; these are its
    /// literals to explore.
    Explore(Vec<Literal>),
}

/// What [`NlsatSolver::analyze_conflict`] derived from a conflict.
pub(super) struct Learnt {
    /// The learnt clause. Every literal is false under the trail the conflict
    /// occurred on, and the clause is entailed by the clause database.
    pub literals: Vec<Literal>,
    /// The level to backjump to before asserting `literals[0]`.
    pub backtrack_level: u32,
    /// Whether `literals[0]` is the *only* literal at the conflict level, and
    /// therefore becomes unit after backjumping. When this is false the clause
    /// is still valid but propagates nothing, and the caller must not assume
    /// the search has made progress.
    pub asserting: bool,
    /// Whether deriving this clause dropped a literal that the arithmetic
    /// theory — not the clause set — had forced. An empty clause from such a
    /// derivation is not a refutation.
    pub theory_dependent: bool,
}

impl NlsatSolver {
    // ========== Conflict Analysis ==========

    /// Analyse a conflict and derive a learnt clause by first-UIP resolution.
    ///
    /// # The invariant that makes this sound
    ///
    /// At every point in the analysis the *resolvent* is exactly the clause
    /// `{ ¬t : t is the trail literal of some variable in `seen` }` — one
    /// literal per seen variable, each of them false under the current trail.
    /// Every step preserves that invariant, and every step is a resolution
    /// against a clause already in the database, so whatever the loop ends up
    /// emitting is *entailed by the clause set*. That is the whole soundness
    /// argument, and it is why literals are reconstructed from the trail here
    /// ([`Self::falsified_literal`]) rather than copied out of the clauses
    /// being resolved: a clause literal and its trail literal have opposite
    /// polarity, and deriving one from the other by hand is exactly the sort
    /// of step that silently emits a clause the solver is not entitled to.
    ///
    /// # What stops the resolution
    ///
    /// Resolution can only eliminate a variable that was *propagated by a
    /// clause* — that clause is the other premise. A variable assigned any
    /// other way (a decision, a unit, or the arithmetic theory) has no clausal
    /// reason to resolve against, so the loop stops there and keeps it. The
    /// invariant above still holds, so the clause is still entailed; it is
    /// simply not guaranteed to be *asserting*, which the caller is told via
    /// [`Learnt::asserting`] and must handle rather than assume.
    ///
    /// # Theory-justified assignments
    ///
    /// A literal the arithmetic layer forced is not a logical consequence of
    /// the clause set — it reflects the arithmetic values the search happens
    /// to have chosen, which are retractable. So while *keeping* such a
    /// literal is always safe, *dropping* one is not, and the one place this
    /// analysis drops literals is the standard level-0 elimination. Any such
    /// drop sets [`Learnt::theory_dependent`], and a caller must not read an
    /// empty clause from a theory-dependent derivation as a refutation.
    pub(super) fn analyze_conflict(&mut self, conflict_id: ClauseId) -> Option<Learnt> {
        self.seen.clear();

        // Track this clause for unsat core
        if self.extract_unsat_core {
            self.conflict_clauses.insert(conflict_id);
        }

        let clause_lits: Vec<Literal> = self.clauses.get(conflict_id)?.literals().to_vec();
        let current_level = self.assignment.level();
        let mut theory_dependent = false;
        // Variables of the resolvent that sit at the current decision level.
        let mut open_at_current_level = 0usize;

        let absorb =
            |solver: &mut Self, var: BoolVar, open: &mut usize, theory_dependent: &mut bool| {
                if !solver.seen.insert(var) {
                    return;
                }
                let level = solver.assignment.bool_level(var);
                if level == current_level {
                    *open += 1;
                    solver.bump_var_activity(var);
                } else if level > 0 {
                    solver.bump_var_activity(var);
                } else {
                    // Level 0: permanently false, so resolution against its unit
                    // reason removes it from the clause. Sound for a clausal
                    // reason; not for one the theory supplied.
                    solver.seen.remove(&var);
                    if solver.is_theory_justified(var) {
                        *theory_dependent = true;
                    }
                }
            };

        for &lit in &clause_lits {
            absorb(
                self,
                lit.var(),
                &mut open_at_current_level,
                &mut theory_dependent,
            );
        }

        // Resolve current-level variables away, newest first, until one is
        // left (the first unique implication point) or none can be resolved.
        let mut trail_idx = self.assignment.trail().len();
        while open_at_current_level > 1 {
            let Some((entry_idx, pivot)) = self.next_seen_on_trail(trail_idx, current_level) else {
                break;
            };
            trail_idx = entry_idx;
            // No clausal reason (a decision, a unit, or a theory
            // propagation): resolution has no second premise, so `pivot`
            // stays in the clause and the loop ends. Keeping a literal never
            // weakens the entailment invariant, so nothing needs flagging.
            let Some(reason_lits) = self.clausal_reason(pivot) else {
                break;
            };
            self.seen.remove(&pivot);
            open_at_current_level -= 1;
            for reason_lit in reason_lits {
                if reason_lit.var() == pivot {
                    continue;
                }
                absorb(
                    self,
                    reason_lit.var(),
                    &mut open_at_current_level,
                    &mut theory_dependent,
                );
            }
        }

        // Materialise the resolvent, current-level literals first so the
        // asserting one (when there is exactly one) lands at index 0.
        let seen_vars: Vec<BoolVar> = self.seen.iter().copied().collect();
        let mut current: Vec<Literal> = Vec::new();
        let mut earlier: Vec<Literal> = Vec::new();
        for var in seen_vars {
            let Some(literal) = self.falsified_literal(var) else {
                // A variable in the resolvent with no truth value cannot be
                // rendered as a false literal, so the derivation cannot be
                // completed; refusing beats emitting a clause of unknown
                // meaning.
                return None;
            };
            if self.assignment.bool_level(var) == current_level {
                current.push(literal);
            } else {
                earlier.push(literal);
            }
        }
        // Deterministic order: the resolvent is collected from a hash set, and
        // a learnt clause whose literal order varies run to run makes the
        // whole search non-reproducible.
        current.sort_unstable_by_key(|l| l.index());
        earlier.sort_unstable_by_key(|l| l.index());

        let asserting = current.len() == 1;
        let mut literals = current;
        literals.extend(earlier);

        let backtrack_level = literals
            .iter()
            .skip(1)
            .map(|l| self.assignment.bool_level(l.var()))
            .max()
            .unwrap_or(0);

        let literals = self.minimize_clause(literals);
        Some(Learnt {
            literals,
            backtrack_level,
            asserting,
            theory_dependent,
        })
    }

    /// The literal of `var` that is *false* under the current assignment —
    /// the form in which `var` belongs to a conflict clause.
    fn falsified_literal(&self, var: BoolVar) -> Option<Literal> {
        let value = self.assignment.bool_value(var);
        if value.is_true() {
            Some(Literal::negative(var))
        } else if value.is_false() {
            Some(Literal::positive(var))
        } else {
            None
        }
    }

    /// Whether `var`'s assignment came from the arithmetic theory.
    fn is_theory_justified(&self, var: BoolVar) -> bool {
        self.assignment
            .trail()
            .iter()
            .find(|e| e.literal.var() == var)
            .is_some_and(|e| matches!(e.justification, Justification::Theory))
    }

    /// The literals of `var`'s reason clause, or `None` when `var` was not
    /// propagated by a clause that is still in the database.
    fn clausal_reason(&mut self, var: BoolVar) -> Option<Vec<Literal>> {
        let reason_id = self
            .assignment
            .trail()
            .iter()
            .find(|e| e.literal.var() == var)
            .and_then(|e| match e.justification {
                Justification::Propagation(reason_id) => Some(reason_id),
                _ => None,
            })?;
        let literals = self.clauses.get(reason_id)?.literals().to_vec();
        if self.extract_unsat_core {
            self.conflict_clauses.insert(reason_id);
        }
        Some(literals)
    }

    /// Scan the trail backwards from `before` for the newest entry that is in
    /// `seen` *and* at `level`, returning its index and variable.
    ///
    /// The level filter is not redundant with the caller's bookkeeping even
    /// though the trail is ordered by level: relying on that ordering to keep
    /// the scan inside the conflict level is an invariant held somewhere else,
    /// and resolving against a literal from an earlier level would eliminate a
    /// variable the resolvent still needs. Checking it here makes the step
    /// correct on its own terms.
    fn next_seen_on_trail(&self, before: usize, level: u32) -> Option<(usize, BoolVar)> {
        let trail = self.assignment.trail();
        let mut idx = before.min(trail.len());
        while idx > 0 {
            idx -= 1;
            let var = trail.get(idx)?.literal.var();
            if self.seen.contains(&var) && self.assignment.bool_level(var) == level {
                return Some((idx, var));
            }
        }
        None
    }

    /// Minimize a learned clause by removing redundant literals.
    pub(super) fn minimize_clause(&self, mut clause: Vec<Literal>) -> Vec<Literal> {
        if clause.len() <= 1 {
            return clause;
        }

        // Keep track of which literals can be removed
        let mut to_remove = Vec::new();

        // Try to remove each literal (except the first asserting literal)
        for i in 1..clause.len() {
            let lit = clause[i];
            let var = lit.var();

            // Check if this literal is redundant
            if self.is_redundant_literal(var, &clause) {
                to_remove.push(i);
            }
        }

        // Remove redundant literals (in reverse order to maintain indices)
        for &idx in to_remove.iter().rev() {
            clause.remove(idx);
        }

        clause
    }

    /// Check if a literal at a variable is redundant in the clause.
    ///
    /// A variable is redundant if every *other* literal in its propagation
    /// reason clause is either decided at level 0 (always true, needs no
    /// justification), already present in `clause`, or itself (transitively)
    /// redundant by the same rule.
    ///
    /// # Why iterative
    ///
    /// This predicate is evaluated on every conflict on the default solving
    /// path (via [`Self::minimize_clause`]), so its cost and termination
    /// matter in the common case, not just at the margin. A direct recursive
    /// implementation with no memoization re-explores shared reason-clause
    /// dependencies from scratch every time they are reached (worst case
    /// exponential in the depth of the implication graph), and has no bound
    /// on recursion depth at all -- a long propagation chain overflows the
    /// native stack, and a reason-clause cycle (never expected from a sound
    /// trail, but not something this function could previously detect
    /// either) would recurse forever. This walks the dependency graph with
    /// an explicit worklist instead, memoizing each variable's resolved
    /// verdict and tracking which variables are still being explored
    /// (`on_stack`) to detect a back-edge.
    ///
    /// # Soundness of memoizing
    ///
    /// For a fixed `clause` and a fixed assignment/clause-database snapshot
    /// (both left untouched for the entire call, exactly as before), this
    /// function is a pure function of the variable being asked about: the
    /// trail, clauses, and `clause` itself never change during the walk, so
    /// caching a variable's verdict cannot change what that verdict *is* --
    /// it only avoids recomputing it. A back-edge is resolved as *not*
    /// redundant, never as redundant: clause minimization is
    /// soundness-sensitive, and treating an unresolved dependency as
    /// redundant could drop a literal the clause actually needs, producing
    /// an unsound (too-strong) learned clause. Treating it as not-redundant
    /// only costs minimization quality, never soundness.
    pub(super) fn is_redundant_literal(&self, var: BoolVar, clause: &[Literal]) -> bool {
        struct Frame {
            var: BoolVar,
            reason_lits: Vec<Literal>,
            next: usize,
        }

        let mut memo: HashMap<BoolVar, bool> = HashMap::new();
        let mut on_stack: HashSet<BoolVar> = HashSet::new();

        // `top` is the frame currently being resolved, owned directly rather
        // than peeked from `stack` -- `stack` holds only the *suspended*
        // ancestors waiting for `top` (or one of its descendants) to finish.
        let mut top = match self.redundant_entry(var, &mut on_stack) {
            EntryOutcome::Resolved(v) => return v,
            EntryOutcome::Explore(reason_lits) => Frame {
                var,
                reason_lits,
                next: 0,
            },
        };
        let mut stack: Vec<Frame> = Vec::new();

        loop {
            let Some(&reason_lit) = top.reason_lits.get(top.next) else {
                // Every reason literal is accounted for: `top.var` is
                // redundant. Resume whichever frame is waiting below it, or
                // return directly if none is -- the empty case is not a
                // failure, it is the answer for the original `var`.
                memo.insert(top.var, true);
                on_stack.remove(&top.var);
                top = match stack.pop() {
                    Some(parent) => parent,
                    None => return true,
                };
                continue;
            };
            top.next += 1;

            if reason_lit.var() == top.var {
                continue; // Skip the propagated literal itself.
            }
            let reason_var = reason_lit.var();
            if self.assignment.bool_level(reason_var) == 0 && !self.is_theory_justified(reason_var)
            {
                // A level-0 literal is permanently false, so a reason that
                // cites it needs no further justification -- provided the
                // assignment that fixed it is a logical consequence. One the
                // arithmetic theory forced is not (it reflects retractable
                // arithmetic choices), so it falls through to the ordinary
                // redundancy test below, which resolves a theory-justified
                // variable as *not* redundant. See `analyze_conflict`'s
                // treatment of the same distinction.
                continue;
            }
            if clause.iter().any(|&cl| cl.var() == reason_var) {
                continue; // Already explicit in the clause being minimized.
            }

            let dependency_redundant = if let Some(&cached) = memo.get(&reason_var) {
                cached
            } else {
                match self.redundant_entry(reason_var, &mut on_stack) {
                    EntryOutcome::Resolved(v) => {
                        memo.insert(reason_var, v);
                        v
                    }
                    EntryOutcome::Explore(reason_lits) => {
                        // Suspend `top` and descend into `reason_var`'s own
                        // reason clause first.
                        stack.push(top);
                        top = Frame {
                            var: reason_var,
                            reason_lits,
                            next: 0,
                        };
                        continue;
                    }
                }
            };

            if dependency_redundant {
                continue; // This dependency checked out; keep resuming `top`.
            }

            // `reason_var` could not be justified, so `top.var` is not
            // redundant either -- mirroring a recursive
            // `if !is_redundant_literal(...) { return false; }`. Every frame
            // still waiting below `top` made that exact same recursive call
            // about `top.var` (or a transitive dependency of it) and so
            // fails the same way in turn: unwind the whole stack rather than
            // resuming any of it.
            memo.insert(top.var, false);
            on_stack.remove(&top.var);
            loop {
                let Some(parent) = stack.pop() else {
                    return false;
                };
                memo.insert(parent.var, false);
                on_stack.remove(&parent.var);
            }
        }
    }

    /// Begin resolving `var` for [`Self::is_redundant_literal`]: resolve it
    /// immediately when possible (no trail entry, a non-propagation
    /// justification, a missing reason clause, or a back-edge to a variable
    /// already being explored), otherwise stake out `on_stack` and hand back
    /// the reason literals to explore.
    fn redundant_entry(&self, var: BoolVar, on_stack: &mut HashSet<BoolVar>) -> EntryOutcome {
        let trail = self.assignment.trail();
        let Some(entry) = trail.iter().find(|e| e.literal.var() == var) else {
            return EntryOutcome::Resolved(false);
        };
        match &entry.justification {
            Justification::Propagation(reason_id) => match self.clauses.get(*reason_id) {
                Some(reason_clause) => {
                    if !on_stack.insert(var) {
                        // Back-edge: `var`'s own resolution is already in
                        // progress higher up this path. A sound trail's
                        // propagation reasons are acyclic by construction
                        // (a reason clause only cites literals assigned
                        // strictly earlier), so this guards against an
                        // inconsistent trail rather than a case expected in
                        // normal operation. Resolve conservatively as "not
                        // redundant" -- see the soundness note on
                        // `is_redundant_literal`.
                        return EntryOutcome::Resolved(false);
                    }
                    EntryOutcome::Explore(reason_clause.literals().to_vec())
                }
                None => EntryOutcome::Resolved(false),
            },
            Justification::Decision | Justification::Unit | Justification::Theory => {
                // Cannot minimize past a decision, unit, or theory literal.
                EntryOutcome::Resolved(false)
            }
        }
    }

    // ========== Backtracking ==========

    /// Backtrack to a given level.
    pub(super) fn backtrack(&mut self, level: u32) {
        // Clear propagation queue
        self.propagation_queue.clear();
        self.conflict_clause = None;

        // Pop assignment levels
        let _unassigned = self.assignment.pop_level(level);

        // Reset arithmetic assignments above this level
        // (Simplified: reset all arithmetic assignments)
        for var in 0..self.num_arith_vars {
            self.assignment.unset_arith(var);
            self.assignment.reset_feasible(var);
        }

        // Every region on the witness ledger was computed against the boolean
        // assignment this backtrack just discarded, so none of them describes
        // a currently-valid feasible set any more; a stale one could offer a
        // point that violates whatever the backjump assigns differently this
        // time. `Self::next_arith_var` picks the freed variables again and
        // the ledger repopulates from scratch.
        self.arith_witnesses.forget_all();

        // Clear evaluation cache
        self.eval_cache.clear();
    }

    // ========== Activity Management ==========

    /// Bump the activity of a variable.
    pub(super) fn bump_var_activity(&mut self, var: BoolVar) {
        if (var as usize) >= self.var_activity.len() {
            self.var_activity.resize(var as usize + 1, 0.0);
        }

        self.var_activity[var as usize] += self.var_activity_inc;

        // Rescale if too large
        if self.var_activity[var as usize] > 1e100 {
            for a in &mut self.var_activity {
                *a *= 1e-100;
            }
            self.var_activity_inc *= 1e-100;
        }
    }

    /// Bump the activity of an arithmetic variable.
    pub(super) fn bump_arith_activity(&mut self, var: Var) {
        if (var as usize) >= self.arith_activity.len() {
            self.arith_activity.resize(var as usize + 1, 0.0);
        }

        self.arith_activity[var as usize] += self.arith_activity_inc;

        // Rescale if too large
        if self.arith_activity[var as usize] > 1e100 {
            for a in &mut self.arith_activity {
                *a *= 1e-100;
            }
            self.arith_activity_inc *= 1e-100;
        }
    }

    /// Decay all activities.
    pub(super) fn decay_activities(&mut self) {
        self.var_activity_inc *= 1.0 / self.var_activity_decay;
        self.arith_activity_inc *= 1.0 / self.arith_activity_decay;
        self.clauses.decay_activities();
    }

    // ========== Restart and Reduction ==========

    /// Compute the Literal Block Distance (LBD) of a clause.
    ///
    /// LBD is the number of distinct decision levels in the clause.
    /// Lower LBD indicates a more "glue" clause.
    pub(super) fn compute_lbd(&self, clause_lits: &[Literal]) -> u32 {
        let mut levels = HashSet::new();
        for &lit in clause_lits {
            let level = self.assignment.bool_level(lit.var());
            if level > 0 {
                levels.insert(level);
            }
        }
        levels.len() as u32
    }

    /// Maybe perform a restart using the restart manager.
    pub(super) fn maybe_restart(&mut self) {
        // Use restart manager to determine if we should restart
        let should_restart = if matches!(
            self.config.restart_strategy,
            RestartStrategy::Glucose { .. }
        ) {
            self.restart_manager
                .should_restart(Some(self.recent_avg_lbd))
        } else {
            self.restart_manager.should_restart(None)
        };

        if should_restart && self.assignment.level() > 0 {
            self.stats.restarts += 1;
            self.backtrack(0);
            self.restart_manager.restart();
        }
    }

    /// Reduce learned clauses.
    pub(super) fn reduce_learned(&mut self) {
        let removed = self
            .clauses
            .reduce_learned(self.config.learned_keep_fraction);
        self.stats.clause_deletions += removed.len() as u64;
    }

    /// Perform dynamic variable reordering based on activity scores.
    pub(super) fn dynamic_reorder(&mut self) {
        if !self.config.dynamic_reordering {
            return;
        }

        // Can only reorder unassigned variables
        let mut unassigned_vars: Vec<(Var, f64)> = (0..self.num_arith_vars)
            .filter(|&var| !self.assignment.is_arith_assigned(var))
            .map(|var| {
                let activity = self
                    .arith_activity
                    .get(var as usize)
                    .copied()
                    .unwrap_or(0.0);
                (var, activity)
            })
            .collect();

        // Sort by activity (highest first)
        unassigned_vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));

        // Rebuild var_order: assigned variables first (in current order), then by activity
        let assigned_vars: Vec<Var> = (0..self.num_arith_vars)
            .filter(|&var| self.assignment.is_arith_assigned(var))
            .collect();

        self.var_order.clear();
        self.var_order.extend(assigned_vars);
        self.var_order
            .extend(unassigned_vars.iter().map(|(var, _)| *var));

        self.stats.reorderings += 1;
    }

    // ========== Helper Methods ==========

    /// Check if the formula is completely assigned.
    pub(super) fn is_complete(&self) -> bool {
        // All boolean variables assigned
        for var in 0..self.num_bool_vars {
            if !self.assignment.is_bool_assigned(var) {
                return false;
            }
        }

        // All arithmetic variables assigned
        for var in 0..self.num_arith_vars {
            if !self.assignment.is_arith_assigned(var) {
                return false;
            }
        }

        true
    }

    /// Generate a random number in [0, 1).
    pub(super) fn random(&mut self) -> f64 {
        self.random_int() as f64 / u64::MAX as f64
    }

    /// Generate a random u64.
    pub(super) fn random_int(&mut self) -> u64 {
        // Simple LCG
        self.random_state = self
            .random_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.random_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Directly register a propagation-justified trail entry for `var`,
    /// bypassing the solver's real unit-propagation machinery so tests can
    /// construct specific (including deliberately cyclic) reason-clause
    /// shapes that a real run of the solver would not produce.
    fn assign_propagated(solver: &mut NlsatSolver, var: BoolVar, reason: ClauseId) {
        solver
            .assignment
            .assign(Literal::positive(var), Justification::Propagation(reason));
    }

    /// Add a clause directly to the clause database, bypassing
    /// `add_clause`'s tautology/dedup/unit-assignment side effects, purely
    /// for use as an `is_redundant_literal` reason clause.
    fn add_reason_clause(solver: &mut NlsatSolver, literals: Vec<Literal>) -> ClauseId {
        solver.clauses.add(literals, 0, false)
    }

    // -----------------------------------------------------------------------
    // First-UIP conflict analysis (independent reimplementation of upstream
    // PR #31's sound-learning work).
    //
    // The property under test in all of these is the one the whole search
    // rests on: the clause `analyze_conflict` hands back must be *entailed*.
    // Its observable signature is that every literal in it is false under the
    // trail the conflict occurred on -- an entailed clause derived by
    // resolution from an all-false conflict clause cannot contain a true
    // literal. `assert_falsified_by_trail` checks exactly that, and the
    // exact-clause assertions pin the specific derivations besides.
    // -----------------------------------------------------------------------

    /// Every literal of a derived clause must be false under the trail.
    fn assert_falsified_by_trail(solver: &NlsatSolver, learnt: &[Literal]) {
        for &lit in learnt {
            assert!(
                solver.assignment.lit_value(lit).is_false(),
                "a resolvent literal must be false under the trail it was derived on"
            );
        }
    }

    /// Set up: level 1 decides `p`, which propagates `a`; level 2 decides `d`,
    /// which propagates `b`. Returns `(a, b, p, d, conflict_clause)` for the
    /// conflict clause `(~a | ~b)`.
    fn two_level_conflict(
        solver: &mut NlsatSolver,
    ) -> (BoolVar, BoolVar, BoolVar, BoolVar, ClauseId) {
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let p = solver.new_bool_var();
        let d = solver.new_bool_var();

        solver.assignment.push_level();
        solver
            .assignment
            .assign(Literal::positive(p), Justification::Decision);
        let reason_a = add_reason_clause(solver, vec![Literal::positive(a), Literal::negative(p)]);
        solver
            .assignment
            .assign(Literal::positive(a), Justification::Propagation(reason_a));

        solver.assignment.push_level();
        solver
            .assignment
            .assign(Literal::positive(d), Justification::Decision);
        let reason_b = add_reason_clause(solver, vec![Literal::positive(b), Literal::negative(d)]);
        solver
            .assignment
            .assign(Literal::positive(b), Justification::Propagation(reason_b));

        let conflict = add_reason_clause(solver, vec![Literal::negative(a), Literal::negative(b)]);
        (a, b, p, d, conflict)
    }

    /// The headline soundness regression. `(~a | ~b)` conflicts with a trail
    /// that made both true; `a` sits at level 1 and `b` at level 2, so the
    /// first-UIP clause is `(~b | ~a)` -- both literals false, backjumping to
    /// level 1 where it becomes unit on `~b`.
    ///
    /// The defect this pins: the lower-level literal used to be emitted
    /// *negated* (`a` instead of `~a`), producing a clause that is true under
    /// the trail and, worse, not entailed by the clause database at all. A
    /// solver that learns non-entailed clauses can refute a satisfiable
    /// formula.
    #[test]
    fn test_pr31_conflict_analysis_keeps_lower_level_literal_polarity() {
        let mut solver = NlsatSolver::new();
        let (a, b, _p, _d, conflict) = two_level_conflict(&mut solver);

        let analysis = solver
            .analyze_conflict(conflict)
            .expect("a fully clause-justified conflict must analyse");

        assert!(analysis.asserting, "one literal sits at the conflict level");
        assert_eq!(analysis.backtrack_level, 1, "`~a` is the level-1 literal");
        assert_eq!(
            analysis.literals,
            vec![Literal::negative(b), Literal::negative(a)],
            "first-UIP resolvent of (~a | ~b) is itself, asserting literal first"
        );
        assert_falsified_by_trail(&solver, &analysis.literals);
        assert!(!analysis.theory_dependent);
    }

    /// When every conflict literal is at the conflict level, resolution runs
    /// all the way back to the decision, which is the unique implication
    /// point.
    #[test]
    fn test_pr31_conflict_analysis_resolves_back_to_the_decision() {
        let mut solver = NlsatSolver::new();
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let d = solver.new_bool_var();

        solver.assignment.push_level();
        solver
            .assignment
            .assign(Literal::positive(d), Justification::Decision);
        let reason_a = add_reason_clause(
            &mut solver,
            vec![Literal::positive(a), Literal::negative(d)],
        );
        solver
            .assignment
            .assign(Literal::positive(a), Justification::Propagation(reason_a));
        let reason_b = add_reason_clause(
            &mut solver,
            vec![Literal::positive(b), Literal::negative(d)],
        );
        solver
            .assignment
            .assign(Literal::positive(b), Justification::Propagation(reason_b));
        let conflict = add_reason_clause(
            &mut solver,
            vec![Literal::negative(a), Literal::negative(b)],
        );

        let analysis = solver
            .analyze_conflict(conflict)
            .expect("a fully clause-justified conflict must analyse");
        assert_eq!(
            analysis.literals,
            vec![Literal::negative(d)],
            "both branches trace back to the decision `d`"
        );
        assert!(analysis.asserting);
        assert_eq!(analysis.backtrack_level, 0);
        assert_falsified_by_trail(&solver, &analysis.literals);
    }

    /// A literal the arithmetic theory forced has no reason *clause*, so
    /// resolution cannot eliminate it. It must stay in the learnt clause --
    /// dropping it (which the previous implementation did, silently) produces
    /// a clause that does not follow from anything.
    #[test]
    fn test_pr31_conflict_analysis_keeps_theory_literal_instead_of_dropping_it() {
        let mut solver = NlsatSolver::new();
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let d = solver.new_bool_var();

        solver.assignment.push_level();
        solver
            .assignment
            .assign(Literal::positive(d), Justification::Decision);
        // `a` and `b` both forced by the theory at the conflict level: neither
        // can be resolved away.
        solver
            .assignment
            .assign(Literal::positive(a), Justification::Theory);
        solver
            .assignment
            .assign(Literal::positive(b), Justification::Theory);
        let conflict = add_reason_clause(
            &mut solver,
            vec![Literal::negative(a), Literal::negative(b)],
        );

        let analysis = solver
            .analyze_conflict(conflict)
            .expect("analysis completes even when nothing can be resolved");
        let mut literals = analysis.literals.clone();
        literals.sort_unstable_by_key(|l| l.index());
        let mut expected = vec![Literal::negative(a), Literal::negative(b)];
        expected.sort_unstable_by_key(|l| l.index());
        assert_eq!(
            literals, expected,
            "both theory-forced literals must survive into the clause"
        );
        assert!(
            !analysis.asserting,
            "two conflict-level literals means the clause propagates nothing"
        );
        assert_falsified_by_trail(&solver, &analysis.literals);
    }

    /// Level-0 literals are dropped by resolution against their unit reasons,
    /// which is sound for a clause-justified assignment. A *theory*-justified
    /// level-0 assignment is a retractable arithmetic choice, not a
    /// consequence, so dropping it must be flagged -- otherwise an empty
    /// clause derived that way would be read as a refutation.
    #[test]
    fn test_pr31_conflict_analysis_flags_dropped_level_zero_theory_literal() {
        let mut solver = NlsatSolver::new();
        let t = solver.new_bool_var();
        let d = solver.new_bool_var();

        // Level 0: the theory forces `t`.
        solver
            .assignment
            .assign(Literal::positive(t), Justification::Theory);
        // Level 1: decide `d`.
        solver.assignment.push_level();
        solver
            .assignment
            .assign(Literal::positive(d), Justification::Decision);
        let conflict = add_reason_clause(
            &mut solver,
            vec![Literal::negative(t), Literal::negative(d)],
        );

        let analysis = solver
            .analyze_conflict(conflict)
            .expect("analysis completes");
        assert!(
            analysis.theory_dependent,
            "the level-0 literal that was dropped was theory-justified"
        );
        assert_eq!(analysis.literals, vec![Literal::negative(d)]);
        assert_falsified_by_trail(&solver, &analysis.literals);
    }

    /// The control for the case above: an ordinary level-0 unit assignment is
    /// dropped without flagging anything, because resolving it away is a real
    /// resolution step.
    #[test]
    fn test_pr31_conflict_analysis_level_zero_unit_drop_is_not_flagged() {
        let mut solver = NlsatSolver::new();
        let u = solver.new_bool_var();
        let d = solver.new_bool_var();

        solver
            .assignment
            .assign(Literal::positive(u), Justification::Unit);
        solver.assignment.push_level();
        solver
            .assignment
            .assign(Literal::positive(d), Justification::Decision);
        let conflict = add_reason_clause(
            &mut solver,
            vec![Literal::negative(u), Literal::negative(d)],
        );

        let analysis = solver
            .analyze_conflict(conflict)
            .expect("analysis completes");
        assert!(!analysis.theory_dependent);
        assert_eq!(analysis.literals, vec![Literal::negative(d)]);
    }

    // -----------------------------------------------------------------------
    // Behaviour-preservation: pin the exact redundancy verdict for concrete,
    // hand-verifiable reason-clause shapes (audit: item 4 asked specifically
    // that dedup/memoization must not change which literals are judged
    // redundant).
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_redundant_literal_dependency_already_in_clause() {
        let mut solver = NlsatSolver::new();
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.assignment.push_level();

        // a's reason is (a, b); b is at level 1 and IS already present in
        // the clause being minimized, so a is redundant without needing to
        // examine b's own justification at all.
        solver
            .assignment
            .assign(Literal::positive(b), Justification::Decision);
        let reason_a = add_reason_clause(
            &mut solver,
            vec![Literal::positive(a), Literal::positive(b)],
        );
        assign_propagated(&mut solver, a, reason_a);

        let clause = [
            Literal::positive(z),
            Literal::positive(a),
            Literal::positive(b),
        ];
        assert!(
            solver.is_redundant_literal(a, &clause),
            "a's only non-trivial dependency (b) is already explicit in the clause"
        );
    }

    #[test]
    fn test_is_redundant_literal_dependency_not_covered_is_not_redundant() {
        let mut solver = NlsatSolver::new();
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let c = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.assignment.push_level();

        // c is a decision (terminal, never redundant); b's reason cites c;
        // a's reason cites b. Neither b nor c is present in the clause
        // being minimized, so a must transitively fail through b and c.
        solver
            .assignment
            .assign(Literal::positive(c), Justification::Decision);
        let reason_b = add_reason_clause(
            &mut solver,
            vec![Literal::positive(b), Literal::positive(c)],
        );
        assign_propagated(&mut solver, b, reason_b);
        let reason_a = add_reason_clause(
            &mut solver,
            vec![Literal::positive(a), Literal::positive(b)],
        );
        assign_propagated(&mut solver, a, reason_a);

        let clause = [Literal::positive(z), Literal::positive(a)];
        assert!(
            !solver.is_redundant_literal(a, &clause),
            "a depends on b depends on c, a Decision literal absent from the clause"
        );
    }

    #[test]
    fn test_is_redundant_literal_level_zero_dependency_always_fine() {
        let mut solver = NlsatSolver::new();
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        // No push_level(): both variables are assigned at level 0.
        solver
            .assignment
            .assign(Literal::positive(b), Justification::Decision);
        let reason_a = add_reason_clause(
            &mut solver,
            vec![Literal::positive(a), Literal::positive(b)],
        );
        assign_propagated(&mut solver, a, reason_a);

        // b is absent from the clause, but its level-0 assignment is always
        // fine regardless -- no recursion into b's own justification needed.
        let clause = [Literal::positive(a)];
        assert!(solver.is_redundant_literal(a, &clause));
    }

    #[test]
    fn test_minimize_clause_removes_exactly_the_redundant_literal() {
        // End-to-end pin through the actual consumer: learned clause
        // (z, a, b) where a is redundant (its only dependency, b, is
        // explicit in the clause) and b is not (its dependency c is a
        // Decision literal absent from the clause). Only `a` must be
        // dropped; z (the asserting literal, index 0) is never examined.
        let mut solver = NlsatSolver::new();
        let z = solver.new_bool_var();
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let c = solver.new_bool_var();
        solver.assignment.push_level();

        solver
            .assignment
            .assign(Literal::positive(c), Justification::Decision);
        let reason_b = add_reason_clause(
            &mut solver,
            vec![Literal::positive(b), Literal::positive(c)],
        );
        assign_propagated(&mut solver, b, reason_b);
        let reason_a = add_reason_clause(
            &mut solver,
            vec![Literal::positive(a), Literal::positive(b)],
        );
        assign_propagated(&mut solver, a, reason_a);

        let clause = vec![
            Literal::positive(z),
            Literal::positive(a),
            Literal::positive(b),
        ];
        let minimized = solver.minimize_clause(clause);

        assert_eq!(
            minimized,
            vec![Literal::positive(z), Literal::positive(b)],
            "a must be dropped (redundant) and b, z must be kept"
        );
    }

    // -----------------------------------------------------------------------
    // Cycle-safety: a reason-clause cycle can never arise from a sound
    // trail (see the doc comment on `is_redundant_literal`), but the
    // function itself had no way to detect one; construct one directly
    // against the trail/clause database to prove termination.
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_redundant_literal_cycle_terminates_conservatively_false() {
        let mut solver = NlsatSolver::new();
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.assignment.push_level();

        // a's reason cites b; b's reason cites a right back.
        let reason_a = add_reason_clause(
            &mut solver,
            vec![Literal::positive(a), Literal::positive(b)],
        );
        let reason_b = add_reason_clause(
            &mut solver,
            vec![Literal::positive(b), Literal::positive(a)],
        );
        assign_propagated(&mut solver, a, reason_a);
        assign_propagated(&mut solver, b, reason_b);

        let clause = [Literal::positive(z)];
        // The primary assertion is that this returns at all.
        let result = solver.is_redundant_literal(a, &clause);
        assert!(
            !result,
            "an unresolvable (cyclic) dependency must resolve to NOT redundant, \
             never to redundant -- the latter could drop a literal the clause needs"
        );
    }

    #[test]
    fn test_is_redundant_literal_reason_citing_only_itself_is_vacuously_redundant() {
        // A degenerate reason clause containing nothing but (repeats of)
        // the propagated literal itself is not a self-cycle: the
        // propagated literal is always skipped (it is not one of the
        // "other" literals a reason must justify), so there is nothing
        // left to check and the loop completes vacuously. This pins that
        // the self-skip is unaffected by the iterative rewrite.
        let mut solver = NlsatSolver::new();
        let a = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.assignment.push_level();

        let reason_a = add_reason_clause(
            &mut solver,
            vec![Literal::positive(a), Literal::positive(a)],
        );
        assign_propagated(&mut solver, a, reason_a);

        let clause = [Literal::positive(z)];
        assert!(solver.is_redundant_literal(a, &clause));
    }

    #[test]
    fn test_is_redundant_literal_three_cycle_terminates_conservatively_false() {
        // a's reason cites b, b's reason cites c, c's reason cites a: a
        // longer cycle than the direct 2-cycle case above, to confirm the
        // `on_stack` back-edge check catches a cycle at any length.
        let mut solver = NlsatSolver::new();
        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let c = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.assignment.push_level();

        let reason_a = add_reason_clause(
            &mut solver,
            vec![Literal::positive(a), Literal::positive(b)],
        );
        let reason_b = add_reason_clause(
            &mut solver,
            vec![Literal::positive(b), Literal::positive(c)],
        );
        let reason_c = add_reason_clause(
            &mut solver,
            vec![Literal::positive(c), Literal::positive(a)],
        );
        assign_propagated(&mut solver, a, reason_a);
        assign_propagated(&mut solver, b, reason_b);
        assign_propagated(&mut solver, c, reason_c);

        let clause = [Literal::positive(z)];
        let result = solver.is_redundant_literal(a, &clause);
        assert!(
            !result,
            "a 3-cycle must resolve to NOT redundant, never to redundant"
        );
    }

    #[test]
    fn test_is_redundant_literal_deep_chain_small_stack() {
        // Build (iteratively) a long propagation chain v_0 <- v_1 <- ... <-
        // v_depth (each v_i's reason cites v_{i-1}), with v_0 a Decision,
        // and check v_depth's redundancy from inside a thread with a
        // deliberately small (128 KiB) stack. A stack overflow aborts the
        // whole process, so "the thread returned at all" is itself part of
        // the assertion.
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut solver = NlsatSolver::new();
                let depth: usize = 6_250;
                let vars: Vec<BoolVar> = (0..=depth).map(|_| solver.new_bool_var()).collect();
                solver.assignment.push_level();
                solver
                    .assignment
                    .assign(Literal::positive(vars[0]), Justification::Decision);
                for i in 1..=depth {
                    let reason = add_reason_clause(
                        &mut solver,
                        vec![Literal::positive(vars[i]), Literal::positive(vars[i - 1])],
                    );
                    assign_propagated(&mut solver, vars[i], reason);
                }
                let clause: Vec<Literal> = Vec::new();
                let result = solver.is_redundant_literal(vars[depth], &clause);
                assert!(
                    !result,
                    "the chain bottoms out at a Decision literal, so it is not redundant"
                );
            })
            .expect("spawning a thread with an explicit stack size must succeed");
        handle
            .join()
            .expect("a deep propagation chain must not overflow a 128 KiB stack");
    }
}
