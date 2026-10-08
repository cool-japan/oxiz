//! Assignment trail for CDCL solver

use crate::clause::ClauseId;
use crate::literal::{LBool, Lit, Var};
#[allow(unused_imports)]
use crate::prelude::*;

/// Reason for an assignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Decision (no antecedent)
    Decision,
    /// Unit propagation from a clause
    Propagation(ClauseId),
    /// Theory propagation
    Theory,
}

/// Information about a variable assignment
#[derive(Debug, Clone, Copy)]
pub struct VarInfo {
    /// Current value
    pub value: LBool,
    /// Decision level at which assigned
    pub level: u32,
    /// Reason for assignment
    pub reason: Reason,
    /// Position in trail
    #[allow(dead_code)]
    pub trail_idx: u32,
}

impl Default for VarInfo {
    fn default() -> Self {
        Self {
            value: LBool::Undef,
            level: 0,
            reason: Reason::Decision,
            trail_idx: 0,
        }
    }
}

/// The assignment trail
#[derive(Debug)]
pub struct Trail {
    /// Sequence of assigned literals
    assignments: Vec<Lit>,
    /// Information for each variable
    var_info: Vec<VarInfo>,
    /// Indices marking the start of each decision level
    level_starts: Vec<usize>,
    /// Current decision level
    current_level: u32,
    /// Propagation queue head
    prop_head: usize,
    /// Scratch buffer for literals that survive a backtrack because their
    /// assignment level is at or below the target level (see
    /// [`Trail::backtrack_to_with_callback`]).  Kept as a field so the
    /// out-of-order rollback path allocates nothing in steady state.
    replay_buf: Vec<Lit>,
}

impl Trail {
    /// Create a new trail for n variables
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        Self {
            assignments: Vec::with_capacity(num_vars),
            var_info: vec![VarInfo::default(); num_vars],
            level_starts: vec![0],
            current_level: 0,
            prop_head: 0,
            replay_buf: Vec::new(),
        }
    }

    /// Get the current decision level
    #[must_use]
    pub fn decision_level(&self) -> u32 {
        self.current_level
    }

    /// Get the value of a variable
    #[must_use]
    pub fn value(&self, var: Var) -> LBool {
        self.var_info
            .get(var.index())
            .map_or(LBool::Undef, |v| v.value)
    }

    /// Get the value of a literal
    #[must_use]
    pub fn lit_value(&self, lit: Lit) -> LBool {
        let val = self.value(lit.var());
        if lit.is_pos() { val } else { val.negate() }
    }

    /// Check if a variable is assigned
    #[must_use]
    pub fn is_assigned(&self, var: Var) -> bool {
        self.value(var).is_defined()
    }

    /// Get the level at which a variable was assigned
    #[must_use]
    pub fn level(&self, var: Var) -> u32 {
        self.var_info.get(var.index()).map_or(0, |v| v.level)
    }

    /// Get the reason for a variable's assignment
    #[must_use]
    pub fn reason(&self, var: Var) -> Reason {
        self.var_info
            .get(var.index())
            .map_or(Reason::Decision, |v| v.reason)
    }

    /// Get the decision variable chosen when entering decision `level`
    /// (1-indexed: level 0 has no decision, only unconditional facts).
    ///
    /// `level_starts[level]` is recorded by [`Self::new_decision_level`]
    /// immediately before the decision literal itself is pushed, and that
    /// position is never overwritten by a later backtrack (a rollback either
    /// truncates the level away entirely or leaves everything at or below
    /// its target level untouched) — so this is a plain, allocation-free
    /// lookup. Returns `None` for level 0 or a level beyond the current
    /// trail, and defensively if the recorded reason ever turns out not to
    /// be a decision (should not happen; see the invariant above).
    #[must_use]
    pub fn decision_var_at_level(&self, level: u32) -> Option<Var> {
        if level == 0 {
            return None;
        }
        let idx = *self.level_starts.get(level as usize)?;
        let lit = self.assignments.get(idx)?;
        let var = lit.var();
        matches!(self.reason(var), Reason::Decision).then_some(var)
    }

    /// Start a new decision level
    pub fn new_decision_level(&mut self) {
        self.current_level += 1;
        self.level_starts.push(self.assignments.len());
    }

    /// Assign a literal as a decision
    pub fn assign_decision(&mut self, lit: Lit) {
        self.assign(lit, Reason::Decision);
    }

    /// Assign a literal due to propagation
    pub fn assign_propagation(&mut self, lit: Lit, clause: ClauseId) {
        self.assign(lit, Reason::Propagation(clause));
    }

    /// Assign a literal due to theory propagation
    pub fn assign_theory(&mut self, lit: Lit) {
        self.assign(lit, Reason::Theory);
    }

    /// Assign a propagated literal at an *explicit* decision level.
    ///
    /// Under chronological backtracking the level at which a literal is implied
    /// is the maximum level over the **other** literals of its reason clause,
    /// which can be strictly below the current decision level (Nadel & Ryvchin,
    /// *Chronological Backtracking*, SAT 2018; Z3 computes the same value in
    /// `solver::propagate_clause` / `assign_core`).  Recording the current
    /// decision level instead over-approximates the literal's dependency set:
    /// it stays logically sound, but the assignment is then thrown away by every
    /// backtrack below that inflated level even though it is still implied.
    ///
    /// Assignments made this way make the trail **non-monotone in level**, which
    /// is why [`Self::backtrack_to_with_callback`] filters by level rather than
    /// truncating at a level boundary.
    pub fn assign_propagation_at(&mut self, lit: Lit, clause: ClauseId, level: u32) {
        self.assign_at(lit, Reason::Propagation(clause), level);
    }

    /// Assign a literal as an unconditional (root-level) fact.
    ///
    /// Used for learned **unit** clauses, which are consequences of the formula
    /// alone and therefore hold at level 0 no matter which decision level the
    /// search happens to be at when they are derived.  Pinning them at level 0
    /// is what makes them survive every later backtrack; recording them at the
    /// current level instead both loses them on the next rollback and plants a
    /// second reason-less "decision" inside a level, which breaks the 1-UIP
    /// termination invariant in the solver's conflict analysis.
    pub fn assign_unit_fact(&mut self, lit: Lit) {
        self.assign_at(lit, Reason::Decision, 0);
    }

    fn assign(&mut self, lit: Lit, reason: Reason) {
        let level = self.current_level;
        self.assign_at(lit, reason, level);
    }

    fn assign_at(&mut self, lit: Lit, reason: Reason, level: u32) {
        debug_assert!(
            level <= self.current_level,
            "cannot assign above the current decision level"
        );
        let var = lit.var();
        let idx = var.index();

        // Resize if needed
        if idx >= self.var_info.len() {
            self.var_info.resize(idx + 1, VarInfo::default());
        }

        let value = if lit.is_pos() {
            LBool::True
        } else {
            LBool::False
        };

        self.var_info[idx] = VarInfo {
            value,
            level,
            reason,
            trail_idx: self.assignments.len() as u32,
        };

        self.assignments.push(lit);
    }

    /// Get the next literal to propagate (if any)
    pub fn next_to_propagate(&mut self) -> Option<Lit> {
        if self.prop_head < self.assignments.len() {
            let lit = self.assignments[self.prop_head];
            self.prop_head += 1;
            Some(lit)
        } else {
            None
        }
    }

    /// Check if there are literals to propagate
    #[must_use]
    pub fn has_pending_propagation(&self) -> bool {
        self.prop_head < self.assignments.len()
    }

    /// Put the literal most recently handed out by [`Self::next_to_propagate`]
    /// back on the propagation queue.
    ///
    /// The trail's central invariant is that every literal *strictly before*
    /// `prop_head` has had **all** of its consequences computed.  `propagate()`
    /// breaks that invariant whenever it aborts: the conflict is detected part
    /// way through a literal's watch list (or its binary-implication edges), so
    /// the remaining watchers of that literal were never examined even though
    /// the head has already moved past it.
    ///
    /// Ordinary CDCL repairs this implicitly — conflict analysis is always
    /// followed by a backtrack, and [`Self::backtrack_to_with_callback`] clamps
    /// the head to the rollback boundary, below the half-processed literal.  A
    /// conflict at decision level 0 has no such backtrack: the solver reports
    /// `Unsat` and returns with the head left past a literal whose conflicting
    /// clause was never re-examined.  A subsequent `solve()` on the same solver
    /// then resumes propagation *after* that literal, never revisits the clause,
    /// and reports `Sat` on a formula refuted by unit propagation alone.
    ///
    /// Re-queueing the literal restores the invariant unconditionally, and is
    /// idempotent: re-propagating an already-assigned literal only re-walks its
    /// watch list.  Reference: MiniSat's `cancelUntil`, which restores the same
    /// property by resetting `qhead` to the backtrack level's trail boundary.
    pub fn requeue_last_propagated(&mut self) {
        debug_assert!(
            self.prop_head > 0,
            "requeue_last_propagated requires a literal to have been dequeued"
        );
        self.prop_head = self.prop_head.saturating_sub(1);
    }

    /// Current position of the propagation head.
    ///
    /// Directly after a rollback this is at most the rollback boundary returned
    /// by [`Self::backtrack_to_with_callback`], which makes it a safe (never
    /// too large) clamp for callers that keep their own cursor into the trail
    /// but do not observe the rollback themselves.
    #[must_use]
    pub fn propagation_head(&self) -> usize {
        self.prop_head
    }

    /// Rewind the propagation head so the whole surviving trail is propagated
    /// again on the next `propagate()`.
    ///
    /// Needed after a rollback that discards assignments *without* discarding
    /// the literals that implied them — the incremental
    /// `Solver::restore_to_trail_size` path, which keeps a committed prefix but
    /// throws away the level-0 consequences derived from it.  Re-propagating a
    /// literal that is still assigned is a no-op, so rewinding is always safe;
    /// it only costs one extra pass over the retained watch lists.
    pub fn reset_propagation_head(&mut self) {
        self.prop_head = 0;
    }

    /// Get the current size of the trail (number of assignments)
    #[must_use]
    pub fn size(&self) -> usize {
        self.assignments.len()
    }

    /// Backtrack to a specific trail size (number of assignments)
    /// This is useful for incremental solving where we want to restore
    /// the exact state at a push point
    ///
    /// # Precondition
    ///
    /// Every literal in the retained prefix must be assigned at level 0: the
    /// method resets the decision level to 0, so a retained literal recorded at
    /// a higher level would leave the trail internally inconsistent.  Both
    /// callers (`Solver::pop` and `Solver::restore_to_trail_size`) capture
    /// `target_size` while the solver sits at the root level, which guarantees
    /// this.
    pub fn backtrack_to_size(&mut self, target_size: usize) {
        debug_assert!(
            self.assignments
                .iter()
                .take(target_size)
                .all(|l| self.var_info[l.var().index()].level == 0),
            "backtrack_to_size must retain only root-level assignments"
        );
        while self.assignments.len() > target_size
            && let Some(lit) = self.assignments.pop()
        {
            let var = lit.var();
            self.var_info[var.index()].value = LBool::Undef;
        }
        // Reset decision level tracking
        self.current_level = 0;
        self.level_starts.truncate(1);
        self.prop_head = self.assignments.len();
    }

    /// Backtrack to a given decision level
    pub fn backtrack_to(&mut self, level: u32) -> usize {
        self.backtrack_to_with_callback(level, |_| {})
    }

    /// Backtrack to a given decision level, calling the callback for each
    /// literal that is actually unassigned.
    ///
    /// Returns the trail index at which the rolled-back region started, i.e. the
    /// length of the definitely-untouched prefix.  Callers that track a cursor
    /// into the trail (theory notification, for instance) must clamp it to this
    /// value rather than to [`Self::size`], because literals that survive the
    /// rollback are re-appended after it and have to be re-processed.
    ///
    /// # Chronological backtracking
    ///
    /// The trail is **not** sorted by decision level: a literal implied at a low
    /// level can be appended while the search sits at a much higher one (see
    /// [`Self::assign_propagation_at`] and [`Self::assign_unit_fact`]).
    /// Truncating the trail at the level boundary — the textbook rollback, valid
    /// only for a level-monotone trail — would therefore discard assignments
    /// that are still implied at or below `level`, silently losing facts and, for
    /// level-0 units, losing them permanently.
    ///
    /// So this filters instead of truncating: everything from the level boundary
    /// onwards is scanned, literals whose assignment level is at or below
    /// `level` are kept (compacted to the end of the surviving trail, with their
    /// recorded trail positions fixed up) and the rest are unassigned.  The
    /// propagation head is rewound to the boundary so the retained literals are
    /// propagated again — their consequences at the higher levels were just
    /// removed, and re-propagating an already-assigned literal is a no-op.
    ///
    /// Reference: Z3's `sat_solver.cpp` `solver::unassign_vars`, which performs
    /// the same keep-and-replay pass.
    pub fn backtrack_to_with_callback<F>(&mut self, level: u32, mut callback: F) -> usize
    where
        F: FnMut(Lit),
    {
        if level >= self.current_level {
            return self.assignments.len();
        }

        let target_idx = self.level_starts[(level + 1) as usize];

        // Reuse the scratch buffer so a rollback never allocates in steady state.
        let mut replay = core::mem::take(&mut self.replay_buf);
        replay.clear();

        for idx in target_idx..self.assignments.len() {
            let lit = self.assignments[idx];
            let var_idx = lit.var().index();
            if self.var_info[var_idx].level <= level {
                replay.push(lit);
            } else {
                self.var_info[var_idx].value = LBool::Undef;
                callback(lit);
            }
        }

        self.assignments.truncate(target_idx);
        // Never advance the head: on a conflict, propagation stops early and
        // literals below the head may still be unpropagated.
        self.prop_head = self.prop_head.min(target_idx);

        for &lit in &replay {
            self.var_info[lit.var().index()].trail_idx = self.assignments.len() as u32;
            self.assignments.push(lit);
        }

        replay.clear();
        self.replay_buf = replay;

        self.level_starts.truncate((level + 1) as usize);
        self.current_level = level;

        target_idx
    }

    /// Get the number of assigned variables
    #[must_use]
    pub fn num_assigned(&self) -> usize {
        self.assignments.len()
    }

    /// Get all assignments
    #[must_use]
    pub fn assignments(&self) -> &[Lit] {
        &self.assignments
    }

    /// Get assignments at current level
    #[must_use]
    pub fn level_assignments(&self) -> &[Lit] {
        let start = *self.level_starts.last().unwrap_or(&0);
        &self.assignments[start..]
    }

    /// Resize to support more variables
    pub fn resize(&mut self, num_vars: usize) {
        if num_vars > self.var_info.len() {
            self.var_info.resize(num_vars, VarInfo::default());
        }
    }

    /// Clear the trail completely
    pub fn clear(&mut self) {
        for lit in &self.assignments {
            self.var_info[lit.var().index()].value = LBool::Undef;
        }
        self.assignments.clear();
        self.level_starts.clear();
        self.level_starts.push(0);
        self.current_level = 0;
        self.prop_head = 0;
        self.replay_buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trail_basic() {
        let mut trail = Trail::new(5);

        assert_eq!(trail.decision_level(), 0);
        assert!(!trail.is_assigned(Var::new(0)));

        trail.new_decision_level();
        trail.assign_decision(Lit::pos(Var::new(0)));

        assert_eq!(trail.decision_level(), 1);
        assert!(trail.is_assigned(Var::new(0)));
        assert!(trail.lit_value(Lit::pos(Var::new(0))).is_true());
        assert!(trail.lit_value(Lit::neg(Var::new(0))).is_false());
    }

    #[test]
    fn test_trail_backtrack() {
        let mut trail = Trail::new(5);

        trail.new_decision_level();
        trail.assign_decision(Lit::pos(Var::new(0)));

        trail.new_decision_level();
        trail.assign_decision(Lit::neg(Var::new(1)));

        assert_eq!(trail.decision_level(), 2);
        assert_eq!(trail.num_assigned(), 2);

        trail.backtrack_to(1);

        assert_eq!(trail.decision_level(), 1);
        assert_eq!(trail.num_assigned(), 1);
        assert!(trail.is_assigned(Var::new(0)));
        assert!(!trail.is_assigned(Var::new(1)));
    }

    #[test]
    fn test_trail_propagation() {
        let mut trail = Trail::new(5);

        trail.new_decision_level();
        trail.assign_decision(Lit::pos(Var::new(0)));
        trail.assign_propagation(Lit::neg(Var::new(1)), ClauseId::new(0));

        assert!(trail.has_pending_propagation());
        assert_eq!(trail.next_to_propagate(), Some(Lit::pos(Var::new(0))));
        assert_eq!(trail.next_to_propagate(), Some(Lit::neg(Var::new(1))));
        assert!(!trail.has_pending_propagation());
    }
}
