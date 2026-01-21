//! Assignment trail for CDCL solver

use crate::clause::ClauseId;
use crate::literal::{LBool, Lit, Var};

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

    fn assign(&mut self, lit: Lit, reason: Reason) {
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
            level: self.current_level,
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

    /// Get the current size of the trail (number of assignments)
    #[must_use]
    pub fn size(&self) -> usize {
        self.assignments.len()
    }

    /// Backtrack to a specific trail size (number of assignments)
    /// This is useful for incremental solving where we want to restore
    /// the exact state at a push point
    pub fn backtrack_to_size(&mut self, target_size: usize) {
        while self.assignments.len() > target_size {
            let lit = self
                .assignments
                .pop()
                .expect("assignments non-empty in loop condition");
            let var = lit.var();
            self.var_info[var.index()].value = LBool::Undef;
        }
        // Reset decision level tracking
        self.current_level = 0;
        self.level_starts.truncate(1);
        self.prop_head = self.assignments.len();
    }

    /// Backtrack to a given decision level
    pub fn backtrack_to(&mut self, level: u32) {
        self.backtrack_to_with_callback(level, |_| {});
    }

    /// Backtrack to a given decision level, calling the callback for each unassigned literal
    pub fn backtrack_to_with_callback<F>(&mut self, level: u32, mut callback: F)
    where
        F: FnMut(Lit),
    {
        if level >= self.current_level {
            return;
        }

        let target_idx = self.level_starts[(level + 1) as usize];

        // Unassign all literals above the target level
        while self.assignments.len() > target_idx {
            let lit = self
                .assignments
                .pop()
                .expect("assignments non-empty in loop condition");
            let var = lit.var();
            self.var_info[var.index()].value = LBool::Undef;
            callback(lit);
        }

        self.level_starts.truncate((level + 1) as usize);
        self.current_level = level;
        self.prop_head = self.assignments.len();
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
