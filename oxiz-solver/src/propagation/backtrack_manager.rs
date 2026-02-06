//! Backtrack Manager for SMT Solver.
//!
//! Manages:
//! - Decision stack and backtracking
//! - Trail of assignments
//! - Undo/redo operations
//! - Incremental solving state

use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;

/// Backtrack manager for incremental SMT solving.
pub struct BacktrackManager {
    /// Decision stack: levels and their decision literals
    decision_stack: Vec<DecisionLevel>,
    /// Current decision level
    current_level: usize,
    /// Trail of all assignments
    trail: Vec<Assignment>,
    /// Variable to trail index mapping
    var_to_trail: FxHashMap<TermId, usize>,
    /// Undo stack for theory-specific operations
    undo_stack: Vec<UndoAction>,
    /// Checkpoint stack for incremental solving
    checkpoints: Vec<Checkpoint>,
    /// Statistics
    stats: BacktrackStats,
}

/// A decision level in the search.
#[derive(Debug, Clone)]
pub struct DecisionLevel {
    /// Level number
    pub level: usize,
    /// Decision literal at this level (if any)
    pub decision: Option<Assignment>,
    /// Trail position at start of this level
    pub trail_start: usize,
}

/// An assignment to a variable.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Variable being assigned
    pub var: TermId,
    /// Assigned value
    pub value: bool,
    /// Decision level at which this was assigned
    pub level: usize,
    /// Reason for assignment (None = decision, Some = propagation)
    pub reason: Option<ReasonClause>,
}

/// Reason for a propagation.
#[derive(Debug, Clone)]
pub enum ReasonClause {
    /// Boolean clause that caused propagation
    BooleanClause(Vec<TermId>),
    /// Theory propagation
    TheoryPropagation(TheoryReason),
}

/// Theory-specific reason for propagation.
#[derive(Debug, Clone)]
pub struct TheoryReason {
    /// Theory identifier
    pub theory_id: usize,
    /// Theory-specific explanation
    pub explanation: Vec<TermId>,
}

/// Undo action for theory operations.
#[derive(Debug, Clone)]
pub enum UndoAction {
    /// Undo an equality assertion
    UndoEquality(TermId, TermId),
    /// Undo a bound update
    UndoBound(TermId, BoundUpdate),
    /// Undo an array store
    UndoArrayStore(TermId),
    /// Generic theory undo with callback identifier
    TheoryUndo(usize, Vec<TermId>),
}

/// Bound update information.
#[derive(Debug, Clone)]
pub struct BoundUpdate {
    /// Old lower bound
    pub old_lower: Option<i64>,
    /// Old upper bound
    pub old_upper: Option<i64>,
}

/// Checkpoint for incremental solving.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Decision level at checkpoint
    pub level: usize,
    /// Trail size at checkpoint
    pub trail_size: usize,
    /// Undo stack size at checkpoint
    pub undo_size: usize,
}

/// Backtrack statistics.
#[derive(Debug, Clone, Default)]
pub struct BacktrackStats {
    /// Number of decisions made
    pub decisions: usize,
    /// Number of propagations
    pub propagations: usize,
    /// Number of backtracks
    pub backtracks: usize,
    /// Number of full restarts
    pub restarts: usize,
    /// Number of undo operations
    pub undos: usize,
    /// Maximum decision level reached
    pub max_level: usize,
}

impl BacktrackManager {
    /// Create a new backtrack manager.
    pub fn new() -> Self {
        Self {
            decision_stack: vec![DecisionLevel {
                level: 0,
                decision: None,
                trail_start: 0,
            }],
            current_level: 0,
            trail: Vec::new(),
            var_to_trail: FxHashMap::default(),
            undo_stack: Vec::new(),
            checkpoints: Vec::new(),
            stats: BacktrackStats::default(),
        }
    }

    /// Make a decision: assign variable at a new decision level.
    pub fn decide(&mut self, var: TermId, value: bool) -> Result<(), String> {
        self.current_level += 1;
        self.stats.decisions += 1;

        if self.current_level > self.stats.max_level {
            self.stats.max_level = self.current_level;
        }

        let assignment = Assignment {
            var,
            value,
            level: self.current_level,
            reason: None, // Decision has no reason
        };

        // Record trail position
        let trail_start = self.trail.len();

        // Push decision level
        self.decision_stack.push(DecisionLevel {
            level: self.current_level,
            decision: Some(assignment.clone()),
            trail_start,
        });

        // Add to trail
        self.assign(assignment)?;

        Ok(())
    }

    /// Propagate: assign variable due to constraint propagation.
    pub fn propagate(
        &mut self,
        var: TermId,
        value: bool,
        reason: ReasonClause,
    ) -> Result<(), String> {
        self.stats.propagations += 1;

        let assignment = Assignment {
            var,
            value,
            level: self.current_level,
            reason: Some(reason),
        };

        self.assign(assignment)?;

        Ok(())
    }

    /// Assign a variable.
    fn assign(&mut self, assignment: Assignment) -> Result<(), String> {
        // Check if already assigned
        if let Some(&trail_idx) = self.var_to_trail.get(&assignment.var) {
            let existing = &self.trail[trail_idx];
            if existing.value != assignment.value {
                return Err(format!(
                    "Conflict: variable {:?} already assigned differently",
                    assignment.var
                ));
            }
            // Already assigned with same value
            return Ok(());
        }

        // Record trail position
        let trail_idx = self.trail.len();
        self.var_to_trail.insert(assignment.var, trail_idx);

        // Add to trail
        self.trail.push(assignment);

        Ok(())
    }

    /// Backtrack to a specific decision level.
    pub fn backtrack(&mut self, target_level: usize) -> Result<(), String> {
        if target_level > self.current_level {
            return Err("Cannot backtrack to higher level".to_string());
        }

        if target_level == self.current_level {
            return Ok(()); // Nothing to do
        }

        self.stats.backtracks += 1;

        // Find trail position for target level
        // We want to keep assignments at target_level, so we use the trail_start of the next level
        let target_trail_pos = if target_level + 1 < self.decision_stack.len() {
            self.decision_stack[target_level + 1].trail_start
        } else {
            // No assignments to remove (shouldn't happen if backtracking to lower level)
            self.trail.len()
        };

        // Undo assignments
        while self.trail.len() > target_trail_pos {
            if let Some(assignment) = self.trail.pop() {
                self.var_to_trail.remove(&assignment.var);
            }
        }

        // Undo theory operations
        self.undo_to_level(target_level)?;

        // Update decision stack
        self.decision_stack.truncate(target_level + 1);
        self.current_level = target_level;

        Ok(())
    }

    /// Restart: backtrack to level 0.
    pub fn restart(&mut self) -> Result<(), String> {
        self.stats.restarts += 1;
        self.backtrack(0)
    }

    /// Record an undo action.
    pub fn record_undo(&mut self, action: UndoAction) {
        self.undo_stack.push(action);
    }

    /// Undo theory operations to target level.
    fn undo_to_level(&mut self, _target_level: usize) -> Result<(), String> {
        // Find how many undo operations to perform
        // (Simplified: undo all operations beyond target level)

        let mut undos_to_apply = Vec::new();

        // Collect undo operations (in reverse order)
        // Pop directly instead of checking last() then popping
        while let Some(undo) = self.undo_stack.pop() {
            undos_to_apply.push(undo);

            // Stop condition (simplified)
            if undos_to_apply.len() > 100 {
                break;
            }
        }

        // Apply undo operations
        for undo in undos_to_apply {
            self.apply_undo(undo)?;
            self.stats.undos += 1;
        }

        Ok(())
    }

    /// Apply an undo operation.
    fn apply_undo(&mut self, _action: UndoAction) -> Result<(), String> {
        // Theory-specific undo logic would go here
        // This is a placeholder
        Ok(())
    }

    /// Create a checkpoint for incremental solving.
    pub fn push_checkpoint(&mut self) {
        self.checkpoints.push(Checkpoint {
            level: self.current_level,
            trail_size: self.trail.len(),
            undo_size: self.undo_stack.len(),
        });
    }

    /// Pop to most recent checkpoint.
    pub fn pop_checkpoint(&mut self) -> Result<(), String> {
        if let Some(checkpoint) = self.checkpoints.pop() {
            self.backtrack(checkpoint.level)?;

            // Restore undo stack size
            self.undo_stack.truncate(checkpoint.undo_size);

            Ok(())
        } else {
            Err("No checkpoint to pop".to_string())
        }
    }

    /// Get current decision level.
    pub fn current_level(&self) -> usize {
        self.current_level
    }

    /// Check if variable is assigned.
    pub fn is_assigned(&self, var: TermId) -> bool {
        self.var_to_trail.contains_key(&var)
    }

    /// Get assignment for variable.
    pub fn get_assignment(&self, var: TermId) -> Option<&Assignment> {
        if let Some(&trail_idx) = self.var_to_trail.get(&var) {
            self.trail.get(trail_idx)
        } else {
            None
        }
    }

    /// Get all assignments at current level.
    pub fn current_assignments(&self) -> &[Assignment] {
        &self.trail
    }

    /// Get decision at specific level.
    pub fn get_decision(&self, level: usize) -> Option<&Assignment> {
        if level < self.decision_stack.len() {
            self.decision_stack[level].decision.as_ref()
        } else {
            None
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &BacktrackStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = BacktrackStats::default();
    }
}

impl Default for BacktrackManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtrack_manager() {
        let mgr = BacktrackManager::new();
        assert_eq!(mgr.current_level(), 0);
        assert_eq!(mgr.stats.decisions, 0);
    }

    #[test]
    fn test_decide() {
        let mut mgr = BacktrackManager::new();

        let var = TermId::from(1);
        mgr.decide(var, true).unwrap();

        assert_eq!(mgr.current_level(), 1);
        assert_eq!(mgr.stats.decisions, 1);
        assert!(mgr.is_assigned(var));
    }

    #[test]
    fn test_propagate() {
        let mut mgr = BacktrackManager::new();

        let var1 = TermId::from(1);
        mgr.decide(var1, true).unwrap();

        let var2 = TermId::from(2);
        let reason = ReasonClause::BooleanClause(vec![var1]);
        mgr.propagate(var2, false, reason).unwrap();

        assert_eq!(mgr.stats.propagations, 1);
        assert!(mgr.is_assigned(var2));
    }

    #[test]
    fn test_backtrack() {
        let mut mgr = BacktrackManager::new();

        let var1 = TermId::from(1);
        mgr.decide(var1, true).unwrap();

        let var2 = TermId::from(2);
        mgr.decide(var2, false).unwrap();

        assert_eq!(mgr.current_level(), 2);

        mgr.backtrack(1).unwrap();

        assert_eq!(mgr.current_level(), 1);
        assert!(mgr.is_assigned(var1));
        assert!(!mgr.is_assigned(var2));
    }

    #[test]
    fn test_restart() {
        let mut mgr = BacktrackManager::new();

        mgr.decide(TermId::from(1), true).unwrap();
        mgr.decide(TermId::from(2), false).unwrap();

        mgr.restart().unwrap();

        assert_eq!(mgr.current_level(), 0);
        assert_eq!(mgr.stats.restarts, 1);
        assert_eq!(mgr.trail.len(), 0);
    }

    #[test]
    fn test_checkpoint() {
        let mut mgr = BacktrackManager::new();

        mgr.decide(TermId::from(1), true).unwrap();
        mgr.push_checkpoint();

        mgr.decide(TermId::from(2), false).unwrap();

        mgr.pop_checkpoint().unwrap();

        assert_eq!(mgr.current_level(), 1);
        assert!(!mgr.is_assigned(TermId::from(2)));
    }

    #[test]
    fn test_get_assignment() {
        let mut mgr = BacktrackManager::new();

        let var = TermId::from(1);
        mgr.decide(var, true).unwrap();

        let assignment = mgr.get_assignment(var).unwrap();
        assert!(assignment.value);
        assert_eq!(assignment.level, 1);
        assert!(assignment.reason.is_none());
    }
}
