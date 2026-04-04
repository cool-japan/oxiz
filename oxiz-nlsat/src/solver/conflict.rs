//! Conflict analysis, backtracking, activity management, and restart/reduction
//! for the NLSAT solver.

use super::NlsatSolver;
use crate::assignment::Justification;
use crate::clause::ClauseId;
use crate::restart::RestartStrategy;
use crate::types::{BoolVar, Literal};
use oxiz_math::polynomial::Var;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashSet;

impl NlsatSolver {
    // ========== Conflict Analysis ==========

    /// Analyze a conflict and learn a clause.
    pub(super) fn analyze_conflict(&mut self, conflict_id: ClauseId) -> (Vec<Literal>, u32) {
        self.learnt_clause.clear();
        self.seen.clear();

        // Track this clause for unsat core
        if self.extract_unsat_core {
            self.conflict_clauses.insert(conflict_id);
        }

        let clause_lits: Vec<Literal> = match self.clauses.get(conflict_id) {
            Some(c) => c.literals().to_vec(),
            None => return (Vec::new(), 0),
        };

        let current_level = self.assignment.level();
        let mut counter = 0; // Number of literals at current level

        // Process conflict clause
        for &lit in &clause_lits {
            let var = lit.var();
            if !self.seen.contains(&var) {
                self.seen.insert(var);
                let level = self.assignment.bool_level(var);

                if level == current_level {
                    counter += 1;
                } else if level > 0 {
                    self.learnt_clause.push(lit.negate());
                    self.bump_var_activity(var);
                }
            }
        }

        // Resolve until we have exactly one literal at current level
        let mut trail_idx = self.assignment.trail().len();
        while counter > 1 && trail_idx > 0 {
            // Find next literal to resolve
            trail_idx -= 1;
            let trail = self.assignment.trail();

            let entry = &trail[trail_idx];
            let lit = entry.literal;
            let var = lit.var();

            if !self.seen.contains(&var) {
                continue;
            }
            self.seen.remove(&var);
            counter -= 1;

            // Get the reason clause
            if let Justification::Propagation(reason_id) = &entry.justification {
                // Track reason clause for unsat core
                if self.extract_unsat_core {
                    self.conflict_clauses.insert(*reason_id);
                }

                let reason_lits: Vec<Literal> = match self.clauses.get(*reason_id) {
                    Some(r) => r.literals().to_vec(),
                    None => continue,
                };

                for reason_lit in reason_lits {
                    if reason_lit == lit {
                        continue;
                    }

                    let reason_var = reason_lit.var();
                    if !self.seen.contains(&reason_var) {
                        self.seen.insert(reason_var);
                        let level = self.assignment.bool_level(reason_var);

                        if level == current_level {
                            counter += 1;
                        } else if level > 0 {
                            self.learnt_clause.push(reason_lit.negate());
                            self.bump_var_activity(reason_var);
                        }
                    }
                }
            }
        }

        // Find the UIP (asserting literal)
        trail_idx = self.assignment.trail().len();
        while trail_idx > 0 {
            trail_idx -= 1;
            let trail = self.assignment.trail();
            let entry = &trail[trail_idx];
            let var = entry.literal.var();

            if self.seen.contains(&var) {
                // This is the asserting literal
                self.learnt_clause.insert(0, entry.literal.negate());
                self.bump_var_activity(var);
                break;
            }
        }

        // Compute backtrack level
        let mut backtrack_level = 0;
        for lit in &self.learnt_clause[1..] {
            let level = self.assignment.bool_level(lit.var());
            backtrack_level = backtrack_level.max(level);
        }

        // Minimize learned clause (optional)
        let minimized = self.minimize_clause(self.learnt_clause.clone());

        (minimized, backtrack_level)
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
    pub(super) fn is_redundant_literal(&self, var: BoolVar, clause: &[Literal]) -> bool {
        // Get the justification for this variable
        let trail = self.assignment.trail();
        let entry = trail.iter().find(|e| e.literal.var() == var);

        if let Some(entry) = entry {
            match &entry.justification {
                Justification::Propagation(reason_id) => {
                    // Check if the reason clause's literals are all in the learned clause
                    if let Some(reason_clause) = self.clauses.get(*reason_id) {
                        let reason_lits = reason_clause.literals();

                        // All literals in reason (except the propagated one) should be
                        // either at level 0 or already in the learned clause
                        for &reason_lit in reason_lits {
                            if reason_lit.var() == var {
                                continue; // Skip the propagated literal
                            }

                            let reason_var = reason_lit.var();
                            let level = self.assignment.bool_level(reason_var);

                            if level == 0 {
                                continue; // Level 0 literals are always fine
                            }

                            // Check if this literal (or its negation) is in the clause
                            let in_clause = clause.iter().any(|&cl| cl.var() == reason_var);

                            if !in_clause {
                                // Check recursively if this variable is redundant
                                if !self.is_redundant_literal(reason_var, clause) {
                                    return false;
                                }
                            }
                        }

                        // All reason literals are covered
                        return true;
                    }
                }
                Justification::Decision | Justification::Unit | Justification::Theory => {
                    // Cannot minimize decision or unit literals
                    return false;
                }
            }
        }

        false
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
