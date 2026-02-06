//! Watched Literal Propagation for Theory Solvers.
#![allow(dead_code, missing_docs)] // Under development
//!
//! Implements efficient watched literal schemes for theory propagation,
//! similar to two-watched literals in CDCL SAT solvers.
//!
//! ## Watched Schemes
//!
//! - **Two-Watched**: Track two literals per constraint
//! - **Multi-Watched**: Track multiple literals for complex constraints
//! - **Lazy Watching**: Defer watch updates until necessary
//!
//! ## References
//!
//! - "Efficient Theory Propagation" (Nieuwenhuis et al., 2006)
//! - Z3's `smt/watched.cpp`

use oxiz_sat::Lit;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Constraint identifier.
pub type ConstraintId = usize;

/// Watch type for constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchType {
    /// Watch for propagation when literal becomes false.
    OnFalse,
    /// Watch for propagation when literal becomes true.
    OnTrue,
    /// Watch for any change.
    OnChange,
}

/// A watched literal with associated constraint.
#[derive(Debug, Clone)]
pub struct Watch {
    /// Constraint being watched.
    pub constraint: ConstraintId,
    /// Type of watch.
    pub watch_type: WatchType,
    /// Index of this watch in the constraint (for multi-watched).
    pub watch_index: usize,
}

impl Watch {
    /// Create a new watch.
    pub fn new(constraint: ConstraintId, watch_type: WatchType, watch_index: usize) -> Self {
        Self {
            constraint,
            watch_type,
            watch_index,
        }
    }
}

/// A constraint with watched literals.
#[derive(Debug, Clone)]
pub struct WatchedConstraint {
    /// Constraint identifier.
    pub id: ConstraintId,
    /// All literals in the constraint.
    pub literals: Vec<Lit>,
    /// Indices of currently watched literals.
    pub watched: Vec<usize>,
    /// Additional data (theory-specific).
    pub data: ConstraintData,
}

/// Theory-specific constraint data.
#[derive(Debug, Clone)]
pub enum ConstraintData {
    /// Generic constraint with no extra data.
    Generic,
    /// Linear arithmetic constraint with coefficients.
    Linear { coeffs: Vec<i64>, bound: i64 },
    /// Cardinality constraint (sum of lits ≤ k).
    Cardinality { k: usize },
    /// Pseudo-Boolean constraint.
    PseudoBoolean { coeffs: Vec<i64>, bound: i64 },
}

/// Configuration for watched propagation.
#[derive(Debug, Clone)]
pub struct WatchedConfig {
    /// Number of literals to watch per constraint.
    pub num_watches: usize,
    /// Enable lazy watch updates.
    pub lazy_updates: bool,
    /// Batch propagation threshold.
    pub batch_threshold: usize,
}

impl Default for WatchedConfig {
    fn default() -> Self {
        Self {
            num_watches: 2,
            lazy_updates: false,
            batch_threshold: 10,
        }
    }
}

/// Statistics for watched propagation.
#[derive(Debug, Clone, Default)]
pub struct WatchedStats {
    /// Propagations triggered.
    pub propagations: u64,
    /// Watch updates performed.
    pub watch_updates: u64,
    /// Conflicts detected.
    pub conflicts: u64,
    /// Batch propagations.
    pub batch_propagations: u64,
}

/// Watched literal propagator.
pub struct WatchedPropagator {
    /// Configuration.
    config: WatchedConfig,
    /// Statistics.
    stats: WatchedStats,
    /// Watch lists: literal -> watches.
    watches: FxHashMap<Lit, Vec<Watch>>,
    /// All constraints.
    constraints: FxHashMap<ConstraintId, WatchedConstraint>,
    /// Next constraint ID.
    next_id: ConstraintId,
    /// Propagation queue.
    prop_queue: VecDeque<(Lit, ConstraintId)>,
}

impl WatchedPropagator {
    /// Create a new watched propagator.
    pub fn new(config: WatchedConfig) -> Self {
        Self {
            config,
            stats: WatchedStats::default(),
            watches: FxHashMap::default(),
            constraints: FxHashMap::default(),
            next_id: 0,
            prop_queue: VecDeque::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(WatchedConfig::default())
    }

    /// Add a constraint to be watched.
    pub fn add_constraint(&mut self, literals: Vec<Lit>, data: ConstraintData) -> ConstraintId {
        let id = self.next_id;
        self.next_id += 1;

        // Choose initial watches
        let num_watches = self.config.num_watches.min(literals.len());
        let watched: Vec<usize> = (0..num_watches).collect();

        // Create constraint
        let constraint = WatchedConstraint {
            id,
            literals: literals.clone(),
            watched: watched.clone(),
            data,
        };

        self.constraints.insert(id, constraint);

        // Add watches
        for &watch_idx in &watched {
            let lit = literals[watch_idx];
            let watch = Watch::new(id, WatchType::OnFalse, watch_idx);

            self.watches.entry(lit).or_default().push(watch);
        }

        id
    }

    /// Remove a constraint.
    pub fn remove_constraint(&mut self, id: ConstraintId) {
        if let Some(constraint) = self.constraints.remove(&id) {
            // Remove all watches for this constraint
            for &watch_idx in &constraint.watched {
                let lit = constraint.literals[watch_idx];

                if let Some(watch_list) = self.watches.get_mut(&lit) {
                    watch_list.retain(|w| w.constraint != id);
                }
            }
        }
    }

    /// Notify that a literal has been assigned.
    ///
    /// Returns propagated literals or a conflict.
    pub fn notify_assigned(&mut self, lit: Lit, value: bool) -> Result<Vec<Lit>, ConstraintId> {
        let mut propagated = Vec::new();

        // Get watches for this literal
        let watch_lit = if value { !lit } else { lit };

        let watches = match self.watches.get(&watch_lit).cloned() {
            Some(w) => w,
            None => return Ok(propagated),
        };

        // Process each watch
        for watch in watches {
            let result = self.process_watch(watch, lit, value)?;

            propagated.extend(result);
        }

        self.stats.propagations += propagated.len() as u64;

        Ok(propagated)
    }

    /// Process a single watch.
    fn process_watch(
        &mut self,
        watch: Watch,
        assigned_lit: Lit,
        assigned_value: bool,
    ) -> Result<Vec<Lit>, ConstraintId> {
        let constraint = match self.constraints.get(&watch.constraint) {
            Some(c) => c.clone(),
            None => return Ok(Vec::new()),
        };

        // Check if we need to find a new watch
        if self.should_update_watch(&constraint, watch.watch_index, assigned_lit, assigned_value) {
            self.stats.watch_updates += 1;

            // Try to find a new literal to watch
            if let Some(new_watch_idx) = self.find_new_watch(&constraint, watch.watch_index) {
                self.update_watch(watch.constraint, watch.watch_index, new_watch_idx);
                return Ok(Vec::new());
            }

            // No new watch found - check for propagation or conflict
            return self.check_propagation(&constraint);
        }

        Ok(Vec::new())
    }

    /// Check if a watch should be updated.
    fn should_update_watch(
        &self,
        constraint: &WatchedConstraint,
        watch_idx: usize,
        assigned_lit: Lit,
        assigned_value: bool,
    ) -> bool {
        // Simplified: update if watched literal is assigned false
        let watched_lit = constraint.literals[watch_idx];
        watched_lit == assigned_lit && !assigned_value
    }

    /// Find a new literal to watch.
    fn find_new_watch(
        &self,
        constraint: &WatchedConstraint,
        _old_watch_idx: usize,
    ) -> Option<usize> {
        // Look for an unwatched, unassigned literal
        for (idx, _lit) in constraint.literals.iter().enumerate() {
            if !constraint.watched.contains(&idx) {
                // Would check assignment status here
                return Some(idx);
            }
        }

        None
    }

    /// Update a watch to a new literal.
    fn update_watch(&mut self, constraint_id: ConstraintId, old_idx: usize, new_idx: usize) {
        if let Some(constraint) = self.constraints.get_mut(&constraint_id) {
            let old_lit = constraint.literals[old_idx];
            let new_lit = constraint.literals[new_idx];

            // Remove old watch
            if let Some(watch_list) = self.watches.get_mut(&old_lit) {
                watch_list.retain(|w| !(w.constraint == constraint_id && w.watch_index == old_idx));
            }

            // Add new watch
            let watch = Watch::new(constraint_id, WatchType::OnFalse, new_idx);
            self.watches.entry(new_lit).or_default().push(watch);

            // Update constraint's watched list
            if let Some(pos) = constraint.watched.iter().position(|&idx| idx == old_idx) {
                constraint.watched[pos] = new_idx;
            }
        }
    }

    /// Check for propagation or conflict in a constraint.
    fn check_propagation(
        &mut self,
        constraint: &WatchedConstraint,
    ) -> Result<Vec<Lit>, ConstraintId> {
        // Simplified: Check based on constraint type
        match &constraint.data {
            ConstraintData::Cardinality { k: _ } => {
                // Would check if exactly k literals can be true
                Ok(Vec::new())
            }
            _ => {
                // Generic: no propagation for now
                Ok(Vec::new())
            }
        }
    }

    /// Get number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> &WatchedStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = WatchedStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_sat::Var;

    #[test]
    fn test_propagator_creation() {
        let propagator = WatchedPropagator::default_config();
        assert_eq!(propagator.num_constraints(), 0);
    }

    #[test]
    fn test_add_constraint() {
        let mut propagator = WatchedPropagator::default_config();

        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::pos(Var::new(1));
        let lit3 = Lit::pos(Var::new(2));

        let id =
            propagator.add_constraint(vec![lit1, lit2, lit3], ConstraintData::Cardinality { k: 2 });

        assert_eq!(propagator.num_constraints(), 1);
        assert!(propagator.constraints.contains_key(&id));
    }

    #[test]
    fn test_remove_constraint() {
        let mut propagator = WatchedPropagator::default_config();

        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::pos(Var::new(1));

        let id = propagator.add_constraint(vec![lit1, lit2], ConstraintData::Generic);

        assert_eq!(propagator.num_constraints(), 1);

        propagator.remove_constraint(id);
        assert_eq!(propagator.num_constraints(), 0);
    }

    #[test]
    fn test_stats() {
        let propagator = WatchedPropagator::default_config();
        assert_eq!(propagator.stats().propagations, 0);
    }
}
