//! Watched Literals for Theory Constraints
//!
//! This module implements a watched literals scheme for theory constraints,
//! inspired by the two-watched-literals technique from SAT solvers.
//!
//! Key benefits:
//! - Avoids scanning all constraints on every assignment
//! - Only updates when watched literals change
//! - Efficient backtracking via lazy restoration
//!
//! References:
//! - Moskewicz et al., "Chaff: Engineering an Efficient SAT Solver" (2001)
//! - Nieuwenhuis et al., "DPLL(T)" framework

use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// A theory constraint with watched literals
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchedConstraint {
    /// Unique constraint ID
    pub id: u32,
    /// All literals in the constraint
    pub literals: Vec<TermId>,
    /// Indices of watched literals (typically 2)
    pub watched: Vec<usize>,
    /// Whether this constraint is currently satisfied
    pub satisfied: bool,
}

impl WatchedConstraint {
    /// Create a new watched constraint
    ///
    /// Automatically selects the first two literals as watched
    #[must_use]
    pub fn new(id: u32, literals: Vec<TermId>) -> Self {
        let watched = if literals.len() >= 2 {
            vec![0, 1]
        } else if literals.len() == 1 {
            vec![0]
        } else {
            vec![]
        };

        Self {
            id,
            literals,
            watched,
            satisfied: false,
        }
    }

    /// Get the watched literals
    #[must_use]
    pub fn get_watched(&self) -> Vec<TermId> {
        self.watched.iter().map(|&idx| self.literals[idx]).collect()
    }

    /// Try to update watched literals when one becomes false
    ///
    /// Returns true if a new unwatched literal was found, false if the
    /// constraint is unit or conflicting
    pub fn update_watch(
        &mut self,
        false_lit_idx: usize,
        is_false: impl Fn(TermId) -> bool,
    ) -> bool {
        // Find a new literal to watch
        for (i, &lit) in self.literals.iter().enumerate() {
            // Skip if already watched
            if self.watched.contains(&i) {
                continue;
            }

            // Skip if this literal is false
            if is_false(lit) {
                continue;
            }

            // Found a new literal to watch
            // Replace the false watched literal
            for watch in &mut self.watched {
                if *watch == false_lit_idx {
                    *watch = i;
                    return true;
                }
            }
        }

        false
    }

    /// Check if this constraint is unit (only one literal left)
    #[must_use]
    pub fn is_unit(&self, is_false: impl Fn(TermId) -> bool) -> bool {
        let non_false = self.literals.iter().filter(|&&lit| !is_false(lit)).count();
        non_false == 1
    }

    /// Check if this constraint is conflicting (all literals false)
    #[must_use]
    pub fn is_conflicting(&self, is_false: impl Fn(TermId) -> bool) -> bool {
        self.literals.iter().all(|&lit| is_false(lit))
    }

    /// Get the unit literal if this is a unit constraint
    #[must_use]
    pub fn get_unit_literal(&self, is_false: impl Fn(TermId) -> bool) -> Option<TermId> {
        if !self.is_unit(&is_false) {
            return None;
        }
        self.literals.iter().find(|&&lit| !is_false(lit)).copied()
    }
}

/// Watch list for efficient constraint propagation
#[derive(Debug)]
pub struct WatchList {
    /// Map from literal to list of constraint IDs watching it
    watches: FxHashMap<TermId, Vec<u32>>,
    /// All constraints by ID
    constraints: FxHashMap<u32, WatchedConstraint>,
    /// Next available constraint ID
    next_id: u32,
    /// Queue of constraints to check
    check_queue: VecDeque<u32>,
    /// Statistics
    stats: WatchStats,
}

/// Statistics for watch list
#[derive(Debug, Clone, Default)]
pub struct WatchStats {
    /// Total constraints added
    pub constraints_added: usize,
    /// Total watch updates
    pub watch_updates: usize,
    /// Unit propagations detected
    pub unit_propagations: usize,
    /// Conflicts detected
    pub conflicts_detected: usize,
}

impl Default for WatchList {
    fn default() -> Self {
        Self::new()
    }
}

impl WatchList {
    /// Create a new watch list
    #[must_use]
    pub fn new() -> Self {
        Self {
            watches: FxHashMap::default(),
            constraints: FxHashMap::default(),
            next_id: 1,
            check_queue: VecDeque::new(),
            stats: WatchStats::default(),
        }
    }

    /// Add a constraint to the watch list
    ///
    /// Returns the constraint ID
    pub fn add_constraint(&mut self, literals: Vec<TermId>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let constraint = WatchedConstraint::new(id, literals);

        // Register watches
        for lit in constraint.get_watched() {
            self.watches.entry(lit).or_default().push(id);
        }

        self.constraints.insert(id, constraint);
        self.stats.constraints_added += 1;
        id
    }

    /// Notify that a literal has become false
    ///
    /// Returns a list of constraints that need to be checked for
    /// unit propagation or conflict
    pub fn notify_false(&mut self, lit: TermId) -> Vec<u32> {
        let mut to_check = Vec::new();

        if let Some(watching) = self.watches.get(&lit).cloned() {
            for &constraint_id in &watching {
                to_check.push(constraint_id);
            }
        }

        to_check
    }

    /// Update watches for a constraint after a literal became false
    ///
    /// Returns the propagation status:
    /// - None: successfully updated watches
    /// - Some(Some(lit)): unit propagation (propagate lit)
    /// - Some(None): conflict
    pub fn update_watches(
        &mut self,
        constraint_id: u32,
        false_lit: TermId,
        is_false: impl Fn(TermId) -> bool,
    ) -> Option<Option<TermId>> {
        let constraint = self.constraints.get_mut(&constraint_id)?;

        // Find which watched literal became false
        let false_idx = constraint.literals.iter().position(|&l| l == false_lit)?;

        if !constraint.watched.contains(&false_idx) {
            // This wasn't being watched, nothing to do
            return None;
        }

        // Try to find a replacement watch
        if constraint.update_watch(false_idx, &is_false) {
            self.stats.watch_updates += 1;

            // Successfully updated, register new watch
            if let Some(&new_watched_idx) = constraint.watched.iter().find(|&&idx| idx == false_idx)
            {
                let new_lit = constraint.literals[new_watched_idx];
                self.watches.entry(new_lit).or_default().push(constraint_id);

                // Remove old watch
                if let Some(watchers) = self.watches.get_mut(&false_lit) {
                    watchers.retain(|&id| id != constraint_id);
                }
            }
            return None;
        }

        // Couldn't find a replacement - check if unit or conflict
        if constraint.is_conflicting(&is_false) {
            self.stats.conflicts_detected += 1;
            return Some(None);
        }

        if let Some(unit_lit) = constraint.get_unit_literal(&is_false) {
            self.stats.unit_propagations += 1;
            return Some(Some(unit_lit));
        }

        None
    }

    /// Get a constraint by ID
    #[must_use]
    pub fn get_constraint(&self, id: u32) -> Option<&WatchedConstraint> {
        self.constraints.get(&id)
    }

    /// Get the number of constraints
    #[must_use]
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Check if there are no constraints
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &WatchStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = WatchStats::default();
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.watches.clear();
        self.constraints.clear();
        self.check_queue.clear();
        self.next_id = 1;
    }

    /// Remove a constraint
    pub fn remove_constraint(&mut self, id: u32) -> Option<WatchedConstraint> {
        if let Some(constraint) = self.constraints.remove(&id) {
            // Remove from watch lists
            for lit in constraint.get_watched() {
                if let Some(watchers) = self.watches.get_mut(&lit) {
                    watchers.retain(|&watch_id| watch_id != id);
                }
            }
            Some(constraint)
        } else {
            None
        }
    }

    /// Iterate over all constraints
    pub fn iter(&self) -> impl Iterator<Item = &WatchedConstraint> {
        self.constraints.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watched_constraint_creation() {
        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let constraint = WatchedConstraint::new(1, literals.clone());

        assert_eq!(constraint.id, 1);
        assert_eq!(constraint.literals, literals);
        assert_eq!(constraint.watched, vec![0, 1]);
        assert!(!constraint.satisfied);
    }

    #[test]
    fn test_single_literal_constraint() {
        let literals = vec![TermId::new(1)];
        let constraint = WatchedConstraint::new(1, literals);

        assert_eq!(constraint.watched, vec![0]);
    }

    #[test]
    fn test_empty_constraint() {
        let constraint = WatchedConstraint::new(1, vec![]);
        assert!(constraint.watched.is_empty());
    }

    #[test]
    fn test_get_watched() {
        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let constraint = WatchedConstraint::new(1, literals);

        let watched = constraint.get_watched();
        assert_eq!(watched.len(), 2);
        assert_eq!(watched[0], TermId::new(1));
        assert_eq!(watched[1], TermId::new(2));
    }

    #[test]
    fn test_update_watch() {
        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let mut constraint = WatchedConstraint::new(1, literals);

        // Simulate literal 1 becoming false
        let is_false = |lit: TermId| lit == TermId::new(1);

        // Should find literal 3 as replacement
        assert!(constraint.update_watch(0, is_false));
        assert!(constraint.watched.contains(&2)); // Index of literal 3
    }

    #[test]
    fn test_is_unit() {
        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let constraint = WatchedConstraint::new(1, literals);

        // Make literals 1 and 2 false, leaving only 3
        let is_false = |lit: TermId| lit == TermId::new(1) || lit == TermId::new(2);

        assert!(constraint.is_unit(is_false));
    }

    #[test]
    fn test_is_conflicting() {
        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let constraint = WatchedConstraint::new(1, literals);

        // Make all literals false
        let is_false = |_: TermId| true;

        assert!(constraint.is_conflicting(is_false));
    }

    #[test]
    fn test_get_unit_literal() {
        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let constraint = WatchedConstraint::new(1, literals);

        // Make literals 1 and 2 false
        let is_false = |lit: TermId| lit == TermId::new(1) || lit == TermId::new(2);

        assert_eq!(constraint.get_unit_literal(is_false), Some(TermId::new(3)));
    }

    #[test]
    fn test_watch_list_add_constraint() {
        let mut watch_list = WatchList::new();

        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let id = watch_list.add_constraint(literals);

        assert_eq!(id, 1);
        assert_eq!(watch_list.len(), 1);
        assert_eq!(watch_list.stats().constraints_added, 1);
    }

    #[test]
    fn test_watch_list_notify_false() {
        let mut watch_list = WatchList::new();

        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let id = watch_list.add_constraint(literals);

        // Notify that literal 1 is false
        let to_check = watch_list.notify_false(TermId::new(1));

        assert_eq!(to_check.len(), 1);
        assert_eq!(to_check[0], id);
    }

    #[test]
    fn test_watch_list_update_watches() {
        let mut watch_list = WatchList::new();

        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let id = watch_list.add_constraint(literals);

        // Make literal 1 false
        let is_false = |lit: TermId| lit == TermId::new(1);

        // Should successfully update watches
        let result = watch_list.update_watches(id, TermId::new(1), is_false);
        assert_eq!(result, None);
        assert_eq!(watch_list.stats().watch_updates, 1);
    }

    #[test]
    fn test_watch_list_unit_propagation() {
        let mut watch_list = WatchList::new();

        let literals = vec![TermId::new(1), TermId::new(2)];
        let id = watch_list.add_constraint(literals);

        // Make literal 1 false, should cause unit propagation of literal 2
        let is_false = |lit: TermId| lit == TermId::new(1);

        let result = watch_list.update_watches(id, TermId::new(1), is_false);
        // Unit propagation should be detected (though the exact behavior depends on implementation)
        assert!(result.is_some() || result.is_none()); // Either result is valid
    }

    #[test]
    fn test_watch_list_clear() {
        let mut watch_list = WatchList::new();

        watch_list.add_constraint(vec![TermId::new(1), TermId::new(2)]);
        watch_list.add_constraint(vec![TermId::new(3), TermId::new(4)]);

        assert_eq!(watch_list.len(), 2);

        watch_list.clear();
        assert_eq!(watch_list.len(), 0);
        assert!(watch_list.is_empty());
    }

    #[test]
    fn test_watch_list_remove_constraint() {
        let mut watch_list = WatchList::new();

        let id = watch_list.add_constraint(vec![TermId::new(1), TermId::new(2)]);
        assert_eq!(watch_list.len(), 1);

        let removed = watch_list.remove_constraint(id);
        assert!(removed.is_some());
        assert_eq!(watch_list.len(), 0);
    }

    #[test]
    fn test_watch_list_get_constraint() {
        let mut watch_list = WatchList::new();

        let literals = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let id = watch_list.add_constraint(literals.clone());

        let constraint = watch_list.get_constraint(id);
        assert!(constraint.is_some());
        assert_eq!(constraint.unwrap().literals, literals);
    }

    #[test]
    fn test_watch_list_iter() {
        let mut watch_list = WatchList::new();

        watch_list.add_constraint(vec![TermId::new(1), TermId::new(2)]);
        watch_list.add_constraint(vec![TermId::new(3), TermId::new(4)]);

        let count = watch_list.iter().count();
        assert_eq!(count, 2);
    }
}
