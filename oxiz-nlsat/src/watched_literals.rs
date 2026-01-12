//! Two-watched literal scheme for efficient unit propagation.
//!
//! The two-watched literal scheme is a fundamental optimization in modern
//! SAT solvers. Instead of scanning all literals in a clause during unit
//! propagation, we maintain two "watched" literals per clause. When a literal
//! becomes false, we only need to update the watch if we can't find another
//! unassigned literal in the clause.
//!
//! Reference: "Chaff: Engineering an Efficient SAT Solver" by Moskewicz et al.

use crate::clause::ClauseId;
use crate::types::{Lbool, Literal, NULL_BOOL_VAR};
use std::collections::HashMap;

/// A watched literal entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatchEntry {
    /// The clause being watched.
    pub clause_id: ClauseId,
    /// The other watched literal in the clause.
    pub blocker: Literal,
}

impl WatchEntry {
    /// Create a new watch entry.
    pub fn new(clause_id: ClauseId, blocker: Literal) -> Self {
        Self { clause_id, blocker }
    }
}

/// Two-watched literal manager.
#[derive(Debug, Clone)]
pub struct WatchedLiterals {
    /// Watch lists: maps each literal to clauses watching it.
    /// watches[lit] = list of clauses where lit is watched
    watches: HashMap<Literal, Vec<WatchEntry>>,
    /// For each clause, stores the two watched literals.
    /// clause_watches[clause_id] = (lit1, lit2)
    clause_watches: HashMap<ClauseId, (Literal, Literal)>,
}

impl WatchedLiterals {
    /// Create a new watched literals manager.
    pub fn new() -> Self {
        Self {
            watches: HashMap::new(),
            clause_watches: HashMap::new(),
        }
    }

    /// Initialize watches for a clause.
    ///
    /// Watches the first two literals of the clause.
    pub fn watch_clause(&mut self, clause_id: ClauseId, literals: &[Literal]) {
        if literals.is_empty() {
            return;
        }

        let lit0 = literals[0];
        let lit1 = if literals.len() > 1 {
            literals[1]
        } else {
            literals[0]
        };

        // Add watches
        self.watches
            .entry(lit0)
            .or_default()
            .push(WatchEntry::new(clause_id, lit1));

        if literals.len() > 1 {
            self.watches
                .entry(lit1)
                .or_default()
                .push(WatchEntry::new(clause_id, lit0));
        }

        // Store which literals are watched for this clause
        self.clause_watches.insert(clause_id, (lit0, lit1));
    }

    /// Remove all watches for a clause.
    pub fn unwatch_clause(&mut self, clause_id: ClauseId) {
        if let Some((lit0, lit1)) = self.clause_watches.remove(&clause_id) {
            // Remove from watch lists
            if let Some(list) = self.watches.get_mut(&lit0) {
                list.retain(|w| w.clause_id != clause_id);
            }
            if let Some(list) = self.watches.get_mut(&lit1) {
                list.retain(|w| w.clause_id != clause_id);
            }
        }
    }

    /// Get the watch list for a literal.
    pub fn get_watches(&self, lit: Literal) -> Option<&[WatchEntry]> {
        self.watches.get(&lit).map(|v| v.as_slice())
    }

    /// Get mutable watch list for a literal.
    pub fn get_watches_mut(&mut self, lit: Literal) -> Option<&mut Vec<WatchEntry>> {
        self.watches.get_mut(&lit)
    }

    /// Get the watched literals for a clause.
    pub fn get_clause_watches(&self, clause_id: ClauseId) -> Option<(Literal, Literal)> {
        self.clause_watches.get(&clause_id).copied()
    }

    /// Update a watched literal for a clause.
    ///
    /// Replaces the old watched literal with a new one.
    pub fn update_watch(
        &mut self,
        clause_id: ClauseId,
        old_lit: Literal,
        new_lit: Literal,
        blocker: Literal,
    ) {
        // Remove old watch
        if let Some(list) = self.watches.get_mut(&old_lit) {
            list.retain(|w| w.clause_id != clause_id);
        }

        // Add new watch
        self.watches
            .entry(new_lit)
            .or_default()
            .push(WatchEntry::new(clause_id, blocker));

        // Update clause watches
        if let Some((lit0, lit1)) = self.clause_watches.get_mut(&clause_id) {
            if *lit0 == old_lit {
                *lit0 = new_lit;
            } else if *lit1 == old_lit {
                *lit1 = new_lit;
            }
        }
    }

    /// Clear all watches.
    pub fn clear(&mut self) {
        self.watches.clear();
        self.clause_watches.clear();
    }

    /// Get the number of clauses being watched.
    pub fn num_watched_clauses(&self) -> usize {
        self.clause_watches.len()
    }

    /// Get the total number of watch entries.
    pub fn num_watch_entries(&self) -> usize {
        self.watches.values().map(|v| v.len()).sum()
    }
}

impl Default for WatchedLiterals {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for watched literal propagation.
#[derive(Debug, Clone, Default)]
pub struct WatchStats {
    /// Number of watch list traversals.
    pub watch_traversals: u64,
    /// Number of watch updates.
    pub watch_updates: u64,
    /// Number of unit propagations via watches.
    pub unit_propagations: u64,
}

/// Helper for efficient unit propagation using watched literals.
pub struct WatchedPropagator {
    watched: WatchedLiterals,
    stats: WatchStats,
}

impl WatchedPropagator {
    /// Create a new watched propagator.
    pub fn new() -> Self {
        Self {
            watched: WatchedLiterals::new(),
            stats: WatchStats::default(),
        }
    }

    /// Get the watched literals manager.
    pub fn watched(&self) -> &WatchedLiterals {
        &self.watched
    }

    /// Get mutable watched literals manager.
    pub fn watched_mut(&mut self) -> &mut WatchedLiterals {
        &mut self.watched
    }

    /// Get the statistics.
    pub fn stats(&self) -> &WatchStats {
        &self.stats
    }

    /// Add a clause to watch.
    pub fn add_clause(&mut self, clause_id: ClauseId, literals: &[Literal]) {
        self.watched.watch_clause(clause_id, literals);
    }

    /// Remove a clause from watches.
    pub fn remove_clause(&mut self, clause_id: ClauseId) {
        self.watched.unwatch_clause(clause_id);
    }

    /// Propagate a literal assignment.
    ///
    /// Returns a list of (clause_id, unit_literal) pairs that became unit.
    /// Returns None if a conflict was detected.
    pub fn propagate<F>(
        &mut self,
        lit: Literal,
        get_clause_literals: &mut F,
        eval_literal: impl Fn(Literal) -> Lbool,
    ) -> Result<Vec<(ClauseId, Literal)>, ClauseId>
    where
        F: FnMut(ClauseId) -> Vec<Literal>,
    {
        let neg_lit = lit.negate();
        let mut unit_clauses = Vec::new();

        // Get watches for ~lit (clauses that become potentially unit when lit is assigned)
        let watch_list = match self.watched.get_watches(neg_lit) {
            Some(list) => list.to_vec(), // Copy to avoid borrow issues
            None => return Ok(unit_clauses),
        };

        self.stats.watch_traversals += 1;

        let mut i = 0;
        while i < watch_list.len() {
            let entry = watch_list[i];
            let clause_id = entry.clause_id;
            let blocker = entry.blocker;

            // Quick check: if blocker is true, clause is satisfied
            if eval_literal(blocker) == Lbool::True {
                i += 1;
                continue;
            }

            // Get clause literals
            let clause_lits = get_clause_literals(clause_id);

            // Find the two watched literals
            let mut watch0 = NULL_BOOL_VAR;
            let mut watch1 = NULL_BOOL_VAR;
            let mut watch0_lit = Literal::new(0, true);
            let mut watch1_lit = Literal::new(0, true);

            if let Some((w0, w1)) = self.watched.get_clause_watches(clause_id) {
                watch0_lit = w0;
                watch1_lit = w1;
                watch0 = w0.var();
                watch1 = w1.var();
            }

            // Try to find a new watch
            let mut found_new_watch = false;
            for &clause_lit in &clause_lits {
                if clause_lit.var() == watch0 || clause_lit.var() == watch1 {
                    continue; // Already watched
                }

                let lit_value = eval_literal(clause_lit);
                if lit_value != Lbool::False {
                    // Found a new literal to watch
                    self.watched
                        .update_watch(clause_id, neg_lit, clause_lit, blocker);
                    self.stats.watch_updates += 1;
                    found_new_watch = true;
                    break;
                }
            }

            if !found_new_watch {
                // Could not find new watch - check if unit or conflict
                let other_watch = if watch0_lit == neg_lit {
                    watch1_lit
                } else {
                    watch0_lit
                };

                match eval_literal(other_watch) {
                    Lbool::False => {
                        // Conflict!
                        return Err(clause_id);
                    }
                    Lbool::Undef => {
                        // Unit clause
                        unit_clauses.push((clause_id, other_watch));
                        self.stats.unit_propagations += 1;
                    }
                    Lbool::True => {
                        // Satisfied
                    }
                }
            }

            i += 1;
        }

        Ok(unit_clauses)
    }
}

impl Default for WatchedPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BoolVar;

    fn make_literal(var: BoolVar, sign: bool) -> Literal {
        Literal::new(var, sign)
    }

    #[test]
    fn test_watched_literals_new() {
        let watched = WatchedLiterals::new();
        assert_eq!(watched.num_watched_clauses(), 0);
        assert_eq!(watched.num_watch_entries(), 0);
    }

    #[test]
    fn test_watch_clause() {
        let mut watched = WatchedLiterals::new();

        let clause_id = 1;
        let literals = vec![make_literal(0, true), make_literal(1, false)];

        watched.watch_clause(clause_id, &literals);

        assert_eq!(watched.num_watched_clauses(), 1);
        assert!(watched.get_clause_watches(clause_id).is_some());
    }

    #[test]
    fn test_unwatch_clause() {
        let mut watched = WatchedLiterals::new();

        let clause_id = 1;
        let literals = vec![make_literal(0, true), make_literal(1, false)];

        watched.watch_clause(clause_id, &literals);
        watched.unwatch_clause(clause_id);

        assert_eq!(watched.num_watched_clauses(), 0);
    }

    #[test]
    fn test_get_watches() {
        let mut watched = WatchedLiterals::new();

        let clause_id = 1;
        let lit = make_literal(0, true);
        let literals = vec![lit, make_literal(1, false)];

        watched.watch_clause(clause_id, &literals);

        let watches = watched.get_watches(lit);
        assert!(watches.is_some());
        assert_eq!(watches.unwrap().len(), 1);
    }

    #[test]
    fn test_watched_propagator_new() {
        let prop = WatchedPropagator::new();
        assert_eq!(prop.stats().watch_traversals, 0);
        assert_eq!(prop.stats().unit_propagations, 0);
    }

    #[test]
    fn test_add_remove_clause() {
        let mut prop = WatchedPropagator::new();

        let clause_id = 1;
        let literals = vec![make_literal(0, true), make_literal(1, false)];

        prop.add_clause(clause_id, &literals);
        assert_eq!(prop.watched().num_watched_clauses(), 1);

        prop.remove_clause(clause_id);
        assert_eq!(prop.watched().num_watched_clauses(), 0);
    }

    #[test]
    fn test_update_watch() {
        let mut watched = WatchedLiterals::new();

        let clause_id = 1;
        let old_lit = make_literal(0, true);
        let new_lit = make_literal(2, true);
        let blocker = make_literal(1, false);
        let literals = vec![old_lit, blocker];

        watched.watch_clause(clause_id, &literals);
        watched.update_watch(clause_id, old_lit, new_lit, blocker);

        let (w0, w1) = watched.get_clause_watches(clause_id).unwrap();
        assert!(w0 == new_lit || w1 == new_lit);
    }
}
