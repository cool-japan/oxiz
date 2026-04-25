//! Unit propagation (BCP) and binary implication graph

use super::*;
#[cfg(feature = "profiling")]
use crate::profiling::{ProfilingCategory, ScopedTimer};

impl Solver {
    /// Unit propagation using two-watched literals
    ///
    /// Updates the current watch list in place so propagation does not rebuild
    /// the list through repeated pushes into a freshly emptied buffer.
    pub(super) fn propagate(&mut self) -> Option<ClauseId> {
        #[cfg(feature = "profiling")]
        let _timer = ScopedTimer::new(ProfilingCategory::SatPropagation);
        while let Some(lit) = self.trail.next_to_propagate() {
            self.stats.propagations += 1;

            // First, propagate binary implications (faster)
            let binary_len = self.binary_graph.get(lit).len();
            for idx in 0..binary_len {
                let (implied_lit, clause_id) = self.binary_graph.get(lit)[idx];
                let value = self.trail.lit_value(implied_lit);
                if value.is_false() {
                    // Conflict in binary clause
                    return Some(clause_id);
                } else if !value.is_defined() {
                    // Propagate
                    self.trail.assign_propagation(implied_lit, clause_id);

                    // Lazy hyper-binary resolution: check if we can learn a binary clause
                    if self.config.enable_lazy_hyper_binary {
                        self.check_hyper_binary_resolution(lit, implied_lit, clause_id);
                    }
                }
            }

            // Take the current watch list, mutate it in place, then move it
            // back once propagation for this literal is finished.
            let mut watches = core::mem::take(self.watches.get_mut(lit));
            let mut conflict_found: Option<ClauseId> = None;
            let mut watch_idx = 0;

            while watch_idx < watches.len() {
                let watcher = watches[watch_idx];

                if self.trail.lit_value(watcher.blocker).is_true() {
                    watch_idx += 1;
                    continue;
                }

                let clause = match self.clauses.get_mut(watcher.clause) {
                    Some(c) if !c.deleted => c,
                    _ => {
                        // Deleted clause - remove its watcher in place.
                        watches.swap_remove(watch_idx);
                        continue;
                    }
                };

                // Make sure the false literal is at position 1
                if clause.lits[0] == lit.negate() {
                    clause.swap(0, 1);
                }

                // If first watch is true, clause is satisfied
                let first = clause.lits[0];
                if self.trail.lit_value(first).is_true() {
                    watches[watch_idx] = Watcher::new(watcher.clause, first);
                    watch_idx += 1;
                    continue;
                }

                // Look for a new watch
                let mut found = false;
                for j in 2..clause.lits.len() {
                    let l = clause.lits[j];
                    if !self.trail.lit_value(l).is_false() {
                        clause.swap(1, j);
                        self.watches
                            .add(clause.lits[1].negate(), Watcher::new(watcher.clause, first));
                        watches.swap_remove(watch_idx);
                        found = true;
                        break;
                    }
                }

                if found {
                    continue;
                }

                // No new watch found - clause is unit or conflicting
                watches[watch_idx] = Watcher::new(watcher.clause, first);

                if self.trail.lit_value(first).is_false() {
                    conflict_found = Some(watcher.clause);
                    break;
                } else {
                    // Unit propagation
                    self.trail.assign_propagation(first, watcher.clause);

                    // Lazy hyper-binary resolution
                    if self.config.enable_lazy_hyper_binary {
                        self.check_hyper_binary_resolution(lit, first, watcher.clause);
                    }

                    watch_idx += 1;
                }
            }

            *self.watches.get_mut(lit) = watches;

            if let Some(conflict) = conflict_found {
                return Some(conflict);
            }
        }

        None
    }

    /// Check for hyper-binary resolution opportunity
    /// When propagating `implied` due to `lit` being assigned, check if we can
    /// learn a binary clause by resolving the reason clauses
    pub(super) fn check_hyper_binary_resolution(
        &mut self,
        _lit: Lit,
        implied: Lit,
        reason_id: ClauseId,
    ) {
        // Only check at higher decision levels to avoid overhead
        if self.trail.decision_level() < 2 {
            return;
        }

        // Get the reason clause
        let reason_clause = match self.clauses.get(reason_id) {
            Some(c) if c.lits.len() >= 2 && c.lits.len() <= 4 => c.lits.clone(),
            _ => return,
        };

        // Check if we can derive a binary clause
        // Look for literals in the reason clause that are assigned at the current level
        let current_level = self.trail.decision_level();
        let mut current_level_lits = SmallVec::<[Lit; 4]>::new();
        let mut has_non_zero_level_other = false;

        for &reason_lit in &reason_clause {
            if reason_lit != implied {
                let var = reason_lit.var();
                let level = self.trail.level(var);
                if level == current_level {
                    current_level_lits.push(reason_lit);
                } else if level > 0 {
                    // There's a literal at a non-zero level other than current
                    // This means the learned clause would depend on that assignment
                    // which is not safe for incremental solving
                    has_non_zero_level_other = true;
                }
            }
        }

        // If there's exactly one literal from the current level besides the implied one,
        // and all others are at level 0, we can safely learn a binary clause.
        // IMPORTANT: We must ensure ALL other literals are at level 0 for the learned
        // clause to be valid when new constraints are added incrementally.
        if current_level_lits.len() == 1 && !has_non_zero_level_other {
            let other_lit = current_level_lits[0];

            // Check if we can create a useful binary clause
            // The reason clause had other_lit FALSE and implied it. So we learn:
            // other_lit | implied (if other_lit is false, implied must be true)
            let binary_clause_lits = [other_lit, implied];

            // Check if this binary clause is new and useful
            // The binary clause is: other_lit | implied
            // This means: ~other_lit -> implied, and ~implied -> other_lit
            if !self.has_binary_implication(other_lit.negate(), implied) {
                // Learn this binary clause on-the-fly
                let clause_id = self.clauses.add_learned(binary_clause_lits.iter().copied());
                // Add correct implications: ~A -> B and ~B -> A for clause (A | B)
                self.binary_graph
                    .add(other_lit.negate(), implied, clause_id);
                self.binary_graph
                    .add(implied.negate(), other_lit, clause_id);
                self.stats.learned_clauses += 1;
            }
        }
    }

    /// Check if a binary implication already exists
    pub(super) fn has_binary_implication(&self, from_lit: Lit, to_lit: Lit) -> bool {
        self.binary_graph
            .get(from_lit)
            .iter()
            .any(|(lit, _)| *lit == to_lit)
    }
}
