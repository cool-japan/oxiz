//! Preprocessing techniques for SAT solving
//!
//! This module implements various preprocessing and simplification techniques:
//! - Blocked Clause Elimination (BCE)
//! - Variable Elimination (VE)
//! - Subsumption Elimination
//! - Pure Literal Elimination
//! - Self-Subsuming Resolution

use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::{Lit, Var};
use std::collections::HashSet;

/// Occurrence list: maps literals to clauses containing them
#[derive(Debug, Clone)]
struct OccurrenceList {
    /// occurrences[lit] = list of clause IDs containing lit
    occurrences: Vec<Vec<ClauseId>>,
}

impl OccurrenceList {
    fn new(num_vars: usize) -> Self {
        Self {
            occurrences: vec![Vec::new(); num_vars * 2],
        }
    }

    fn add(&mut self, lit: Lit, clause_id: ClauseId) {
        self.occurrences[lit.code() as usize].push(clause_id);
    }

    fn get(&self, lit: Lit) -> &[ClauseId] {
        &self.occurrences[lit.code() as usize]
    }

    fn remove(&mut self, lit: Lit, clause_id: ClauseId) {
        let list = &mut self.occurrences[lit.code() as usize];
        if let Some(pos) = list.iter().position(|&id| id == clause_id) {
            list.swap_remove(pos);
        }
    }

    fn clear(&mut self) {
        for list in &mut self.occurrences {
            list.clear();
        }
    }
}

/// Preprocessing context
#[derive(Debug)]
pub struct Preprocessor {
    /// Number of variables
    num_vars: usize,
    /// Occurrence lists
    occurrences: OccurrenceList,
    /// Eliminated variables
    eliminated: HashSet<Var>,
    /// Clauses to remove
    removed_clauses: HashSet<ClauseId>,
}

impl Preprocessor {
    /// Create a new preprocessor
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            occurrences: OccurrenceList::new(num_vars),
            eliminated: HashSet::new(),
            removed_clauses: HashSet::new(),
        }
    }

    /// Build occurrence lists from clause database
    pub fn build_occurrences(&mut self, clauses: &ClauseDatabase) {
        self.occurrences.clear();
        for i in 0..clauses.len() {
            let clause_id = ClauseId::new(i as u32);
            if let Some(clause) = clauses.get(clause_id)
                && !clause.deleted
            {
                for &lit in &clause.lits {
                    self.occurrences.add(lit, clause_id);
                }
            }
        }
    }

    /// Check if a clause is a tautology (contains both l and ~l)
    fn is_tautology(lits: &[Lit]) -> bool {
        let mut seen = HashSet::new();
        for &lit in lits {
            if seen.contains(&lit.negate()) {
                return true;
            }
            seen.insert(lit);
        }
        false
    }

    /// Check if clause C is blocked on literal l
    ///
    /// A clause C is blocked on literal l ∈ C if for every clause D with ~l ∈ D,
    /// the resolvent of C and D on l is a tautology.
    fn is_blocked(&self, clause_lits: &[Lit], blocking_lit: Lit, clauses: &ClauseDatabase) -> bool {
        // Get all clauses containing ~blocking_lit
        let neg_lit = blocking_lit.negate();

        for &other_clause_id in self.occurrences.get(neg_lit) {
            if let Some(other_clause) = clauses.get(other_clause_id) {
                if other_clause.deleted {
                    continue;
                }

                // Compute resolvent: (C \ {l}) ∪ (D \ {~l})
                let mut resolvent = Vec::new();

                // Add literals from C except blocking_lit
                for &lit in clause_lits {
                    if lit != blocking_lit {
                        resolvent.push(lit);
                    }
                }

                // Add literals from D except ~blocking_lit
                for &lit in &other_clause.lits {
                    if lit != neg_lit {
                        resolvent.push(lit);
                    }
                }

                // Check if resolvent is a tautology
                if !Self::is_tautology(&resolvent) {
                    return false;
                }
            }
        }

        true
    }

    /// Blocked Clause Elimination
    ///
    /// Remove clauses that are blocked on some literal.
    /// Returns the number of clauses eliminated.
    pub fn blocked_clause_elimination(&mut self, clauses: &mut ClauseDatabase) -> usize {
        let mut eliminated = 0;
        self.build_occurrences(clauses);

        // Try to eliminate each clause
        let clause_ids: Vec<_> = (0..clauses.len())
            .map(|i| ClauseId::new(i as u32))
            .collect();

        for clause_id in clause_ids {
            if self.removed_clauses.contains(&clause_id) {
                continue;
            }

            if let Some(clause) = clauses.get(clause_id) {
                if clause.deleted || clause.learned {
                    continue;
                }

                let lits = clause.lits.clone();

                // Try each literal as blocking literal
                for &lit in &lits {
                    if self.is_blocked(&lits, lit, clauses) {
                        // Mark clause for removal
                        if let Some(clause) = clauses.get_mut(clause_id) {
                            clause.deleted = true;
                        }
                        self.removed_clauses.insert(clause_id);
                        eliminated += 1;

                        // Update occurrence lists
                        for &l in &lits {
                            self.occurrences.remove(l, clause_id);
                        }
                        break;
                    }
                }
            }
        }

        eliminated
    }

    /// Pure Literal Elimination
    ///
    /// A pure literal is one that appears only in positive or only in negative form.
    /// All clauses containing a pure literal can be satisfied and removed.
    /// Returns the number of clauses eliminated.
    pub fn pure_literal_elimination(&mut self, clauses: &mut ClauseDatabase) -> usize {
        let mut eliminated = 0;
        self.build_occurrences(clauses);

        // Find pure literals
        let mut pure_literals = Vec::new();

        for v in 0..self.num_vars {
            let var = Var(v as u32);
            let pos_lit = Lit::pos(var);
            let neg_lit = Lit::neg(var);

            let pos_occurs = !self.occurrences.get(pos_lit).is_empty();
            let neg_occurs = !self.occurrences.get(neg_lit).is_empty();

            if pos_occurs && !neg_occurs {
                pure_literals.push(pos_lit);
            } else if neg_occurs && !pos_occurs {
                pure_literals.push(neg_lit);
            }
        }

        // Remove clauses containing pure literals
        for lit in pure_literals {
            for &clause_id in self.occurrences.get(lit).iter() {
                if !self.removed_clauses.contains(&clause_id)
                    && let Some(clause) = clauses.get_mut(clause_id)
                    && !clause.deleted
                    && !clause.learned
                {
                    clause.deleted = true;
                    self.removed_clauses.insert(clause_id);
                    eliminated += 1;
                }
            }
        }

        eliminated
    }

    /// Subsumption Elimination
    ///
    /// A clause C subsumes clause D if C ⊆ D (every literal in C is in D).
    /// If C subsumes D, then D can be removed.
    /// Returns the number of clauses eliminated.
    pub fn subsumption_elimination(&mut self, clauses: &mut ClauseDatabase) -> usize {
        let mut eliminated = 0;
        self.build_occurrences(clauses);

        let clause_ids: Vec<_> = (0..clauses.len())
            .map(|i| ClauseId::new(i as u32))
            .collect();

        for i in 0..clause_ids.len() {
            let clause_id = clause_ids[i];

            if self.removed_clauses.contains(&clause_id) {
                continue;
            }

            let clause_lits = if let Some(clause) = clauses.get(clause_id) {
                if clause.deleted || clause.learned {
                    continue;
                }
                clause.lits.clone()
            } else {
                continue;
            };

            // Check if this clause subsumes any other clause
            for &other_id in &clause_ids[(i + 1)..] {
                if self.removed_clauses.contains(&other_id) {
                    continue;
                }

                let other_lits = if let Some(other_clause) = clauses.get(other_id) {
                    if other_clause.deleted || other_clause.learned {
                        continue;
                    }
                    &other_clause.lits
                } else {
                    continue;
                };

                // Check if clause_lits ⊆ other_lits
                if clause_lits.iter().all(|lit| other_lits.contains(lit)) {
                    // clause subsumes other - remove other
                    if let Some(other_clause) = clauses.get_mut(other_id) {
                        other_clause.deleted = true;
                    }
                    self.removed_clauses.insert(other_id);
                    eliminated += 1;
                }
            }
        }

        eliminated
    }

    /// Variable Elimination (Bounded Variable Elimination)
    ///
    /// Eliminate a variable by resolving all pairs of clauses containing v and ~v,
    /// but only if the number of resulting clauses is not too large.
    /// Returns the number of variables eliminated.
    #[allow(dead_code)]
    pub fn variable_elimination(&mut self, clauses: &mut ClauseDatabase, limit: usize) -> usize {
        let mut eliminated = 0;
        self.build_occurrences(clauses);

        for v in 0..self.num_vars {
            let var = Var(v as u32);
            if self.eliminated.contains(&var) {
                continue;
            }

            let pos_lit = Lit::pos(var);
            let neg_lit = Lit::neg(var);

            let pos_clauses: Vec<_> = self.occurrences.get(pos_lit).to_vec();
            let neg_clauses: Vec<_> = self.occurrences.get(neg_lit).to_vec();

            // Bound check: only eliminate if cost is reasonable
            let resolvents = pos_clauses.len() * neg_clauses.len();
            let current = pos_clauses.len() + neg_clauses.len();

            if resolvents > limit || resolvents > current {
                continue;
            }

            // Generate all resolvents
            let mut new_clauses = Vec::new();

            for &pos_clause_id in &pos_clauses {
                for &neg_clause_id in &neg_clauses {
                    let pos_lits = if let Some(c) = clauses.get(pos_clause_id) {
                        &c.lits
                    } else {
                        continue;
                    };

                    let neg_lits = if let Some(c) = clauses.get(neg_clause_id) {
                        &c.lits
                    } else {
                        continue;
                    };

                    // Compute resolvent
                    let mut resolvent = Vec::new();

                    for &lit in pos_lits {
                        if lit != pos_lit {
                            resolvent.push(lit);
                        }
                    }

                    for &lit in neg_lits {
                        if lit != neg_lit && !resolvent.contains(&lit) {
                            resolvent.push(lit);
                        }
                    }

                    // Check if resolvent is tautology
                    if Self::is_tautology(&resolvent) {
                        continue;
                    }

                    new_clauses.push(resolvent);
                }
            }

            // Eliminate the variable
            self.eliminated.insert(var);
            eliminated += 1;

            // Remove old clauses
            for &clause_id in &pos_clauses {
                if let Some(clause) = clauses.get_mut(clause_id) {
                    clause.deleted = true;
                }
                self.removed_clauses.insert(clause_id);
            }

            for &clause_id in &neg_clauses {
                if let Some(clause) = clauses.get_mut(clause_id) {
                    clause.deleted = true;
                }
                self.removed_clauses.insert(clause_id);
            }

            // Add new clauses
            for resolvent in new_clauses {
                if !resolvent.is_empty() {
                    clauses.add_original(resolvent);
                }
            }
        }

        eliminated
    }

    /// Failed Literal Probing
    ///
    /// Try to assign each literal and propagate. If a literal leads to a conflict,
    /// we can infer its negation must be true (failed literal).
    /// Returns the number of failed literals found.
    pub fn failed_literal_probing(&mut self, clauses: &mut ClauseDatabase) -> usize {
        use crate::trail::Trail;
        use crate::watched::{WatchLists, Watcher};

        let mut found = 0;
        self.build_occurrences(clauses);

        // Create temporary trail and watch lists for probing
        let mut trail = Trail::new(self.num_vars);
        let mut watches = WatchLists::new(self.num_vars);

        // Build watch lists from current clauses
        for i in 0..clauses.len() {
            let clause_id = ClauseId::new(i as u32);
            if let Some(clause) = clauses.get(clause_id) {
                if clause.deleted || clause.lits.len() < 2 {
                    continue;
                }

                let lit0 = clause.lits[0];
                let lit1 = clause.lits[1];
                watches.add(lit0.negate(), Watcher::new(clause_id, lit1));
                watches.add(lit1.negate(), Watcher::new(clause_id, lit0));
            }
        }

        // Try to probe each literal
        let mut failed_literals = Vec::new();

        for v in 0..self.num_vars {
            let var = Var(v as u32);
            if trail.is_assigned(var) {
                continue;
            }

            for &polarity in &[false, true] {
                let probe_lit = if polarity {
                    Lit::pos(var)
                } else {
                    Lit::neg(var)
                };

                // Save trail state
                let saved_level = trail.decision_level();

                // Try to assign the literal
                trail.new_decision_level();
                trail.assign_decision(probe_lit);

                // Propagate and check for conflict
                let conflict = self.propagate_probe(&mut trail, &watches, clauses);

                // Backtrack
                trail.backtrack_to(saved_level);

                if conflict {
                    // Found a failed literal! Add its negation as a unit clause
                    failed_literals.push(probe_lit.negate());
                    found += 1;
                    break;
                }
            }
        }

        // Add all failed literals as unit clauses
        for lit in failed_literals {
            clauses.add_original([lit]);
        }

        found
    }

    /// Helper for propagating during probing
    fn propagate_probe(
        &self,
        trail: &mut crate::trail::Trail,
        watches: &crate::watched::WatchLists,
        clauses: &ClauseDatabase,
    ) -> bool {
        use crate::literal::LBool;

        while let Some(lit) = trail.next_to_propagate() {
            let watch_list = watches.get(lit);

            for &watcher in watch_list {
                let clause_id = watcher.clause;
                let blocker = watcher.blocker;

                // Check blocker literal
                if trail.lit_value(blocker) == LBool::True {
                    continue;
                }

                let clause = match clauses.get(clause_id) {
                    Some(c) if !c.deleted => c,
                    _ => continue,
                };

                // Find the two watched literals
                let mut first = clause.lits[0];
                let mut second = clause.lits[1];

                if first == lit.negate() {
                    std::mem::swap(&mut first, &mut second);
                }

                // Try to find a new watch
                let mut found_new_watch = false;
                for &other_lit in &clause.lits[2..] {
                    if trail.lit_value(other_lit) != LBool::False {
                        // Found a new watch - would need to update watches but we're read-only
                        found_new_watch = true;
                        break;
                    }
                }

                if !found_new_watch {
                    // Check if other watch is false (conflict)
                    if trail.lit_value(first) == LBool::False {
                        return true; // Conflict found
                    }

                    // Unit propagation
                    if !trail.is_assigned(first.var()) {
                        trail.assign_propagation(first, clause_id);
                    }
                }
            }
        }

        false // No conflict
    }

    /// Bounded Variable Addition (BVA)
    ///
    /// Introduce new variables to simplify formulas by factoring out common literals.
    /// For example, if we have clauses (a ∨ b ∨ c) and (a ∨ b ∨ d), we can replace them with:
    /// (a ∨ b ∨ x), (~x ∨ c), (~x ∨ d)
    /// where x is a fresh variable. This can reduce total clause size and improve solving.
    ///
    /// Returns the number of variables added.
    #[allow(dead_code)]
    pub fn bounded_variable_addition(
        &mut self,
        clauses: &mut ClauseDatabase,
        max_vars_to_add: usize,
    ) -> usize {
        let mut vars_added = 0;
        self.build_occurrences(clauses);

        // Collect all clause pairs with sufficient overlap
        let clause_ids: Vec<_> = (0..clauses.len())
            .map(|i| ClauseId::new(i as u32))
            .filter(|&id| {
                clauses
                    .get(id)
                    .is_some_and(|c| !c.deleted && !c.learned && c.lits.len() >= 3)
            })
            .collect();

        for i in 0..clause_ids.len() {
            if vars_added >= max_vars_to_add {
                break;
            }

            let clause1_id = clause_ids[i];
            let clause1_lits = match clauses.get(clause1_id) {
                Some(c) if !c.deleted => c.lits.clone(),
                _ => continue,
            };

            for &clause2_id in &clause_ids[(i + 1)..] {
                if vars_added >= max_vars_to_add {
                    break;
                }

                let clause2_lits = match clauses.get(clause2_id) {
                    Some(c) if !c.deleted => c.lits.clone(),
                    _ => continue,
                };

                // Find common literals
                let common: Vec<Lit> = clause1_lits
                    .iter()
                    .filter(|&lit| clause2_lits.contains(lit))
                    .copied()
                    .collect();

                // Only apply BVA if we have at least 2 common literals
                if common.len() < 2 {
                    continue;
                }

                // Check if it's beneficial (reduces total clause size)
                let unique1: Vec<Lit> = clause1_lits
                    .iter()
                    .filter(|lit| !common.contains(lit))
                    .copied()
                    .collect();

                let unique2: Vec<Lit> = clause2_lits
                    .iter()
                    .filter(|lit| !common.contains(lit))
                    .copied()
                    .collect();

                // Original size: |clause1| + |clause2|
                let original_size = clause1_lits.len() + clause2_lits.len();

                // New size: |common| + 1 + |unique1| + 1 + |unique2| + 1
                // = |common| + |unique1| + |unique2| + 3
                let new_size = common.len() + unique1.len() + unique2.len() + 3;

                // Only add variable if it reduces total size
                if new_size >= original_size {
                    continue;
                }

                // Create a new variable
                let new_var = Var::new((self.num_vars + vars_added) as u32);
                let new_lit = Lit::pos(new_var);

                // Create new clauses:
                // 1. (common literals) ∨ new_var
                // 2. ~new_var ∨ (unique literals from clause1)
                // 3. ~new_var ∨ (unique literals from clause2)

                let mut new_clause1 = common.clone();
                new_clause1.push(new_lit);

                let mut new_clause2 = vec![new_lit.negate()];
                new_clause2.extend(&unique1);

                let mut new_clause3 = vec![new_lit.negate()];
                new_clause3.extend(&unique2);

                // Remove old clauses
                if let Some(c) = clauses.get_mut(clause1_id) {
                    c.deleted = true;
                }
                if let Some(c) = clauses.get_mut(clause2_id) {
                    c.deleted = true;
                }
                self.removed_clauses.insert(clause1_id);
                self.removed_clauses.insert(clause2_id);

                // Add new clauses
                if !new_clause1.is_empty() {
                    clauses.add_original(new_clause1);
                }
                if !new_clause2.is_empty() {
                    clauses.add_original(new_clause2);
                }
                if !new_clause3.is_empty() {
                    clauses.add_original(new_clause3);
                }

                vars_added += 1;
                break; // Process next clause pair
            }
        }

        // Update num_vars to reflect new variables
        self.num_vars += vars_added;

        vars_added
    }

    /// Apply all preprocessing techniques
    ///
    /// Returns (clauses_eliminated, vars_eliminated)
    pub fn preprocess(&mut self, clauses: &mut ClauseDatabase) -> (usize, usize) {
        let mut total_clauses = 0;
        let total_vars = 0;

        // Iteratively apply preprocessing until fixpoint
        loop {
            let mut changed = false;

            // Pure literal elimination
            let pure_elim = self.pure_literal_elimination(clauses);
            if pure_elim > 0 {
                total_clauses += pure_elim;
                changed = true;
            }

            // Subsumption elimination
            let subsumption = self.subsumption_elimination(clauses);
            if subsumption > 0 {
                total_clauses += subsumption;
                changed = true;
            }

            // Blocked clause elimination
            let bce = self.blocked_clause_elimination(clauses);
            if bce > 0 {
                total_clauses += bce;
                changed = true;
            }

            if !changed {
                break;
            }
        }

        (total_clauses, total_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tautology_detection() {
        let v0 = Var(0);
        let l0 = Lit::pos(v0);
        let l0_neg = Lit::neg(v0);

        // Tautology: x0 ∨ ~x0
        assert!(Preprocessor::is_tautology(&[l0, l0_neg]));

        // Not a tautology: x0 ∨ x0
        assert!(!Preprocessor::is_tautology(&[l0, l0]));
    }

    #[test]
    fn test_pure_literal() {
        let mut clauses = ClauseDatabase::new();
        let v0 = Var(0);
        let v1 = Var(1);

        // Add clauses: (x0 ∨ x1), (x0)
        // x0 and x1 are pure (only positive)
        clauses.add_original([Lit::pos(v0), Lit::pos(v1)]);
        clauses.add_original([Lit::pos(v0)]);

        let mut prep = Preprocessor::new(2);
        let eliminated = prep.pure_literal_elimination(&mut clauses);

        assert_eq!(eliminated, 2);
    }

    #[test]
    fn test_subsumption() {
        let mut clauses = ClauseDatabase::new();
        let v0 = Var(0);
        let v1 = Var(1);

        // Add clauses: (x0), (x0 ∨ x1)
        // First clause subsumes second
        clauses.add_original([Lit::pos(v0)]);
        clauses.add_original([Lit::pos(v0), Lit::pos(v1)]);

        let mut prep = Preprocessor::new(2);
        let eliminated = prep.subsumption_elimination(&mut clauses);

        assert_eq!(eliminated, 1);
    }
}
