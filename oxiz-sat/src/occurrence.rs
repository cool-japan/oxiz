//! Literal occurrence lists for efficient clause operations
//!
//! This module tracks which clauses contain each literal, enabling fast
//! operations like:
//! - Finding all clauses containing a literal
//! - Subsumption checking
//! - Variable elimination
//! - Resolution
//!
//! References:
//! - "Effective Preprocessing in SAT Through Variable and Clause Elimination"
//! - "Bounded Variable Elimination in Simple Sat Solvers"

use crate::clause::ClauseId;
use crate::literal::Lit;
use smallvec::SmallVec;

/// Statistics for occurrence tracking
#[derive(Debug, Clone, Default)]
pub struct OccurrenceStats {
    /// Total number of occurrences tracked
    pub total_occurrences: usize,
    /// Number of literals with occurrences
    pub active_literals: usize,
    /// Maximum occurrences for a single literal
    pub max_occurrences: usize,
    /// Average occurrences per literal
    pub avg_occurrences: f64,
}

impl OccurrenceStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Occurrence List Statistics:");
        println!("  Total occurrences: {}", self.total_occurrences);
        println!("  Active literals: {}", self.active_literals);
        println!("  Max occurrences: {}", self.max_occurrences);
        println!("  Avg occurrences: {:.1}", self.avg_occurrences);
    }
}

/// Occurrence list manager
///
/// Maintains for each literal a list of clauses that contain it.
/// This enables O(1) lookup of clauses containing a literal.
#[derive(Debug)]
pub struct OccurrenceList {
    /// occurrences[lit] = list of clause IDs containing lit
    occurrences: Vec<SmallVec<[ClauseId; 4]>>,
    /// Statistics
    stats: OccurrenceStats,
}

impl Default for OccurrenceList {
    fn default() -> Self {
        Self::new()
    }
}

impl OccurrenceList {
    /// Create a new occurrence list manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            occurrences: Vec::new(),
            stats: OccurrenceStats::default(),
        }
    }

    /// Resize for new number of variables
    pub fn resize(&mut self, num_vars: usize) {
        // Each variable has 2 literals (positive and negative)
        self.occurrences.resize(num_vars * 2, SmallVec::new());
    }

    /// Add a clause containing the given literal
    pub fn add(&mut self, lit: Lit, clause_id: ClauseId) {
        let idx = lit.code() as usize;

        // Ensure we have space
        if idx >= self.occurrences.len() {
            self.occurrences.resize(idx + 1, SmallVec::new());
        }

        // Add if not already present
        if !self.occurrences[idx].contains(&clause_id) {
            self.occurrences[idx].push(clause_id);
        }
    }

    /// Remove a clause from a literal's occurrence list
    pub fn remove(&mut self, lit: Lit, clause_id: ClauseId) {
        let idx = lit.code() as usize;

        if idx < self.occurrences.len()
            && let Some(pos) = self.occurrences[idx].iter().position(|&id| id == clause_id)
        {
            self.occurrences[idx].swap_remove(pos);
        }
    }

    /// Get all clauses containing a literal
    #[must_use]
    pub fn get(&self, lit: Lit) -> &[ClauseId] {
        let idx = lit.code() as usize;
        if idx < self.occurrences.len() {
            &self.occurrences[idx]
        } else {
            &[]
        }
    }

    /// Get number of occurrences for a literal
    #[must_use]
    pub fn count(&self, lit: Lit) -> usize {
        self.get(lit).len()
    }

    /// Check if a literal has any occurrences
    #[must_use]
    pub fn has_occurrences(&self, lit: Lit) -> bool {
        !self.get(lit).is_empty()
    }

    /// Get total number of occurrences for a variable (positive + negative)
    #[must_use]
    pub fn var_occurrence_count(&self, var_idx: usize) -> usize {
        if var_idx * 2 + 1 < self.occurrences.len() {
            self.occurrences[var_idx * 2].len() + self.occurrences[var_idx * 2 + 1].len()
        } else {
            0
        }
    }

    /// Clear all occurrence lists for a literal
    pub fn clear_literal(&mut self, lit: Lit) {
        let idx = lit.code() as usize;
        if idx < self.occurrences.len() {
            self.occurrences[idx].clear();
        }
    }

    /// Clear all occurrences
    pub fn clear(&mut self) {
        for list in &mut self.occurrences {
            list.clear();
        }
        self.stats = OccurrenceStats::default();
    }

    /// Update statistics
    pub fn update_stats(&mut self) {
        self.stats.total_occurrences = 0;
        self.stats.active_literals = 0;
        self.stats.max_occurrences = 0;

        for list in &self.occurrences {
            if !list.is_empty() {
                self.stats.active_literals += 1;
                self.stats.total_occurrences += list.len();
                self.stats.max_occurrences = self.stats.max_occurrences.max(list.len());
            }
        }

        if self.stats.active_literals > 0 {
            self.stats.avg_occurrences =
                self.stats.total_occurrences as f64 / self.stats.active_literals as f64;
        } else {
            self.stats.avg_occurrences = 0.0;
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &OccurrenceStats {
        &self.stats
    }

    /// Find clauses that contain all given literals (for resolution)
    ///
    /// Returns clause IDs that appear in all literal occurrence lists
    #[must_use]
    pub fn find_common_clauses(&self, lits: &[Lit]) -> Vec<ClauseId> {
        if lits.is_empty() {
            return Vec::new();
        }

        // Start with first literal's occurrences
        let mut common: Vec<ClauseId> = self.get(lits[0]).to_vec();

        // Intersect with remaining literals
        for &lit in &lits[1..] {
            let occurrences = self.get(lit);
            common.retain(|id| occurrences.contains(id));

            if common.is_empty() {
                break;
            }
        }

        common
    }

    /// Find pure literals (appear only in one polarity)
    ///
    /// Returns list of literals that appear only positive or only negative
    #[must_use]
    pub fn find_pure_literals(&self) -> Vec<Lit> {
        let mut pure = Vec::new();

        // Check pairs of literals (positive and negative)
        for var_idx in 0..self.occurrences.len() / 2 {
            let pos_idx = var_idx * 2;
            let neg_idx = var_idx * 2 + 1;

            let pos_count = self.occurrences[pos_idx].len();
            let neg_count = self.occurrences[neg_idx].len();

            // Reconstruct literal from index
            if pos_count > 0 && neg_count == 0 {
                // Only positive occurrences
                pure.push(Lit::from_code(pos_idx as u32));
            } else if neg_count > 0 && pos_count == 0 {
                // Only negative occurrences
                pure.push(Lit::from_code(neg_idx as u32));
            }
        }

        pure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_occurrence_list_creation() {
        let list = OccurrenceList::new();
        assert_eq!(list.occurrences.len(), 0);
    }

    #[test]
    fn test_add_and_get() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        let lit = Lit::pos(Var::new(0));
        list.add(lit, ClauseId(1));
        list.add(lit, ClauseId(2));

        let occurrences = list.get(lit);
        assert_eq!(occurrences.len(), 2);
        assert!(occurrences.contains(&ClauseId(1)));
        assert!(occurrences.contains(&ClauseId(2)));
    }

    #[test]
    fn test_duplicate_add() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        let lit = Lit::pos(Var::new(0));
        list.add(lit, ClauseId(1));
        list.add(lit, ClauseId(1)); // Duplicate

        assert_eq!(list.count(lit), 1);
    }

    #[test]
    fn test_remove() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        let lit = Lit::pos(Var::new(0));
        list.add(lit, ClauseId(1));
        list.add(lit, ClauseId(2));

        list.remove(lit, ClauseId(1));

        let occurrences = list.get(lit);
        assert_eq!(occurrences.len(), 1);
        assert!(occurrences.contains(&ClauseId(2)));
    }

    #[test]
    fn test_count() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        let lit = Lit::pos(Var::new(0));
        list.add(lit, ClauseId(1));
        list.add(lit, ClauseId(2));
        list.add(lit, ClauseId(3));

        assert_eq!(list.count(lit), 3);
    }

    #[test]
    fn test_has_occurrences() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        let lit = Lit::pos(Var::new(0));
        assert!(!list.has_occurrences(lit));

        list.add(lit, ClauseId(1));
        assert!(list.has_occurrences(lit));
    }

    #[test]
    fn test_var_occurrence_count() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        let pos_lit = Lit::pos(Var::new(0));
        let neg_lit = Lit::neg(Var::new(0));

        list.add(pos_lit, ClauseId(1));
        list.add(pos_lit, ClauseId(2));
        list.add(neg_lit, ClauseId(3));

        assert_eq!(list.var_occurrence_count(0), 3);
    }

    #[test]
    fn test_clear_literal() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        let lit = Lit::pos(Var::new(0));
        list.add(lit, ClauseId(1));
        list.add(lit, ClauseId(2));

        list.clear_literal(lit);

        assert_eq!(list.count(lit), 0);
    }

    #[test]
    fn test_clear() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        list.add(Lit::pos(Var::new(0)), ClauseId(1));
        list.add(Lit::neg(Var::new(1)), ClauseId(2));

        list.clear();

        assert_eq!(list.count(Lit::pos(Var::new(0))), 0);
        assert_eq!(list.count(Lit::neg(Var::new(1))), 0);
    }

    #[test]
    fn test_update_stats() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        list.add(Lit::pos(Var::new(0)), ClauseId(1));
        list.add(Lit::pos(Var::new(0)), ClauseId(2));
        list.add(Lit::neg(Var::new(1)), ClauseId(3));

        list.update_stats();

        let stats = list.stats();
        assert_eq!(stats.total_occurrences, 3);
        assert_eq!(stats.active_literals, 2);
        assert_eq!(stats.max_occurrences, 2);
    }

    #[test]
    fn test_find_common_clauses() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::pos(Var::new(1));

        list.add(lit1, ClauseId(1));
        list.add(lit1, ClauseId(2));
        list.add(lit2, ClauseId(2));
        list.add(lit2, ClauseId(3));

        let common = list.find_common_clauses(&[lit1, lit2]);
        assert_eq!(common.len(), 1);
        assert!(common.contains(&ClauseId(2)));
    }

    #[test]
    fn test_find_pure_literals() {
        let mut list = OccurrenceList::new();
        list.resize(10);

        // Variable 0: only positive occurrences (pure)
        list.add(Lit::pos(Var::new(0)), ClauseId(1));
        list.add(Lit::pos(Var::new(0)), ClauseId(2));

        // Variable 1: both polarities (not pure)
        list.add(Lit::pos(Var::new(1)), ClauseId(3));
        list.add(Lit::neg(Var::new(1)), ClauseId(4));

        // Variable 2: only negative occurrences (pure)
        list.add(Lit::neg(Var::new(2)), ClauseId(5));

        let pure = list.find_pure_literals();
        assert_eq!(pure.len(), 2);
        assert!(pure.contains(&Lit::pos(Var::new(0))));
        assert!(pure.contains(&Lit::neg(Var::new(2))));
    }

    #[test]
    fn test_resize() {
        let mut list = OccurrenceList::new();
        list.resize(5);

        assert!(list.occurrences.len() >= 10); // 5 vars * 2 literals

        list.resize(10);
        assert!(list.occurrences.len() >= 20); // 10 vars * 2 literals
    }
}
