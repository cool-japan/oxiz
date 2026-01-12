//! Clause representation and database

use crate::literal::Lit;
use smallvec::SmallVec;

/// Unique identifier for a clause
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClauseId(pub u32);

impl ClauseId {
    /// The null clause ID (indicates no clause)
    pub const NULL: Self = Self(u32::MAX);

    /// Create a new clause ID
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Check if this is a null ID
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }

    /// Get the raw index
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Clause tier for tiered database management
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ClauseTier {
    /// Tier 3: Local clauses (recently learned, deleted aggressively)
    Local = 3,
    /// Tier 2: Mid-tier clauses (useful but not essential, deleted conservatively)
    Mid = 2,
    /// Tier 1: Core/GLUE clauses (very high quality, rarely deleted)
    Core = 1,
}

/// A clause is a disjunction of literals
///
/// Cache-line aligned for better memory performance
#[derive(Debug, Clone)]
#[repr(align(64))]
pub struct Clause {
    /// The literals in this clause
    pub lits: SmallVec<[Lit; 4]>,
    /// Whether this is a learned clause
    pub learned: bool,
    /// Activity for clause deletion heuristic
    pub activity: f64,
    /// LBD (Literal Block Distance) for quality metric
    pub lbd: u32,
    /// Whether this clause is marked for deletion
    pub deleted: bool,
    /// Tier for tiered database management (only used for learned clauses)
    pub tier: ClauseTier,
    /// Number of times this clause was used in conflict analysis (for tier promotion)
    pub usage_count: u32,
}

impl Clause {
    /// Create a new clause
    #[must_use]
    pub fn new(lits: impl IntoIterator<Item = Lit>, learned: bool) -> Self {
        Self {
            lits: lits.into_iter().collect(),
            learned,
            activity: 0.0,
            lbd: 0,
            deleted: false,
            tier: ClauseTier::Local, // All learned clauses start in Local tier
            usage_count: 0,
        }
    }

    /// Create an original (non-learned) clause
    #[must_use]
    pub fn original(lits: impl IntoIterator<Item = Lit>) -> Self {
        Self::new(lits, false)
    }

    /// Create a learned clause
    #[must_use]
    pub fn learned(lits: impl IntoIterator<Item = Lit>) -> Self {
        Self::new(lits, true)
    }

    /// Get the number of literals
    #[must_use]
    pub fn len(&self) -> usize {
        self.lits.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lits.is_empty()
    }

    /// Check if this is a unit clause
    #[must_use]
    pub fn is_unit(&self) -> bool {
        self.lits.len() == 1
    }

    /// Check if this is a binary clause
    #[must_use]
    pub fn is_binary(&self) -> bool {
        self.lits.len() == 2
    }

    /// Get the first literal (for unit clauses)
    #[must_use]
    pub fn unit_lit(&self) -> Option<Lit> {
        if self.is_unit() {
            Some(self.lits[0])
        } else {
            None
        }
    }

    /// Swap literals at indices i and j
    pub fn swap(&mut self, i: usize, j: usize) {
        self.lits.swap(i, j);
    }

    /// Increment usage count and potentially promote tier
    pub fn record_usage(&mut self) {
        self.usage_count += 1;

        // Promote to Mid tier after 3 uses
        if self.usage_count >= 3 && self.tier == ClauseTier::Local {
            self.tier = ClauseTier::Mid;
        }
        // Promote to Core tier after 10 uses or if LBD ≤ 2
        else if (self.usage_count >= 10 || self.lbd <= 2) && self.tier == ClauseTier::Mid {
            self.tier = ClauseTier::Core;
        }
    }

    /// Promote clause to Core tier (for GLUE clauses)
    pub fn promote_to_core(&mut self) {
        self.tier = ClauseTier::Core;
    }

    /// Normalize clause: remove duplicates, sort literals, check for tautology
    /// Returns true if clause is a tautology (contains both l and ~l)
    pub fn normalize(&mut self) -> bool {
        if self.lits.is_empty() {
            return false;
        }

        // Sort literals for better cache locality and faster operations
        self.lits.sort_unstable_by_key(|lit| lit.code());

        // Remove duplicates and check for tautology in a single pass
        let mut write_idx = 0;
        let mut prev_lit = self.lits[0];

        for read_idx in 1..self.lits.len() {
            let curr_lit = self.lits[read_idx];

            // Check for tautology (complementary literals)
            if curr_lit == prev_lit.negate() {
                return true;
            }

            // Skip duplicates
            if curr_lit != prev_lit {
                write_idx += 1;
                self.lits[write_idx] = curr_lit;
                prev_lit = curr_lit;
            }
        }

        // Truncate to remove duplicates
        self.lits.truncate(write_idx + 1);
        false
    }

    /// Check if this clause subsumes another clause
    /// A clause C subsumes D if C ⊆ D (all literals of C are in D)
    #[must_use]
    pub fn subsumes(&self, other: &Clause) -> bool {
        if self.lits.len() > other.lits.len() {
            return false;
        }

        // Both clauses should be sorted for efficient checking
        let mut i = 0;
        let mut j = 0;

        while i < self.lits.len() && j < other.lits.len() {
            if self.lits[i] == other.lits[j] {
                i += 1;
                j += 1;
            } else if self.lits[i].code() < other.lits[j].code() {
                // Literal from self not in other
                return false;
            } else {
                j += 1;
            }
        }

        i == self.lits.len()
    }

    /// Check if this clause is a self-subsuming resolvent of another clause
    /// Returns the literal to remove from other if self-subsumption is possible
    #[must_use]
    pub fn self_subsuming_resolvent(&self, other: &Clause) -> Option<Lit> {
        if self.lits.len() >= other.lits.len() {
            return None;
        }

        let mut diff_lit = None;
        let mut matches = 0;

        for &other_lit in &other.lits {
            if self.lits.contains(&other_lit) {
                matches += 1;
            } else if self.lits.contains(&other_lit.negate()) {
                if diff_lit.is_some() {
                    return None; // More than one difference
                }
                diff_lit = Some(other_lit);
            }
        }

        // Self-subsuming resolution requires exactly one complementary literal
        // and all other literals of self must be in other
        if matches == self.lits.len() - 1 && diff_lit.is_some() {
            diff_lit
        } else {
            None
        }
    }
}

/// Statistics for clause database
#[derive(Debug, Clone, Default)]
pub struct ClauseDatabaseStats {
    /// Number of clauses in each tier
    pub tier_counts: [usize; 3], // [Core, Mid, Local]
    /// Total LBD sum for computing average
    pub total_lbd: u64,
    /// Number of clauses with LBD counted
    pub lbd_count: usize,
    /// Distribution of clause sizes
    pub size_distribution: [usize; 10], // [binary, ternary, 4-lit, ..., 10+]
    /// Number of clause promotions
    pub promotions: usize,
    /// Number of clause demotions
    pub demotions: usize,
}

impl ClauseDatabaseStats {
    /// Get average LBD across all learned clauses
    #[must_use]
    pub fn avg_lbd(&self) -> f64 {
        if self.lbd_count == 0 {
            0.0
        } else {
            self.total_lbd as f64 / self.lbd_count as f64
        }
    }

    /// Display statistics
    pub fn display(&self) {
        println!("Clause Database Statistics:");
        println!("  Tier distribution:");
        println!("    Core:  {}", self.tier_counts[0]);
        println!("    Mid:   {}", self.tier_counts[1]);
        println!("    Local: {}", self.tier_counts[2]);
        println!("  Average LBD: {:.2}", self.avg_lbd());
        println!("  Size distribution:");
        for (i, &count) in self.size_distribution.iter().enumerate() {
            if count > 0 {
                let size = if i < 9 {
                    format!("{}", i + 2)
                } else {
                    "10+".to_string()
                };
                println!("    {} literals: {}", size, count);
            }
        }
        println!(
            "  Promotions: {}, Demotions: {}",
            self.promotions, self.demotions
        );
    }
}

/// Database of clauses with memory pool
#[derive(Debug)]
pub struct ClauseDatabase {
    /// All clauses
    clauses: Vec<Clause>,
    /// Number of original clauses
    num_original: usize,
    /// Number of learned clauses
    num_learned: usize,
    /// Free list for reusing deleted clause slots (memory pool)
    free_list: Vec<ClauseId>,
    /// Statistics
    stats: ClauseDatabaseStats,
}

impl Default for ClauseDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl ClauseDatabase {
    /// Create a new clause database
    #[must_use]
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            num_original: 0,
            num_learned: 0,
            free_list: Vec::new(),
            stats: ClauseDatabaseStats::default(),
        }
    }

    /// Get statistics about the clause database
    #[must_use]
    pub fn stats(&self) -> &ClauseDatabaseStats {
        &self.stats
    }

    /// Update statistics for a clause
    fn update_stats_add(&mut self, clause: &Clause) {
        if clause.learned {
            // Update tier count
            let tier_idx = match clause.tier {
                ClauseTier::Core => 0,
                ClauseTier::Mid => 1,
                ClauseTier::Local => 2,
            };
            self.stats.tier_counts[tier_idx] += 1;

            // Update LBD stats
            if clause.lbd > 0 {
                self.stats.total_lbd += clause.lbd as u64;
                self.stats.lbd_count += 1;
            }
        }

        // Update size distribution (only for clauses with 2+ literals)
        if clause.len() >= 2 {
            let size_idx = if clause.len() >= 12 {
                9 // 10+ bucket
            } else {
                clause.len() - 2
            };
            self.stats.size_distribution[size_idx] += 1;
        }
    }

    /// Update statistics when removing a clause
    fn update_stats_remove(&mut self, clause: &Clause) {
        if clause.learned {
            // Update tier count
            let tier_idx = match clause.tier {
                ClauseTier::Core => 0,
                ClauseTier::Mid => 1,
                ClauseTier::Local => 2,
            };
            if self.stats.tier_counts[tier_idx] > 0 {
                self.stats.tier_counts[tier_idx] -= 1;
            }

            // Update LBD stats
            if clause.lbd > 0 && self.stats.lbd_count > 0 {
                self.stats.total_lbd = self.stats.total_lbd.saturating_sub(clause.lbd as u64);
                self.stats.lbd_count -= 1;
            }
        }

        // Update size distribution (only for clauses with 2+ literals)
        if clause.len() >= 2 {
            let size_idx = if clause.len() >= 12 {
                9 // 10+ bucket
            } else {
                clause.len() - 2
            };
            if self.stats.size_distribution[size_idx] > 0 {
                self.stats.size_distribution[size_idx] -= 1;
            }
        }
    }

    /// Add a clause to the database
    ///
    /// Uses the memory pool (free list) to reuse deleted clause slots when available
    pub fn add(&mut self, clause: Clause) -> ClauseId {
        // Update statistics
        self.update_stats_add(&clause);

        // Try to reuse a slot from the free list
        if let Some(id) = self.free_list.pop() {
            // Reuse this slot
            if let Some(slot) = self.clauses.get_mut(id.index()) {
                *slot = clause.clone();
                if clause.learned {
                    self.num_learned += 1;
                } else {
                    self.num_original += 1;
                }
                return id;
            }
        }

        // No free slot available, allocate new
        let id = ClauseId::new(self.clauses.len() as u32);
        if clause.learned {
            self.num_learned += 1;
        } else {
            self.num_original += 1;
        }
        self.clauses.push(clause);
        id
    }

    /// Add an original clause
    pub fn add_original(&mut self, lits: impl IntoIterator<Item = Lit>) -> ClauseId {
        self.add(Clause::original(lits))
    }

    /// Add a learned clause
    pub fn add_learned(&mut self, lits: impl IntoIterator<Item = Lit>) -> ClauseId {
        self.add(Clause::learned(lits))
    }

    /// Get a clause by ID
    #[must_use]
    pub fn get(&self, id: ClauseId) -> Option<&Clause> {
        self.clauses.get(id.index())
    }

    /// Get a mutable reference to a clause
    pub fn get_mut(&mut self, id: ClauseId) -> Option<&mut Clause> {
        self.clauses.get_mut(id.index())
    }

    /// Mark a clause as deleted
    ///
    /// The deleted clause slot is added to the free list for reuse (memory pool)
    pub fn remove(&mut self, id: ClauseId) {
        if let Some(clause) = self.clauses.get_mut(id.index())
            && !clause.deleted
        {
            // Clone necessary info for stats update
            let clause_copy = clause.clone();

            clause.deleted = true;
            if clause.learned {
                self.num_learned -= 1;
            } else {
                self.num_original -= 1;
            }
            // Add to free list for reuse
            self.free_list.push(id);

            // Update statistics after marking as deleted
            self.update_stats_remove(&clause_copy);
        }
    }

    /// Compact the database by removing deleted clauses from the free list
    ///
    /// This should be called periodically to prevent the free list from growing too large
    pub fn compact(&mut self) {
        // Limit free list size to avoid memory bloat
        const MAX_FREE_LIST_SIZE: usize = 1000;

        if self.free_list.len() > MAX_FREE_LIST_SIZE {
            // Keep only the most recent freed slots
            self.free_list
                .drain(0..self.free_list.len() - MAX_FREE_LIST_SIZE);
        }
    }

    /// Get the number of active clauses
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_original + self.num_learned
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of original clauses
    #[must_use]
    pub fn num_original(&self) -> usize {
        self.num_original
    }

    /// Get the number of learned clauses
    #[must_use]
    pub fn num_learned(&self) -> usize {
        self.num_learned
    }

    /// Iterate over all non-deleted clause IDs
    pub fn iter_ids(&self) -> impl Iterator<Item = ClauseId> + '_ {
        self.clauses
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.deleted)
            .map(|(i, _)| ClauseId::new(i as u32))
    }

    /// Bump activity of a clause
    pub fn bump_activity(&mut self, id: ClauseId, increment: f64) {
        if let Some(clause) = self.get_mut(id) {
            clause.activity += increment;
        }
    }

    /// Decay all clause activities
    pub fn decay_activity(&mut self, factor: f64) {
        for clause in &mut self.clauses {
            if !clause.deleted {
                clause.activity *= factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_clause_creation() {
        let lits = vec![
            Lit::pos(Var::new(0)),
            Lit::neg(Var::new(1)),
            Lit::pos(Var::new(2)),
        ];
        let clause = Clause::original(lits.clone());

        assert_eq!(clause.len(), 3);
        assert!(!clause.is_unit());
        assert!(!clause.is_binary());
        assert!(!clause.learned);
    }

    #[test]
    fn test_clause_database() {
        let mut db = ClauseDatabase::new();

        let c1 = db.add_original([Lit::pos(Var::new(0)), Lit::neg(Var::new(1))]);
        let _c2 = db.add_learned([Lit::pos(Var::new(2))]);

        assert_eq!(db.len(), 2);
        assert_eq!(db.num_original(), 1);
        assert_eq!(db.num_learned(), 1);

        db.remove(c1);
        assert_eq!(db.len(), 1);
        assert_eq!(db.num_original(), 0);
    }

    #[test]
    fn test_clause_normalize() {
        let mut clause = Clause::original([
            Lit::pos(Var::new(2)),
            Lit::pos(Var::new(0)),
            Lit::pos(Var::new(2)), // duplicate
            Lit::pos(Var::new(1)),
        ]);

        let is_tautology = clause.normalize();
        assert!(!is_tautology);
        assert_eq!(clause.len(), 3); // duplicate removed
        // Check sorted order
        assert_eq!(clause.lits[0], Lit::pos(Var::new(0)));
        assert_eq!(clause.lits[1], Lit::pos(Var::new(1)));
        assert_eq!(clause.lits[2], Lit::pos(Var::new(2)));
    }

    #[test]
    fn test_clause_normalize_tautology() {
        let mut clause = Clause::original([
            Lit::pos(Var::new(0)),
            Lit::neg(Var::new(0)), // tautology
            Lit::pos(Var::new(1)),
        ]);

        let is_tautology = clause.normalize();
        assert!(is_tautology);
    }

    #[test]
    fn test_clause_subsumes() {
        let mut c1 = Clause::original([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        let mut c2 = Clause::original([
            Lit::pos(Var::new(0)),
            Lit::pos(Var::new(1)),
            Lit::pos(Var::new(2)),
        ]);

        c1.normalize();
        c2.normalize();

        assert!(c1.subsumes(&c2)); // c1 ⊆ c2
        assert!(!c2.subsumes(&c1)); // c2 ⊈ c1
    }

    #[test]
    fn test_clause_self_subsuming_resolvent() {
        // C1: (a v b), C2: (~a v b v c)
        // C1 can strengthen C2 to (b v c) by removing ~a
        let mut c1 = Clause::original([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        let mut c2 = Clause::original([
            Lit::neg(Var::new(0)),
            Lit::pos(Var::new(1)),
            Lit::pos(Var::new(2)),
        ]);

        c1.normalize();
        c2.normalize();

        if let Some(lit_to_remove) = c1.self_subsuming_resolvent(&c2) {
            assert_eq!(lit_to_remove, Lit::neg(Var::new(0)));
        } else {
            panic!("Expected self-subsuming resolvent");
        }
    }
}
