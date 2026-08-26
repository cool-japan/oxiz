//! Clause representation and database

use crate::literal::Lit;
#[allow(unused_imports)]
use crate::prelude::*;
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
    /// Activity for clause deletion heuristic
    pub activity: f64,
    /// Whether this is a learned clause
    pub learned: bool,
    /// LBD (Literal Block Distance) for quality metric
    pub lbd: u32,
    /// Keep hot metadata at the front of the struct so propagation and clause
    /// management usually touch a single cache line before reading literals.
    pub deleted: bool,
    /// The literals in this clause
    ///
    /// Inline capacity could be raised from 4 to 8 literals for free (an 8+32=40B
    /// `SmallVec` pushes the payload to ~59B, still 64B total once padded) — left
    /// at 4 pending a measured decision; do not change this without one.
    pub lits: SmallVec<[Lit; 4]>,
    /// Tier for tiered database management (only used for learned clauses)
    pub tier: ClauseTier,
    /// Number of times this clause was used in conflict analysis (for tier promotion)
    pub usage_count: u32,
}

// Compile-time size guard: `#[repr(align(64))]`
// only forces the size to a *multiple* of 64 bytes, not a cap, so a future field could
// silently push `Clause` past one cache line into two. `ClauseDatabase` stores `Vec<Clause>`,
// so every slot pays the full stride these asserts pin at exactly 64 bytes.
const _: () = assert!(
    core::mem::size_of::<Clause>() == 64,
    "Clause must stay exactly one cache line; a new field pushed it past 64 bytes"
);
const _: () = assert!(core::mem::align_of::<Clause>() == 64);

impl Clause {
    /// Create a new clause
    #[must_use]
    pub fn new(lits: impl IntoIterator<Item = Lit>, learned: bool) -> Self {
        Self {
            activity: 0.0,
            learned,
            lbd: 0,
            deleted: false,
            lits: lits.into_iter().collect(),
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

    /// Assign this clause's tier from its LBD at the moment it is learned,
    /// rather than waiting for [`Self::record_usage`] to promote it later.
    ///
    /// A freshly-learned glue-2 clause otherwise starts in `Local` and is
    /// exposed to the aggressive per-cycle `Local` sweep before it has had a
    /// chance to be reused even once. On formulas whose short low-glue
    /// clauses are load-bearing from the moment they are derived (dense
    /// propagation cascades), losing them to an early sweep measurably hurts
    /// — so a clause this good is protected immediately instead of on
    /// probation. The assignment only ever moves a clause to a *more*
    /// protected tier (lower [`ClauseTier`] discriminant): it never
    /// undoes a promotion `record_usage`/`promote_to_core` already granted.
    pub fn assign_tier_from_lbd(&mut self) {
        let by_lbd = if self.lbd <= 2 {
            ClauseTier::Core
        } else if self.lbd <= 6 {
            ClauseTier::Mid
        } else {
            ClauseTier::Local
        };
        if (by_lbd as u8) <= (self.tier as u8) {
            self.tier = by_lbd;
        }
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
    ///
    /// Concretely: if `self` contains `¬l`, `other` contains `l`, and
    /// `self \ {¬l} ⊆ other \ {l}`, then resolving the two on that variable
    /// yields `other \ {l}` — a clause subsuming `other` — so `l` is returned
    /// and may be dropped from `other`.
    ///
    /// Unlike [`Clause::subsumes`] this makes no assumption about literal
    /// order (clause literals are permuted in place by watch selection, so a
    /// sorted representation cannot be relied on).
    ///
    /// The length guard rejects only a *strictly longer* `self`: equal lengths
    /// are the single most common self-subsumption shape in practice
    /// (`(a ∨ b ∨ c)` against `(a ∨ b ∨ ¬c)`, which strengthens one of them to
    /// `(a ∨ b)`), and rejecting them — as this did until the pass in
    /// `solver/self_subsumption.rs` was wired up and found the rule firing on
    /// nothing — silently discards most of the technique's value. A strictly
    /// longer `self` cannot succeed anyway: the match count below could then
    /// never reach `self.lits.len() - 1`, so the guard is a shortcut, not a
    /// semantic restriction.
    #[must_use]
    pub fn self_subsuming_resolvent(&self, other: &Clause) -> Option<Lit> {
        if self.lits.len() > other.lits.len() {
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
    /// # Soundness: why removed slots are NOT recycled
    ///
    /// `remove` only *marks* a clause deleted and relies on **lazy** watcher
    /// cleanup: stale watch-list entries pointing at a removed clause are detached
    /// on-the-fly during propagation (via the `deleted` flag), not eagerly when the
    /// clause is removed. If we reused a freed `ClauseId` for a *different* clause,
    /// those not-yet-cleaned stale watchers would suddenly reference a live but
    /// unrelated clause. Propagation does not re-validate that the watched literal
    /// belongs to the clause, so it could force a bogus unit propagation (or corrupt
    /// the real watchers' positions via its in-place swaps) — an unsound result that
    /// can flip SAT instances to UNSAT.
    ///
    /// Until a full watch-list garbage-collection pass exists (which would rewrite
    /// every watcher for a relocated clause), we therefore always allocate a fresh
    /// slot, guaranteeing a `ClauseId` maps to exactly one clause for its entire
    /// lifetime. Freed slots stay marked `deleted` and are reclaimed only by their
    /// lazy watcher-cleanup path. `free_list` is retained for stats/compaction and a
    /// future GC, but is intentionally never popped here.
    pub fn add(&mut self, clause: Clause) -> ClauseId {
        // Update statistics
        self.update_stats_add(&clause);

        // Always allocate a new slot (see soundness note above — no free_list reuse).
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

    /// Multiply every live clause's activity by `factor`, preserving their
    /// relative order.
    ///
    /// The clause-activity bump increment used by the solver's clause-decay
    /// step grows geometrically (it is divided by the decay factor on every
    /// conflict), which is the standard trick for making "bump" cheap without
    /// touching every clause — but left unchecked it eventually approaches
    /// `f64::MAX`. This is the paired rescue: shrink every activity (and the
    /// increment, by the caller) by a constant factor before that happens, an
    /// O(n) pass that is amortized away by firing only once every ~100k+
    /// conflicts rather than on every single one.
    pub fn rescale_activity(&mut self, factor: f64) {
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

    // Regression: equal-length clauses differing in exactly one polarity are
    // the most common self-subsumption shape, and the length guard used to
    // reject them outright — which made the whole rule fire on almost nothing.
    #[test]
    fn test_issue36_self_subsuming_resolvent_accepts_equal_lengths() {
        let (a, b, c) = (Var::new(0), Var::new(1), Var::new(2));
        let c1 = Clause::original([Lit::pos(a), Lit::pos(b), Lit::pos(c)]);
        let c2 = Clause::original([Lit::pos(a), Lit::pos(b), Lit::neg(c)]);

        assert_eq!(
            c1.self_subsuming_resolvent(&c2),
            Some(Lit::neg(c)),
            "(a ∨ b ∨ c) must strengthen (a ∨ b ∨ ¬c) to (a ∨ b)"
        );
        assert_eq!(
            c2.self_subsuming_resolvent(&c1),
            Some(Lit::pos(c)),
            "the relation is symmetric at equal length"
        );
    }

    // A strictly longer strengthener can never succeed; the guard rejecting it
    // is a shortcut, so the answer must be `None` either way.
    #[test]
    fn test_issue36_self_subsuming_resolvent_rejects_longer_strengthener() {
        let (a, b, c) = (Var::new(0), Var::new(1), Var::new(2));
        let longer = Clause::original([Lit::pos(a), Lit::pos(b), Lit::neg(c)]);
        let shorter = Clause::original([Lit::pos(a), Lit::pos(c)]);
        assert_eq!(longer.self_subsuming_resolvent(&shorter), None);
    }

    #[test]
    fn test_pr26_assign_tier_from_lbd_promotes_glue_clauses_immediately() {
        let mut c = Clause::learned([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        assert_eq!(c.tier, ClauseTier::Local);
        c.lbd = 2;
        c.assign_tier_from_lbd();
        assert_eq!(
            c.tier,
            ClauseTier::Core,
            "glue <= 2 must be protected immediately, not left to record_usage"
        );
    }

    #[test]
    fn test_pr26_assign_tier_from_lbd_mid_range() {
        let mut c = Clause::learned([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        c.lbd = 5;
        c.assign_tier_from_lbd();
        assert_eq!(c.tier, ClauseTier::Mid);
    }

    #[test]
    fn test_pr26_assign_tier_from_lbd_high_glue_stays_local() {
        let mut c = Clause::learned([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        c.lbd = 20;
        c.assign_tier_from_lbd();
        assert_eq!(c.tier, ClauseTier::Local);
    }

    #[test]
    fn test_pr26_assign_tier_from_lbd_never_demotes() {
        // A clause already promoted to Core by usage must not be pulled back
        // down to Mid/Local by a later, worse LBD-based assignment.
        let mut c = Clause::learned([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        c.promote_to_core();
        c.lbd = 50; // would map to Local on its own
        c.assign_tier_from_lbd();
        assert_eq!(
            c.tier,
            ClauseTier::Core,
            "assign_tier_from_lbd must be one-way upward only"
        );
    }

    #[test]
    fn test_pr26_rescale_activity_preserves_relative_order() {
        let mut db = ClauseDatabase::new();
        let a = db.add_learned([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        let b = db.add_learned([Lit::pos(Var::new(2)), Lit::pos(Var::new(3))]);
        if let Some(clause) = db.get_mut(a) {
            clause.activity = 10.0;
        }
        if let Some(clause) = db.get_mut(b) {
            clause.activity = 20.0;
        }
        db.rescale_activity(1e-10);
        let act_a = db.get(a).expect("clause a").activity;
        let act_b = db.get(b).expect("clause b").activity;
        assert!(act_a < act_b, "relative order must survive a rescale");
        assert!(act_a > 0.0 && act_a.is_finite());
        assert!(act_b > 0.0 && act_b.is_finite());
    }
}
