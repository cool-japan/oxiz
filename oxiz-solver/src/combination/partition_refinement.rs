//! Partition Refinement for Theory Combination.
//!
//! This module implements partition refinement algorithms for managing
//! equality arrangements in Nelson-Oppen combination:
//! - Set partition enumeration
//! - Partition refinement with constraints
//! - Efficient arrangement generation
//! - Backtrackable partition data structures
//!
//! ## Partition Refinement
//!
//! Given a set of terms {t1, ..., tn}, we need to enumerate all possible
//! **partitions** (equivalence relations) over these terms. Each partition
//! represents a possible equality arrangement.
//!
//! ## Bell Numbers
//!
//! The number of partitions of n elements is given by the Bell number B(n):
//! - B(1) = 1
//! - B(2) = 2
//! - B(3) = 5
//! - B(4) = 15
//! - B(5) = 52
//!
//! This grows very quickly, so efficient enumeration and pruning is critical.
//!
//! ## Partition Refinement Algorithm
//!
//! Starting from the finest partition (all singletons), we can:
//! 1. Merge classes based on constraints
//! 2. Enumerate coarser partitions
//! 3. Backtrack when conflicts arise
//!
//! ## References
//!
//! - Knuth TAOCP Vol 4A: "Combinatorial Algorithms, Part 1"
//! - Restricted Growth Strings for partition enumeration
//! - Z3's theory combination implementation

#![allow(missing_docs)]
#![allow(dead_code)]

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Term identifier.
pub type TermId = u32;

/// Theory identifier.
pub type TheoryId = u32;

/// Decision level.
pub type DecisionLevel = u32;

/// Class identifier in a partition.
pub type ClassId = usize;

/// Equality between terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Equality {
    /// Left-hand side.
    pub lhs: TermId,
    /// Right-hand side.
    pub rhs: TermId,
}

impl Equality {
    /// Create new equality.
    pub fn new(lhs: TermId, rhs: TermId) -> Self {
        if lhs <= rhs {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }
}

/// Set partition of terms.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Equivalence classes.
    classes: Vec<FxHashSet<TermId>>,

    /// Term to class mapping.
    term_to_class: FxHashMap<TermId, ClassId>,

    /// Representative for each class.
    representatives: Vec<TermId>,
}

impl Partition {
    /// Create finest partition (all singletons).
    pub fn finest(terms: &[TermId]) -> Self {
        let mut classes = Vec::new();
        let mut term_to_class = FxHashMap::default();
        let mut representatives = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            let mut class = FxHashSet::default();
            class.insert(term);
            classes.push(class);
            term_to_class.insert(term, i);
            representatives.push(term);
        }

        Self {
            classes,
            term_to_class,
            representatives,
        }
    }

    /// Create coarsest partition (all terms in one class).
    pub fn coarsest(terms: &[TermId]) -> Self {
        if terms.is_empty() {
            return Self {
                classes: Vec::new(),
                term_to_class: FxHashMap::default(),
                representatives: Vec::new(),
            };
        }

        let mut class = FxHashSet::default();
        let mut term_to_class = FxHashMap::default();

        for &term in terms {
            class.insert(term);
            term_to_class.insert(term, 0);
        }

        Self {
            classes: vec![class],
            term_to_class,
            representatives: vec![terms[0]],
        }
    }

    /// Merge two classes.
    pub fn merge(&mut self, t1: TermId, t2: TermId) -> Result<(), String> {
        let c1 = *self.term_to_class.get(&t1).ok_or("Term not in partition")?;
        let c2 = *self.term_to_class.get(&t2).ok_or("Term not in partition")?;

        if c1 == c2 {
            return Ok(());
        }

        // Merge smaller into larger
        let (src, dst) = if self.classes[c1].len() < self.classes[c2].len() {
            (c1, c2)
        } else {
            (c2, c1)
        };

        // Move all terms from src to dst
        let src_terms: Vec<_> = self.classes[src].iter().copied().collect();
        for term in src_terms {
            self.classes[dst].insert(term);
            self.term_to_class.insert(term, dst);
        }

        // Clear source class
        self.classes[src].clear();

        Ok(())
    }

    /// Get all equalities implied by this partition.
    pub fn get_equalities(&self) -> Vec<Equality> {
        let mut equalities = Vec::new();

        for class in &self.classes {
            if class.len() > 1 {
                let terms: Vec<_> = class.iter().copied().collect();
                // Use star topology: all terms equal to first term
                let rep = terms[0];
                for &term in &terms[1..] {
                    equalities.push(Equality::new(rep, term));
                }
            }
        }

        equalities
    }

    /// Get number of non-empty classes.
    pub fn num_classes(&self) -> usize {
        self.classes.iter().filter(|c| !c.is_empty()).count()
    }

    /// Check if two terms are in the same class.
    pub fn are_equal(&self, t1: TermId, t2: TermId) -> bool {
        if let (Some(&c1), Some(&c2)) = (self.term_to_class.get(&t1), self.term_to_class.get(&t2)) {
            c1 == c2
        } else {
            false
        }
    }

    /// Get representative for a term.
    pub fn get_representative(&self, term: TermId) -> Option<TermId> {
        self.term_to_class
            .get(&term)
            .and_then(|&class_id| self.representatives.get(class_id))
            .copied()
    }

    /// Get all terms in the same class as a term.
    pub fn get_class(&self, term: TermId) -> Option<&FxHashSet<TermId>> {
        self.term_to_class
            .get(&term)
            .and_then(|&class_id| self.classes.get(class_id))
    }

    /// Clone partition.
    pub fn clone_partition(&self) -> Partition {
        self.clone()
    }
}

/// Partition refinement algorithm.
pub struct PartitionRefinement {
    /// Current partition.
    partition: Partition,

    /// Refinement history for backtracking.
    history: Vec<Partition>,

    /// Decision levels.
    decision_levels: Vec<DecisionLevel>,

    /// Current decision level.
    current_level: DecisionLevel,
}

impl PartitionRefinement {
    /// Create new refinement starting from finest partition.
    pub fn new(terms: &[TermId]) -> Self {
        Self {
            partition: Partition::finest(terms),
            history: Vec::new(),
            decision_levels: Vec::new(),
            current_level: 0,
        }
    }

    /// Refine with equality.
    pub fn refine(&mut self, eq: Equality) -> Result<(), String> {
        self.history.push(self.partition.clone_partition());
        self.decision_levels.push(self.current_level);
        self.partition.merge(eq.lhs, eq.rhs)
    }

    /// Refine with multiple equalities.
    pub fn refine_batch(&mut self, equalities: &[Equality]) -> Result<(), String> {
        for &eq in equalities {
            self.refine(eq)?;
        }
        Ok(())
    }

    /// Get current partition.
    pub fn current(&self) -> &Partition {
        &self.partition
    }

    /// Backtrack one step.
    pub fn backtrack_step(&mut self) -> Result<(), String> {
        self.partition = self.history.pop().ok_or("No refinement to backtrack")?;
        self.decision_levels.pop();
        Ok(())
    }

    /// Backtrack to decision level.
    pub fn backtrack(&mut self, level: DecisionLevel) -> Result<(), String> {
        while !self.decision_levels.is_empty() {
            if let Some(&last_level) = self.decision_levels.last() {
                if last_level > level {
                    self.backtrack_step()?;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.current_level = level;
        Ok(())
    }

    /// Push decision level.
    pub fn push_decision_level(&mut self) {
        self.current_level += 1;
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.decision_levels.clear();
    }
}

/// Partition enumerator using Restricted Growth Strings.
pub struct PartitionEnumerator {
    /// Number of elements.
    n: usize,

    /// Terms being partitioned.
    terms: Vec<TermId>,

    /// Current RGS (Restricted Growth String).
    rgs: Vec<usize>,

    /// Maximum value seen so far.
    max_val: usize,

    /// Is enumeration complete?
    done: bool,
}

impl PartitionEnumerator {
    /// Create new enumerator.
    pub fn new(terms: Vec<TermId>) -> Self {
        let n = terms.len();
        Self {
            n,
            terms,
            rgs: vec![0; n],
            max_val: 0,
            done: n == 0,
        }
    }

    /// Get next partition.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Partition> {
        if self.done {
            return None;
        }

        // Build partition from current RGS
        let partition = self.rgs_to_partition();

        // Generate next RGS
        self.next_rgs();

        Some(partition)
    }

    /// Convert RGS to partition.
    fn rgs_to_partition(&self) -> Partition {
        let mut classes: Vec<FxHashSet<TermId>> = vec![FxHashSet::default(); self.max_val + 1];
        let mut term_to_class = FxHashMap::default();
        let mut representatives = vec![0; self.max_val + 1];

        for (i, &class_id) in self.rgs.iter().enumerate() {
            let term = self.terms[i];
            classes[class_id].insert(term);
            term_to_class.insert(term, class_id);

            if representatives[class_id] == 0 || term < representatives[class_id] {
                representatives[class_id] = term;
            }
        }

        Partition {
            classes,
            term_to_class,
            representatives,
        }
    }

    /// Generate next RGS.
    fn next_rgs(&mut self) {
        // Find rightmost position that can be incremented
        let mut i = self.n;
        while i > 0 {
            i -= 1;

            let can_increment = if i == 0 {
                false
            } else {
                let max_up_to_i = self.rgs[..i].iter().max().copied().unwrap_or(0);
                self.rgs[i] <= max_up_to_i
            };

            if can_increment {
                self.rgs[i] += 1;

                // Update max_val
                self.max_val = self.rgs.iter().max().copied().unwrap_or(0);

                // Reset suffix to 0
                for j in (i + 1)..self.n {
                    self.rgs[j] = 0;
                }

                return;
            }
        }

        self.done = true;
    }

    /// Reset enumerator.
    pub fn reset(&mut self) {
        self.rgs = vec![0; self.n];
        self.max_val = 0;
        self.done = self.n == 0;
    }

    /// Get number of remaining partitions (approximate).
    pub fn count_remaining(&self) -> usize {
        // Bell number computation (simplified)
        bell_number(self.n)
    }
}

/// Compute Bell number B(n).
fn bell_number(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    // Use Stirling numbers (simplified for small n)
    match n {
        0 => 1,
        1 => 1,
        2 => 2,
        3 => 5,
        4 => 15,
        5 => 52,
        6 => 203,
        7 => 877,
        8 => 4140,
        _ => usize::MAX, // Too large
    }
}

/// Configuration for partition refinement.
#[derive(Debug, Clone)]
pub struct PartitionRefinementConfig {
    /// Enable partition enumeration.
    pub enable_enumeration: bool,

    /// Maximum partitions to enumerate.
    pub max_partitions: usize,

    /// Enable constraint-guided refinement.
    pub constraint_guided: bool,

    /// Enable backtracking.
    pub enable_backtracking: bool,
}

impl Default for PartitionRefinementConfig {
    fn default() -> Self {
        Self {
            enable_enumeration: true,
            max_partitions: 1000,
            constraint_guided: true,
            enable_backtracking: true,
        }
    }
}

/// Statistics for partition refinement.
#[derive(Debug, Clone, Default)]
pub struct PartitionRefinementStats {
    /// Refinements performed.
    pub refinements: u64,
    /// Partitions enumerated.
    pub partitions_enumerated: u64,
    /// Backtracks.
    pub backtracks: u64,
    /// Constraints applied.
    pub constraints_applied: u64,
}

/// Partition refinement manager.
pub struct PartitionRefinementManager {
    /// Configuration.
    config: PartitionRefinementConfig,

    /// Statistics.
    stats: PartitionRefinementStats,

    /// Refinement algorithm.
    refinement: PartitionRefinement,

    /// Enumerator (if enumeration enabled).
    enumerator: Option<PartitionEnumerator>,

    /// Constraint queue.
    constraints: VecDeque<Equality>,
}

impl PartitionRefinementManager {
    /// Create new manager.
    pub fn new(terms: Vec<TermId>) -> Self {
        Self::with_config(terms, PartitionRefinementConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(terms: Vec<TermId>, config: PartitionRefinementConfig) -> Self {
        let enumerator = if config.enable_enumeration {
            Some(PartitionEnumerator::new(terms.clone()))
        } else {
            None
        };

        Self {
            config,
            stats: PartitionRefinementStats::default(),
            refinement: PartitionRefinement::new(&terms),
            enumerator,
            constraints: VecDeque::new(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &PartitionRefinementStats {
        &self.stats
    }

    /// Add constraint.
    pub fn add_constraint(&mut self, eq: Equality) {
        self.constraints.push_back(eq);
        self.stats.constraints_applied += 1;
    }

    /// Apply constraints and refine.
    pub fn apply_constraints(&mut self) -> Result<(), String> {
        while let Some(eq) = self.constraints.pop_front() {
            self.refinement.refine(eq)?;
            self.stats.refinements += 1;
        }
        Ok(())
    }

    /// Get current partition.
    pub fn current_partition(&self) -> &Partition {
        self.refinement.current()
    }

    /// Get next enumerated partition.
    pub fn next_partition(&mut self) -> Option<Partition> {
        if let Some(ref mut enumerator) = self.enumerator {
            if self.stats.partitions_enumerated >= self.config.max_partitions as u64 {
                return None;
            }

            let partition = enumerator.next();
            if partition.is_some() {
                self.stats.partitions_enumerated += 1;
            }
            partition
        } else {
            None
        }
    }

    /// Backtrack to decision level.
    pub fn backtrack(&mut self, level: DecisionLevel) -> Result<(), String> {
        if !self.config.enable_backtracking {
            return Ok(());
        }

        self.refinement.backtrack(level)?;
        self.stats.backtracks += 1;
        Ok(())
    }

    /// Push decision level.
    pub fn push_decision_level(&mut self) {
        self.refinement.push_decision_level();
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.refinement.clear_history();
        self.constraints.clear();

        if let Some(ref mut enumerator) = self.enumerator {
            enumerator.reset();
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PartitionRefinementStats::default();
    }
}

/// Partition comparison utilities.
pub struct PartitionComparator;

impl PartitionComparator {
    /// Check if p1 is finer than p2.
    pub fn is_finer(p1: &Partition, p2: &Partition) -> bool {
        // p1 is finer if every class in p1 is a subset of some class in p2
        for class1 in &p1.classes {
            if class1.is_empty() {
                continue;
            }

            // Check if all terms in class1 are in the same class in p2
            let first_term = *class1.iter().next().expect("Non-empty class");
            let p2_class = p2.term_to_class.get(&first_term);

            for &term in class1 {
                if p2.term_to_class.get(&term) != p2_class {
                    return false;
                }
            }
        }

        true
    }

    /// Check if partitions are equal.
    pub fn are_equal(p1: &Partition, p2: &Partition) -> bool {
        Self::is_finer(p1, p2) && Self::is_finer(p2, p1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finest_partition() {
        let terms = vec![1, 2, 3];
        let partition = Partition::finest(&terms);

        assert_eq!(partition.num_classes(), 3);
        assert!(!partition.are_equal(1, 2));
    }

    #[test]
    fn test_coarsest_partition() {
        let terms = vec![1, 2, 3];
        let partition = Partition::coarsest(&terms);

        assert_eq!(partition.num_classes(), 1);
        assert!(partition.are_equal(1, 2));
        assert!(partition.are_equal(2, 3));
    }

    #[test]
    fn test_partition_merge() {
        let terms = vec![1, 2, 3, 4];
        let mut partition = Partition::finest(&terms);

        partition.merge(1, 2).expect("Merge failed");
        assert_eq!(partition.num_classes(), 3);
        assert!(partition.are_equal(1, 2));
        assert!(!partition.are_equal(1, 3));
    }

    #[test]
    fn test_partition_equalities() {
        let terms = vec![1, 2, 3];
        let mut partition = Partition::finest(&terms);

        partition.merge(1, 2).expect("Merge failed");
        partition.merge(2, 3).expect("Merge failed");

        let equalities = partition.get_equalities();
        assert_eq!(equalities.len(), 2); // Star topology
    }

    #[test]
    fn test_refinement() {
        let terms = vec![1, 2, 3, 4];
        let mut refinement = PartitionRefinement::new(&terms);

        refinement
            .refine(Equality::new(1, 2))
            .expect("Refine failed");
        assert!(refinement.current().are_equal(1, 2));
    }

    #[test]
    fn test_refinement_backtrack() {
        let terms = vec![1, 2, 3, 4];
        let mut refinement = PartitionRefinement::new(&terms);

        refinement
            .refine(Equality::new(1, 2))
            .expect("Refine failed");
        refinement.backtrack_step().expect("Backtrack failed");

        assert!(!refinement.current().are_equal(1, 2));
    }

    #[test]
    fn test_bell_number() {
        assert_eq!(bell_number(0), 1);
        assert_eq!(bell_number(1), 1);
        assert_eq!(bell_number(2), 2);
        assert_eq!(bell_number(3), 5);
        assert_eq!(bell_number(4), 15);
    }

    #[test]
    fn test_partition_enumerator() {
        let terms = vec![1, 2, 3];
        let mut enumerator = PartitionEnumerator::new(terms);

        let mut count = 0;
        while enumerator.next().is_some() {
            count += 1;
        }

        assert_eq!(count, 5); // B(3) = 5
    }

    #[test]
    fn test_manager() {
        let terms = vec![1, 2, 3];
        let mut manager = PartitionRefinementManager::new(terms);

        manager.add_constraint(Equality::new(1, 2));
        manager.apply_constraints().expect("Apply failed");

        assert!(manager.current_partition().are_equal(1, 2));
    }

    #[test]
    fn test_partition_comparison() {
        let terms = vec![1, 2, 3];

        let finest = Partition::finest(&terms);
        let coarsest = Partition::coarsest(&terms);

        assert!(PartitionComparator::is_finer(&finest, &coarsest));
        assert!(!PartitionComparator::is_finer(&coarsest, &finest));
    }

    #[test]
    fn test_representative() {
        let terms = vec![1, 2, 3];
        let mut partition = Partition::finest(&terms);

        partition.merge(1, 2).expect("Merge failed");

        let rep1 = partition.get_representative(1);
        let rep2 = partition.get_representative(2);

        assert_eq!(rep1, rep2);
    }

    #[test]
    fn test_get_class() {
        let terms = vec![1, 2, 3, 4];
        let mut partition = Partition::finest(&terms);

        partition.merge(1, 2).expect("Merge failed");
        partition.merge(2, 3).expect("Merge failed");

        let class = partition.get_class(1).expect("No class");
        assert_eq!(class.len(), 3);
        assert!(class.contains(&1));
        assert!(class.contains(&2));
        assert!(class.contains(&3));
    }
}
