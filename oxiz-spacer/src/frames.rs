//! Frame management for PDR (Property Directed Reachability).
//!
//! Frames F_0, F_1, ..., F_N represent over-approximations of reachable states.
//!
//! Key invariants:
//! - F_0 = Init (initial states)
//! - F_i ⊇ F_{i+1} (monotonicity: each frame is an over-approximation of the next)
//! - F_i ∧ T ⊆ F_{i+1} (consecution: transitions from F_i stay in F_{i+1})
//!
//! Reference: Z3's `muz/spacer/spacer_context.h` frames class.

use crate::chc::PredId;
use oxiz_core::TermId;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Represents infinity level for inductive lemmas
pub const INFTY_LEVEL: u32 = u32::MAX;

/// Check if a level is infinity
#[inline]
pub fn is_infty_level(level: u32) -> bool {
    level == INFTY_LEVEL
}

/// A lemma in the frame sequence
#[derive(Debug, Clone)]
pub struct Lemma {
    /// Unique identifier for this lemma
    pub id: LemmaId,
    /// The lemma formula (negation of blocked states)
    pub formula: TermId,
    /// Current level of the lemma
    level: u32,
    /// Initial level when the lemma was created
    init_level: u32,
    /// Number of times this lemma has been bumped (priority boost)
    bumped: u16,
    /// Weakness score for prioritization
    weakness: u16,
    /// Whether this lemma came from an external source
    external: bool,
    /// Whether this lemma is blocked by a counter-example to pushing
    blocked: bool,
    /// Whether this is a background assumption
    background: bool,
}

impl Lemma {
    /// Create a new lemma at a given level
    pub fn new(id: LemmaId, formula: TermId, level: u32) -> Self {
        Self {
            id,
            formula,
            level,
            init_level: level,
            bumped: 0,
            weakness: 0,
            external: false,
            blocked: false,
            background: false,
        }
    }

    /// Check if this lemma is inductive (at infinity level)
    #[inline]
    #[must_use]
    pub fn is_inductive(&self) -> bool {
        is_infty_level(self.level)
    }

    /// Get the current level
    #[inline]
    #[must_use]
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Get the initial level
    #[inline]
    #[must_use]
    pub fn init_level(&self) -> u32 {
        self.init_level
    }

    /// Set the level (for propagation)
    #[inline]
    pub fn set_level(&mut self, level: u32) {
        self.level = level;
    }

    /// Bump the lemma priority
    #[inline]
    pub fn bump(&mut self) {
        self.bumped = self.bumped.saturating_add(1);
    }

    /// Get the bump count
    #[inline]
    #[must_use]
    pub fn bumped(&self) -> u16 {
        self.bumped
    }

    /// Get the weakness score
    #[inline]
    #[must_use]
    pub fn weakness(&self) -> u16 {
        self.weakness
    }

    /// Set the weakness score
    #[inline]
    pub fn set_weakness(&mut self, weakness: u16) {
        self.weakness = weakness;
    }

    /// Check if this is an external lemma
    #[inline]
    #[must_use]
    pub fn is_external(&self) -> bool {
        self.external
    }

    /// Mark as external
    #[inline]
    pub fn set_external(&mut self, external: bool) {
        self.external = external;
    }

    /// Check if blocked by CTP
    #[inline]
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.blocked
    }

    /// Set blocked status
    #[inline]
    pub fn set_blocked(&mut self, blocked: bool) {
        self.blocked = blocked;
    }

    /// Check if this is a background assumption
    #[inline]
    #[must_use]
    pub fn is_background(&self) -> bool {
        self.background
    }

    /// Mark as background
    #[inline]
    pub fn set_background(&mut self, background: bool) {
        self.background = background;
    }
}

/// Unique identifier for a lemma
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LemmaId(pub u32);

impl LemmaId {
    /// Create a new lemma ID
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Comparison for lemmas by level (for sorting)
impl PartialEq for Lemma {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Lemma {}

impl PartialOrd for Lemma {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Lemma {
    fn cmp(&self, other: &Self) -> Ordering {
        self.level
            .cmp(&other.level)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Frame sequence for a single predicate transformer
#[derive(Debug)]
pub struct PredicateFrames {
    /// The predicate this frame sequence is for
    pub pred: PredId,
    /// All lemmas (both active and pinned)
    lemmas: Vec<Lemma>,
    /// Active lemma IDs (not subsumed)
    active: FxHashSet<LemmaId>,
    /// Background invariants
    bg_invariants: SmallVec<[LemmaId; 4]>,
    /// Number of frames
    num_frames: u32,
    /// Next lemma ID
    next_lemma_id: u32,
    /// Whether lemmas are sorted
    sorted: bool,
}

impl PredicateFrames {
    /// Create a new frame sequence for a predicate
    pub fn new(pred: PredId) -> Self {
        Self {
            pred,
            lemmas: Vec::new(),
            active: FxHashSet::default(),
            bg_invariants: SmallVec::new(),
            num_frames: 1, // Start with F_0
            next_lemma_id: 0,
            sorted: true,
        }
    }

    /// Get the number of frames
    #[inline]
    #[must_use]
    pub fn num_frames(&self) -> u32 {
        self.num_frames
    }

    /// Add a new frame
    pub fn add_frame(&mut self) {
        self.num_frames += 1;
    }

    /// Add a lemma at a given level
    pub fn add_lemma(&mut self, formula: TermId, level: u32) -> LemmaId {
        let id = LemmaId(self.next_lemma_id);
        self.next_lemma_id += 1;

        let lemma = Lemma::new(id, formula, level);
        self.lemmas.push(lemma);
        self.active.insert(id);
        self.sorted = false;

        id
    }

    /// Add a background invariant
    pub fn add_background(&mut self, formula: TermId) -> LemmaId {
        let id = self.add_lemma(formula, INFTY_LEVEL);
        if let Some(lemma) = self.get_lemma_mut(id) {
            lemma.set_background(true);
        }
        self.bg_invariants.push(id);
        id
    }

    /// Get a lemma by ID
    #[must_use]
    pub fn get_lemma(&self, id: LemmaId) -> Option<&Lemma> {
        self.lemmas.get(id.0 as usize)
    }

    /// Get a mutable lemma by ID
    pub fn get_lemma_mut(&mut self, id: LemmaId) -> Option<&mut Lemma> {
        self.lemmas.get_mut(id.0 as usize)
    }

    /// Get all active lemmas
    #[inline]
    pub fn active_lemmas(&self) -> impl Iterator<Item = &Lemma> {
        self.active.iter().filter_map(|&id| self.get_lemma(id))
    }

    /// Get lemmas at a specific level
    pub fn lemmas_at_level(&self, level: u32) -> impl Iterator<Item = &Lemma> {
        self.active_lemmas().filter(move |l| l.level() == level)
    }

    /// Get lemmas at or above a level (for frame queries)
    pub fn lemmas_geq_level(&self, level: u32) -> impl Iterator<Item = &Lemma> {
        self.active_lemmas().filter(move |l| l.level() >= level)
    }

    /// Get all inductive lemmas
    #[inline]
    pub fn inductive_lemmas(&self) -> impl Iterator<Item = &Lemma> {
        self.active_lemmas().filter(|l| l.is_inductive())
    }

    /// Get background invariants
    pub fn background_invariants(&self) -> impl Iterator<Item = &Lemma> {
        self.bg_invariants
            .iter()
            .filter_map(|&id| self.get_lemma(id))
    }

    /// Propagate a lemma to a higher level
    pub fn propagate(&mut self, id: LemmaId, new_level: u32) -> bool {
        if let Some(lemma) = self.get_lemma_mut(id)
            && new_level > lemma.level()
        {
            lemma.set_level(new_level);
            self.sorted = false;
            return true;
        }
        false
    }

    /// Propagate lemmas to infinity (inductive)
    pub fn propagate_to_infinity(&mut self, from_level: u32) {
        for lemma in &mut self.lemmas {
            if self.active.contains(&lemma.id) && lemma.level() >= from_level {
                lemma.set_level(INFTY_LEVEL);
            }
        }
        self.sorted = false;
    }

    /// Try to propagate all lemmas at a level to the next level
    /// Returns true if all lemmas were successfully propagated
    pub fn propagate_level(&mut self, level: u32) -> bool {
        let mut all_propagated = true;
        let lemma_ids: Vec<_> = self.lemmas_at_level(level).map(|l| l.id).collect();

        for id in lemma_ids {
            if !self.propagate(id, level + 1) {
                all_propagated = false;
            }
        }

        all_propagated
    }

    /// Deactivate a lemma (mark as subsumed)
    pub fn deactivate(&mut self, id: LemmaId) {
        self.active.remove(&id);
    }

    /// Check if a lemma is active
    #[must_use]
    pub fn is_active(&self, id: LemmaId) -> bool {
        self.active.contains(&id)
    }

    /// Get number of active lemmas
    #[must_use]
    pub fn num_active_lemmas(&self) -> usize {
        self.active.len()
    }

    /// Get number of inductive lemmas
    #[must_use]
    pub fn num_inductive(&self) -> usize {
        self.inductive_lemmas().count()
    }

    /// Remove subsumed lemmas using syntactic subsumption
    /// Returns the number of lemmas removed
    pub fn remove_subsumed_syntactic(&mut self) -> usize {
        let mut to_deactivate = Vec::new();

        // Collect all active lemma IDs and formulas
        let active_lemmas: Vec<_> = self
            .active
            .iter()
            .filter_map(|&id| self.get_lemma(id).map(|l| (id, l.formula)))
            .collect();

        // Check for syntactic duplicates
        for i in 0..active_lemmas.len() {
            for j in (i + 1)..active_lemmas.len() {
                if active_lemmas[i].1 == active_lemmas[j].1 {
                    // Same formula - remove one (keep the one with lower level)
                    let lemma_i = self.get_lemma(active_lemmas[i].0);
                    let lemma_j = self.get_lemma(active_lemmas[j].0);

                    if let (Some(li), Some(lj)) = (lemma_i, lemma_j) {
                        if li.level() > lj.level() {
                            to_deactivate.push(active_lemmas[i].0);
                        } else {
                            to_deactivate.push(active_lemmas[j].0);
                        }
                    }
                }
            }
        }

        // Remove duplicates from deactivation list
        to_deactivate.sort_unstable_by_key(|id| id.0);
        to_deactivate.dedup();

        // Actually deactivate
        let count = to_deactivate.len();
        for id in to_deactivate {
            self.deactivate(id);
        }

        count
    }

    /// Sort lemmas by level (for efficient queries)
    pub fn sort_lemmas(&mut self) {
        if !self.sorted {
            // We don't actually need to sort the storage, just mark as sorted
            // since we filter through active set anyway
            self.sorted = true;
        }
    }

    /// Clear all lemmas (reset)
    pub fn clear(&mut self) {
        self.lemmas.clear();
        self.active.clear();
        self.bg_invariants.clear();
        self.num_frames = 1;
        self.next_lemma_id = 0;
        self.sorted = true;
    }

    /// Compress frames by removing lemmas at lower levels that have been pushed
    /// This is a memory optimization to reduce the number of stored lemmas
    /// Returns the number of lemmas removed
    pub fn compress(&mut self, keep_above_level: u32) -> usize {
        let mut removed = 0;

        // Collect lemmas to remove (those below keep_above_level and not inductive)
        let to_remove: Vec<_> = self
            .active
            .iter()
            .filter_map(|&id| {
                self.get_lemma(id).and_then(|l| {
                    if l.level() < keep_above_level && !l.is_inductive() {
                        Some(id)
                    } else {
                        None
                    }
                })
            })
            .collect();

        // Deactivate these lemmas
        for id in to_remove {
            self.deactivate(id);
            removed += 1;
        }

        removed
    }

    /// Get memory usage statistics
    #[must_use]
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        (
            self.lemmas.len(),        // Total lemmas
            self.active.len(),        // Active lemmas
            self.bg_invariants.len(), // Background invariants
        )
    }
}

/// Global frame manager for all predicates
#[derive(Debug)]
pub struct FrameManager {
    /// Per-predicate frame sequences
    frames: rustc_hash::FxHashMap<PredId, PredicateFrames>,
    /// Current global frame level
    current_level: u32,
}

impl Default for FrameManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameManager {
    /// Create a new frame manager
    pub fn new() -> Self {
        Self {
            frames: rustc_hash::FxHashMap::default(),
            current_level: 0,
        }
    }

    /// Get or create frames for a predicate
    pub fn get_or_create(&mut self, pred: PredId) -> &mut PredicateFrames {
        self.frames
            .entry(pred)
            .or_insert_with(|| PredicateFrames::new(pred))
    }

    /// Get frames for a predicate
    #[must_use]
    pub fn get(&self, pred: PredId) -> Option<&PredicateFrames> {
        self.frames.get(&pred)
    }

    /// Get mutable frames for a predicate
    pub fn get_mut(&mut self, pred: PredId) -> Option<&mut PredicateFrames> {
        self.frames.get_mut(&pred)
    }

    /// Get the current global level
    #[inline]
    #[must_use]
    pub fn current_level(&self) -> u32 {
        self.current_level
    }

    /// Advance to the next level
    pub fn next_level(&mut self) {
        self.current_level += 1;
        for frames in self.frames.values_mut() {
            frames.add_frame();
        }
    }

    /// Add a lemma for a predicate at a level
    pub fn add_lemma(&mut self, pred: PredId, formula: TermId, level: u32) -> LemmaId {
        self.get_or_create(pred).add_lemma(formula, level)
    }

    /// Check for fixpoint: all lemmas at current level are inductive
    #[must_use]
    pub fn is_fixpoint(&self) -> bool {
        let level = self.current_level;
        for frames in self.frames.values() {
            // Check if there are any non-inductive lemmas at current level
            if frames.lemmas_at_level(level).any(|l| !l.is_inductive()) {
                return false;
            }
        }
        true
    }

    /// Try to propagate all frames
    /// Returns true if a fixpoint is detected
    pub fn propagate(&mut self) -> bool {
        // Try to push all lemmas to higher levels
        for level in 1..=self.current_level {
            let mut all_propagated = true;

            for frames in self.frames.values_mut() {
                if !frames.propagate_level(level) {
                    all_propagated = false;
                }
            }

            // If all lemmas at this level were pushed, we have a fixpoint
            if all_propagated && level == self.current_level {
                // Promote all lemmas at this level to infinity
                for frames in self.frames.values_mut() {
                    frames.propagate_to_infinity(level);
                }
                return true;
            }
        }

        false
    }

    /// Compress all frames by removing old lemmas
    /// Returns the total number of lemmas removed
    pub fn compress(&mut self, keep_above_level: u32) -> usize {
        let mut total_removed = 0;
        for frames in self.frames.values_mut() {
            total_removed += frames.compress(keep_above_level);
        }
        total_removed
    }

    /// Get total memory usage statistics across all predicates
    #[must_use]
    pub fn total_memory_stats(&self) -> (usize, usize, usize) {
        let mut total_lemmas = 0;
        let mut total_active = 0;
        let mut total_bg = 0;

        for frames in self.frames.values() {
            let (lemmas, active, bg) = frames.memory_stats();
            total_lemmas += lemmas;
            total_active += active;
            total_bg += bg;
        }

        (total_lemmas, total_active, total_bg)
    }

    /// Reset all frames
    pub fn reset(&mut self) {
        for frames in self.frames.values_mut() {
            frames.clear();
        }
        self.current_level = 0;
    }

    /// Get statistics
    pub fn stats(&self) -> FrameStats {
        let mut total_lemmas = 0;
        let mut total_inductive = 0;
        let mut max_level = 0;

        for frames in self.frames.values() {
            total_lemmas += frames.num_active_lemmas();
            total_inductive += frames.num_inductive();
            max_level = max_level.max(frames.num_frames());
        }

        FrameStats {
            total_lemmas,
            total_inductive,
            num_predicates: self.frames.len(),
            max_level,
            current_level: self.current_level,
        }
    }
}

/// Statistics about the frame sequence
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Total number of active lemmas
    pub total_lemmas: usize,
    /// Number of inductive lemmas
    pub total_inductive: usize,
    /// Number of predicates with frames
    pub num_predicates: usize,
    /// Maximum frame level
    pub max_level: u32,
    /// Current level
    pub current_level: u32,
}

/// Priority queue for lemmas (by level, lower first)
#[derive(Debug)]
pub struct LemmaQueue {
    heap: BinaryHeap<std::cmp::Reverse<(u32, LemmaId, PredId)>>,
}

impl Default for LemmaQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl LemmaQueue {
    /// Create a new lemma queue
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Push a lemma to the queue
    pub fn push(&mut self, level: u32, lemma: LemmaId, pred: PredId) {
        self.heap.push(std::cmp::Reverse((level, lemma, pred)));
    }

    /// Pop the lemma with lowest level
    pub fn pop(&mut self) -> Option<(u32, LemmaId, PredId)> {
        self.heap.pop().map(|r| r.0)
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get the number of items
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Clear the queue
    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lemma_creation() {
        let id = LemmaId::new(0);
        let formula = oxiz_core::TermId::new(42);
        let lemma = Lemma::new(id, formula, 1);

        assert_eq!(lemma.level(), 1);
        assert_eq!(lemma.init_level(), 1);
        assert!(!lemma.is_inductive());
    }

    #[test]
    fn test_inductive_lemma() {
        let id = LemmaId::new(0);
        let formula = oxiz_core::TermId::new(42);
        let mut lemma = Lemma::new(id, formula, 1);

        lemma.set_level(INFTY_LEVEL);
        assert!(lemma.is_inductive());
    }

    #[test]
    fn test_predicate_frames() {
        let pred = PredId::new(0);
        let mut frames = PredicateFrames::new(pred);

        let f1 = oxiz_core::TermId::new(1);
        let f2 = oxiz_core::TermId::new(2);

        let id1 = frames.add_lemma(f1, 1);
        let id2 = frames.add_lemma(f2, 2);

        assert_eq!(frames.num_active_lemmas(), 2);
        assert_eq!(frames.lemmas_at_level(1).count(), 1);
        assert_eq!(frames.lemmas_at_level(2).count(), 1);

        // Propagate lemma 1 to level 2
        assert!(frames.propagate(id1, 2));
        assert_eq!(frames.lemmas_at_level(2).count(), 2);
        assert_eq!(frames.lemmas_at_level(1).count(), 0);

        // Deactivate lemma 2
        frames.deactivate(id2);
        assert!(!frames.is_active(id2));
        assert_eq!(frames.num_active_lemmas(), 1);
    }

    #[test]
    fn test_frame_manager() {
        let mut manager = FrameManager::new();
        let pred = PredId::new(0);

        let f1 = oxiz_core::TermId::new(1);
        manager.add_lemma(pred, f1, 0);

        assert_eq!(manager.current_level(), 0);

        manager.next_level();
        assert_eq!(manager.current_level(), 1);

        let stats = manager.stats();
        assert_eq!(stats.total_lemmas, 1);
        assert_eq!(stats.num_predicates, 1);
    }

    #[test]
    fn test_lemma_queue() {
        let mut queue = LemmaQueue::new();

        queue.push(3, LemmaId::new(0), PredId::new(0));
        queue.push(1, LemmaId::new(1), PredId::new(0));
        queue.push(2, LemmaId::new(2), PredId::new(0));

        // Should pop in order of level (lowest first)
        let (level, _, _) = queue.pop().unwrap();
        assert_eq!(level, 1);

        let (level, _, _) = queue.pop().unwrap();
        assert_eq!(level, 2);

        let (level, _, _) = queue.pop().unwrap();
        assert_eq!(level, 3);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_propagate_to_infinity() {
        let pred = PredId::new(0);
        let mut frames = PredicateFrames::new(pred);

        let f1 = oxiz_core::TermId::new(1);
        let f2 = oxiz_core::TermId::new(2);
        let f3 = oxiz_core::TermId::new(3);

        frames.add_lemma(f1, 1);
        frames.add_lemma(f2, 2);
        frames.add_lemma(f3, 3);

        // Propagate all lemmas at level >= 2 to infinity
        frames.propagate_to_infinity(2);

        assert_eq!(frames.lemmas_at_level(1).count(), 1);
        assert_eq!(frames.inductive_lemmas().count(), 2);
    }
}
