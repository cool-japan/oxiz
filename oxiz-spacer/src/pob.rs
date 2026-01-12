//! Proof Obligation (POB) management.
//!
//! POBs track states that need to be blocked at various frames.
//! Each POB represents a bad state that must be proven unreachable.
//!
//! Reference: Z3's `muz/spacer/spacer_context.h` pob class.

use crate::chc::PredId;
use crate::frames::LemmaId;
use oxiz_core::TermId;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

/// Unique identifier for a proof obligation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PobId(pub u32);

impl PobId {
    /// Create a new POB ID
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// A proof obligation: a state that must be blocked
#[derive(Debug, Clone)]
pub struct Pob {
    /// Unique identifier
    pub id: PobId,
    /// The predicate this POB is for
    pub pred: PredId,
    /// The post-condition (bad state to block)
    pub post: TermId,
    /// The level at which to block
    level: u32,
    /// Depth in the derivation tree (for prioritization)
    depth: u32,
    /// Parent POB (if this was derived from another)
    parent: Option<PobId>,
    /// Bindings for skolem constants
    bindings: SmallVec<[TermId; 4]>,
    /// Whether this POB is closed (blocked)
    closed: bool,
    /// The lemma that blocked this POB (if closed)
    blocking_lemma: Option<LemmaId>,
    /// Weakness score for generalization
    weakness: u16,
    /// Whether this POB is expandable
    expandable: bool,
}

impl Pob {
    /// Create a new proof obligation
    pub fn new(id: PobId, pred: PredId, post: TermId, level: u32, depth: u32) -> Self {
        Self {
            id,
            pred,
            post,
            level,
            depth,
            parent: None,
            bindings: SmallVec::new(),
            closed: false,
            blocking_lemma: None,
            weakness: 0,
            expandable: true,
        }
    }

    /// Create a derived POB (with parent)
    pub fn derived(
        id: PobId,
        pred: PredId,
        post: TermId,
        level: u32,
        depth: u32,
        parent: PobId,
    ) -> Self {
        let mut pob = Self::new(id, pred, post, level, depth);
        pob.parent = Some(parent);
        pob
    }

    /// Get the level
    #[inline]
    #[must_use]
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Set the level
    #[inline]
    pub fn set_level(&mut self, level: u32) {
        self.level = level;
    }

    /// Get the depth
    #[inline]
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Get the parent POB
    #[inline]
    #[must_use]
    pub fn parent(&self) -> Option<PobId> {
        self.parent
    }

    /// Check if this POB is closed (blocked)
    #[inline]
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Close this POB with a blocking lemma
    pub fn close(&mut self, lemma: LemmaId) {
        self.closed = true;
        self.blocking_lemma = Some(lemma);
    }

    /// Get the blocking lemma
    #[inline]
    #[must_use]
    pub fn blocking_lemma(&self) -> Option<LemmaId> {
        self.blocking_lemma
    }

    /// Get bindings
    #[inline]
    pub fn bindings(&self) -> &[TermId] {
        &self.bindings
    }

    /// Add a binding
    pub fn add_binding(&mut self, binding: TermId) {
        self.bindings.push(binding);
    }

    /// Set bindings
    pub fn set_bindings(&mut self, bindings: impl IntoIterator<Item = TermId>) {
        self.bindings = bindings.into_iter().collect();
    }

    /// Get weakness
    #[inline]
    #[must_use]
    pub fn weakness(&self) -> u16 {
        self.weakness
    }

    /// Set weakness
    #[inline]
    pub fn set_weakness(&mut self, weakness: u16) {
        self.weakness = weakness;
    }

    /// Check if expandable
    #[inline]
    #[must_use]
    pub fn is_expandable(&self) -> bool {
        self.expandable
    }

    /// Set expandable
    #[inline]
    pub fn set_expandable(&mut self, expandable: bool) {
        self.expandable = expandable;
    }

    /// Check if this is a root POB (no parent)
    #[inline]
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Get the priority for the POB queue
    /// Lower is higher priority: (level, depth)
    #[inline]
    #[must_use]
    pub fn priority(&self) -> (u32, u32) {
        (self.level, self.depth)
    }
}

/// Comparison for POBs by priority (level, then depth)
impl PartialEq for Pob {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Pob {}

impl PartialOrd for Pob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower level and depth = higher priority
        self.priority().cmp(&other.priority())
    }
}

/// Priority wrapper for POBs (min-heap by level, then depth)
#[derive(Debug, Clone, PartialEq, Eq)]
struct PobPriority {
    level: u32,
    depth: u32,
    id: PobId,
}

impl PartialOrd for PobPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PobPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse comparison for min-heap behavior
        other
            .level
            .cmp(&self.level)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// POB queue with priority ordering
#[derive(Debug)]
pub struct PobQueue {
    /// Priority queue (min-heap by level, then depth)
    heap: BinaryHeap<PobPriority>,
    /// All POBs
    pobs: Vec<Pob>,
    /// Next POB ID
    next_id: AtomicU32,
    /// Maximum level for this queue
    max_level: u32,
    /// Cached count of open (unclosed) POBs
    open_pobs_count: AtomicU32,
}

impl Default for PobQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl PobQueue {
    /// Create a new POB queue
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            pobs: Vec::new(),
            next_id: AtomicU32::new(0),
            max_level: 0,
            open_pobs_count: AtomicU32::new(0),
        }
    }

    /// Create a new POB and add it to the queue
    pub fn create(&mut self, pred: PredId, post: TermId, level: u32, depth: u32) -> PobId {
        let id = PobId(self.next_id.fetch_add(1, AtomicOrdering::Relaxed));
        let pob = Pob::new(id, pred, post, level, depth);
        self.pobs.push(pob);
        self.max_level = self.max_level.max(level);

        self.heap.push(PobPriority { level, depth, id });

        // Increment open POB count
        self.open_pobs_count.fetch_add(1, AtomicOrdering::Relaxed);

        id
    }

    /// Create a derived POB and add it to the queue
    pub fn create_derived(
        &mut self,
        pred: PredId,
        post: TermId,
        level: u32,
        depth: u32,
        parent: PobId,
    ) -> PobId {
        let id = PobId(self.next_id.fetch_add(1, AtomicOrdering::Relaxed));
        let pob = Pob::derived(id, pred, post, level, depth, parent);
        self.pobs.push(pob);
        self.max_level = self.max_level.max(level);

        self.heap.push(PobPriority { level, depth, id });

        // Increment open POB count
        self.open_pobs_count.fetch_add(1, AtomicOrdering::Relaxed);

        id
    }

    /// Get a POB by ID
    #[inline]
    #[must_use]
    pub fn get(&self, id: PobId) -> Option<&Pob> {
        self.pobs.get(id.0 as usize)
    }

    /// Get a mutable POB by ID
    pub fn get_mut(&mut self, id: PobId) -> Option<&mut Pob> {
        self.pobs.get_mut(id.0 as usize)
    }

    /// Pop the highest priority POB (lowest level, then depth)
    pub fn pop(&mut self) -> Option<PobId> {
        while let Some(priority) = self.heap.pop() {
            // Check if this POB is still valid (not closed)
            if let Some(pob) = self.get(priority.id)
                && !pob.is_closed()
                && pob.level() == priority.level
            {
                return Some(priority.id);
            }
        }
        None
    }

    /// Peek at the highest priority POB without removing
    pub fn peek(&self) -> Option<PobId> {
        for priority in self.heap.iter() {
            if let Some(pob) = self.get(priority.id)
                && !pob.is_closed()
                && pob.level() == priority.level
            {
                return Some(priority.id);
            }
        }
        None
    }

    /// Re-queue a POB (e.g., after level change)
    pub fn requeue(&mut self, id: PobId) {
        if let Some(pob) = self.get(id)
            && !pob.is_closed()
        {
            self.heap.push(PobPriority {
                level: pob.level(),
                depth: pob.depth(),
                id,
            });
        }
    }

    /// Close a POB with a blocking lemma
    pub fn close(&mut self, id: PobId, lemma: LemmaId) {
        if let Some(pob) = self.get_mut(id)
            && !pob.is_closed()
        {
            pob.close(lemma);
            // Decrement open POB count
            self.open_pobs_count.fetch_sub(1, AtomicOrdering::Relaxed);
        }
    }

    /// Check if the queue is empty (no open POBs)
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.open_count() == 0
    }

    /// Get the number of open (unclosed) POBs (cached)
    #[inline]
    #[must_use]
    pub fn open_count(&self) -> usize {
        self.open_pobs_count.load(AtomicOrdering::Relaxed) as usize
    }

    /// Get the total number of POBs
    #[inline]
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.pobs.len()
    }

    /// Get the maximum level
    #[inline]
    #[must_use]
    pub fn max_level(&self) -> u32 {
        self.max_level
    }

    /// Get all open POBs at a given level
    pub fn open_at_level(&self, level: u32) -> impl Iterator<Item = &Pob> {
        self.pobs
            .iter()
            .filter(move |p| !p.is_closed() && p.level() == level)
    }

    /// Get all open POBs for a predicate
    pub fn open_for_pred(&self, pred: PredId) -> impl Iterator<Item = &Pob> {
        self.pobs
            .iter()
            .filter(move |p| !p.is_closed() && p.pred == pred)
    }

    /// Clear all POBs
    pub fn clear(&mut self) {
        self.heap.clear();
        self.pobs.clear();
        self.next_id = AtomicU32::new(0);
        self.max_level = 0;
    }

    /// Find a POB by post-condition and parent
    #[must_use]
    pub fn find(&self, parent: Option<PobId>, post: TermId) -> Option<&Pob> {
        self.pobs
            .iter()
            .find(|p| p.parent() == parent && p.post == post)
    }

    /// Check subsumption: is there a POB that subsumes the given state?
    #[must_use]
    pub fn is_subsumed(&self, pred: PredId, _post: TermId, level: u32) -> bool {
        // A POB is subsumed if there's a closed POB at a higher level
        // with a more general (subsuming) post-condition
        // For now, simple check: any closed POB at higher level for same predicate
        self.pobs
            .iter()
            .any(|p| p.pred == pred && p.is_closed() && p.level() >= level)
    }

    /// Get the derivation trace from a POB to root
    pub fn trace_to_root(&self, id: PobId) -> Vec<PobId> {
        let mut trace = Vec::new();
        let mut current = Some(id);

        while let Some(pob_id) = current {
            trace.push(pob_id);
            current = self.get(pob_id).and_then(|p| p.parent());
        }

        trace.reverse();
        trace
    }
}

/// POB manager for multiple predicates
#[derive(Debug)]
pub struct PobManager {
    /// Per-predicate POB queues
    queues: rustc_hash::FxHashMap<PredId, PobQueue>,
    /// Global queue for priority ordering across predicates
    global_queue: BinaryHeap<PobPriority>,
    /// POB ID to predicate mapping
    pob_to_pred: rustc_hash::FxHashMap<PobId, PredId>,
    /// Global next ID
    next_id: AtomicU32,
}

impl Default for PobManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PobManager {
    /// Create a new POB manager
    pub fn new() -> Self {
        Self {
            queues: rustc_hash::FxHashMap::default(),
            global_queue: BinaryHeap::new(),
            pob_to_pred: rustc_hash::FxHashMap::default(),
            next_id: AtomicU32::new(0),
        }
    }

    /// Create a POB for a predicate
    pub fn create(&mut self, pred: PredId, post: TermId, level: u32, depth: u32) -> PobId {
        let id = PobId(self.next_id.fetch_add(1, AtomicOrdering::Relaxed));
        let pob = Pob::new(id, pred, post, level, depth);

        let queue = self.queues.entry(pred).or_default();
        queue.pobs.push(pob);
        queue.max_level = queue.max_level.max(level);

        self.pob_to_pred.insert(id, pred);
        self.global_queue.push(PobPriority { level, depth, id });

        id
    }

    /// Create a derived POB
    pub fn create_derived(
        &mut self,
        pred: PredId,
        post: TermId,
        level: u32,
        depth: u32,
        parent: PobId,
    ) -> PobId {
        let id = PobId(self.next_id.fetch_add(1, AtomicOrdering::Relaxed));
        let pob = Pob::derived(id, pred, post, level, depth, parent);

        let queue = self.queues.entry(pred).or_default();
        queue.pobs.push(pob);
        queue.max_level = queue.max_level.max(level);

        self.pob_to_pred.insert(id, pred);
        self.global_queue.push(PobPriority { level, depth, id });

        id
    }

    /// Get a POB by ID
    #[must_use]
    pub fn get(&self, id: PobId) -> Option<&Pob> {
        let pred = self.pob_to_pred.get(&id)?;
        let queue = self.queues.get(pred)?;
        queue.pobs.iter().find(|p| p.id == id)
    }

    /// Get a mutable POB by ID
    pub fn get_mut(&mut self, id: PobId) -> Option<&mut Pob> {
        let pred = *self.pob_to_pred.get(&id)?;
        let queue = self.queues.get_mut(&pred)?;
        queue.pobs.iter_mut().find(|p| p.id == id)
    }

    /// Pop the highest priority POB globally
    pub fn pop(&mut self) -> Option<PobId> {
        while let Some(priority) = self.global_queue.pop() {
            if let Some(pob) = self.get(priority.id)
                && !pob.is_closed()
                && pob.level() == priority.level
            {
                return Some(priority.id);
            }
        }
        None
    }

    /// Close a POB
    pub fn close(&mut self, id: PobId, lemma: LemmaId) {
        if let Some(pob) = self.get_mut(id) {
            pob.close(lemma);
        }
    }

    /// Get the queue for a predicate
    #[must_use]
    pub fn queue(&self, pred: PredId) -> Option<&PobQueue> {
        self.queues.get(&pred)
    }

    /// Check if all POBs are closed
    #[must_use]
    pub fn all_closed(&self) -> bool {
        self.queues.values().all(|q| q.is_empty())
    }

    /// Get total open POB count
    #[must_use]
    pub fn open_count(&self) -> usize {
        self.queues.values().map(|q| q.open_count()).sum()
    }

    /// Clear all POBs
    pub fn clear(&mut self) {
        self.queues.clear();
        self.global_queue.clear();
        self.pob_to_pred.clear();
        self.next_id = AtomicU32::new(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pob_creation() {
        let id = PobId::new(0);
        let pred = PredId::new(0);
        let post = oxiz_core::TermId::new(42);
        let pob = Pob::new(id, pred, post, 1, 0);

        assert_eq!(pob.level(), 1);
        assert_eq!(pob.depth(), 0);
        assert!(!pob.is_closed());
        assert!(pob.is_root());
    }

    #[test]
    fn test_pob_derived() {
        let id = PobId::new(1);
        let pred = PredId::new(0);
        let post = oxiz_core::TermId::new(42);
        let parent = PobId::new(0);
        let pob = Pob::derived(id, pred, post, 2, 1, parent);

        assert_eq!(pob.parent(), Some(parent));
        assert!(!pob.is_root());
    }

    #[test]
    fn test_pob_queue() {
        let mut queue = PobQueue::new();
        let pred = PredId::new(0);

        let post1 = oxiz_core::TermId::new(1);
        let post2 = oxiz_core::TermId::new(2);
        let post3 = oxiz_core::TermId::new(3);

        // Add POBs at different levels
        let _id3 = queue.create(pred, post3, 3, 0);
        let id1 = queue.create(pred, post1, 1, 0);
        let _id2 = queue.create(pred, post2, 2, 0);

        // Should pop in level order (lowest first)
        let popped = queue.pop();
        assert_eq!(popped, Some(id1));
    }

    #[test]
    fn test_pob_close() {
        let mut queue = PobQueue::new();
        let pred = PredId::new(0);
        let post = oxiz_core::TermId::new(42);
        let lemma = LemmaId::new(0);

        let id = queue.create(pred, post, 1, 0);
        assert!(!queue.is_empty());

        queue.close(id, lemma);
        assert!(queue.is_empty());

        let pob = queue.get(id).unwrap();
        assert!(pob.is_closed());
        assert_eq!(pob.blocking_lemma(), Some(lemma));
    }

    #[test]
    fn test_pob_trace() {
        let mut queue = PobQueue::new();
        let pred = PredId::new(0);

        let post1 = oxiz_core::TermId::new(1);
        let post2 = oxiz_core::TermId::new(2);
        let post3 = oxiz_core::TermId::new(3);

        let id1 = queue.create(pred, post1, 1, 0);
        let id2 = queue.create_derived(pred, post2, 2, 1, id1);
        let id3 = queue.create_derived(pred, post3, 3, 2, id2);

        let trace = queue.trace_to_root(id3);
        assert_eq!(trace, vec![id1, id2, id3]);
    }

    #[test]
    fn test_pob_manager() {
        let mut manager = PobManager::new();
        let pred1 = PredId::new(0);
        let pred2 = PredId::new(1);

        let post1 = oxiz_core::TermId::new(1);
        let post2 = oxiz_core::TermId::new(2);

        let id1 = manager.create(pred1, post1, 2, 0);
        let id2 = manager.create(pred2, post2, 1, 0);

        // Should pop id2 first (lower level)
        let popped = manager.pop();
        assert_eq!(popped, Some(id2));

        let popped = manager.pop();
        assert_eq!(popped, Some(id1));
    }
}
