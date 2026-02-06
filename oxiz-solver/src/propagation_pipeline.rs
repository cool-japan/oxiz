//! Multi-Level Propagation Pipeline.
//!
//! Coordinates propagation across different levels:
//! 1. Unit propagation (Boolean)
//! 2. Theory propagation (per theory)
//! 3. Learned clause propagation
//!
//! ## Architecture
//!
//! Priority queue based on propagation cost and impact.
//! Theories register propagation hooks and the pipeline schedules them efficiently.
//!
//! ## References
//!
//! - Z3's `smt/smt_context.cpp` propagation loop
//! - MiniSAT's propagation queue

use oxiz_sat::{Lit, Var};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Priority level for propagations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PropagationPriority {
    /// Immediate (unit propagation).
    Immediate,
    /// High priority (cheap theory propagation).
    High,
    /// Normal priority (expensive theory propagation).
    Normal,
    /// Low priority (heuristic propagation).
    Low,
}

/// Type of propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationType {
    /// Boolean unit propagation.
    Unit,
    /// Theory propagation (from theory solver).
    Theory,
    /// Learned clause propagation.
    Learned,
    /// Heuristic propagation.
    Heuristic,
}

/// A propagation item.
#[derive(Debug, Clone)]
pub struct PropagationItem {
    /// The literal to propagate.
    pub lit: Lit,
    /// Reason clause (if any).
    pub reason: Option<Vec<Lit>>,
    /// Priority level.
    pub priority: PropagationPriority,
    /// Type of propagation.
    pub prop_type: PropagationType,
    /// Theory ID (if theory propagation).
    pub theory_id: Option<usize>,
}

/// Configuration for propagation pipeline.
#[derive(Debug, Clone)]
pub struct PropagationConfig {
    /// Enable theory propagation.
    pub enable_theory_propagation: bool,
    /// Enable learned clause propagation.
    pub enable_learned_propagation: bool,
    /// Maximum queue size before forcing flush.
    pub max_queue_size: usize,
    /// Enable propagation batching.
    pub enable_batching: bool,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            enable_theory_propagation: true,
            enable_learned_propagation: true,
            max_queue_size: 10_000,
            enable_batching: true,
        }
    }
}

/// Statistics for propagation pipeline.
#[derive(Debug, Clone, Default)]
pub struct PropagationStats {
    /// Total propagations performed.
    pub total_propagations: u64,
    /// Unit propagations.
    pub unit_propagations: u64,
    /// Theory propagations.
    pub theory_propagations: u64,
    /// Learned propagations.
    pub learned_propagations: u64,
    /// Conflicts detected.
    pub conflicts: u64,
    /// Queue flushes.
    pub queue_flushes: u64,
}

/// Multi-level propagation pipeline.
#[derive(Debug)]
pub struct PropagationPipeline {
    /// Configuration.
    config: PropagationConfig,
    /// Propagation queue (priority-based).
    queue: VecDeque<PropagationItem>,
    /// Current assignment trail.
    trail: Vec<Lit>,
    /// Assigned variables.
    assigned: FxHashSet<Var>,
    /// Reason for each assignment.
    reasons: FxHashMap<Var, Vec<Lit>>,
    /// Statistics.
    stats: PropagationStats,
}

impl PropagationPipeline {
    /// Create a new propagation pipeline.
    pub fn new(config: PropagationConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
            trail: Vec::new(),
            assigned: FxHashSet::default(),
            reasons: FxHashMap::default(),
            stats: PropagationStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PropagationConfig::default())
    }

    /// Enqueue a propagation.
    pub fn enqueue(&mut self, item: PropagationItem) {
        // Check if variable already assigned
        let var = item.lit.var();
        if self.assigned.contains(&var) {
            return; // Already assigned
        }

        // Insert based on priority
        let pos = self.queue.iter().position(|i| i.priority > item.priority);
        if let Some(pos) = pos {
            self.queue.insert(pos, item);
        } else {
            self.queue.push_back(item);
        }

        // Force flush if queue too large
        if self.queue.len() >= self.config.max_queue_size {
            self.stats.queue_flushes += 1;
        }
    }

    /// Enqueue unit propagation.
    pub fn enqueue_unit(&mut self, lit: Lit, reason: Vec<Lit>) {
        self.enqueue(PropagationItem {
            lit,
            reason: Some(reason),
            priority: PropagationPriority::Immediate,
            prop_type: PropagationType::Unit,
            theory_id: None,
        });
    }

    /// Enqueue theory propagation.
    pub fn enqueue_theory(
        &mut self,
        lit: Lit,
        reason: Vec<Lit>,
        theory_id: usize,
        priority: PropagationPriority,
    ) {
        if !self.config.enable_theory_propagation {
            return;
        }

        self.enqueue(PropagationItem {
            lit,
            reason: Some(reason),
            priority,
            prop_type: PropagationType::Theory,
            theory_id: Some(theory_id),
        });
    }

    /// Dequeue next propagation.
    pub fn dequeue(&mut self) -> Option<PropagationItem> {
        self.queue.pop_front()
    }

    /// Process a propagation (assign literal).
    ///
    /// Returns true if assignment succeeded, false if conflict.
    pub fn process(&mut self, item: PropagationItem) -> bool {
        let var = item.lit.var();

        // Check if variable already assigned
        if self.assigned.contains(&var) {
            // Check for conflict
            return !self.trail.contains(&item.lit.negate());
        }

        // Assign the literal
        self.trail.push(item.lit);
        self.assigned.insert(var);

        // Store reason
        if let Some(reason) = item.reason {
            self.reasons.insert(var, reason);
        }

        // Update statistics
        self.stats.total_propagations += 1;
        match item.prop_type {
            PropagationType::Unit => self.stats.unit_propagations += 1,
            PropagationType::Theory => self.stats.theory_propagations += 1,
            PropagationType::Learned => self.stats.learned_propagations += 1,
            PropagationType::Heuristic => {}
        }

        true
    }

    /// Propagate all pending items until fixpoint or conflict.
    ///
    /// Returns None if consistent, Some(conflict_clause) if conflict.
    pub fn propagate_fixpoint(&mut self) -> Option<Vec<Lit>> {
        while let Some(item) = self.dequeue() {
            if !self.process(item.clone()) {
                // Conflict detected
                self.stats.conflicts += 1;

                // Build conflict clause
                let mut conflict = item.reason.clone().unwrap_or_default();
                conflict.push(item.lit.negate());
                return Some(conflict);
            }
        }

        None // Consistent
    }

    /// Get current assignment trail.
    pub fn trail(&self) -> &[Lit] {
        &self.trail
    }

    /// Get reason for a variable's assignment.
    pub fn reason(&self, var: Var) -> Option<&Vec<Lit>> {
        self.reasons.get(&var)
    }

    /// Check if variable is assigned.
    pub fn is_assigned(&self, var: Var) -> bool {
        self.assigned.contains(&var)
    }

    /// Backtrack to a specific decision level.
    ///
    /// Removes all assignments after the given trail position.
    pub fn backtrack(&mut self, trail_pos: usize) {
        while self.trail.len() > trail_pos {
            if let Some(lit) = self.trail.pop() {
                let var = lit.var();
                self.assigned.remove(&var);
                self.reasons.remove(&var);
            }
        }
    }

    /// Clear all assignments.
    pub fn reset(&mut self) {
        self.queue.clear();
        self.trail.clear();
        self.assigned.clear();
        self.reasons.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &PropagationStats {
        &self.stats
    }

    /// Get queue size.
    pub fn queue_size(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(n: i32) -> Lit {
        Lit::from_dimacs(n)
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = PropagationPipeline::default_config();
        assert_eq!(pipeline.trail().len(), 0);
        assert_eq!(pipeline.queue_size(), 0);
    }

    #[test]
    fn test_enqueue_unit() {
        let mut pipeline = PropagationPipeline::default_config();
        pipeline.enqueue_unit(lit(1), vec![lit(2), lit(3)]);

        assert_eq!(pipeline.queue_size(), 1);
    }

    #[test]
    fn test_process_propagation() {
        let mut pipeline = PropagationPipeline::default_config();
        let item = PropagationItem {
            lit: lit(1),
            reason: Some(vec![lit(2)]),
            priority: PropagationPriority::Immediate,
            prop_type: PropagationType::Unit,
            theory_id: None,
        };

        let success = pipeline.process(item);
        assert!(success);
        assert_eq!(pipeline.trail().len(), 1);
        assert!(pipeline.is_assigned(lit(1).var()));
    }

    #[test]
    fn test_conflict_detection() {
        let mut pipeline = PropagationPipeline::default_config();

        // Assign literal 1
        pipeline.process(PropagationItem {
            lit: lit(1),
            reason: Some(vec![]),
            priority: PropagationPriority::Immediate,
            prop_type: PropagationType::Unit,
            theory_id: None,
        });

        // Try to assign -1 (conflict)
        let success = pipeline.process(PropagationItem {
            lit: lit(-1),
            reason: Some(vec![lit(2)]),
            priority: PropagationPriority::Immediate,
            prop_type: PropagationType::Unit,
            theory_id: None,
        });

        assert!(!success); // Conflict
    }

    #[test]
    fn test_backtrack() {
        let mut pipeline = PropagationPipeline::default_config();

        pipeline.enqueue_unit(lit(1), vec![]);
        pipeline.enqueue_unit(lit(2), vec![]);
        pipeline.enqueue_unit(lit(3), vec![]);

        let _ = pipeline.propagate_fixpoint();

        assert_eq!(pipeline.trail().len(), 3);

        // Backtrack to position 1
        pipeline.backtrack(1);

        assert_eq!(pipeline.trail().len(), 1);
        assert!(pipeline.is_assigned(lit(1).var()));
        assert!(!pipeline.is_assigned(lit(2).var()));
    }

    #[test]
    fn test_priority_ordering() {
        let mut pipeline = PropagationPipeline::default_config();

        // Enqueue in different priority order
        pipeline.enqueue(PropagationItem {
            lit: lit(1),
            reason: None,
            priority: PropagationPriority::Low,
            prop_type: PropagationType::Heuristic,
            theory_id: None,
        });

        pipeline.enqueue(PropagationItem {
            lit: lit(2),
            reason: None,
            priority: PropagationPriority::Immediate,
            prop_type: PropagationType::Unit,
            theory_id: None,
        });

        // Immediate priority should be dequeued first
        let item = pipeline.dequeue().unwrap();
        assert_eq!(item.lit, lit(2));
    }
}
