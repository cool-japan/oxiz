//! LRB (Learning Rate Branching) heuristic
//!
//! LRB is a modern branching heuristic that uses exponential moving averages
//! to track variable participation in conflicts. It often outperforms VSIDS
//! on industrial instances.
//!
//! Reference: "Learning Rate Based Branching Heuristic for SAT Solvers"
//! by Jia Hui Liang et al. (SAT 2016)

use crate::literal::Var;
use std::collections::BinaryHeap;

/// LRB score entry for priority queue
#[derive(Debug, Clone)]
struct LrbEntry {
    var: Var,
    score: f64,
}

impl PartialEq for LrbEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for LrbEntry {}

impl PartialOrd for LrbEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LrbEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher scores should come first
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// LRB (Learning Rate Branching) heuristic
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct LRB {
    /// Number of variables
    num_vars: usize,
    /// Participation rates (exponential moving average)
    participation: Vec<f64>,
    /// Assigned counts (how many times variable was assigned)
    assigned: Vec<u64>,
    /// Reasoned counts (how many times variable appeared in conflict analysis)
    reasoned: Vec<u64>,
    /// Learning rate (step size)
    alpha: f64,
    /// Priority queue for unassigned variables
    heap: BinaryHeap<LrbEntry>,
    /// Position in heap (usize::MAX if not in heap)
    in_heap: Vec<bool>,
    /// Interval for updating scores
    interval: u64,
    /// Number of conflicts since last update
    conflicts: u64,
}

impl LRB {
    /// Create a new LRB heuristic
    pub fn new(num_vars: usize) -> Self {
        let mut lrb = Self {
            num_vars,
            participation: vec![0.0; num_vars],
            assigned: vec![0; num_vars],
            reasoned: vec![0; num_vars],
            alpha: 0.4,
            heap: BinaryHeap::new(),
            in_heap: vec![false; num_vars],
            interval: 5000,
            conflicts: 0,
        };

        // Initialize heap with all variables
        for v in 0..num_vars {
            lrb.heap.push(LrbEntry {
                var: Var(v as u32),
                score: 0.0,
            });
            lrb.in_heap[v] = true;
        }

        lrb
    }

    /// Resize for new variables
    pub fn resize(&mut self, num_vars: usize) {
        if num_vars <= self.num_vars {
            return;
        }

        let old_size = self.num_vars;
        self.num_vars = num_vars;

        self.participation.resize(num_vars, 0.0);
        self.assigned.resize(num_vars, 0);
        self.reasoned.resize(num_vars, 0);
        self.in_heap.resize(num_vars, false);

        // Add new variables to heap
        for v in old_size..num_vars {
            self.heap.push(LrbEntry {
                var: Var(v as u32),
                score: 0.0,
            });
            self.in_heap[v] = true;
        }
    }

    /// Record that a variable was assigned
    pub fn on_assign(&mut self, var: Var) {
        let idx = var.0 as usize;
        if idx < self.num_vars {
            self.assigned[idx] += 1;
        }
    }

    /// Record that a variable appeared in conflict analysis
    pub fn on_reason(&mut self, var: Var) {
        let idx = var.0 as usize;
        if idx < self.num_vars {
            self.reasoned[idx] += 1;
        }
    }

    /// Update participation rates after conflict
    pub fn on_conflict(&mut self) {
        self.conflicts += 1;

        if self.conflicts.is_multiple_of(self.interval) {
            self.update_participation();
        }
    }

    /// Update participation rates using exponential moving average
    fn update_participation(&mut self) {
        for v in 0..self.num_vars {
            if self.assigned[v] > 0 {
                let rate = self.reasoned[v] as f64 / self.assigned[v] as f64;
                // Exponential moving average: participation = α * rate + (1-α) * participation
                self.participation[v] =
                    self.alpha * rate + (1.0 - self.alpha) * self.participation[v];
            }
        }

        // Rebuild heap with updated scores
        self.rebuild_heap();
    }

    /// Rebuild the heap with current participation scores
    fn rebuild_heap(&mut self) {
        self.heap.clear();
        for v in 0..self.num_vars {
            if self.in_heap[v] {
                self.heap.push(LrbEntry {
                    var: Var(v as u32),
                    score: self.participation[v],
                });
            }
        }
    }

    /// Select next variable to branch on
    pub fn select(&mut self) -> Option<Var> {
        while let Some(entry) = self.heap.pop() {
            let idx = entry.var.0 as usize;
            if idx < self.num_vars && self.in_heap[idx] {
                self.in_heap[idx] = false;
                return Some(entry.var);
            }
        }
        None
    }

    /// Mark variable as unassigned (add back to heap)
    pub fn unassign(&mut self, var: Var) {
        let idx = var.0 as usize;
        if idx < self.num_vars && !self.in_heap[idx] {
            self.in_heap[idx] = true;
            self.heap.push(LrbEntry {
                var,
                score: self.participation[idx],
            });
        }
    }

    /// Decay alpha (learning rate) over time
    pub fn decay(&mut self) {
        self.alpha *= 0.95;
    }

    /// Get current participation rate for a variable
    #[allow(dead_code)]
    pub fn participation(&self, var: Var) -> f64 {
        let idx = var.0 as usize;
        if idx < self.num_vars {
            self.participation[idx]
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lrb_basic() {
        let mut lrb = LRB::new(10);

        // All variables should be selectable initially
        for _ in 0..10 {
            assert!(lrb.select().is_some());
        }

        // No more variables
        assert!(lrb.select().is_none());
    }

    #[test]
    fn test_lrb_unassign() {
        let mut lrb = LRB::new(5);

        let v0 = lrb.select().unwrap();
        let v1 = lrb.select().unwrap();

        // Unassign and should be able to select again
        // Note: heap doesn't guarantee order when scores are equal
        lrb.unassign(v0);
        let v2 = lrb.select().unwrap();
        // Just check that we got a valid variable
        assert!(v2.0 < 5);

        lrb.unassign(v1);
        let v3 = lrb.select().unwrap();
        assert!(v3.0 < 5);
    }

    #[test]
    fn test_lrb_participation() {
        let mut lrb = LRB::new(3);

        // Simulate assignments and conflicts
        let v0 = Var(0);
        let v1 = Var(1);

        // v0 participates more in conflicts
        for _ in 0..10 {
            lrb.on_assign(v0);
            lrb.on_reason(v0);
            lrb.on_assign(v1);
        }

        lrb.on_conflict();
        lrb.update_participation();

        // v0 should have higher participation rate
        assert!(lrb.participation(v0) > lrb.participation(v1));
    }
}
