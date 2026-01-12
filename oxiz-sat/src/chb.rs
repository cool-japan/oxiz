//! CHB (Conflict History-Based) branching heuristic
//!
//! CHB is a modern branching heuristic that maintains a score for each variable
//! based on how recently it participated in conflicts.

use crate::literal::Var;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Variable with its CHB score
#[derive(Debug, Clone)]
struct VarScore {
    var: Var,
    score: f64,
}

impl PartialEq for VarScore {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.var == other.var
    }
}

impl Eq for VarScore {}

impl PartialOrd for VarScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VarScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher scores first (max heap)
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.var.index().cmp(&self.var.index()))
    }
}

/// CHB branching heuristic
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct CHB {
    /// Variable scores based on conflict history
    scores: Vec<f64>,
    /// Step size for score updates (alpha)
    step_size: f64,
    /// Step size decay factor
    step_decay: f64,
    /// Minimum step size
    min_step_size: f64,
    /// Priority queue of variables
    heap: BinaryHeap<VarScore>,
    /// Track which variables are in the heap
    in_heap: Vec<bool>,
    /// Position map for faster updates
    positions: FxHashMap<Var, usize>,
}

impl CHB {
    /// Create a new CHB heuristic
    pub fn new(num_vars: usize) -> Self {
        Self {
            scores: vec![0.0; num_vars],
            step_size: 0.4,
            step_decay: 0.000001,
            min_step_size: 0.06,
            heap: BinaryHeap::new(),
            in_heap: vec![false; num_vars],
            positions: FxHashMap::default(),
        }
    }

    /// Insert a variable into the heap
    pub fn insert(&mut self, var: Var) {
        let idx = var.index();
        if idx >= self.scores.len() {
            self.scores.resize(idx + 1, 0.0);
            self.in_heap.resize(idx + 1, false);
        }

        if !self.in_heap[idx] {
            self.heap.push(VarScore {
                var,
                score: self.scores[idx],
            });
            self.in_heap[idx] = true;
        }
    }

    /// Bump the score of a variable (participated in conflict)
    pub fn bump(&mut self, var: Var) {
        let idx = var.index();
        if idx >= self.scores.len() {
            return;
        }

        self.scores[idx] += self.step_size;

        // If variable is in heap, we need to rebuild it (lazy approach)
        // In practice, we'd want a more efficient updateable heap
    }

    /// Decay step size after each conflict
    pub fn decay(&mut self) {
        self.step_size -= self.step_decay;
        if self.step_size < self.min_step_size {
            self.step_size = self.min_step_size;
        }
    }

    /// Get the variable with the highest score
    pub fn pop_max(&mut self) -> Option<Var> {
        while let Some(vs) = self.heap.pop() {
            let idx = vs.var.index();
            if idx < self.in_heap.len() && self.in_heap[idx] {
                self.in_heap[idx] = false;
                return Some(vs.var);
            }
        }
        None
    }

    /// Check if a variable is in the heap
    #[allow(dead_code)]
    pub fn contains(&self, var: Var) -> bool {
        let idx = var.index();
        idx < self.in_heap.len() && self.in_heap[idx]
    }

    /// Clear the heuristic
    pub fn clear(&mut self) {
        self.scores.clear();
        self.heap.clear();
        self.in_heap.clear();
        self.positions.clear();
        self.step_size = 0.4;
    }

    /// Rebuild the heap with current scores
    pub fn rebuild_heap(&mut self) {
        self.heap.clear();
        for (idx, &score) in self.scores.iter().enumerate() {
            if self.in_heap[idx] {
                self.heap.push(VarScore {
                    var: Var::new(idx as u32),
                    score,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chb_basic() {
        let mut chb = CHB::new(10);
        let v0 = Var::new(0);
        let v1 = Var::new(1);

        chb.insert(v0);
        chb.insert(v1);

        chb.bump(v1);
        chb.bump(v1);

        // Check scores
        assert!(chb.scores[1] > chb.scores[0]);

        // Rebuild to ensure heap reflects current scores
        chb.rebuild_heap();

        // v1 should have higher score and be returned first
        let first = chb.pop_max();
        let second = chb.pop_max();

        // After 2 bumps, v1 should have score 0.8, v0 should have score 0.0
        // So v1 should be first
        assert_eq!(first, Some(v1));
        assert_eq!(second, Some(v0));
    }

    #[test]
    fn test_chb_decay() {
        let mut chb = CHB::new(10);
        let initial_step = chb.step_size;

        chb.decay();
        assert!(chb.step_size < initial_step);

        // Decay many times
        for _ in 0..10000 {
            chb.decay();
        }

        // Should not go below minimum
        assert!(chb.step_size >= chb.min_step_size);
    }
}
