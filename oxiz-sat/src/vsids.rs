//! VSIDS (Variable State Independent Decaying Sum) branching heuristic

use crate::literal::Var;

/// VSIDS branching heuristic
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct VSIDS {
    /// Activity score for each variable
    activity: Vec<f64>,
    /// Activity increment
    increment: f64,
    /// Decay factor
    decay: f64,
    /// Priority queue (binary heap indices)
    heap: Vec<Var>,
    /// Position in heap for each variable
    heap_pos: Vec<usize>,
}

impl VSIDS {
    /// Create a new VSIDS instance for n variables
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        let mut vsids = Self {
            activity: vec![0.0; num_vars],
            increment: 1.0,
            decay: 0.95,
            heap: Vec::with_capacity(num_vars),
            heap_pos: vec![usize::MAX; num_vars],
        };

        // Initialize heap with all variables
        for i in 0..num_vars {
            vsids.heap.push(Var::new(i as u32));
            vsids.heap_pos[i] = i;
        }

        vsids
    }

    /// Get the activity of a variable
    #[must_use]
    #[allow(dead_code)]
    pub fn activity(&self, var: Var) -> f64 {
        self.activity.get(var.index()).copied().unwrap_or(0.0)
    }

    /// Bump the activity of a variable
    pub fn bump(&mut self, var: Var) {
        let idx = var.index();
        if idx >= self.activity.len() {
            self.resize(idx + 1);
        }

        self.activity[idx] += self.increment;

        // Rescale if activity gets too large
        if self.activity[idx] > 1e100 {
            self.rescale();
        }

        // Move up in heap
        if self.heap_pos[idx] != usize::MAX {
            self.sift_up(self.heap_pos[idx]);
        }
    }

    /// Decay all activities
    pub fn decay(&mut self) {
        self.increment /= self.decay;
    }

    /// Pop the variable with highest activity
    pub fn pop_max(&mut self) -> Option<Var> {
        if self.heap.is_empty() {
            return None;
        }

        let max_var = self.heap[0];
        let last = self
            .heap
            .pop()
            .expect("heap non-empty after is_empty check");

        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.heap_pos[last.index()] = 0;
            self.sift_down(0);
        }

        self.heap_pos[max_var.index()] = usize::MAX;
        Some(max_var)
    }

    /// Insert a variable back into the heap
    pub fn insert(&mut self, var: Var) {
        let idx = var.index();
        if idx >= self.heap_pos.len() {
            self.resize(idx + 1);
        }

        if self.heap_pos[idx] == usize::MAX {
            let pos = self.heap.len();
            self.heap.push(var);
            self.heap_pos[idx] = pos;
            self.sift_up(pos);
        }
    }

    /// Check if a variable is in the heap
    #[must_use]
    pub fn contains(&self, var: Var) -> bool {
        let idx = var.index();
        idx < self.heap_pos.len() && self.heap_pos[idx] != usize::MAX
    }

    fn sift_up(&mut self, mut pos: usize) {
        let var = self.heap[pos];
        let act = self.activity[var.index()];

        while pos > 0 {
            let parent = (pos - 1) / 2;
            let parent_var = self.heap[parent];
            let parent_act = self.activity[parent_var.index()];

            if act <= parent_act {
                break;
            }

            self.heap[pos] = parent_var;
            self.heap_pos[parent_var.index()] = pos;
            pos = parent;
        }

        self.heap[pos] = var;
        self.heap_pos[var.index()] = pos;
    }

    fn sift_down(&mut self, mut pos: usize) {
        let var = self.heap[pos];
        let act = self.activity[var.index()];

        loop {
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;

            let mut max_pos = pos;
            let mut max_act = act;

            if left < self.heap.len() {
                let left_var = self.heap[left];
                let left_act = self.activity[left_var.index()];
                if left_act > max_act {
                    max_pos = left;
                    max_act = left_act;
                }
            }

            if right < self.heap.len() {
                let right_var = self.heap[right];
                let right_act = self.activity[right_var.index()];
                if right_act > max_act {
                    max_pos = right;
                }
            }

            if max_pos == pos {
                break;
            }

            let max_var = self.heap[max_pos];
            self.heap[pos] = max_var;
            self.heap_pos[max_var.index()] = pos;
            pos = max_pos;
        }

        self.heap[pos] = var;
        self.heap_pos[var.index()] = pos;
    }

    fn rescale(&mut self) {
        for act in &mut self.activity {
            *act *= 1e-100;
        }
        self.increment *= 1e-100;
    }

    fn resize(&mut self, num_vars: usize) {
        let old_len = self.activity.len();
        self.activity.resize(num_vars, 0.0);
        self.heap_pos.resize(num_vars, usize::MAX);

        // Add new variables to heap
        for i in old_len..num_vars {
            let var = Var::new(i as u32);
            let pos = self.heap.len();
            self.heap.push(var);
            self.heap_pos[i] = pos;
        }
    }

    /// Get the number of variables in the heap
    #[must_use]
    #[allow(dead_code)]
    pub fn heap_size(&self) -> usize {
        self.heap.len()
    }

    /// Clear the VSIDS state
    pub fn clear(&mut self) {
        for act in &mut self.activity {
            *act = 0.0;
        }
        self.increment = 1.0;
        self.heap.clear();
        for pos in &mut self.heap_pos {
            *pos = usize::MAX;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsids_basic() {
        let mut vsids = VSIDS::new(5);

        // All variables start with 0 activity
        assert_eq!(vsids.activity(Var::new(0)), 0.0);

        // Bump variable 2
        vsids.bump(Var::new(2));
        assert!(vsids.activity(Var::new(2)) > 0.0);

        // Variable 2 should be on top
        let max = vsids.pop_max().unwrap();
        assert_eq!(max, Var::new(2));
    }

    #[test]
    fn test_vsids_ordering() {
        let mut vsids = VSIDS::new(5);

        // Bump variables with different amounts
        vsids.bump(Var::new(0));
        vsids.bump(Var::new(1));
        vsids.bump(Var::new(1));
        vsids.bump(Var::new(2));
        vsids.bump(Var::new(2));
        vsids.bump(Var::new(2));

        // Should come out in order of activity
        assert_eq!(vsids.pop_max().unwrap(), Var::new(2));
        assert_eq!(vsids.pop_max().unwrap(), Var::new(1));
        assert_eq!(vsids.pop_max().unwrap(), Var::new(0));
    }
}
