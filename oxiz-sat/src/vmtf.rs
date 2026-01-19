//! Variable Move-To-Front (VMTF) branching heuristic
//!
//! VMTF is a modern branching heuristic used in state-of-the-art SAT solvers
//! like Kissat and CaDiCaL. It maintains a doubly-linked list of variables,
//! and when a variable is involved in a conflict, it's moved to the front.
//!
//! Compared to VSIDS:
//! - Simpler implementation (no heap operations)
//! - Lower overhead (constant-time operations)
//! - Often competitive or better performance
//! - No periodic rescaling needed

use crate::literal::Var;

/// Node in the VMTF doubly-linked list
#[derive(Debug, Clone, Copy)]
struct VmtfNode {
    /// Previous variable in the list (None if this is the head)
    prev: Option<Var>,
    /// Next variable in the list (None if this is the tail)
    next: Option<Var>,
    /// Timestamp when this variable was last moved to front
    timestamp: u64,
}

impl VmtfNode {
    fn new() -> Self {
        Self {
            prev: None,
            next: None,
            timestamp: 0,
        }
    }
}

/// Variable Move-To-Front (VMTF) branching heuristic
///
/// Maintains a doubly-linked list of all variables. When a variable
/// is bumped (involved in a conflict), it's moved to the front of the list.
/// Variable selection picks the first unassigned variable from the front.
#[derive(Debug, Clone)]
pub struct VMTF {
    /// Doubly-linked list nodes for each variable
    nodes: Vec<VmtfNode>,
    /// Head of the list (most recently bumped variable)
    head: Option<Var>,
    /// Tail of the list (least recently bumped variable)
    tail: Option<Var>,
    /// Current timestamp (incremented on each bump)
    timestamp: u64,
    /// Queue head for iteration during variable selection
    queue_head: Option<Var>,
}

impl VMTF {
    /// Create a new VMTF instance with the given number of variables
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        let mut vmtf = Self {
            nodes: vec![VmtfNode::new(); num_vars],
            head: None,
            tail: None,
            timestamp: 0,
            queue_head: None,
        };

        // Initialize the doubly-linked list with all variables
        if num_vars > 0 {
            vmtf.head = Some(Var::new(0));
            vmtf.tail = Some(Var::new((num_vars - 1) as u32));

            for i in 0..num_vars {
                vmtf.nodes[i].prev = if i > 0 {
                    Some(Var::new((i - 1) as u32))
                } else {
                    None
                };
                vmtf.nodes[i].next = if i < num_vars - 1 {
                    Some(Var::new((i + 1) as u32))
                } else {
                    None
                };
                vmtf.nodes[i].timestamp = i as u64;
            }
        }

        vmtf.queue_head = vmtf.head;
        vmtf
    }

    /// Resize the VMTF to accommodate more variables
    pub fn resize(&mut self, new_num_vars: usize) {
        let old_num_vars = self.nodes.len();
        if new_num_vars <= old_num_vars {
            return;
        }

        // Add new nodes
        self.nodes.resize(new_num_vars, VmtfNode::new());

        // Link new variables to the list
        if old_num_vars == 0 {
            // Starting from empty, same as new()
            self.head = Some(Var::new(0));
            self.tail = Some(Var::new((new_num_vars - 1) as u32));

            for i in 0..new_num_vars {
                self.nodes[i].prev = if i > 0 {
                    Some(Var::new((i - 1) as u32))
                } else {
                    None
                };
                self.nodes[i].next = if i < new_num_vars - 1 {
                    Some(Var::new((i + 1) as u32))
                } else {
                    None
                };
                self.nodes[i].timestamp = self.timestamp;
                self.timestamp += 1;
            }
        } else {
            // Append new variables to the existing tail
            let old_tail = self
                .tail
                .expect("tail exists when extending non-empty list");
            let first_new = Var::new(old_num_vars as u32);

            // Connect old tail to first new variable
            self.nodes[old_tail.index()].next = Some(first_new);

            for i in old_num_vars..new_num_vars {
                self.nodes[i].prev = if i == old_num_vars {
                    Some(old_tail)
                } else {
                    Some(Var::new((i - 1) as u32))
                };
                self.nodes[i].next = if i < new_num_vars - 1 {
                    Some(Var::new((i + 1) as u32))
                } else {
                    None
                };
                self.nodes[i].timestamp = self.timestamp;
                self.timestamp += 1;
            }

            // Update tail
            self.tail = Some(Var::new((new_num_vars - 1) as u32));
        }
    }

    /// Bump a variable (move it to the front of the list)
    ///
    /// This is called when a variable is involved in a conflict
    pub fn bump(&mut self, var: Var) {
        let idx = var.index();
        if idx >= self.nodes.len() {
            return;
        }

        // If already at head, just update timestamp
        if Some(var) == self.head {
            self.timestamp += 1;
            self.nodes[idx].timestamp = self.timestamp;
            return;
        }

        // Remove from current position
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;

        if let Some(prev_var) = prev {
            self.nodes[prev_var.index()].next = next;
        }
        if let Some(next_var) = next {
            self.nodes[next_var.index()].prev = prev;
        }

        // Update tail if we're removing the tail
        if Some(var) == self.tail {
            self.tail = prev;
        }

        // Insert at head
        self.nodes[idx].prev = None;
        self.nodes[idx].next = self.head;
        self.timestamp += 1;
        self.nodes[idx].timestamp = self.timestamp;

        if let Some(old_head) = self.head {
            self.nodes[old_head.index()].prev = Some(var);
        }
        self.head = Some(var);

        // If list was empty, update tail
        if self.tail.is_none() {
            self.tail = Some(var);
        }

        // Reset queue head when we bump
        self.queue_head = self.head;
    }

    /// Select the next unassigned variable from the front of the list
    ///
    /// Returns `None` when reaching the end of the list.
    /// Call `reset_queue()` to start from the beginning again.
    #[must_use]
    pub fn select(&mut self) -> Option<Var> {
        self.queue_head
    }

    /// Advance the queue to the next variable
    ///
    /// This is called after selecting a variable that was assigned.
    /// The caller should keep calling `select()` and `advance()` until
    /// `select()` returns an unassigned variable or `None`.
    pub fn advance(&mut self) {
        if let Some(current) = self.queue_head {
            self.queue_head = self.nodes[current.index()].next;
        }
    }

    /// Reset the queue to start iteration from the beginning
    pub fn reset_queue(&mut self) {
        self.queue_head = self.head;
    }

    /// Get the activity/timestamp of a variable (for statistics)
    #[must_use]
    pub fn activity(&self, var: Var) -> u64 {
        let idx = var.index();
        if idx < self.nodes.len() {
            self.nodes[idx].timestamp
        } else {
            0
        }
    }

    /// Get statistics about the VMTF state
    #[must_use]
    pub fn stats(&self) -> VmtfStats {
        VmtfStats {
            num_vars: self.nodes.len(),
            current_timestamp: self.timestamp,
        }
    }
}

/// Statistics for VMTF
#[derive(Debug, Clone, Copy)]
pub struct VmtfStats {
    /// Number of variables being tracked
    pub num_vars: usize,
    /// Current timestamp value
    pub current_timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmtf_creation() {
        let vmtf = VMTF::new(5);
        assert_eq!(vmtf.nodes.len(), 5);
        assert_eq!(vmtf.head, Some(Var::new(0)));
        assert_eq!(vmtf.tail, Some(Var::new(4)));
    }

    #[test]
    fn test_vmtf_select_sequential() {
        let mut vmtf = VMTF::new(5);

        // Initially should select variables in order 0, 1, 2, 3, 4
        assert_eq!(vmtf.select(), Some(Var::new(0)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(1)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(2)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(3)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(4)));
        vmtf.advance();
        assert_eq!(vmtf.select(), None);
    }

    #[test]
    fn test_vmtf_bump() {
        let mut vmtf = VMTF::new(5);

        // Bump variable 3 - it should move to front
        vmtf.bump(Var::new(3));
        vmtf.reset_queue();

        // Now should select 3 first
        assert_eq!(vmtf.select(), Some(Var::new(3)));
        vmtf.advance();

        // Then the rest in original order (excluding 3)
        assert_eq!(vmtf.select(), Some(Var::new(0)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(1)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(2)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(4)));
    }

    #[test]
    fn test_vmtf_multiple_bumps() {
        let mut vmtf = VMTF::new(5);

        // Bump variables in sequence: 2, 4, 1
        vmtf.bump(Var::new(2));
        vmtf.bump(Var::new(4));
        vmtf.bump(Var::new(1));
        vmtf.reset_queue();

        // Should now select in order: 1, 4, 2, 0, 3
        // (most recently bumped first)
        assert_eq!(vmtf.select(), Some(Var::new(1)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(4)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(2)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(0)));
        vmtf.advance();
        assert_eq!(vmtf.select(), Some(Var::new(3)));
    }

    #[test]
    fn test_vmtf_resize() {
        let mut vmtf = VMTF::new(3);
        vmtf.resize(6);

        assert_eq!(vmtf.nodes.len(), 6);
        vmtf.reset_queue();

        // Should be able to select all 6 variables
        for _i in 0..6 {
            assert!(vmtf.select().is_some());
            vmtf.advance();
        }
        assert_eq!(vmtf.select(), None);
    }

    #[test]
    fn test_vmtf_activity() {
        let mut vmtf = VMTF::new(5);

        let v0_activity = vmtf.activity(Var::new(0));
        vmtf.bump(Var::new(0));
        let v0_activity_after = vmtf.activity(Var::new(0));

        // Activity should increase after bump
        assert!(v0_activity_after > v0_activity);
    }

    #[test]
    fn test_vmtf_stats() {
        let mut vmtf = VMTF::new(10);
        let stats = vmtf.stats();

        assert_eq!(stats.num_vars, 10);

        // Bump a variable and check timestamp increases
        vmtf.bump(Var::new(0));
        let stats_after = vmtf.stats();
        assert!(stats_after.current_timestamp > stats.current_timestamp);
    }

    #[test]
    fn test_vmtf_bump_head() {
        let mut vmtf = VMTF::new(5);

        // Bump the head variable
        let head_var = vmtf.head.unwrap();
        let timestamp_before = vmtf.activity(head_var);

        vmtf.bump(head_var);

        // Timestamp should increase even though it's already at head
        assert!(vmtf.activity(head_var) > timestamp_before);
        assert_eq!(vmtf.head, Some(head_var));
    }

    #[test]
    fn test_vmtf_bump_tail() {
        let mut vmtf = VMTF::new(5);

        // Bump the tail variable
        let tail_var = Var::new(4);
        vmtf.bump(tail_var);

        // Tail should move to head
        assert_eq!(vmtf.head, Some(tail_var));
        // New tail should be variable 3
        assert_eq!(vmtf.tail, Some(Var::new(3)));
    }
}
