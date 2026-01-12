//! Union-Find data structure for congruence closure

/// An undo entry for reverting a union operation
#[derive(Debug, Clone, Copy)]
pub struct UndoEntry {
    /// The node whose parent was changed (the loser that became a child)
    node: u32,
    /// The old parent value (should be the node itself since it was a root)
    old_parent: u32,
    /// The winner root (only valid if rank_incremented is true)
    winner: u32,
    /// Whether rank was incremented (only for the winner root)
    rank_incremented: bool,
}

/// Union-Find with path compression and union by rank
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// Parent pointers (root points to itself)
    parent: Vec<u32>,
    /// Rank for union by rank
    rank: Vec<u32>,
    /// Trail of undo entries for backtracking
    trail: Vec<UndoEntry>,
    /// Trail size at each decision level
    trail_limits: Vec<usize>,
}

impl UnionFind {
    /// Create a new Union-Find with n elements
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
            trail: Vec::new(),
            trail_limits: vec![0],
        }
    }

    /// Find the representative of an element with path compression
    #[inline]
    pub fn find(&mut self, mut x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }

        // Path compression
        while self.parent[x as usize] != root {
            let next = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = next;
        }

        root
    }

    /// Find the representative of an element without path compression (immutable)
    #[inline]
    #[must_use]
    pub fn find_no_compress(&self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            x = self.parent[x as usize];
        }
        x
    }

    /// Check if two elements are in the same set (immutable version)
    #[inline]
    #[must_use]
    pub fn same_no_compress(&self, x: u32, y: u32) -> bool {
        self.find_no_compress(x) == self.find_no_compress(y)
    }

    /// Union two elements, returns true if they were in different sets
    /// This version tracks the merge for backtracking
    pub fn union(&mut self, x: u32, y: u32) -> bool {
        let root_x = self.find_no_compress(x);
        let root_y = self.find_no_compress(y);

        if root_x == root_y {
            return false;
        }

        // Union by rank (track for undo)
        match self.rank[root_x as usize].cmp(&self.rank[root_y as usize]) {
            std::cmp::Ordering::Less => {
                // root_x becomes child of root_y
                self.trail.push(UndoEntry {
                    node: root_x,
                    old_parent: root_x,
                    winner: root_y,
                    rank_incremented: false,
                });
                self.parent[root_x as usize] = root_y;
            }
            std::cmp::Ordering::Greater => {
                // root_y becomes child of root_x
                self.trail.push(UndoEntry {
                    node: root_y,
                    old_parent: root_y,
                    winner: root_x,
                    rank_incremented: false,
                });
                self.parent[root_y as usize] = root_x;
            }
            std::cmp::Ordering::Equal => {
                // root_y becomes child of root_x, rank increases
                self.trail.push(UndoEntry {
                    node: root_y,
                    old_parent: root_y,
                    winner: root_x,
                    rank_incremented: true,
                });
                self.parent[root_y as usize] = root_x;
                self.rank[root_x as usize] += 1;
            }
        }

        true
    }

    /// Union two elements without tracking (for non-incremental use)
    pub fn union_no_trail(&mut self, x: u32, y: u32) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        match self.rank[root_x as usize].cmp(&self.rank[root_y as usize]) {
            std::cmp::Ordering::Less => {
                self.parent[root_x as usize] = root_y;
            }
            std::cmp::Ordering::Greater => {
                self.parent[root_y as usize] = root_x;
            }
            std::cmp::Ordering::Equal => {
                self.parent[root_y as usize] = root_x;
                self.rank[root_x as usize] += 1;
            }
        }

        true
    }

    /// Check if two elements are in the same set
    #[inline]
    pub fn same(&mut self, x: u32, y: u32) -> bool {
        self.find(x) == self.find(y)
    }

    /// Add a new element
    pub fn add(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    /// Get the number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Push a new decision level (save current trail position)
    pub fn push(&mut self) {
        self.trail_limits.push(self.trail.len());
    }

    /// Pop to previous decision level, undoing all unions since then
    pub fn pop(&mut self) {
        if let Some(limit) = self.trail_limits.pop() {
            // Undo all merges since the limit
            while self.trail.len() > limit {
                if let Some(entry) = self.trail.pop() {
                    // Restore the old parent
                    self.parent[entry.node as usize] = entry.old_parent;

                    // If rank was incremented, decrement it on the winner root
                    if entry.rank_incremented {
                        self.rank[entry.winner as usize] -= 1;
                    }
                }
            }
        }
    }

    /// Backtrack to a specific decision level
    pub fn backtrack_to(&mut self, level: usize) {
        while self.trail_limits.len() > level + 1 {
            self.pop();
        }
    }

    /// Get the current decision level
    #[must_use]
    pub fn decision_level(&self) -> usize {
        self.trail_limits.len().saturating_sub(1)
    }

    /// Get the trail size
    #[must_use]
    pub fn trail_size(&self) -> usize {
        self.trail.len()
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.parent.clear();
        self.rank.clear();
        self.trail.clear();
        self.trail_limits.clear();
        self.trail_limits.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);

        assert!(!uf.same(0, 1));
        assert!(uf.union(0, 1));
        assert!(uf.same(0, 1));

        assert!(!uf.same(1, 2));
        assert!(uf.union(1, 2));
        assert!(uf.same(0, 2));
    }

    #[test]
    fn test_union_find_redundant() {
        let mut uf = UnionFind::new(3);

        assert!(uf.union(0, 1));
        assert!(!uf.union(0, 1)); // Already in same set
        assert!(!uf.union(1, 0)); // Already in same set
    }

    #[test]
    fn test_union_find_add() {
        let mut uf = UnionFind::new(2);

        let x = uf.add();
        assert_eq!(x, 2);
        assert!(!uf.same(0, x));

        uf.union(0, x);
        assert!(uf.same(0, x));
    }

    #[test]
    fn test_union_find_push_pop() {
        let mut uf = UnionFind::new(5);

        // Initial state: all separate
        assert!(!uf.same_no_compress(0, 1));
        assert!(!uf.same_no_compress(2, 3));

        // Level 0: merge 0 and 1
        uf.union(0, 1);
        assert!(uf.same_no_compress(0, 1));

        // Push to level 1
        uf.push();

        // Level 1: merge 2 and 3
        uf.union(2, 3);
        assert!(uf.same_no_compress(2, 3));

        // Also merge 0 with 2
        uf.union(0, 2);
        assert!(uf.same_no_compress(0, 2));
        assert!(uf.same_no_compress(1, 3)); // Transitive

        // Pop back to level 0
        uf.pop();

        // 0 and 1 should still be merged
        assert!(uf.same_no_compress(0, 1));

        // 2 and 3 should be separate again
        assert!(!uf.same_no_compress(2, 3));

        // 0 and 2 should be separate again
        assert!(!uf.same_no_compress(0, 2));
    }

    #[test]
    fn test_union_find_multiple_levels() {
        let mut uf = UnionFind::new(6);

        // Level 0
        uf.union(0, 1);

        uf.push(); // Level 1
        uf.union(2, 3);

        uf.push(); // Level 2
        uf.union(4, 5);
        uf.union(0, 4); // Merge two groups

        assert!(uf.same_no_compress(0, 1));
        assert!(uf.same_no_compress(2, 3));
        assert!(uf.same_no_compress(4, 5));
        assert!(uf.same_no_compress(0, 5)); // Through 0-4-5

        // Pop to level 1
        uf.pop();
        assert!(uf.same_no_compress(0, 1));
        assert!(uf.same_no_compress(2, 3));
        assert!(!uf.same_no_compress(4, 5)); // Undone
        assert!(!uf.same_no_compress(0, 4)); // Undone

        // Pop to level 0
        uf.pop();
        assert!(uf.same_no_compress(0, 1));
        assert!(!uf.same_no_compress(2, 3)); // Undone
    }

    #[test]
    fn test_union_find_backtrack_to() {
        let mut uf = UnionFind::new(4);

        uf.union(0, 1); // Level 0

        uf.push(); // Level 1
        uf.union(1, 2);

        uf.push(); // Level 2
        uf.union(2, 3);

        assert!(uf.same_no_compress(0, 3));

        // Backtrack to level 0
        uf.backtrack_to(0);

        assert!(uf.same_no_compress(0, 1));
        assert!(!uf.same_no_compress(1, 2));
        assert!(!uf.same_no_compress(2, 3));
    }
}
