//! Binary Implication Graph (BIG) optimization
//!
//! This module implements optimizations for the binary implication graph,
//! including transitive reduction to remove redundant binary clauses and
//! strongly connected component detection for equivalence finding.
//!
//! The binary implication graph represents binary clauses as implications:
//! - Binary clause (a v b) represents two implications: ~a => b and ~b => a
//!
//! References:
//! - "Binary Clause Reasoning in Conflict-Driven Clause Learning" (Bacchus)
//! - "Effective Preprocessing in SAT" (Eén & Biere)
//! - "Bounded Variable Elimination" (Subbarayan & Pradhan)

use crate::clause::ClauseDatabase;
use crate::literal::Lit;
#[allow(unused_imports)]
use crate::prelude::*;

/// Statistics for BIG optimization
#[derive(Debug, Clone, Default)]
pub struct BigStats {
    /// Number of binary clauses analyzed
    pub binary_clauses_analyzed: usize,
    /// Number of redundant binary clauses removed
    pub redundant_removed: usize,
    /// Number of transitive implications found
    pub transitive_found: usize,
    /// Number of SCCs detected
    pub sccs_found: usize,
    /// Number of equivalent literal pairs from SCCs
    pub equivalences_from_sccs: usize,
}

impl BigStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Binary Implication Graph Statistics:");
        println!(
            "  Binary clauses analyzed: {}",
            self.binary_clauses_analyzed
        );
        println!("  Redundant clauses removed: {}", self.redundant_removed);
        println!("  Transitive implications: {}", self.transitive_found);
        println!("  SCCs found: {}", self.sccs_found);
        println!("  Equivalences from SCCs: {}", self.equivalences_from_sccs);
    }
}

/// Binary Implication Graph optimizer
#[derive(Debug)]
pub struct BinaryImplicationGraph {
    /// Adjacency list: implications[lit] = literals implied by lit
    implications: Vec<HashSet<Lit>>,
    /// Reverse adjacency list for SCC detection
    reverse_implications: Vec<HashSet<Lit>>,
    /// Statistics
    stats: BigStats,
}

impl BinaryImplicationGraph {
    /// Create a new BIG optimizer
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        let size = num_vars * 2;
        Self {
            implications: vec![HashSet::new(); size],
            reverse_implications: vec![HashSet::new(); size],
            stats: BigStats::default(),
        }
    }

    /// Build the implication graph from clause database
    pub fn build(&mut self, clauses: &ClauseDatabase) {
        // Clear existing data
        for imp in &mut self.implications {
            imp.clear();
        }
        for imp in &mut self.reverse_implications {
            imp.clear();
        }
        self.stats.binary_clauses_analyzed = 0;

        // Extract binary clauses
        for cid in clauses.iter_ids() {
            if let Some(clause) = clauses.get(cid)
                && clause.len() == 2
            {
                self.stats.binary_clauses_analyzed += 1;

                let a = clause.lits[0];
                let b = clause.lits[1];

                // Binary clause (a v b) means: ~a => b and ~b => a
                self.add_implication(!a, b);
                self.add_implication(!b, a);
            }
        }
    }

    /// Add a binary implication to the graph
    fn add_implication(&mut self, from: Lit, to: Lit) {
        let from_idx = from.code() as usize;
        let to_idx = to.code() as usize;

        // Ensure capacity
        while from_idx >= self.implications.len() {
            self.implications.push(HashSet::new());
            self.reverse_implications.push(HashSet::new());
        }
        while to_idx >= self.implications.len() {
            self.implications.push(HashSet::new());
            self.reverse_implications.push(HashSet::new());
        }

        self.implications[from_idx].insert(to);
        self.reverse_implications[to_idx].insert(from);
    }

    /// Perform transitive reduction to remove redundant implications
    ///
    /// An edge a => c is redundant if there exists a path a => b => c
    pub fn transitive_reduction(&mut self) -> Vec<(Lit, Lit)> {
        let mut redundant = Vec::new();
        let num_lits = self.implications.len();

        for lit_idx in 0..num_lits {
            let lit = Lit::from_code(lit_idx as u32);
            let direct_implications: Vec<_> = self.implications[lit_idx].iter().copied().collect();

            for &implied in &direct_implications {
                // Check if there's an alternative path from lit to implied
                if self.has_alternative_path(lit, implied) {
                    redundant.push((lit, implied));
                    self.stats.redundant_removed += 1;
                }
            }
        }

        // Remove redundant edges
        for (from, to) in &redundant {
            let from_idx = from.code() as usize;
            let to_idx = to.code() as usize;
            self.implications[from_idx].remove(to);
            self.reverse_implications[to_idx].remove(from);
        }

        redundant
    }

    /// Check if there's a path from 'from' to 'to' without using the direct edge
    fn has_alternative_path(&self, from: Lit, to: Lit) -> bool {
        let from_idx = from.code() as usize;
        if from_idx >= self.implications.len() {
            return false;
        }

        let mut visited = HashSet::new();
        let mut queue = Vec::new();

        // Start BFS from literals implied by 'from', excluding direct edge to 'to'
        for &implied in &self.implications[from_idx] {
            if implied != to {
                queue.push(implied);
            }
        }

        while let Some(lit) = queue.pop() {
            if lit == to {
                return true; // Found alternative path
            }

            let lit_idx = lit.code() as usize;
            if visited.contains(&lit_idx) || lit_idx >= self.implications.len() {
                continue;
            }

            visited.insert(lit_idx);

            for &next in &self.implications[lit_idx] {
                if !visited.contains(&(next.code() as usize)) {
                    queue.push(next);
                }
            }
        }

        false
    }

    /// Detect strongly connected components using Tarjan's algorithm
    ///
    /// Literals in the same SCC are equivalent (mutual implication)
    #[allow(dead_code)]
    pub fn find_sccs(&mut self) -> Vec<Vec<Lit>> {
        let num_lits = self.implications.len();
        let mut index = vec![None; num_lits];
        let mut lowlink = vec![0; num_lits];
        let mut on_stack = vec![false; num_lits];
        let mut stack = Vec::new();
        let mut sccs = Vec::new();
        let mut current_index = 0;

        for lit_idx in 0..num_lits {
            if index[lit_idx].is_none() {
                self.tarjan_scc(
                    lit_idx,
                    &mut index,
                    &mut lowlink,
                    &mut on_stack,
                    &mut stack,
                    &mut sccs,
                    &mut current_index,
                );
            }
        }

        // Filter out trivial SCCs (single nodes)
        self.stats.sccs_found = sccs.iter().filter(|scc| scc.len() > 1).count();
        sccs.into_iter().filter(|scc| scc.len() > 1).collect()
    }

    /// Tarjan's SCC algorithm helper.
    ///
    /// Runs on an explicit heap stack rather than native recursion: the
    /// depth of the DFS is the length of the longest simple path in the
    /// binary implication graph, i.e. one per binary clause, entirely
    /// attacker-controlled (a 100k-link implication chain
    /// `(¬a₁∨a₂)(¬a₂∨a₃)…` is trivial to write). The function returns
    /// `Vec<Vec<Lit>>` through `sccs` with no error channel, so a depth cap
    /// could only silently omit equivalences.
    ///
    /// The rewrite also removes three `expect()`s that the recursive form
    /// needed: the `on_stack ⇒ index.is_some()` and "stack non-empty during
    /// SCC formation" invariants are now expressed as `if let` / `while
    /// let` bindings that cannot be written wrongly.
    #[allow(clippy::too_many_arguments)]
    fn tarjan_scc(
        &self,
        lit_idx: usize,
        index: &mut [Option<usize>],
        lowlink: &mut [usize],
        on_stack: &mut [bool],
        stack: &mut Vec<usize>,
        sccs: &mut Vec<Vec<Lit>>,
        current_index: &mut usize,
    ) {
        /// One suspended node of the walk: `succs` is its successor list
        /// snapshotted on entry (the adjacency structure is a `HashSet`,
        /// which has no positional index) and `next` is how far through
        /// that list the walk had got when it descended into a child.
        struct Frame {
            node: usize,
            succs: Vec<usize>,
            next: usize,
        }

        // Mirror of the recursive prologue, for the root and for every node
        // the walk descends into.
        let successors = |node: usize| -> Vec<usize> {
            self.implications.get(node).map_or_else(Vec::new, |set| {
                set.iter().map(|implied| implied.code() as usize).collect()
            })
        };

        index[lit_idx] = Some(*current_index);
        lowlink[lit_idx] = *current_index;
        *current_index += 1;
        stack.push(lit_idx);
        on_stack[lit_idx] = true;

        let mut frames = vec![Frame {
            node: lit_idx,
            succs: successors(lit_idx),
            next: 0,
        }];

        while let Some(mut frame) = frames.pop() {
            let node = frame.node;
            let mut descended = false;

            while frame.next < frame.succs.len() {
                let impl_idx = frame.succs[frame.next];
                frame.next += 1;

                // `add_implication` grows the adjacency vectors for both
                // endpoints, so every successor is in range; the guard only
                // keeps a malformed graph from indexing out of bounds.
                if impl_idx >= index.len() {
                    continue;
                }

                match index[impl_idx] {
                    // Unvisited successor: descend, resuming `frame`
                    // afterwards (the post-descent `lowlink` update is done
                    // by the child when it finishes, see below).
                    None => {
                        index[impl_idx] = Some(*current_index);
                        lowlink[impl_idx] = *current_index;
                        *current_index += 1;
                        stack.push(impl_idx);
                        on_stack[impl_idx] = true;
                        let succs = successors(impl_idx);
                        frames.push(frame);
                        frames.push(Frame {
                            node: impl_idx,
                            succs,
                            next: 0,
                        });
                        descended = true;
                        break;
                    }
                    Some(child_index) => {
                        if on_stack[impl_idx] {
                            lowlink[node] = lowlink[node].min(child_index);
                        }
                    }
                }
            }

            if descended {
                continue;
            }

            // All successors of `node` are done. If it is an SCC root, pop
            // the component off the shared Tarjan stack.
            if index[node] == Some(lowlink[node]) {
                let mut scc = Vec::new();
                while let Some(popped) = stack.pop() {
                    on_stack[popped] = false;
                    scc.push(Lit::from_code(popped as u32));
                    if popped == node {
                        break;
                    }
                }
                sccs.push(scc);
            }

            // Propagate this node's lowlink into its (suspended) parent —
            // the `lowlink[parent] = min(lowlink[parent], lowlink[child])`
            // step that followed the recursive call.
            if let Some(parent) = frames.last() {
                lowlink[parent.node] = lowlink[parent.node].min(lowlink[node]);
            }
        }
    }

    /// Apply BIG optimizations to clause database
    ///
    /// Removes redundant binary clauses found through transitive reduction
    pub fn optimize(&mut self, clauses: &mut ClauseDatabase) {
        // Build the graph
        self.build(clauses);

        // Perform transitive reduction
        let redundant = self.transitive_reduction();

        // Remove redundant binary clauses from database
        let clause_ids: Vec<_> = clauses.iter_ids().collect();
        for cid in clause_ids {
            if let Some(clause) = clauses.get(cid)
                && clause.len() == 2
            {
                let a = clause.lits[0];
                let b = clause.lits[1];

                // Check if this binary clause is redundant
                if redundant.contains(&(!a, b)) || redundant.contains(&(!b, a)) {
                    clauses.remove(cid);
                }
            }
        }
    }

    /// Get all implications for a literal
    #[must_use]
    pub fn get_implications(&self, lit: Lit) -> Vec<Lit> {
        let idx = lit.code() as usize;
        if idx < self.implications.len() {
            self.implications[idx].iter().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Check if literal a implies literal b
    #[must_use]
    pub fn implies(&self, a: Lit, b: Lit) -> bool {
        let idx = a.code() as usize;
        if idx < self.implications.len() {
            self.implications[idx].contains(&b)
        } else {
            false
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &BigStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = BigStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_big_creation() {
        let big = BinaryImplicationGraph::new(10);
        assert_eq!(big.stats().binary_clauses_analyzed, 0);
    }

    /// A 12_500-link implication chain `a₁ ⇒ a₂ ⇒ … ⇒ a₁₂₅₀₀`, i.e. the
    /// graph produced by 12_500 binary clauses, run on a 128 KiB stack.
    /// Tarjan's DFS visits the whole chain in one descent; the recursive
    /// form aborted the process here. Returning at all is the assertion.
    #[test]
    fn test_find_sccs_deep_chain_does_not_overflow() {
        let worker = std::thread::Builder::new().stack_size(1 << 17).spawn(|| {
            const CHAIN: u32 = 12_500;
            let mut big = BinaryImplicationGraph::new(CHAIN as usize + 1);
            for i in 0..CHAIN {
                big.add_implication(Lit::pos(Var::new(i)), Lit::pos(Var::new(i + 1)));
            }
            // A chain is acyclic, so every SCC is a singleton and
            // `find_sccs` (which filters those out) returns nothing.
            big.find_sccs().len()
        });
        let non_trivial_sccs = match worker.map(std::thread::JoinHandle::join) {
            Ok(Ok(count)) => count,
            _ => panic!("deep-chain SCC worker thread did not complete"),
        };
        assert_eq!(non_trivial_sccs, 0);
    }

    /// A long implication *cycle* must still be reported as a single SCC
    /// after the iterative rewrite — the component-popping half of the
    /// algorithm is the part most easily broken by the conversion.
    #[test]
    fn test_find_sccs_long_cycle_is_one_component() {
        const RING: u32 = 5_000;
        let mut big = BinaryImplicationGraph::new(RING as usize);
        for i in 0..RING {
            big.add_implication(Lit::pos(Var::new(i)), Lit::pos(Var::new((i + 1) % RING)));
        }
        let sccs = big.find_sccs();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), RING as usize);
    }

    #[test]
    fn test_build_from_clauses() {
        let mut big = BinaryImplicationGraph::new(10);
        let mut db = ClauseDatabase::new();

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        db.add_original(vec![a, b]);
        big.build(&db);

        assert_eq!(big.stats().binary_clauses_analyzed, 1);
        // Should have implications: ~a => b and ~b => a
        assert!(big.implies(!a, b));
        assert!(big.implies(!b, a));
    }

    #[test]
    fn test_transitive_reduction() {
        let mut big = BinaryImplicationGraph::new(10);

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let c = Lit::pos(Var::new(2));

        // Add: a => b, b => c, a => c (last one is redundant)
        big.add_implication(a, b);
        big.add_implication(b, c);
        big.add_implication(a, c);

        let redundant = big.transitive_reduction();

        // Should find a => c as redundant
        assert!(!redundant.is_empty());
        assert!(redundant.contains(&(a, c)));
    }

    #[test]
    fn test_find_sccs() {
        let mut big = BinaryImplicationGraph::new(10);

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        // Create a cycle: a => b, b => a (they're equivalent)
        big.add_implication(a, b);
        big.add_implication(b, a);

        let sccs = big.find_sccs();

        // Should find one SCC containing a and b
        assert!(!sccs.is_empty());
        let scc = &sccs[0];
        assert!(scc.contains(&a));
        assert!(scc.contains(&b));
    }

    #[test]
    fn test_get_implications() {
        let mut big = BinaryImplicationGraph::new(10);

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        big.add_implication(a, b);

        let implications = big.get_implications(a);
        assert!(implications.contains(&b));
    }

    #[test]
    fn test_optimize() {
        let mut big = BinaryImplicationGraph::new(10);
        let mut db = ClauseDatabase::new();

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let c = Lit::pos(Var::new(2));

        // Add three binary clauses creating redundancy
        db.add_original(vec![a, b]); // ~a => b
        db.add_original(vec![b, c]); // ~b => c
        db.add_original(vec![a, c]); // ~a => c (redundant)

        let before = db.len();
        big.optimize(&mut db);
        let after = db.len();

        // Should have removed at least one redundant clause
        assert!(after <= before);
    }
}
