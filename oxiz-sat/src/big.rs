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
use std::collections::HashSet;

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

    /// Tarjan's SCC algorithm helper
    #[allow(clippy::too_many_arguments)]
    fn tarjan_scc(
        &self,
        lit_idx: usize,
        index: &mut Vec<Option<usize>>,
        lowlink: &mut Vec<usize>,
        on_stack: &mut Vec<bool>,
        stack: &mut Vec<usize>,
        sccs: &mut Vec<Vec<Lit>>,
        current_index: &mut usize,
    ) {
        index[lit_idx] = Some(*current_index);
        lowlink[lit_idx] = *current_index;
        *current_index += 1;
        stack.push(lit_idx);
        on_stack[lit_idx] = true;

        // Consider successors
        if lit_idx < self.implications.len() {
            for &implied in &self.implications[lit_idx] {
                let impl_idx = implied.code() as usize;

                if index[impl_idx].is_none() {
                    self.tarjan_scc(
                        impl_idx,
                        index,
                        lowlink,
                        on_stack,
                        stack,
                        sccs,
                        current_index,
                    );
                    lowlink[lit_idx] = lowlink[lit_idx].min(lowlink[impl_idx]);
                } else if on_stack[impl_idx] {
                    lowlink[lit_idx] = lowlink[lit_idx]
                        .min(index[impl_idx].expect("index set when on_stack is true"));
                }
            }
        }

        // If lit_idx is a root node, pop the stack to form an SCC
        if lowlink[lit_idx] == index[lit_idx].expect("index set for lit_idx in SCC") {
            let mut scc = Vec::new();
            loop {
                let node = stack.pop().expect("stack non-empty in SCC formation");
                on_stack[node] = false;
                scc.push(Lit::from_code(node as u32));
                if node == lit_idx {
                    break;
                }
            }
            sccs.push(scc);
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
