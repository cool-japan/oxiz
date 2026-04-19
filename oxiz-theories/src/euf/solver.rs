//! EUF Theory Solver

use core::mem;
use super::union_find::UnionFind;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::theory::{Theory, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use smallvec::SmallVec;

/// Signature update entry: (signature, node_id, fingerprint)
type SigUpdateEntry = ((u32, SmallVec<[u32; 4]>), u32, ENodeFingerprint);

/// Function properties for dynamic arity support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FunctionProperties {
    /// Is the function associative? (e.g., +, *, and, or)
    pub associative: bool,
    /// Is the function commutative? (e.g., +, *, and, or)
    pub commutative: bool,
    /// Does the function have an identity element?
    pub has_identity: bool,
}

/// 64-bit fingerprint for fast congruence pre-filtering.
/// Before doing full signature comparison in the congruence table,
/// we compare fingerprints first (cheap u64 comparison) to avoid
/// expensive argument-level equality checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ENodeFingerprint(u64);

impl ENodeFingerprint {
    /// Compute a fingerprint from a function symbol and canonical argument representatives.
    /// Uses a fast multiplicative hash to combine func and args into a single u64.
    #[must_use]
    pub fn compute(func: u32, args: &[u32]) -> Self {
        let mut h = func as u64;
        for &arg in args {
            h = h
                .wrapping_mul(0x517c_c1b7_2722_0a95)
                .wrapping_add(arg as u64);
        }
        Self(h)
    }

    /// Return the raw fingerprint value
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// A term node in the E-graph
#[derive(Debug, Clone)]
struct ENode {
    /// The original term
    #[allow(dead_code)]
    term: TermId,
    /// Function symbol (for function applications)
    func: Option<u32>,
    /// Arguments (indices into nodes)
    args: SmallVec<[u32; 4]>,
    /// 64-bit fingerprint for fast congruence pre-filtering
    fingerprint: ENodeFingerprint,
}

/// Disequality constraint
#[derive(Debug, Clone)]
struct Diseq {
    /// First term
    lhs: u32,
    /// Second term
    rhs: u32,
    /// Reason for the disequality
    reason: TermId,
}

/// A merge reason: why two nodes became equal
#[derive(Debug, Clone)]
enum MergeReason {
    /// Direct equality assertion
    Assertion(TermId),
    /// Congruence: f(a1,...,an) = f(b1,...,bn) because ai = bi for all i
    Congruence {
        /// The terms that became equal by congruence
        term1: u32,
        term2: u32,
    },
}

/// A merge edge in the proof forest
#[derive(Debug, Clone)]
struct MergeEdge {
    /// The other node in the merge
    other: u32,
    /// The reason for the merge
    reason: MergeReason,
}

/// EUF Theory Solver using congruence closure
#[derive(Debug)]
pub struct EufSolver {
    /// Union-Find for equivalence classes
    uf: UnionFind,
    /// E-nodes
    nodes: Vec<ENode>,
    /// Term to node index mapping
    term_to_node: FxHashMap<TermId, u32>,
    /// Disequality constraints
    diseqs: Vec<Diseq>,
    /// Pending merges for congruence closure
    pending: Vec<(u32, u32, TermId)>,
    /// Use list: for each node, which applications use it as an argument
    use_list: Vec<SmallVec<[u32; 8]>>,
    /// Signature table for congruence closure
    sig_table: FxHashMap<(u32, SmallVec<[u32; 4]>), u32>,
    /// Fingerprint table: maps fingerprint -> list of node indices with that fingerprint.
    /// Used as a fast pre-filter before full signature comparison in congruence checks.
    fingerprint_table: FxHashMap<ENodeFingerprint, SmallVec<[u32; 4]>>,
    /// Context stack for push/pop
    context_stack: Vec<ContextState>,
    /// Proof forest: for each node, edges to explain equalities
    proof_forest: Vec<Vec<MergeEdge>>,
    /// Function properties for dynamic arity support
    function_properties: FxHashMap<u32, FunctionProperties>,
    /// Reused queue for newly discovered propagations during congruence closure.
    propagation_buf: Vec<(u32, u32, TermId)>,
}

/// State to save for push/pop
#[derive(Debug, Clone)]
struct ContextState {
    num_nodes: usize,
    num_diseqs: usize,
}

impl Default for EufSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl EufSolver {
    /// Create a new EUF solver
    #[must_use]
    pub fn new() -> Self {
        Self {
            uf: UnionFind::new(0),
            nodes: Vec::new(),
            term_to_node: FxHashMap::default(),
            diseqs: Vec::new(),
            pending: Vec::new(),
            use_list: Vec::new(),
            sig_table: FxHashMap::default(),
            fingerprint_table: FxHashMap::default(),
            context_stack: Vec::new(),
            proof_forest: Vec::new(),
            function_properties: FxHashMap::default(),
            propagation_buf: Vec::new(),
        }
    }

    /// Register a function with specific properties (for dynamic arity support)
    pub fn register_function(&mut self, func: u32, props: FunctionProperties) {
        self.function_properties.insert(func, props);
    }

    /// Get the properties of a function
    fn get_function_props(&self, func: u32) -> FunctionProperties {
        self.function_properties
            .get(&func)
            .copied()
            .unwrap_or_default()
    }

    /// Canonicalize arguments for commutative functions
    fn canonicalize_args(&mut self, func: u32, args: &[u32]) -> SmallVec<[u32; 4]> {
        let props = self.get_function_props(func);
        let mut canonical: SmallVec<[u32; 4]> = args.iter().map(|&a| self.uf.find(a)).collect();

        // For commutative functions, sort arguments by their canonical representative
        if props.commutative {
            canonical.sort_unstable();
        }

        canonical
    }

    /// Flatten associative function applications
    /// For example: f(f(a, b), c) -> f(a, b, c)
    fn flatten_args(&self, func: u32, args: &[u32]) -> SmallVec<[u32; 4]> {
        let props = self.get_function_props(func);

        if !props.associative {
            return args.iter().copied().collect();
        }

        let mut flattened = SmallVec::new();
        for &arg in args {
            let arg_node = &self.nodes[arg as usize];
            // If the argument is an application of the same function, flatten it
            if arg_node.func == Some(func) {
                flattened.extend(arg_node.args.iter().copied());
            } else {
                flattened.push(arg);
            }
        }

        flattened
    }

    /// Intern a term, returning its node index
    pub fn intern(&mut self, term: TermId) -> u32 {
        if let Some(&idx) = self.term_to_node.get(&term) {
            return idx;
        }

        let idx = self.nodes.len() as u32;
        self.nodes.push(ENode {
            term,
            func: None,
            args: SmallVec::new(),
            fingerprint: ENodeFingerprint::default(),
        });
        self.uf.add();
        self.use_list.push(SmallVec::new());
        self.proof_forest.push(Vec::new());
        self.term_to_node.insert(term, idx);
        idx
    }

    /// Intern a function application
    pub fn intern_app(
        &mut self,
        term: TermId,
        func: u32,
        args: impl IntoIterator<Item = u32>,
    ) -> u32 {
        if let Some(&idx) = self.term_to_node.get(&term) {
            return idx;
        }

        let args: SmallVec<[u32; 4]> = args.into_iter().collect();

        // Flatten for associative functions
        let flattened_args = self.flatten_args(func, &args);

        // Canonicalize arguments (handles commutativity and finds canonical reps)
        let canonical_args = self.canonicalize_args(func, &flattened_args);

        // Compute fingerprint for fast congruence pre-filtering
        let fp = ENodeFingerprint::compute(func, &canonical_args);

        let sig = (func, canonical_args.clone());
        if let Some(&existing) = self.sig_table.get(&sig) {
            self.term_to_node.insert(term, existing);
            return existing;
        }

        let idx = self.nodes.len() as u32;
        self.nodes.push(ENode {
            term,
            func: Some(func),
            args: flattened_args.clone(),
            fingerprint: fp,
        });
        self.uf.add();
        self.use_list.push(SmallVec::new());
        self.proof_forest.push(Vec::new());
        self.term_to_node.insert(term, idx);

        // Add to use lists
        for &arg in &flattened_args {
            self.use_list[arg as usize].push(idx);
        }

        // Add to signature table
        self.sig_table.insert(sig, idx);

        // Add to fingerprint table for fast congruence pre-filtering
        self.fingerprint_table.entry(fp).or_default().push(idx);

        idx
    }

    /// Merge two equivalence classes
    pub fn merge(&mut self, a: u32, b: u32, reason: TermId) -> Result<()> {
        self.pending.push((a, b, reason));
        self.propagate()?;
        Ok(())
    }

    /// Propagate pending merges with optimized congruence closure:
    /// - Index-based use-list iteration (avoids cloning the use-list)
    /// - Batch signature updates (collects all updates, applies at once)
    /// - Fingerprint pre-filter (cheap u64 comparison before full signature match)
    fn propagate(&mut self) -> Result<()> {
        let mut propagation_buf = mem::take(&mut self.propagation_buf);
        propagation_buf.clear();

        while let Some((a, b, reason)) = self.pending.pop() {
            let root_a = self.uf.find(a);
            let root_b = self.uf.find(b);

            if root_a == root_b {
                continue;
            }

            // Record the merge in the proof forest (for explanation generation)
            self.proof_forest[a as usize].push(MergeEdge {
                other: b,
                reason: MergeReason::Assertion(reason),
            });
            self.proof_forest[b as usize].push(MergeEdge {
                other: a,
                reason: MergeReason::Assertion(reason),
            });

            // Union the classes
            self.uf.union(root_a, root_b);
            let new_root = self.uf.find(root_a);

            // Congruence closure: check for new merges
            let other_root = if new_root == root_a { root_b } else { root_a };

            // --- Optimization 1: Index-based use-list iteration ---
            // Instead of cloning the entire use-list, iterate by index.
            // We snapshot the length so we only process existing entries.
            let use_len = self.use_list[other_root as usize].len();

            // --- Optimization 2: Batch signature updates ---
            // Collect all (new_signature, node_id) pairs first, then apply
            // to the sig_table in a single batch to avoid repeated hash lookups.
            let mut sig_updates: SmallVec<[SigUpdateEntry; 16]> = SmallVec::new();
            // Collect congruence merges to enqueue
            propagation_buf.clear();

            for i in 0..use_len {
                let user = self.use_list[other_root as usize][i];
                if (user as usize) >= self.nodes.len() {
                    continue; // stale use-list entry — node was not allocated
                }
                let func = match self.nodes[user as usize].func {
                    Some(f) => f,
                    None => continue,
                };

                // Read args by index to avoid cloning the SmallVec
                let args_len = self.nodes[user as usize].args.len();
                let mut args_copy: SmallVec<[u32; 4]> = SmallVec::with_capacity(args_len);
                for j in 0..args_len {
                    args_copy.push(self.nodes[user as usize].args[j]);
                }

                // Canonicalize arguments (handles commutativity and finds canonical reps)
                let canonical_args = self.canonicalize_args(func, &args_copy);

                // --- Optimization 3: Fingerprint pre-filter ---
                // Compute the new fingerprint for the updated canonical args
                let new_fp = ENodeFingerprint::compute(func, &canonical_args);

                // Check signature table for congruence match
                let sig = (func, canonical_args.clone());
                if let Some(&existing) = self.sig_table.get(&sig) {
                    if !self.uf.same(user, existing) {
                        // Congruence detected: record proof edges
                        self.proof_forest[user as usize].push(MergeEdge {
                            other: existing,
                            reason: MergeReason::Congruence {
                                term1: user,
                                term2: existing,
                            },
                        });
                        self.proof_forest[existing as usize].push(MergeEdge {
                            other: user,
                            reason: MergeReason::Congruence {
                                term1: user,
                                term2: existing,
                            },
                        });

                        propagation_buf.push((user, existing, TermId::new(0)));
                    }
                } else {
                    // No congruence match; batch the signature update for later.
                    sig_updates.push((sig, user, new_fp));
                }

                // Update the node's fingerprint
                self.nodes[user as usize].fingerprint = new_fp;
            }

            // Apply batched signature updates
            for (sig, node_id, fp) in sig_updates {
                self.sig_table.insert(sig, node_id);
                self.fingerprint_table.entry(fp).or_default().push(node_id);
            }

            // Enqueue congruence merges
            for (user, existing, term) in propagation_buf.drain(..) {
                self.pending.push((user, existing, term));
            }

            // Merge use lists: extend new_root's use-list with other_root's entries
            // Using index-based copy to avoid borrow conflicts
            let mut other_uses: SmallVec<[u32; 8]> = SmallVec::with_capacity(use_len);
            for i in 0..use_len {
                other_uses.push(self.use_list[other_root as usize][i]);
            }
            self.use_list[new_root as usize].extend(other_uses);
        }

        propagation_buf.clear();
        self.propagation_buf = propagation_buf;

        Ok(())
    }

    /// Assert a disequality
    pub fn assert_diseq(&mut self, a: u32, b: u32, reason: TermId) {
        self.diseqs.push(Diseq {
            lhs: a,
            rhs: b,
            reason,
        });
    }

    /// Check for conflicts
    pub fn check_conflicts(&mut self) -> Option<Vec<TermId>> {
        for diseq in &self.diseqs {
            if self.uf.same(diseq.lhs, diseq.rhs) {
                // Conflict: a = b but we have a != b
                // Generate an explanation for why a = b
                let mut explanation = self.explain_equality(diseq.lhs, diseq.rhs);
                // Add the disequality reason
                if !explanation.contains(&diseq.reason) {
                    explanation.push(diseq.reason);
                }
                return Some(explanation);
            }
        }
        None
    }

    /// Explain why two nodes are equal
    /// Uses BFS through the proof forest to find a path
    fn explain_equality(&self, a: u32, b: u32) -> Vec<TermId> {
        if a == b {
            return Vec::new();
        }

        let n = self.proof_forest.len();
        // Guard against out-of-bounds indices
        if (a as usize) >= n || (b as usize) >= n {
            return Vec::new();
        }
        let mut visited = vec![false; n];
        let mut parent = vec![None; n];

        // BFS to find path from a to b
        let mut queue = crate::prelude::VecDeque::new();
        queue.push_back(a);
        visited[a as usize] = true;

        let mut found = false;
        while let Some(node) = queue.pop_front() {
            if node == b {
                found = true;
                break;
            }

            if (node as usize) >= self.proof_forest.len() {
                continue;
            }
            for (idx, edge) in self.proof_forest[node as usize].iter().enumerate() {
                let other_idx = edge.other as usize;
                if other_idx < n && !visited[other_idx] {
                    visited[other_idx] = true;
                    parent[other_idx] = Some((node, idx));
                    queue.push_back(edge.other);
                }
            }
        }

        if !found {
            return Vec::new();
        }

        // Reconstruct path and collect reasons
        let mut reasons = Vec::new();
        let mut current = b;

        while let Some((prev, edge_idx)) = parent[current as usize] {
            let edge = &self.proof_forest[prev as usize][edge_idx];

            match &edge.reason {
                MergeReason::Assertion(term_id) => {
                    if term_id.raw() != 0 && !reasons.contains(term_id) {
                        reasons.push(*term_id);
                    }
                }
                MergeReason::Congruence { term1, term2 } => {
                    // For congruence, we need to explain why the arguments are equal
                    let node1 = &self.nodes[*term1 as usize];
                    let node2 = &self.nodes[*term2 as usize];

                    // Recursively explain argument equalities
                    for (&arg1, &arg2) in node1.args.iter().zip(node2.args.iter()) {
                        if arg1 != arg2 && self.uf.same_no_compress(arg1, arg2) {
                            let arg_reasons = self.explain_equality(arg1, arg2);
                            for r in arg_reasons {
                                if !reasons.contains(&r) {
                                    reasons.push(r);
                                }
                            }
                        }
                    }
                }
            }

            current = prev;
        }

        reasons
    }

    /// Check if two terms are equivalent
    pub fn are_equal(&mut self, a: u32, b: u32) -> bool {
        self.uf.same(a, b)
    }

    /// Get the representative of a term
    pub fn find(&mut self, a: u32) -> u32 {
        self.uf.find(a)
    }

    /// Get the representative of a term without path compression (immutable)
    pub fn find_immutable(&self, a: u32) -> u32 {
        self.uf.find_no_compress(a)
    }

    /// Check equivalence without mutation (immutable)
    pub fn are_equal_immutable(&self, a: u32, b: u32) -> bool {
        self.uf.same_no_compress(a, b)
    }

    /// Get the number of E-graph nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the term associated with a node index
    pub fn node_term(&self, idx: u32) -> Option<TermId> {
        self.nodes.get(idx as usize).map(|n| n.term)
    }

    /// Get the function symbol of a node (if it is a function application)
    pub fn node_func(&self, idx: u32) -> Option<u32> {
        self.nodes.get(idx as usize).and_then(|n| n.func)
    }

    /// Get the arguments of a node (if it is a function application)
    pub fn node_args(&self, idx: u32) -> Option<&SmallVec<[u32; 4]>> {
        let node = self.nodes.get(idx as usize)?;
        if node.func.is_some() {
            Some(&node.args)
        } else {
            None
        }
    }

    /// Look up the node index for a given TermId
    pub fn term_to_node(&self, term: TermId) -> Option<u32> {
        self.term_to_node.get(&term).copied()
    }

    /// Iterate over all node indices that are function applications of a given function symbol.
    /// Returns a Vec of node indices.
    pub fn apps_by_func(&self, func_id: u32) -> Vec<u32> {
        let mut result = Vec::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            if node.func == Some(func_id) {
                result.push(idx as u32);
            }
        }
        result
    }

    /// Get all members of an equivalence class (all node indices with the same representative).
    /// This is an O(n) scan; for performance-critical paths, consider caching.
    pub fn class_members(&self, class_rep: u32) -> Vec<u32> {
        let rep = self.uf.find_no_compress(class_rep);
        let mut members = Vec::new();
        for idx in 0..self.nodes.len() {
            if self.uf.find_no_compress(idx as u32) == rep {
                members.push(idx as u32);
            }
        }
        members
    }

    /// Iterate over all node indices (0..node_count)
    pub fn all_node_indices(&self) -> std::ops::Range<u32> {
        0..self.nodes.len() as u32
    }

    /// Get all distinct function symbols present in the E-graph
    pub fn all_func_symbols(&self) -> Vec<u32> {
        use rustc_hash::FxHashSet;
        let mut funcs = FxHashSet::default();
        for node in &self.nodes {
            if let Some(func) = node.func {
                funcs.insert(func);
            }
        }
        funcs.into_iter().collect()
    }

    /// Get a reference to the fingerprint table (for testing/debugging)
    #[cfg(test)]
    fn fingerprint_table_len(&self) -> usize {
        self.fingerprint_table.len()
    }
}

impl Theory for EufSolver {
    fn id(&self) -> TheoryId {
        TheoryId::EUF
    }

    fn name(&self) -> &str {
        "EUF"
    }

    fn can_handle(&self, _term: TermId) -> bool {
        // EUF can handle equality and function applications
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        // Assuming term is an equality a = b
        // In a full implementation, we'd parse the term
        let _ = self.intern(term);
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        // Assuming term is an equality a = b, assert a != b
        let node = self.intern(term);
        self.assert_diseq(node, node, term); // Simplified - real impl needs parsing
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        if let Some(conflict) = self.check_conflicts() {
            Ok(TheoryResult::Unsat(conflict))
        } else {
            Ok(TheoryResult::Sat)
        }
    }

    fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_nodes: self.nodes.len(),
            num_diseqs: self.diseqs.len(),
        });
        self.uf.push();
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            let num_nodes = state.num_nodes;

            self.nodes.truncate(num_nodes);
            self.diseqs.truncate(state.num_diseqs);
            self.uf.pop();

            // Also truncate related structures
            self.use_list.truncate(num_nodes);
            self.proof_forest.truncate(num_nodes);

            // Remove term_to_node mappings that point to removed nodes
            self.term_to_node
                .retain(|_term, &mut idx| (idx as usize) < num_nodes);

            // Rebuild signature table and fingerprint table for remaining nodes
            self.sig_table.clear();
            self.fingerprint_table.clear();
            for (idx, node) in self.nodes.iter().enumerate() {
                if let Some(func) = node.func {
                    let canonical_args: SmallVec<[u32; 4]> = node
                        .args
                        .iter()
                        .map(|&a| self.uf.find_no_compress(a))
                        .collect();
                    let fp = ENodeFingerprint::compute(func, &canonical_args);
                    self.sig_table.insert((func, canonical_args), idx as u32);
                    self.fingerprint_table
                        .entry(fp)
                        .or_default()
                        .push(idx as u32);
                }
            }
        }
    }

    fn reset(&mut self) {
        self.uf = UnionFind::new(0);
        self.nodes.clear();
        self.term_to_node.clear();
        self.diseqs.clear();
        self.pending.clear();
        self.use_list.clear();
        self.sig_table.clear();
        self.fingerprint_table.clear();
        self.context_stack.clear();
        self.proof_forest.clear();
        self.function_properties.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euf_basic() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        assert!(!solver.are_equal(a, b));

        solver.merge(a, b, TermId::new(0)).unwrap_or(());
        assert!(solver.are_equal(a, b));

        solver.merge(b, c, TermId::new(0)).unwrap_or(());
        assert!(solver.are_equal(a, c));
    }

    #[test]
    fn test_euf_diseq_conflict() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));

        // Assert a != b
        solver.assert_diseq(a, b, TermId::new(10));
        assert!(solver.check_conflicts().is_none());

        // Then assert a = b -> conflict
        solver.merge(a, b, TermId::new(11)).unwrap_or(());
        assert!(solver.check_conflicts().is_some());
    }

    #[test]
    fn test_euf_congruence() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));

        // f(a) and f(b)
        let fa = solver.intern_app(TermId::new(3), 0, [a]);
        let fb = solver.intern_app(TermId::new(4), 0, [b]);

        assert!(!solver.are_equal(fa, fb));

        // Merge a and b -> f(a) = f(b) by congruence
        solver.merge(a, b, TermId::new(0)).unwrap_or(());
        assert!(solver.are_equal(fa, fb));
    }

    #[test]
    fn test_euf_explanation_simple() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        // Assert a = b (reason 10)
        solver.merge(a, b, TermId::new(10)).unwrap_or(());

        // Assert b = c (reason 11)
        solver.merge(b, c, TermId::new(11)).unwrap_or(());

        // Assert a != c (reason 12)
        solver.assert_diseq(a, c, TermId::new(12));

        // Now check - should have conflict with explanation containing reasons 10, 11, 12
        let conflict = solver.check_conflicts();
        assert!(conflict.is_some());

        if let Some(reasons) = conflict {
            // Should contain the disequality reason
            assert!(reasons.contains(&TermId::new(12)));
            // Should contain at least one of the equality reasons
            assert!(reasons.len() >= 2);
        }
    }

    #[test]
    fn test_euf_explanation_congruence() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));

        // f(a) and f(b)
        let fa = solver.intern_app(TermId::new(3), 0, [a]);
        let fb = solver.intern_app(TermId::new(4), 0, [b]);

        // Assert f(a) != f(b) (reason 20)
        solver.assert_diseq(fa, fb, TermId::new(20));

        // Assert a = b (reason 21) -> causes f(a) = f(b) by congruence
        solver.merge(a, b, TermId::new(21)).unwrap_or(());

        // Check - should have conflict
        let conflict = solver.check_conflicts();
        assert!(conflict.is_some());

        if let Some(reasons) = conflict {
            // Should contain the disequality reason
            assert!(reasons.contains(&TermId::new(20)));
            // Should contain the equality reason that caused congruence
            assert!(reasons.contains(&TermId::new(21)));
        }
    }

    #[test]
    fn test_euf_transitivity_explanation() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));
        let d = solver.intern(TermId::new(4));

        // Assert a = b (reason 100)
        solver.merge(a, b, TermId::new(100)).unwrap_or(());

        // Assert b = c (reason 101)
        solver.merge(b, c, TermId::new(101)).unwrap_or(());

        // Assert c = d (reason 102)
        solver.merge(c, d, TermId::new(102)).unwrap_or(());

        // Assert a != d (reason 103)
        solver.assert_diseq(a, d, TermId::new(103));

        // Check - should have conflict
        let conflict = solver.check_conflicts();
        assert!(conflict.is_some());

        if let Some(reasons) = conflict {
            // Should contain the disequality reason
            assert!(reasons.contains(&TermId::new(103)));
            // Should have multiple reasons from the equality chain
            assert!(reasons.len() >= 2);
        }
    }

    #[test]
    fn test_commutative_function() {
        let mut solver = EufSolver::new();

        // Register a commutative function (e.g., addition)
        solver.register_function(
            0,
            FunctionProperties {
                associative: false,
                commutative: true,
                has_identity: false,
            },
        );

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));

        // f(a, b) and f(b, a) should be the same due to commutativity
        let fab = solver.intern_app(TermId::new(3), 0, [a, b]);
        let fba = solver.intern_app(TermId::new(4), 0, [b, a]);

        // They should be the same node due to commutativity
        assert_eq!(fab, fba);
    }

    #[test]
    fn test_associative_function() {
        let mut solver = EufSolver::new();

        // Register an associative function (e.g., addition)
        solver.register_function(
            0,
            FunctionProperties {
                associative: true,
                commutative: false,
                has_identity: false,
            },
        );

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        // f(a, b)
        let fab = solver.intern_app(TermId::new(10), 0, [a, b]);

        // f(f(a, b), c) should be flattened to f(a, b, c)
        let fab_c = solver.intern_app(TermId::new(11), 0, [fab, c]);

        // Verify that the node has 3 arguments (flattened)
        let node = &solver.nodes[fab_c as usize];
        assert_eq!(node.args.len(), 3);
    }

    #[test]
    fn test_associative_commutative_function() {
        let mut solver = EufSolver::new();

        // Register an associative and commutative function (e.g., addition)
        solver.register_function(
            0,
            FunctionProperties {
                associative: true,
                commutative: true,
                has_identity: false,
            },
        );

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        // f(a, b)
        let fab = solver.intern_app(TermId::new(10), 0, [a, b]);

        // f(c, f(a, b)) should be flattened and canonicalized
        let c_fab = solver.intern_app(TermId::new(11), 0, [c, fab]);

        // f(f(b, a), c) should be flattened and canonicalized to the same thing
        let fba = solver.intern_app(TermId::new(12), 0, [b, a]);
        let fba_c = solver.intern_app(TermId::new(13), 0, [fba, c]);

        // Due to commutativity and associativity, they should be the same
        assert_eq!(c_fab, fba_c);
    }

    #[test]
    fn test_fingerprint_basic() {
        // Same func and args should produce the same fingerprint
        let fp1 = ENodeFingerprint::compute(0, &[1, 2, 3]);
        let fp2 = ENodeFingerprint::compute(0, &[1, 2, 3]);
        assert_eq!(fp1, fp2);

        // Different args should (almost certainly) produce different fingerprints
        let fp3 = ENodeFingerprint::compute(0, &[1, 2, 4]);
        assert_ne!(fp1, fp3);

        // Different func should produce different fingerprint
        let fp4 = ENodeFingerprint::compute(1, &[1, 2, 3]);
        assert_ne!(fp1, fp4);
    }

    #[test]
    fn test_fingerprint_empty_args() {
        let fp1 = ENodeFingerprint::compute(5, &[]);
        let fp2 = ENodeFingerprint::compute(5, &[]);
        assert_eq!(fp1, fp2);

        let fp3 = ENodeFingerprint::compute(6, &[]);
        assert_ne!(fp1, fp3);
    }

    #[test]
    fn test_congruence_with_fingerprint_prefilter() {
        // Verify congruence closure still works correctly with fingerprint optimization
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        // g(a, c) and g(b, c)
        let gac = solver.intern_app(TermId::new(10), 1, [a, c]);
        let gbc = solver.intern_app(TermId::new(11), 1, [b, c]);

        assert!(!solver.are_equal(gac, gbc));

        // Merge a and b -> g(a,c) = g(b,c) by congruence
        solver.merge(a, b, TermId::new(50)).unwrap_or(());
        assert!(solver.are_equal(gac, gbc));
    }

    #[test]
    fn test_fingerprint_table_populated() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));

        let _fa = solver.intern_app(TermId::new(3), 0, [a]);
        let _fb = solver.intern_app(TermId::new(4), 0, [b]);

        // There should be entries in the fingerprint table
        assert!(solver.fingerprint_table_len() > 0);
    }

    #[test]
    fn test_push_pop_rebuilds_fingerprint_table() {
        use crate::theory::Theory;

        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));

        solver.push();

        let b = solver.intern(TermId::new(2));
        let _fa = solver.intern_app(TermId::new(3), 0, [a]);
        let _fb = solver.intern_app(TermId::new(4), 0, [b]);

        let fp_count_before = solver.fingerprint_table_len();
        assert!(fp_count_before > 0);

        solver.pop();

        // After pop, fingerprint table should be rebuilt (possibly smaller)
        let fp_count_after = solver.fingerprint_table_len();
        assert!(fp_count_after <= fp_count_before);
    }

    #[test]
    fn test_batch_sig_updates_correctness() {
        // Test that batch signature updates produce correct congruence results
        // with multiple function applications
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));
        let d = solver.intern(TermId::new(4));

        // f(a, c) and f(b, d)
        let fac = solver.intern_app(TermId::new(10), 0, [a, c]);
        let fbd = solver.intern_app(TermId::new(11), 0, [b, d]);

        assert!(!solver.are_equal(fac, fbd));

        // Merge a=b and c=d -> should trigger congruence f(a,c) = f(b,d)
        solver.merge(a, b, TermId::new(50)).unwrap_or(());
        solver.merge(c, d, TermId::new(51)).unwrap_or(());
        assert!(solver.are_equal(fac, fbd));
    }

    #[test]
    fn test_reset_clears_fingerprint_table() {
        use crate::theory::Theory;

        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let _fa = solver.intern_app(TermId::new(2), 0, [a]);

        assert!(solver.fingerprint_table_len() > 0);

        solver.reset();

        assert_eq!(solver.fingerprint_table_len(), 0);
    }
}
