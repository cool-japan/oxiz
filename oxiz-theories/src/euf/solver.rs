//! EUF Theory Solver

use super::union_find::UnionFind;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::theory::{Theory, TheoryId, TheoryResult};
use core::mem;
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use smallvec::SmallVec;

/// Signature update entry used in batched congruence-closure updates.
#[derive(Debug)]
struct SigUpdateEntry {
    func: u32,
    args: SmallVec<[u32; 4]>,
    node: u32,
    fp: ENodeFingerprint,
}

/// Records an insertion into sig_table or fingerprint_table for undo on pop().
#[derive(Debug, Clone)]
enum SigTrailEntry {
    /// Inserted key into sig_table; undo removes this key.
    InsertedSig { key: (u32, SmallVec<[u32; 4]>) },
    /// Pushed node_idx into fingerprint_table[fp]; undo removes it from the bucket.
    InsertedFingerprint { fp: ENodeFingerprint, node_idx: u32 },
}

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
    /// Function symbol index; `u32::MAX` (= `ENode::NO_FUNC`) means leaf (no application).
    /// Placed first so that the hot `func` discriminant is at offset 0 of the struct.
    func: u32,
    /// 64-bit fingerprint for fast congruence pre-filtering.
    /// Placed second (after the 4-byte func + 4-byte implicit pad) so it aligns to 8 bytes
    /// without additional padding waste.
    fingerprint: ENodeFingerprint,
    /// Arguments (indices into nodes)
    args: SmallVec<[u32; 4]>,
    /// The original term
    term: TermId,
}

impl ENode {
    /// Sentinel value meaning "no function symbol" (leaf node).
    const NO_FUNC: u32 = u32::MAX;

    /// Create a leaf node (no function application).
    fn leaf(term: TermId) -> Self {
        ENode {
            func: Self::NO_FUNC,
            fingerprint: ENodeFingerprint::default(),
            args: SmallVec::new(),
            term,
        }
    }

    /// Create a function application node.
    fn app(
        func: u32,
        args: SmallVec<[u32; 4]>,
        fingerprint: ENodeFingerprint,
        term: TermId,
    ) -> Self {
        debug_assert!(
            func != Self::NO_FUNC,
            "func must not be u32::MAX (reserved sentinel)"
        );
        ENode {
            func,
            fingerprint,
            args,
            term,
        }
    }

    /// Returns true if this node is a function application (not a leaf).
    #[inline]
    fn is_app(&self) -> bool {
        self.func != Self::NO_FUNC
    }
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
    /// Proof forest: for each node, edges to explain equalities.
    /// SmallVec<[MergeEdge; 4]> avoids heap allocation for nodes with ≤4 proof edges,
    /// which covers the vast majority of E-graph nodes in practice.
    proof_forest: Vec<SmallVec<[MergeEdge; 4]>>,
    /// Function properties for dynamic arity support
    function_properties: FxHashMap<u32, FunctionProperties>,
    /// Reused queue for newly discovered propagations during congruence closure.
    propagation_buf: Vec<(u32, u32, TermId)>,
    /// Undo trail for sig_table and fingerprint_table insertions.
    sig_trail: Vec<SigTrailEntry>,
    /// Scope checkpoints into sig_trail, parallel to uf.trail_limits.
    sig_trail_limits: Vec<usize>,
    /// Reusable BFS queue for explain_equality — avoids per-call VecDeque allocation.
    explain_queue: crate::prelude::VecDeque<u32>,
    /// Reusable visited flags for explain_equality — resized to proof_forest.len() and cleared at entry.
    explain_visited: Vec<bool>,
    /// Reusable parent-pointer table for explain_equality — parallel to explain_visited.
    explain_parent: Vec<Option<(u32, usize)>>,
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
            sig_trail: Vec::new(),
            sig_trail_limits: Vec::new(),
            explain_queue: crate::prelude::VecDeque::new(),
            explain_visited: Vec::new(),
            explain_parent: Vec::new(),
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
        self.canonicalize_args_with_props(&props, args)
    }

    /// Canonicalize arguments given pre-fetched function properties.
    /// Used in hot paths to hoist the `get_function_props` hashmap lookup out of inner loops.
    fn canonicalize_args_with_props(
        &mut self,
        props: &FunctionProperties,
        args: &[u32],
    ) -> SmallVec<[u32; 4]> {
        let mut canonical: SmallVec<[u32; 4]> = args.iter().map(|&a| self.uf.find(a)).collect();

        // For commutative functions, sort arguments by their canonical representative
        if props.commutative {
            canonical.sort_unstable();
        }

        canonical
    }

    /// Canonicalize arguments into a caller-owned buffer to avoid per-call allocation.
    /// Clears `buf` first, then pushes the canonical representative of each arg.
    /// For commutative functions the results are sorted in-place.
    ///
    /// This is the allocation-free variant used in the hot inner loop of `propagate`.
    fn canonicalize_args_with_props_into(
        &mut self,
        props: &FunctionProperties,
        args: &[u32],
        buf: &mut SmallVec<[u32; 4]>,
    ) {
        buf.clear();
        for &a in args {
            buf.push(self.uf.find(a));
        }
        if props.commutative {
            buf.sort_unstable();
        }
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
            if arg_node.is_app() && arg_node.func == func {
                flattened.extend(arg_node.args.iter().copied());
            } else {
                flattened.push(arg);
            }
        }

        flattened
    }

    /// Intern a term, returning its node index
    #[inline]
    pub fn intern(&mut self, term: TermId) -> u32 {
        if let Some(&idx) = self.term_to_node.get(&term) {
            return idx;
        }

        let idx = self.nodes.len() as u32;
        self.nodes.push(ENode::leaf(term));
        self.uf.add();
        self.use_list.push(SmallVec::new());
        self.proof_forest.push(SmallVec::new());
        self.term_to_node.insert(term, idx);
        idx
    }

    /// Intern a function application
    #[inline]
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
        self.nodes
            .push(ENode::app(func, flattened_args.clone(), fp, term));
        self.uf.add();
        self.use_list.push(SmallVec::new());
        self.proof_forest.push(SmallVec::new());
        self.term_to_node.insert(term, idx);

        // Add to use lists
        for &arg in &flattened_args {
            self.use_list[arg as usize].push(idx);
        }

        // Add to signature table. When inside a push scope, record the insertion
        // in the undo trail so pop() can remove it without rebuilding the table.
        // `canonical_args` is moved (no extra clone needed).
        if !self.sig_trail_limits.is_empty() {
            self.sig_trail.push(SigTrailEntry::InsertedSig {
                key: (func, canonical_args),
            });
        }
        self.sig_table.insert(sig, idx);

        // Add to fingerprint table for fast congruence pre-filtering
        self.fingerprint_table.entry(fp).or_default().push(idx);
        if !self.sig_trail_limits.is_empty() {
            self.sig_trail
                .push(SigTrailEntry::InsertedFingerprint { fp, node_idx: idx });
        }

        idx
    }

    /// Merge two equivalence classes
    #[inline]
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

            // --- Change A: Reusable canonicalization buffer ---
            // Declared once outside the loop so the SmallVec's heap backing (if it
            // ever spills past the inline capacity of 4) is allocated at most once
            // per merge event rather than once per use-list entry.
            let mut canon_buf: SmallVec<[u32; 4]> = SmallVec::new();

            for i in 0..use_len {
                let user = self.use_list[other_root as usize][i];
                if (user as usize) >= self.nodes.len() {
                    continue; // stale use-list entry — node was not allocated
                }
                let node_func_val = self.nodes[user as usize].func;
                if node_func_val == ENode::NO_FUNC {
                    continue;
                }
                let func = node_func_val;

                // Read args by index to avoid cloning the SmallVec
                let args_len = self.nodes[user as usize].args.len();
                let mut args_copy: SmallVec<[u32; 4]> = SmallVec::with_capacity(args_len);
                for j in 0..args_len {
                    args_copy.push(self.nodes[user as usize].args[j]);
                }

                // Fetch function properties once per use-list entry (per unique func),
                // then pass to canonicalize_args_with_props_into to avoid repeated lookups.
                let props = self.get_function_props(func);

                // Canonicalize arguments into the reusable buffer (avoids per-iteration alloc).
                self.canonicalize_args_with_props_into(&props, &args_copy, &mut canon_buf);

                // --- Optimization 3: Fingerprint pre-filter ---
                // Compute the new fingerprint for the updated canonical args
                let new_fp = ENodeFingerprint::compute(func, &canon_buf);

                // Fast-exit guard before costly sig_table.get:
                // `sig_table.get` hashes over (u32, SmallVec) which is expensive.
                // If no entry with this fingerprint exists in fingerprint_table, skip
                // the sig lookup — but still update sig_updates and the node fingerprint
                // so the invariant (fingerprint_table tracks all live fps) is maintained.
                if !self.fingerprint_table.contains_key(&new_fp) {
                    sig_updates.push(SigUpdateEntry {
                        func,
                        args: canon_buf.clone(),
                        node: user,
                        fp: new_fp,
                    });
                    self.nodes[user as usize].fingerprint = new_fp;
                    continue;
                }

                // Check signature table for congruence match
                let sig = (func, canon_buf.clone());
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
                    // We must clone canon_buf here because it is reused on the next iteration.
                    sig_updates.push(SigUpdateEntry {
                        func,
                        args: canon_buf.clone(),
                        node: user,
                        fp: new_fp,
                    });
                }

                // Update the node's fingerprint
                self.nodes[user as usize].fingerprint = new_fp;
            }

            // Apply batched signature updates. When inside a push scope, record
            // each insertion into the trail so pop() can undo them without rebuild.
            // The guard is hoisted above the clone to avoid unnecessary SmallVec
            // allocations on non-incremental workloads (no active push scope).
            let in_scope = !self.sig_trail_limits.is_empty();
            for entry in sig_updates {
                let SigUpdateEntry {
                    func,
                    args,
                    node,
                    fp,
                } = entry;
                if in_scope {
                    // Clone before consuming args in the insert call.
                    self.sig_trail.push(SigTrailEntry::InsertedSig {
                        key: (func, args.clone()),
                    });
                }
                self.sig_table.insert((func, args), node);
                self.fingerprint_table.entry(fp).or_default().push(node);
                if in_scope {
                    self.sig_trail
                        .push(SigTrailEntry::InsertedFingerprint { fp, node_idx: node });
                }
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
        // First find the conflicting disequality by index so that we can drop
        // the shared borrow on `self.diseqs` before calling `explain_equality`
        // (which needs `&mut self`).
        let conflict_idx = self
            .diseqs
            .iter()
            .position(|d| self.uf.same(d.lhs, d.rhs))?;

        let (lhs, rhs, reason) = {
            let d = &self.diseqs[conflict_idx];
            (d.lhs, d.rhs, d.reason)
        };

        // Borrow of self.diseqs is fully released here.
        let mut explanation = self.explain_equality(lhs, rhs);
        if !explanation.contains(&reason) {
            explanation.push(reason);
        }
        Some(explanation)
    }

    /// Explain why two nodes are equal.
    ///
    /// Uses BFS through the proof forest to find a path from `a` to `b`.
    /// Reusable buffers (`explain_queue`, `explain_visited`, `explain_parent`) are
    /// moved out of `self` via `mem::take` at entry and restored at exit so that
    /// recursive calls (for congruence sub-explanations) each work on a fresh,
    /// independently sized set of buffers without re-allocating from the heap once
    /// the buffers are warm.
    fn explain_equality(&mut self, a: u32, b: u32) -> Vec<TermId> {
        if a == b {
            return Vec::new();
        }

        let n = self.proof_forest.len();
        // Guard against out-of-bounds indices
        if (a as usize) >= n || (b as usize) >= n {
            return Vec::new();
        }

        // Take reusable buffers out of self so recursive calls (for congruence
        // sub-explanations) do not conflict with the current borrow.
        let mut queue = mem::take(&mut self.explain_queue);
        let mut visited = mem::take(&mut self.explain_visited);
        let mut parent = mem::take(&mut self.explain_parent);

        // Reset / resize in-place — existing heap capacity is retained.
        queue.clear();
        visited.clear();
        visited.resize(n, false);
        parent.clear();
        parent.resize(n, None);

        // BFS to find path from a to b
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
            // Restore buffers before returning so they are available for the next call.
            self.explain_queue = queue;
            self.explain_visited = visited;
            self.explain_parent = parent;
            return Vec::new();
        }

        // Collect the (prev, edge_idx) pairs from the parent chain into a local
        // Vec before dropping the parent borrow — this lets us recursively call
        // explain_equality below without conflicting with `parent`.
        let mut path: Vec<(u32, usize)> = Vec::new();
        let mut current = b;
        while let Some((prev, edge_idx)) = parent[current as usize] {
            path.push((prev, edge_idx));
            current = prev;
        }

        // Restore buffers now — recursive calls may reuse them safely.
        self.explain_queue = queue;
        self.explain_visited = visited;
        self.explain_parent = parent;

        // Reconstruct path and collect reasons
        let mut reasons = Vec::new();

        for (prev, edge_idx) in path {
            let reason = self.proof_forest[prev as usize][edge_idx].reason.clone();

            match reason {
                MergeReason::Assertion(term_id) => {
                    if term_id.raw() != 0 && !reasons.contains(&term_id) {
                        reasons.push(term_id);
                    }
                }
                MergeReason::Congruence { term1, term2 } => {
                    // For congruence, we need to explain why the arguments are equal
                    let args1: SmallVec<[u32; 4]> = self.nodes[term1 as usize].args.clone();
                    let args2: SmallVec<[u32; 4]> = self.nodes[term2 as usize].args.clone();

                    // Recursively explain argument equalities
                    for (&arg1, &arg2) in args1.iter().zip(args2.iter()) {
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
        }

        reasons
    }

    /// Check if two terms are equivalent
    #[inline]
    pub fn are_equal(&mut self, a: u32, b: u32) -> bool {
        self.uf.same(a, b)
    }

    /// Get the representative of a term
    #[inline]
    pub fn find(&mut self, a: u32) -> u32 {
        self.uf.find(a)
    }

    /// Get the representative of a term without path compression (immutable)
    #[inline]
    pub fn find_immutable(&self, a: u32) -> u32 {
        self.uf.find_no_compress(a)
    }

    /// Check equivalence without mutation (immutable)
    #[inline]
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
        self.nodes
            .get(idx as usize)
            .and_then(|n| if n.is_app() { Some(n.func) } else { None })
    }

    /// Get the arguments of a node (if it is a function application)
    pub fn node_args(&self, idx: u32) -> Option<&SmallVec<[u32; 4]>> {
        let node = self.nodes.get(idx as usize)?;
        if node.is_app() {
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
            if node.is_app() && node.func == func_id {
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
            if node.is_app() {
                funcs.insert(node.func);
            }
        }
        funcs.into_iter().collect()
    }

    /// Get the fingerprint table size (for testing/debugging)
    #[cfg(test)]
    fn fingerprint_table_len(&self) -> usize {
        self.fingerprint_table.len()
    }

    /// Get the sig table size (for testing/debugging)
    #[cfg(test)]
    fn sig_table_len(&self) -> usize {
        self.sig_table.len()
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
        // Record sig_trail checkpoint, mirroring uf.trail_limits.push(...)
        self.sig_trail_limits.push(self.sig_trail.len());
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

            // Rewind sig_trail to the saved limit, undoing all sig/fp insertions
            // made since the matching push().  Mirrors UnionFind::pop() exactly.
            if let Some(sig_limit) = self.sig_trail_limits.pop() {
                while self.sig_trail.len() > sig_limit {
                    if let Some(entry) = self.sig_trail.pop() {
                        match entry {
                            SigTrailEntry::InsertedSig { key } => {
                                self.sig_table.remove(&key);
                            }
                            SigTrailEntry::InsertedFingerprint { fp, node_idx } => {
                                if let Some(bucket) = self.fingerprint_table.get_mut(&fp) {
                                    // Remove in LIFO order: the last push is the first to undo.
                                    if let Some(pos) = bucket.iter().rposition(|&n| n == node_idx) {
                                        bucket.swap_remove(pos);
                                    }
                                    if bucket.is_empty() {
                                        self.fingerprint_table.remove(&fp);
                                    }
                                }
                            }
                        }
                    }
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
        self.sig_trail.clear();
        self.sig_trail_limits.clear();
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

    /// Test that the fingerprint pre-filter does not cause false negatives:
    /// - Merging unrelated args must NOT produce spurious congruence merges.
    /// - Merging the right args MUST still produce congruence merges.
    #[test]
    fn test_fingerprint_prefilter_short_circuits() {
        let mut solver = EufSolver::new();
        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));
        let f_sym = 100u32;
        let fa = solver.intern_app(TermId::new(10), f_sym, [a]);
        let fb = solver.intern_app(TermId::new(11), f_sym, [b]);

        // Merge a = c (NOT a = b)
        solver.merge(a, c, TermId::new(20)).unwrap_or(());
        // f(a) and f(b) should NOT be merged (root(a) != root(b))
        assert!(
            !solver.are_equal(fa, fb),
            "f(a) and f(b) should not be merged without a=b"
        );

        // Now merge a = b (so root(a) == root(b))
        solver.merge(a, b, TermId::new(21)).unwrap_or(());
        // After a=b, congruence should derive f(a)=f(b)
        assert!(
            solver.are_equal(fa, fb),
            "f(a) and f(b) should be merged after a=b"
        );
    }

    /// Test the critical invariant: multi-step merges that route through an
    /// intermediate shared root must still produce congruence.
    /// This catches the bug where Change A's `continue` skips the fingerprint-table
    /// update, leaving the invariant broken for subsequent merges.
    #[test]
    fn test_fingerprint_prefilter_invariant_multi_merge() {
        let mut solver = EufSolver::new();
        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));
        let f_sym = 200u32;
        let fa = solver.intern_app(TermId::new(10), f_sym, [a]);
        let fb = solver.intern_app(TermId::new(11), f_sym, [b]);

        // merge(a, c): fa re-canonicalizes to f([c]); new fp may not be in table yet.
        // The pre-filter must still update fingerprint_table so the next step works.
        solver.merge(a, c, TermId::new(20)).unwrap_or(());
        assert!(
            !solver.are_equal(fa, fb),
            "f(a) and f(b) should not be merged yet"
        );

        // merge(b, c): fb re-canonicalizes to f([c]); fp IS now in table; congruence fires.
        solver.merge(b, c, TermId::new(21)).unwrap_or(());
        assert!(
            solver.are_equal(fa, fb),
            "f(a) and f(b) should be merged after a=c and b=c (both share root c)"
        );
    }

    /// Verify that using the reusable canon_buf (Change A) does not corrupt results
    /// when two different intern_app calls with different arities share the same solver.
    /// The buffer is cleared and refilled each iteration, so results must remain correct
    /// even across applications with different argument lists.
    #[test]
    fn test_canonicalize_buf_is_reused() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        let f_sym = 300u32;

        // Two applications with different argument sets
        let fab = solver.intern_app(TermId::new(10), f_sym, [a, b]);
        let fbc = solver.intern_app(TermId::new(11), f_sym, [b, c]);

        // Neither should be equal to each other initially
        assert!(!solver.are_equal(fab, fbc));

        // Merging a = b triggers propagate, which exercises the reused canon_buf
        // on use-list entries for both f(a,b) and f(b,c).
        solver.merge(a, b, TermId::new(50)).unwrap_or(());

        // f(a,b) has canonical args [root(a), root(b)] = [r, r]; if root(b) = root(c) differs
        // they must still be distinct.
        assert!(!solver.are_equal(fab, fbc));

        // Now merge b = c so the solver exercises propagate again with the same buf
        solver.merge(b, c, TermId::new(51)).unwrap_or(());

        // After a=b and b=c, a=b=c.  f(a,b) canonical = [root, root], f(b,c) = [root, root]
        // so congruence must unify them.
        assert!(
            solver.are_equal(fab, fbc),
            "f(a,b) and f(b,c) must be equal once a=b=c"
        );
    }

    /// Verify that the incremental sig_trail correctly restores sig_table and
    /// fingerprint_table to exactly the pre-push state, matching what a full
    /// rebuild would have produced.
    #[test]
    fn test_incremental_sig_trail_matches_rebuild() {
        use crate::theory::Theory;

        let mut solver = EufSolver::new();
        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let f_sym = 100u32;
        let fa = solver.intern_app(TermId::new(10), f_sym, [a]);
        // Capture state BEFORE push
        let sig_before = solver.sig_table_len();
        let fp_before = solver.fingerprint_table_len();

        solver.push();
        let c = solver.intern(TermId::new(3));
        let fc = solver.intern_app(TermId::new(11), f_sym, [c]);
        solver.merge(a, c, TermId::new(20)).expect("merge a=c");
        // Now pop — should restore to pre-push state
        solver.pop();

        let sig_after = solver.sig_table_len();
        let fp_after = solver.fingerprint_table_len();
        assert_eq!(
            sig_before, sig_after,
            "sig_table size should match pre-push state after pop"
        );
        assert_eq!(
            fp_before, fp_after,
            "fingerprint_table size should match pre-push state after pop"
        );
        // The merge done during the push scope must be undone
        assert!(
            !solver.are_equal(fa, fc),
            "terms merged during push scope should not be equal after pop"
        );
        let _ = (b, fc);
    }

    /// Verify that a 3-level push/pop stack completely rewinds all sig/fp state.
    #[test]
    fn test_push_pop_stack_depth_3() {
        use crate::theory::Theory;

        let mut solver = EufSolver::new();
        let f = 100u32;
        let a = solver.intern(TermId::new(1));

        // Level 1
        solver.push();
        let b = solver.intern(TermId::new(2));
        let fab = solver.intern_app(TermId::new(10), f, [a, b]);

        // Level 2
        solver.push();
        let c = solver.intern(TermId::new(3));
        let fbc = solver.intern_app(TermId::new(11), f, [b, c]);
        solver.merge(a, b, TermId::new(20)).expect("merge a=b");

        // Level 3
        solver.push();
        let d = solver.intern(TermId::new(4));
        solver.merge(b, c, TermId::new(21)).expect("merge b=c");

        // Pop all three levels
        solver.pop(); // back to level 2 state
        solver.pop(); // back to level 1 state
        solver.pop(); // back to initial state

        // After all pops, no merges should remain
        assert!(
            !solver.are_equal(a, b),
            "a and b should not be equal after full pop"
        );
        let _ = (fab, fbc, c, d);
    }

    #[test]
    fn test_enode_size_regression() {
        // Guards against ENode growing larger than expected.
        // ENode fields: func (4B), fingerprint (8B), args (SmallVec=32B), term (4B)
        // With alignment padding the size should be ≤ 56 bytes.
        let size = std::mem::size_of::<ENode>();
        assert!(size <= 56, "ENode size should be ≤56 bytes, got {}", size);
    }

    #[test]
    fn test_leaf_constructor_uses_sentinel() {
        let t = TermId::from(42u32);
        let node = ENode::leaf(t);
        assert!(!node.is_app(), "leaf node should not be an app");
        assert_eq!(
            node.func,
            ENode::NO_FUNC,
            "leaf node func should be NO_FUNC sentinel"
        );
        assert!(node.args.is_empty(), "leaf node should have no args");
    }
}
