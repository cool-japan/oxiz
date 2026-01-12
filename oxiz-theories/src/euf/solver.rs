//! EUF Theory Solver

use super::union_find::UnionFind;
use crate::theory::{Theory, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

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
    /// Context stack for push/pop
    context_stack: Vec<ContextState>,
    /// Proof forest: for each node, edges to explain equalities
    proof_forest: Vec<Vec<MergeEdge>>,
    /// Function properties for dynamic arity support
    function_properties: FxHashMap<u32, FunctionProperties>,
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
            context_stack: Vec::new(),
            proof_forest: Vec::new(),
            function_properties: FxHashMap::default(),
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

        idx
    }

    /// Merge two equivalence classes
    pub fn merge(&mut self, a: u32, b: u32, reason: TermId) -> Result<()> {
        self.pending.push((a, b, reason));
        self.propagate()?;
        Ok(())
    }

    /// Propagate pending merges
    fn propagate(&mut self) -> Result<()> {
        while let Some((a, b, reason)) = self.pending.pop() {
            let root_a = self.uf.find(a);
            let root_b = self.uf.find(b);

            if root_a == root_b {
                continue;
            }

            // Record the merge in the proof forest (for explanation generation)
            // Add edge from a to b with the reason
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

            // For each term that uses the merged class
            let uses: SmallVec<[u32; 8]> = self.use_list[other_root as usize].clone();
            for &user in &uses {
                let node = &self.nodes[user as usize];
                if let Some(func) = node.func {
                    // Clone args to avoid borrow checker issues
                    let args = node.args.clone();
                    // Use canonicalize_args to handle commutativity
                    let canonical_args = self.canonicalize_args(func, &args);

                    let sig = (func, canonical_args);
                    if let Some(&existing) = self.sig_table.get(&sig)
                        && !self.uf.same(user, existing)
                    {
                        // Congruence: merge user with existing
                        // Record congruence edge in proof forest
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

                        self.pending.push((user, existing, TermId::new(0)));
                    }
                }
            }

            // Merge use lists
            self.use_list[new_root as usize].extend(uses);
        }

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
        let mut visited = vec![false; n];
        let mut parent = vec![None; n];

        // BFS to find path from a to b
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(a);
        visited[a as usize] = true;

        let mut found = false;
        while let Some(node) = queue.pop_front() {
            if node == b {
                found = true;
                break;
            }

            for (idx, edge) in self.proof_forest[node as usize].iter().enumerate() {
                if !visited[edge.other as usize] {
                    visited[edge.other as usize] = true;
                    parent[edge.other as usize] = Some((node, idx));
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
            self.nodes.truncate(state.num_nodes);
            self.diseqs.truncate(state.num_diseqs);
            self.uf.pop();

            // Also truncate related structures
            self.use_list.truncate(state.num_nodes);
            self.proof_forest.truncate(state.num_nodes);

            // Rebuild signature table for remaining nodes
            self.sig_table.clear();
            for (idx, node) in self.nodes.iter().enumerate() {
                if let Some(func) = node.func {
                    let canonical_args: SmallVec<[u32; 4]> = node
                        .args
                        .iter()
                        .map(|&a| self.uf.find_no_compress(a))
                        .collect();
                    self.sig_table.insert((func, canonical_args), idx as u32);
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

        solver.merge(a, b, TermId::new(0)).unwrap();
        assert!(solver.are_equal(a, b));

        solver.merge(b, c, TermId::new(0)).unwrap();
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
        solver.merge(a, b, TermId::new(11)).unwrap();
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
        solver.merge(a, b, TermId::new(0)).unwrap();
        assert!(solver.are_equal(fa, fb));
    }

    #[test]
    fn test_euf_explanation_simple() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        // Assert a = b (reason 10)
        solver.merge(a, b, TermId::new(10)).unwrap();

        // Assert b = c (reason 11)
        solver.merge(b, c, TermId::new(11)).unwrap();

        // Assert a != c (reason 12)
        solver.assert_diseq(a, c, TermId::new(12));

        // Now check - should have conflict with explanation containing reasons 10, 11, 12
        let conflict = solver.check_conflicts();
        assert!(conflict.is_some());

        let reasons = conflict.unwrap();
        // Should contain the disequality reason
        assert!(reasons.contains(&TermId::new(12)));
        // Should contain at least one of the equality reasons
        assert!(reasons.len() >= 2);
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
        solver.merge(a, b, TermId::new(21)).unwrap();

        // Check - should have conflict
        let conflict = solver.check_conflicts();
        assert!(conflict.is_some());

        let reasons = conflict.unwrap();
        // Should contain the disequality reason
        assert!(reasons.contains(&TermId::new(20)));
        // Should contain the equality reason that caused congruence
        assert!(reasons.contains(&TermId::new(21)));
    }

    #[test]
    fn test_euf_transitivity_explanation() {
        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));
        let d = solver.intern(TermId::new(4));

        // Assert a = b (reason 100)
        solver.merge(a, b, TermId::new(100)).unwrap();

        // Assert b = c (reason 101)
        solver.merge(b, c, TermId::new(101)).unwrap();

        // Assert c = d (reason 102)
        solver.merge(c, d, TermId::new(102)).unwrap();

        // Assert a != d (reason 103)
        solver.assert_diseq(a, d, TermId::new(103));

        // Check - should have conflict
        let conflict = solver.check_conflicts();
        assert!(conflict.is_some());

        let reasons = conflict.unwrap();
        // Should contain the disequality reason
        assert!(reasons.contains(&TermId::new(103)));
        // Should have multiple reasons from the equality chain
        assert!(reasons.len() >= 2);
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
}
