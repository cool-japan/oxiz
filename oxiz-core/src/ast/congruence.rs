//! Congruence closure for efficient equality reasoning
//!
//! This module implements an advanced congruence closure data structure for maintaining
//! and reasoning about equalities between terms. It's a fundamental component
//! for equality reasoning in SMT solvers.
//!
//! Features:
//! - Backtrackable union-find for incremental solving (push/pop)
//! - Explanation tracking for proof generation
//! - Efficient worklist-based propagation
//! - Disequality reasoning and conflict detection

use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};

/// Explanation for why two terms are equal
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Explanation {
    /// Given equality (asserted)
    Given,
    /// Congruence: f(a1,...,an) = f(b1,...,bn) because ai = bi for all i
    Congruence(Vec<(TermId, TermId)>),
    /// Transitivity: a = c via b (a = b and b = c)
    Transitivity(TermId),
}

/// Undo operation for backtracking
#[derive(Debug, Clone)]
enum UndoOp {
    /// Undo a merge operation: restore parent[child] = old_parent
    Merge { child: TermId, old_parent: TermId },
    /// Undo a lookup insertion
    LookupInsert { key: (TermId, Vec<TermId>) },
    /// Undo a use list insertion
    UseListInsert { arg: TermId, parent: TermId },
}

/// Congruence closure data structure with advanced features
///
/// Maintains equivalence classes of terms under the congruence relation.
/// Two terms are congruent if they have the same function symbol and their
/// arguments are pairwise equivalent.
#[derive(Debug, Clone)]
pub struct CongruenceClosure {
    /// Union-find parent pointers
    parent: FxHashMap<TermId, TermId>,
    /// Rank for union-by-rank heuristic
    rank: FxHashMap<TermId, usize>,
    /// Explanations for why terms are equal
    explanations: FxHashMap<(TermId, TermId), Explanation>,
    /// Lookup table for congruence: maps (function, args) to term
    lookup: FxHashMap<(TermId, Vec<TermId>), TermId>,
    /// Use list: for each term, which terms use it as an argument
    use_list: FxHashMap<TermId, Vec<TermId>>,
    /// Worklist for pending propagations
    worklist: Vec<TermId>,
    /// Disequalities: set of pairs (a, b) where a ≠ b
    diseqs: FxHashSet<(TermId, TermId)>,
    /// Undo trail for backtracking
    undo_trail: Vec<UndoOp>,
    /// Scope levels for push/pop
    scope_levels: Vec<usize>,
}

impl CongruenceClosure {
    /// Create a new empty congruence closure
    #[must_use]
    pub fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
            explanations: FxHashMap::default(),
            lookup: FxHashMap::default(),
            use_list: FxHashMap::default(),
            worklist: Vec::new(),
            diseqs: FxHashSet::default(),
            undo_trail: Vec::new(),
            scope_levels: vec![0],
        }
    }

    /// Find the representative of a term's equivalence class (without path compression for backtracking)
    pub fn find(&mut self, term: TermId) -> TermId {
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(term) {
            e.insert(term);
            self.rank.insert(term, 0);
            return term;
        }

        // Iterative find to avoid path compression (which would break undo trail)
        let mut current = term;
        while let Some(&parent) = self.parent.get(&current) {
            if parent == current {
                return current;
            }
            current = parent;
        }
        current
    }

    /// Find with path halving (lighter compression that's easier to undo)
    #[allow(dead_code)]
    fn find_with_halving(&mut self, term: TermId) -> TermId {
        let mut current = term;
        loop {
            let parent = match self.parent.get(&current) {
                Some(&p) if p != current => p,
                _ => return current,
            };

            // Path halving: make current point to grandparent
            if let Some(&grandparent) = self.parent.get(&parent)
                && grandparent != parent
            {
                self.parent.insert(current, grandparent);
            }

            current = parent;
        }
    }

    /// Check if two terms are in the same equivalence class
    pub fn are_equal(&mut self, a: TermId, b: TermId) -> bool {
        self.find(a) == self.find(b)
    }

    /// Push a new scope for backtracking
    pub fn push(&mut self) {
        self.scope_levels.push(self.undo_trail.len());
    }

    /// Pop the most recent scope, undoing all operations since the last push
    pub fn pop(&mut self) {
        if self.scope_levels.len() <= 1 {
            return; // Cannot pop base level
        }

        let target_level = self
            .scope_levels
            .pop()
            .expect("scope_levels has elements after length check");

        // Undo all operations back to the target level
        while self.undo_trail.len() > target_level {
            if let Some(op) = self.undo_trail.pop() {
                match op {
                    UndoOp::Merge { child, old_parent } => {
                        self.parent.insert(child, old_parent);
                    }
                    UndoOp::LookupInsert { key } => {
                        self.lookup.remove(&key);
                    }
                    UndoOp::UseListInsert { arg, parent } => {
                        if let Some(list) = self.use_list.get_mut(&arg) {
                            list.retain(|&p| p != parent);
                        }
                    }
                }
            }
        }

        // Clear worklist
        self.worklist.clear();
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.parent.clear();
        self.rank.clear();
        self.explanations.clear();
        self.lookup.clear();
        self.use_list.clear();
        self.worklist.clear();
        self.diseqs.clear();
        self.undo_trail.clear();
        self.scope_levels = vec![0];
    }

    /// Add a disequality constraint: a ≠ b
    /// Returns None if no conflict, or Some((a, b)) if this creates a conflict
    pub fn assert_diseq(&mut self, a: TermId, b: TermId) -> Option<(TermId, TermId)> {
        let a_root = self.find(a);
        let b_root = self.find(b);

        // Conflict: asserting a ≠ b but they're already equal
        if a_root == b_root {
            return Some((a, b));
        }

        // Normalize the pair
        let pair = if a_root.0 < b_root.0 {
            (a_root, b_root)
        } else {
            (b_root, a_root)
        };

        self.diseqs.insert(pair);
        None
    }

    /// Check if asserting a = b would violate any disequality
    fn check_diseq_conflict(&mut self, a: TermId, b: TermId) -> Option<(TermId, TermId)> {
        let a_root = self.find(a);
        let b_root = self.find(b);

        let pair = if a_root.0 < b_root.0 {
            (a_root, b_root)
        } else {
            (b_root, a_root)
        };

        if self.diseqs.contains(&pair) {
            Some(pair)
        } else {
            None
        }
    }

    /// Get explanation for why two terms are equal
    #[must_use]
    pub fn get_explanation(&self, a: TermId, b: TermId) -> Option<Explanation> {
        let key = if a.0 < b.0 { (a, b) } else { (b, a) };
        self.explanations.get(&key).cloned()
    }

    /// Add a term to the congruence closure
    pub fn add_term(&mut self, term: TermId, manager: &TermManager) {
        // Initialize the term if not present
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(term) {
            e.insert(term);
            self.rank.insert(term, 0);
        }

        // Get term structure
        if let Some(t) = manager.get(term) {
            match &t.kind {
                // Binary operations
                TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::And(args)
                | TermKind::Or(args)
                    if !args.is_empty() =>
                {
                    // Normalize args by finding representatives
                    let normalized_args: Vec<_> = args.iter().map(|&a| self.find(a)).collect();

                    // Create lookup key
                    let key = (term, normalized_args.clone());

                    // Check for congruence
                    if let Some(&existing) = self.lookup.get(&key) {
                        if existing != term {
                            // Found congruent term, add to worklist
                            self.worklist.push(term);
                            self.worklist.push(existing);
                        }
                    } else {
                        self.lookup.insert(key.clone(), term);
                        self.undo_trail.push(UndoOp::LookupInsert { key });
                    }

                    // Update use lists
                    for &arg in args {
                        self.use_list.entry(arg).or_default().push(term);
                        self.undo_trail
                            .push(UndoOp::UseListInsert { arg, parent: term });
                    }
                }

                TermKind::Eq(lhs, rhs)
                | TermKind::Lt(lhs, rhs)
                | TermKind::Le(lhs, rhs)
                | TermKind::Gt(lhs, rhs)
                | TermKind::Ge(lhs, rhs)
                | TermKind::Sub(lhs, rhs)
                | TermKind::Div(lhs, rhs)
                | TermKind::Mod(lhs, rhs)
                | TermKind::Implies(lhs, rhs)
                | TermKind::Xor(lhs, rhs)
                | TermKind::BvAnd(lhs, rhs) => {
                    let args = vec![self.find(*lhs), self.find(*rhs)];
                    let key = (term, args);

                    if let Some(&existing) = self.lookup.get(&key) {
                        if existing != term {
                            self.worklist.push(term);
                            self.worklist.push(existing);
                        }
                    } else {
                        self.lookup.insert(key.clone(), term);
                        self.undo_trail.push(UndoOp::LookupInsert { key });
                    }

                    self.use_list.entry(*lhs).or_default().push(term);
                    self.undo_trail.push(UndoOp::UseListInsert {
                        arg: *lhs,
                        parent: term,
                    });
                    self.use_list.entry(*rhs).or_default().push(term);
                    self.undo_trail.push(UndoOp::UseListInsert {
                        arg: *rhs,
                        parent: term,
                    });
                }

                TermKind::Not(arg) | TermKind::Neg(arg) | TermKind::BvNot(arg) => {
                    let args = vec![self.find(*arg)];
                    let key = (term, args);

                    if let Some(&existing) = self.lookup.get(&key) {
                        if existing != term {
                            self.worklist.push(term);
                            self.worklist.push(existing);
                        }
                    } else {
                        self.lookup.insert(key.clone(), term);
                        self.undo_trail.push(UndoOp::LookupInsert { key });
                    }

                    self.use_list.entry(*arg).or_default().push(term);
                    self.undo_trail.push(UndoOp::UseListInsert {
                        arg: *arg,
                        parent: term,
                    });
                }

                TermKind::Ite(cond, then_branch, else_branch) => {
                    let args = vec![
                        self.find(*cond),
                        self.find(*then_branch),
                        self.find(*else_branch),
                    ];
                    let key = (term, args);

                    if let Some(&existing) = self.lookup.get(&key) {
                        if existing != term {
                            self.worklist.push(term);
                            self.worklist.push(existing);
                        }
                    } else {
                        self.lookup.insert(key.clone(), term);
                        self.undo_trail.push(UndoOp::LookupInsert { key });
                    }

                    self.use_list.entry(*cond).or_default().push(term);
                    self.undo_trail.push(UndoOp::UseListInsert {
                        arg: *cond,
                        parent: term,
                    });
                    self.use_list.entry(*then_branch).or_default().push(term);
                    self.undo_trail.push(UndoOp::UseListInsert {
                        arg: *then_branch,
                        parent: term,
                    });
                    self.use_list.entry(*else_branch).or_default().push(term);
                    self.undo_trail.push(UndoOp::UseListInsert {
                        arg: *else_branch,
                        parent: term,
                    });
                }

                // Constants and variables have no arguments
                _ => {}
            }
        }
    }

    /// Merge two equivalence classes with explanation
    /// Returns Some(conflict) if this merge violates a disequality, None otherwise
    pub fn merge(
        &mut self,
        a: TermId,
        b: TermId,
        explanation: Explanation,
    ) -> Option<(TermId, TermId)> {
        let a_root = self.find(a);
        let b_root = self.find(b);

        if a_root == b_root {
            return None; // Already in same class
        }

        // Check for disequality conflict
        if let Some(conflict) = self.check_diseq_conflict(a_root, b_root) {
            return Some(conflict);
        }

        // Union by rank
        let a_rank = self.rank.get(&a_root).copied().unwrap_or(0);
        let b_rank = self.rank.get(&b_root).copied().unwrap_or(0);

        let (child, parent) = if a_rank < b_rank {
            (a_root, b_root)
        } else if a_rank > b_rank {
            (b_root, a_root)
        } else {
            // Equal ranks: increase parent's rank
            self.rank.insert(a_root, a_rank + 1);
            (b_root, a_root)
        };

        // Record undo operation
        let old_parent = self.parent.get(&child).copied().unwrap_or(child);
        self.undo_trail.push(UndoOp::Merge { child, old_parent });

        // Perform the merge
        self.parent.insert(child, parent);

        // Store explanation
        let key = if a.0 < b.0 { (a, b) } else { (b, a) };
        self.explanations.insert(key, explanation);

        // Add merged terms to worklist for propagation
        self.worklist.push(a_root);
        self.worklist.push(b_root);

        None
    }

    /// Merge without explanation (for internal use)
    fn merge_internal(&mut self, a: TermId, b: TermId) {
        let _ = self.merge(a, b, Explanation::Given);
    }

    /// Process all pending merges and propagate congruences using worklist
    pub fn close(&mut self, manager: &TermManager) {
        let mut processed = FxHashSet::default();

        while let Some(term) = self.worklist.pop() {
            let root = self.find(term);

            // Skip if already processed this root
            if !processed.insert(root) {
                continue;
            }

            // Get all terms that use this term as an argument
            let parents: Vec<_> = self.use_list.get(&term).cloned().unwrap_or_default();
            let root_parents: Vec<_> = self.use_list.get(&root).cloned().unwrap_or_default();

            let all_parents: Vec<_> = parents.into_iter().chain(root_parents).collect();

            // Check for congruent parents
            for i in 0..all_parents.len() {
                for j in (i + 1)..all_parents.len() {
                    let parent_a = all_parents[i];
                    let parent_b = all_parents[j];

                    // Check if parent_a and parent_b are congruent
                    if self.are_congruent(parent_a, parent_b, manager) {
                        let pa_root = self.find(parent_a);
                        let pb_root = self.find(parent_b);

                        if pa_root != pb_root {
                            // Merge congruent terms
                            let arg_pairs = self.get_argument_pairs(parent_a, parent_b, manager);
                            let explanation = Explanation::Congruence(arg_pairs);
                            self.merge_internal(pa_root, pb_root);

                            // Store explanation for the original terms
                            let key = if parent_a.0 < parent_b.0 {
                                (parent_a, parent_b)
                            } else {
                                (parent_b, parent_a)
                            };
                            self.explanations.insert(key, explanation);
                        }
                    }
                }
            }
        }
    }

    /// Get the pairs of arguments that justify congruence
    fn get_argument_pairs(
        &mut self,
        a: TermId,
        b: TermId,
        manager: &TermManager,
    ) -> Vec<(TermId, TermId)> {
        let term_a = manager.get(a);
        let term_b = manager.get(b);

        match (term_a, term_b) {
            (Some(ta), Some(tb)) => match (&ta.kind, &tb.kind) {
                (TermKind::Add(args_a), TermKind::Add(args_b))
                | (TermKind::Mul(args_a), TermKind::Mul(args_b))
                | (TermKind::And(args_a), TermKind::And(args_b))
                | (TermKind::Or(args_a), TermKind::Or(args_b)) => args_a
                    .iter()
                    .zip(args_b.iter())
                    .map(|(&x, &y)| (x, y))
                    .collect(),

                (TermKind::Not(a), TermKind::Not(b))
                | (TermKind::Neg(a), TermKind::Neg(b))
                | (TermKind::BvNot(a), TermKind::BvNot(b)) => vec![(*a, *b)],

                (TermKind::Eq(a1, a2), TermKind::Eq(b1, b2))
                | (TermKind::Lt(a1, a2), TermKind::Lt(b1, b2))
                | (TermKind::Le(a1, a2), TermKind::Le(b1, b2))
                | (TermKind::Sub(a1, a2), TermKind::Sub(b1, b2)) => {
                    vec![(*a1, *b1), (*a2, *b2)]
                }

                (TermKind::Ite(c1, t1, e1), TermKind::Ite(c2, t2, e2)) => {
                    vec![(*c1, *c2), (*t1, *t2), (*e1, *e2)]
                }

                _ => Vec::new(),
            },
            _ => Vec::new(),
        }
    }

    /// Check if two terms are congruent (same function symbol, equivalent arguments)
    fn are_congruent(&mut self, a: TermId, b: TermId, manager: &TermManager) -> bool {
        let term_a = manager.get(a);
        let term_b = manager.get(b);

        match (term_a, term_b) {
            (Some(ta), Some(tb)) => {
                // Check if they have the same structure and equivalent arguments
                match (&ta.kind, &tb.kind) {
                    (TermKind::Add(args_a), TermKind::Add(args_b))
                    | (TermKind::Mul(args_a), TermKind::Mul(args_b))
                    | (TermKind::And(args_a), TermKind::And(args_b))
                    | (TermKind::Or(args_a), TermKind::Or(args_b)) => {
                        if args_a.len() != args_b.len() {
                            return false;
                        }
                        args_a
                            .iter()
                            .zip(args_b.iter())
                            .all(|(&a, &b)| self.are_equal(a, b))
                    }

                    (TermKind::Not(a), TermKind::Not(b))
                    | (TermKind::Neg(a), TermKind::Neg(b))
                    | (TermKind::BvNot(a), TermKind::BvNot(b)) => self.are_equal(*a, *b),

                    (TermKind::Eq(a1, a2), TermKind::Eq(b1, b2))
                    | (TermKind::Lt(a1, a2), TermKind::Lt(b1, b2))
                    | (TermKind::Le(a1, a2), TermKind::Le(b1, b2))
                    | (TermKind::Sub(a1, a2), TermKind::Sub(b1, b2)) => {
                        self.are_equal(*a1, *b1) && self.are_equal(*a2, *b2)
                    }

                    (TermKind::Ite(c1, t1, e1), TermKind::Ite(c2, t2, e2)) => {
                        self.are_equal(*c1, *c2)
                            && self.are_equal(*t1, *t2)
                            && self.are_equal(*e1, *e2)
                    }

                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Get all terms in the same equivalence class as the given term
    #[must_use]
    pub fn get_class(&mut self, term: TermId) -> Vec<TermId> {
        let root = self.find(term);
        let terms: Vec<_> = self.parent.keys().copied().collect();
        terms
            .into_iter()
            .filter(|&t| self.find(t) == root)
            .collect()
    }

    /// Get the number of equivalence classes
    #[must_use]
    pub fn num_classes(&mut self) -> usize {
        let terms: Vec<_> = self.parent.keys().copied().collect();
        let mut roots: Vec<_> = terms.iter().map(|&t| self.find(t)).collect();
        roots.sort_unstable();
        roots.dedup();
        roots.len()
    }
}

impl Default for CongruenceClosure {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_closure() {
        let cc = CongruenceClosure::new();
        assert_eq!(cc.parent.len(), 0);
    }

    #[test]
    fn test_find_creates_class() {
        let mut cc = CongruenceClosure::new();
        let term = TermId(1);
        let root = cc.find(term);
        assert_eq!(root, term);
        assert_eq!(cc.find(term), root);
    }

    #[test]
    fn test_merge_same_class() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);

        cc.merge(a, b, Explanation::Given);
        assert!(cc.are_equal(a, b));
    }

    #[test]
    fn test_transitivity() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);

        cc.merge(a, b, Explanation::Given);
        cc.merge(b, c, Explanation::Given);

        // a = b and b = c implies a = c
        assert!(cc.are_equal(a, c));
    }

    #[test]
    fn test_get_class() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);

        cc.merge(a, b, Explanation::Given);
        cc.merge(b, c, Explanation::Given);

        let class = cc.get_class(a);
        assert_eq!(class.len(), 3);
        assert!(class.contains(&a));
        assert!(class.contains(&b));
        assert!(class.contains(&c));
    }

    #[test]
    fn test_num_classes() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);
        let d = TermId(4);

        cc.find(a);
        cc.find(b);
        cc.find(c);
        cc.find(d);
        assert_eq!(cc.num_classes(), 4);

        cc.merge(a, b, Explanation::Given);
        assert_eq!(cc.num_classes(), 3);

        cc.merge(c, d, Explanation::Given);
        assert_eq!(cc.num_classes(), 2);

        cc.merge(a, c, Explanation::Given);
        assert_eq!(cc.num_classes(), 1);
    }

    #[test]
    fn test_add_term_simple() {
        let mut manager = TermManager::new();
        let mut cc = CongruenceClosure::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        cc.add_term(x, &manager);

        assert!(cc.parent.contains_key(&x));
    }

    #[test]
    fn test_basic_usage() {
        let mut manager = TermManager::new();
        let mut cc = CongruenceClosure::new();

        // Create some simple terms
        let a = manager.mk_var("a", manager.sorts.int_sort);
        let b = manager.mk_var("b", manager.sorts.int_sort);
        let c = manager.mk_var("c", manager.sorts.int_sort);

        cc.add_term(a, &manager);
        cc.add_term(b, &manager);
        cc.add_term(c, &manager);

        // Initially all different
        assert!(!cc.are_equal(a, b));
        assert!(!cc.are_equal(b, c));

        // Merge a and b
        cc.merge(a, b, Explanation::Given);
        assert!(cc.are_equal(a, b));

        // Merge b and c
        cc.merge(b, c, Explanation::Given);
        assert!(cc.are_equal(b, c));
        assert!(cc.are_equal(a, c)); // Transitivity
    }

    #[test]
    fn test_push_pop() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);

        // Initial scope
        cc.merge(a, b, Explanation::Given);
        assert!(cc.are_equal(a, b));

        // Push new scope
        cc.push();
        cc.merge(b, c, Explanation::Given);
        assert!(cc.are_equal(a, c));

        // Pop scope - should undo b = c but keep a = b
        cc.pop();
        assert!(cc.are_equal(a, b));
        assert!(!cc.are_equal(b, c));
    }

    #[test]
    fn test_diseq_conflict() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);

        // Assert a != b
        assert!(cc.assert_diseq(a, b).is_none());

        // Try to merge a and b - should create conflict
        let conflict = cc.merge(a, b, Explanation::Given);
        assert!(conflict.is_some());
    }

    #[test]
    fn test_diseq_after_merge() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);

        // Merge a and b first
        cc.merge(a, b, Explanation::Given);

        // Try to assert a != b - should create conflict
        let conflict = cc.assert_diseq(a, b);
        assert!(conflict.is_some());
    }

    #[test]
    fn test_explanation() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);

        cc.merge(a, b, Explanation::Given);

        // Should have an explanation for this merge
        let exp = cc.get_explanation(a, b);
        assert!(exp.is_some());
        assert_eq!(exp.unwrap(), Explanation::Given);
    }
}
