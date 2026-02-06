//! Equality Propagation Engine for Theory Combination.
#![allow(dead_code)] // Under development
//!
//! Implements efficient equality propagation between theories using:
//! - Congruence closure with union-find
//! - E-graph for term rewriting
//! - Equality explanation generation
//! - Watched equalities for lazy propagation

use oxiz_core::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Equality propagation engine.
pub struct EqualityPropagator {
    /// Union-find for equality classes
    union_find: UnionFind,
    /// Congruence closure data structures
    congruence: CongruenceData,
    /// Pending equalities to propagate
    pending: VecDeque<(TermId, TermId, Explanation)>,
    /// Watched equalities: term → watchers
    watched: FxHashMap<TermId, Vec<EqualityWatch>>,
    /// E-graph for term canonicalization
    egraph: EGraph,
    /// Statistics
    stats: EqualityPropStats,
}

/// Union-find data structure for equivalence classes.
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// Parent pointers
    parent: FxHashMap<TermId, TermId>,
    /// Rank for union-by-rank
    rank: FxHashMap<TermId, usize>,
    /// Size of equivalence class
    size: FxHashMap<TermId, usize>,
}

/// Congruence closure data.
#[derive(Debug, Clone)]
pub struct CongruenceData {
    /// Use list: term → terms that use it
    use_list: FxHashMap<TermId, Vec<TermId>>,
    /// Lookup table: (function, args) → term
    lookup: FxHashMap<CongruenceKey, TermId>,
    /// Pending congruence checks
    pending_congruences: VecDeque<(TermId, TermId)>,
}

/// Key for congruence lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CongruenceKey {
    /// Function/operator
    pub function: TermKind,
    /// Canonical arguments (equivalence class representatives)
    pub args: Vec<TermId>,
}

/// E-graph for term canonicalization.
#[derive(Debug, Clone)]
pub struct EGraph {
    /// E-class membership: term → e-class
    eclass: FxHashMap<TermId, EClassId>,
    /// E-class contents: e-class → terms
    nodes: FxHashMap<EClassId, Vec<TermId>>,
    /// E-class data
    data: FxHashMap<EClassId, EClassData>,
    /// Next available e-class ID
    next_id: EClassId,
}

/// E-class identifier.
pub type EClassId = usize;

/// Data associated with an e-class.
#[derive(Debug, Clone)]
pub struct EClassData {
    /// Representative term
    pub representative: TermId,
    /// Size of e-class
    pub size: usize,
    /// Parent e-classes (for congruence)
    pub parents: Vec<EClassId>,
}

/// Explanation for an equality.
#[derive(Debug, Clone)]
pub enum Explanation {
    /// Given equality (axiom)
    Given,
    /// Equality by reflexivity
    Reflexivity,
    /// Equality by transitivity
    Transitivity(TermId, Box<Explanation>, Box<Explanation>),
    /// Equality by congruence
    Congruence(Vec<(TermId, TermId, Box<Explanation>)>),
    /// Theory propagation
    TheoryPropagation(TheoryExplanation),
}

/// Theory-specific explanation.
#[derive(Debug, Clone)]
pub struct TheoryExplanation {
    /// Theory ID
    pub theory_id: usize,
    /// Antecedent equalities
    pub antecedents: Vec<(TermId, TermId)>,
}

/// Watched equality for lazy propagation.
#[derive(Debug, Clone)]
pub struct EqualityWatch {
    /// Left-hand side
    pub lhs: TermId,
    /// Right-hand side
    pub rhs: TermId,
    /// Callback ID
    pub callback: usize,
}

/// Equality propagation statistics.
#[derive(Debug, Clone, Default)]
pub struct EqualityPropStats {
    /// Equalities propagated
    pub equalities_propagated: usize,
    /// Congruences found
    pub congruences_found: usize,
    /// E-graph merges
    pub egraph_merges: usize,
    /// Explanations generated
    pub explanations_generated: usize,
    /// Watched equality triggers
    pub watch_triggers: usize,
}

impl UnionFind {
    /// Create a new union-find structure.
    pub fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
            size: FxHashMap::default(),
        }
    }

    /// Find the representative of a set.
    pub fn find(&mut self, x: TermId) -> TermId {
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(x) {
            e.insert(x);
            self.rank.insert(x, 0);
            self.size.insert(x, 1);
            return x;
        }

        let parent = self.parent[&x];
        if parent != x {
            // Path compression
            let root = self.find(parent);
            self.parent.insert(x, root);
            root
        } else {
            x
        }
    }

    /// Union two sets.
    pub fn union(&mut self, x: TermId, y: TermId) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false; // Already in same set
        }

        let rank_x = self.rank.get(&root_x).copied().unwrap_or(0);
        let rank_y = self.rank.get(&root_y).copied().unwrap_or(0);

        // Union by rank
        if rank_x < rank_y {
            self.parent.insert(root_x, root_y);
            let size_x = self.size.get(&root_x).copied().unwrap_or(1);
            *self.size.entry(root_y).or_insert(1) += size_x;
        } else if rank_x > rank_y {
            self.parent.insert(root_y, root_x);
            let size_y = self.size.get(&root_y).copied().unwrap_or(1);
            *self.size.entry(root_x).or_insert(1) += size_y;
        } else {
            self.parent.insert(root_y, root_x);
            *self.rank.entry(root_x).or_insert(0) += 1;
            let size_y = self.size.get(&root_y).copied().unwrap_or(1);
            *self.size.entry(root_x).or_insert(1) += size_y;
        }

        true
    }

    /// Check if two elements are in the same set.
    pub fn connected(&mut self, x: TermId, y: TermId) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get size of the set containing x.
    pub fn set_size(&mut self, x: TermId) -> usize {
        let root = self.find(x);
        self.size[&root]
    }
}

impl EqualityPropagator {
    /// Create a new equality propagator.
    pub fn new() -> Self {
        Self {
            union_find: UnionFind::new(),
            congruence: CongruenceData::new(),
            pending: VecDeque::new(),
            watched: FxHashMap::default(),
            egraph: EGraph::new(),
            stats: EqualityPropStats::default(),
        }
    }

    /// Assert an equality.
    pub fn assert_equality(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        explanation: Explanation,
        tm: &TermManager,
    ) -> Result<(), String> {
        // Check if already equal
        if self.union_find.connected(lhs, rhs) {
            return Ok(());
        }

        // Add to pending queue
        self.pending.push_back((lhs, rhs, explanation));

        // Propagate all pending equalities
        self.propagate(tm)?;

        Ok(())
    }

    /// Propagate all pending equalities.
    fn propagate(&mut self, tm: &TermManager) -> Result<(), String> {
        while let Some((lhs, rhs, explanation)) = self.pending.pop_front() {
            self.propagate_equality(lhs, rhs, explanation, tm)?;
        }

        // Check for new congruences
        self.check_congruences(tm)?;

        Ok(())
    }

    /// Propagate a single equality.
    fn propagate_equality(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        _explanation: Explanation,
        _tm: &TermManager,
    ) -> Result<(), String> {
        // Union in union-find
        if !self.union_find.union(lhs, rhs) {
            return Ok(()); // Already merged
        }

        self.stats.equalities_propagated += 1;

        // Merge in e-graph
        self.egraph.merge(lhs, rhs);
        self.stats.egraph_merges += 1;

        // Update use lists
        self.congruence.merge_use_lists(lhs, rhs);

        // Trigger watches
        self.trigger_watches(lhs, rhs)?;

        // Add parents to pending congruence checks
        let lhs_parents = self.congruence.get_parents(lhs);
        let rhs_parents = self.congruence.get_parents(rhs);

        for lhs_parent in lhs_parents {
            for &rhs_parent in &rhs_parents {
                self.congruence
                    .pending_congruences
                    .push_back((lhs_parent, rhs_parent));
            }
        }

        Ok(())
    }

    /// Check for new congruences.
    fn check_congruences(&mut self, tm: &TermManager) -> Result<(), String> {
        while let Some((t1, t2)) = self.congruence.pending_congruences.pop_front() {
            // Check if they have congruent arguments
            if self.are_congruent(t1, t2, tm)? {
                self.stats.congruences_found += 1;

                // Generate congruence explanation
                let explanation = self.generate_congruence_explanation(t1, t2, tm)?;

                // Assert equality
                self.pending.push_back((t1, t2, explanation));
            }
        }

        Ok(())
    }

    /// Check if two terms are congruent.
    fn are_congruent(&mut self, t1: TermId, t2: TermId, tm: &TermManager) -> Result<bool, String> {
        let term1 = tm.get(t1).ok_or("term not found")?;
        let term2 = tm.get(t2).ok_or("term not found")?;

        // Must have same kind
        if std::mem::discriminant(&term1.kind) != std::mem::discriminant(&term2.kind) {
            return Ok(false);
        }

        // Get arguments
        let args1 = self.get_args(&term1.kind);
        let args2 = self.get_args(&term2.kind);

        if args1.len() != args2.len() {
            return Ok(false);
        }

        // Check if all arguments are equal
        for (arg1, arg2) in args1.iter().zip(args2.iter()) {
            if !self.union_find.connected(*arg1, *arg2) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Generate explanation for congruence.
    fn generate_congruence_explanation(
        &mut self,
        t1: TermId,
        t2: TermId,
        tm: &TermManager,
    ) -> Result<Explanation, String> {
        let term1 = tm.get(t1).ok_or("term not found")?;
        let term2 = tm.get(t2).ok_or("term not found")?;

        let args1 = self.get_args(&term1.kind);
        let args2 = self.get_args(&term2.kind);

        let mut arg_explanations = Vec::new();

        for (arg1, arg2) in args1.iter().zip(args2.iter()) {
            let expl = self.explain_equality(*arg1, *arg2)?;
            arg_explanations.push((*arg1, *arg2, Box::new(expl)));
        }

        self.stats.explanations_generated += 1;

        Ok(Explanation::Congruence(arg_explanations))
    }

    /// Explain why two terms are equal.
    pub fn explain_equality(&mut self, lhs: TermId, rhs: TermId) -> Result<Explanation, String> {
        if lhs == rhs {
            return Ok(Explanation::Reflexivity);
        }

        if !self.union_find.connected(lhs, rhs) {
            return Err("Terms are not equal".to_string());
        }

        // Simplified: return a generic explanation
        // Full implementation would trace union-find path
        Ok(Explanation::Given)
    }

    /// Watch an equality.
    pub fn watch_equality(&mut self, lhs: TermId, rhs: TermId, callback: usize) {
        let watch = EqualityWatch { lhs, rhs, callback };

        self.watched.entry(lhs).or_default().push(watch.clone());
        self.watched.entry(rhs).or_default().push(watch);
    }

    /// Trigger watches when an equality is established.
    fn trigger_watches(&mut self, lhs: TermId, rhs: TermId) -> Result<(), String> {
        let mut triggered = Vec::new();

        // Check watches on lhs
        if let Some(watches) = self.watched.get(&lhs) {
            for watch in watches {
                if self.union_find.connected(watch.lhs, watch.rhs) {
                    triggered.push(watch.callback);
                }
            }
        }

        // Check watches on rhs
        if let Some(watches) = self.watched.get(&rhs) {
            for watch in watches {
                if self.union_find.connected(watch.lhs, watch.rhs) {
                    triggered.push(watch.callback);
                }
            }
        }

        self.stats.watch_triggers += triggered.len();

        Ok(())
    }

    /// Get arguments of a term.
    fn get_args(&self, kind: &TermKind) -> Vec<TermId> {
        match kind {
            TermKind::And(args) | TermKind::Or(args) => args.to_vec(),
            TermKind::Not(arg) => vec![*arg],
            TermKind::Eq(l, r) | TermKind::Le(l, r) | TermKind::Lt(l, r) => vec![*l, *r],
            TermKind::Add(args) | TermKind::Mul(args) => args.to_vec(),
            _ => vec![],
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &EqualityPropStats {
        &self.stats
    }
}

impl CongruenceData {
    /// Create new congruence data.
    pub fn new() -> Self {
        Self {
            use_list: FxHashMap::default(),
            lookup: FxHashMap::default(),
            pending_congruences: VecDeque::new(),
        }
    }

    /// Merge use lists when two terms become equal.
    pub fn merge_use_lists(&mut self, t1: TermId, t2: TermId) {
        // Simplified implementation
        let t1_uses = self.use_list.get(&t1).cloned().unwrap_or_default();
        let t2_uses = self.use_list.get(&t2).cloned().unwrap_or_default();

        let mut merged = t1_uses;
        merged.extend(t2_uses);

        self.use_list.insert(t1, merged.clone());
        self.use_list.insert(t2, merged);
    }

    /// Get parent terms.
    pub fn get_parents(&self, t: TermId) -> Vec<TermId> {
        self.use_list.get(&t).cloned().unwrap_or_default()
    }
}

impl EGraph {
    /// Create a new e-graph.
    pub fn new() -> Self {
        Self {
            eclass: FxHashMap::default(),
            nodes: FxHashMap::default(),
            data: FxHashMap::default(),
            next_id: 0,
        }
    }

    /// Get e-class for a term.
    pub fn get_eclass(&mut self, term: TermId) -> EClassId {
        if let Some(&id) = self.eclass.get(&term) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;

            self.eclass.insert(term, id);
            self.nodes.insert(id, vec![term]);
            self.data.insert(
                id,
                EClassData {
                    representative: term,
                    size: 1,
                    parents: Vec::new(),
                },
            );

            id
        }
    }

    /// Merge two terms in the e-graph.
    pub fn merge(&mut self, t1: TermId, t2: TermId) {
        let id1 = self.get_eclass(t1);
        let id2 = self.get_eclass(t2);

        if id1 == id2 {
            return;
        }

        // Merge smaller into larger
        let size1 = self.data[&id1].size;
        let size2 = self.data[&id2].size;

        let (smaller, larger) = if size1 < size2 {
            (id1, id2)
        } else {
            (id2, id1)
        };

        // Update e-class membership
        let smaller_nodes = self.nodes[&smaller].clone();
        for &node in &smaller_nodes {
            self.eclass.insert(node, larger);
        }

        // Merge node lists
        if let Some(larger_nodes) = self.nodes.get_mut(&larger) {
            larger_nodes.extend(smaller_nodes);
        }
        self.nodes.remove(&smaller);

        // Update data
        let smaller_size = self.data.get(&smaller).map(|d| d.size).unwrap_or(0);
        if let Some(larger_data) = self.data.get_mut(&larger) {
            larger_data.size += smaller_size;
        }
        self.data.remove(&smaller);
    }
}

impl Default for EqualityPropagator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for UnionFind {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CongruenceData {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new();

        let t1 = TermId::from(1);
        let t2 = TermId::from(2);
        let t3 = TermId::from(3);

        assert!(!uf.connected(t1, t2));

        uf.union(t1, t2);
        assert!(uf.connected(t1, t2));

        uf.union(t2, t3);
        assert!(uf.connected(t1, t3));
    }

    #[test]
    fn test_equality_propagator() {
        let prop = EqualityPropagator::new();
        assert_eq!(prop.stats.equalities_propagated, 0);
    }

    #[test]
    fn test_egraph() {
        let mut eg = EGraph::new();

        let t1 = TermId::from(1);
        let t2 = TermId::from(2);

        let id1 = eg.get_eclass(t1);
        let id2 = eg.get_eclass(t2);

        assert_ne!(id1, id2);

        eg.merge(t1, t2);

        let id1_after = eg.get_eclass(t1);
        let id2_after = eg.get_eclass(t2);

        assert_eq!(id1_after, id2_after);
    }
}
