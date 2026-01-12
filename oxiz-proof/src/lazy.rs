//! Lazy proof evaluation for efficient proof processing.
//!
//! This module provides lazy evaluation strategies that defer computation
//! until results are actually needed, improving performance for large proofs.

use crate::proof::{Proof, ProofNode, ProofNodeId, ProofStep};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;

/// A lazily-evaluated proof node.
#[derive(Debug, Clone)]
pub struct LazyNode {
    /// Node ID
    pub id: ProofNodeId,
    /// Cached node data (None if not yet evaluated)
    cached: Rc<RefCell<Option<ProofNode>>>,
    /// Proof reference for lazy loading
    proof: Rc<Proof>,
}

impl LazyNode {
    /// Create a new lazy node.
    fn new(id: ProofNodeId, proof: Rc<Proof>) -> Self {
        Self {
            id,
            cached: Rc::new(RefCell::new(None)),
            proof,
        }
    }

    /// Force evaluation and get the node.
    pub fn force(&self) -> Option<ProofNode> {
        // Check cache first
        if let Some(node) = self.cached.borrow().as_ref() {
            return Some(node.clone());
        }

        // Load from proof
        if let Some(node) = self.proof.get_node(self.id) {
            *self.cached.borrow_mut() = Some(node.clone());
            Some(node.clone())
        } else {
            None
        }
    }

    /// Check if node has been evaluated.
    pub fn is_forced(&self) -> bool {
        self.cached.borrow().is_some()
    }

    /// Get the conclusion without full evaluation (if cached).
    pub fn conclusion_if_cached(&self) -> Option<String> {
        self.cached
            .borrow()
            .as_ref()
            .map(|node| node.conclusion().to_string())
    }
}

/// Lazy proof wrapper for deferred computation.
pub struct LazyProof {
    /// The underlying proof
    proof: Rc<Proof>,
    /// Lazy node cache
    node_cache: RefCell<FxHashMap<ProofNodeId, LazyNode>>,
    /// Statistics on cache hits/misses
    stats: RefCell<LazyStats>,
}

/// Statistics for lazy evaluation.
#[derive(Debug, Clone, Default)]
pub struct LazyStats {
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of forced evaluations
    pub forced_evaluations: usize,
}

impl LazyStats {
    /// Get the cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

impl LazyProof {
    /// Create a new lazy proof wrapper.
    pub fn new(proof: Proof) -> Self {
        Self {
            proof: Rc::new(proof),
            node_cache: RefCell::new(FxHashMap::default()),
            stats: RefCell::new(LazyStats::default()),
        }
    }

    /// Get a lazy node by ID.
    pub fn get_lazy_node(&self, id: ProofNodeId) -> Option<LazyNode> {
        // Check cache
        if let Some(lazy_node) = self.node_cache.borrow().get(&id) {
            self.stats.borrow_mut().cache_hits += 1;
            return Some(lazy_node.clone());
        }

        // Create new lazy node
        self.stats.borrow_mut().cache_misses += 1;
        if self.proof.get_node(id).is_some() {
            let lazy_node = LazyNode::new(id, Rc::clone(&self.proof));
            self.node_cache.borrow_mut().insert(id, lazy_node.clone());
            Some(lazy_node)
        } else {
            None
        }
    }

    /// Force evaluation of a node.
    pub fn force_node(&self, id: ProofNodeId) -> Option<ProofNode> {
        if let Some(lazy_node) = self.get_lazy_node(id) {
            self.stats.borrow_mut().forced_evaluations += 1;
            lazy_node.force()
        } else {
            None
        }
    }

    /// Get premises of a node lazily.
    pub fn get_premises_lazy(&self, id: ProofNodeId) -> Vec<LazyNode> {
        if let Some(node) = self.force_node(id) {
            if let ProofStep::Inference { premises, .. } = &node.step {
                premises
                    .iter()
                    .filter_map(|&p| self.get_lazy_node(p))
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Check if a node is an axiom without forcing full evaluation.
    pub fn is_axiom_lazy(&self, id: ProofNodeId) -> bool {
        if let Some(node) = self.force_node(id) {
            matches!(node.step, ProofStep::Axiom { .. })
        } else {
            false
        }
    }

    /// Get the number of nodes in the proof.
    pub fn len(&self) -> usize {
        self.proof.len()
    }

    /// Check if the proof is empty.
    pub fn is_empty(&self) -> bool {
        self.proof.is_empty()
    }

    /// Get lazy evaluation statistics.
    pub fn get_stats(&self) -> LazyStats {
        self.stats.borrow().clone()
    }

    /// Clear the node cache.
    pub fn clear_cache(&self) {
        self.node_cache.borrow_mut().clear();
    }

    /// Get the underlying proof reference.
    pub fn proof(&self) -> &Proof {
        &self.proof
    }
}

/// Lazy iterator over proof nodes.
pub struct LazyNodeIterator {
    lazy_proof: Rc<LazyProof>,
    current_index: usize,
    node_ids: Vec<ProofNodeId>,
}

impl LazyNodeIterator {
    /// Create a new lazy iterator.
    pub fn new(lazy_proof: &LazyProof) -> Self {
        let node_ids = lazy_proof.proof.nodes().iter().map(|n| n.id).collect();
        Self {
            lazy_proof: Rc::new(LazyProof::new((*lazy_proof.proof).clone())),
            current_index: 0,
            node_ids,
        }
    }
}

impl Iterator for LazyNodeIterator {
    type Item = LazyNode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.node_ids.len() {
            let id = self.node_ids[self.current_index];
            self.current_index += 1;
            self.lazy_proof.get_lazy_node(id)
        } else {
            None
        }
    }
}

/// Lazy dependency resolver.
pub struct LazyDependencyResolver {
    lazy_proof: LazyProof,
}

impl LazyDependencyResolver {
    /// Create a new lazy dependency resolver.
    pub fn new(proof: Proof) -> Self {
        Self {
            lazy_proof: LazyProof::new(proof),
        }
    }

    /// Get all dependencies of a node lazily.
    pub fn get_dependencies(&self, id: ProofNodeId) -> Vec<ProofNodeId> {
        let mut dependencies = Vec::new();
        let mut visited = FxHashMap::default();
        self.collect_dependencies(id, &mut dependencies, &mut visited);
        dependencies
    }

    // Helper: Recursively collect dependencies
    fn collect_dependencies(
        &self,
        id: ProofNodeId,
        deps: &mut Vec<ProofNodeId>,
        visited: &mut FxHashMap<ProofNodeId, bool>,
    ) {
        if visited.contains_key(&id) {
            return;
        }
        visited.insert(id, true);

        let premises = self.lazy_proof.get_premises_lazy(id);
        for premise in premises {
            self.collect_dependencies(premise.id, deps, visited);
        }

        deps.push(id);
    }

    /// Check if node A depends on node B.
    pub fn depends_on(&self, a: ProofNodeId, b: ProofNodeId) -> bool {
        self.get_dependencies(a).contains(&b)
    }

    /// Get evaluation statistics.
    pub fn get_stats(&self) -> LazyStats {
        self.lazy_proof.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_node_new() {
        let proof = Proof::new();
        let proof_rc = Rc::new(proof);
        let lazy_node = LazyNode::new(ProofNodeId(0), proof_rc);
        assert_eq!(lazy_node.id, ProofNodeId(0));
        assert!(!lazy_node.is_forced());
    }

    #[test]
    fn test_lazy_proof_new() {
        let proof = Proof::new();
        let lazy_proof = LazyProof::new(proof);
        assert!(lazy_proof.is_empty());
    }

    #[test]
    fn test_get_lazy_node() {
        let mut proof = Proof::new();
        let id = proof.add_axiom("x = x");
        let lazy_proof = LazyProof::new(proof);

        let lazy_node = lazy_proof.get_lazy_node(id);
        assert!(lazy_node.is_some());
    }

    #[test]
    fn test_force_node() {
        let mut proof = Proof::new();
        let id = proof.add_axiom("x = x");
        let lazy_proof = LazyProof::new(proof);

        let node = lazy_proof.force_node(id);
        assert!(node.is_some());
        assert_eq!(node.unwrap().conclusion(), "x = x");
    }

    #[test]
    fn test_is_axiom_lazy() {
        let mut proof = Proof::new();
        let id = proof.add_axiom("x = x");
        let lazy_proof = LazyProof::new(proof);

        assert!(lazy_proof.is_axiom_lazy(id));
    }

    #[test]
    fn test_lazy_stats() {
        let mut proof = Proof::new();
        let id = proof.add_axiom("x = x");
        let lazy_proof = LazyProof::new(proof);

        // First access - cache miss
        let _ = lazy_proof.get_lazy_node(id);
        let stats = lazy_proof.get_stats();
        assert_eq!(stats.cache_misses, 1);

        // Second access - cache hit
        let _ = lazy_proof.get_lazy_node(id);
        let stats = lazy_proof.get_stats();
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_lazy_stats_hit_rate() {
        let stats = LazyStats {
            cache_hits: 7,
            cache_misses: 3,
            forced_evaluations: 10,
        };
        assert_eq!(stats.hit_rate(), 0.7);
    }

    #[test]
    fn test_clear_cache() {
        let mut proof = Proof::new();
        let id = proof.add_axiom("x = x");
        let lazy_proof = LazyProof::new(proof);

        let _ = lazy_proof.get_lazy_node(id);
        lazy_proof.clear_cache();

        // Should be cache miss after clear
        let _ = lazy_proof.get_lazy_node(id);
        let stats = lazy_proof.get_stats();
        assert_eq!(stats.cache_misses, 2); // One before clear, one after
    }

    #[test]
    fn test_lazy_dependency_resolver() {
        let mut proof = Proof::new();
        let ax1 = proof.add_axiom("x = x");
        let ax2 = proof.add_axiom("y = y");
        let res = proof.add_inference("resolution", vec![ax1, ax2], "x = x or y = y");

        let resolver = LazyDependencyResolver::new(proof);
        let deps = resolver.get_dependencies(res);
        assert_eq!(deps.len(), 3); // ax1, ax2, res
    }

    #[test]
    fn test_depends_on() {
        let mut proof = Proof::new();
        let ax1 = proof.add_axiom("x = x");
        let ax2 = proof.add_axiom("y = y");
        let res = proof.add_inference("resolution", vec![ax1, ax2], "x = x or y = y");

        let resolver = LazyDependencyResolver::new(proof);
        assert!(resolver.depends_on(res, ax1));
        assert!(resolver.depends_on(res, ax2));
        assert!(!resolver.depends_on(ax1, res));
    }

    #[test]
    fn test_get_premises_lazy() {
        let mut proof = Proof::new();
        let ax1 = proof.add_axiom("x = x");
        let ax2 = proof.add_axiom("y = y");
        let res = proof.add_inference("resolution", vec![ax1, ax2], "x = x or y = y");

        let lazy_proof = LazyProof::new(proof);
        let premises = lazy_proof.get_premises_lazy(res);
        assert_eq!(premises.len(), 2);
    }
}
