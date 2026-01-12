//! Proof traversal and transformation utilities.
//!
//! This module provides various ways to traverse and transform proof trees,
//! including visitors, iterators, and transformation passes.

use crate::proof::{Proof, ProofNode, ProofNodeId, ProofStep};
use std::collections::{HashSet, VecDeque};

/// Visitor trait for proof tree traversal.
pub trait ProofVisitor {
    /// Visit a proof node.
    fn visit_node(&mut self, proof: &Proof, node: &ProofNode);

    /// Visit an axiom node.
    fn visit_axiom(&mut self, _proof: &Proof, _id: ProofNodeId, _conclusion: &str) {}

    /// Visit an inference node.
    fn visit_inference(
        &mut self,
        _proof: &Proof,
        _id: ProofNodeId,
        _rule: &str,
        _premises: &[ProofNodeId],
        _conclusion: &str,
    ) {
    }
}

/// Proof tree traversal order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalOrder {
    /// Pre-order: visit node before its children.
    PreOrder,
    /// Post-order: visit node after its children.
    PostOrder,
    /// Breadth-first: visit nodes level by level.
    BreadthFirst,
}

/// Traverse a proof tree with a visitor.
pub fn traverse<V: ProofVisitor>(proof: &Proof, visitor: &mut V, order: TraversalOrder) {
    if let Some(root) = proof.root() {
        let mut visited = HashSet::new();
        match order {
            TraversalOrder::PreOrder => traverse_pre_order(proof, root, visitor, &mut visited),
            TraversalOrder::PostOrder => traverse_post_order(proof, root, visitor, &mut visited),
            TraversalOrder::BreadthFirst => traverse_breadth_first(proof, visitor),
        }
    }
}

/// Pre-order traversal (root, then children).
fn traverse_pre_order<V: ProofVisitor>(
    proof: &Proof,
    node_id: ProofNodeId,
    visitor: &mut V,
    visited: &mut HashSet<ProofNodeId>,
) {
    if visited.contains(&node_id) {
        return;
    }
    visited.insert(node_id);

    if let Some(node) = proof.get_node(node_id) {
        // Visit this node first
        visitor.visit_node(proof, node);

        match &node.step {
            ProofStep::Axiom { conclusion } => {
                visitor.visit_axiom(proof, node_id, conclusion);
            }
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                ..
            } => {
                visitor.visit_inference(proof, node_id, rule, premises, conclusion);

                // Then visit children
                for &premise in premises {
                    traverse_pre_order(proof, premise, visitor, visited);
                }
            }
        }
    }
}

/// Post-order traversal (children first, then root).
fn traverse_post_order<V: ProofVisitor>(
    proof: &Proof,
    node_id: ProofNodeId,
    visitor: &mut V,
    visited: &mut HashSet<ProofNodeId>,
) {
    if visited.contains(&node_id) {
        return;
    }
    visited.insert(node_id);

    if let Some(node) = proof.get_node(node_id) {
        // Visit children first
        if let ProofStep::Inference { premises, .. } = &node.step {
            for &premise in premises {
                traverse_post_order(proof, premise, visitor, visited);
            }
        }

        // Then visit this node
        visitor.visit_node(proof, node);

        match &node.step {
            ProofStep::Axiom { conclusion } => {
                visitor.visit_axiom(proof, node_id, conclusion);
            }
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                ..
            } => {
                visitor.visit_inference(proof, node_id, rule, premises, conclusion);
            }
        }
    }
}

/// Breadth-first traversal (level by level).
fn traverse_breadth_first<V: ProofVisitor>(proof: &Proof, visitor: &mut V) {
    if let Some(root) = proof.root() {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back(root);
        visited.insert(root);

        while let Some(node_id) = queue.pop_front() {
            if let Some(node) = proof.get_node(node_id) {
                visitor.visit_node(proof, node);

                match &node.step {
                    ProofStep::Axiom { conclusion } => {
                        visitor.visit_axiom(proof, node_id, conclusion);
                    }
                    ProofStep::Inference {
                        rule,
                        premises,
                        conclusion,
                        ..
                    } => {
                        visitor.visit_inference(proof, node_id, rule, premises, conclusion);

                        for &premise in premises {
                            if !visited.contains(&premise) {
                                visited.insert(premise);
                                queue.push_back(premise);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Collect all nodes in topological order (leaves first, root last).
#[must_use]
pub fn topological_order(proof: &Proof) -> Vec<ProofNodeId> {
    let mut order = Vec::new();
    if let Some(root) = proof.root() {
        let mut visited = HashSet::new();
        collect_topological(proof, root, &mut order, &mut visited);
    }
    order
}

fn collect_topological(
    proof: &Proof,
    node_id: ProofNodeId,
    order: &mut Vec<ProofNodeId>,
    visited: &mut HashSet<ProofNodeId>,
) {
    if visited.contains(&node_id) {
        return;
    }
    visited.insert(node_id);

    if let Some(node) = proof.get_node(node_id)
        && let ProofStep::Inference { premises, .. } = &node.step
    {
        for &premise in premises {
            collect_topological(proof, premise, order, visited);
        }
    }

    order.push(node_id);
}

/// Find all paths from leaves to root.
#[must_use]
pub fn find_all_paths(proof: &Proof) -> Vec<Vec<ProofNodeId>> {
    let mut paths = Vec::new();
    if let Some(root) = proof.root() {
        let mut current_path = Vec::new();
        collect_paths(proof, root, &mut current_path, &mut paths);
    }
    paths
}

fn collect_paths(
    proof: &Proof,
    node_id: ProofNodeId,
    current_path: &mut Vec<ProofNodeId>,
    all_paths: &mut Vec<Vec<ProofNodeId>>,
) {
    current_path.push(node_id);

    if let Some(node) = proof.get_node(node_id) {
        match &node.step {
            ProofStep::Axiom { .. } => {
                // Reached a leaf - save the path
                all_paths.push(current_path.clone());
            }
            ProofStep::Inference { premises, .. } => {
                // Continue down each premise
                for &premise in premises {
                    collect_paths(proof, premise, current_path, all_paths);
                }
            }
        }
    }

    current_path.pop();
}

/// A visitor that counts nodes by type.

#[derive(Debug, Default)]
pub struct NodeCounter {
    /// Number of axiom nodes.
    pub axioms: usize,
    /// Number of inference nodes.
    pub inferences: usize,
}

impl ProofVisitor for NodeCounter {
    fn visit_node(&mut self, _proof: &Proof, node: &ProofNode) {
        match node.step {
            ProofStep::Axiom { .. } => self.axioms += 1,
            ProofStep::Inference { .. } => self.inferences += 1,
        }
    }
}

/// A visitor that collects all conclusions.

#[derive(Debug, Default)]
pub struct ConclusionCollector {
    /// All conclusions in the proof.
    pub conclusions: Vec<String>,
}

impl ProofVisitor for ConclusionCollector {
    fn visit_node(&mut self, _proof: &Proof, node: &ProofNode) {
        let conclusion = match &node.step {
            ProofStep::Axiom { conclusion } => conclusion.clone(),
            ProofStep::Inference { conclusion, .. } => conclusion.clone(),
        };
        self.conclusions.push(conclusion);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_counter() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let _p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");

        let mut counter = NodeCounter::default();
        traverse(&proof, &mut counter, TraversalOrder::PreOrder);

        assert_eq!(counter.axioms, 2);
        assert_eq!(counter.inferences, 1);
    }

    #[test]
    fn test_conclusion_collector() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let _p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");

        let mut collector = ConclusionCollector::default();
        traverse(&proof, &mut collector, TraversalOrder::PreOrder);

        assert_eq!(collector.conclusions.len(), 3);
        assert!(collector.conclusions.contains(&"p".to_string()));
        assert!(collector.conclusions.contains(&"q".to_string()));
        assert!(collector.conclusions.contains(&"(and p q)".to_string()));
    }

    #[test]
    fn test_topological_order() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");

        let order = topological_order(&proof);
        assert_eq!(order.len(), 3);

        // The root (p3) should come last in topological order
        assert_eq!(order[order.len() - 1], p3);
    }

    #[test]
    fn test_find_all_paths() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let _p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");

        let paths = find_all_paths(&proof);
        assert_eq!(paths.len(), 2); // Two paths: one through p, one through q
    }

    #[test]
    fn test_traversal_orders() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let _p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");

        // Test all traversal orders don't crash
        let mut counter = NodeCounter::default();
        traverse(&proof, &mut counter, TraversalOrder::PreOrder);
        assert_eq!(counter.axioms, 2);

        let mut counter = NodeCounter::default();
        traverse(&proof, &mut counter, TraversalOrder::PostOrder);
        assert_eq!(counter.axioms, 2);

        let mut counter = NodeCounter::default();
        traverse(&proof, &mut counter, TraversalOrder::BreadthFirst);
        assert_eq!(counter.axioms, 2);
    }
}
