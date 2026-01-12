//! Proof Generation for Congruence Closure
//!
//! Generates proofs (explanations) for equalities and conflicts in the EUF theory.
//! This is crucial for:
//! - UNSAT core extraction
//! - Theory propagation explanations
//! - Interpolation

use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::fmt;

/// A proof step in the congruence closure
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofStep {
    /// Given equality (axiom): a = b
    Given {
        /// Left term
        left: TermId,
        /// Right term
        right: TermId,
        /// Reason (e.g., literal ID)
        reason: u32,
    },

    /// Reflexivity: a = a
    Refl {
        /// The term
        term: TermId,
    },

    /// Symmetry: a = b → b = a
    Symm {
        /// Original proof
        proof: Box<ProofStep>,
    },

    /// Transitivity: a = b ∧ b = c → a = c
    Trans {
        /// Left proof (a = b)
        left: Box<ProofStep>,
        /// Right proof (b = c)
        right: Box<ProofStep>,
    },

    /// Congruence: f(a₁,...,aₙ) = f(b₁,...,bₙ) if aᵢ = bᵢ for all i
    Cong {
        /// Function symbol
        func: TermId,
        /// Proofs for argument equalities
        arg_proofs: Vec<ProofStep>,
    },
}

impl ProofStep {
    /// Extract the left and right terms from a proof
    pub fn terms(&self) -> (TermId, TermId) {
        match self {
            ProofStep::Given { left, right, .. } => (*left, *right),
            ProofStep::Refl { term } => (*term, *term),
            ProofStep::Symm { proof } => {
                let (l, r) = proof.terms();
                (r, l)
            }
            ProofStep::Trans { left, right } => {
                let (l, _) = left.terms();
                let (_, r) = right.terms();
                (l, r)
            }
            ProofStep::Cong { func, .. } => (*func, *func), // Simplified
        }
    }

    /// Get the reasons (axioms) used in this proof
    pub fn reasons(&self) -> Vec<u32> {
        let mut reasons = Vec::new();
        self.collect_reasons(&mut reasons);
        reasons
    }

    fn collect_reasons(&self, reasons: &mut Vec<u32>) {
        match self {
            ProofStep::Given { reason, .. } => {
                reasons.push(*reason);
            }
            ProofStep::Refl { .. } => {}
            ProofStep::Symm { proof } => {
                proof.collect_reasons(reasons);
            }
            ProofStep::Trans { left, right } => {
                left.collect_reasons(reasons);
                right.collect_reasons(reasons);
            }
            ProofStep::Cong { arg_proofs, .. } => {
                for proof in arg_proofs {
                    proof.collect_reasons(reasons);
                }
            }
        }
    }

    /// Compute the size of the proof (number of steps)
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            ProofStep::Given { .. } | ProofStep::Refl { .. } => 1,
            ProofStep::Symm { proof } => 1 + proof.size(),
            ProofStep::Trans { left, right } => 1 + left.size() + right.size(),
            ProofStep::Cong { arg_proofs, .. } => {
                1 + arg_proofs.iter().map(ProofStep::size).sum::<usize>()
            }
        }
    }
}

impl fmt::Display for ProofStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProofStep::Given {
                left,
                right,
                reason,
            } => {
                write!(f, "Given({:?} = {:?}, reason={})", left, right, reason)
            }
            ProofStep::Refl { term } => write!(f, "Refl({:?})", term),
            ProofStep::Symm { proof } => write!(f, "Symm({})", proof),
            ProofStep::Trans { left, right } => write!(f, "Trans({}, {})", left, right),
            ProofStep::Cong { func, arg_proofs } => {
                write!(f, "Cong({:?}, [", func)?;
                for (i, proof) in arg_proofs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", proof)?;
                }
                write!(f, "])")
            }
        }
    }
}

/// Proof forest for efficient proof construction
///
/// Stores parent pointers with proof annotations.
#[derive(Debug)]
pub struct ProofForest {
    /// Parent of each term (for union-find)
    parent: FxHashMap<TermId, TermId>,
    /// Proof from term to its parent
    proof_to_parent: FxHashMap<TermId, ProofStep>,
    /// Rank for union-by-rank
    rank: FxHashMap<TermId, u32>,
}

impl Default for ProofForest {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofForest {
    /// Create a new proof forest
    #[must_use]
    pub fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            proof_to_parent: FxHashMap::default(),
            rank: FxHashMap::default(),
        }
    }

    /// Make a term (ensure it exists)
    pub fn make(&mut self, term: TermId) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(term) {
            e.insert(term);
            self.rank.insert(term, 0);
        }
    }

    /// Find the representative of a term
    #[must_use]
    pub fn find(&self, term: TermId) -> TermId {
        let mut current = term;
        while let Some(&parent) = self.parent.get(&current) {
            if parent == current {
                return current;
            }
            current = parent;
        }
        term
    }

    /// Union two terms with a proof
    pub fn union(&mut self, a: TermId, b: TermId, proof: ProofStep) {
        let root_a = self.find(a);
        let root_b = self.find(b);

        if root_a == root_b {
            return;
        }

        let rank_a = self.rank.get(&root_a).copied().unwrap_or(0);
        let rank_b = self.rank.get(&root_b).copied().unwrap_or(0);

        match rank_a.cmp(&rank_b) {
            std::cmp::Ordering::Less => {
                self.parent.insert(root_a, root_b);
                self.proof_to_parent.insert(root_a, proof);
            }
            std::cmp::Ordering::Greater => {
                self.parent.insert(root_b, root_a);
                self.proof_to_parent.insert(root_b, proof);
            }
            std::cmp::Ordering::Equal => {
                self.parent.insert(root_b, root_a);
                self.proof_to_parent.insert(root_b, proof);
                self.rank.insert(root_a, rank_a + 1);
            }
        }
    }

    /// Explain why two terms are equal
    pub fn explain(&self, a: TermId, b: TermId) -> Option<ProofStep> {
        if a == b {
            return Some(ProofStep::Refl { term: a });
        }

        // Find paths from a and b to their common root
        let path_a = self.path_to_root(a);
        let path_b = self.path_to_root(b);

        if path_a.is_empty() || path_b.is_empty() {
            return None;
        }

        // Check if they have the same root
        if path_a.last() != path_b.last() {
            return None;
        }

        // Build proof: a →* root ←* b
        let proof_a_to_root = self.build_proof_path(&path_a)?;
        let proof_b_to_root = self.build_proof_path(&path_b)?;

        // Combine: a = root ∧ b = root → a = b
        Some(ProofStep::Trans {
            left: Box::new(proof_a_to_root),
            right: Box::new(ProofStep::Symm {
                proof: Box::new(proof_b_to_root),
            }),
        })
    }

    /// Find path from term to root
    fn path_to_root(&self, term: TermId) -> Vec<TermId> {
        let mut path = vec![term];
        let mut current = term;

        while let Some(&parent) = self.parent.get(&current) {
            if parent == current {
                break;
            }
            path.push(parent);
            current = parent;
        }

        path
    }

    /// Build proof for a path
    fn build_proof_path(&self, path: &[TermId]) -> Option<ProofStep> {
        if path.len() == 1 {
            return Some(ProofStep::Refl { term: path[0] });
        }

        let mut current_proof = self.proof_to_parent.get(&path[0])?.clone();

        for &term in path.iter().take(path.len() - 1).skip(1) {
            let next_proof = self.proof_to_parent.get(&term)?.clone();
            current_proof = ProofStep::Trans {
                left: Box::new(current_proof),
                right: Box::new(next_proof),
            };
        }

        Some(current_proof)
    }

    /// Reset the forest
    pub fn reset(&mut self) {
        self.parent.clear();
        self.proof_to_parent.clear();
        self.rank.clear();
    }
}

/// Conflict explanation
///
/// When a conflict is detected (e.g., a ≠ b asserted but a = b derived),
/// we need to explain why a = b.
#[derive(Debug, Clone)]
pub struct Conflict {
    /// The two terms that are equal but asserted to be disequal
    pub left: TermId,
    /// The right term in the conflict
    pub right: TermId,
    /// Proof of equality
    pub proof: ProofStep,
    /// Disequality reason
    pub diseq_reason: u32,
}

impl Conflict {
    /// Create a new conflict
    #[must_use]
    pub fn new(left: TermId, right: TermId, proof: ProofStep, diseq_reason: u32) -> Self {
        Self {
            left,
            right,
            proof,
            diseq_reason,
        }
    }

    /// Get all reasons involved in this conflict
    pub fn reasons(&self) -> Vec<u32> {
        let mut reasons = self.proof.reasons();
        reasons.push(self.diseq_reason);
        reasons
    }

    /// Get the conflict clause
    pub fn clause(&self) -> SmallVec<[u32; 8]> {
        let reasons = self.reasons();
        SmallVec::from_vec(reasons)
    }
}

impl fmt::Display for Conflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Conflict: {:?} != {:?} (diseq={}) but {:?} = {:?} (proof={})",
            self.left, self.right, self.diseq_reason, self.left, self.right, self.proof
        )
    }
}

/// Proof manager for EUF solver
#[derive(Debug)]
pub struct ProofManager {
    /// Proof forest
    forest: ProofForest,
    /// Conflicts detected
    conflicts: Vec<Conflict>,
}

impl Default for ProofManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofManager {
    /// Create a new proof manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            forest: ProofForest::new(),
            conflicts: Vec::new(),
        }
    }

    /// Add a term
    pub fn add_term(&mut self, term: TermId) {
        self.forest.make(term);
    }

    /// Merge two terms with a proof
    pub fn merge(&mut self, a: TermId, b: TermId, proof: ProofStep) {
        self.forest.make(a);
        self.forest.make(b);
        self.forest.union(a, b, proof);
    }

    /// Check if two terms are equal
    #[must_use]
    pub fn are_equal(&self, a: TermId, b: TermId) -> bool {
        self.forest.find(a) == self.forest.find(b)
    }

    /// Explain why two terms are equal
    pub fn explain(&self, a: TermId, b: TermId) -> Option<ProofStep> {
        self.forest.explain(a, b)
    }

    /// Record a conflict
    pub fn add_conflict(&mut self, left: TermId, right: TermId, diseq_reason: u32) {
        if let Some(proof) = self.explain(left, right) {
            self.conflicts
                .push(Conflict::new(left, right, proof, diseq_reason));
        }
    }

    /// Get all conflicts
    #[must_use]
    pub fn conflicts(&self) -> &[Conflict] {
        &self.conflicts
    }

    /// Clear conflicts
    pub fn clear_conflicts(&mut self) {
        self.conflicts.clear();
    }

    /// Reset the manager
    pub fn reset(&mut self) {
        self.forest.reset();
        self.conflicts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_reflexivity() {
        let a = TermId::new(1);
        let proof = ProofStep::Refl { term: a };

        assert_eq!(proof.terms(), (a, a));
        assert_eq!(proof.size(), 1);
    }

    #[test]
    fn test_proof_symmetry() {
        let a = TermId::new(1);
        let b = TermId::new(2);

        let given = ProofStep::Given {
            left: a,
            right: b,
            reason: 0,
        };

        let symm = ProofStep::Symm {
            proof: Box::new(given),
        };

        assert_eq!(symm.terms(), (b, a));
    }

    #[test]
    fn test_proof_transitivity() {
        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        let ab = ProofStep::Given {
            left: a,
            right: b,
            reason: 0,
        };

        let bc = ProofStep::Given {
            left: b,
            right: c,
            reason: 1,
        };

        let trans = ProofStep::Trans {
            left: Box::new(ab),
            right: Box::new(bc),
        };

        assert_eq!(trans.terms(), (a, c));
        assert_eq!(trans.reasons(), vec![0, 1]);
    }

    #[test]
    fn test_proof_forest() {
        let mut forest = ProofForest::new();

        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        forest.make(a);
        forest.make(b);
        forest.make(c);

        let proof_ab = ProofStep::Given {
            left: a,
            right: b,
            reason: 0,
        };

        let proof_bc = ProofStep::Given {
            left: b,
            right: c,
            reason: 1,
        };

        forest.union(a, b, proof_ab);
        forest.union(b, c, proof_bc);

        assert_eq!(forest.find(a), forest.find(c));
    }

    #[test]
    fn test_proof_manager() {
        let mut manager = ProofManager::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        manager.add_term(a);
        manager.add_term(b);

        assert!(!manager.are_equal(a, b));

        let proof = ProofStep::Given {
            left: a,
            right: b,
            reason: 0,
        };

        manager.merge(a, b, proof);

        assert!(manager.are_equal(a, b));
    }
}
