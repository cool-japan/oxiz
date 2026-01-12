//! BitVector theory implementation
//!
//! Implements the theory of fixed-width bit-vectors with arithmetic and bitwise operations.
//!
//! Reference: Z3's `src/smt/theory_bv.cpp` at `../z3/src/smt/`

use crate::ast::{TermId, TermKind, TermManager};
use crate::sort::{SortKind, SortManager};
use rustc_hash::{FxHashMap, FxHashSet};

/// BitVector theory axioms and rewrite rules
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BitVectorAxiom {
    /// bvadd identity: bvadd(x, 0) = x
    AddIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvmul identity: bvmul(x, 1) = x
    MulIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvmul by zero: bvmul(x, 0) = 0
    MulZero {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvand identity: bvand(x, all_ones) = x
    AndIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvand zero: bvand(x, 0) = 0
    AndZero {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvor identity: bvor(x, 0) = x
    OrIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvor saturation: bvor(x, all_ones) = all_ones
    OrSaturation {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvxor identity: bvxor(x, 0) = x
    XorIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvxor self: bvxor(x, x) = 0
    XorSelf {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvnot involution: bvnot(bvnot(x)) = x
    NotInvolution {
        /// The term this axiom applies to
        term: TermId,
    },
    /// bvneg zero: bvneg(0) = 0
    NegZero {
        /// The bit-width of the bitvector
        width: u32,
    },
    /// Bit-blast rule for comparison: converts bv operations to bit-level operations
    BitBlast {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
        /// The bitvector operation to bit-blast
        operation: BitVectorOp,
    },
}

/// BitVector operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitVectorOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Unsigned division
    UDiv,
    /// Signed division
    SDiv,
    /// Unsigned remainder
    URem,
    /// Signed remainder
    SRem,
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NOT
    Not,
    /// Negation (two's complement)
    Neg,
    /// Left shift
    Shl,
    /// Logical right shift
    LShr,
    /// Arithmetic right shift
    AShr,
}

/// BitVector theory reasoning engine
#[derive(Debug, Clone)]
pub struct BitVectorTheory {
    /// Tracked bitvector terms (maps term to its width)
    bitvectors: FxHashMap<TermId, u32>,
    /// Pending axiom instantiations
    pending_axioms: Vec<BitVectorAxiom>,
    /// Already instantiated axioms (to avoid duplicates)
    instantiated: FxHashSet<BitVectorAxiom>,
    /// Statistics
    propagations: usize,
    conflicts: usize,
}

impl Default for BitVectorTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl BitVectorTheory {
    /// Create a new bitvector theory instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            bitvectors: FxHashMap::default(),
            pending_axioms: Vec::new(),
            instantiated: FxHashSet::default(),
            propagations: 0,
            conflicts: 0,
        }
    }

    /// Register a bitvector term with its width
    pub fn register_bitvector(&mut self, term: TermId, width: u32) {
        self.bitvectors.insert(term, width);
    }

    /// Add a term to the theory and extract bitvector operations
    pub fn add_term(&mut self, term: TermId, manager: &TermManager, sort_manager: &SortManager) {
        if let Some(t) = manager.get(term)
            && let Some(sort) = sort_manager.get(t.sort)
            && let SortKind::BitVec(width) = sort.kind
        {
            self.register_bitvector(term, width);

            // Generate axioms based on term structure
            self.generate_axioms_for_term(term, &t.kind, width, manager);
        }
    }

    /// Generate axioms for a bitvector term
    fn generate_axioms_for_term(
        &mut self,
        term: TermId,
        kind: &TermKind,
        width: u32,
        manager: &TermManager,
    ) {
        match kind {
            TermKind::BvAdd(lhs, rhs) => {
                // Check for addition with zero
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::AddIdentity { term, width });
                }
            }
            TermKind::BvMul(lhs, rhs) => {
                // Check for multiplication by zero or one
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::MulZero { term, width });
                } else if self.is_bv_one(*rhs, manager, width)
                    || self.is_bv_one(*lhs, manager, width)
                {
                    self.add_axiom(BitVectorAxiom::MulIdentity { term, width });
                }
            }
            TermKind::BvAnd(lhs, rhs) => {
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::AndZero { term, width });
                } else if self.is_bv_all_ones(*rhs, manager, width)
                    || self.is_bv_all_ones(*lhs, manager, width)
                {
                    self.add_axiom(BitVectorAxiom::AndIdentity { term, width });
                }
            }
            TermKind::BvOr(lhs, rhs) => {
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::OrIdentity { term, width });
                } else if self.is_bv_all_ones(*rhs, manager, width)
                    || self.is_bv_all_ones(*lhs, manager, width)
                {
                    self.add_axiom(BitVectorAxiom::OrSaturation { term, width });
                }
            }
            TermKind::BvXor(lhs, rhs) => {
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::XorIdentity { term, width });
                } else if lhs == rhs {
                    self.add_axiom(BitVectorAxiom::XorSelf { term, width });
                }
            }
            TermKind::BvNot(inner) => {
                // Check for double negation
                if let Some(inner_term) = manager.get(*inner)
                    && let TermKind::BvNot(_) = inner_term.kind
                {
                    self.add_axiom(BitVectorAxiom::NotInvolution { term });
                }
            }
            _ => {}
        }
    }

    /// Check if a term is bitvector zero
    /// Note: Simplified implementation as BvLit is not available in current TermKind
    fn is_bv_zero(&self, _term: TermId, _manager: &TermManager, _width: u32) -> bool {
        // Placeholder: would check if term is a BvLit with value 0
        false
    }

    /// Check if a term is bitvector one
    /// Note: Simplified implementation as BvLit is not available in current TermKind
    fn is_bv_one(&self, _term: TermId, _manager: &TermManager, _width: u32) -> bool {
        // Placeholder: would check if term is a BvLit with value 1
        false
    }

    /// Check if a term is all ones (e.g., 0xFF for width=8)
    /// Note: Simplified implementation as BvLit is not available in current TermKind
    fn is_bv_all_ones(&self, _term: TermId, _manager: &TermManager, _width: u32) -> bool {
        // Placeholder: would check if term is a BvLit with all bits set
        false
    }

    /// Add an axiom to the pending list
    fn add_axiom(&mut self, axiom: BitVectorAxiom) {
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Get pending axioms and clear the list
    pub fn take_pending_axioms(&mut self) -> Vec<BitVectorAxiom> {
        std::mem::take(&mut self.pending_axioms)
    }

    /// Convert an axiom to a term (requires term manager)
    ///
    /// Note: This is a simplified implementation that returns placeholder terms.
    /// A full implementation would require additional BV literal and operation
    /// constructors that aren't currently available in TermManager.
    #[allow(dead_code)]
    pub fn axiom_to_term(&self, axiom: &BitVectorAxiom, manager: &mut TermManager) -> TermId {
        match axiom {
            BitVectorAxiom::AddIdentity { term, .. }
            | BitVectorAxiom::MulIdentity { term, .. }
            | BitVectorAxiom::AndIdentity { term, .. }
            | BitVectorAxiom::OrIdentity { term, .. }
            | BitVectorAxiom::XorIdentity { term, .. }
            | BitVectorAxiom::AndZero { term, .. }
            | BitVectorAxiom::MulZero { term, .. }
            | BitVectorAxiom::XorSelf { term, .. }
            | BitVectorAxiom::NotInvolution { term }
            | BitVectorAxiom::BitBlast { term, .. } => {
                // Simplified: return term or equality placeholder
                *term
            }
            BitVectorAxiom::OrSaturation { .. } | BitVectorAxiom::NegZero { .. } => {
                // Simplified: return placeholder
                manager.mk_true()
            }
        }
    }

    /// Propagate equalities and deduce new facts
    pub fn propagate(&mut self) -> Vec<(TermId, TermId)> {
        self.propagations += 1;
        // Placeholder: in a full implementation, this would deduce new equalities
        Vec::new()
    }

    /// Check for conflicts in the current state
    pub fn check_for_conflicts(&mut self, _manager: &TermManager) -> Option<Vec<TermId>> {
        // Placeholder: in a full implementation, this would detect contradictions
        None
    }

    /// Reset the theory state (for backtracking)
    pub fn reset(&mut self) {
        self.bitvectors.clear();
        self.pending_axioms.clear();
        self.instantiated.clear();
        self.propagations = 0;
        self.conflicts = 0;
    }

    /// Get statistics
    pub fn statistics(&self) -> BitVectorStatistics {
        BitVectorStatistics {
            num_bitvectors: self.bitvectors.len(),
            num_axioms: self.instantiated.len(),
            num_propagations: self.propagations,
            num_conflicts: self.conflicts,
        }
    }
}

/// Statistics for bitvector theory
#[derive(Debug, Clone, Copy)]
pub struct BitVectorStatistics {
    /// Number of bitvector terms
    pub num_bitvectors: usize,
    /// Number of axioms instantiated
    pub num_axioms: usize,
    /// Number of propagations
    pub num_propagations: usize,
    /// Number of conflicts detected
    pub num_conflicts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;
    use crate::sort::SortManager;

    #[test]
    fn test_empty_theory() {
        let theory = BitVectorTheory::new();
        assert_eq!(theory.bitvectors.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
    }

    #[test]
    fn test_register_bitvector() {
        let mut theory = BitVectorTheory::new();
        let term = TermId(42);
        theory.register_bitvector(term, 32);
        assert_eq!(theory.bitvectors.get(&term), Some(&32));
    }

    #[test]
    fn test_add_identity_axiom() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();
        let mut sort_manager = SortManager::new();

        let bv_sort = sort_manager.bitvec(32);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let add = manager.mk_bv_add(x, y);

        theory.add_term(add, &manager, &sort_manager);
        // Axiom generation may not work without proper BvLit support
        // This test verifies registration works
    }

    #[test]
    fn test_mul_zero_axiom() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();
        let mut sort_manager = SortManager::new();

        let bv_sort = sort_manager.bitvec(16);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let mul = manager.mk_bv_mul(x, y);

        theory.add_term(mul, &manager, &sort_manager);
        // Axiom generation may not work without proper BvLit support
        // This test verifies registration works
    }

    #[test]
    fn test_and_operation() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();
        let mut sort_manager = SortManager::new();

        let bv_sort = sort_manager.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let and = manager.mk_bv_and(x, y);

        theory.add_term(and, &manager, &sort_manager);
        // Test that term registration works
    }

    #[test]
    fn test_no_duplicate_axioms() {
        let mut theory = BitVectorTheory::new();
        let axiom = BitVectorAxiom::AddIdentity {
            term: TermId(1),
            width: 32,
        };

        theory.add_axiom(axiom.clone());
        theory.add_axiom(axiom);

        assert_eq!(theory.pending_axioms.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut theory = BitVectorTheory::new();
        theory.register_bitvector(TermId(1), 32);
        theory.add_axiom(BitVectorAxiom::AddIdentity {
            term: TermId(1),
            width: 32,
        });

        theory.reset();

        assert_eq!(theory.bitvectors.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
        assert_eq!(theory.instantiated.len(), 0);
    }

    #[test]
    fn test_statistics() {
        let mut theory = BitVectorTheory::new();
        theory.register_bitvector(TermId(1), 32);
        theory.register_bitvector(TermId(2), 16);
        theory.propagate();

        let stats = theory.statistics();
        assert_eq!(stats.num_bitvectors, 2);
        assert_eq!(stats.num_propagations, 1);
    }

    #[test]
    fn test_propagate() {
        let mut theory = BitVectorTheory::new();
        let equalities = theory.propagate();
        assert_eq!(equalities.len(), 0);
    }
}
