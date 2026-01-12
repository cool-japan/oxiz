//! FloatingPoint theory implementation
//!
//! Implements the theory of IEEE 754 floating-point arithmetic.
//!
//! Reference: Z3's `src/smt/theory_fpa.cpp` at `../z3/src/smt/`

use crate::ast::{RoundingMode, TermId, TermKind, TermManager};
use crate::sort::{SortId, SortKind, SortManager};
use rustc_hash::{FxHashMap, FxHashSet};

/// FloatingPoint theory axioms and properties
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FloatingPointAxiom {
    /// Addition identity: fp.add(rm, x, +0) = x
    AddIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The rounding mode used in the operation
        rounding_mode: RoundingMode,
    },
    /// Multiplication identity: fp.mul(rm, x, 1.0) = x
    MulIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The rounding mode used in the operation
        rounding_mode: RoundingMode,
    },
    /// Multiplication by zero: fp.mul(rm, x, +0) = +0 (for normal x)
    MulZero {
        /// The term this axiom applies to
        term: TermId,
        /// The rounding mode used in the operation
        rounding_mode: RoundingMode,
    },
    /// Negation involution: fp.neg(fp.neg(x)) = x
    NegInvolution {
        /// The term this axiom applies to
        term: TermId,
    },
    /// Absolute value: fp.abs(x) >= +0
    AbsNonNegative {
        /// The term this axiom applies to
        term: TermId,
    },
    /// NaN propagation: any operation with NaN produces NaN
    NaNPropagation {
        /// The term this axiom applies to
        term: TermId,
        /// The floating-point operation that propagates NaN
        operation: FloatingPointOp,
    },
    /// Infinity properties
    InfinityAxiom {
        /// The term this axiom applies to
        term: TermId,
        /// The specific infinity property being asserted
        property: InfinityProperty,
    },
    /// Comparison with NaN always false (except fp.isNaN)
    CompareNaN {
        /// The left-hand side of the comparison
        lhs: TermId,
        /// The right-hand side of the comparison
        rhs: TermId,
    },
}

/// FloatingPoint operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatingPointOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Remainder
    Rem,
    /// Fused multiply-add
    Fma,
    /// Square root
    Sqrt,
    /// Round to integral
    RoundToIntegral,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Absolute value
    Abs,
    /// Negation
    Neg,
}

/// Infinity properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InfinityProperty {
    /// +∞ + +∞ = +∞
    AddPosInf,
    /// +∞ * x = +∞ (for x > 0)
    MulPosInf,
    /// +∞ > x (for finite x)
    CompareGreater,
}

/// FloatingPoint theory reasoning engine
#[derive(Debug, Clone)]
pub struct FloatingPointTheory {
    /// Tracked floating-point terms (maps term to its sort)
    floats: FxHashMap<TermId, SortId>,
    /// Special values (NaN, +Inf, -Inf, +0, -0)
    special_values: FxHashMap<SortId, SpecialValues>,
    /// Pending axiom instantiations
    pending_axioms: Vec<FloatingPointAxiom>,
    /// Already instantiated axioms (to avoid duplicates)
    instantiated: FxHashSet<FloatingPointAxiom>,
    /// Statistics
    propagations: usize,
    conflicts: usize,
}

/// Special floating-point values for a given sort
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SpecialValues {
    /// Positive zero
    pos_zero: Option<TermId>,
    /// Negative zero
    neg_zero: Option<TermId>,
    /// Positive infinity
    pos_inf: Option<TermId>,
    /// Negative infinity
    neg_inf: Option<TermId>,
    /// Not-a-Number
    nan: Option<TermId>,
}

impl Default for FloatingPointTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl FloatingPointTheory {
    /// Create a new floating-point theory instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            floats: FxHashMap::default(),
            special_values: FxHashMap::default(),
            pending_axioms: Vec::new(),
            instantiated: FxHashSet::default(),
            propagations: 0,
            conflicts: 0,
        }
    }

    /// Register a floating-point term
    pub fn register_float(&mut self, term: TermId, sort: SortId) {
        self.floats.insert(term, sort);
    }

    /// Add a term to the theory and extract floating-point operations
    pub fn add_term(&mut self, term: TermId, manager: &TermManager, sort_manager: &SortManager) {
        if let Some(t) = manager.get(term)
            && let Some(sort) = sort_manager.get(t.sort)
            && matches!(sort.kind, SortKind::FloatingPoint { .. })
        {
            self.register_float(term, t.sort);

            // Generate axioms based on term structure
            self.generate_axioms_for_term(term, &t.kind, manager);
        }
    }

    /// Generate axioms for a floating-point term
    fn generate_axioms_for_term(&mut self, term: TermId, kind: &TermKind, manager: &TermManager) {
        match kind {
            TermKind::FpAdd(rm, lhs, rhs) => {
                // Check for addition with zero
                if self.is_fp_pos_zero(*rhs, manager) || self.is_fp_pos_zero(*lhs, manager) {
                    self.add_axiom(FloatingPointAxiom::AddIdentity {
                        term,
                        rounding_mode: *rm,
                    });
                }
                // Check for NaN
                if self.is_fp_nan(*lhs, manager) || self.is_fp_nan(*rhs, manager) {
                    self.add_axiom(FloatingPointAxiom::NaNPropagation {
                        term,
                        operation: FloatingPointOp::Add,
                    });
                }
            }
            TermKind::FpMul(rm, lhs, rhs) => {
                // Check for multiplication by zero or one
                if self.is_fp_pos_zero(*rhs, manager) || self.is_fp_pos_zero(*lhs, manager) {
                    self.add_axiom(FloatingPointAxiom::MulZero {
                        term,
                        rounding_mode: *rm,
                    });
                }
                // Check for NaN
                if self.is_fp_nan(*lhs, manager) || self.is_fp_nan(*rhs, manager) {
                    self.add_axiom(FloatingPointAxiom::NaNPropagation {
                        term,
                        operation: FloatingPointOp::Mul,
                    });
                }
            }
            TermKind::FpNeg(inner) => {
                // Check for double negation
                if let Some(inner_term) = manager.get(*inner)
                    && let TermKind::FpNeg(_) = inner_term.kind
                {
                    self.add_axiom(FloatingPointAxiom::NegInvolution { term });
                }
            }
            TermKind::FpAbs(_) => {
                self.add_axiom(FloatingPointAxiom::AbsNonNegative { term });
            }
            _ => {}
        }
    }

    /// Check if a term is positive zero
    fn is_fp_pos_zero(&self, _term: TermId, _manager: &TermManager) -> bool {
        // Placeholder: would check if term is a FpLit with value +0.0
        false
    }

    /// Check if a term is NaN
    fn is_fp_nan(&self, _term: TermId, _manager: &TermManager) -> bool {
        // Placeholder: would check if term is a FpLit with NaN value
        false
    }

    /// Add an axiom to the pending list
    fn add_axiom(&mut self, axiom: FloatingPointAxiom) {
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Get pending axioms and clear the list
    pub fn take_pending_axioms(&mut self) -> Vec<FloatingPointAxiom> {
        std::mem::take(&mut self.pending_axioms)
    }

    /// Convert an axiom to a term (requires term manager)
    pub fn axiom_to_term(&self, axiom: &FloatingPointAxiom, manager: &TermManager) -> TermId {
        match axiom {
            FloatingPointAxiom::AddIdentity { term, .. } => *term,
            FloatingPointAxiom::MulIdentity { term, .. } => *term,
            FloatingPointAxiom::MulZero { term, .. } => *term,
            FloatingPointAxiom::NegInvolution { term } => *term,
            FloatingPointAxiom::AbsNonNegative { term } => {
                // fp.abs(x) >= +0 becomes a comparison
                // Placeholder: would create actual comparison term
                *term
            }
            FloatingPointAxiom::NaNPropagation { term, .. } => {
                // Any op with NaN yields NaN
                *term
            }
            FloatingPointAxiom::InfinityAxiom { term, .. } => *term,
            FloatingPointAxiom::CompareNaN { .. } => {
                // Comparison with NaN is false
                manager.mk_false()
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
        self.floats.clear();
        self.special_values.clear();
        self.pending_axioms.clear();
        self.instantiated.clear();
        self.propagations = 0;
        self.conflicts = 0;
    }

    /// Get statistics
    pub fn statistics(&self) -> FloatingPointStatistics {
        FloatingPointStatistics {
            num_floats: self.floats.len(),
            num_axioms: self.instantiated.len(),
            num_propagations: self.propagations,
            num_conflicts: self.conflicts,
        }
    }
}

/// Statistics for floating-point theory
#[derive(Debug, Clone, Copy)]
pub struct FloatingPointStatistics {
    /// Number of floating-point terms
    pub num_floats: usize,
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
        let theory = FloatingPointTheory::new();
        assert_eq!(theory.floats.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
    }

    #[test]
    fn test_register_float() {
        let mut theory = FloatingPointTheory::new();
        let term = TermId(42);
        let sort = SortId(1);
        theory.register_float(term, sort);
        assert_eq!(theory.floats.get(&term), Some(&sort));
    }

    #[test]
    fn test_no_duplicate_axioms() {
        let mut theory = FloatingPointTheory::new();
        let axiom = FloatingPointAxiom::NegInvolution { term: TermId(1) };

        theory.add_axiom(axiom.clone());
        theory.add_axiom(axiom);

        assert_eq!(theory.pending_axioms.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut theory = FloatingPointTheory::new();
        theory.register_float(TermId(1), SortId(1));
        theory.add_axiom(FloatingPointAxiom::NegInvolution { term: TermId(1) });

        theory.reset();

        assert_eq!(theory.floats.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
        assert_eq!(theory.instantiated.len(), 0);
    }

    #[test]
    fn test_statistics() {
        let mut theory = FloatingPointTheory::new();
        theory.register_float(TermId(1), SortId(1));
        theory.register_float(TermId(2), SortId(1));
        theory.propagate();

        let stats = theory.statistics();
        assert_eq!(stats.num_floats, 2);
        assert_eq!(stats.num_propagations, 1);
    }

    #[test]
    fn test_propagate() {
        let mut theory = FloatingPointTheory::new();
        let equalities = theory.propagate();
        assert_eq!(equalities.len(), 0);
    }

    #[test]
    fn test_rounding_modes() {
        use RoundingMode::*;
        let modes = [RNE, RNA, RTP, RTN, RTZ];
        assert_eq!(modes.len(), 5);
    }

    #[test]
    fn test_axiom_to_term() {
        let theory = FloatingPointTheory::new();
        let manager = TermManager::new();
        let axiom = FloatingPointAxiom::NegInvolution { term: TermId(42) };
        let term = theory.axiom_to_term(&axiom, &manager);
        assert_eq!(term, TermId(42));
    }

    #[test]
    fn test_add_term_registration() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();
        let mut sort_manager = SortManager::new();

        // Create a Float32 sort (exponent=8, significand=24)
        let fp_sort = sort_manager.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);

        theory.add_term(x, &manager, &sort_manager);
        assert!(theory.floats.contains_key(&x));
    }
}
