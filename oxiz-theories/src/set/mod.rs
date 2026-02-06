//! Set Theory Solver for SMT
//!
//! This module implements a comprehensive set theory solver for SMT,
//! supporting operations on finite and infinite sets including:
//! - Set union, intersection, difference, complement
//! - Membership constraints (element ∈ Set)
//! - Subset relations (Set1 ⊆ Set2)
//! - Cardinality constraints (|Set| = k, |Set| ≤ k, etc.)
//! - Powerset operations
//! - Set comprehension and enumeration
//!
//! # Architecture
//!
//! The solver uses a combination of techniques:
//! - **BDD-based representation** for efficient set operations
//! - **Constraint propagation** for membership and cardinality
//! - **Nelson-Oppen** integration for theory combination
//! - **Finite domain reasoning** for bounded cardinality sets
//!
//! # Example
//!
//! ```ignore
//! use oxiz_theories::set::{SetSolver, SetExpr};
//!
//! let mut solver = SetSolver::new();
//!
//! // Create set variables
//! let s1 = solver.new_set_var("S1");
//! let s2 = solver.new_set_var("S2");
//!
//! // Assert: S1 ∪ S2 = ∅
//! solver.assert_union_empty(s1, s2);
//!
//! // Check satisfiability
//! let result = solver.check();
//! ```
//!
//! # References
//!
//! Based on algorithms from:
//! - Z3's theory_set.cpp
//! - "Decision Procedures for Set Constraints" (Zarba, 2004)
//! - "Complete Decision Procedures for Satisfiability Problems" (Cantone et al.)

#![allow(missing_docs)]

mod cardinality;
mod finite_sets;
mod membership;
mod operations;
mod powerset;
mod solver;
mod subset;

// Re-export public types
pub use cardinality::{
    CardConstraint, CardConstraintKind, CardDomain, CardPropagator, CardResult, CardStats,
};
pub use finite_sets::{
    EnumSet, FiniteSetEnumerator, SetElement, SetEnumConfig, SetEnumResult, SetEnumStats,
};
pub use membership::{
    MemberConstraint, MemberDomain, MemberPropagator, MemberResult, MemberStats, MemberVar,
};
pub use operations::{
    SetBinOp, SetComplement, SetDifference, SetIntersection, SetOp, SetOpBuilder, SetOpResult,
    SetOpStats, SetUnion,
};
pub use powerset::{
    PowersetBuilder, PowersetConstraint, PowersetIter, PowersetResult, PowersetStats,
};
pub use solver::{
    SetConfig, SetConstraint, SetExpr, SetResult, SetSolver, SetStats, SetVar, SetVarId,
};
pub use subset::{
    SubsetConstraint, SubsetDomain, SubsetGraph, SubsetPropagator, SubsetResult, SubsetStats,
};

/// Set sort kinds
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SetSort {
    /// Set of integers
    IntSet,
    /// Set of reals
    RealSet,
    /// Set of bitvectors
    BvSet,
    /// Set of elements from a given sort
    ElementSet(u32),
    /// Set of sets (for nested set reasoning)
    SetSet(Box<SetSort>),
}

/// Set literals for assignment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetLiteral {
    /// x ∈ S
    Member {
        element: u32,
        set: SetVarId,
        sign: bool,
    },
    /// S1 ⊆ S2
    Subset {
        lhs: SetVarId,
        rhs: SetVarId,
        sign: bool,
    },
    /// S1 = S2
    Equal {
        lhs: SetVarId,
        rhs: SetVarId,
        sign: bool,
    },
    /// |S| op k (where op is =, ≤, ≥, <, >)
    Cardinality {
        set: SetVarId,
        op: CardConstraintKind,
        bound: i64,
    },
}

/// Set theory conflict explanation
#[derive(Debug, Clone)]
pub struct SetConflict {
    /// Conflicting literals
    pub literals: Vec<SetLiteral>,
    /// Explanation message
    pub reason: String,
    /// Proof steps (for UNSAT core)
    pub proof_steps: Vec<SetProofStep>,
}

/// Set proof step for explanation
#[derive(Debug, Clone)]
pub enum SetProofStep {
    /// Assumption from assertion
    Assume(SetLiteral),
    /// Propagation by membership
    MemberProp {
        element: u32,
        from: SetVarId,
        to: SetVarId,
    },
    /// Propagation by subset
    SubsetProp {
        from: SetVarId,
        mid: SetVarId,
        to: SetVarId,
    },
    /// Cardinality bound conflict
    CardConflict {
        set: SetVarId,
        lower: i64,
        upper: i64,
    },
    /// Empty set conflict
    EmptyConflict { set: SetVarId },
}

impl SetSort {
    /// Check if this is a nested set sort
    pub fn is_nested(&self) -> bool {
        matches!(self, SetSort::SetSet(_))
    }

    /// Get the depth of nesting
    pub fn nesting_depth(&self) -> usize {
        match self {
            SetSort::SetSet(inner) => 1 + inner.nesting_depth(),
            _ => 0,
        }
    }

    /// Get the element sort for this set sort
    pub fn element_sort(&self) -> Option<&SetSort> {
        match self {
            SetSort::SetSet(inner) => Some(inner),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_sort_nesting() {
        let int_set = SetSort::IntSet;
        assert_eq!(int_set.nesting_depth(), 0);
        assert!(!int_set.is_nested());

        let set_of_sets = SetSort::SetSet(Box::new(SetSort::IntSet));
        assert_eq!(set_of_sets.nesting_depth(), 1);
        assert!(set_of_sets.is_nested());

        let deep_nested = SetSort::SetSet(Box::new(SetSort::SetSet(Box::new(SetSort::RealSet))));
        assert_eq!(deep_nested.nesting_depth(), 2);
        assert!(deep_nested.is_nested());
    }

    #[test]
    fn test_set_sort_element() {
        let set_of_sets = SetSort::SetSet(Box::new(SetSort::IntSet));
        match set_of_sets.element_sort() {
            Some(SetSort::IntSet) => {}
            _ => panic!("Expected IntSet element sort"),
        }
    }

    #[test]
    fn test_set_literal_equality() {
        let lit1 = SetLiteral::Member {
            element: 42,
            set: SetVarId(0),
            sign: true,
        };
        let lit2 = SetLiteral::Member {
            element: 42,
            set: SetVarId(0),
            sign: true,
        };
        assert_eq!(lit1, lit2);

        let lit3 = SetLiteral::Member {
            element: 42,
            set: SetVarId(0),
            sign: false,
        };
        assert_ne!(lit1, lit3);
    }
}
