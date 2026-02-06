//! Array theory solver
//!
//! Implements the theory of arrays with read (select) and write (store) operations.
//! This module provides a comprehensive array theory implementation including:
//!
//! ## Core Theory
//! - Basic array axioms:
//!   - Read-over-write same: select(store(a, i, v), i) = v
//!   - Read-over-write different: i ≠ j → select(store(a, i, v), j) = select(a, j)
//!   - Extensionality: (∀i. select(a, i) = select(b, i)) → a = b
//!
//! ## Advanced Features
//! - Multi-dimensional array support with N-dimensional indexing
//! - Quantifier elimination for array formulas
//! - Decidable array property fragments (Bradley-Manna-Sipma)
//! - Cardinality constraints and finite model reasoning
//! - Array abstraction and refinement (CEGAR)
//! - Interpolation for array formulas
//!
//! ## Modules
//! - `solver`: Core array solver with basic axioms
//! - `multi_dimensional`: Multi-dimensional array support
//! - `quantifier_elim`: Array quantifier elimination
//! - `property_fragments`: Decidable array fragments
//! - `cardinality`: Cardinality constraints and reasoning
//! - `abstraction`: Array abstraction and refinement
//! - `interpolation`: Interpolation for arrays

#![allow(missing_docs)]

mod abstraction;
mod cardinality;
mod interpolation;
mod multi_dimensional;
mod property_fragments;
mod quantifier_elim;
mod solver;

// Re-export main solver
pub use solver::ArraySolver;

// Re-export multi-dimensional support
pub use multi_dimensional::{
    ArrayDimensions, DimensionIterator, FlatteningStrategy, IndexDifferenceAnalyzer,
    IndexSequenceRelation, MultiDimArrayManager, MultiDimLemma, MultiDimLemmaGenerator,
    MultiDimSelect, MultiDimStore, SortId,
};

// Re-export quantifier elimination
pub use quantifier_elim::{
    ArrayFormulaSimplifier, ArrayFragment, ArrayFragmentAnalyzer, ArrayQuantifierEliminator,
    ArrayQuantifierPattern, EliminationResult, IndexTermCollector, InstantiationHeuristic,
    QuantifiedVar, QuantifierEliminationContext, QuantifierInstantiationStrategy, QuantifierType,
};

// Re-export property fragments
pub use property_fragments::{
    ArithOp, ArrayTerm, BMSFragment, DecidabilityChecker, DecidabilityReport, FragmentClass,
    IndexTerm, PropertyFragmentClassifier, QuantifierComplexity, QuantifierComplexityAnalyzer,
    UpdatePattern, UpdatePatternAnalyzer, UpdatePatternType, ValueTerm,
};

// Re-export cardinality
pub use cardinality::{
    ArrayCardinalityConstraint, ArrayOperation, BoundType, CardinalityBound,
    CardinalityConstraintManager, CardinalityImplication, CardinalityReasoner,
    FiniteModelConstraint, PigeonholeConstraint, SymmetryPattern,
};

// Re-export abstraction
pub use abstraction::{
    AbstractArrayState, AbstractCheckResult, AbstractDomain, AbstractIndex, AbstractionPredicate,
    ArrayAbstractOp, ArrayAbstractionEngine, ArrayInvariant, ArrayInvariantGenerator, ArraySummary,
    CEGARLoop, CEGARResult, ConditionType, InvariantType, ParityDomain, RefinementStep, SignDomain,
    SummaryCondition,
};

// Re-export interpolation
pub use interpolation::{
    ArrayInterpolationEngine, Interpolant, InterpolationProblem, InterpolationStrategy,
    InterpolationStrengthAnalyzer, InterpolationVerifier, SequenceInterpolationEngine, StepResult,
    StrengthMetrics, TreeInterpolationEngine, TreeInterpolationProblem, TreeNode,
    VerificationResult, VerificationStep,
};
