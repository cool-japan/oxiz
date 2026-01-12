//! Abstract Syntax Tree for OxiZ
//!
//! Terms are arena-allocated and referenced via `TermId` indices.
//! This provides cache-friendly access and cheap copies.

pub mod congruence;
pub mod context;
pub mod egraph;
pub mod interpolation;
mod manager;
pub mod model;
pub mod normal_forms;
pub mod parallel;
pub mod pattern;
pub mod pool;
pub mod proof;
pub mod proof_transform;
pub mod rewriting;
pub mod simd;
mod term;
pub mod traversal;
pub mod utils;
pub mod validation;

pub use congruence::{CongruenceClosure, Explanation};
pub use context::{Context, NamedAssertion, NamedContext};
pub use egraph::{EClass, EClassId, EGraph, EGraphStats, ENode, ENodeKind};
pub use interpolation::{InterpolationContext, InterpolationStats};
pub use manager::{GCStatistics, SubstitutionBuilder, TermManager};
pub use model::{FunctionInterpretation, Model, ModelValue};
pub use normal_forms::{
    eliminate_universal_quantifiers, extract_cnf_clauses, is_cnf, is_dnf, is_nnf, simplify_boolean,
    skolemize, to_cnf, to_dnf, to_nnf,
};
pub use parallel::{
    ParallelStats, parallel_all, parallel_all_distinct, parallel_any, parallel_count,
    parallel_filter, parallel_find, parallel_map, parallel_partition, parallel_reduce,
};
pub use pattern::{Pattern, PatternRewriteRule, Substitution, instantiate_pattern, match_pattern};
pub use pool::{MemoryStats, TermPool};
pub use proof::{Proof, ProofId, ProofNode, ProofRule, ProofStatistics};
pub use proof_transform::{
    CertificateFormat, NaturalDeductionNode, NaturalDeductionProof, NaturalDeductionRule,
    generate_certificate, resolution_to_natural_deduction,
};
pub use rewriting::{
    ArithmeticConstantFoldingRule, ArithmeticIdentityRule, BooleanAbsorptionRule,
    BooleanPropagationRule, ComparisonNormalizationRule, DeMorganRule, DistributivityRule,
    DoubleNegationRule, EqualitySimplificationRule, ImplicationEliminationRule,
    IteSimplificationRule, RewriteRule, RewriteRuleSet, standard_rules,
};
pub use simd::{simd_all_equal, simd_compare_termids, simd_contains, simd_hash_termids};
pub use term::{InstPattern, RoundingMode, Term, TermId, TermKind};
pub use traversal::{
    TermVisitor, TraversalError, VisitorAction, collect_free_vars, collect_subterms, compute_depth,
    contains_term, count_nodes, get_children, map_terms, traverse,
};
pub use utils::{
    TermStatistics, alpha_equivalent, collect_unique_subterms, compute_statistics,
    count_operations, find_terms, flatten_associative, is_ground, max_term_id, structural_hash,
    structurally_equal, term_complexity,
};
pub use validation::{CachedEvaluator, eval_term, validate_assertion, validate_model};
