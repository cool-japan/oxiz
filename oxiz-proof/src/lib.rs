//! # oxiz-proof
//!
//! Proof generation and checking for the OxiZ SMT solver.
//!
//! This crate provides machine-checkable proof output in various formats,
//! enabling verification of solver results by external proof checkers.
//!
//! ## Beyond Z3
//!
//! While Z3 supports generic proofs, OxiZ aims to generate **machine-checkable
//! proofs** (Alethe, LFSC) by default, making it more suitable for certified
//! verification workflows.
//!
//! ## Supported Formats
//!
//! - **DRAT**: For SAT core proofs
//! - **Alethe**: SMT-LIB proof format
//! - **Carcara**: Proof checker compatibility for Alethe format
//! - **LFSC**: Logical Framework with Side Conditions
//! - **Coq**: Export to Coq proof assistant
//! - **Lean**: Export to Lean theorem prover (Lean 3 & 4)
//! - **Isabelle**: Export to Isabelle/HOL
//! - **Theory**: Theory-specific proof steps
//! - **Checker**: Proof validation infrastructure
//! - **PCC**: Proof-carrying code generation
//! - **Merge**: Proof merging and slicing utilities
//! - **Diff**: Proof comparison and similarity metrics
//! - **Normalize**: Proof normalization for canonical representation

pub mod alethe;
pub mod carcara;
pub mod checker;
pub mod compress;
pub mod conversion;
pub mod coq;
pub mod craig;
pub mod diff;
pub mod drat;
pub mod explanation;
pub mod fingerprint;
pub mod format;
pub mod heuristic;
pub mod incremental;
pub mod interpolant;
pub mod isabelle;
pub mod lazy;
pub mod lean;
pub mod lfsc;
pub mod merge;
pub mod metadata;
pub mod mmap;
pub mod normalize;
pub mod parallel;
pub mod pattern;
pub mod pcc;
pub mod proof;
pub mod rules;
pub mod sat_integration;
pub mod simplify;
pub mod streaming;
pub mod template;
pub mod theory;
pub mod unsat_core;
pub mod validation;
pub mod visualization;

// Internal modules
mod builder;
mod cnf;
mod convert;
mod premise;
mod quantifier;
mod resolution;

// Public modules with useful analysis and utility tools
pub mod stats;
pub mod traversal;

// Re-exports
pub use alethe::{AletheProof, AletheProofProducer, AletheRule, AletheStep};
pub use carcara::{CarcaraProof, to_carcara_format, validate_for_carcara};
pub use checker::{CheckError, CheckResult, Checkable, CheckerConfig, ErrorSeverity, ProofChecker};
pub use compress::{
    CompressionConfig, CompressionResult, ProofCompressor, get_dependency_cone, trim_to_conclusion,
};
pub use conversion::{ConversionError, ConversionResult, FormatConverter};
pub use coq::{CoqExporter, export_theory_to_coq, export_to_coq};
pub use diff::{ProofDiff, ProofSimilarity, compute_similarity, diff_proofs};
pub use drat::{DratProof, DratProofProducer, DratStep};
pub use explanation::{ExplainedStep, ProofComplexity, ProofExplainer, Verbosity};
pub use fingerprint::{FingerprintDatabase, FingerprintGenerator, ProofFingerprint, SizeFeatures};
pub use format::ProofFormat;
pub use heuristic::{HeuristicType, ProofHeuristic, StrategyLearner};
pub use incremental::{IncrementalProofBuilder, IncrementalStats, ProofRecorder};
pub use interpolant::{Color, Interpolant, InterpolantExtractor, Partition};
pub use isabelle::{IsabelleExporter, export_theory_to_isabelle, export_to_isabelle};
pub use lazy::{LazyDependencyResolver, LazyNode, LazyProof, LazyStats};
pub use lean::{LeanExporter, export_theory_to_lean, export_to_lean, export_to_lean3};
pub use lfsc::{LfscDecl, LfscProof, LfscProofProducer, LfscSort, LfscTerm};
pub use merge::{merge_proofs, slice_proof, slice_proof_multi};
pub use metadata::{Difficulty, Priority, ProofMetadata, Strategy};
pub use mmap::{MmapConfig, MmapProof, MmapProofStorage};
pub use normalize::{canonicalize_conclusions, normalize_proof};
pub use parallel::{ParallelCheckResult, ParallelConfig, ParallelProcessor, ParallelStatsComputer};
pub use pattern::{LemmaPattern, PatternExtractor, PatternStructure};
pub use pcc::{CodeLocation, PccBuilder, ProofCarryingCode, SafetyProperty, VerificationCondition};
pub use proof::{Proof, ProofNode, ProofNodeId, ProofStats, ProofStep};
pub use rules::{
    Clause, CnfValidator, Literal, ResolutionValidator, RuleValidation, TheoryLemmaValidator,
    UnitPropagationValidator,
};
#[cfg(feature = "sat-integration")]
pub use sat_integration::{
    ProofRecordingSolver, drat_clause_to_sat, drat_lit_to_sat, sat_clause_to_drat, sat_lit_to_drat,
};
pub use simplify::{ProofSimplifier, SimplificationConfig, SimplificationStats, simplify_proof};
pub use stats::{DetailedProofStats, ProofQuality, TheoryProofStats};
pub use streaming::{
    ProofChunk, ProofChunkIterator, ProofStreamer, StreamConfig, StreamingProofBuilder,
};
pub use template::{ProofTemplate, TemplateIdentifier, TemplateStep};
pub use theory::{
    ArithProofRecorder, ArrayProofRecorder, EufProofRecorder, ProofTerm, TheoryProof,
    TheoryProofProducer, TheoryRule, TheoryStep, TheoryStepId,
};
pub use unsat_core::{UnsatCore, extract_minimal_unsat_core, extract_unsat_core, get_core_labels};
pub use validation::{FormatValidator, ValidationError, ValidationResult};
pub use visualization::{ProofVisualizer, VisualizationFormat};

// Craig interpolation
pub use craig::{
    ArrayInterpolator, CraigInterpolator, EufInterpolator, InterpolantColor, InterpolantPartition,
    InterpolantTerm, InterpolationAlgorithm, InterpolationConfig, InterpolationError,
    InterpolationStats, LiaInterpolator, SequenceInterpolator, Symbol, TheoryInterpolator,
    TreeInterpolator, TreeNode,
};
