//! SAT Preprocessing Techniques - Advanced Extensions.
//!
//! This module extends the basic preprocessing functionality with:
//! - Advanced variable elimination (BVE with cost analysis)
//! - Gate extraction and recognition
//! - Bounded variable addition

pub mod advanced;
pub mod gate_extraction;
pub mod variable_elimination;

pub use advanced::{AdvancedPreprocessor, Clause, PreprocessingConfig, PreprocessingStats};
pub use gate_extraction::{Circuit, Gate, GateConfig, GateExtractor, GateStats, GateType};
pub use variable_elimination::{
    AsymmetricEliminator, BoundedVariableAddition, EliminationConfig, EliminationStats,
    VariableEliminator,
};
