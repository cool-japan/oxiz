//! Cylindrical Algebraic Decomposition (CAD).
//!
//! CAD is a fundamental algorithm for quantifier elimination over real closed fields.
//! It decomposes R^n into cylindrically arranged cells where polynomials have constant signs.
//!
//! ## Algorithm Phases
//!
//! 1. **Projection**: Eliminate variables recursively to reach R^1
//! 2. **Base**: Decompose R^1 based on projection polynomials
//! 3. **Lifting**: Lift decomposition back up to R^n
//!
//! ## References
//!
//! - Collins: "Quantifier Elimination for Real Closed Fields by CAD" (1975)
//! - Z3's `qe/qe_arith_plugin.cpp`

pub mod base;
pub mod cell_decomposition;
pub mod lifting;
pub mod projection;
pub mod sample;

pub use base::{BaseCase, BaseConfig, BaseStats};
pub use cell_decomposition::{
    CellDecomposition, CellId, DecomposedCell, DecompositionConfig, DecompositionError,
    DecompositionStats, Level,
};
pub use lifting::{Cell, CellType, LiftingConfig, LiftingEngine, LiftingStats, SampleStrategy};
pub use projection::{
    ProjectionConfig, ProjectionEngine, ProjectionLevel, ProjectionOperator, ProjectionStats,
};
pub use sample::{
    CellType as SampleCellType, SampleConfig, SamplePoint, SampleSelector, SampleStats,
    SampleStrategy as SelectorStrategy,
};
