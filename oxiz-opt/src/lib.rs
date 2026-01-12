//! # oxiz-opt
//!
//! Optimization engine for OxiZ: MaxSMT and Optimization Modulo Theories (OMT).
//!
//! This crate provides optimization capabilities on top of the SMT solver,
//! including weighted MaxSAT, MaxSMT, and objective optimization.
//!
//! ## Features
//!
//! - **MaxSAT**: Core-guided algorithms (Fu-Malik, OLL, MSU3, WMax, RC2)
//! - **MaxSMT**: Optimization of soft SMT constraints
//! - **OMT**: Objective function optimization (minimize/maximize)
//! - **Pareto**: Multi-objective optimization with Pareto front enumeration
//! - **Hybrid**: Stochastic Local Search combined with exact methods
//! - **Portfolio**: Algorithm selection and parallel solving
//! - **Advanced**: Hitting set methods (MaxHS, IHS), SortMax, PMRES
//!
//! ## Algorithms
//!
//! ### Core-Guided MaxSAT
//! - **Fu-Malik**: Basic core-guided algorithm with assumption-based core extraction
//! - **OLL**: Opportunistic Literal Learning with incremental totalizer encoding
//! - **MSU3**: Iterative relaxation algorithm
//! - **WMax**: Stratified weighted MaxSAT solver
//! - **RC2**: Modern core-guided algorithm using relaxable cardinality constraints
//!
//! ### Advanced Algorithms
//! - **MaxHS**: Hitting set based MaxSAT solver
//! - **IHS**: Implicit Hitting Sets algorithm
//! - **SortMax**: Uses sorting networks for cardinality constraints
//! - **PMRES**: Partial MaxRes with core-guided relaxation
//!
//! ### Hybrid & Portfolio
//! - **Hybrid**: Combines SLS (Stochastic Local Search) with exact solvers
//! - **Portfolio**: Algorithm selection based on problem features
//! - **LNS**: Large Neighborhood Search with various operators
//!
//! ## Examples
//!
//! ### Basic MaxSAT Usage
//!
//! ```rust,ignore
//! use oxiz_opt::{MaxSatSolver, Weight};
//! use oxiz_sat::{Lit, Var};
//!
//! let mut solver = MaxSatSolver::new();
//!
//! // Add hard constraint: x0 must be true
//! solver.add_hard([Lit::pos(Var::new(0))]);
//!
//! // Add soft constraints with weights
//! solver.add_soft_weighted([Lit::neg(Var::new(0))], Weight::from(5));
//! solver.add_soft([Lit::pos(Var::new(1))]);
//!
//! // Solve and get optimal solution
//! match solver.solve() {
//!     Ok(result) => {
//!         println!("Result: {}", result);
//!         println!("Cost: {}", solver.cost());
//!     }
//!     Err(e) => println!("Error: {}", e),
//! }
//! ```
//!
//! ### Algorithm Selection
//!
//! ```rust,ignore
//! use oxiz_opt::{MaxSatSolver, MaxSatConfig, MaxSatAlgorithm};
//!
//! // Use OLL algorithm for instances with many cores
//! let config = MaxSatConfig::builder()
//!     .algorithm(MaxSatAlgorithm::Oll)
//!     .max_iterations(10000)
//!     .build();
//!
//! let mut solver = MaxSatSolver::with_config(config);
//! // ... add constraints and solve
//! ```
//!
//! ### Weighted MaxSAT
//!
//! ```rust,ignore
//! use oxiz_opt::{MaxSatSolver, MaxSatAlgorithm, MaxSatConfig, Weight};
//!
//! let config = MaxSatConfig::builder()
//!     .algorithm(MaxSatAlgorithm::WMax)
//!     .build();
//!
//! let mut solver = MaxSatSolver::with_config(config);
//!
//! // Add soft constraints with different weights
//! solver.add_soft_weighted(clause1, Weight::from(10));
//! solver.add_soft_weighted(clause2, Weight::from(5));
//! solver.add_soft_weighted(clause3, Weight::from(2));
//! ```
//!
//! For complete working examples, see the `examples/` directory:
//! - `scheduling.rs` - Task scheduling with preferences
//! - `resource_allocation.rs` - Resource allocation optimization
//! - `algorithm_comparison.rs` - Comparing different algorithms
//!
//! ## Reference
//!
//! Z3's `opt/` directory, particularly:
//! - `opt_context.cpp` - Main optimization context
//! - `maxsmt.cpp` - MaxSMT solver
//! - `maxcore.cpp` - Core-guided MaxSAT
//! - `optsmt.cpp` - Objective optimization
//! - `opt_pareto.cpp` - Pareto optimization

pub mod cardinality_network;
pub mod context;
pub mod hybrid;
pub mod ihs;
pub mod lns;
pub mod maxhs;
pub mod maxsat;
pub mod maxsmt;
pub mod objective;
pub mod omt;
pub mod pareto;
pub mod pmres;
pub mod portfolio;
pub mod preprocess;
pub mod rc2;
pub mod sls;
pub mod smtlib;
pub mod smtlib_commands;
pub mod sortmax;
pub mod totalizer;
pub mod wcnf;

// Re-exports for convenience
pub use cardinality_network::{
    CardinalityClause, CardinalityNetworkEncoding, SortingNetwork,
    encode_at_most_k_cardinality_network,
};
pub use context::{
    ModelValue, OptConfig, OptContext, OptError, OptResult, OptStats, SoftConstraint,
    SoftConstraintId,
};
pub use hybrid::{HybridConfig, HybridError, HybridSolver, HybridStats, HybridStrategy};
pub use ihs::{IhsConfig, IhsError, IhsSolver, IhsStats};
pub use lns::{
    Assignment, LnsConfig, LnsError, LnsResult, LnsSolver, LnsStats, NeighborhoodOp,
    NeighborhoodOperator, RestartManager, RestartStrategy,
};
pub use maxhs::{MaxHsConfig, MaxHsError, MaxHsSolver, MaxHsStats};
pub use maxsat::{
    Core, MaxSatAlgorithm, MaxSatConfig, MaxSatConfigBuilder, MaxSatError, MaxSatResult,
    MaxSatSolver, MaxSatStats, SoftClause, SoftId, Weight,
};
pub use maxsmt::{
    MaxSmtConfig, MaxSmtError, MaxSmtResult, MaxSmtSolver, MaxSmtStats, SmtCore, SoftSmtConstraint,
    SoftSmtId,
};
pub use objective::{
    LinearObjective, Objective, ObjectiveBound, ObjectiveConfig, ObjectiveError, ObjectiveId,
    ObjectiveKind, ObjectiveManager, ObjectiveOptimizer, ObjectiveResult, ObjectiveStats,
};
pub use omt::{
    ArithConstraint, ComparisonOp, OmtConfig, OmtError, OmtObjective, OmtResult, OmtSolver,
    OmtStats, OmtStrategy,
};
pub use pareto::{
    ObjectiveBox, ObjectivePoint, ParetoConfig, ParetoError, ParetoFront, ParetoResult,
    ParetoSolution, ParetoSolver, ParetoStats,
};
pub use pmres::{PmresConfig, PmresSolver, PmresStats};
pub use portfolio::{
    PortfolioConfig, PortfolioError, PortfolioSolver, PortfolioStats, PortfolioStrategy,
    ProblemFeatures,
};
pub use preprocess::{PreprocessConfig, PreprocessStats, Preprocessor};
pub use rc2::{Rc2Config, Rc2Error, Rc2Solver, Rc2Stats};
pub use sls::{SlsConfig, SlsError, SlsSolver, SlsStats};
pub use smtlib_commands::{
    CommandResponse, ObjectiveValue, ObjectiveValueKind, ObjectivesResponse, OptCommand,
    OptCommandProcessor, format_weight, parse_weight,
};
pub use sortmax::SortMaxSolver;
pub use totalizer::{
    CardinalityEncoding, IncrementalTotalizer, Totalizer, TotalizerClause, encode_at_most_k,
    encode_at_most_one_pairwise,
};
pub use wcnf::{WcnfError, WcnfInstance};
