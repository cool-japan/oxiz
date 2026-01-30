//! Model Building for SMT Solvers.

pub mod advanced_builder;
pub mod builder;
pub mod completion;
pub mod minimizer;

pub use advanced_builder::{
    AdvancedModelBuilder, ArrayValue, Model as AdvancedModel,
    ModelBuilderConfig as AdvancedModelBuilderConfig,
    ModelBuilderStats as AdvancedModelBuilderStats, ModelValue, Theory, Value as ModelValue2,
};
pub use builder::{Model, ModelBuilder, ModelBuilderConfig, ModelBuilderStats, Value, VarId};
pub use completion::{CompletionConfig, CompletionStats, CompletionStrategy, ModelCompleter};
pub use minimizer::{
    Assignment, MinimizationStrategy, MinimizerConfig, MinimizerStats, ModelMinimizer,
};
