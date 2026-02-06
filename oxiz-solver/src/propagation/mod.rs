//! Propagation subsystem.

pub mod backtrack_manager;
pub mod pipeline;
pub mod priority_queue;
pub mod theory_propagator;
pub mod watched_propagator;

pub use backtrack_manager::{
    Assignment, BacktrackManager, BacktrackStats, BoundUpdate, Checkpoint, DecisionLevel,
    ReasonClause, TheoryReason, UndoAction,
};
pub use pipeline::{
    Propagation as PipelinePropagation, PropagationConfig, PropagationLevel, PropagationPipeline,
    PropagationReason, PropagationStats, PropagationWatcher, TheoryId as PipelineTheoryId,
};
pub use priority_queue::{
    PriorityQueueConfig, PriorityQueueStats, Propagation, PropagationId, PropagationQueue,
    PropagationType,
};
pub use theory_propagator::{
    Explanation, PropagationResult, PropagatorConfig, PropagatorManager, PropagatorStats, TheoryId,
    TheoryPropagator,
};
pub use watched_propagator::{
    ConstraintData, ConstraintId, Watch, WatchType, WatchedConfig, WatchedConstraint,
    WatchedPropagator, WatchedStats,
};
