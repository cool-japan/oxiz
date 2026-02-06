//! E-matching engine for quantifier instantiation
//!
//! This module provides a comprehensive E-matching implementation for efficient
//! quantifier instantiation in SMT solving. E-matching is the core technique used
//! to find instantiations of universally quantified formulas.
//!
//! # Architecture
//!
//! The E-matching engine consists of several key components:
//!
//! - **Pattern Compilation**: Patterns are compiled into efficient code trees
//! - **Indexing**: Term indices for fast lookup of matching candidates
//! - **Mod-time Optimization**: Tracks modifications to avoid redundant work
//! - **Relevancy Propagation**: Focuses instantiation on relevant terms
//! - **Multi-pattern Matching**: Optimized handling of multiple triggers
//!
//! # Algorithm Reference
//!
//! This implementation is based on Z3's E-matching algorithms as described in:
//! - src/sat/smt/q_ematch.cpp
//! - src/sat/smt/q_mam.cpp (Multi-pattern matching)
//! - src/sat/smt/q_model_fixer.cpp
//!
//! # Example
//!
//! ```ignore
//! use oxiz_core::ematching::{EmatchEngine, EmatchConfig};
//! use oxiz_core::ast::TermManager;
//!
//! let mut tm = TermManager::new();
//! let mut engine = EmatchEngine::new(EmatchConfig::default());
//!
//! // Register quantifiers and their patterns
//! // ... add quantifiers ...
//!
//! // Perform E-matching rounds
//! let instantiations = engine.match_round(&mut tm);
//! ```

pub mod code_tree;
pub mod fingerprint;
pub mod heuristics;
pub mod index;
pub mod mod_time;
pub mod multi_pattern;
pub mod pattern;
pub mod quantifier_inst;
pub mod relevancy;
pub mod substitution;
pub mod trigger;

pub use code_tree::{CodeTree, CodeTreeBuilder, CodeTreeNode, Instruction, InstructionKind};
pub use fingerprint::{FingerprintCache, FingerprintConfig, TermFingerprint, compute_fingerprint};
pub use heuristics::{
    ConflictDrivenHeuristic, GreedyHeuristic, HeuristicConfig, HybridHeuristic,
    InstantiationHeuristic, InstantiationPriority,
};
pub use index::{EgraphIndex, IndexConfig, IndexStats, InvertedIndex, TermIndex, TermIndexEntry};
pub use mod_time::{
    ModTime, ModTimeManager, ModTimeOptimization, ModTimeStats, ModificationTracker,
};
pub use multi_pattern::{
    MultiPatternBuilder, MultiPatternConfig, MultiPatternMatcher, MultiPatternStats,
    SharedPatternCache,
};
pub use pattern::{
    Pattern, PatternCompiler, PatternConfig, PatternKind, PatternNode, PatternStats,
    PatternVariable,
};
pub use quantifier_inst::{
    EmatchConfig, EmatchEngine, EmatchStats, InstantiationCache, InstantiationContext,
    InstantiationResult, QuantifierInfo,
};
pub use relevancy::{
    RelevancyConfig, RelevancyPropagator, RelevancyScore, RelevancyStats, RelevancyTracker,
};
pub use substitution::{
    Substitution, SubstitutionBuilder, SubstitutionCache, SubstitutionConfig, SubstitutionStats,
};
pub use trigger::{
    Trigger, TriggerConfig, TriggerGenerator, TriggerQuality, TriggerSelection, TriggerStats,
};

use crate::ast::{TermId, TermManager};
use crate::error::Result;

/// Main E-matching facade providing a unified interface
#[derive(Debug)]
pub struct EmatchingEngine {
    /// Quantifier instantiation engine
    engine: EmatchEngine,
    /// Trigger generator
    trigger_gen: TriggerGenerator,
    /// Configuration
    #[allow(dead_code)]
    config: EmatchingConfig,
}

/// Configuration for the E-matching system
#[derive(Debug, Clone)]
pub struct EmatchingConfig {
    /// E-matching engine configuration
    pub ematch: EmatchConfig,
    /// Trigger generation configuration
    pub trigger: TriggerConfig,
    /// Maximum instantiations per round
    pub max_instances_per_round: usize,
    /// Maximum total instantiations
    pub max_total_instances: usize,
    /// Enable relevancy-based filtering
    pub use_relevancy: bool,
    /// Enable mod-time optimization
    pub use_mod_time: bool,
}

impl Default for EmatchingConfig {
    fn default() -> Self {
        Self {
            ematch: EmatchConfig::default(),
            trigger: TriggerConfig::default(),
            max_instances_per_round: 1000,
            max_total_instances: 100000,
            use_relevancy: true,
            use_mod_time: true,
        }
    }
}

impl EmatchingEngine {
    /// Create a new E-matching engine
    pub fn new(config: EmatchingConfig) -> Self {
        Self {
            engine: EmatchEngine::new(config.ematch.clone()),
            trigger_gen: TriggerGenerator::new(config.trigger.clone()),
            config,
        }
    }

    /// Register a quantified formula for E-matching
    pub fn register_quantifier(
        &mut self,
        quant_id: TermId,
        manager: &mut TermManager,
    ) -> Result<()> {
        // Generate triggers if not explicitly provided
        let triggers = self.trigger_gen.generate_triggers(quant_id, manager)?;

        // Register with the E-matching engine
        self.engine
            .register_quantifier(quant_id, triggers, manager)?;

        Ok(())
    }

    /// Perform one round of E-matching
    pub fn match_round(&mut self, manager: &mut TermManager) -> Result<Vec<TermId>> {
        self.engine.match_round(manager)
    }

    /// Get statistics
    pub fn stats(&self) -> &EmatchStats {
        self.engine.stats()
    }

    /// Reset the engine state
    pub fn reset(&mut self) {
        self.engine.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ematching_config_default() {
        let config = EmatchingConfig::default();
        assert_eq!(config.max_instances_per_round, 1000);
        assert_eq!(config.max_total_instances, 100000);
        assert!(config.use_relevancy);
        assert!(config.use_mod_time);
    }

    #[test]
    fn test_ematching_engine_creation() {
        let config = EmatchingConfig::default();
        let _engine = EmatchingEngine::new(config);
        // Just verify it constructs without error
    }
}
