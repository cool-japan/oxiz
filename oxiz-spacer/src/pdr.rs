//! Property Directed Reachability (PDR/IC3) algorithm.
//!
//! This implements the Spacer algorithm for solving Constrained Horn Clauses.
//!
//! Reference: Z3's `muz/spacer/spacer_context.cpp`
//!
//! ## Algorithm Overview
//!
//! 1. Initialize: F_0 = Init, F_i = True for i > 0
//! 2. Main loop:
//!    a. Check if Bad is reachable from F_N
//!    b. If reachable: create POB and try to block
//!    c. If blocked: propagate lemmas, check for fixpoint
//!    d. If fixpoint: SAFE
//!    e. If counterexample: UNSAFE

use crate::chc::{ChcSystem, PredId, PredicateApp, Rule};
use crate::frames::{FrameManager, LemmaId};
use crate::pob::{PobId, PobManager};
use crate::reach::{CexState, Counterexample, ReachFactStore};
use crate::smt::{SmtError, SmtSolver};
use oxiz_core::{TermId, TermManager};
use smallvec::SmallVec;
use thiserror::Error;
use tracing::{debug, trace};

/// Errors that can occur during Spacer solving
#[derive(Error, Debug)]
pub enum SpacerError {
    /// The CHC system is empty
    #[error("empty CHC system")]
    EmptySystem,
    /// No query found in the system
    #[error("no query found in CHC system")]
    NoQuery,
    /// SMT solver error
    #[error("SMT solver error: {0}")]
    SolverError(String),
    /// SMT error from solver
    #[error("SMT error: {0}")]
    Smt(#[from] SmtError),
    /// Resource limit exceeded
    #[error("resource limit exceeded")]
    ResourceLimit,
    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result of Spacer solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpacerResult {
    /// Property holds - system is safe
    /// Contains inductive invariants for each predicate
    Safe,
    /// Counterexample found - system is unsafe
    Unsafe,
    /// Could not determine within resource limits
    Unknown,
}

/// Configuration for Spacer
#[derive(Debug, Clone)]
pub struct SpacerConfig {
    /// Maximum number of frames
    pub max_level: u32,
    /// Maximum number of POBs to process
    pub max_pobs: u32,
    /// Maximum number of SMT queries
    pub max_smt_queries: u32,
    /// Enable inductive generalization
    pub use_inductive_gen: bool,
    /// Enable counterexample-guided abstraction refinement
    pub use_cegar: bool,
    /// Verbosity level (0 = quiet, 1 = normal, 2 = verbose)
    pub verbosity: u32,
}

impl Default for SpacerConfig {
    fn default() -> Self {
        Self {
            max_level: 1000,
            max_pobs: 100000,
            max_smt_queries: 1_000_000,
            use_inductive_gen: true,
            use_cegar: true,
            verbosity: 0,
        }
    }
}

/// Statistics from Spacer solving
#[derive(Debug, Clone, Default)]
pub struct SpacerStats {
    /// Number of frames created
    pub num_frames: u32,
    /// Number of lemmas learned
    pub num_lemmas: u32,
    /// Number of inductive lemmas
    pub num_inductive: u32,
    /// Number of POBs processed
    pub num_pobs: u32,
    /// Number of POBs blocked
    pub num_blocked: u32,
    /// Number of SMT queries
    pub num_smt_queries: u32,
    /// Number of propagation attempts
    pub num_propagations: u32,
    /// Number of POBs subsumed
    pub num_subsumed: u32,
    /// Number of MIC (minimal inductive core) attempts
    pub num_mic_attempts: u32,
    /// Number of CTG (counterexample-guided) strengthenings
    pub num_ctg_strengthenings: u32,
    /// Number of lazy model extractions deferred
    pub num_lazy_models_deferred: u32,
    /// Number of lazy generalizations deferred
    pub num_lazy_generalizations_deferred: u32,
    /// Number of under-approximation states tracked
    pub num_under_approx_states: u32,
    /// Number of under-approximation cache hits
    pub num_under_approx_hits: u32,
    /// Number of SMT queries avoided via under-approximation
    pub num_under_approx_avoided_queries: u32,
    /// Total solving time (microseconds)
    pub total_time_us: u64,
    /// Time spent in reachability checks (microseconds)
    pub reachability_time_us: u64,
    /// Time spent in blocking (microseconds)
    pub blocking_time_us: u64,
    /// Time spent in propagation (microseconds)
    pub propagation_time_us: u64,
    /// Time spent in generalization (microseconds)
    pub generalization_time_us: u64,
}

/// The Spacer solver for Constrained Horn Clauses
pub struct Spacer<'a> {
    /// Term manager for creating formulas
    terms: &'a mut TermManager,
    /// The CHC system to solve
    system: &'a ChcSystem,
    /// Configuration
    config: SpacerConfig,
    /// Frame manager
    frames: FrameManager,
    /// POB manager
    pobs: PobManager,
    /// Reach facts
    reach_facts: ReachFactStore,
    /// Statistics
    stats: SpacerStats,
    /// Current counterexample (if found)
    counterexample: Option<Counterexample>,
}

impl<'a> Spacer<'a> {
    /// Create a new Spacer solver
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        Self::with_config(terms, system, SpacerConfig::default())
    }

    /// Create a new Spacer solver with configuration
    pub fn with_config(
        terms: &'a mut TermManager,
        system: &'a ChcSystem,
        config: SpacerConfig,
    ) -> Self {
        Self {
            terms,
            system,
            config,
            frames: FrameManager::new(),
            pobs: PobManager::new(),
            reach_facts: ReachFactStore::new(),
            stats: SpacerStats::default(),
            counterexample: None,
        }
    }

    /// Solve the CHC system
    pub fn solve(&mut self) -> Result<SpacerResult, SpacerError> {
        // Validate system
        if self.system.is_empty() {
            // Empty system is trivially safe - nothing can go wrong
            return Ok(SpacerResult::Safe);
        }

        if self.system.queries().next().is_none() {
            return Err(SpacerError::NoQuery);
        }

        // Initialize frames for all predicates
        self.initialize()?;

        // Main PDR loop
        loop {
            // Check resource limits
            if self.stats.num_frames > self.config.max_level {
                return Ok(SpacerResult::Unknown);
            }
            if self.stats.num_pobs > self.config.max_pobs {
                return Ok(SpacerResult::Unknown);
            }
            if self.stats.num_smt_queries > self.config.max_smt_queries {
                return Ok(SpacerResult::Unknown);
            }

            // Try to find a counterexample at the current level
            match self.check_reachability()? {
                ReachabilityResult::Unreachable => {
                    // Try to propagate lemmas
                    if self.propagate()? {
                        // Fixpoint found - system is safe
                        return Ok(SpacerResult::Safe);
                    }
                    // Move to next level
                    self.frames.next_level();
                    self.stats.num_frames = self.stats.num_frames.saturating_add(1);
                }
                ReachabilityResult::Reachable(pob_id) => {
                    // Try to block the POB
                    match self.block(pob_id)? {
                        BlockResult::Blocked => {
                            // Continue processing POBs
                        }
                        BlockResult::Counterexample => {
                            // Real counterexample found
                            return Ok(SpacerResult::Unsafe);
                        }
                    }
                }
            }
        }
    }

    /// Initialize the solver
    fn initialize(&mut self) -> Result<(), SpacerError> {
        // Initialize frames for all predicates
        for pred in self.system.predicates() {
            self.frames.get_or_create(pred.id);
        }

        // Process init rules to establish initial reach facts
        for rule in self.system.entries() {
            self.process_init_rule(rule)?;
        }

        Ok(())
    }

    /// Process an init rule
    fn process_init_rule(&mut self, rule: &Rule) -> Result<(), SpacerError> {
        if let Some(head_pred) = rule.head_predicate() {
            // The constraint of the init rule defines initial states
            let init_fact = rule.body.constraint;
            self.reach_facts.add(head_pred, init_fact, rule.id, true);
        }
        Ok(())
    }

    /// Check reachability of bad states
    fn check_reachability(&mut self) -> Result<ReachabilityResult, SpacerError> {
        let level = self.frames.current_level();

        // Check each query rule
        for query in self.system.queries() {
            // Get body predicates of the query
            for body_app in &query.body.predicates {
                // Check if bad state is reachable at current level
                // Pass the query constraint to properly check reachability
                if self.is_bad_reachable(body_app, query.body.constraint, level)? {
                    // Create a POB for the bad state
                    let pob_id = self.pobs.create(
                        body_app.pred,
                        query.body.constraint,
                        level,
                        0, // depth 0 for initial POBs
                    );
                    self.stats.num_pobs = self.stats.num_pobs.saturating_add(1);
                    return Ok(ReachabilityResult::Reachable(pob_id));
                }
            }
        }

        Ok(ReachabilityResult::Unreachable)
    }

    /// Check if a bad state is reachable
    fn is_bad_reachable(
        &mut self,
        app: &PredicateApp,
        query_constraint: TermId,
        level: u32,
    ) -> Result<bool, SpacerError> {
        // Build frame formula for this predicate at this level
        let frame_formula = self.build_frame_formula(app.pred, level);

        // Create temporary SMT solver for this query
        let mut smt = SmtSolver::new(self.terms, self.system);

        // Query: Is F_level(pred) /\ query_constraint SAT?
        // This checks if the bad state (defined by query_constraint) is reachable
        // given the current invariant approximation (frame_formula)
        let is_sat =
            match smt.is_state_reachable(app.pred, query_constraint, level, frame_formula)? {
                Some(_model) => {
                    debug!("Bad state reachable at level {}", level);
                    true
                }
                None => false,
            };

        self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
        Ok(is_sat)
    }

    /// Block a proof obligation
    fn block(&mut self, pob_id: PobId) -> Result<BlockResult, SpacerError> {
        // Extract POB data first to avoid holding borrow
        let (level, pred, post) = {
            let pob = self
                .pobs
                .get(pob_id)
                .ok_or_else(|| SpacerError::Internal("POB not found".to_string()))?;
            (pob.level(), pob.pred, pob.post)
        };

        // Check if already blocked by existing lemma
        if self.is_blocked_by_lemma(pred, post, level)? {
            if let Some(lemma_id) = self.find_blocking_lemma(pred, post, level) {
                self.pobs.close(pob_id, lemma_id);
                self.stats.num_blocked = self.stats.num_blocked.saturating_add(1);
            }
            return Ok(BlockResult::Blocked);
        }

        // Level 0: must check if truly reachable from init
        if level == 0 {
            // Check if the bad state is satisfiable with initial states
            if self.is_init_reachable(pred, post)? {
                // Construct counterexample
                self.build_counterexample(pob_id)?;
                return Ok(BlockResult::Counterexample);
            }
        }

        // Try to find a predecessor
        match self.find_predecessor(pob_id)? {
            Some(pred_pob_id) => {
                // Found predecessor - need to block it first
                // Recursively block the predecessor
                self.block(pred_pob_id)
            }
            None => {
                // No predecessor found - can generate blocking lemma
                let lemma = self.generalize_blocking_lemma(pob_id)?;
                let lemma_id = self.frames.add_lemma(pred, lemma, level);
                self.pobs.close(pob_id, lemma_id);
                self.stats.num_blocked = self.stats.num_blocked.saturating_add(1);
                self.stats.num_lemmas = self.stats.num_lemmas.saturating_add(1);
                Ok(BlockResult::Blocked)
            }
        }
    }

    /// Check if a state is blocked by an existing lemma
    fn is_blocked_by_lemma(
        &mut self,
        pred: PredId,
        state: TermId,
        level: u32,
    ) -> Result<bool, SpacerError> {
        // Check if any lemma at this level or higher blocks the state
        if let Some(pred_frames) = self.frames.get(pred) {
            // Collect lemma formulas to check
            let lemmas: Vec<TermId> = pred_frames
                .lemmas_geq_level(level)
                .map(|l| l.formula)
                .collect();

            // Check each lemma
            for lemma in lemmas {
                let mut smt = SmtSolver::new(self.terms, self.system);
                if smt.is_blocked_by(lemma, state)? {
                    self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
                    return Ok(true);
                }
                self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
            }
        }
        Ok(false)
    }

    /// Find a lemma that blocks a state
    fn find_blocking_lemma(&self, pred: PredId, _state: TermId, level: u32) -> Option<LemmaId> {
        // Find the first lemma that blocks the state
        // In a full implementation, we would check each lemma to see if it blocks the state
        // For now, return the first lemma at the level (if any)
        if let Some(pred_frames) = self.frames.get(pred) {
            pred_frames
                .lemmas_geq_level(level)
                .next()
                .map(|lemma| lemma.id)
        } else {
            None
        }
    }

    /// Check if a state is reachable from initial states
    fn is_init_reachable(&mut self, pred: PredId, _state: TermId) -> Result<bool, SpacerError> {
        // Check if state is satisfiable with init reach facts
        for _fact in self.reach_facts.for_pred(pred) {
            // In real implementation: check if fact /\ state is SAT
            self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
        }
        Ok(false)
    }

    /// Find a predecessor state for a POB
    fn find_predecessor(&mut self, pob_id: PobId) -> Result<Option<PobId>, SpacerError> {
        // Extract POB info first to avoid holding borrow
        let (pred, level, depth) = {
            let pob = self
                .pobs
                .get(pob_id)
                .ok_or_else(|| SpacerError::Internal("POB not found".to_string()))?;
            (pob.pred, pob.level(), pob.depth())
        };

        if level == 0 {
            return Ok(None);
        }

        // Collect rules that derive this predicate
        let rules: Vec<_> = self.system.rules_by_head(pred).collect();

        // Find rules that can derive this predicate
        for rule in rules {
            // Check if the transition is feasible
            if self.is_transition_feasible(rule, pob_id)? {
                // Create predecessor POBs for body predicates
                // In full implementation, we'd create POBs for all body predicates
                // For now, create POB for first body predicate (if any)
                if let Some(first_body_app) = rule.body.predicates.first() {
                    let pred_pob = self.pobs.create_derived(
                        first_body_app.pred,
                        rule.body.constraint,
                        level - 1,
                        depth + 1,
                        pob_id,
                    );
                    self.stats.num_pobs = self.stats.num_pobs.saturating_add(1);
                    return Ok(Some(pred_pob));
                }
            }
        }

        Ok(None)
    }

    /// Check if a transition is feasible
    fn is_transition_feasible(
        &mut self,
        _rule: &Rule,
        _pob_id: PobId,
    ) -> Result<bool, SpacerError> {
        // In real implementation:
        // 1. Get current state from POB
        // 2. Check if rule.body.constraint /\ F_{level-1}(body_preds) /\ post is SAT
        self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
        Ok(false)
    }

    /// Generalize a blocking lemma
    fn generalize_blocking_lemma(&mut self, pob_id: PobId) -> Result<TermId, SpacerError> {
        let pob = self
            .pobs
            .get(pob_id)
            .ok_or_else(|| SpacerError::Internal("POB not found".to_string()))?;

        // Basic generalization: negate the bad state
        // In real implementation, apply inductive generalization
        let lemma = self.terms.mk_not(pob.post);

        if self.config.use_inductive_gen {
            // Try to strengthen the lemma inductively
            // This would involve MIC (Minimal Inductive Clause)
        }

        Ok(lemma)
    }

    /// Build a counterexample trace
    fn build_counterexample(&mut self, pob_id: PobId) -> Result<(), SpacerError> {
        let mut cex = Counterexample::new();

        // Trace back from POB to initial state
        let mut current = Some(pob_id);
        while let Some(id) = current {
            if let Some(pob) = self.pobs.get(id) {
                cex.push(CexState {
                    pred: pob.pred,
                    state: pob.post,
                    rule: None,
                    assignments: SmallVec::new(),
                });
                current = pob.parent();
            } else {
                break;
            }
        }

        cex.reverse();
        self.counterexample = Some(cex);
        Ok(())
    }

    /// Propagate lemmas to higher levels
    fn propagate(&mut self) -> Result<bool, SpacerError> {
        self.stats.num_propagations = self.stats.num_propagations.saturating_add(1);

        // Try to push lemmas to higher levels
        let current_level = self.frames.current_level();

        for level in 1..=current_level {
            let mut all_pushed = true;

            // Collect all predicates to process
            let pred_ids: Vec<_> = self.system.predicates().map(|p| p.id).collect();

            for pred_id in pred_ids {
                // Collect lemmas to push (immutable borrow)
                let lemmas_to_push: Vec<_> = if let Some(pred_frames) = self.frames.get(pred_id) {
                    pred_frames.lemmas_at_level(level).map(|l| l.id).collect()
                } else {
                    Vec::new()
                };

                // Check and propagate each lemma
                for lemma_id in lemmas_to_push {
                    // Check if lemma can be pushed: F_level /\ T => lemma'
                    let can_push = self.can_push_lemma(pred_id, lemma_id, level)?;

                    if can_push {
                        if let Some(pred_frames) = self.frames.get_mut(pred_id) {
                            pred_frames.propagate(lemma_id, level + 1);
                        }
                    } else {
                        all_pushed = false;
                    }
                }
            }

            // If all lemmas at this level were pushed, we found a fixpoint
            if all_pushed && level == current_level {
                // Mark all pushed lemmas as inductive
                let pred_ids: Vec<_> = self.system.predicates().map(|p| p.id).collect();
                for pred_id in pred_ids {
                    if let Some(pred_frames) = self.frames.get_mut(pred_id) {
                        pred_frames.propagate_to_infinity(level);
                    }
                }
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check if a lemma can be pushed to the next level
    fn can_push_lemma(
        &mut self,
        pred: PredId,
        lemma_id: LemmaId,
        level: u32,
    ) -> Result<bool, SpacerError> {
        // Get the lemma formula
        let lemma = if let Some(pred_frames) = self.frames.get(pred) {
            if let Some(lemma_data) = pred_frames.get_lemma(lemma_id) {
                lemma_data.formula
            } else {
                return Ok(false);
            }
        } else {
            return Ok(false);
        };

        // Build frame formula at current level
        let frame_formula = self.build_frame_formula(pred, level);

        // Check if lemma is inductive: F_level /\ T => lemma'
        // This is checked by verifying UNSAT of: F_level /\ T /\ ¬lemma'
        let mut smt = SmtSolver::new(self.terms, self.system);
        let can_push = smt.is_lemma_inductive(pred, lemma, level, frame_formula)?;

        self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
        trace!(
            "Lemma {:?} at level {} can_push: {}",
            lemma_id, level, can_push
        );
        Ok(can_push)
    }

    /// Get the counterexample (if found)
    #[must_use]
    pub fn counterexample(&self) -> Option<&Counterexample> {
        self.counterexample.as_ref()
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &SpacerStats {
        &self.stats
    }

    /// Get inductive invariants for all predicates
    pub fn invariants(&self) -> Vec<(PredId, Vec<TermId>)> {
        let mut result = Vec::new();

        for pred in self.system.predicates() {
            if let Some(pred_frames) = self.frames.get(pred.id) {
                let invs: Vec<TermId> = pred_frames.inductive_lemmas().map(|l| l.formula).collect();
                if !invs.is_empty() {
                    result.push((pred.id, invs));
                }
            }
        }

        result
    }

    /// Reset the solver for a new run
    pub fn reset(&mut self) {
        self.frames.reset();
        self.pobs.clear();
        self.reach_facts.clear();
        self.stats = SpacerStats::default();
        self.counterexample = None;
    }

    /// Build a frame formula for a predicate at a given level
    /// Returns the conjunction of all lemmas at level or higher
    fn build_frame_formula(&mut self, pred: PredId, level: u32) -> TermId {
        if let Some(pred_frames) = self.frames.get(pred) {
            let lemmas: Vec<TermId> = pred_frames
                .lemmas_geq_level(level)
                .map(|l| l.formula)
                .collect();

            if lemmas.is_empty() {
                // No lemmas, frame is true
                self.terms.mk_true()
            } else if lemmas.len() == 1 {
                lemmas[0]
            } else {
                // Conjunction of all lemmas
                self.terms.mk_and(lemmas)
            }
        } else {
            // No frames for this predicate, return true
            self.terms.mk_true()
        }
    }
}

/// Result of reachability check
enum ReachabilityResult {
    /// Bad state is unreachable at current level
    Unreachable,
    /// Bad state is reachable, POB created
    Reachable(PobId),
}

/// Result of blocking a POB
enum BlockResult {
    /// POB was successfully blocked
    Blocked,
    /// A real counterexample was found
    Counterexample,
}

/// Legacy interface for backward compatibility
pub struct LegacySpacer {
    result: SpacerResult,
}

impl LegacySpacer {
    /// Create a new legacy Spacer solver
    pub fn new() -> Self {
        Self {
            result: SpacerResult::Unknown,
        }
    }

    /// Solve (placeholder for legacy interface)
    pub fn solve(&mut self) -> SpacerResult {
        self.result.clone()
    }
}

impl Default for LegacySpacer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chc::PredicateApp;

    #[test]
    fn test_spacer_creation() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let constraint = terms.mk_eq(x, zero);

        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            constraint,
            inv,
            [x],
        );

        let spacer = Spacer::new(&mut terms, &system);
        assert_eq!(spacer.stats().num_frames, 0);
    }

    #[test]
    fn test_spacer_config() {
        let config = SpacerConfig {
            max_level: 100,
            max_pobs: 1000,
            max_smt_queries: 10000,
            use_inductive_gen: true,
            use_cegar: false,
            verbosity: 1,
        };

        assert_eq!(config.max_level, 100);
        assert_eq!(config.max_smt_queries, 10000);
        assert!(config.use_inductive_gen);
        assert!(!config.use_cegar);
    }

    #[test]
    fn test_spacer_empty_system() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();

        let mut spacer = Spacer::new(&mut terms, &system);
        let result = spacer.solve();

        // Empty system is trivially safe - nothing can go wrong
        assert!(matches!(result, Ok(SpacerResult::Safe)));
    }

    #[test]
    fn test_spacer_no_query() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let constraint = terms.mk_true();

        // Only init rule, no query
        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            constraint,
            inv,
            [x],
        );

        let mut spacer = Spacer::new(&mut terms, &system);
        let result = spacer.solve();

        assert!(matches!(result, Err(SpacerError::NoQuery)));
    }

    #[test]
    #[ignore = "Requires complete arithmetic theory integration"]
    fn test_spacer_simple_safe() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);

        // Init: x = 0 => Inv(x)
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let init_constraint = terms.mk_eq(x, zero);

        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            init_constraint,
            inv,
            [x],
        );

        // Trans: Inv(x) /\ x' = x + 1 /\ x' < 10 => Inv(x')
        let x_prime = terms.mk_var("x'", terms.sorts.int_sort);
        let one = terms.mk_int(1);
        let ten = terms.mk_int(10);
        let x_plus_one = terms.mk_add([x, one]);
        let trans_eq = terms.mk_eq(x_prime, x_plus_one);
        let bound = terms.mk_lt(x_prime, ten);
        let trans_constraint = terms.mk_and([trans_eq, bound]);

        system.add_transition_rule(
            [
                ("x".to_string(), terms.sorts.int_sort),
                ("x'".to_string(), terms.sorts.int_sort),
            ],
            [PredicateApp::new(inv, [x])],
            trans_constraint,
            inv,
            [x_prime],
        );

        // Query: Inv(x) /\ x < 0 => false
        let neg_constraint = terms.mk_lt(x, zero);
        system.add_query(
            [("x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(inv, [x])],
            neg_constraint,
        );

        let mut spacer = Spacer::new(&mut terms, &system);
        let result = spacer.solve();

        // The system should be safe (x >= 0 is invariant)
        // Note: With placeholder SMT, this returns Safe due to is_bad_reachable returning false
        assert!(matches!(result, Ok(SpacerResult::Safe)));
    }

    #[test]
    fn test_legacy_spacer() {
        let spacer = LegacySpacer::new();
        assert!(matches!(spacer.result, SpacerResult::Unknown));
    }
}
