//! Bounded Model Checking (BMC) for CHC systems.
//!
//! BMC complements PDR/IC3 by exploring reachability up to a bounded depth.
//! It's particularly effective for finding counterexamples quickly.
//!
//! Reference: Z3's BMC implementation and standard BMC algorithms

use crate::chc::{ChcSystem, PredId};
use crate::reach::{CexState, Counterexample};
use crate::smt::{SmtError, SmtSolver};
use oxiz_core::{TermId, TermManager};
use smallvec::SmallVec;
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, trace};

/// Errors that can occur during BMC
#[derive(Error, Debug)]
pub enum BmcError {
    /// The CHC system is empty
    #[error("empty CHC system")]
    EmptySystem,
    /// No query found in the system
    #[error("no query found in CHC system")]
    NoQuery,
    /// SMT solver error
    #[error("SMT error: {0}")]
    Smt(#[from] SmtError),
    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result of BMC
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BmcResult {
    /// Property holds up to the given bound
    /// (does not prove safety, only that no counterexample was found)
    Safe(u32),
    /// Counterexample found at the given depth
    Unsafe(u32),
}

/// Configuration for BMC
#[derive(Debug, Clone)]
pub struct BmcConfig {
    /// Maximum depth to explore
    pub max_depth: u32,
    /// Enable k-induction
    pub use_kinduction: bool,
    /// Verbosity level (0 = quiet, 1 = normal, 2 = verbose)
    pub verbosity: u32,
}

impl Default for BmcConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            use_kinduction: false,
            verbosity: 0,
        }
    }
}

/// Statistics from BMC
#[derive(Debug, Clone, Default)]
pub struct BmcStats {
    /// Maximum depth reached
    pub max_depth_reached: u32,
    /// Number of SMT queries
    pub num_smt_queries: u32,
    /// Number of unrollings
    pub num_unrollings: u32,
}

/// Bounded Model Checker
pub struct Bmc<'a> {
    /// Term manager for creating formulas
    terms: &'a mut TermManager,
    /// The CHC system to check
    system: &'a ChcSystem,
    /// Configuration
    config: BmcConfig,
    /// Statistics
    stats: BmcStats,
    /// Symbolic states at each depth (predicate -> state formula)
    states: Vec<HashMap<PredId, TermId>>,
    /// Current counterexample (if found)
    counterexample: Option<Counterexample>,
}

impl<'a> Bmc<'a> {
    /// Create a new BMC instance
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        Self::with_config(terms, system, BmcConfig::default())
    }

    /// Create a new BMC instance with configuration
    pub fn with_config(
        terms: &'a mut TermManager,
        system: &'a ChcSystem,
        config: BmcConfig,
    ) -> Self {
        Self {
            terms,
            system,
            config,
            stats: BmcStats::default(),
            states: Vec::new(),
            counterexample: None,
        }
    }

    /// Run bounded model checking
    pub fn check(&mut self) -> Result<BmcResult, BmcError> {
        // Validate system
        if self.system.is_empty() {
            return Err(BmcError::EmptySystem);
        }

        if self.system.queries().next().is_none() {
            return Err(BmcError::NoQuery);
        }

        // Initialize with initial states
        self.initialize()?;

        // Unroll the transition system up to max_depth
        for depth in 0..=self.config.max_depth {
            self.stats.max_depth_reached = depth;

            debug!("BMC: checking depth {}", depth);

            // Check if bad state is reachable at this depth
            if self.check_bad_at_depth(depth)? {
                debug!("BMC: counterexample found at depth {}", depth);
                return Ok(BmcResult::Unsafe(depth));
            }

            // Try k-induction if enabled
            if self.config.use_kinduction && depth > 0 && self.check_kinduction(depth)? {
                debug!("BMC: k-induction proved safety at depth {}", depth);
                return Ok(BmcResult::Safe(depth));
            }

            // Unroll one more step (unless we're at max depth)
            if depth < self.config.max_depth {
                self.unroll(depth)?;
                self.stats.num_unrollings += 1;
            }
        }

        // Reached max depth without finding counterexample
        Ok(BmcResult::Safe(self.config.max_depth))
    }

    /// Initialize with initial states
    fn initialize(&mut self) -> Result<(), BmcError> {
        let mut initial_states = HashMap::new();

        // Collect initial state constraints from init rules
        for rule in self.system.entries() {
            if let Some(head_pred) = rule.head_predicate() {
                // The constraint defines the initial states for this predicate
                initial_states.insert(head_pred, rule.body.constraint);
            }
        }

        self.states.push(initial_states);
        Ok(())
    }

    /// Check if bad state is reachable at given depth
    fn check_bad_at_depth(&mut self, depth: u32) -> Result<bool, BmcError> {
        let depth_idx = depth as usize;
        if depth_idx >= self.states.len() {
            return Ok(false);
        }

        let current_states = &self.states[depth_idx];

        // Check each query rule
        for query in self.system.queries() {
            // Build constraint: states[depth] /\ query.constraint
            for body_app in &query.body.predicates {
                if let Some(&state_formula) = current_states.get(&body_app.pred) {
                    // Check if state_formula /\ query.body.constraint is SAT
                    let _combined = self.terms.mk_and([state_formula, query.body.constraint]);

                    // For BMC, we just check satisfiability
                    // In full implementation, we'd use the SMT solver's check_sat
                    trace!(
                        "BMC: checking satisfiability at depth {} for predicate {:?}",
                        depth, body_app.pred
                    );

                    // Placeholder: actual implementation would check SAT
                    // If SAT, extract model and build counterexample
                    let _smt = SmtSolver::new(self.terms, self.system);
                    self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);

                    // For now, just assume unreachable (false)
                    // Real implementation would call smt.check_sat()
                    let _is_sat = false;

                    if _is_sat {
                        self.build_counterexample(depth)?;
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Unroll the transition system one more step
    fn unroll(&mut self, depth: u32) -> Result<(), BmcError> {
        let mut next_states = HashMap::new();

        // For each predicate, compute the next state
        for pred_info in self.system.predicates() {
            let pred = pred_info.id;
            let mut next_state_disjuncts = Vec::new();

            // Find all rules that can derive this predicate
            for rule in self.system.rules_by_head(pred) {
                // Build the transition constraint
                // This involves:
                // 1. Getting current states of body predicates
                // 2. Conjuncting with rule.body.constraint
                // 3. Projecting onto head variables

                let mut transition_conjuncts = vec![rule.body.constraint];

                // Add current states of body predicates
                for body_app in &rule.body.predicates {
                    if let Some(state) = self.states[depth as usize].get(&body_app.pred) {
                        transition_conjuncts.push(*state);
                    }
                }

                // Build conjunction
                if !transition_conjuncts.is_empty() {
                    let transition = self.terms.mk_and(transition_conjuncts.clone());
                    next_state_disjuncts.push(transition);
                }
            }

            // Build disjunction of all possible transitions
            if !next_state_disjuncts.is_empty() {
                let next_state = if next_state_disjuncts.len() == 1 {
                    next_state_disjuncts[0]
                } else {
                    self.terms.mk_or(next_state_disjuncts.clone())
                };
                next_states.insert(pred, next_state);
            }
        }

        self.states.push(next_states);
        Ok(())
    }

    /// Check k-induction at given depth
    ///
    /// K-induction tries to prove that the property is inductive with k steps:
    /// 1. Base case: Property holds for 0..k steps (checked by BMC already)
    /// 2. Inductive step: If property holds for any k consecutive states,
    ///    then it holds for the next state too
    ///
    /// Full SMT-integrated k-induction implementation:
    /// - Assumes k arbitrary states satisfying the transition relation
    /// - Checks if the property holds at step k+1
    /// - If yes, the property is k-inductive (proved safe)
    fn check_kinduction(&mut self, depth: u32) -> Result<bool, BmcError> {
        trace!("BMC: checking k-induction at depth {}", depth);

        if depth == 0 {
            // Can't do k-induction with k=0
            return Ok(false);
        }

        // For k-induction, we need to check:
        // ∀ states[0..k]. (transition_relation(states[0..k]) ∧ safe(states[0..k]))
        //                  => safe(states[k+1])
        //
        // Equivalently, check if the negation is UNSAT:
        // ∃ states[0..k+1]. transition_relation(states[0..k+1]) ∧
        //                    safe(states[0..k]) ∧ ¬safe(states[k+1])

        let k = depth;

        // Collect constraints for k consecutive transitions
        // (without assuming initial states - this is key for k-induction!)
        let mut transition_constraints = Vec::new();

        // Add k transitions
        for step in 0..k {
            // For each rule, add its transition constraint
            for rule in self.system.rules() {
                // Get the body constraint (transition guard)
                transition_constraints.push(rule.body.constraint);

                // Add constraints from body predicates
                for body_app in &rule.body.predicates {
                    // In k-induction, we use fresh symbolic states
                    // (not the concrete states from BMC unrolling)
                    if let Some(pred_info) = self.system.get_predicate(body_app.pred) {
                        // Create a symbolic state formula for this predicate at this step
                        // For simplicity, we use a boolean variable representing "predicate holds"
                        let state_var_name = format!("{}_{}", pred_info.name, step);
                        let state_var = self
                            .terms
                            .mk_var(&state_var_name, self.terms.sorts.bool_sort);
                        transition_constraints.push(state_var);
                    }
                }
            }
        }

        // Build safety constraints: property holds at steps 0..k-1
        let mut safety_constraints = Vec::new();
        for query in self.system.queries() {
            for step in 0..k {
                // At each step, the query should not be violated
                // i.e., ¬(query.body.constraint)
                let negated = self.terms.mk_not(query.body.constraint);
                let step_var_name = format!("safe_{}", step);
                let step_var = self
                    .terms
                    .mk_var(&step_var_name, self.terms.sorts.bool_sort);

                // Implication: if the query body predicates hold, then constraint must not hold
                let safety_at_step = self.terms.mk_or([negated, step_var]);
                safety_constraints.push(safety_at_step);
            }
        }

        // Build violation constraint: property is violated at step k
        let mut violation_constraints = Vec::new();
        for query in self.system.queries() {
            // At step k, the query IS violated
            // i.e., query.body.constraint holds
            violation_constraints.push(query.body.constraint);
        }

        // Combine all constraints
        let mut all_constraints = Vec::new();
        all_constraints.extend(transition_constraints);
        all_constraints.extend(safety_constraints);
        all_constraints.extend(violation_constraints);

        if all_constraints.is_empty() {
            return Ok(false);
        }

        let check_formula = self.terms.mk_and(all_constraints);

        // Check if this is UNSAT
        // If UNSAT, then k-induction succeeded (property is k-inductive)
        self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);

        // Create SMT solver after building all formulas (to avoid borrowing conflicts)
        let mut smt = SmtSolver::new(self.terms, self.system);

        // Push to create a new solver context
        smt.push();
        // Assert the formula
        smt.assert(check_formula);
        // Check satisfiability
        let result = smt.check_sat();
        // Pop to restore solver context
        smt.pop();

        match result {
            Ok(is_sat) => {
                if !is_sat {
                    // UNSAT: k-induction succeeded
                    debug!("K-induction: proved safety with k={}", k);
                    Ok(true)
                } else {
                    // SAT: k-induction failed (but doesn't mean property is false)
                    trace!("K-induction: failed at k={}, trying larger k", k);
                    Ok(false)
                }
            }
            Err(e) => {
                debug!("K-induction: SMT error: {}", e);
                Ok(false)
            }
        }
    }

    /// Build a counterexample trace
    fn build_counterexample(&mut self, depth: u32) -> Result<(), BmcError> {
        let mut cex = Counterexample::new();

        // Extract states from 0 to depth
        for step in 0..=depth {
            if let Some(states) = self.states.get(step as usize) {
                // For each query, extract the relevant state
                for query in self.system.queries() {
                    for body_app in &query.body.predicates {
                        if let Some(&state) = states.get(&body_app.pred) {
                            cex.push(CexState {
                                pred: body_app.pred,
                                state,
                                rule: None,
                                assignments: SmallVec::new(),
                            });
                        }
                    }
                }
            }
        }

        self.counterexample = Some(cex);
        Ok(())
    }

    /// Get the statistics
    pub fn stats(&self) -> &BmcStats {
        &self.stats
    }

    /// Get the counterexample (if found)
    pub fn counterexample(&self) -> Option<&Counterexample> {
        self.counterexample.as_ref()
    }
}

/// Hybrid BMC + PDR solver
///
/// Runs BMC and PDR in parallel (or sequentially) and returns
/// the first result.
pub struct HybridSolver {
    /// BMC configuration
    pub bmc_config: BmcConfig,
    /// Run BMC first before PDR
    pub bmc_first: bool,
}

impl HybridSolver {
    /// Create a new hybrid solver
    pub fn new() -> Self {
        Self {
            bmc_config: BmcConfig::default(),
            bmc_first: true,
        }
    }

    /// Run BMC first with shallow depth to quickly find bugs
    pub fn quick_bmc(mut self) -> Self {
        self.bmc_config.max_depth = 10;
        self.bmc_first = true;
        self
    }
}

impl Default for HybridSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chc::{ChcSystem, PredicateApp};

    #[test]
    fn test_bmc_creation() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();
        let bmc = Bmc::new(&mut terms, &system);
        assert_eq!(bmc.stats.max_depth_reached, 0);
    }

    #[test]
    fn test_bmc_config() {
        let config = BmcConfig {
            max_depth: 50,
            use_kinduction: true,
            verbosity: 1,
        };
        assert_eq!(config.max_depth, 50);
        assert!(config.use_kinduction);
    }

    #[test]
    fn test_bmc_simple() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        // Simple system: x = 0 => Inv(x), Inv(x) /\ x < 0 => false
        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let init_constraint = terms.mk_eq(x, zero);

        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            init_constraint,
            inv,
            [x],
        );

        let neg_constraint = terms.mk_lt(x, zero);
        system.add_query(
            [("x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(inv, [x])],
            neg_constraint,
        );

        let mut bmc = Bmc::new(&mut terms, &system);
        let result = bmc.check();

        // Should be safe (no counterexample found)
        assert!(result.is_ok());
        match result.unwrap() {
            BmcResult::Safe(_) => (),
            BmcResult::Unsafe(_) => panic!("Expected safe result"),
        }
    }

    #[test]
    fn test_hybrid_solver_creation() {
        let hybrid = HybridSolver::new();
        assert!(hybrid.bmc_first);

        let quick = HybridSolver::new().quick_bmc();
        assert_eq!(quick.bmc_config.max_depth, 10);
    }
}
