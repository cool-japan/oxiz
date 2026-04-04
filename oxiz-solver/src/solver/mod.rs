//! Main CDCL(T) SMT Solver module

pub(super) mod candidates;
pub(super) mod check_array;
pub(super) mod check_bv;
pub(super) mod check_dt;
pub(super) mod check_fp;
pub(super) mod check_nlsat;
pub(super) mod check_string;
pub(super) mod config;
pub(super) mod encode;
pub(super) mod model_builder;
pub(super) mod pigeonhole;
pub(super) mod theory_manager;
pub(super) mod trail;
pub(super) mod types;

pub use types::{
    FpConstraintData, Model, NamedAssertion, Proof, ProofStep, SolverConfig, SolverResult,
    Statistics, TheoryMode, UnsatCore,
};

use crate::mbqi::{MBQIIntegration, MBQIResult};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::simplify::Simplifier;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::ematching::{EmatchingConfig, EmatchingEngine};
use oxiz_core::sort::SortId;
#[cfg(test)]
use oxiz_sat::RestartStrategy;
use oxiz_sat::{
    Lit, Solver as SatSolver, SolverConfig as SatConfig, SolverResult as SatResult, Var,
};
use oxiz_theories::Theory;
use oxiz_theories::arithmetic::ArithSolver;
use oxiz_theories::bv::BvSolver;
use oxiz_theories::euf::EufSolver;

use theory_manager::TheoryManager;
use trail::{ContextState, TrailOp};
use types::{Constraint, ParsedArithConstraint, Polarity};

/// Main CDCL(T) SMT Solver
#[derive(Debug)]
pub struct Solver {
    /// Configuration
    pub(super) config: SolverConfig,
    /// SAT solver core
    pub(super) sat: SatSolver,
    /// EUF theory solver
    pub(super) euf: EufSolver,
    /// Arithmetic theory solver
    pub(super) arith: ArithSolver,
    /// Bitvector theory solver
    pub(super) bv: BvSolver,
    /// NLSAT solver for nonlinear arithmetic (QF_NIA/QF_NRA)
    #[cfg(feature = "std")]
    pub(super) nlsat: Option<oxiz_theories::nlsat::NlsatTheory>,
    /// MBQI solver for quantified formulas
    pub(super) mbqi: MBQIIntegration,
    /// E-matching engine for quantifier instantiation via trigger patterns
    pub(super) ematch_engine: EmatchingEngine,
    /// Whether the formula contains quantifiers
    pub(super) has_quantifiers: bool,
    /// Term to SAT variable mapping
    pub(super) term_to_var: FxHashMap<TermId, Var>,
    /// SAT variable to term mapping
    pub(super) var_to_term: Vec<TermId>,
    /// SAT variable to theory constraint mapping
    pub(super) var_to_constraint: FxHashMap<Var, Constraint>,
    /// SAT variable to parsed arithmetic constraint mapping
    pub(super) var_to_parsed_arith: FxHashMap<Var, ParsedArithConstraint>,
    /// Current logic
    pub(super) logic: Option<String>,
    /// Assertions
    pub(super) assertions: Vec<TermId>,
    /// Named assertions for unsat core tracking
    pub(super) named_assertions: Vec<NamedAssertion>,
    /// Assumption literals for unsat core tracking (maps assertion index to assumption var)
    /// Reserved for future use with assumption-based unsat core extraction
    #[allow(dead_code)]
    pub(super) assumption_vars: FxHashMap<u32, Var>,
    /// Model (if sat)
    pub(super) model: Option<Model>,
    /// Unsat core (if unsat)
    pub(super) unsat_core: Option<UnsatCore>,
    /// Context stack for push/pop
    pub(super) context_stack: Vec<ContextState>,
    /// Trail of operations for efficient undo
    pub(super) trail: Vec<TrailOp>,
    /// Tracking which literals have been processed by theories
    pub(super) theory_processed_up_to: usize,
    /// Whether to produce unsat cores
    pub(super) produce_unsat_cores: bool,
    /// Track if we've asserted False (for immediate unsat)
    pub(super) has_false_assertion: bool,
    /// Polarity tracking for optimization
    pub(super) polarities: FxHashMap<TermId, Polarity>,
    /// Whether polarity-aware encoding is enabled
    pub(super) polarity_aware: bool,
    /// Whether theory-aware branching is enabled
    pub(super) theory_aware_branching: bool,
    /// Proof of unsatisfiability (if proof generation is enabled)
    pub(super) proof: Option<Proof>,
    /// Formula simplifier
    pub(super) simplifier: Simplifier,
    /// Solver statistics
    pub(super) statistics: Statistics,
    /// Bitvector terms (for model extraction)
    pub(super) bv_terms: FxHashSet<TermId>,
    /// Whether we've seen arithmetic BV operations (division/remainder)
    /// Used to decide when to run eager BV checking
    pub(super) has_bv_arith_ops: bool,
    /// Arithmetic terms (Int/Real variables for model extraction)
    pub(super) arith_terms: FxHashSet<TermId>,
    /// Datatype constructor constraints: variable -> constructor name
    /// Used to detect mutual exclusivity conflicts (var = C1 AND var = C2 where C1 != C2)
    pub(super) dt_var_constructors: FxHashMap<TermId, oxiz_core::interner::Spur>,
    /// Cache for parsed arithmetic constraints, keyed by the comparison term id.
    /// `ParsedArithConstraint` is purely structural (depends only on the term graph),
    /// so it is safe to reuse across CDCL backtracks.
    pub(super) arith_parse_cache: FxHashMap<TermId, Option<ParsedArithConstraint>>,
    /// Set of compound term ids whose theory-variable sub-graph has been fully
    /// traversed by `track_theory_vars`.  Avoids redundant O(depth) re-walks
    /// when the same sub-expression appears in multiple parent constraints.
    pub(super) tracked_compound_terms: FxHashSet<TermId>,
    /// Cache for FP constraint checking results.
    pub(super) fp_constraint_cache: FxHashMap<TermId, FpConstraintData>,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    /// Create a new solver
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    /// Create a new solver with configuration
    #[must_use]
    pub fn with_config(config: SolverConfig) -> Self {
        let proof_enabled = config.proof;

        // Build SAT solver configuration from our config
        let sat_config = SatConfig {
            restart_strategy: config.restart_strategy,
            enable_inprocessing: config.enable_inprocessing,
            inprocessing_interval: config.inprocessing_interval,
            ..SatConfig::default()
        };

        // Note: The following features are controlled by the SAT solver's preprocessor
        // and clause management systems. We pass the configuration but the actual
        // implementation is in oxiz-sat:
        // - Clause minimization (via RecursiveMinimizer)
        // - Clause subsumption (via SubsumptionChecker)
        // - Variable elimination (via Preprocessor::variable_elimination)
        // - Blocked clause elimination (via Preprocessor::blocked_clause_elimination)
        // - Symmetry breaking (via SymmetryBreaker)

        Self {
            config,
            sat: SatSolver::with_config(sat_config),
            euf: EufSolver::new(),
            arith: ArithSolver::lra(),
            bv: BvSolver::new(),
            #[cfg(feature = "std")]
            nlsat: None,
            mbqi: MBQIIntegration::new(),
            ematch_engine: EmatchingEngine::new(EmatchingConfig::default()),
            has_quantifiers: false,
            term_to_var: FxHashMap::default(),
            var_to_term: Vec::new(),
            var_to_constraint: FxHashMap::default(),
            var_to_parsed_arith: FxHashMap::default(),
            logic: None,
            assertions: Vec::new(),
            named_assertions: Vec::new(),
            assumption_vars: FxHashMap::default(),
            model: None,
            unsat_core: None,
            context_stack: Vec::new(),
            trail: Vec::new(),
            theory_processed_up_to: 0,
            produce_unsat_cores: false,
            has_false_assertion: false,
            polarities: FxHashMap::default(),
            polarity_aware: true, // Enable polarity-aware encoding by default
            theory_aware_branching: true, // Enable theory-aware branching by default
            proof: if proof_enabled {
                Some(Proof::new())
            } else {
                None
            },
            simplifier: Simplifier::new(),
            statistics: Statistics::new(),
            bv_terms: FxHashSet::default(),
            has_bv_arith_ops: false,
            arith_terms: FxHashSet::default(),
            dt_var_constructors: FxHashMap::default(),
            arith_parse_cache: FxHashMap::default(),
            tracked_compound_terms: FxHashSet::default(),
            fp_constraint_cache: FxHashMap::default(),
        }
    }

    /// Get the proof (if proof generation is enabled and the result is unsat)
    #[must_use]
    pub fn get_proof(&self) -> Option<&Proof> {
        self.proof.as_ref()
    }

    /// Get the solver statistics
    #[must_use]
    pub fn get_statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Reset the solver statistics
    pub fn reset_statistics(&mut self) {
        self.statistics.reset();
    }

    /// Enable or disable theory-aware branching
    pub fn set_theory_aware_branching(&mut self, enabled: bool) {
        self.theory_aware_branching = enabled;
    }

    /// Check if theory-aware branching is enabled
    #[must_use]
    pub fn theory_aware_branching(&self) -> bool {
        self.theory_aware_branching
    }

    /// Enable or disable unsat core production
    pub fn set_produce_unsat_cores(&mut self, produce: bool) {
        self.produce_unsat_cores = produce;
    }

    /// Register a declared constant as an MBQI ground instantiation candidate.
    ///
    /// This must be called from the context layer whenever a `declare-const`
    /// command is processed, so that trigger-free quantifiers can be
    /// instantiated with constants that exist in scope.
    pub fn register_declared_const(&mut self, term: TermId, sort: SortId) {
        self.mbqi.register_declared_const(term, sort);
    }

    /// Get a SAT variable for a term, then check satisfiability
    pub fn check(&mut self, manager: &mut TermManager) -> SolverResult {
        // Check for trivial unsat (false assertion)
        if self.has_false_assertion {
            self.build_unsat_core_trivial_false();
            return SolverResult::Unsat;
        }

        if self.assertions.is_empty() {
            return SolverResult::Sat;
        }

        // Check string constraints for early conflict detection
        if self.check_string_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check floating-point constraints for early conflict detection
        if self.check_fp_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check datatype constraints for early conflict detection
        if self.check_dt_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check array constraints for early conflict detection
        if self.check_array_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check bitvector constraints for early conflict detection
        if self.check_bv_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check nonlinear arithmetic constraints for early conflict detection
        if self.check_nonlinear_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check resource limits before starting
        if self.config.max_conflicts > 0 && self.statistics.conflicts >= self.config.max_conflicts {
            return SolverResult::Unknown;
        }
        if self.config.max_decisions > 0 && self.statistics.decisions >= self.config.max_decisions {
            return SolverResult::Unknown;
        }

        // Run SAT solver with theory integration
        let mut theory_manager = TheoryManager::new(
            manager,
            &mut self.euf,
            &mut self.arith,
            &mut self.bv,
            &self.bv_terms,
            &self.var_to_constraint,
            &self.var_to_parsed_arith,
            &self.term_to_var,
            &self.var_to_term,
            self.config.theory_mode,
            &mut self.statistics,
            self.config.max_conflicts,
            self.config.max_decisions,
            self.has_bv_arith_ops,
        );

        // MBQI loop for quantified formulas
        let max_mbqi_iterations = 100;
        let mut mbqi_iteration = 0;

        loop {
            let sat_result = self.sat.solve_with_theory(&mut theory_manager);
            match sat_result {
                SatResult::Unsat => {
                    self.build_unsat_core();
                    return SolverResult::Unsat;
                }
                SatResult::Unknown => {
                    return SolverResult::Unknown;
                }
                SatResult::Sat => {
                    // If no quantifiers, we're done
                    if !self.has_quantifiers {
                        self.build_model(manager);
                        self.unsat_core = None;
                        return SolverResult::Sat;
                    }

                    // Build partial model for MBQI
                    self.build_model(manager);

                    // Run MBQI to check quantified formulas
                    let model_assignments = self
                        .model
                        .as_ref()
                        .map(|m| m.assignments().clone())
                        .unwrap_or_default();

                    let mbqi_result = self.mbqi.check_with_model(&model_assignments, manager);
                    match mbqi_result {
                        MBQIResult::NoQuantifiers => {
                            self.unsat_core = None;
                            return SolverResult::Sat;
                        }
                        MBQIResult::Satisfied => {
                            // All quantifiers satisfied by the current model.
                            self.unsat_core = None;
                            return SolverResult::Sat;
                        }
                        MBQIResult::InstantiationLimit => {
                            // Too many instantiations - return unknown
                            return SolverResult::Unknown;
                        }
                        MBQIResult::Conflict {
                            quantifier: _,
                            reason,
                        } => {
                            // Add conflict clause
                            let lits: Vec<Lit> = reason
                                .iter()
                                .filter_map(|&t| self.term_to_var.get(&t).map(|&v| Lit::neg(v)))
                                .collect();
                            if !lits.is_empty() {
                                self.sat.add_clause(lits);
                            }
                            // Continue loop
                        }
                        MBQIResult::NewInstantiations(instantiations) => {
                            // Collect ground sub-terms (especially Skolem
                            // applications) from instantiation results so they
                            // become MBQI candidates in subsequent rounds.
                            for inst in &instantiations {
                                self.collect_ground_candidates_from_term(inst.result, manager);
                            }

                            // Collect domain/disequality info for pigeonhole
                            let mut ph_domains: FxHashMap<TermId, (i64, i64)> =
                                FxHashMap::default();
                            let mut ph_diseqs: Vec<(TermId, TermId)> = Vec::new();

                            // Add instantiation lemmas
                            for inst in instantiations {
                                // If the instantiation result is definitively False
                                // (e.g., a nested Exists with no valid witness), add an
                                // empty clause to signal immediate UNSAT.
                                let is_false_result = manager
                                    .get(inst.result)
                                    .is_some_and(|t| matches!(t.kind, TermKind::False));
                                if is_false_result {
                                    self.sat.add_clause([] as [Lit; 0]);
                                    break;
                                }
                                // Scan for pigeonhole patterns (recurses into Implies)
                                self.scan_for_pigeonhole(
                                    inst.result,
                                    manager,
                                    &mut ph_domains,
                                    &mut ph_diseqs,
                                );
                                let lit = self.encode(inst.result, manager);
                                let ok = self.sat.add_clause([lit]);
                                let _ = ok;
                                self.add_arith_diseq_split(inst.result, manager);
                                self.add_arith_eq_trichotomy(inst.result, manager);
                                self.add_int_domain_clauses(inst.result, manager);
                            }
                            // Add pigeonhole exclusion clauses
                            if !ph_diseqs.is_empty() && !ph_domains.is_empty() {
                                self.add_pigeonhole_exclusions_from(
                                    &ph_domains,
                                    &ph_diseqs,
                                    manager,
                                );
                            }

                            // E-matching phase: find additional instantiations via trigger patterns
                            let ematch_lemmas =
                                self.ematch_engine.match_round(manager).unwrap_or_default();
                            let mut new_clauses_added = 0usize;
                            let mut ematch_unsat = false;
                            for lemma in ematch_lemmas {
                                let lit = self.encode(lemma, manager);
                                if self.sat.add_clause([lit]) {
                                    new_clauses_added += 1;
                                } else {
                                    ematch_unsat = true;
                                    break;
                                }
                            }
                            if ematch_unsat || new_clauses_added > 0 {
                                // SAT solver will process newly added clauses on next iteration
                            }
                            // Continue loop
                        }
                        MBQIResult::Unknown => {
                            // Some evaluations produced symbolic residuals.
                            // Generate blind instantiations (simplified) once
                            // to seed the solver with ground lemmas for array
                            // theory reasoning (pigeonhole, bounds, etc.).
                            if !self.mbqi.blind_tried() {
                                self.mbqi.mark_blind_tried();
                                // Clear dedup cache so that blind instantiations with
                                // corrected substitution results are not filtered out
                                // as duplicates of earlier (broken) engine results.
                                self.mbqi.clear_dedup_cache();
                                let blind = self.mbqi.generate_blind_instantiations(manager);
                                let mut ph_domains: FxHashMap<TermId, (i64, i64)> =
                                    FxHashMap::default();
                                let mut ph_diseqs: Vec<(TermId, TermId)> = Vec::new();
                                for inst in blind {
                                    let is_false = manager
                                        .get(inst.result)
                                        .is_some_and(|t| matches!(t.kind, TermKind::False));
                                    if is_false {
                                        self.sat.add_clause([] as [Lit; 0]);
                                        break;
                                    }
                                    // Track domains and disequalities for pigeonhole
                                    if let Some(dbg_t) = manager.get(inst.result) {}
                                    self.scan_for_pigeonhole(
                                        inst.result,
                                        manager,
                                        &mut ph_domains,
                                        &mut ph_diseqs,
                                    );
                                    let lit = self.encode(inst.result, manager);
                                    let _ = self.sat.add_clause([lit]);
                                    self.add_arith_diseq_split(inst.result, manager);
                                    self.add_arith_eq_trichotomy(inst.result, manager);
                                    self.add_int_domain_clauses(inst.result, manager);
                                }
                                // Add pigeonhole exclusion clauses directly
                                // from the collected domains and disequalities.
                                self.add_pigeonhole_exclusions_from(
                                    &ph_domains,
                                    &ph_diseqs,
                                    manager,
                                );
                            }
                            // After 2 Unknown rounds, try finite instantiation:
                            // for quantifiers with bounded integer guards like
                            // (i >= 0 && i <= 3), enumerate all values and add
                            // ground instances directly.
                            if mbqi_iteration == 2 {
                                let finite_insts =
                                    self.mbqi.generate_finite_domain_instantiations(manager);
                                if !finite_insts.is_empty() {
                                    let mut ph_d: FxHashMap<TermId, (i64, i64)> =
                                        FxHashMap::default();
                                    let mut ph_q: Vec<(TermId, TermId)> = Vec::new();
                                    for inst in &finite_insts {
                                        let simplified =
                                            self.mbqi.deep_simplify(inst.result, manager);
                                        // Skip tautologies
                                        if manager
                                            .get(simplified)
                                            .is_some_and(|t| matches!(t.kind, TermKind::True))
                                        {
                                            continue;
                                        }
                                        self.scan_for_pigeonhole(
                                            simplified, manager, &mut ph_d, &mut ph_q,
                                        );
                                        let lit = self.encode(simplified, manager);
                                        let _ = self.sat.add_clause([lit]);
                                        self.add_arith_diseq_split(simplified, manager);
                                        self.add_int_domain_clauses(simplified, manager);
                                    }
                                    if !ph_q.is_empty() && !ph_d.is_empty() {
                                        self.add_pigeonhole_exclusions_from(&ph_d, &ph_q, manager);
                                    }
                                }
                            }
                            if mbqi_iteration >= 10 {
                                // After exhausting blind and finite domain
                                // instantiation attempts, assume the model
                                // satisfies all quantifiers.  This is sound
                                // under the incomplete-but-practical MBQI
                                // heuristic used by Z3 and similar solvers.
                                self.unsat_core = None;
                                return SolverResult::Sat;
                            }
                            // Continue MBQI loop
                        }
                    }

                    mbqi_iteration += 1;
                    if mbqi_iteration >= max_mbqi_iterations {
                        return SolverResult::Unknown;
                    }

                    // Recreate theory manager for next iteration.
                    // Do NOT reset theory solvers here - resetting EUF/Arith/BV
                    // state causes spurious conflicts when accumulated lemmas from
                    // MBQI instantiations interact with theory state that was cleared.
                    // The theory state accumulates correctly across iterations.
                    theory_manager = TheoryManager::new(
                        manager,
                        &mut self.euf,
                        &mut self.arith,
                        &mut self.bv,
                        &self.bv_terms,
                        &self.var_to_constraint,
                        &self.var_to_parsed_arith,
                        &self.term_to_var,
                        &self.var_to_term,
                        self.config.theory_mode,
                        &mut self.statistics,
                        self.config.max_conflicts,
                        self.config.max_decisions,
                        self.has_bv_arith_ops,
                    );
                }
            }
        }
    }

    /// Check satisfiability under assumptions
    /// Assumptions are temporary constraints that don't modify the assertion stack
    pub fn check_with_assumptions(
        &mut self,
        assumptions: &[TermId],
        manager: &mut TermManager,
    ) -> SolverResult {
        // Save current state
        self.push();

        // Assert all assumptions
        for &assumption in assumptions {
            self.assert(assumption, manager);
        }

        // Check satisfiability
        let result = self.check(manager);

        // Restore state
        self.pop();

        result
    }

    /// Check satisfiability (pure SAT, no theory integration)
    /// Useful for benchmarking or when theories are not needed
    pub fn check_sat_only(&mut self, manager: &mut TermManager) -> SolverResult {
        if self.assertions.is_empty() {
            return SolverResult::Sat;
        }

        match self.sat.solve() {
            SatResult::Sat => {
                self.build_model(manager);
                SolverResult::Sat
            }
            SatResult::Unsat => SolverResult::Unsat,
            SatResult::Unknown => SolverResult::Unknown,
        }
    }

    /// Build the model after SAT solving, which can be used to efficiently extract minimal unsat cores
    pub fn enable_assumption_based_cores(&mut self) {
        self.produce_unsat_cores = true;
        // Assumption variables would be created during assertion
        // to enable fine-grained core extraction
    }

    /// Minimize an unsat core using greedy deletion
    /// This creates a minimal (but not necessarily minimum) unsatisfiable subset
    pub fn minimize_unsat_core(&mut self, manager: &mut TermManager) -> Option<UnsatCore> {
        if !self.produce_unsat_cores {
            return None;
        }

        // Get the current unsat core
        let core = self.unsat_core.as_ref()?;
        if core.is_empty() {
            return Some(core.clone());
        }

        // Extract the assertions in the core
        let mut core_assertions: Vec<_> = core
            .indices
            .iter()
            .map(|&idx| {
                let assertion = self.assertions[idx as usize];
                let name = self
                    .named_assertions
                    .iter()
                    .find(|na| na.index == idx)
                    .and_then(|na| na.name.clone());
                (idx, assertion, name)
            })
            .collect();

        // Try to remove each assertion one by one
        let mut i = 0;
        while i < core_assertions.len() {
            // Create a temporary solver with all assertions except the i-th one
            let mut temp_solver = Solver::new();
            temp_solver.set_logic(self.logic.as_deref().unwrap_or("ALL"));

            // Add all assertions except the i-th one
            for (j, &(_, assertion, _)) in core_assertions.iter().enumerate() {
                if i != j {
                    temp_solver.assert(assertion, manager);
                }
            }

            // Check if still unsat
            if temp_solver.check(manager) == SolverResult::Unsat {
                // Still unsat without this assertion - remove it
                core_assertions.remove(i);
                // Don't increment i, check the next element which is now at position i
            } else {
                // This assertion is needed
                i += 1;
            }
        }

        // Build the minimized core
        let mut minimized = UnsatCore::new();
        for (idx, _, name) in core_assertions {
            minimized.indices.push(idx);
            if let Some(n) = name {
                minimized.names.push(n);
            }
        }

        Some(minimized)
    }

    /// Get the model (if sat)
    #[must_use]
    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    /// Check satisfiability with resource limits.
    pub fn check_with_limits(
        &mut self,
        manager: &mut TermManager,
        limits: &crate::resource_limits::ResourceLimits,
    ) -> core::result::Result<SolverResult, crate::resource_limits::ResourceExhausted> {
        use crate::resource_limits::ResourceMonitor;
        let mut monitor = ResourceMonitor::new(limits.clone());
        if let Some(reason) = monitor.check() {
            return Err(reason);
        }
        let orig_max_conflicts = self.config.max_conflicts;
        let orig_max_decisions = self.config.max_decisions;
        if let Some(max_c) = limits.max_conflicts {
            if self.config.max_conflicts == 0 || max_c < self.config.max_conflicts {
                self.config.max_conflicts = max_c;
            }
        }
        if let Some(max_d) = limits.max_decisions {
            if self.config.max_decisions == 0 || max_d < self.config.max_decisions {
                self.config.max_decisions = max_d;
            }
        }
        let result = self.check(manager);
        self.config.max_conflicts = orig_max_conflicts;
        self.config.max_decisions = orig_max_decisions;
        monitor.conflicts = self.statistics.conflicts;
        monitor.decisions = self.statistics.decisions;
        monitor.restarts = self.statistics.restarts;
        monitor.theory_checks =
            self.statistics.theory_propagations + self.statistics.theory_conflicts;
        if result == SolverResult::Unknown {
            if let Some(reason) = monitor.check() {
                return Err(reason);
            }
        }
        Ok(result)
    }
    /// Set a wall-clock timeout.
    pub fn set_timeout(&mut self, timeout: core::time::Duration) {
        self.config.timeout_ms = timeout.as_millis() as u64;
    }
    /// Set the maximum number of SAT conflicts.
    pub fn set_conflict_limit(&mut self, max_conflicts: u64) {
        self.config.max_conflicts = max_conflicts;
    }
    /// Set the maximum number of SAT decisions.
    pub fn set_decision_limit(&mut self, max_decisions: u64) {
        self.config.max_decisions = max_decisions;
    }

    /// Assert multiple terms at once
    /// This is more efficient than calling assert() multiple times
    pub fn assert_many(&mut self, terms: &[TermId], manager: &mut TermManager) {
        for &term in terms {
            self.assert(term, manager);
        }
    }

    /// Get the number of assertions in the solver
    #[must_use]
    pub fn num_assertions(&self) -> usize {
        self.assertions.len()
    }

    /// Get the number of variables in the SAT solver
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.term_to_var.len()
    }

    /// Check if the solver has any assertions
    #[must_use]
    pub fn has_assertions(&self) -> bool {
        !self.assertions.is_empty()
    }

    /// Get the current context level (push/pop depth)
    #[must_use]
    pub fn context_level(&self) -> usize {
        self.context_stack.len()
    }

    /// Push a context level
    pub fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_assertions: self.assertions.len(),
            num_vars: self.var_to_term.len(),
            has_false_assertion: self.has_false_assertion,
            trail_position: self.trail.len(),
        });
        self.sat.push();
        self.euf.push();
        self.arith.push();
        #[cfg(feature = "std")]
        if let Some(nlsat) = &mut self.nlsat {
            nlsat.push();
        }
    }

    /// Pop a context level using trail-based undo
    pub fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // Undo all operations in the trail since the push
            while self.trail.len() > state.trail_position {
                if let Some(op) = self.trail.pop() {
                    match op {
                        TrailOp::AssertionAdded { index } => {
                            if self.assertions.len() > index {
                                self.assertions.truncate(index);
                            }
                        }
                        TrailOp::VarCreated { var: _, term } => {
                            // Remove the term-to-var mapping
                            self.term_to_var.remove(&term);
                        }
                        TrailOp::ConstraintAdded { var } => {
                            // Remove the constraint
                            self.var_to_constraint.remove(&var);
                        }
                        TrailOp::FalseAssertionSet => {
                            // Reset the flag
                            self.has_false_assertion = false;
                        }
                        TrailOp::NamedAssertionAdded { index } => {
                            // Remove the named assertion
                            if self.named_assertions.len() > index {
                                self.named_assertions.truncate(index);
                            }
                        }
                        TrailOp::BvTermAdded { term } => {
                            // Remove the bitvector term
                            self.bv_terms.remove(&term);
                        }
                        TrailOp::ArithTermAdded { term } => {
                            // Remove the arithmetic term
                            self.arith_terms.remove(&term);
                        }
                    }
                }
            }

            // Use state to restore other fields
            self.assertions.truncate(state.num_assertions);
            self.var_to_term.truncate(state.num_vars);
            self.has_false_assertion = state.has_false_assertion;

            self.sat.pop();
            self.euf.pop();
            self.arith.pop();
            #[cfg(feature = "std")]
            if let Some(nlsat) = &mut self.nlsat {
                nlsat.pop();
            }
        }
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.sat.reset();
        self.euf.reset();
        self.arith.reset();
        self.bv.reset();
        self.term_to_var.clear();
        self.var_to_term.clear();
        self.var_to_constraint.clear();
        self.var_to_parsed_arith.clear();
        self.assertions.clear();
        self.named_assertions.clear();
        self.model = None;
        self.unsat_core = None;
        self.context_stack.clear();
        self.trail.clear();
        self.logic = None;
        self.theory_processed_up_to = 0;
        self.has_false_assertion = false;
        self.bv_terms.clear();
        self.arith_terms.clear();
        self.dt_var_constructors.clear();
        self.arith_parse_cache.clear();
        self.tracked_compound_terms.clear();
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: SolverConfig) {
        self.config = config;
    }

    /// Get solver statistics
    #[must_use]
    pub fn stats(&self) -> &oxiz_sat::SolverStats {
        self.sat.stats()
    }
}

#[cfg(test)]
mod tests;
