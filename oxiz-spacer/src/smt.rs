//! SMT solver integration for Spacer
//!
//! Provides incremental SMT queries, model extraction, and interpolation support.
//!
//! Reference: Z3's `muz/spacer/spacer_context.cpp` solver integration

use crate::chc::{ChcSystem, PredId, Rule};
use crate::interp::Interpolator;
use oxiz_core::{TermId, TermManager};
use oxiz_solver::{Context, SolverResult};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use thiserror::Error;
use tracing::{debug, trace};

/// Errors from SMT queries
#[derive(Error, Debug)]
pub enum SmtError {
    /// Solver returned unknown
    #[error("solver returned unknown")]
    Unknown,
    /// Internal solver error
    #[error("internal solver error: {0}")]
    Internal(String),
}

/// SMT solver interface for Spacer
pub struct SmtSolver<'a> {
    /// The SMT context
    ctx: Context,
    /// Term manager (borrowed from context)
    terms: &'a mut TermManager,
    /// CHC system
    system: &'a ChcSystem,
    /// Current assertion level (for push/pop tracking)
    level: u32,
    /// Cache of predicate frame formulas
    frame_cache: FxHashMap<(PredId, u32), TermId>,
    /// Interpolator for Craig interpolation
    interpolator: Interpolator,
    /// Statistics
    stats: SmtStats,
}

/// Statistics for SMT queries
#[derive(Debug, Clone, Default)]
pub struct SmtStats {
    /// Number of check-sat queries
    pub num_queries: u64,
    /// Number of SAT results
    pub num_sat: u64,
    /// Number of UNSAT results
    pub num_unsat: u64,
    /// Number of UNKNOWN results
    pub num_unknown: u64,
    /// Number of push operations
    pub num_push: u64,
    /// Number of pop operations
    pub num_pop: u64,
    /// Total time spent in check-sat (microseconds)
    pub total_check_sat_time_us: u64,
    /// Total time spent in model extraction (microseconds)
    pub total_model_extraction_time_us: u64,
    /// Frame cache hits
    pub frame_cache_hits: u64,
    /// Frame cache misses
    pub frame_cache_misses: u64,
}

impl<'a> SmtSolver<'a> {
    /// Create a new SMT solver for Spacer
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        let mut ctx = Context::new();
        ctx.set_logic("HORN"); // Use HORN logic for CHC solving

        Self {
            ctx,
            terms,
            system,
            level: 0,
            frame_cache: FxHashMap::default(),
            interpolator: Interpolator::new(),
            stats: SmtStats::default(),
        }
    }

    /// Push a solver context
    pub fn push(&mut self) {
        self.ctx.push();
        self.level += 1;
        self.stats.num_push += 1;
        trace!("SMT push to level {}", self.level);
    }

    /// Pop a solver context
    pub fn pop(&mut self) {
        if self.level > 0 {
            self.ctx.pop();
            self.level -= 1;
            self.stats.num_pop += 1;
            trace!("SMT pop to level {}", self.level);
        }
    }

    /// Assert a formula
    pub fn assert(&mut self, formula: TermId) {
        self.ctx.assert(formula);
        trace!("SMT assert formula");
    }

    /// Check satisfiability
    pub fn check_sat(&mut self) -> Result<bool, SmtError> {
        use std::time::Instant;

        self.stats.num_queries += 1;
        let start = Instant::now();
        let result = self.ctx.check_sat();
        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.total_check_sat_time_us += elapsed;

        match result {
            SolverResult::Sat => {
                self.stats.num_sat += 1;
                debug!("SMT query: SAT ({}µs)", elapsed);
                Ok(true)
            }
            SolverResult::Unsat => {
                self.stats.num_unsat += 1;
                debug!("SMT query: UNSAT ({}µs)", elapsed);
                Ok(false)
            }
            SolverResult::Unknown => {
                self.stats.num_unknown += 1;
                debug!("SMT query: UNKNOWN ({}µs)", elapsed);
                Err(SmtError::Unknown)
            }
        }
    }

    /// Check if a state is reachable: F_level(pred) ∧ state is SAT?
    pub fn is_state_reachable(
        &mut self,
        pred: PredId,
        state: TermId,
        level: u32,
        frame_formula: TermId,
    ) -> Result<Option<Model>, SmtError> {
        // Check if frame formula is cached
        if self.frame_cache.contains_key(&(pred, level)) {
            self.stats.frame_cache_hits += 1;
            trace!("Frame cache hit for predicate {:?} level {}", pred, level);
        } else {
            self.stats.frame_cache_misses += 1;
            self.frame_cache.insert((pred, level), frame_formula);
            trace!("Frame cache miss for predicate {:?} level {}", pred, level);
        }

        self.push();

        // Assert frame formula: F_level(pred)
        self.assert(frame_formula);

        // Assert the state
        self.assert(state);

        let is_sat = self.check_sat()?;
        let result = if is_sat {
            // Extract model
            Some(self.extract_model(pred))
        } else {
            None
        };

        self.pop();
        Ok(result)
    }

    /// Check if a transition is feasible:
    /// F_level(body_preds) ∧ transition_constraint ∧ post is SAT?
    pub fn is_transition_feasible(
        &mut self,
        rule: &Rule,
        body_frames: &[(PredId, TermId)],
        post: TermId,
    ) -> Result<Option<Model>, SmtError> {
        self.push();

        // Assert frames for body predicates
        for (_pred, frame_formula) in body_frames {
            self.assert(*frame_formula);
        }

        // Assert transition constraint
        self.assert(rule.body.constraint);

        // Assert post-condition
        self.assert(post);

        let is_sat = self.check_sat()?;
        let result = if is_sat {
            Some(self.extract_model(PredId::new(0))) // Will be refined
        } else {
            None
        };

        self.pop();
        Ok(result)
    }

    /// Check if a lemma is inductive at a level:
    /// F_level(pred) ∧ T ∧ ¬lemma' is UNSAT?
    pub fn is_lemma_inductive(
        &mut self,
        pred: PredId,
        lemma: TermId,
        _level: u32,
        frame_formula: TermId,
    ) -> Result<bool, SmtError> {
        self.push();

        // Assert frame at level
        self.assert(frame_formula);

        // Assert all transition rules for this predicate
        for rule in self.system.rules_by_head(pred) {
            // Assert: body ∧ constraint => head
            // Encode as: ¬(body ∧ constraint) ∨ head
            // For induction check, we want:
            // F_level ∧ body ∧ constraint => lemma'
            // Which is equivalent to checking UNSAT of:
            // F_level ∧ body ∧ constraint ∧ ¬lemma'

            // For now, simplified: just check the lemma holds
            self.assert(rule.body.constraint);
        }

        // Assert negation of lemma in next state
        let not_lemma = self.terms.mk_not(lemma);
        self.assert(not_lemma);

        let is_sat = self.check_sat()?;
        let is_inductive = !is_sat; // UNSAT means inductive

        self.pop();
        Ok(is_inductive)
    }

    /// Check if state is blocked by a lemma:
    /// lemma ∧ state is UNSAT?
    pub fn is_blocked_by(&mut self, lemma: TermId, state: TermId) -> Result<bool, SmtError> {
        self.push();

        self.assert(lemma);
        self.assert(state);

        let is_sat = self.check_sat()?;
        let is_blocked = !is_sat;

        self.pop();
        Ok(is_blocked)
    }

    /// Extract model from current satisfying assignment
    fn extract_model(&mut self, _pred: PredId) -> Model {
        use std::time::Instant;

        let start = Instant::now();

        // Use Context's get_model() method instead of accessing solver directly
        let model = if let Some(_model_data) = self.ctx.get_model() {
            let assignments = if let Some(_pred_decl) = self.system.get_predicate(_pred) {
                // Extract term IDs from model
                // For now, return empty - will be refined when we have proper variable tracking
                Vec::new()
            } else {
                Vec::new()
            };

            Model { assignments }
        } else {
            Model {
                assignments: Vec::new(),
            }
        };

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.total_model_extraction_time_us += elapsed;
        trace!("Model extraction: {}µs", elapsed);

        model
    }

    /// Generalize a cube using model-based projection
    /// Given a model M that satisfies cube C, find a minimal generalization
    pub fn generalize_cube(
        &mut self,
        cube: &[TermId],
        _pred: PredId,
        _model: &Model,
    ) -> Vec<TermId> {
        // MBP: Model-Based Projection
        // Try to drop literals from the cube while maintaining unsatisfiability

        let mut generalized = cube.to_vec();
        let mut i = 0;

        while i < generalized.len() {
            // Try removing literal i
            let removed = generalized.remove(i);

            // Check if the remaining cube is still sufficient
            self.push();

            // Assert all remaining literals
            for &lit in &generalized {
                self.assert(lit);
            }

            // Check if the cube is still unsatisfiable with the bad state
            // (This is a simplified version - real MBP is more sophisticated)
            let is_sat = self.check_sat().unwrap_or(true);

            self.pop();

            if is_sat {
                // Need this literal, put it back
                generalized.insert(i, removed);
                i += 1;
            }
            // else: successfully removed, continue with same index
        }

        generalized
    }

    /// Compute interpolant between A and B where A ∧ B is UNSAT
    /// Returns a formula I such that:
    /// - A => I
    /// - I ∧ B is UNSAT
    /// - I only uses common variables
    pub fn interpolate(&mut self, a: TermId, b: TermId) -> Result<TermId, SmtError> {
        // Check that A ∧ B is UNSAT
        self.push();
        self.assert(a);
        self.assert(b);
        let is_sat = self.check_sat()?;
        self.pop();

        if is_sat {
            return Err(SmtError::Internal(
                "Cannot interpolate SAT formula".to_string(),
            ));
        }

        // Use the Interpolator to compute Craig interpolant
        // This projects A onto common variables with B
        let interp = self
            .interpolator
            .interpolate(self.terms, a, b)
            .map_err(|e| SmtError::Internal(format!("Interpolation failed: {}", e)))?;

        Ok(interp.formula)
    }

    /// Get the current statistics
    pub fn stats(&self) -> &SmtStats {
        &self.stats
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.ctx.reset();
        self.level = 0;
        self.frame_cache.clear();
        self.stats = SmtStats::default();
    }
}

/// A model (satisfying assignment)
#[derive(Debug, Clone)]
pub struct Model {
    /// Variable assignments (term values)
    pub assignments: Vec<TermId>,
}

impl Model {
    /// Create an empty model
    pub fn new() -> Self {
        Self {
            assignments: Vec::new(),
        }
    }

    /// Get the value assigned to a variable (by index)
    pub fn get(&self, index: usize) -> Option<TermId> {
        self.assignments.get(index).copied()
    }

    /// Get all assignments
    pub fn assignments(&self) -> &[TermId] {
        &self.assignments
    }

    /// Check if model is empty
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

/// Model-based generalization result
#[derive(Debug, Clone)]
pub struct MbpResult {
    /// Generalized formula (cube without unnecessary literals)
    pub cube: SmallVec<[TermId; 8]>,
    /// Literals that were eliminated
    pub eliminated: SmallVec<[TermId; 4]>,
    /// Whether the generalization is inductive
    pub is_inductive: bool,
}

impl MbpResult {
    /// Create a new MBP result
    pub fn new(cube: impl IntoIterator<Item = TermId>) -> Self {
        Self {
            cube: cube.into_iter().collect(),
            eliminated: SmallVec::new(),
            is_inductive: false,
        }
    }

    /// Mark as inductive
    pub fn set_inductive(&mut self, inductive: bool) {
        self.is_inductive = inductive;
    }

    /// Add an eliminated literal
    pub fn add_eliminated(&mut self, lit: TermId) {
        self.eliminated.push(lit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smt_solver_creation() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();

        let solver = SmtSolver::new(&mut terms, &system);
        assert_eq!(solver.level, 0);
        assert_eq!(solver.stats.num_queries, 0);
    }

    #[test]
    fn test_smt_push_pop() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();

        let mut solver = SmtSolver::new(&mut terms, &system);

        assert_eq!(solver.level, 0);
        solver.push();
        assert_eq!(solver.level, 1);
        solver.push();
        assert_eq!(solver.level, 2);
        solver.pop();
        assert_eq!(solver.level, 1);
        solver.pop();
        assert_eq!(solver.level, 0);
    }

    #[test]
    fn test_smt_basic_sat() {
        let mut terms = TermManager::new();
        let t = terms.mk_true();
        let system = ChcSystem::new();

        let mut solver = SmtSolver::new(&mut terms, &system);
        solver.assert(t);

        let result = solver.check_sat();
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_smt_basic_unsat() {
        let mut terms = TermManager::new();
        let f = terms.mk_false();
        let system = ChcSystem::new();

        let mut solver = SmtSolver::new(&mut terms, &system);
        solver.assert(f);

        let result = solver.check_sat();
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_model_creation() {
        let model = Model::new();
        assert!(model.is_empty());
        assert_eq!(model.assignments().len(), 0);
    }

    #[test]
    fn test_mbp_result() {
        let cube = [TermId::new(1), TermId::new(2), TermId::new(3)];
        let mut mbp = MbpResult::new(cube);

        assert_eq!(mbp.cube.len(), 3);
        assert!(!mbp.is_inductive);

        mbp.set_inductive(true);
        assert!(mbp.is_inductive);

        mbp.add_eliminated(TermId::new(2));
        assert_eq!(mbp.eliminated.len(), 1);
    }
}
