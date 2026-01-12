//! Counterexample-Guided (CTG) strengthening with conflict analysis.
//!
//! CTG uses counterexamples to inductiveness to guide lemma strengthening.
//! When a lemma is not inductive, we analyze the conflict (counterexample)
//! to determine which literals to add to make it inductive.
//!
//! Reference: Z3's `muz/spacer/spacer_generalizers.cpp` CTI/CTG implementation.

use crate::chc::{ChcSystem, PredId};
use crate::smt::{Model, SmtSolver};
use oxiz_core::{TermId, TermManager};
use smallvec::SmallVec;
use thiserror::Error;
use tracing::{debug, trace};

/// Errors from CTG strengthening
#[derive(Error, Debug)]
pub enum CtgError {
    /// SMT error during CTG
    #[error("SMT error: {0}")]
    Smt(String),
    /// No counterexample to inductiveness found
    #[error("no counterexample to inductiveness")]
    NoCounterexample,
    /// Invalid lemma
    #[error("invalid lemma: {0}")]
    InvalidLemma(String),
}

/// Result of CTG strengthening
#[derive(Debug, Clone)]
pub struct CtgResult {
    /// The strengthened lemma (cube)
    pub lemma: SmallVec<[TermId; 8]>,
    /// Literals added during strengthening
    pub added: SmallVec<[TermId; 4]>,
    /// Whether the result is now inductive
    pub is_inductive: bool,
    /// Number of SMT queries used
    pub num_queries: u32,
    /// The conflict core (minimal set of literals from the counterexample)
    pub conflict_core: SmallVec<[TermId; 4]>,
}

impl CtgResult {
    /// Create a new CTG result
    pub fn new(lemma: impl IntoIterator<Item = TermId>) -> Self {
        Self {
            lemma: lemma.into_iter().collect(),
            added: SmallVec::new(),
            is_inductive: false,
            num_queries: 0,
            conflict_core: SmallVec::new(),
        }
    }

    /// Convert to a formula (conjunction)
    pub fn to_formula(&self, terms: &mut TermManager) -> TermId {
        if self.lemma.is_empty() {
            terms.mk_true()
        } else if self.lemma.len() == 1 {
            self.lemma[0]
        } else {
            terms.mk_and(self.lemma.iter().copied())
        }
    }
}

/// CTG strengthener with full conflict analysis
pub struct CtgStrengthener<'a> {
    /// Term manager
    terms: &'a mut TermManager,
    /// CHC system
    system: &'a ChcSystem,
    /// Number of queries performed
    num_queries: u32,
}

impl<'a> CtgStrengthener<'a> {
    /// Create a new CTG strengthener
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        Self {
            terms,
            system,
            num_queries: 0,
        }
    }

    /// Strengthen a lemma using counterexample-guided analysis
    ///
    /// Algorithm:
    /// 1. Check if lemma is inductive
    /// 2. If not, extract counterexample model
    /// 3. Analyze conflict: determine why inductiveness fails
    /// 4. Extract conflict core (minimal relevant literals)
    /// 5. Add literals from conflict core to strengthen lemma
    /// 6. Repeat until inductive or max iterations
    pub fn strengthen(
        &mut self,
        pred: PredId,
        lemma: &[TermId],
        level: u32,
        frame_formula: TermId,
        max_iterations: u32,
    ) -> Result<CtgResult, CtgError> {
        debug!(
            "CTG strengthening for predicate {:?} with {} literals",
            pred,
            lemma.len()
        );

        if lemma.is_empty() {
            return Err(CtgError::InvalidLemma(
                "Cannot strengthen empty lemma".to_string(),
            ));
        }

        let mut current_lemma = lemma.to_vec();
        let mut result = CtgResult::new(lemma.iter().copied());
        let mut iteration = 0;

        while iteration < max_iterations {
            iteration += 1;

            // Build current lemma formula
            let lemma_formula = if current_lemma.len() == 1 {
                current_lemma[0]
            } else {
                self.terms.mk_and(current_lemma.clone())
            };

            // Check if lemma is inductive
            let mut smt = SmtSolver::new(self.terms, self.system);
            self.num_queries += 1;

            match smt.is_lemma_inductive(pred, lemma_formula, level, frame_formula) {
                Ok(true) => {
                    // Lemma is inductive!
                    result.lemma = current_lemma.into_iter().collect();
                    result.is_inductive = true;
                    result.num_queries = self.num_queries;
                    debug!("CTG: lemma is inductive after {} iterations", iteration);
                    return Ok(result);
                }
                Ok(false) => {
                    // Not inductive, get counterexample
                    trace!("CTG: lemma not inductive, analyzing counterexample");

                    // Extract counterexample model
                    match self.get_counterexample_to_inductiveness(
                        pred,
                        lemma_formula,
                        level,
                        frame_formula,
                    ) {
                        Ok(Some(cti_model)) => {
                            // Analyze conflict and extract core
                            let conflict_core =
                                self.analyze_conflict(pred, &current_lemma, &cti_model)?;

                            if conflict_core.is_empty() {
                                // Can't strengthen further
                                debug!("CTG: empty conflict core, cannot strengthen further");
                                break;
                            }

                            // Add literals from conflict core
                            let mut added_any = false;
                            for &lit in &conflict_core {
                                if !current_lemma.contains(&lit) {
                                    current_lemma.push(lit);
                                    result.added.push(lit);
                                    added_any = true;
                                    trace!("CTG: added literal from conflict core");
                                }
                            }

                            result.conflict_core = conflict_core;

                            if !added_any {
                                // No new literals to add
                                debug!("CTG: no new literals to add from conflict");
                                break;
                            }
                        }
                        Ok(None) => {
                            return Err(CtgError::NoCounterexample);
                        }
                        Err(e) => {
                            debug!("CTG: SMT error getting counterexample: {}", e);
                            return Err(CtgError::Smt(e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    debug!("CTG: SMT error checking inductiveness: {}", e);
                    return Err(CtgError::Smt(e.to_string()));
                }
            }
        }

        // Max iterations reached or couldn't strengthen further
        result.lemma = current_lemma.into_iter().collect();
        result.num_queries = self.num_queries;
        debug!(
            "CTG: completed after {} iterations (inductive: {})",
            iteration, result.is_inductive
        );
        Ok(result)
    }

    /// Get counterexample to inductiveness (CTI)
    ///
    /// Returns a model where:
    /// - The current state satisfies the frame and lemma
    /// - The next state violates the lemma (via some transition)
    fn get_counterexample_to_inductiveness(
        &mut self,
        pred: PredId,
        lemma: TermId,
        _level: u32,
        frame_formula: TermId,
    ) -> Result<Option<Model>, crate::smt::SmtError> {
        // Use is_state_reachable which returns Option<Model>
        let mut smt = SmtSolver::new(self.terms, self.system);
        self.num_queries += 1;

        // Check if there's a state reachable that violates the lemma
        // Construct: frame ∧ lemma is reachable
        smt.is_state_reachable(pred, lemma, 0, frame_formula)
    }

    /// Analyze conflict to extract minimal conflict core
    ///
    /// Given a counterexample model, determine which literals
    /// from the model are essential for the conflict.
    ///
    /// Uses conflict-driven clause learning (CDCL) style analysis.
    fn analyze_conflict(
        &mut self,
        pred: PredId,
        current_lemma: &[TermId],
        cti_model: &Model,
    ) -> Result<SmallVec<[TermId; 4]>, CtgError> {
        debug!(
            "Analyzing conflict with {} model assignments",
            cti_model.assignments().len()
        );

        let mut conflict_core = SmallVec::new();

        // Extract relevant constraints from the model
        let model_constraints = self.extract_model_constraints(pred, cti_model)?;

        if model_constraints.is_empty() {
            return Ok(conflict_core);
        }

        // Compute minimal unsatisfiable core (MUS) from model constraints
        // Try to find smallest subset that still conflicts with lemma

        let mut candidate_core: Vec<TermId> = model_constraints.clone();

        // Greedy minimization: try removing each literal
        let mut i = 0;
        while i < candidate_core.len() {
            let removed = candidate_core.remove(i);

            // Check if remaining core still creates a conflict
            if self.is_conflict_core(&candidate_core, current_lemma)? {
                // Still a conflict without this literal, keep it removed
                trace!("CTG: removed non-essential literal from conflict core");
            } else {
                // Need this literal for the conflict
                candidate_core.insert(i, removed);
                i += 1;
            }
        }

        conflict_core = candidate_core.into_iter().collect();

        debug!("CTG: conflict core has {} literals", conflict_core.len());
        Ok(conflict_core)
    }

    /// Extract relevant constraints from a model
    fn extract_model_constraints(
        &mut self,
        pred: PredId,
        model: &Model,
    ) -> Result<Vec<TermId>, CtgError> {
        let mut constraints = Vec::new();

        // Get predicate info
        let pred_info = self
            .system
            .get_predicate(pred)
            .ok_or_else(|| CtgError::InvalidLemma(format!("Predicate {:?} not found", pred)))?;

        // For each model assignment, create a constraint
        for (idx, &value) in model.assignments().iter().enumerate() {
            if idx < pred_info.params.len() {
                let sort = pred_info.params[idx];
                let var_name = format!("{}_{}", pred_info.name, idx);
                let var = self.terms.mk_var(&var_name, sort);

                // Create equality constraint: var = value
                let constraint = self.terms.mk_eq(var, value);
                constraints.push(constraint);
            }
        }

        Ok(constraints)
    }

    /// Check if a set of constraints forms a conflict core with the lemma
    fn is_conflict_core(&mut self, core: &[TermId], lemma: &[TermId]) -> Result<bool, CtgError> {
        if core.is_empty() {
            return Ok(false);
        }

        // Build lemma formula before creating SMT solver (to avoid borrowing conflicts)
        let lemma_formula = if lemma.is_empty() {
            self.terms.mk_true()
        } else if lemma.len() == 1 {
            lemma[0]
        } else {
            self.terms.mk_and(lemma.iter().copied())
        };

        let mut smt = SmtSolver::new(self.terms, self.system);
        self.num_queries += 1;

        smt.push();

        // Assert core constraints
        for &constraint in core {
            smt.assert(constraint);
        }

        // Assert lemma (should be satisfied)
        smt.assert(lemma_formula);

        // Check SAT: if UNSAT, the core conflicts with lemma
        let result = smt.check_sat();
        smt.pop();

        match result {
            Ok(is_sat) => Ok(!is_sat), // UNSAT means it's a conflict
            Err(e) => Err(CtgError::Smt(e.to_string())),
        }
    }

    /// Get the number of SMT queries performed
    pub fn num_queries(&self) -> u32 {
        self.num_queries
    }

    /// Reset query counter
    pub fn reset_queries(&mut self) {
        self.num_queries = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctg_strengthener_creation() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();
        let _ctg = CtgStrengthener::new(&mut terms, &system);
    }

    #[test]
    fn test_ctg_result() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let y = terms.mk_var("y", terms.sorts.int_sort);

        let mut result = CtgResult::new(vec![x]);
        assert_eq!(result.lemma.len(), 1);
        assert!(!result.is_inductive);

        result.added.push(y);
        result.is_inductive = true;

        assert!(result.is_inductive);
        assert_eq!(result.added.len(), 1);
    }

    #[test]
    fn test_ctg_result_to_formula() {
        use oxiz_core::TermKind;

        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let y = terms.mk_var("y", terms.sorts.int_sort);

        let result = CtgResult::new(vec![x, y]);
        let formula = result.to_formula(&mut terms);

        // Should be a conjunction
        if let Some(term) = terms.get(formula) {
            assert!(matches!(term.kind, TermKind::And(_)));
        }
    }
}
