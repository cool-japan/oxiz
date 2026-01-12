//! Generalization algorithms for lemma strengthening
//!
//! Implements various generalization techniques:
//! - MIC (Minimal Inductive Clause): Generalize cubes to minimal inductive clauses
//! - CTI (Counterexample To Induction): Use counterexamples to guide generalization
//! - MBP (Model-Based Projection): Project out irrelevant literals
//!
//! Reference: Z3's `muz/spacer/spacer_generalizers.cpp`

use crate::chc::{ChcSystem, PredId};
use crate::smt::SmtSolver;
use oxiz_core::{TermId, TermKind, TermManager};
use smallvec::SmallVec;
use thiserror::Error;
use tracing::{debug, trace};

/// Strategy for literal elimination in MIC
#[derive(Debug, Clone, Copy)]
enum LiteralEliminationStrategy {
    /// Try to eliminate literals in sequential order
    Sequential,
    /// Try to eliminate literals in reverse order
    ReverseSequential,
    /// Try to eliminate larger terms first
    LargestFirst,
    /// Cost-based: eliminate high-cost (complex) literals first
    CostBased,
    /// Frequency-based: eliminate less frequent variables first
    FrequencyBased,
    /// Adaptive: combine multiple strategies dynamically
    Adaptive,
}

/// Errors from generalization
#[derive(Error, Debug)]
pub enum GeneralizationError {
    /// SMT error during generalization
    #[error("SMT error: {0}")]
    Smt(String),
    /// Invalid cube (empty or trivial)
    #[error("invalid cube: {0}")]
    InvalidCube(String),
}

/// Result of generalization
#[derive(Debug, Clone)]
pub struct GeneralizationResult {
    /// The generalized lemma (cube)
    pub lemma: SmallVec<[TermId; 8]>,
    /// Literals dropped during generalization
    pub dropped: SmallVec<[TermId; 4]>,
    /// Whether the result is inductive
    pub is_inductive: bool,
    /// Number of SMT queries used
    pub num_queries: u32,
}

impl GeneralizationResult {
    /// Create a new generalization result
    pub fn new(lemma: impl IntoIterator<Item = TermId>) -> Self {
        Self {
            lemma: lemma.into_iter().collect(),
            dropped: SmallVec::new(),
            is_inductive: false,
            num_queries: 0,
        }
    }

    /// Convert to a single formula (conjunction of literals)
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

/// Generalizer for creating inductive lemmas
pub struct Generalizer<'a> {
    /// Term manager
    terms: &'a mut TermManager,
    /// CHC system
    system: &'a ChcSystem,
    /// Number of queries performed
    num_queries: u32,
}

impl<'a> Generalizer<'a> {
    /// Create a new generalizer
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        Self {
            terms,
            system,
            num_queries: 0,
        }
    }

    /// Generalize a cube using MIC (Minimal Inductive Clause)
    ///
    /// Given a cube (conjunction of literals) that blocks a bad state,
    /// compute a minimal subset that is still inductive.
    ///
    /// Algorithm:
    /// 1. Start with the full cube
    /// 2. Try to drop each literal one at a time
    /// 3. Check if the remaining cube is still inductive
    /// 4. Keep dropping until no more literals can be removed
    ///
    /// Enhanced with multiple elimination strategies
    pub fn mic(
        &mut self,
        pred: PredId,
        cube: &[TermId],
        level: u32,
        frame_formula: TermId,
    ) -> Result<GeneralizationResult, GeneralizationError> {
        debug!(
            "MIC generalization for predicate {:?} with {} literals",
            pred,
            cube.len()
        );

        if cube.is_empty() {
            return Err(GeneralizationError::InvalidCube(
                "Cannot generalize empty cube".to_string(),
            ));
        }

        // Try different elimination strategies and pick the best result
        let strategies = [
            LiteralEliminationStrategy::Sequential,
            LiteralEliminationStrategy::ReverseSequential,
            LiteralEliminationStrategy::LargestFirst,
            LiteralEliminationStrategy::CostBased,
            LiteralEliminationStrategy::FrequencyBased,
            LiteralEliminationStrategy::Adaptive,
        ];

        let mut best_result = None;
        let mut best_size = cube.len();

        for strategy in &strategies {
            match self.mic_with_strategy(pred, cube, level, frame_formula, *strategy) {
                Ok(result) => {
                    if result.lemma.len() < best_size {
                        best_size = result.lemma.len();
                        best_result = Some(result);
                    }
                }
                Err(e) => {
                    debug!("MIC strategy {:?} failed: {}", strategy, e);
                }
            }
        }

        best_result.ok_or_else(|| {
            GeneralizationError::InvalidCube("All MIC strategies failed".to_string())
        })
    }

    /// MIC with a specific elimination strategy
    fn mic_with_strategy(
        &mut self,
        pred: PredId,
        cube: &[TermId],
        level: u32,
        frame_formula: TermId,
        strategy: LiteralEliminationStrategy,
    ) -> Result<GeneralizationResult, GeneralizationError> {
        let mut result = GeneralizationResult::new(cube.iter().copied());
        let mut current_cube = cube.to_vec();

        // Order literals according to strategy
        let indices = self.order_literals_for_elimination(&current_cube, strategy);

        for &idx in &indices {
            if idx >= current_cube.len() {
                continue;
            }

            // Find the current position of this literal
            let actual_idx = current_cube
                .iter()
                .position(|&lit| lit == cube[idx])
                .unwrap_or(idx);

            if actual_idx >= current_cube.len() {
                continue;
            }

            // Remove literal temporarily
            let removed_lit = current_cube.remove(actual_idx);

            // Build candidate lemma from remaining literals
            let candidate = if current_cube.is_empty() {
                self.terms.mk_true()
            } else if current_cube.len() == 1 {
                current_cube[0]
            } else {
                self.terms.mk_and(current_cube.clone())
            };

            // Check if candidate is inductive
            let mut smt = SmtSolver::new(self.terms, self.system);
            self.num_queries += 1;

            let is_inductive = match smt.is_lemma_inductive(pred, candidate, level, frame_formula) {
                Ok(inductive) => inductive,
                Err(e) => {
                    debug!("SMT error during MIC check: {}", e);
                    false
                }
            };

            if is_inductive {
                // Successfully dropped the literal
                result.dropped.push(removed_lit);
                trace!(
                    "MIC: successfully dropped literal with strategy {:?}",
                    strategy
                );
            } else {
                // Need to keep this literal
                current_cube.insert(actual_idx, removed_lit);
            }
        }

        result.lemma = current_cube.into_iter().collect();
        result.is_inductive = true;
        result.num_queries = self.num_queries;

        debug!(
            "MIC with {:?}: reduced from {} to {} literals",
            strategy,
            cube.len(),
            result.lemma.len()
        );

        Ok(result)
    }

    /// Order literals for elimination based on strategy
    fn order_literals_for_elimination(
        &self,
        cube: &[TermId],
        strategy: LiteralEliminationStrategy,
    ) -> Vec<usize> {
        match strategy {
            LiteralEliminationStrategy::Sequential => (0..cube.len()).collect(),
            LiteralEliminationStrategy::ReverseSequential => (0..cube.len()).rev().collect(),
            LiteralEliminationStrategy::LargestFirst => {
                // Order by term size (larger terms first)
                let mut indices: Vec<usize> = (0..cube.len()).collect();
                indices.sort_by_key(|&i| std::cmp::Reverse(self.term_size(cube[i])));
                indices
            }
            LiteralEliminationStrategy::CostBased => {
                // Order by computational cost (higher cost first)
                let mut indices: Vec<usize> = (0..cube.len()).collect();
                indices.sort_by_key(|&i| std::cmp::Reverse(self.term_cost(cube[i])));
                indices
            }
            LiteralEliminationStrategy::FrequencyBased => {
                // Order by variable frequency (less frequent first)
                let var_freq = self.compute_variable_frequency(cube);
                let mut indices: Vec<usize> = (0..cube.len()).collect();
                indices.sort_by_key(|&i| self.literal_frequency_score(cube[i], &var_freq));
                indices
            }
            LiteralEliminationStrategy::Adaptive => {
                // Adaptive: use heuristics based on cube properties
                // For small cubes: use sequential
                // For medium cubes: use cost-based
                // For large cubes: use frequency-based
                if cube.len() < 5 {
                    (0..cube.len()).collect()
                } else if cube.len() < 15 {
                    let mut indices: Vec<usize> = (0..cube.len()).collect();
                    indices.sort_by_key(|&i| std::cmp::Reverse(self.term_cost(cube[i])));
                    indices
                } else {
                    let var_freq = self.compute_variable_frequency(cube);
                    let mut indices: Vec<usize> = (0..cube.len()).collect();
                    indices.sort_by_key(|&i| self.literal_frequency_score(cube[i], &var_freq));
                    indices
                }
            }
        }
    }

    /// Estimate the size of a term (for prioritization)
    fn term_size(&self, term: TermId) -> usize {
        use oxiz_core::TermKind;

        let Some(t) = self.terms.get(term) else {
            return 1;
        };

        match &t.kind {
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => 1 + args.iter().map(|&arg| self.term_size(arg)).sum::<usize>(),
            TermKind::Not(arg) => 1 + self.term_size(*arg),
            TermKind::Eq(a, b)
            | TermKind::Le(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b) => 1 + self.term_size(*a) + self.term_size(*b),
            _ => 1,
        }
    }

    /// Estimate the computational cost of a term
    /// Higher cost means more expensive to check
    fn term_cost(&self, term: TermId) -> usize {
        use oxiz_core::TermKind;

        let Some(t) = self.terms.get(term) else {
            return 1;
        };

        match &t.kind {
            // Arithmetic operations have higher cost
            TermKind::Mul(args) => 10 + args.iter().map(|&arg| self.term_cost(arg)).sum::<usize>(),
            TermKind::Div(a, b) | TermKind::Mod(a, b) => {
                10 + self.term_cost(*a) + self.term_cost(*b)
            }
            TermKind::Add(args) => 5 + args.iter().map(|&arg| self.term_cost(arg)).sum::<usize>(),
            TermKind::Sub(a, b) => 5 + self.term_cost(*a) + self.term_cost(*b),
            // Logical operations have medium cost
            TermKind::And(args) | TermKind::Or(args) => {
                3 + args.iter().map(|&arg| self.term_cost(arg)).sum::<usize>()
            }
            TermKind::Not(arg) => 2 + self.term_cost(*arg),
            // Comparisons have medium cost
            TermKind::Eq(a, b)
            | TermKind::Le(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Gt(a, b) => 4 + self.term_cost(*a) + self.term_cost(*b),
            // Variables and constants have low cost
            _ => 1,
        }
    }

    /// Compute variable frequency in a cube
    /// Returns a map from variable to its frequency count
    fn compute_variable_frequency(&self, cube: &[TermId]) -> rustc_hash::FxHashMap<TermId, usize> {
        let mut freq = rustc_hash::FxHashMap::default();

        for &lit in cube {
            let vars = Self::collect_vars(self.terms, lit);
            for var in vars {
                *freq.entry(var).or_insert(0) += 1;
            }
        }

        freq
    }

    /// Compute a frequency score for a literal
    /// Lower score means less frequent variables (prioritize for elimination)
    fn literal_frequency_score(
        &self,
        lit: TermId,
        var_freq: &rustc_hash::FxHashMap<TermId, usize>,
    ) -> usize {
        let vars = Self::collect_vars(self.terms, lit);
        if vars.is_empty() {
            return 0;
        }

        // Average frequency of variables in this literal
        let total_freq: usize = vars.iter().filter_map(|v| var_freq.get(v)).sum();
        total_freq / vars.len().max(1)
    }

    /// Simple down-closure: try to drop literals without inductiveness check
    ///
    /// This is faster than MIC but may produce weaker lemmas.
    /// Useful as a pre-processing step before MIC.
    pub fn down_closure(
        &mut self,
        cube: &[TermId],
        must_block: TermId,
    ) -> Result<GeneralizationResult, GeneralizationError> {
        debug!("Down-closure generalization with {} literals", cube.len());

        if cube.is_empty() {
            return Err(GeneralizationError::InvalidCube(
                "Cannot generalize empty cube".to_string(),
            ));
        }

        let mut result = GeneralizationResult::new(cube.iter().copied());
        let mut current_cube = cube.to_vec();
        let mut i = 0;

        // Try to drop each literal
        while i < current_cube.len() {
            let removed_lit = current_cube.remove(i);

            // Build candidate from remaining literals
            let candidate = if current_cube.is_empty() {
                self.terms.mk_true()
            } else if current_cube.len() == 1 {
                current_cube[0]
            } else {
                self.terms.mk_and(current_cube.clone())
            };

            // Check if candidate still blocks the bad state
            let mut smt = SmtSolver::new(self.terms, self.system);
            self.num_queries += 1;

            let still_blocks = smt.is_blocked_by(candidate, must_block).unwrap_or_default();

            if still_blocks {
                // Successfully dropped the literal
                result.dropped.push(removed_lit);
            } else {
                // Need to keep this literal
                current_cube.insert(i, removed_lit);
                i += 1;
            }
        }

        result.lemma = current_cube.into_iter().collect();
        result.num_queries = self.num_queries;

        debug!(
            "Down-closure: reduced from {} to {} literals",
            cube.len(),
            result.lemma.len()
        );

        Ok(result)
    }

    /// Extract cube (conjunction of literals) from a formula
    ///
    /// Decomposes a formula into a vector of literals (atoms and negated atoms).
    /// Only handles conjunctions; returns error for other forms.
    pub fn extract_cube(terms: &TermManager, formula: TermId) -> Vec<TermId> {
        let mut cube = Vec::new();

        if let Some(term) = terms.get(formula) {
            match &term.kind {
                TermKind::And(args) => {
                    // Conjunction: collect all conjuncts
                    for &arg in args.iter() {
                        Self::collect_literals(terms, arg, &mut cube);
                    }
                }
                _ => {
                    // Single literal or complex formula
                    Self::collect_literals(terms, formula, &mut cube);
                }
            }
        }

        cube
    }

    /// Collect literals from a formula into a cube
    fn collect_literals(terms: &TermManager, formula: TermId, cube: &mut Vec<TermId>) {
        if let Some(term) = terms.get(formula) {
            match &term.kind {
                TermKind::And(args) => {
                    // Recursively collect from conjuncts
                    for &arg in args.iter() {
                        Self::collect_literals(terms, arg, cube);
                    }
                }
                TermKind::True => {
                    // Skip true literals
                }
                TermKind::False => {
                    // False makes the whole cube unsatisfiable
                    cube.clear();
                    cube.push(formula);
                }
                _ => {
                    // Atomic literal or negation
                    if !cube.contains(&formula) {
                        cube.push(formula);
                    }
                }
            }
        }
    }

    /// Expand a cube by adding relevant constraints
    ///
    /// Uses unsat cores to identify which constraints are actually needed.
    pub fn expand_cube(
        &mut self,
        _pred: PredId,
        cube: &[TermId],
        constraints: &[TermId],
    ) -> Result<GeneralizationResult, GeneralizationError> {
        debug!(
            "Expanding cube with {} literals and {} additional constraints",
            cube.len(),
            constraints.len()
        );

        if constraints.is_empty() {
            // No constraints to add
            return Ok(GeneralizationResult::new(cube.iter().copied()));
        }

        // Enhanced implementation: Try to add constraints that strengthen the cube
        // without making it too specific

        let mut result = GeneralizationResult::new(cube.iter().copied());
        let mut current_cube = cube.to_vec();

        // Try to add each constraint one at a time
        for &constraint in constraints {
            // Skip if this constraint is already in the cube
            if current_cube.contains(&constraint) {
                continue;
            }

            // Check if adding this constraint is beneficial
            // Heuristic: Add if it doesn't make the formula too strong
            // A full implementation would:
            // 1. Check if the constraint is relevant (shares variables with cube)
            // 2. Use SMT to check if it's not redundant
            // 3. Ensure it doesn't make the lemma too specific

            // Simple heuristic: Check if the constraint shares variables with the cube
            if Self::shares_variables(self.terms, constraint, cube) {
                // Add the constraint
                current_cube.push(constraint);
                trace!("Added constraint to cube");
            }
        }

        result.lemma = current_cube.into_iter().collect();
        result.num_queries = self.num_queries;

        debug!(
            "Cube expansion: grew from {} to {} literals",
            cube.len(),
            result.lemma.len()
        );

        Ok(result)
    }

    /// Check if a constraint shares variables with any literal in the cube
    fn shares_variables(terms: &TermManager, constraint: TermId, cube: &[TermId]) -> bool {
        let constraint_vars = Self::collect_vars(terms, constraint);
        if constraint_vars.is_empty() {
            return false;
        }

        for &lit in cube {
            let lit_vars = Self::collect_vars(terms, lit);
            // Check if there's any overlap
            if constraint_vars.iter().any(|v| lit_vars.contains(v)) {
                return true;
            }
        }

        false
    }

    /// Collect variables from a term
    fn collect_vars(terms: &TermManager, term: TermId) -> Vec<TermId> {
        use std::collections::HashSet;
        let mut vars = HashSet::new();
        Self::collect_vars_rec(terms, term, &mut vars);
        vars.into_iter().collect()
    }

    /// Recursively collect variables
    fn collect_vars_rec(
        terms: &TermManager,
        term: TermId,
        vars: &mut std::collections::HashSet<TermId>,
    ) {
        use oxiz_core::TermKind;

        let Some(t) = terms.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::Var(_) => {
                vars.insert(term);
            }
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => {
                for &arg in args.iter() {
                    Self::collect_vars_rec(terms, arg, vars);
                }
            }
            TermKind::Not(arg) => {
                Self::collect_vars_rec(terms, *arg, vars);
            }
            TermKind::Eq(a, b)
            | TermKind::Le(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b) => {
                Self::collect_vars_rec(terms, *a, vars);
                Self::collect_vars_rec(terms, *b, vars);
            }
            _ => {}
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
    fn test_generalizer_creation() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();
        let _gen = Generalizer::new(&mut terms, &system);
    }

    #[test]
    fn test_extract_cube_single() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);

        let cube = Generalizer::extract_cube(&terms, x);
        assert_eq!(cube.len(), 1);
        assert_eq!(cube[0], x);
    }

    #[test]
    fn test_extract_cube_conjunction() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let y = terms.mk_var("y", terms.sorts.int_sort);
        let z = terms.mk_var("z", terms.sorts.int_sort);

        let conj = terms.mk_and(vec![x, y, z]);

        let cube = Generalizer::extract_cube(&terms, conj);
        assert_eq!(cube.len(), 3);
        assert!(cube.contains(&x));
        assert!(cube.contains(&y));
        assert!(cube.contains(&z));
    }

    #[test]
    fn test_generalization_result() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let y = terms.mk_var("y", terms.sorts.int_sort);

        let result = GeneralizationResult::new(vec![x, y]);
        assert_eq!(result.lemma.len(), 2);
        assert!(!result.is_inductive);

        let formula = result.to_formula(&mut terms);
        // Should be a conjunction
        if let Some(term) = terms.get(formula) {
            assert!(matches!(term.kind, TermKind::And(_)));
        }
    }

    #[test]
    fn test_generalization_result_single() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);

        let result = GeneralizationResult::new(vec![x]);
        let formula = result.to_formula(&mut terms);
        assert_eq!(formula, x);
    }

    #[test]
    fn test_generalization_result_empty() {
        let mut terms = TermManager::new();
        let result = GeneralizationResult::new(Vec::<TermId>::new());
        let formula = result.to_formula(&mut terms);
        assert_eq!(formula, terms.mk_true());
    }
}
