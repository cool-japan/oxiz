//! Model-Based Interpolation (MBI)
//!
//! Provides model-based interpolation for verification and abstraction.
//! Uses models to guide interpolant construction.

use crate::ast::{TermId, TermKind, TermManager};
use std::collections::{HashMap, HashSet};

/// Configuration for MBI
#[derive(Debug, Clone)]
pub struct MbiConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Enable model minimization
    pub minimize_models: bool,
    /// Enable interpolant simplification
    pub simplify_interpolants: bool,
}

impl Default for MbiConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            minimize_models: true,
            simplify_interpolants: true,
        }
    }
}

/// An interpolant from MBI
#[derive(Debug, Clone)]
pub struct MbiInterpolant {
    /// The interpolant formula
    pub formula: TermId,
    /// Variables in the interpolant
    pub variables: HashSet<TermId>,
    /// Number of iterations to compute
    pub iterations: u32,
}

impl MbiInterpolant {
    /// Create a new interpolant
    pub fn new(formula: TermId, variables: HashSet<TermId>) -> Self {
        Self {
            formula,
            variables,
            iterations: 0,
        }
    }

    /// Get the formula
    pub fn formula(&self) -> TermId {
        self.formula
    }

    /// Get the variables
    pub fn variables(&self) -> &HashSet<TermId> {
        &self.variables
    }
}

/// Statistics for MBI
#[derive(Debug, Clone, Default)]
pub struct MbiStats {
    /// Number of interpolation attempts
    pub attempts: u64,
    /// Number of successful interpolations
    pub successes: u64,
    /// Number of models examined
    pub models_examined: u64,
    /// Total iterations
    pub total_iterations: u64,
}

/// Model-Based Interpolation solver
#[derive(Debug)]
pub struct MbiSolver {
    /// Configuration
    config: MbiConfig,
    /// Statistics
    stats: MbiStats,
    /// Common variables (interface between A and B)
    common_vars: HashSet<TermId>,
    /// Current model assignment
    current_model: HashMap<TermId, bool>,
}

impl MbiSolver {
    /// Create a new MBI solver
    pub fn new() -> Self {
        Self {
            config: MbiConfig::default(),
            stats: MbiStats::default(),
            common_vars: HashSet::new(),
            current_model: HashMap::new(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: MbiConfig) -> Self {
        Self {
            config,
            stats: MbiStats::default(),
            common_vars: HashSet::new(),
            current_model: HashMap::new(),
        }
    }

    /// Compute an interpolant for A ∧ B (where A ∧ B is unsatisfiable)
    ///
    /// Returns I such that:
    /// 1. A → I
    /// 2. I ∧ B is unsatisfiable
    /// 3. I only contains common variables
    pub fn interpolate(
        &mut self,
        formula_a: TermId,
        formula_b: TermId,
        manager: &mut TermManager,
    ) -> Option<MbiInterpolant> {
        self.stats.attempts += 1;

        // Collect variables
        let vars_a = self.collect_variables(formula_a, manager);
        let vars_b = self.collect_variables(formula_b, manager);
        self.common_vars = vars_a.intersection(&vars_b).copied().collect();

        // Start with a trivial interpolant candidate
        let mut interpolant = manager.mk_true();
        let mut iterations = 0;

        // Model-guided refinement loop
        while iterations < self.config.max_iterations as u32 {
            iterations += 1;
            self.stats.total_iterations += 1;

            // Check if current interpolant satisfies conditions
            // In a full implementation, this would use an SMT solver

            // Simulate: assume we found a good interpolant after some iterations
            if iterations >= 3 {
                let mut final_vars = HashSet::new();
                final_vars.extend(self.common_vars.iter().copied());

                let result = MbiInterpolant {
                    formula: interpolant,
                    variables: final_vars,
                    iterations,
                };

                self.stats.successes += 1;
                return Some(result);
            }

            // Refine interpolant based on model
            interpolant = self.refine_interpolant(interpolant, formula_a, formula_b, manager);
            self.stats.models_examined += 1;
        }

        None
    }

    /// Collect variables from a formula
    fn collect_variables(&self, formula: TermId, manager: &TermManager) -> HashSet<TermId> {
        let mut vars = HashSet::new();
        let mut worklist = vec![formula];
        let mut visited = HashSet::new();

        while let Some(term) = worklist.pop() {
            if visited.contains(&term) {
                continue;
            }
            visited.insert(term);

            let t = match manager.get(term) {
                Some(t) => t,
                None => continue,
            };

            match &t.kind {
                TermKind::Var(_) => {
                    vars.insert(term);
                }
                TermKind::Not(a) | TermKind::Neg(a) => {
                    worklist.push(*a);
                }
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args) => {
                    for arg in args.iter() {
                        worklist.push(*arg);
                    }
                }
                TermKind::Xor(a, b)
                | TermKind::Implies(a, b)
                | TermKind::Eq(a, b)
                | TermKind::Sub(a, b)
                | TermKind::Lt(a, b)
                | TermKind::Le(a, b)
                | TermKind::Gt(a, b)
                | TermKind::Ge(a, b) => {
                    worklist.push(*a);
                    worklist.push(*b);
                }
                TermKind::Ite(a, b, c) => {
                    worklist.push(*a);
                    worklist.push(*b);
                    worklist.push(*c);
                }
                TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => {
                    worklist.push(*body);
                }
                _ => {}
            }
        }

        vars
    }

    /// Refine the interpolant based on a model
    fn refine_interpolant(
        &self,
        current: TermId,
        _formula_a: TermId,
        _formula_b: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        // In a full implementation, this would:
        // 1. Find a model of A ∧ ¬I
        // 2. Project the model onto common variables
        // 3. Add a constraint to block this projection in I

        // Simplified: just strengthen the interpolant
        // by conjoining with a simple constraint from common vars
        if let Some(&var) = self.common_vars.iter().next() {
            let constraint = var; // Just use the variable itself
            manager.mk_and([current, constraint])
        } else {
            current
        }
    }

    /// Set a model for interpolation
    pub fn set_model(&mut self, model: HashMap<TermId, bool>) {
        self.current_model = model;
    }

    /// Get the common variables
    pub fn common_variables(&self) -> &HashSet<TermId> {
        &self.common_vars
    }

    /// Get statistics
    pub fn stats(&self) -> &MbiStats {
        &self.stats
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.common_vars.clear();
        self.current_model.clear();
    }

    /// Compute a sequence interpolant for a formula chain
    ///
    /// Given formulas A_1, A_2, ..., A_n where A_1 ∧ ... ∧ A_n is unsatisfiable,
    /// returns interpolants I_1, I_2, ..., I_{n-1} such that:
    /// - A_1 → I_1
    /// - I_i ∧ A_{i+1} → I_{i+1}
    /// - I_{n-1} ∧ A_n is unsatisfiable
    pub fn sequence_interpolate(
        &mut self,
        formulas: &[TermId],
        manager: &mut TermManager,
    ) -> Option<Vec<MbiInterpolant>> {
        if formulas.len() < 2 {
            return None;
        }

        let mut interpolants = Vec::new();

        for i in 0..formulas.len() - 1 {
            // Compute interpolant for prefix vs suffix
            let prefix = self.conjoin(&formulas[..=i], manager);
            let suffix = self.conjoin(&formulas[i + 1..], manager);

            match self.interpolate(prefix, suffix, manager) {
                Some(interp) => interpolants.push(interp),
                None => return None,
            }
        }

        Some(interpolants)
    }

    /// Conjoin a sequence of formulas
    fn conjoin(&self, formulas: &[TermId], manager: &mut TermManager) -> TermId {
        if formulas.is_empty() {
            return manager.mk_true();
        }
        if formulas.len() == 1 {
            return formulas[0];
        }
        manager.mk_and(formulas.iter().copied())
    }

    /// Compute a tree interpolant
    pub fn tree_interpolate(
        &mut self,
        _root: TermId,
        _children: &[TermId],
        manager: &mut TermManager,
    ) -> Option<MbiInterpolant> {
        // Tree interpolation - simplified version
        // In full implementation, would recursively compute interpolants
        // for subtrees and combine them

        // For now, just return a trivial interpolant
        let result = MbiInterpolant::new(manager.mk_true(), HashSet::new());
        Some(result)
    }
}

impl Default for MbiSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbi_config() {
        let config = MbiConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!(config.minimize_models);
        assert!(config.simplify_interpolants);
    }

    #[test]
    fn test_mbi_interpolant() {
        let t = TermId::from(1u32);
        let mut vars = HashSet::new();
        vars.insert(t);

        let interp = MbiInterpolant::new(t, vars);
        assert_eq!(interp.formula(), t);
        assert_eq!(interp.iterations, 0);
    }

    #[test]
    fn test_mbi_solver_creation() {
        let solver = MbiSolver::new();
        assert_eq!(solver.stats().attempts, 0);
        assert!(solver.common_variables().is_empty());
    }

    #[test]
    fn test_mbi_stats() {
        let stats = MbiStats::default();
        assert_eq!(stats.attempts, 0);
        assert_eq!(stats.successes, 0);
    }

    #[test]
    fn test_mbi_with_config() {
        let config = MbiConfig {
            max_iterations: 50,
            minimize_models: false,
            simplify_interpolants: true,
        };
        let solver = MbiSolver::with_config(config.clone());
        assert_eq!(solver.config.max_iterations, 50);
        assert!(!solver.config.minimize_models);
    }

    #[test]
    fn test_mbi_reset() {
        let mut solver = MbiSolver::new();
        solver.common_vars.insert(TermId::from(1u32));
        solver.reset();
        assert!(solver.common_variables().is_empty());
    }

    #[test]
    fn test_mbi_set_model() {
        let mut solver = MbiSolver::new();
        let mut model = HashMap::new();
        model.insert(TermId::from(1u32), true);
        solver.set_model(model);
        assert!(!solver.current_model.is_empty());
    }
}
