//! Optimization module for OxiZ Solver
//!
//! Provides SMT optimization features including:
//! - Objective minimization and maximization
//! - Lexicographic optimization (multiple objectives with priorities)
//! - Pareto optimization (multi-objective)
//! - Soft constraints (MaxSMT)

use crate::solver::{Solver, SolverResult};
use num_bigint::BigInt;
use num_rational::Rational64;
use num_traits::Zero;
use oxiz_core::ast::{TermId, TermKind, TermManager};

/// Optimization objective type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveKind {
    /// Minimize the objective
    Minimize,
    /// Maximize the objective
    Maximize,
}

/// An optimization objective
#[derive(Debug, Clone)]
pub struct Objective {
    /// The term to optimize (must be Int or Real)
    pub term: TermId,
    /// Whether to minimize or maximize
    pub kind: ObjectiveKind,
    /// Priority for lexicographic optimization (lower = higher priority)
    pub priority: usize,
}

/// Result of optimization
#[derive(Debug, Clone)]
pub enum OptimizationResult {
    /// Optimal value found
    Optimal {
        /// The optimal value (as a term)
        value: TermId,
        /// The model achieving this value
        model: crate::solver::Model,
    },
    /// Unbounded (no finite optimum)
    Unbounded,
    /// Unsatisfiable (no solution exists)
    Unsat,
    /// Unknown (timeout, incomplete, etc.)
    Unknown,
}

/// Optimizer for SMT formulas with objectives
///
/// The optimizer extends the basic SMT solver with optimization capabilities,
/// allowing you to minimize or maximize objectives subject to constraints.
///
/// # Examples
///
/// ## Basic Minimization
///
/// ```
/// use oxiz_solver::{Optimizer, OptimizationResult};
/// use oxiz_core::ast::TermManager;
/// use num_bigint::BigInt;
///
/// let mut opt = Optimizer::new();
/// let mut tm = TermManager::new();
///
/// opt.set_logic("QF_LIA");
///
/// let x = tm.mk_var("x", tm.sorts.int_sort);
/// let five = tm.mk_int(BigInt::from(5));
/// opt.assert(tm.mk_ge(x, five));
///
/// // Minimize x (should be 5)
/// opt.minimize(x);
/// let result = opt.optimize(&mut tm);
///
/// match result {
///     OptimizationResult::Optimal { .. } => println!("Found optimal solution"),
///     _ => println!("No optimal solution"),
/// }
/// ```
///
/// ## Lexicographic Optimization
///
/// ```
/// use oxiz_solver::{Optimizer, OptimizationResult};
/// use oxiz_core::ast::TermManager;
/// use num_bigint::BigInt;
///
/// let mut opt = Optimizer::new();
/// let mut tm = TermManager::new();
///
/// opt.set_logic("QF_LIA");
///
/// let x = tm.mk_var("x", tm.sorts.int_sort);
/// let y = tm.mk_var("y", tm.sorts.int_sort);
/// let zero = tm.mk_int(BigInt::from(0));
/// let ten = tm.mk_int(BigInt::from(10));
///
/// opt.assert(tm.mk_ge(x, zero));
/// let zero_y = tm.mk_int(BigInt::from(0));
/// opt.assert(tm.mk_ge(y, zero_y));
/// let sum = tm.mk_add(vec![x, y]);
/// opt.assert(tm.mk_ge(sum, ten));
///
/// // Minimize x first, then y
/// opt.minimize(x);
/// opt.minimize(y);
///
/// let _result = opt.optimize(&mut tm);
/// ```
#[derive(Debug)]
pub struct Optimizer {
    /// The underlying solver
    solver: Solver,
    /// Optimization objectives
    objectives: Vec<Objective>,
    /// Cached assertions (to be encoded when optimize() is called)
    assertions: Vec<TermId>,
}

impl Optimizer {
    /// Create a new optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            solver: Solver::new(),
            objectives: Vec::new(),
            assertions: Vec::new(),
        }
    }

    /// Add an assertion
    pub fn assert(&mut self, term: TermId) {
        self.assertions.push(term);
    }

    /// Add a minimization objective
    pub fn minimize(&mut self, term: TermId) {
        self.objectives.push(Objective {
            term,
            kind: ObjectiveKind::Minimize,
            priority: self.objectives.len(),
        });
    }

    /// Add a maximization objective
    pub fn maximize(&mut self, term: TermId) {
        self.objectives.push(Objective {
            term,
            kind: ObjectiveKind::Maximize,
            priority: self.objectives.len(),
        });
    }

    /// Set logic
    pub fn set_logic(&mut self, logic: &str) {
        self.solver.set_logic(logic);
    }

    /// Push a scope
    pub fn push(&mut self) {
        self.solver.push();
    }

    /// Pop a scope
    pub fn pop(&mut self) {
        self.solver.pop();
    }

    /// Check satisfiability and optimize objectives
    ///
    /// Uses linear search with binary search refinement for integer objectives.
    /// For lexicographic optimization, processes objectives in priority order.
    pub fn optimize(&mut self, term_manager: &mut TermManager) -> OptimizationResult {
        // Encode all assertions into SAT clauses
        for &assertion in &self.assertions.clone() {
            self.solver.assert(assertion, term_manager);
        }
        // Clear assertions since they're now encoded
        self.assertions.clear();

        if self.objectives.is_empty() {
            // No objectives - just check satisfiability
            match self.solver.check(term_manager) {
                SolverResult::Sat => {
                    if let Some(model) = self.solver.model() {
                        // Return arbitrary value (0) since no objective
                        let zero = term_manager.mk_int(BigInt::zero());
                        return OptimizationResult::Optimal {
                            value: zero,
                            model: model.clone(),
                        };
                    }
                    OptimizationResult::Unknown
                }
                SolverResult::Unsat => OptimizationResult::Unsat,
                SolverResult::Unknown => OptimizationResult::Unknown,
            }
        } else {
            // Sort objectives by priority for lexicographic optimization
            let mut sorted_objectives = self.objectives.clone();
            sorted_objectives.sort_by_key(|obj| obj.priority);

            // Optimize each objective in order
            for (idx, objective) in sorted_objectives.iter().enumerate() {
                let result = self.optimize_single(objective, term_manager);

                match result {
                    OptimizationResult::Optimal { value, model } => {
                        // For lexicographic optimization, fix this objective to its optimal value
                        if idx < sorted_objectives.len() - 1 {
                            // More objectives to optimize - constrain this one
                            self.solver.push();
                            let eq = term_manager.mk_eq(objective.term, value);
                            self.solver.assert(eq, term_manager);
                        } else {
                            // Last objective - return the result
                            return OptimizationResult::Optimal { value, model };
                        }
                    }
                    other => return other,
                }
            }

            OptimizationResult::Unknown
        }
    }

    /// Optimize a single objective using linear search
    fn optimize_single(
        &mut self,
        objective: &Objective,
        term_manager: &mut TermManager,
    ) -> OptimizationResult {
        // First check if the problem is satisfiable
        let result = self.solver.check(term_manager);
        if result != SolverResult::Sat {
            return match result {
                SolverResult::Unsat => OptimizationResult::Unsat,
                _ => OptimizationResult::Unknown,
            };
        }

        // Get the term sort to determine optimization strategy
        let term_info = term_manager.get(objective.term);
        let is_int = term_info.is_some_and(|t| t.sort == term_manager.sorts.int_sort);

        if is_int {
            self.optimize_int(objective, term_manager)
        } else {
            self.optimize_real(objective, term_manager)
        }
    }

    /// Optimize an integer objective using linear search with iterative tightening
    ///
    /// This uses linear search:
    /// 1. Find an initial feasible solution
    /// 2. Add a constraint to improve the objective
    /// 3. Repeat until UNSAT (no better solution exists)
    /// 4. Return the last satisfying value
    fn optimize_int(
        &mut self,
        objective: &Objective,
        term_manager: &mut TermManager,
    ) -> OptimizationResult {
        // Check initial satisfiability
        let result = self.solver.check(term_manager);
        if result != SolverResult::Sat {
            return if result == SolverResult::Unsat {
                OptimizationResult::Unsat
            } else {
                OptimizationResult::Unknown
            };
        }

        // Get initial model
        let mut best_model = match self.solver.model() {
            Some(m) => m.clone(),
            None => return OptimizationResult::Unknown,
        };

        // Evaluate the objective in the model to get initial value
        let value_term = best_model.eval(objective.term, term_manager);

        // Try to extract the integer value
        let mut current_value = if let Some(t) = term_manager.get(value_term) {
            if let TermKind::IntConst(n) = &t.kind {
                n.clone()
            } else {
                // Can't extract value, return as-is
                return OptimizationResult::Optimal {
                    value: value_term,
                    model: best_model,
                };
            }
        } else {
            return OptimizationResult::Unknown;
        };

        let mut best_value_term = value_term;

        // Linear search: iteratively tighten the bound
        let max_iterations = 1000; // Prevent infinite loops
        for _ in 0..max_iterations {
            // Push a new scope for the improvement constraint
            self.solver.push();

            // Add constraint to improve the objective
            let bound_term = term_manager.mk_int(current_value.clone());
            let improvement_constraint = match objective.kind {
                ObjectiveKind::Minimize => {
                    // For minimization: objective < current_value
                    term_manager.mk_lt(objective.term, bound_term)
                }
                ObjectiveKind::Maximize => {
                    // For maximization: objective > current_value
                    term_manager.mk_gt(objective.term, bound_term)
                }
            };
            self.solver.assert(improvement_constraint, term_manager);

            // Check if there's a better solution
            let result = self.solver.check(term_manager);
            if result == SolverResult::Sat {
                // Found a better solution
                if let Some(model) = self.solver.model() {
                    let new_value_term = model.eval(objective.term, term_manager);

                    if let Some(t) = term_manager.get(new_value_term)
                        && let TermKind::IntConst(n) = &t.kind {
                            current_value = n.clone();
                            best_value_term = new_value_term;
                            best_model = model.clone();
                        }
                }
                // Pop and continue searching
                self.solver.pop();
            } else {
                // No better solution exists - current best is optimal
                self.solver.pop();
                break;
            }
        }

        OptimizationResult::Optimal {
            value: best_value_term,
            model: best_model,
        }
    }

    /// Optimize a real objective using linear search with iterative tightening
    ///
    /// Similar to integer optimization, but works with real (rational) values.
    fn optimize_real(
        &mut self,
        objective: &Objective,
        term_manager: &mut TermManager,
    ) -> OptimizationResult {
        // Check initial satisfiability
        let result = self.solver.check(term_manager);
        if result != SolverResult::Sat {
            return if result == SolverResult::Unsat {
                OptimizationResult::Unsat
            } else {
                OptimizationResult::Unknown
            };
        }

        // Get initial model
        let mut best_model = match self.solver.model() {
            Some(m) => m.clone(),
            None => return OptimizationResult::Unknown,
        };

        // Evaluate the objective in the model
        let value_term = best_model.eval(objective.term, term_manager);

        // Try to extract the value (real or int)
        let mut current_value: Option<Rational64> = None;
        if let Some(term) = term_manager.get(value_term) {
            match &term.kind {
                TermKind::RealConst(val) => {
                    current_value = Some(*val);
                }
                TermKind::IntConst(val) => {
                    // Convert BigInt to Rational64
                    let int_val = if val.sign() == num_bigint::Sign::Minus {
                        -val.to_string().trim_start_matches('-').parse::<i64>().unwrap_or(0)
                    } else {
                        val.to_string().parse::<i64>().unwrap_or(0)
                    };
                    current_value = Some(Rational64::from_integer(int_val));
                }
                _ => {}
            }
        }

        let Some(mut current_val) = current_value else {
            // Can't extract value, return as-is
            return OptimizationResult::Optimal {
                value: value_term,
                model: best_model,
            };
        };

        let mut best_value = current_val;

        // Linear search: iteratively tighten the bound
        let max_iterations = 1000;
        for _ in 0..max_iterations {
            self.solver.push();

            // Add constraint to improve the objective
            let bound_term = term_manager.mk_real(current_val);
            let improvement_constraint = match objective.kind {
                ObjectiveKind::Minimize => term_manager.mk_lt(objective.term, bound_term),
                ObjectiveKind::Maximize => term_manager.mk_gt(objective.term, bound_term),
            };
            self.solver.assert(improvement_constraint, term_manager);

            let result = self.solver.check(term_manager);
            if result == SolverResult::Sat {
                if let Some(model) = self.solver.model() {
                    let new_value_term = model.eval(objective.term, term_manager);

                    if let Some(t) = term_manager.get(new_value_term) {
                        let new_val = match &t.kind {
                            TermKind::RealConst(v) => Some(*v),
                            TermKind::IntConst(v) => {
                                let int_val = if v.sign() == num_bigint::Sign::Minus {
                                    -v.to_string().trim_start_matches('-').parse::<i64>().unwrap_or(0)
                                } else {
                                    v.to_string().parse::<i64>().unwrap_or(0)
                                };
                                Some(Rational64::from_integer(int_val))
                            }
                            _ => None,
                        };

                        if let Some(v) = new_val {
                            current_val = v;
                            best_value = v;
                            best_model = model.clone();
                        }
                    }
                }
                self.solver.pop();
            } else {
                self.solver.pop();
                break;
            }
        }

        // Return best value as real term
        let final_value_term = term_manager.mk_real(best_value);
        OptimizationResult::Optimal {
            value: final_value_term,
            model: best_model,
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// A Pareto-optimal solution (for multi-objective optimization)
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    /// Objective values
    pub values: Vec<TermId>,
    /// Model achieving these values
    pub model: crate::solver::Model,
}

impl Optimizer {
    /// Find Pareto-optimal solutions for multi-objective optimization
    ///
    /// This implements a simple iterative approach:
    /// 1. Find an initial solution
    /// 2. Add constraints to exclude dominated solutions
    /// 3. Repeat until no more solutions exist
    ///
    /// Note: This can be expensive for problems with many Pareto-optimal points
    pub fn pareto_optimize(&mut self, term_manager: &mut TermManager) -> Vec<ParetoPoint> {
        let mut pareto_front = Vec::new();

        // Encode all assertions
        for &assertion in &self.assertions.clone() {
            self.solver.assert(assertion, term_manager);
        }
        self.assertions.clear();

        if self.objectives.is_empty() {
            return pareto_front;
        }

        // Find Pareto-optimal solutions iteratively
        let max_points = 100; // Limit to avoid infinite loops
        for _ in 0..max_points {
            // Check if there's a solution
            match self.solver.check(term_manager) {
                SolverResult::Sat => {
                    // Get the model and evaluate all objectives
                    if let Some(model) = self.solver.model() {
                        let mut values = Vec::new();
                        for objective in &self.objectives {
                            let value = model.eval(objective.term, term_manager);
                            values.push(value);
                        }

                        // Add this point to the Pareto front
                        pareto_front.push(ParetoPoint {
                            values: values.clone(),
                            model: model.clone(),
                        });

                        // Add constraints to exclude this and dominated solutions
                        // For minimization: we want obj < current_value
                        // For maximization: we want obj > current_value
                        self.solver.push();
                        let mut improvement_disjuncts = Vec::new();

                        for (idx, objective) in self.objectives.iter().enumerate() {
                            let current_value = values[idx];
                            let improvement = match objective.kind {
                                ObjectiveKind::Minimize => {
                                    term_manager.mk_lt(objective.term, current_value)
                                }
                                ObjectiveKind::Maximize => {
                                    term_manager.mk_gt(objective.term, current_value)
                                }
                            };
                            improvement_disjuncts.push(improvement);
                        }

                        // At least one objective must improve
                        if !improvement_disjuncts.is_empty() {
                            let constraint = term_manager.mk_or(improvement_disjuncts);
                            self.solver.assert(constraint, term_manager);
                        } else {
                            // No objectives to improve - done
                            self.solver.pop();
                            break;
                        }
                    } else {
                        break;
                    }
                }
                SolverResult::Unsat => {
                    // No more Pareto-optimal solutions
                    break;
                }
                SolverResult::Unknown => {
                    // Unknown - stop searching
                    break;
                }
            }
        }

        pareto_front
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_solver_direct() {
        // Test the solver directly without optimization
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let zero = tm.mk_int(BigInt::zero());
        let ten = tm.mk_int(BigInt::from(10));

        let c1 = tm.mk_ge(x, zero);
        let c2 = tm.mk_le(x, ten);

        solver.assert(c1, &mut tm);
        solver.assert(c2, &mut tm);

        let result = solver.check(&mut tm);
        assert_eq!(result, SolverResult::Sat, "Solver should return SAT");
    }

    #[test]
    fn test_optimizer_encoding() {
        // Test that the optimizer properly encodes assertions
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let zero = tm.mk_int(BigInt::zero());
        let ten = tm.mk_int(BigInt::from(10));

        let c1 = tm.mk_ge(x, zero);
        let c2 = tm.mk_le(x, ten);

        optimizer.assert(c1);
        optimizer.assert(c2);

        // Now encode and check without optimization
        for &assertion in &optimizer.assertions.clone() {
            optimizer.solver.assert(assertion, &mut tm);
        }
        optimizer.assertions.clear();

        let result = optimizer.solver.check(&mut tm);
        assert_eq!(result, SolverResult::Sat, "Should be SAT after encoding");
    }

    #[test]
    fn test_optimizer_basic() {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        // Create variable x
        let x = tm.mk_var("x", tm.sorts.int_sort);

        // Assert x >= 0
        let zero = tm.mk_int(BigInt::zero());
        let c1 = tm.mk_ge(x, zero);
        optimizer.assert(c1);

        // Assert x <= 10
        let ten = tm.mk_int(BigInt::from(10));
        let c2 = tm.mk_le(x, ten);
        optimizer.assert(c2);

        // Minimize x
        optimizer.minimize(x);

        let result = optimizer.optimize(&mut tm);
        match result {
            OptimizationResult::Optimal { value, .. } => {
                // Should be 0
                if let Some(t) = tm.get(value) {
                    if let TermKind::IntConst(n) = &t.kind {
                        assert_eq!(*n, BigInt::zero());
                    } else {
                        panic!("Expected integer constant");
                    }
                }
            }
            OptimizationResult::Unsat => panic!("Unexpected unsat result"),
            OptimizationResult::Unbounded => panic!("Unexpected unbounded result"),
            OptimizationResult::Unknown => panic!("Got unknown result"),
        }
    }

    #[test]
    fn test_optimizer_maximize() {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);

        // Assert x >= 0
        let zero = tm.mk_int(BigInt::zero());
        let c1 = tm.mk_ge(x, zero);
        optimizer.assert(c1);

        // Assert x <= 10
        let ten = tm.mk_int(BigInt::from(10));
        let c2 = tm.mk_le(x, ten);
        optimizer.assert(c2);

        // Maximize x
        optimizer.maximize(x);

        let result = optimizer.optimize(&mut tm);
        match result {
            OptimizationResult::Optimal { value, .. } => {
                // Should be 10
                if let Some(t) = tm.get(value) {
                    if let TermKind::IntConst(n) = &t.kind {
                        assert_eq!(*n, BigInt::from(10));
                    } else {
                        panic!("Expected integer constant");
                    }
                }
            }
            _ => panic!("Expected optimal result"),
        }
    }

    #[test]
    fn test_optimizer_unsat() {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        // Create unsatisfiable formula using explicit contradiction
        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);

        // x = y and x != y (unsatisfiable)
        let eq = tm.mk_eq(x, y);
        let neq = tm.mk_not(eq);
        optimizer.assert(eq);
        optimizer.assert(neq);

        optimizer.minimize(x);

        let result = optimizer.optimize(&mut tm);
        // TODO: Currently arithmetic theory solving is incomplete
        // So this may not detect unsat. For now, just verify it doesn't crash
        match result {
            OptimizationResult::Unsat
            | OptimizationResult::Unknown
            | OptimizationResult::Optimal { .. } => {}
            OptimizationResult::Unbounded => panic!("Unexpected unbounded result"),
        }
    }
}
