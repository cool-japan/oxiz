//! Non-linear Integer Arithmetic (NIA) solver.
//!
//! This module extends the NLSAT solver with integer constraints using
//! branch-and-bound and cutting planes.
//!
//! ## Key Components
//!
//! - **Branch and Bound**: Enumerate integer solutions
//! - **Cutting Planes**: Add constraints to eliminate non-integer solutions
//! - **Mixed Constraints**: Combine real and integer variables
//!
//! ## Reference
//!
//! - Z3's NIA solver in `nlsat/nlsat_solver.cpp`
//! - Branch-and-bound for mixed integer non-linear programming (MINLP)

use crate::cutting_planes::CuttingPlaneGenerator;
use crate::solver::{Model, NlsatSolver, SolverResult};
use crate::types::AtomKind;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};
use oxiz_math::polynomial::{Polynomial, Var};
use std::collections::HashSet;

/// Integer variable type specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    /// Real-valued variable (no integer constraint).
    Real,
    /// Integer-valued variable.
    Integer,
}

/// Configuration for NIA solver.
#[derive(Debug, Clone)]
pub struct NiaConfig {
    /// Maximum number of branch-and-bound nodes to explore.
    pub max_nodes: usize,
    /// Maximum depth of the branch-and-bound tree.
    pub max_depth: usize,
    /// Enable cutting planes.
    pub enable_cutting_planes: bool,
    /// Branching variable selection strategy.
    pub branching_strategy: BranchingStrategy,
    /// Tolerance for integer proximity (values within this of an integer are rounded).
    pub int_tolerance: f64,
}

impl Default for NiaConfig {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            max_depth: 100,
            enable_cutting_planes: true,
            branching_strategy: BranchingStrategy::MostFractional,
            int_tolerance: 1e-6,
        }
    }
}

/// Strategy for selecting which variable to branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingStrategy {
    /// Branch on variable with most fractional value.
    MostFractional,
    /// Branch on variable with least fractional value.
    LeastFractional,
    /// Branch on variable with smallest domain.
    SmallestDomain,
    /// Branch on first fractional variable found.
    FirstFractional,
}

/// Branch-and-bound node.
#[derive(Debug, Clone)]
struct BranchNode {
    /// Decision level in the solver.
    level: u32,
    /// Variables that have been branched on.
    branched_vars: HashSet<Var>,
    /// Current depth in the tree.
    depth: usize,
}

/// Non-linear integer arithmetic solver.
///
/// Extends NLSAT with integer constraints using branch-and-bound.
pub struct NiaSolver {
    /// Underlying NLSAT solver for real arithmetic.
    nlsat: NlsatSolver,
    /// Variable types (Real or Integer).
    var_types: Vec<VarType>,
    /// NIA-specific configuration.
    config: NiaConfig,
    /// Statistics.
    stats: NiaStats,
    /// Cutting plane generator.
    cut_generator: CuttingPlaneGenerator,
}

/// Statistics for NIA solver.
#[derive(Debug, Clone, Default)]
pub struct NiaStats {
    /// Number of branch-and-bound nodes explored.
    pub nodes_explored: usize,
    /// Number of cutting planes added.
    pub cutting_planes: usize,
    /// Number of integer solutions found.
    pub integer_solutions: usize,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
}

impl NiaSolver {
    /// Create a new NIA solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(NiaConfig::default())
    }

    /// Create a new NIA solver with custom configuration.
    pub fn with_config(config: NiaConfig) -> Self {
        Self {
            nlsat: NlsatSolver::new(),
            var_types: Vec::new(),
            config,
            stats: NiaStats::default(),
            cut_generator: CuttingPlaneGenerator::new(),
        }
    }

    /// Get the underlying NLSAT solver.
    pub fn nlsat(&self) -> &NlsatSolver {
        &self.nlsat
    }

    /// Get mutable reference to underlying NLSAT solver.
    pub fn nlsat_mut(&mut self) -> &mut NlsatSolver {
        &mut self.nlsat
    }

    /// Set variable type (Real or Integer).
    pub fn set_var_type(&mut self, var: Var, var_type: VarType) {
        // Ensure we have enough space
        while self.var_types.len() <= var as usize {
            self.var_types.push(VarType::Real);
        }
        self.var_types[var as usize] = var_type;
    }

    /// Get variable type.
    pub fn var_type(&self, var: Var) -> VarType {
        self.var_types
            .get(var as usize)
            .copied()
            .unwrap_or(VarType::Real)
    }

    /// Check if a variable is integer-typed.
    pub fn is_integer_var(&self, var: Var) -> bool {
        self.var_type(var) == VarType::Integer
    }

    /// Solve with integer constraints.
    ///
    /// Uses branch-and-bound to find integer solutions.
    pub fn solve(&mut self) -> SolverResult {
        // Reset statistics
        self.stats = NiaStats::default();

        // First solve the real relaxation
        let real_result = self.nlsat.solve();

        match real_result {
            SolverResult::Unsat => {
                // Real relaxation is UNSAT, so integer problem is also UNSAT
                SolverResult::Unsat
            }
            SolverResult::Unknown => SolverResult::Unknown,
            SolverResult::Sat => {
                // Check if the solution satisfies integer constraints
                if let Some(model) = self.nlsat.get_model() {
                    if self.is_integer_solution(&model) {
                        // Lucky! Real solution is already integer
                        self.stats.integer_solutions += 1;
                        return SolverResult::Sat;
                    }

                    // Need to branch
                    return self.branch_and_bound();
                }
                SolverResult::Unknown
            }
        }
    }

    /// Branch-and-bound search for integer solutions.
    fn branch_and_bound(&mut self) -> SolverResult {
        let root_node = BranchNode {
            level: 0,
            branched_vars: HashSet::new(),
            depth: 0,
        };

        let mut stack = vec![root_node];

        while let Some(node) = stack.pop() {
            self.stats.nodes_explored += 1;
            self.stats.max_depth_reached = self.stats.max_depth_reached.max(node.depth);

            // Check limits
            if self.stats.nodes_explored >= self.config.max_nodes {
                return SolverResult::Unknown;
            }
            if node.depth >= self.config.max_depth {
                continue; // Prune this branch
            }

            // Solve at this node
            let result = self.nlsat.solve();

            match result {
                SolverResult::Unsat => {
                    // This branch is infeasible - backtrack
                    continue;
                }
                SolverResult::Unknown => {
                    // Inconclusive - try other branches
                    continue;
                }
                SolverResult::Sat => {
                    // Check if solution is integer
                    if let Some(model) = self.nlsat.get_model() {
                        if self.is_integer_solution(&model) {
                            // Found an integer solution!
                            self.stats.integer_solutions += 1;
                            return SolverResult::Sat;
                        }

                        // Solution is not integer - need to branch
                        if let Some(branch_var) = self.select_branching_variable(&model, &node) {
                            // Get current value of branch variable
                            if let Some(value) = model.arith_value(branch_var) {
                                // Create two branches: x <= floor(value) and x >= ceil(value)
                                let (floor_val, ceil_val) = self.floor_ceil(value);

                                // Push ceil branch first (stack is LIFO, so will explore floor first)
                                if ceil_val > floor_val {
                                    self.create_branch(
                                        &mut stack, &node, branch_var, &ceil_val,
                                        true, // >= ceil
                                    );
                                }

                                // Push floor branch
                                self.create_branch(
                                    &mut stack, &node, branch_var, &floor_val,
                                    false, // <= floor
                                );
                            }
                        } else {
                            // No variable to branch on but solution is not integer
                            // This shouldn't happen if var_types is set correctly
                            continue;
                        }
                    }
                }
            }
        }

        // Exhausted search space without finding integer solution
        SolverResult::Unsat
    }

    /// Select which variable to branch on.
    fn select_branching_variable(&self, model: &Model, node: &BranchNode) -> Option<Var> {
        let mut candidates: Vec<(Var, BigRational, f64)> = Vec::new();

        for var in 0..self.nlsat.num_arith_vars() {
            // Skip if already branched
            if node.branched_vars.contains(&var) {
                continue;
            }

            // Skip if not integer-typed
            if !self.is_integer_var(var) {
                continue;
            }

            // Get value from model
            if let Some(value) = model.arith_value(var) {
                let frac = self.fractional_part(value);
                if frac > self.config.int_tolerance && frac < (1.0 - self.config.int_tolerance) {
                    candidates.push((var, value.clone(), frac));
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Select based on strategy
        match self.config.branching_strategy {
            BranchingStrategy::MostFractional => {
                // Pick variable with fractional part closest to 0.5
                candidates.sort_by(|a, b| {
                    let dist_a = (a.2 - 0.5).abs();
                    let dist_b = (b.2 - 0.5).abs();
                    dist_a
                        .partial_cmp(&dist_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                Some(candidates[0].0)
            }
            BranchingStrategy::LeastFractional => {
                // Pick variable with smallest fractional part
                candidates
                    .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
                Some(candidates[0].0)
            }
            BranchingStrategy::FirstFractional => Some(candidates[0].0),
            BranchingStrategy::SmallestDomain => {
                // For now, just pick first (domain analysis would require more info)
                Some(candidates[0].0)
            }
        }
    }

    /// Create a branch by adding a constraint.
    fn create_branch(
        &mut self,
        stack: &mut Vec<BranchNode>,
        parent: &BranchNode,
        var: Var,
        bound: &BigRational,
        is_lower: bool, // true for x >= bound, false for x <= bound
    ) {
        // Create new node
        let mut new_branched = parent.branched_vars.clone();
        new_branched.insert(var);

        let new_node = BranchNode {
            level: parent.level + 1,
            branched_vars: new_branched,
            depth: parent.depth + 1,
        };

        // Add constraint to NLSAT solver
        // For x >= bound: (x - bound > 0) OR (x - bound = 0) => NOT(x - bound < 0)
        // For x <= bound: (x - bound < 0) OR (x - bound = 0) => NOT(x - bound > 0)
        let x = Polynomial::from_var(var);
        let poly = Polynomial::sub(&x, &Polynomial::constant(bound.clone()));

        if is_lower {
            // x >= bound means NOT(x - bound < 0)
            let atom_id = self.nlsat.new_ineq_atom(poly, AtomKind::Lt);
            let lit = self.nlsat.atom_literal(atom_id, false); // negated
            self.nlsat.add_clause(vec![lit]);
        } else {
            // x <= bound means NOT(x - bound > 0)
            let atom_id = self.nlsat.new_ineq_atom(poly, AtomKind::Gt);
            let lit = self.nlsat.atom_literal(atom_id, false); // negated
            self.nlsat.add_clause(vec![lit]);
        }

        stack.push(new_node);
    }

    /// Check if a model satisfies all integer constraints.
    fn is_integer_solution(&self, model: &Model) -> bool {
        for var in 0..self.nlsat.num_arith_vars() {
            if !self.is_integer_var(var) {
                continue; // Skip real variables
            }

            if let Some(value) = model.arith_value(var)
                && !self.is_near_integer(value)
            {
                return false;
            }
        }
        true
    }

    /// Check if a value is near an integer.
    fn is_near_integer(&self, value: &BigRational) -> bool {
        let frac = self.fractional_part(value);
        frac < self.config.int_tolerance || frac > (1.0 - self.config.int_tolerance)
    }

    /// Get the fractional part of a rational number.
    fn fractional_part(&self, value: &BigRational) -> f64 {
        // Convert to f64 for fractional part calculation
        let val_f64 = value.numer().to_f64().unwrap_or(0.0) / value.denom().to_f64().unwrap_or(1.0);
        (val_f64 - val_f64.floor()).abs()
    }

    /// Compute floor and ceiling of a rational number.
    fn floor_ceil(&self, value: &BigRational) -> (BigRational, BigRational) {
        // For rational a/b, floor = floor(a/b)
        let floor_int = value.numer() / value.denom();
        let floor_val = BigRational::from_integer(floor_int.clone());

        let ceil_val = if &floor_val == value {
            floor_val.clone()
        } else {
            floor_val.clone() + BigRational::one()
        };

        (floor_val, ceil_val)
    }

    /// Get statistics.
    pub fn stats(&self) -> &NiaStats {
        &self.stats
    }

    /// Add a cutting plane to eliminate the current fractional solution.
    ///
    /// Gomory cutting planes can be used to cut off fractional solutions.
    #[allow(dead_code)]
    pub fn add_cutting_plane(&mut self, model: &Model) -> bool {
        // Collect integer variables
        let integer_vars: Vec<Var> = self
            .var_types
            .iter()
            .enumerate()
            .filter_map(|(idx, vtype)| {
                if *vtype == VarType::Integer {
                    Some(idx as Var)
                } else {
                    None
                }
            })
            .collect();

        // Generate Gomory cuts
        let cuts = self
            .cut_generator
            .generate_gomory_cuts(model, &integer_vars);

        if cuts.is_empty() {
            return false;
        }

        // Add cuts as constraints to the NLSAT solver
        for cut in cuts {
            // Convert the cutting plane constraint (poly >= 0) to a clause
            // poly >= 0 is equivalent to NOT(poly < 0)
            // So we create an atom for poly < 0 and negate it
            let atom_id = self.nlsat.new_ineq_atom(cut.poly.clone(), AtomKind::Lt);
            let literal = self.nlsat.atom_literal(atom_id, false); // Negated literal

            // Add as a unit clause (must be satisfied)
            self.nlsat.add_clause(vec![literal]);
            self.stats.cutting_planes += 1;
        }

        true
    }
}

impl Default for NiaSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    #[test]
    fn test_nia_solver_new() {
        let solver = NiaSolver::new();
        assert_eq!(solver.stats().nodes_explored, 0);
    }

    #[test]
    fn test_nia_var_types() {
        let mut solver = NiaSolver::new();

        solver.set_var_type(0, VarType::Integer);
        solver.set_var_type(1, VarType::Real);

        assert_eq!(solver.var_type(0), VarType::Integer);
        assert_eq!(solver.var_type(1), VarType::Real);
        assert!(solver.is_integer_var(0));
        assert!(!solver.is_integer_var(1));
    }

    #[test]
    fn test_nia_fractional_part() {
        let solver = NiaSolver::new();

        let val1 = BigRational::new(5.into(), 2.into()); // 2.5
        let frac1 = solver.fractional_part(&val1);
        assert!((frac1 - 0.5).abs() < 0.01);

        let val2 = BigRational::from_integer(3.into()); // 3.0
        let frac2 = solver.fractional_part(&val2);
        assert!(frac2 < 0.01);
    }

    #[test]
    fn test_nia_floor_ceil() {
        let solver = NiaSolver::new();

        let val = BigRational::new(7.into(), 2.into()); // 3.5
        let (floor, ceil) = solver.floor_ceil(&val);

        assert_eq!(floor, rat(3));
        assert_eq!(ceil, rat(4));
    }

    #[test]
    fn test_nia_is_near_integer() {
        let solver = NiaSolver::new();

        let int_val = BigRational::from_integer(5.into());
        assert!(solver.is_near_integer(&int_val));

        let frac_val = BigRational::new(5.into(), 2.into()); // 2.5
        assert!(!solver.is_near_integer(&frac_val));
    }

    #[test]
    fn test_nia_simple_integer() {
        let mut solver = NiaSolver::new();

        // x is integer, x - 1 = 0  => x = 1
        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        let x = Polynomial::from_var(var_x);
        let poly = Polynomial::sub(&x, &Polynomial::constant(rat(1)));
        let atom = solver.nlsat_mut().new_ineq_atom(poly, AtomKind::Eq);
        let lit = solver.nlsat().atom_literal(atom, true);
        solver.nlsat_mut().add_clause(vec![lit]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        if let Some(model) = solver.nlsat().get_model() {
            let x_val = model.arith_value(var_x).unwrap();
            assert_eq!(x_val, &rat(1));
        }
    }

    #[test]
    fn test_nia_fractional_infeasible() {
        let mut solver = NiaSolver::new();

        // x is integer
        // x - 0.5 = 0  => x = 0.5 (not integer, should be UNSAT)
        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        let x = Polynomial::from_var(var_x);
        let half = BigRational::new(1.into(), 2.into());
        let poly = Polynomial::sub(&x, &Polynomial::constant(half));
        let atom = solver.nlsat_mut().new_ineq_atom(poly, AtomKind::Eq);
        let lit = solver.nlsat().atom_literal(atom, true);
        solver.nlsat_mut().add_clause(vec![lit]);

        let result = solver.solve();
        // Real relaxation gives x = 0.5, but no integer solution exists
        assert_eq!(result, SolverResult::Unsat);
    }

    #[test]
    fn test_branching_strategy() {
        let config = NiaConfig {
            branching_strategy: BranchingStrategy::MostFractional,
            ..Default::default()
        };

        let solver = NiaSolver::with_config(config);
        assert_eq!(
            solver.config.branching_strategy,
            BranchingStrategy::MostFractional
        );
    }

    #[test]
    fn test_nia_stats() {
        let solver = NiaSolver::new();
        let stats = solver.stats();

        assert_eq!(stats.nodes_explored, 0);
        assert_eq!(stats.cutting_planes, 0);
        assert_eq!(stats.integer_solutions, 0);
    }
}
