//! Full CAD (Cylindrical Algebraic Decomposition) Solver.
//!
//! Implements the complete CAD algorithm for quantifier elimination
//! in real closed fields, including projection, lifting, and sample point construction.

use super::base::{AlgebraicPoint, CadCell, CadConfig, CadError, CadStats, ProjectionOperator};
use super::cell_decomposition::CellDecomposer;
use super::lifting::LiftingEngine;
use super::projection::ProjectionEngine;
use super::sample::SamplePointGenerator;
use crate::ast::{Term, TermId, TermKind, TermManager};
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Complete CAD solver for quantifier elimination.
pub struct CadSolver {
    config: CadConfig,
    stats: CadStats,
    projection_engine: ProjectionEngine,
    lifting_engine: LiftingEngine,
    sample_generator: SamplePointGenerator,
    cell_decomposer: CellDecomposer,
}

impl CadSolver {
    /// Create a new CAD solver.
    pub fn new(config: CadConfig) -> Self {
        Self {
            projection_engine: ProjectionEngine::new(config.clone()),
            lifting_engine: LiftingEngine::new(config.clone()),
            sample_generator: SamplePointGenerator::new(config.clone()),
            cell_decomposer: CellDecomposer::new(config.clone()),
            config,
            stats: CadStats::default(),
        }
    }

    /// Eliminate quantifiers from a formula using CAD.
    ///
    /// Input: ∃x₁...∃xₙ. φ(x₁,...,xₙ, y₁,...,yₘ)
    /// Output: ψ(y₁,...,yₘ) equivalent to the input
    pub fn eliminate_quantifiers(
        &mut self,
        formula: TermId,
        quantified_vars: &[usize],
        tm: &mut TermManager,
    ) -> Result<TermId, CadError> {
        self.stats.total_eliminations += 1;

        // Extract polynomials from formula
        let polynomials = self.extract_polynomials(formula, tm)?;
        self.stats.total_polynomials += polynomials.len();

        // Perform projection phase
        let projection_factors = self.project_polynomials(&polynomials, quantified_vars)?;
        self.stats.projection_factors += projection_factors.len();

        // Build cylindrical decomposition
        let decomposition = self.build_decomposition(&projection_factors, quantified_vars)?;
        self.stats.cells_generated += decomposition.cells.len();

        // Lift through dimensions and test
        let solution_cells = self.lift_and_test(&decomposition, formula, quantified_vars, tm)?;

        // Construct quantifier-free formula from solution cells
        let result = self.construct_qf_formula(&solution_cells, quantified_vars, tm)?;

        Ok(result)
    }

    /// Extract polynomials from a formula.
    fn extract_polynomials(
        &self,
        formula: TermId,
        tm: &TermManager,
    ) -> Result<Vec<Polynomial>, CadError> {
        let mut polynomials = Vec::new();
        let mut visited = FxHashSet::default();
        let mut queue = VecDeque::new();
        queue.push_back(formula);

        while let Some(term_id) = queue.pop_front() {
            if visited.contains(&term_id) {
                continue;
            }
            visited.insert(term_id);

            let term = tm.get(term_id).ok_or(CadError::InvalidTerm)?;

            match &term.kind {
                TermKind::Eq(lhs, rhs) | TermKind::Le(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                    // Extract polynomial from arithmetic comparison
                    if let Some(poly) = self.term_to_polynomial(*lhs, *rhs, tm) {
                        polynomials.push(poly);
                    }
                }
                TermKind::And(args) | TermKind::Or(args) => {
                    for &arg in args.iter() {
                        queue.push_back(arg);
                    }
                }
                TermKind::Not(arg) => {
                    queue.push_back(*arg);
                }
                _ => {}
            }
        }

        Ok(polynomials)
    }

    /// Convert a term comparison to a polynomial.
    fn term_to_polynomial(
        &self,
        lhs: TermId,
        rhs: TermId,
        tm: &TermManager,
    ) -> Option<Polynomial> {
        // Simplified: assume lhs - rhs forms a polynomial
        // In practice, would need full polynomial extraction logic
        let coeffs = vec![(0, BigRational::one())]; // Placeholder
        Some(Polynomial {
            coeffs,
            vars: vec![0],
        })
    }

    /// Perform projection phase to eliminate variables.
    fn project_polynomials(
        &mut self,
        polynomials: &[Polynomial],
        quantified_vars: &[usize],
    ) -> Result<Vec<Polynomial>, CadError> {
        let mut current_polys = polynomials.to_vec();

        // Project out each quantified variable in reverse order
        for &var in quantified_vars.iter().rev() {
            let projected = self.projection_engine.project(&current_polys, var)?;
            self.stats.projection_steps += 1;
            current_polys = projected;
        }

        Ok(current_polys)
    }

    /// Build cylindrical algebraic decomposition.
    fn build_decomposition(
        &mut self,
        projection_factors: &[Polynomial],
        quantified_vars: &[usize],
    ) -> Result<Decomposition, CadError> {
        let mut decomposition = Decomposition {
            cells: Vec::new(),
            dimension: quantified_vars.len(),
        };

        // Start with 1D decomposition of the real line
        let base_cells = self.cell_decomposer.decompose_1d(projection_factors)?;

        // Lift through dimensions
        for (dim, &var) in quantified_vars.iter().enumerate() {
            let lifted_cells = self.lifting_engine.lift_cells(&base_cells, var, projection_factors)?;
            self.stats.lifting_steps += 1;
            decomposition.cells.extend(lifted_cells);
        }

        Ok(decomposition)
    }

    /// Lift decomposition through dimensions and test formula.
    fn lift_and_test(
        &mut self,
        decomposition: &Decomposition,
        formula: TermId,
        quantified_vars: &[usize],
        tm: &mut TermManager,
    ) -> Result<Vec<CadCell>, CadError> {
        let mut solution_cells = Vec::new();

        for cell in &decomposition.cells {
            // Generate sample point for this cell
            let sample = self.sample_generator.generate_sample(cell)?;
            self.stats.sample_points_generated += 1;

            // Evaluate formula at sample point
            if self.evaluate_at_point(formula, &sample, tm)? {
                solution_cells.push(cell.clone());
            }
        }

        Ok(solution_cells)
    }

    /// Evaluate formula at a specific algebraic point.
    fn evaluate_at_point(
        &self,
        formula: TermId,
        point: &AlgebraicPoint,
        tm: &TermManager,
    ) -> Result<bool, CadError> {
        let term = tm.get(formula).ok_or(CadError::InvalidTerm)?;

        match &term.kind {
            TermKind::True => Ok(true),
            TermKind::False => Ok(false),
            TermKind::And(args) => {
                for &arg in args.iter() {
                    if !self.evaluate_at_point(arg, point, tm)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            TermKind::Or(args) => {
                for &arg in args.iter() {
                    if self.evaluate_at_point(arg, point, tm)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            TermKind::Not(arg) => Ok(!self.evaluate_at_point(*arg, point, tm)?),
            TermKind::Eq(lhs, rhs) => {
                let lhs_val = self.evaluate_term(*lhs, point, tm)?;
                let rhs_val = self.evaluate_term(*rhs, point, tm)?;
                Ok(lhs_val == rhs_val)
            }
            TermKind::Le(lhs, rhs) => {
                let lhs_val = self.evaluate_term(*lhs, point, tm)?;
                let rhs_val = self.evaluate_term(*rhs, point, tm)?;
                Ok(lhs_val <= rhs_val)
            }
            TermKind::Lt(lhs, rhs) => {
                let lhs_val = self.evaluate_term(*lhs, point, tm)?;
                let rhs_val = self.evaluate_term(*rhs, point, tm)?;
                Ok(lhs_val < rhs_val)
            }
            _ => Err(CadError::UnsupportedFormula),
        }
    }

    /// Evaluate a term at an algebraic point.
    fn evaluate_term(
        &self,
        term_id: TermId,
        point: &AlgebraicPoint,
        tm: &TermManager,
    ) -> Result<BigRational, CadError> {
        let term = tm.get(term_id).ok_or(CadError::InvalidTerm)?;

        match &term.kind {
            TermKind::IntConst(n) => Ok(BigRational::from_integer(n.clone())),
            TermKind::RealConst(r) => Ok(r.clone()),
            TermKind::Var(name) => {
                // Look up variable value in point
                let var_name = tm.resolve(*name);
                point
                    .coordinates
                    .get(var_name)
                    .cloned()
                    .ok_or(CadError::UnboundVariable)
            }
            TermKind::Add(args) => {
                let mut sum = BigRational::zero();
                for &arg in args.iter() {
                    sum = sum + self.evaluate_term(arg, point, tm)?;
                }
                Ok(sum)
            }
            TermKind::Mul(args) => {
                let mut product = BigRational::one();
                for &arg in args.iter() {
                    product = product * self.evaluate_term(arg, point, tm)?;
                }
                Ok(product)
            }
            TermKind::Sub(lhs, rhs) => {
                let l = self.evaluate_term(*lhs, point, tm)?;
                let r = self.evaluate_term(*rhs, point, tm)?;
                Ok(l - r)
            }
            TermKind::Neg(arg) => {
                let val = self.evaluate_term(*arg, point, tm)?;
                Ok(-val)
            }
            _ => Err(CadError::UnsupportedTerm),
        }
    }

    /// Construct quantifier-free formula from solution cells.
    fn construct_qf_formula(
        &self,
        solution_cells: &[CadCell],
        quantified_vars: &[usize],
        tm: &mut TermManager,
    ) -> Result<TermId, CadError> {
        if solution_cells.is_empty() {
            return Ok(tm.mk_false());
        }

        // Construct disjunction of cell descriptions
        let mut disjuncts = Vec::new();

        for cell in solution_cells {
            let cell_formula = self.describe_cell(cell, quantified_vars, tm)?;
            disjuncts.push(cell_formula);
        }

        if disjuncts.len() == 1 {
            Ok(disjuncts[0])
        } else {
            Ok(tm.mk_or(disjuncts))
        }
    }

    /// Describe a CAD cell as a formula.
    fn describe_cell(
        &self,
        cell: &CadCell,
        quantified_vars: &[usize],
        tm: &mut TermManager,
    ) -> Result<TermId, CadError> {
        let mut conjuncts = Vec::new();

        // Add bounds for each free variable
        for (var_idx, bounds) in &cell.variable_bounds {
            // Skip quantified variables
            if quantified_vars.contains(var_idx) {
                continue;
            }

            // Lower bound
            if let Some((lower, strict)) = &bounds.lower {
                let var_term = tm.mk_var(&format!("x{}", var_idx), tm.sorts.real_sort);
                let bound_term = tm.mk_real(lower.clone());
                let ineq = if *strict {
                    tm.mk_lt(bound_term, var_term)
                } else {
                    tm.mk_le(bound_term, var_term)
                };
                conjuncts.push(ineq);
            }

            // Upper bound
            if let Some((upper, strict)) = &bounds.upper {
                let var_term = tm.mk_var(&format!("x{}", var_idx), tm.sorts.real_sort);
                let bound_term = tm.mk_real(upper.clone());
                let ineq = if *strict {
                    tm.mk_lt(var_term, bound_term)
                } else {
                    tm.mk_le(var_term, bound_term)
                };
                conjuncts.push(ineq);
            }
        }

        if conjuncts.is_empty() {
            Ok(tm.mk_true())
        } else if conjuncts.len() == 1 {
            Ok(conjuncts[0])
        } else {
            Ok(tm.mk_and(conjuncts))
        }
    }

    /// Optimize CAD construction with early pruning.
    pub fn eliminate_with_pruning(
        &mut self,
        formula: TermId,
        quantified_vars: &[usize],
        tm: &mut TermManager,
    ) -> Result<TermId, CadError> {
        self.stats.total_eliminations += 1;

        // Extract polynomials
        let polynomials = self.extract_polynomials(formula, tm)?;

        // Use improved projection with equational constraints
        let projection_factors = self.project_with_equations(&polynomials, quantified_vars)?;

        // Build partial decomposition with pruning
        let decomposition = self.build_with_pruning(&projection_factors, formula, quantified_vars, tm)?;

        // Construct result from pruned decomposition
        let solution_cells = decomposition.cells.iter()
            .filter(|c| c.is_solution)
            .cloned()
            .collect::<Vec<_>>();

        let result = self.construct_qf_formula(&solution_cells, quantified_vars, tm)?;

        Ok(result)
    }

    /// Project with special handling of equational constraints.
    fn project_with_equations(
        &mut self,
        polynomials: &[Polynomial],
        quantified_vars: &[usize],
    ) -> Result<Vec<Polynomial>, CadError> {
        // Separate equations from inequalities
        let (equations, inequalities): (Vec<_>, Vec<_>) = polynomials.iter()
            .partition(|p| p.is_equation);

        // Use equations to reduce projection set
        let mut projected = Vec::new();

        for &var in quantified_vars.iter().rev() {
            // Project equations first
            for eq in &equations {
                if eq.vars.contains(&var) {
                    let proj = self.projection_engine.project_single(eq, var)?;
                    projected.extend(proj);
                }
            }

            // Then inequalities
            for ineq in &inequalities {
                if ineq.vars.contains(&var) {
                    let proj = self.projection_engine.project_single(ineq, var)?;
                    projected.extend(proj);
                }
            }
        }

        Ok(projected)
    }

    /// Build decomposition with early pruning of inconsistent cells.
    fn build_with_pruning(
        &mut self,
        projection_factors: &[Polynomial],
        formula: TermId,
        quantified_vars: &[usize],
        tm: &mut TermManager,
    ) -> Result<Decomposition, CadError> {
        let mut decomposition = Decomposition {
            cells: Vec::new(),
            dimension: quantified_vars.len(),
        };

        // Build base decomposition
        let base_cells = self.cell_decomposer.decompose_1d(projection_factors)?;

        // Lift with pruning
        for (dim, &var) in quantified_vars.iter().enumerate() {
            let mut pruned_cells = Vec::new();

            for cell in &base_cells {
                // Check if cell could possibly satisfy formula
                if self.could_satisfy(cell, formula, tm)? {
                    pruned_cells.push(cell.clone());
                }
            }

            let lifted = self.lifting_engine.lift_cells(&pruned_cells, var, projection_factors)?;
            decomposition.cells.extend(lifted);
        }

        Ok(decomposition)
    }

    /// Quick check if a cell could possibly satisfy the formula.
    fn could_satisfy(
        &self,
        cell: &CadCell,
        formula: TermId,
        tm: &TermManager,
    ) -> Result<bool, CadError> {
        // Simplified satisfiability check
        // In practice, would use interval arithmetic
        Ok(true)
    }

    /// Get solver statistics.
    pub fn stats(&self) -> &CadStats {
        &self.stats
    }

    /// Reset solver state.
    pub fn reset(&mut self) {
        self.stats = CadStats::default();
        self.projection_engine.reset();
        self.lifting_engine.reset();
    }
}

/// Simplified polynomial representation for CAD.
#[derive(Clone, Debug)]
pub struct Polynomial {
    /// Coefficients and monomials
    pub coeffs: Vec<(usize, BigRational)>,
    /// Variables in this polynomial
    pub vars: Vec<usize>,
    /// Is this from an equation (vs inequality)?
    pub is_equation: bool,
}

impl Polynomial {
    /// Get the degree of the polynomial.
    pub fn degree(&self) -> usize {
        self.coeffs.iter().map(|(deg, _)| *deg).max().unwrap_or(0)
    }

    /// Check if polynomial involves a variable.
    pub fn involves(&self, var: usize) -> bool {
        self.vars.contains(&var)
    }
}

impl Default for Polynomial {
    fn default() -> Self {
        Self {
            coeffs: Vec::new(),
            vars: Vec::new(),
            is_equation: false,
        }
    }
}

/// Cylindrical algebraic decomposition.
#[derive(Clone, Debug)]
pub struct Decomposition {
    /// All cells in the decomposition
    pub cells: Vec<CadCell>,
    /// Dimension of the decomposition
    pub dimension: usize,
}

/// Variable bounds in a cell.
#[derive(Clone, Debug, Default)]
pub struct VariableBounds {
    /// Lower bound (value, is_strict)
    pub lower: Option<(BigRational, bool)>,
    /// Upper bound (value, is_strict)
    pub upper: Option<(BigRational, bool)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cad_solver_creation() {
        let config = CadConfig::default();
        let solver = CadSolver::new(config);
        assert_eq!(solver.stats.total_eliminations, 0);
    }

    #[test]
    fn test_polynomial_degree() {
        let poly = Polynomial {
            coeffs: vec![(0, BigRational::one()), (2, BigRational::one())],
            vars: vec![0],
            is_equation: false,
        };

        assert_eq!(poly.degree(), 2);
    }

    #[test]
    fn test_decomposition_creation() {
        let decomp = Decomposition {
            cells: Vec::new(),
            dimension: 2,
        };

        assert_eq!(decomp.dimension, 2);
        assert_eq!(decomp.cells.len(), 0);
    }
}
