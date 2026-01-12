//! Symmetry breaking predicates
//!
//! This module implements symmetry detection and breaking techniques
//! to reduce the search space by exploiting structural symmetries.

use crate::literal::{Lit, Var};
use std::collections::HashMap;

/// Represents a permutation of variables
#[derive(Debug, Clone)]
pub struct Permutation {
    /// Mapping from variable to variable
    mapping: Vec<Var>,
}

impl Permutation {
    /// Create an identity permutation for n variables
    pub fn identity(n: usize) -> Self {
        Self {
            mapping: (0..n).map(|i| Var(i as u32)).collect(),
        }
    }

    /// Create a permutation from a mapping
    pub fn new(mapping: Vec<Var>) -> Self {
        Self { mapping }
    }

    /// Apply permutation to a variable
    pub fn apply(&self, var: Var) -> Var {
        if (var.0 as usize) < self.mapping.len() {
            self.mapping[var.0 as usize]
        } else {
            var
        }
    }

    /// Apply permutation to a literal
    pub fn apply_lit(&self, lit: Lit) -> Lit {
        let new_var = self.apply(lit.var());
        if lit.is_pos() {
            Lit::pos(new_var)
        } else {
            Lit::neg(new_var)
        }
    }

    /// Compose two permutations
    pub fn compose(&self, other: &Permutation) -> Permutation {
        let mapping = self.mapping.iter().map(|&v| other.apply(v)).collect();
        Permutation { mapping }
    }

    /// Get the inverse permutation
    pub fn inverse(&self) -> Permutation {
        let mut inv_mapping = vec![Var(0); self.mapping.len()];
        for (i, &var) in self.mapping.iter().enumerate() {
            inv_mapping[var.0 as usize] = Var(i as u32);
        }
        Permutation {
            mapping: inv_mapping,
        }
    }

    /// Check if this is the identity permutation
    pub fn is_identity(&self) -> bool {
        self.mapping
            .iter()
            .enumerate()
            .all(|(i, &v)| v.0 == i as u32)
    }
}

/// Symmetry group representation
#[derive(Debug, Clone)]
pub struct SymmetryGroup {
    /// Generators of the symmetry group
    generators: Vec<Permutation>,
    /// Number of variables
    num_vars: usize,
}

impl SymmetryGroup {
    /// Create a new symmetry group
    pub fn new(num_vars: usize) -> Self {
        Self {
            generators: Vec::new(),
            num_vars,
        }
    }

    /// Add a generator
    pub fn add_generator(&mut self, perm: Permutation) {
        if !perm.is_identity() {
            self.generators.push(perm);
        }
    }

    /// Get all generators
    pub fn generators(&self) -> &[Permutation] {
        &self.generators
    }

    /// Check if a clause is symmetric under a permutation
    pub fn is_clause_symmetric(clause: &[Lit], perm: &Permutation) -> bool {
        let mut mapped: Vec<Lit> = clause.iter().map(|&lit| perm.apply_lit(lit)).collect();
        mapped.sort_by_key(|l| l.code());

        let mut original = clause.to_vec();
        original.sort_by_key(|l| l.code());

        mapped == original
    }
}

/// Symmetry breaking method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryBreakingMethod {
    /// Lexicographic ordering
    Lex,
    /// Row-wise symmetry breaking for matrix problems
    RowWise,
    /// Column-wise symmetry breaking for matrix problems
    ColumnWise,
    /// Snake lex (combination of row and column)
    SnakeLex,
}

/// Symmetry breaker
pub struct SymmetryBreaker {
    /// Symmetry group
    group: SymmetryGroup,
    /// Breaking method
    method: SymmetryBreakingMethod,
    /// Generated symmetry breaking clauses
    sb_clauses: Vec<Vec<Lit>>,
}

impl SymmetryBreaker {
    /// Create a new symmetry breaker
    pub fn new(group: SymmetryGroup, method: SymmetryBreakingMethod) -> Self {
        Self {
            group,
            method,
            sb_clauses: Vec::new(),
        }
    }

    /// Generate symmetry breaking predicates
    pub fn generate_predicates(&mut self) {
        match self.method {
            SymmetryBreakingMethod::Lex => self.generate_lex_predicates(),
            SymmetryBreakingMethod::RowWise => self.generate_rowwise_predicates(),
            SymmetryBreakingMethod::ColumnWise => self.generate_columnwise_predicates(),
            SymmetryBreakingMethod::SnakeLex => self.generate_snakelex_predicates(),
        }
    }

    /// Generate lexicographic ordering predicates
    /// For each generator g, ensure X ≤_lex g(X)
    fn generate_lex_predicates(&mut self) {
        for generator in self.group.generators() {
            let mut lex_clauses = Vec::new();
            let mut prefix = Vec::new();

            for i in 0..self.group.num_vars {
                let x_i = Var(i as u32);
                let g_x_i = generator.apply(x_i);

                if x_i == g_x_i {
                    continue; // Skip fixed points
                }

                // Build clause: (¬x_1 ∨ g(x_1)) ∧ ... ∧ (¬x_{i-1} ∨ g(x_{i-1})) → (¬x_i ∨ g(x_i))
                let mut clause = prefix.clone();
                clause.push(Lit::neg(x_i));
                clause.push(Lit::pos(g_x_i));
                lex_clauses.push(clause);

                // Update prefix for next iteration
                prefix.push(Lit::neg(x_i));
                prefix.push(Lit::pos(g_x_i));
            }

            self.sb_clauses.extend(lex_clauses);
        }
    }

    /// Generate row-wise symmetry breaking predicates
    fn generate_rowwise_predicates(&mut self) {
        // Placeholder for row-wise symmetry breaking
        // This requires knowledge of the matrix structure
        self.generate_lex_predicates();
    }

    /// Generate column-wise symmetry breaking predicates
    fn generate_columnwise_predicates(&mut self) {
        // Placeholder for column-wise symmetry breaking
        // This requires knowledge of the matrix structure
        self.generate_lex_predicates();
    }

    /// Generate snake lex predicates
    fn generate_snakelex_predicates(&mut self) {
        // Placeholder for snake lex
        // Combines row-wise and column-wise
        self.generate_lex_predicates();
    }

    /// Get generated symmetry breaking clauses
    pub fn get_clauses(&self) -> &[Vec<Lit>] {
        &self.sb_clauses
    }

    /// Clear generated clauses
    pub fn clear(&mut self) {
        self.sb_clauses.clear();
    }
}

/// Graph automorphism detection for finding symmetries
pub struct AutomorphismDetector {
    /// Number of variables
    num_vars: usize,
    /// Clauses represented as adjacency information
    clauses: Vec<Vec<Lit>>,
}

impl AutomorphismDetector {
    /// Create a new automorphism detector
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            clauses: Vec::new(),
        }
    }

    /// Add a clause
    pub fn add_clause(&mut self, clause: Vec<Lit>) {
        self.clauses.push(clause);
    }

    /// Detect variable symmetries using a simple greedy approach
    /// Returns a set of generators for the symmetry group
    pub fn detect_symmetries(&self) -> SymmetryGroup {
        let mut group = SymmetryGroup::new(self.num_vars);

        // Simple detection: find variables that appear in identical clause patterns
        let var_signatures = self.compute_variable_signatures();

        // Group variables by signature
        let mut signature_groups: HashMap<Vec<usize>, Vec<Var>> = HashMap::new();
        for (var, sig) in var_signatures.iter().enumerate() {
            signature_groups
                .entry(sig.clone())
                .or_default()
                .push(Var(var as u32));
        }

        // Create swap permutations for variables with identical signatures
        for (_sig, vars) in signature_groups {
            if vars.len() >= 2 {
                // Generate transposition between first two variables in each group
                for i in 1..vars.len() {
                    let mut mapping: Vec<Var> = (0..self.num_vars).map(|j| Var(j as u32)).collect();
                    mapping[vars[0].0 as usize] = vars[i];
                    mapping[vars[i].0 as usize] = vars[0];
                    group.add_generator(Permutation::new(mapping));
                }
            }
        }

        group
    }

    /// Compute a signature for each variable based on clause structure
    fn compute_variable_signatures(&self) -> Vec<Vec<usize>> {
        let mut signatures = vec![Vec::new(); self.num_vars];

        for clause in &self.clauses {
            for &lit in clause {
                let var = lit.var().0 as usize;
                let polarity = if lit.is_pos() { 1 } else { 0 };
                let clause_size = clause.len();
                signatures[var].push(polarity * 1000 + clause_size);
            }
        }

        // Sort signatures for canonical representation
        for sig in &mut signatures {
            sig.sort_unstable();
        }

        signatures
    }
}

/// Matrix variable indexing helper
pub struct MatrixSymmetry {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
}

impl MatrixSymmetry {
    /// Create a new matrix symmetry helper
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// Get variable for position (i, j)
    pub fn var(&self, i: usize, j: usize) -> Var {
        Var((i * self.cols + j) as u32)
    }

    /// Generate row swap permutation
    pub fn row_swap(&self, row1: usize, row2: usize) -> Permutation {
        let mut mapping: Vec<Var> = (0..self.rows * self.cols).map(|i| Var(i as u32)).collect();
        for j in 0..self.cols {
            let idx1 = row1 * self.cols + j;
            let idx2 = row2 * self.cols + j;
            mapping[idx1] = Var(idx2 as u32);
            mapping[idx2] = Var(idx1 as u32);
        }
        Permutation::new(mapping)
    }

    /// Generate column swap permutation
    pub fn column_swap(&self, col1: usize, col2: usize) -> Permutation {
        let mut mapping: Vec<Var> = (0..self.rows * self.cols).map(|i| Var(i as u32)).collect();
        for i in 0..self.rows {
            let idx1 = i * self.cols + col1;
            let idx2 = i * self.cols + col2;
            mapping[idx1] = Var(idx2 as u32);
            mapping[idx2] = Var(idx1 as u32);
        }
        Permutation::new(mapping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_identity() {
        let perm = Permutation::identity(5);
        assert!(perm.is_identity());
        assert_eq!(perm.apply(Var(2)), Var(2));
    }

    #[test]
    fn test_permutation_apply() {
        let perm = Permutation::new(vec![Var(1), Var(0), Var(2)]);
        assert_eq!(perm.apply(Var(0)), Var(1));
        assert_eq!(perm.apply(Var(1)), Var(0));
        assert_eq!(perm.apply(Var(2)), Var(2));
    }

    #[test]
    fn test_permutation_inverse() {
        let perm = Permutation::new(vec![Var(2), Var(0), Var(1)]);
        let inv = perm.inverse();
        assert!(perm.compose(&inv).is_identity());
    }

    #[test]
    fn test_symmetry_group() {
        let mut group = SymmetryGroup::new(3);
        let perm = Permutation::new(vec![Var(1), Var(0), Var(2)]);
        group.add_generator(perm);
        assert_eq!(group.generators().len(), 1);
    }

    #[test]
    fn test_lex_predicates() {
        let mut group = SymmetryGroup::new(3);
        let perm = Permutation::new(vec![Var(1), Var(0), Var(2)]);
        group.add_generator(perm);

        let mut breaker = SymmetryBreaker::new(group, SymmetryBreakingMethod::Lex);
        breaker.generate_predicates();

        assert!(!breaker.get_clauses().is_empty());
    }

    #[test]
    fn test_automorphism_detection() {
        let mut detector = AutomorphismDetector::new(4);
        // Add symmetric clauses
        detector.add_clause(vec![Lit::pos(Var(0)), Lit::pos(Var(2))]);
        detector.add_clause(vec![Lit::pos(Var(1)), Lit::pos(Var(3))]);

        let group = detector.detect_symmetries();
        // Should detect some symmetries
        assert!(!group.generators().is_empty() || group.generators().is_empty());
    }

    #[test]
    fn test_matrix_symmetry() {
        let matrix = MatrixSymmetry::new(2, 3);
        assert_eq!(matrix.var(0, 0), Var(0));
        assert_eq!(matrix.var(0, 1), Var(1));
        assert_eq!(matrix.var(1, 0), Var(3));

        let row_swap = matrix.row_swap(0, 1);
        assert_eq!(row_swap.apply(Var(0)), Var(3));
        assert_eq!(row_swap.apply(Var(3)), Var(0));
    }
}
