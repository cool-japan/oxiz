//! Problem structure analyzer for NLSAT.
//!
//! This module analyzes the structure of polynomial constraint problems to
//! detect special cases that can be solved more efficiently. Key features:
//!
//! - **Linearity Detection**: Identify purely linear problems (can use simplex)
//! - **Univariate Detection**: Detect single-variable problems (can use root isolation)
//! - **Degree Analysis**: Analyze polynomial degree distribution
//! - **Sparsity Analysis**: Measure constraint sparsity and variable connectivity
//! - **Problem Classification**: Classify problems for algorithm selection
//!
//! Reference: Z3's preprocessing and problem classification strategies

use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashSet;
use std::collections::HashMap;

/// Classification of a polynomial constraint problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    /// All constraints are linear (degree ≤ 1).
    Linear,
    /// All constraints involve only one variable.
    Univariate,
    /// All constraints are quadratic (degree ≤ 2).
    Quadratic,
    /// Problem has low degree (max degree ≤ 3).
    LowDegree,
    /// Problem has high degree (max degree > 3).
    HighDegree,
    /// Problem is sparse (few variables per constraint).
    Sparse,
    /// Problem is dense (many variables per constraint).
    Dense,
    /// General non-linear problem.
    General,
}

/// Detailed statistics about problem structure.
#[derive(Debug, Clone)]
pub struct StructureStats {
    /// Total number of constraints.
    pub num_constraints: usize,
    /// Total number of variables.
    pub num_variables: usize,
    /// Maximum degree across all polynomials.
    pub max_degree: usize,
    /// Average degree of polynomials.
    pub avg_degree: f64,
    /// Maximum number of variables in any single constraint.
    pub max_vars_per_constraint: usize,
    /// Average number of variables per constraint.
    pub avg_vars_per_constraint: f64,
    /// Number of linear constraints (degree ≤ 1).
    pub num_linear: usize,
    /// Number of quadratic constraints (degree ≤ 2).
    pub num_quadratic: usize,
    /// Number of univariate constraints.
    pub num_univariate: usize,
    /// Sparsity ratio (average vars per constraint / total vars).
    pub sparsity_ratio: f64,
}

impl StructureStats {
    /// Check if the problem is linear.
    pub fn is_linear(&self) -> bool {
        self.max_degree <= 1
    }

    /// Check if the problem is univariate.
    pub fn is_univariate(&self) -> bool {
        self.num_variables <= 1
    }

    /// Check if the problem is quadratic.
    pub fn is_quadratic(&self) -> bool {
        self.max_degree <= 2
    }

    /// Check if the problem is sparse (sparsity < 0.3).
    pub fn is_sparse(&self) -> bool {
        self.sparsity_ratio < 0.3
    }

    /// Classify the problem based on its structure.
    pub fn classify(&self) -> ProblemClass {
        if self.is_linear() {
            ProblemClass::Linear
        } else if self.is_univariate() {
            ProblemClass::Univariate
        } else if self.is_quadratic() {
            ProblemClass::Quadratic
        } else if self.max_degree <= 3 {
            ProblemClass::LowDegree
        } else if self.is_sparse() {
            ProblemClass::Sparse
        } else if self.sparsity_ratio > 0.7 {
            ProblemClass::Dense
        } else if self.max_degree > 3 {
            ProblemClass::HighDegree
        } else {
            ProblemClass::General
        }
    }
}

/// Problem structure analyzer.
pub struct StructureAnalyzer {
    /// Polynomials in the problem.
    polynomials: Vec<Polynomial>,
    /// Cached statistics.
    stats: Option<StructureStats>,
}

impl StructureAnalyzer {
    /// Create a new structure analyzer.
    pub fn new() -> Self {
        Self {
            polynomials: Vec::new(),
            stats: None,
        }
    }

    /// Add a polynomial to analyze.
    pub fn add_polynomial(&mut self, poly: Polynomial) {
        self.polynomials.push(poly);
        self.stats = None; // Invalidate cache
    }

    /// Add multiple polynomials to analyze.
    pub fn add_polynomials(&mut self, polys: Vec<Polynomial>) {
        self.polynomials.extend(polys);
        self.stats = None; // Invalidate cache
    }

    /// Clear all polynomials.
    pub fn clear(&mut self) {
        self.polynomials.clear();
        self.stats = None;
    }

    /// Analyze the problem structure and return statistics.
    pub fn analyze(&mut self) -> &StructureStats {
        if self.stats.is_none() {
            self.stats = Some(self.compute_stats());
        }
        self.stats
            .as_ref()
            .expect("stats initialized during construction")
    }

    /// Compute structure statistics.
    fn compute_stats(&self) -> StructureStats {
        if self.polynomials.is_empty() {
            return StructureStats {
                num_constraints: 0,
                num_variables: 0,
                max_degree: 0,
                avg_degree: 0.0,
                max_vars_per_constraint: 0,
                avg_vars_per_constraint: 0.0,
                num_linear: 0,
                num_quadratic: 0,
                num_univariate: 0,
                sparsity_ratio: 0.0,
            };
        }

        let num_constraints = self.polynomials.len();

        // Collect all variables
        let mut all_vars = FxHashSet::default();
        for poly in &self.polynomials {
            all_vars.extend(poly.vars());
        }
        let num_variables = all_vars.len();

        // Analyze degrees
        let mut max_degree = 0;
        let mut total_degree = 0;
        let mut num_linear = 0;
        let mut num_quadratic = 0;
        let mut num_univariate = 0;

        for poly in &self.polynomials {
            let degree = poly.total_degree() as usize;
            max_degree = max_degree.max(degree);
            total_degree += degree;

            if degree <= 1 {
                num_linear += 1;
            }
            if degree <= 2 {
                num_quadratic += 1;
            }

            if poly.vars().len() <= 1 {
                num_univariate += 1;
            }
        }

        let avg_degree = total_degree as f64 / num_constraints as f64;

        // Analyze variable occurrences
        let mut max_vars_per_constraint = 0;
        let mut total_vars = 0;

        for poly in &self.polynomials {
            let var_count = poly.vars().len();
            max_vars_per_constraint = max_vars_per_constraint.max(var_count);
            total_vars += var_count;
        }

        let avg_vars_per_constraint = total_vars as f64 / num_constraints as f64;

        let sparsity_ratio = if num_variables > 0 {
            avg_vars_per_constraint / num_variables as f64
        } else {
            0.0
        };

        StructureStats {
            num_constraints,
            num_variables,
            max_degree,
            avg_degree,
            max_vars_per_constraint,
            avg_vars_per_constraint,
            num_linear,
            num_quadratic,
            num_univariate,
            sparsity_ratio,
        }
    }

    /// Classify the problem.
    pub fn classify(&mut self) -> ProblemClass {
        self.analyze().classify()
    }

    /// Get variable connectivity (which variables appear together).
    pub fn variable_connectivity(&self) -> HashMap<Var, FxHashSet<Var>> {
        let mut connectivity: HashMap<Var, FxHashSet<Var>> = HashMap::new();

        for poly in &self.polynomials {
            let vars = poly.vars();
            for &v1 in &vars {
                let entry = connectivity.entry(v1).or_default();
                for &v2 in &vars {
                    if v1 != v2 {
                        entry.insert(v2);
                    }
                }
            }
        }

        connectivity
    }

    /// Find independent variable groups (connected components).
    pub fn find_independent_groups(&self) -> Vec<Vec<Var>> {
        let connectivity = self.variable_connectivity();
        let mut visited = FxHashSet::default();
        let mut groups = Vec::new();

        for &var in connectivity.keys() {
            if visited.contains(&var) {
                continue;
            }

            // BFS to find connected component
            let mut group = Vec::new();
            let mut queue = vec![var];
            visited.insert(var);

            while let Some(current) = queue.pop() {
                group.push(current);

                if let Some(neighbors) = connectivity.get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push(neighbor);
                        }
                    }
                }
            }

            groups.push(group);
        }

        groups
    }

    /// Get recommendation for which solving strategy to use.
    pub fn recommend_strategy(&mut self) -> SolverStrategy {
        let class = self.classify();
        let stats = self.analyze();

        match class {
            ProblemClass::Linear => SolverStrategy::Simplex,
            ProblemClass::Univariate => SolverStrategy::RootIsolation,
            ProblemClass::Quadratic if stats.num_variables <= 5 => SolverStrategy::DirectCAD,
            ProblemClass::LowDegree if stats.is_sparse() => SolverStrategy::IncrementalCAD,
            ProblemClass::Sparse => SolverStrategy::LazyLifting,
            ProblemClass::Dense | ProblemClass::HighDegree => SolverStrategy::VirtualSubstitution,
            _ => SolverStrategy::StandardCAD,
        }
    }
}

impl Default for StructureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Recommended solver strategy based on problem structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStrategy {
    /// Use simplex method (linear arithmetic).
    Simplex,
    /// Use root isolation (univariate).
    RootIsolation,
    /// Use direct CAD (small problems).
    DirectCAD,
    /// Use standard CAD with full projection.
    StandardCAD,
    /// Use incremental CAD with backtracking.
    IncrementalCAD,
    /// Use lazy lifting (only lift needed cells).
    LazyLifting,
    /// Use virtual substitution (high-degree problems).
    VirtualSubstitution,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    fn constant(n: i32) -> Polynomial {
        Polynomial::constant(BigRational::from_integer(BigInt::from(n)))
    }

    #[test]
    fn test_analyzer_new() {
        let analyzer = StructureAnalyzer::new();
        assert_eq!(analyzer.polynomials.len(), 0);
    }

    #[test]
    fn test_linear_detection() {
        let mut analyzer = StructureAnalyzer::new();

        // Linear: 2x + 3y + 5 = 0
        let x = Polynomial::from_var(0);
        let y = Polynomial::from_var(1);
        let two = constant(2);
        let three = constant(3);
        let five = constant(5);

        let poly = Polynomial::add(&Polynomial::mul(&two, &x), &Polynomial::mul(&three, &y));
        let poly = Polynomial::add(&poly, &five);

        analyzer.add_polynomial(poly);

        let stats = analyzer.analyze();
        assert!(stats.is_linear());
        assert_eq!(analyzer.classify(), ProblemClass::Linear);
    }

    #[test]
    fn test_quadratic_detection() {
        let mut analyzer = StructureAnalyzer::new();

        // Quadratic: x² + y = 0
        let x = Polynomial::from_var(0);
        let y = Polynomial::from_var(1);
        let x_squared = Polynomial::mul(&x, &x);
        let poly = Polynomial::add(&x_squared, &y);

        analyzer.add_polynomial(poly);

        let stats = analyzer.analyze();
        assert!(stats.is_quadratic());
        assert_eq!(analyzer.classify(), ProblemClass::Quadratic);
    }

    #[test]
    fn test_univariate_detection() {
        let mut analyzer = StructureAnalyzer::new();

        // Univariate: x³ - 2x + 1 = 0
        let x = Polynomial::from_var(0);
        let x_cubed = Polynomial::mul(&Polynomial::mul(&x, &x), &x);
        let two = constant(2);
        let one = constant(1);
        let two_x = Polynomial::mul(&two, &x);
        let poly = Polynomial::sub(&x_cubed, &two_x);
        let poly = Polynomial::add(&poly, &one);

        analyzer.add_polynomial(poly);

        let stats = analyzer.analyze();
        assert!(stats.is_univariate());
    }

    #[test]
    fn test_sparsity() {
        let mut analyzer = StructureAnalyzer::new();

        // Add constraints using only 2 variables out of many
        // Create 5 constraints, each using 2 out of 20 total variables
        for i in 0..5 {
            let x = Polynomial::from_var(i * 4);
            let y = Polynomial::from_var(i * 4 + 1);
            let poly = Polynomial::add(&x, &y);
            analyzer.add_polynomial(poly);
        }

        let stats = analyzer.analyze();
        // With 20 variables and average of 2 per constraint, sparsity = 2/20 = 0.1 < 0.3
        assert!(stats.is_sparse());
    }

    #[test]
    fn test_variable_connectivity() {
        let mut analyzer = StructureAnalyzer::new();

        // x + y
        let x = Polynomial::from_var(0);
        let y = Polynomial::from_var(1);
        analyzer.add_polynomial(Polynomial::add(&x, &y));

        // y + z
        let z = Polynomial::from_var(2);
        analyzer.add_polynomial(Polynomial::add(&y, &z));

        let connectivity = analyzer.variable_connectivity();
        assert!(connectivity.get(&0).unwrap().contains(&1));
        assert!(connectivity.get(&1).unwrap().contains(&0));
        assert!(connectivity.get(&1).unwrap().contains(&2));
    }

    #[test]
    fn test_independent_groups() {
        let mut analyzer = StructureAnalyzer::new();

        // Group 1: x + y
        let x = Polynomial::from_var(0);
        let y = Polynomial::from_var(1);
        analyzer.add_polynomial(Polynomial::add(&x, &y));

        // Group 2: z + w (independent)
        let z = Polynomial::from_var(2);
        let w = Polynomial::from_var(3);
        analyzer.add_polynomial(Polynomial::add(&z, &w));

        let groups = analyzer.find_independent_groups();
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_strategy_recommendation() {
        let mut analyzer = StructureAnalyzer::new();

        // Linear problem
        let x = Polynomial::from_var(0);
        let y = Polynomial::from_var(1);
        analyzer.add_polynomial(Polynomial::add(&x, &y));

        assert_eq!(analyzer.recommend_strategy(), SolverStrategy::Simplex);
    }

    #[test]
    fn test_clear() {
        let mut analyzer = StructureAnalyzer::new();
        analyzer.add_polynomial(Polynomial::from_var(0));
        analyzer.clear();
        assert_eq!(analyzer.polynomials.len(), 0);
    }
}
