//! Variable ordering strategies for NLSAT.
//!
//! Variable ordering is crucial for CAD efficiency. A good ordering can reduce
//! the number of cells exponentially. This module implements several heuristics:
//!
//! - **Brown's heuristic**: Minimize the number of cells by analyzing polynomial structure
//! - **Degree-based**: Order by maximum degree in polynomials
//! - **Occurrence-based**: Order by number of polynomial occurrences
//! - **Static**: User-provided or default ordering
//!
//! Reference:
//! - Brown, "An improved projection operation for cylindrical algebraic decomposition" (1978)
//! - Hong, "Heuristic strategies for CAD" (1990)

use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering as CmpOrdering;

/// Strategy for ordering variables in CAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderingStrategy {
    /// Use Brown's heuristic (minimize projection complexity).
    Brown,
    /// Order by maximum degree (higher degree first).
    MaxDegree,
    /// Order by minimum degree (lower degree first).
    MinDegree,
    /// Order by occurrence count (more occurrences first).
    MaxOccurrence,
    /// Order by occurrence count (fewer occurrences first).
    MinOccurrence,
    /// Use the given static ordering.
    Static,
}

/// Variable ordering computer.
pub struct VariableOrdering {
    /// Ordering strategy.
    strategy: OrderingStrategy,
    /// Polynomials to analyze.
    polynomials: Vec<Polynomial>,
    /// All variables found in polynomials.
    variables: FxHashSet<Var>,
}

impl VariableOrdering {
    /// Create a new variable ordering computer.
    pub fn new(strategy: OrderingStrategy, polynomials: Vec<Polynomial>) -> Self {
        let mut variables = FxHashSet::default();
        for poly in &polynomials {
            for var in poly.vars() {
                variables.insert(var);
            }
        }

        Self {
            strategy,
            polynomials,
            variables,
        }
    }

    /// Compute the variable ordering.
    pub fn compute(&self) -> Vec<Var> {
        match self.strategy {
            OrderingStrategy::Brown => self.brown_ordering(),
            OrderingStrategy::MaxDegree => self.degree_ordering(false),
            OrderingStrategy::MinDegree => self.degree_ordering(true),
            OrderingStrategy::MaxOccurrence => self.occurrence_ordering(false),
            OrderingStrategy::MinOccurrence => self.occurrence_ordering(true),
            OrderingStrategy::Static => {
                let mut vars: Vec<Var> = self.variables.iter().copied().collect();
                vars.sort_unstable();
                vars
            }
        }
    }

    /// Brown's heuristic: minimize the complexity of the projection.
    ///
    /// The heuristic tries to choose variables such that:
    /// 1. Variables appearing in fewer polynomials come first
    /// 2. Variables with lower total degree come first
    /// 3. Variables that reduce the number of resultants come first
    ///
    /// This is an approximation since the exact optimization is NP-hard.
    fn brown_ordering(&self) -> Vec<Var> {
        let mut remaining: FxHashSet<Var> = self.variables.clone();
        let mut ordering = Vec::with_capacity(remaining.len());

        while !remaining.is_empty() {
            // Compute score for each remaining variable
            let mut scores: Vec<(Var, f64)> = remaining
                .iter()
                .map(|&var| {
                    let score = self.brown_score(var, &remaining);
                    (var, score)
                })
                .collect();

            // Sort by score (lower is better)
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(CmpOrdering::Equal));

            // Choose the variable with the lowest score
            let chosen = scores[0].0;
            ordering.push(chosen);
            remaining.remove(&chosen);
        }

        ordering
    }

    /// Compute Brown's score for a variable.
    ///
    /// Lower score means the variable should come earlier in the ordering.
    fn brown_score(&self, var: Var, remaining: &FxHashSet<Var>) -> f64 {
        let mut score = 0.0;

        // Count polynomials containing this variable
        let mut poly_count = 0;
        let mut total_degree = 0;
        let mut max_degree = 0;

        for poly in &self.polynomials {
            // Only consider polynomials with variables still in the remaining set
            let poly_vars: FxHashSet<Var> = poly.vars().into_iter().collect();
            if !poly_vars.iter().any(|v| remaining.contains(v)) {
                continue;
            }

            if poly.vars().contains(&var) {
                poly_count += 1;
                let deg = poly.degree(var);
                total_degree += deg;
                max_degree = max_degree.max(deg);
            }
        }

        // Score components:
        // 1. Polynomial count (prefer variables in fewer polynomials)
        score += poly_count as f64 * 100.0;

        // 2. Total degree (prefer lower total degree)
        score += total_degree as f64 * 10.0;

        // 3. Maximum degree (prefer lower max degree)
        score += max_degree as f64;

        // 4. Number of potential resultants (rough estimate)
        // If this variable appears in k polynomials, we'll need roughly k*(k-1)/2 resultants
        if poly_count > 1 {
            let resultant_count = (poly_count * (poly_count - 1)) / 2;
            score += resultant_count as f64 * 50.0;
        }

        score
    }

    /// Order by polynomial degree.
    ///
    /// If `ascending` is true, variables with lower max degree come first.
    /// Otherwise, variables with higher max degree come first.
    fn degree_ordering(&self, ascending: bool) -> Vec<Var> {
        let mut var_degrees: Vec<(Var, u32)> = self
            .variables
            .iter()
            .map(|&var| {
                let max_deg = self
                    .polynomials
                    .iter()
                    .map(|p| p.degree(var))
                    .max()
                    .unwrap_or(0);
                (var, max_deg)
            })
            .collect();

        var_degrees.sort_by(|a, b| {
            let cmp = a.1.cmp(&b.1);
            if ascending { cmp } else { cmp.reverse() }
        });

        var_degrees.into_iter().map(|(var, _)| var).collect()
    }

    /// Order by occurrence count.
    ///
    /// If `ascending` is true, variables appearing in fewer polynomials come first.
    /// Otherwise, variables appearing in more polynomials come first.
    fn occurrence_ordering(&self, ascending: bool) -> Vec<Var> {
        let mut var_counts: Vec<(Var, usize)> = self
            .variables
            .iter()
            .map(|&var| {
                let count = self
                    .polynomials
                    .iter()
                    .filter(|p| p.vars().contains(&var))
                    .count();
                (var, count)
            })
            .collect();

        var_counts.sort_by(|a, b| {
            let cmp = a.1.cmp(&b.1);
            if ascending { cmp } else { cmp.reverse() }
        });

        var_counts.into_iter().map(|(var, _)| var).collect()
    }

    /// Get statistics about the polynomial set.
    pub fn stats(&self) -> OrderingStats {
        let num_vars = self.variables.len();
        let num_polys = self.polynomials.len();

        let mut max_degree = 0;
        let mut total_terms = 0;

        for poly in &self.polynomials {
            max_degree = max_degree.max(poly.total_degree());
            total_terms += poly.num_terms();
        }

        let avg_terms = if num_polys > 0 {
            total_terms as f64 / num_polys as f64
        } else {
            0.0
        };

        OrderingStats {
            num_vars,
            num_polys,
            max_degree,
            avg_terms,
        }
    }
}

/// Statistics about a polynomial set for ordering.
#[derive(Debug, Clone)]
pub struct OrderingStats {
    /// Number of variables.
    pub num_vars: usize,
    /// Number of polynomials.
    pub num_polys: usize,
    /// Maximum total degree.
    pub max_degree: u32,
    /// Average number of terms per polynomial.
    pub avg_terms: f64,
}

/// Analyze the impact of different variable orderings.
pub struct OrderingAnalyzer {
    /// Polynomials to analyze.
    polynomials: Vec<Polynomial>,
}

impl OrderingAnalyzer {
    /// Create a new ordering analyzer.
    pub fn new(polynomials: Vec<Polynomial>) -> Self {
        Self { polynomials }
    }

    /// Estimate the number of CAD cells for a given ordering.
    ///
    /// This is a rough heuristic based on polynomial degrees and occurrences.
    pub fn estimate_cells(&self, ordering: &[Var]) -> f64 {
        if ordering.is_empty() {
            return 1.0;
        }

        let mut estimate = 1.0;

        for &var in ordering {
            // For each variable, estimate the number of new cells created
            // This depends on:
            // 1. Number of polynomials containing this variable
            // 2. Their degrees in this variable
            // 3. Expected number of roots

            let mut total_roots_estimate = 0.0;

            for poly in &self.polynomials {
                if poly.vars().contains(&var) {
                    let deg = poly.degree(var);
                    // Rough estimate: a degree-d polynomial has at most d roots
                    // In practice, expect about d/2 real roots on average
                    total_roots_estimate += (deg as f64) * 0.5;
                }
            }

            // Each variable adds cells: between roots, at roots
            // Approximate: 2 * num_roots + 1 cells per variable level
            let cells_this_level = 2.0 * total_roots_estimate + 1.0;
            estimate *= cells_this_level;
        }

        estimate
    }

    /// Compare different ordering strategies and return the best one.
    pub fn best_strategy(&self) -> OrderingStrategy {
        let strategies = [
            OrderingStrategy::Brown,
            OrderingStrategy::MinDegree,
            OrderingStrategy::MaxDegree,
            OrderingStrategy::MinOccurrence,
            OrderingStrategy::MaxOccurrence,
        ];

        let mut best_strategy = OrderingStrategy::Brown;
        let mut best_estimate = f64::MAX;

        for &strategy in &strategies {
            let ordering = VariableOrdering::new(strategy, self.polynomials.clone());
            let var_order = ordering.compute();
            let estimate = self.estimate_cells(&var_order);

            if estimate < best_estimate {
                best_estimate = estimate;
                best_strategy = strategy;
            }
        }

        best_strategy
    }

    /// Get detailed comparison of all strategies.
    pub fn compare_all(&self) -> FxHashMap<OrderingStrategy, f64> {
        let strategies = [
            OrderingStrategy::Brown,
            OrderingStrategy::MinDegree,
            OrderingStrategy::MaxDegree,
            OrderingStrategy::MinOccurrence,
            OrderingStrategy::MaxOccurrence,
            OrderingStrategy::Static,
        ];

        let mut results = FxHashMap::default();

        for &strategy in &strategies {
            let ordering = VariableOrdering::new(strategy, self.polynomials.clone());
            let var_order = ordering.compute();
            let estimate = self.estimate_cells(&var_order);
            results.insert(strategy, estimate);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    #[allow(dead_code)]
    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_static_ordering() {
        // p1 = x + y, p2 = x^2 + 1
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[])]);

        let ordering = VariableOrdering::new(OrderingStrategy::Static, vec![p1, p2]);
        let order = ordering.compute();

        assert_eq!(order.len(), 2);
        assert!(order.contains(&0));
        assert!(order.contains(&1));
    }

    #[test]
    fn test_degree_ordering() {
        // p1 = x (degree 1), p2 = y^2 (degree 2), p3 = z^3 (degree 3)
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(1, 2)])]);
        let p3 = Polynomial::from_coeffs_int(&[(1, &[(2, 3)])]);

        let ordering = VariableOrdering::new(
            OrderingStrategy::MinDegree,
            vec![p1.clone(), p2.clone(), p3.clone()],
        );
        let order = ordering.compute();

        // Should order: x (deg 1), y (deg 2), z (deg 3)
        assert_eq!(order, vec![0, 1, 2]);

        let ordering_max = VariableOrdering::new(OrderingStrategy::MaxDegree, vec![p1, p2, p3]);
        let order_max = ordering_max.compute();

        // Should order: z (deg 3), y (deg 2), x (deg 1)
        assert_eq!(order_max, vec![2, 1, 0]);
    }

    #[test]
    fn test_occurrence_ordering() {
        // p1 = x, p2 = x + y, p3 = x + y + z
        // x appears in 3, y appears in 2, z appears in 1
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)])]);
        let p3 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)]), (1, &[(2, 1)])]);

        let ordering = VariableOrdering::new(
            OrderingStrategy::MinOccurrence,
            vec![p1.clone(), p2.clone(), p3.clone()],
        );
        let order = ordering.compute();

        // z appears least (1), y appears more (2), x appears most (3)
        assert_eq!(order, vec![2, 1, 0]);

        let ordering_max = VariableOrdering::new(OrderingStrategy::MaxOccurrence, vec![p1, p2, p3]);
        let order_max = ordering_max.compute();

        // x appears most (3), y appears less (2), z appears least (1)
        assert_eq!(order_max, vec![0, 1, 2]);
    }

    #[test]
    fn test_brown_ordering() {
        // Simple case: x^2 + 1, y + 1
        // x has higher degree, y has lower degree
        // Brown's heuristic should prefer y first (lower degree, simpler)
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(1, 1)]), (1, &[])]);

        let ordering = VariableOrdering::new(OrderingStrategy::Brown, vec![p1, p2]);
        let order = ordering.compute();

        // Should prefer y first (simpler, linear)
        assert_eq!(order[0], 1);
        assert_eq!(order[1], 0);
    }

    #[test]
    fn test_ordering_stats() {
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (2, &[(0, 1)]), (1, &[])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(1, 1)]), (1, &[])]);

        let ordering = VariableOrdering::new(OrderingStrategy::Static, vec![p1, p2]);
        let stats = ordering.stats();

        assert_eq!(stats.num_vars, 2);
        assert_eq!(stats.num_polys, 2);
        assert_eq!(stats.max_degree, 2);
        assert_eq!(stats.avg_terms, 2.5); // (3 + 2) / 2
    }

    #[test]
    fn test_ordering_analyzer() {
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-4, &[])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(1, 1)]), (-1, &[])]);

        let analyzer = OrderingAnalyzer::new(vec![p1, p2]);

        // Test cell estimation
        let estimate1 = analyzer.estimate_cells(&[0, 1]);
        let estimate2 = analyzer.estimate_cells(&[1, 0]);

        // Both should be positive
        assert!(estimate1 > 0.0);
        assert!(estimate2 > 0.0);

        // Test strategy comparison
        let best = analyzer.best_strategy();
        assert!(matches!(
            best,
            OrderingStrategy::Brown
                | OrderingStrategy::MinDegree
                | OrderingStrategy::MaxDegree
                | OrderingStrategy::MinOccurrence
                | OrderingStrategy::MaxOccurrence
        ));
    }

    #[test]
    fn test_compare_all_strategies() {
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[(1, 1)]), (-1, &[])]);
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 2)])]);

        let analyzer = OrderingAnalyzer::new(vec![p1, p2]);
        let comparison = analyzer.compare_all();

        // Should have all strategies
        assert_eq!(comparison.len(), 6);
        assert!(comparison.contains_key(&OrderingStrategy::Brown));
        assert!(comparison.contains_key(&OrderingStrategy::MinDegree));
        assert!(comparison.contains_key(&OrderingStrategy::MaxDegree));
        assert!(comparison.contains_key(&OrderingStrategy::MinOccurrence));
        assert!(comparison.contains_key(&OrderingStrategy::MaxOccurrence));
        assert!(comparison.contains_key(&OrderingStrategy::Static));
    }
}
