//! Projection Phase of CAD.
//!
//! The projection phase computes projection polynomials that eliminate one
//! variable, producing a CAD of lower dimension. This continues recursively
//! until reaching dimension 1 (real line).
//!
//! ## Projection Operators
//!
//! - **Collins Projection**: Original CAD projection (1975)
//! - **McCallum Projection**: Improved projection (1988)
//! - **Brown Projection**: Further optimization (2001)
//!
//! ## References
//!
//! - Collins: "Quantifier Elimination for Real Closed Fields by Cylindrical
//!   Algebraic Decomposition" (1975)
//! - McCallum: "An Improved Projection Operation for CAD" (1988)
//! - Z3's `qe/qe_arith_plugin.cpp`

use crate::Term;
use rustc_hash::FxHashSet;

/// Variable identifier.
pub type VarId = usize;

/// Polynomial represented as term.
pub type Polynomial = Term;

/// Projection operator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionOperator {
    /// Collins projection (conservative, always works).
    Collins,
    /// McCallum projection (more efficient, requires well-orientedness).
    McCallum,
    /// Brown projection (optimized for sparse polynomials).
    Brown,
}

/// Configuration for projection phase.
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    /// Projection operator to use.
    pub operator: ProjectionOperator,
    /// Enable early termination optimization.
    pub early_termination: bool,
    /// Enable projection caching.
    pub enable_caching: bool,
    /// Maximum projection depth.
    pub max_depth: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            operator: ProjectionOperator::McCallum,
            early_termination: true,
            enable_caching: true,
            max_depth: 100,
        }
    }
}

/// Statistics for projection phase.
#[derive(Debug, Clone, Default)]
pub struct ProjectionStats {
    /// Polynomials projected.
    pub polynomials_projected: u64,
    /// Resultants computed.
    pub resultants_computed: u64,
    /// Discriminants computed.
    pub discriminants_computed: u64,
    /// Leading coefficients computed.
    pub leading_coeffs_computed: u64,
    /// Cache hits.
    pub cache_hits: u64,
}

/// Projection result for one level.
#[derive(Debug, Clone)]
pub struct ProjectionLevel {
    /// Variable being eliminated.
    pub variable: VarId,
    /// Projected polynomials (in lower dimension).
    pub projected_polys: Vec<Polynomial>,
    /// Number of polynomials before projection.
    pub input_count: usize,
    /// Number of polynomials after projection.
    pub output_count: usize,
}

/// CAD projection engine.
pub struct ProjectionEngine {
    /// Current variable ordering.
    variable_order: Vec<VarId>,
    /// Configuration.
    config: ProjectionConfig,
    /// Statistics.
    stats: ProjectionStats,
    /// Projection cache (polynomial pair -> resultant).
    cache: FxHashSet<(usize, usize)>,
}

impl ProjectionEngine {
    /// Create a new projection engine.
    pub fn new(variable_order: Vec<VarId>, config: ProjectionConfig) -> Self {
        Self {
            variable_order,
            config,
            stats: ProjectionStats::default(),
            cache: FxHashSet::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config(variable_order: Vec<VarId>) -> Self {
        Self::new(variable_order, ProjectionConfig::default())
    }

    /// Project a set of polynomials to eliminate a variable.
    pub fn project(&mut self, polys: &[Polynomial], var: VarId) -> ProjectionLevel {
        self.stats.polynomials_projected += polys.len() as u64;

        let projected = match self.config.operator {
            ProjectionOperator::Collins => self.collins_project(polys, var),
            ProjectionOperator::McCallum => self.mccallum_project(polys, var),
            ProjectionOperator::Brown => self.brown_project(polys, var),
        };

        ProjectionLevel {
            variable: var,
            projected_polys: projected.clone(),
            input_count: polys.len(),
            output_count: projected.len(),
        }
    }

    /// Collins projection operator.
    ///
    /// For each polynomial p:
    /// 1. Discriminant(p, x)
    /// 2. Leading coefficient of p w.r.t. x
    ///
    /// For each pair (p, q):
    /// 3. Resultant(p, q, x)
    fn collins_project(&mut self, polys: &[Polynomial], _var: VarId) -> Vec<Polynomial> {
        let mut result = Vec::new();

        // Step 1 & 2: For each polynomial
        for poly in polys {
            // Discriminant
            if let Some(disc) = self.discriminant(poly) {
                result.push(disc);
                self.stats.discriminants_computed += 1;
            }

            // Leading coefficient
            if let Some(lc) = self.leading_coefficient(poly) {
                result.push(lc);
                self.stats.leading_coeffs_computed += 1;
            }
        }

        // Step 3: Pairwise resultants
        for i in 0..polys.len() {
            for j in i + 1..polys.len() {
                if let Some(res) = self.resultant(&polys[i], &polys[j]) {
                    result.push(res);
                    self.stats.resultants_computed += 1;
                }
            }
        }

        result
    }

    /// McCallum projection operator.
    ///
    /// More efficient than Collins, assumes well-orientedness.
    /// Projects discriminants and resultants, but not always leading coefficients.
    fn mccallum_project(&mut self, polys: &[Polynomial], _var: VarId) -> Vec<Polynomial> {
        let mut result = Vec::new();

        // Discriminants
        for poly in polys {
            if let Some(disc) = self.discriminant(poly) {
                result.push(disc);
                self.stats.discriminants_computed += 1;
            }
        }

        // Pairwise resultants
        for i in 0..polys.len() {
            for j in i + 1..polys.len() {
                if let Some(res) = self.resultant(&polys[i], &polys[j]) {
                    result.push(res);
                    self.stats.resultants_computed += 1;
                }
            }
        }

        result
    }

    /// Brown projection operator.
    ///
    /// Optimized for sparse polynomials, uses only necessary projections.
    fn brown_project(&mut self, polys: &[Polynomial], var: VarId) -> Vec<Polynomial> {
        // Simplified: Fall back to McCallum for now
        // Brown projection requires more sophisticated analysis of polynomial structure
        self.mccallum_project(polys, var)
    }

    /// Compute discriminant of a polynomial.
    ///
    /// disc(p) = resultant(p, ∂p/∂x, x)
    fn discriminant(&mut self, _poly: &Polynomial) -> Option<Polynomial> {
        // Simplified implementation
        // Would compute: res(p, ∂p/∂x)
        None
    }

    /// Get leading coefficient of polynomial w.r.t. variable.
    fn leading_coefficient(&mut self, _poly: &Polynomial) -> Option<Polynomial> {
        // Simplified implementation
        // Would extract leading coefficient as a polynomial in other variables
        None
    }

    /// Compute resultant of two polynomials.
    ///
    /// The resultant eliminates the main variable.
    fn resultant(&mut self, _p: &Polynomial, _q: &Polynomial) -> Option<Polynomial> {
        // Simplified implementation
        // Would use Sylvester matrix or subresultant PRS
        None
    }

    /// Project through all levels (full projection).
    pub fn project_all(&mut self, polys: &[Polynomial]) -> Vec<ProjectionLevel> {
        let mut levels = Vec::new();
        let mut current_polys = polys.to_vec();

        for &var in &self.variable_order.clone() {
            let level = self.project(&current_polys, var);
            current_polys = level.projected_polys.clone();
            levels.push(level);

            if current_polys.is_empty() || levels.len() >= self.config.max_depth {
                break;
            }
        }

        levels
    }

    /// Get statistics.
    pub fn stats(&self) -> &ProjectionStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ProjectionStats::default();
    }

    /// Clear cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for ProjectionEngine {
    fn default() -> Self {
        Self::default_config(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let vars = vec![0, 1, 2];
        let engine = ProjectionEngine::default_config(vars);
        assert_eq!(engine.stats().polynomials_projected, 0);
    }

    #[test]
    fn test_projection_operator_variants() {
        assert_ne!(ProjectionOperator::Collins, ProjectionOperator::McCallum);
        assert_ne!(ProjectionOperator::McCallum, ProjectionOperator::Brown);
    }

    #[test]
    fn test_default_config() {
        let config = ProjectionConfig::default();
        assert_eq!(config.operator, ProjectionOperator::McCallum);
        assert!(config.early_termination);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_projection_level() {
        let level = ProjectionLevel {
            variable: 0,
            projected_polys: Vec::new(),
            input_count: 5,
            output_count: 10,
        };

        assert_eq!(level.variable, 0);
        assert_eq!(level.input_count, 5);
        assert_eq!(level.output_count, 10);
    }

    #[test]
    fn test_stats() {
        let mut engine = ProjectionEngine::default();
        engine.stats.polynomials_projected = 42;

        assert_eq!(engine.stats().polynomials_projected, 42);

        engine.reset_stats();
        assert_eq!(engine.stats().polynomials_projected, 0);
    }

    #[test]
    fn test_clear_cache() {
        let mut engine = ProjectionEngine::default();
        engine.cache.insert((0, 1));

        assert!(!engine.cache.is_empty());

        engine.clear_cache();
        assert!(engine.cache.is_empty());
    }
}
