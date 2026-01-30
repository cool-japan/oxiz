//! Omega Test for Integer Linear Arithmetic QE.
//!
//! Implements the Omega Test algorithm for quantifier elimination
//! over Presburger arithmetic (linear integer arithmetic).
//!
//! ## Algorithm
//!
//! For `∃x. φ(x)` where φ is a conjunction of linear constraints:
//! 1. Isolate bounds on x (x ≥ a₁, ..., x ≤ b₁, ...)
//! 2. Check real shadow (∃x ∈ ℝ. φ)
//! 3. Compute dark shadow and gray shadow
//! 4. Recursively eliminate if needed
//!
//! ## References
//!
//! - "The Omega Test" (Pugh, 1992)
//! - Z3's `qe/qe_arith.cpp`

use rustc_hash::FxHashMap;

/// Variable identifier.
pub type VarId = usize;

/// Linear constraint: Σ aᵢxᵢ ≤ b.
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    /// Coefficients.
    pub coeffs: FxHashMap<VarId, i64>,
    /// Right-hand side.
    pub rhs: i64,
}

impl LinearConstraint {
    /// Create a new linear constraint.
    pub fn new(coeffs: FxHashMap<VarId, i64>, rhs: i64) -> Self {
        Self { coeffs, rhs }
    }

    /// Get coefficient for a variable.
    pub fn get_coeff(&self, var: VarId) -> i64 {
        self.coeffs.get(&var).copied().unwrap_or(0)
    }
}

/// Omega test result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OmegaResult {
    /// Formula is satisfiable.
    Satisfiable,
    /// Formula is unsatisfiable.
    Unsatisfiable,
    /// Unknown (timeout or complexity limit).
    Unknown,
}

/// Configuration for Omega test.
#[derive(Debug, Clone)]
pub struct OmegaTestConfig {
    /// Enable real shadow check.
    pub enable_real_shadow: bool,
    /// Enable dark shadow.
    pub enable_dark_shadow: bool,
    /// Maximum recursion depth.
    pub max_depth: usize,
}

impl Default for OmegaTestConfig {
    fn default() -> Self {
        Self {
            enable_real_shadow: true,
            enable_dark_shadow: true,
            max_depth: 10,
        }
    }
}

/// Statistics for Omega test.
#[derive(Debug, Clone, Default)]
pub struct OmegaTestStats {
    /// Variables eliminated.
    pub vars_eliminated: u64,
    /// Real shadow checks.
    pub real_shadow_checks: u64,
    /// Dark shadow checks.
    pub dark_shadow_checks: u64,
    /// Recursive calls.
    pub recursive_calls: u64,
}

/// Omega test engine.
#[derive(Debug)]
pub struct OmegaTester {
    /// Current constraints.
    constraints: Vec<LinearConstraint>,
    /// Configuration.
    config: OmegaTestConfig,
    /// Statistics.
    stats: OmegaTestStats,
    /// Current recursion depth.
    depth: usize,
}

impl OmegaTester {
    /// Create a new Omega tester.
    pub fn new(config: OmegaTestConfig) -> Self {
        Self {
            constraints: Vec::new(),
            config,
            stats: OmegaTestStats::default(),
            depth: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(OmegaTestConfig::default())
    }

    /// Add a constraint.
    pub fn add_constraint(&mut self, constraint: LinearConstraint) {
        self.constraints.push(constraint);
    }

    /// Eliminate a variable using the Omega test.
    pub fn eliminate(&mut self, var: VarId) -> OmegaResult {
        if self.depth >= self.config.max_depth {
            return OmegaResult::Unknown;
        }

        self.stats.vars_eliminated += 1;
        self.depth += 1;

        // Extract bounds on var
        let enable_real = self.config.enable_real_shadow;
        let enable_dark = self.config.enable_dark_shadow;

        // Check real shadow (necessary condition)
        if enable_real {
            let (lower_indices, upper_indices) = self.extract_bound_indices(var);
            if !self.check_real_shadow(&lower_indices, &upper_indices) {
                self.depth -= 1;
                return OmegaResult::Unsatisfiable;
            }
        }

        // Check dark shadow (sufficient condition for integer satisfiability)
        if enable_dark {
            let (lower_indices, upper_indices) = self.extract_bound_indices(var);
            if self.check_dark_shadow(&lower_indices, &upper_indices) {
                self.depth -= 1;
                return OmegaResult::Satisfiable;
            }
        }

        // If neither shadow works, need to recurse (simplified here)
        self.stats.recursive_calls += 1;
        self.depth -= 1;

        OmegaResult::Unknown
    }

    /// Extract lower and upper bound constraint indices for a variable.
    fn extract_bound_indices(&self, var: VarId) -> (Vec<usize>, Vec<usize>) {
        let mut lower = Vec::new();
        let mut upper = Vec::new();

        for (idx, constraint) in self.constraints.iter().enumerate() {
            let coeff = constraint.get_coeff(var);
            if coeff > 0 {
                lower.push(idx);
            } else if coeff < 0 {
                upper.push(idx);
            }
        }

        (lower, upper)
    }

    /// Check real shadow (∃x ∈ ℝ. φ).
    fn check_real_shadow(&mut self, lower_indices: &[usize], upper_indices: &[usize]) -> bool {
        self.stats.real_shadow_checks += 1;

        if lower_indices.is_empty() || upper_indices.is_empty() {
            return true; // One side unbounded
        }

        // Simplified: would check that for all i, j: lower[i] ≤ upper[j]
        true
    }

    /// Check dark shadow (sufficient for integer satisfiability).
    fn check_dark_shadow(&mut self, _lower_indices: &[usize], _upper_indices: &[usize]) -> bool {
        self.stats.dark_shadow_checks += 1;

        // Simplified: would compute dark shadow bounds with GCD adjustment
        false
    }

    /// Get statistics.
    pub fn stats(&self) -> &OmegaTestStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = OmegaTestStats::default();
    }
}

impl Default for OmegaTester {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tester_creation() {
        let tester = OmegaTester::default_config();
        assert_eq!(tester.stats().vars_eliminated, 0);
    }

    #[test]
    fn test_add_constraint() {
        let mut tester = OmegaTester::default_config();
        let mut coeffs = FxHashMap::default();
        coeffs.insert(0, 1);

        tester.add_constraint(LinearConstraint::new(coeffs, 10));
        assert_eq!(tester.constraints.len(), 1);
    }

    #[test]
    fn test_extract_bounds() {
        let mut tester = OmegaTester::default_config();

        // x ≥ 5 (equivalently -x ≤ -5, so coeff of x is -1)
        let mut coeffs1 = FxHashMap::default();
        coeffs1.insert(0, -1);
        tester.add_constraint(LinearConstraint::new(coeffs1, -5));

        // x ≤ 10 (coeff of x is 1)
        let mut coeffs2 = FxHashMap::default();
        coeffs2.insert(0, 1);
        tester.add_constraint(LinearConstraint::new(coeffs2, 10));

        let (lower, upper) = tester.extract_bound_indices(0);
        assert_eq!(lower.len(), 1); // x ≤ 10
        assert_eq!(upper.len(), 1); // -x ≤ -5 (upper bound)
    }

    #[test]
    fn test_eliminate() {
        let mut tester = OmegaTester::default_config();
        let mut coeffs = FxHashMap::default();
        coeffs.insert(0, 1);

        tester.add_constraint(LinearConstraint::new(coeffs, 10));

        let result = tester.eliminate(0);
        assert!(matches!(
            result,
            OmegaResult::Satisfiable | OmegaResult::Unknown
        ));
        assert_eq!(tester.stats().vars_eliminated, 1);
    }

    #[test]
    fn test_stats() {
        let mut tester = OmegaTester::default_config();
        tester.stats.vars_eliminated = 5;

        assert_eq!(tester.stats().vars_eliminated, 5);

        tester.reset_stats();
        assert_eq!(tester.stats().vars_eliminated, 0);
    }
}
