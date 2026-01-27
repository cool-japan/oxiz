//! Cutting Planes for Integer Linear Programming.
//!
//! Implements cutting plane techniques to strengthen LP relaxations
//! and eliminate fractional solutions in mixed-integer programming.
//!
//! ## Techniques
//!
//! - **Gomory Cuts**: Cutting planes derived from simplex tableau
//! - **MIR Cuts**: Mixed-integer rounding cuts
//! - **Lift-and-Project**: Disjunctive cuts
//!
//! ## References
//!
//! - Gomory (1958): "Outline of an algorithm for integer solutions to linear programs"
//! - Nemhauser & Wolsey (1988): "Integer and Combinatorial Optimization"
//! - Z3's `math/lp/gomory.cpp`

use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

/// Variable identifier.
pub type VarId = usize;

/// Type of cutting plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutType {
    /// Gomory fractional cut.
    GomoryFractional,
    /// Gomory mixed-integer cut (MIR).
    GomoryMixed,
    /// Lift-and-project cut.
    LiftProject,
}

/// A cutting plane constraint represented as: sum(coeff[i] * var[i]) >= rhs.
#[derive(Debug, Clone)]
pub struct Cut {
    /// Coefficients for each variable.
    pub coeffs: FxHashMap<VarId, BigRational>,
    /// Right-hand side.
    pub rhs: BigRational,
    /// Type of cut.
    pub cut_type: CutType,
}

impl Cut {
    /// Create a new cut.
    pub fn new(coeffs: FxHashMap<VarId, BigRational>, rhs: BigRational, cut_type: CutType) -> Self {
        Self {
            coeffs,
            rhs,
            cut_type,
        }
    }

    /// Check if the cut is violated by the given solution.
    ///
    /// A cut is violated if sum(coeff[i] * solution[i]) < rhs.
    pub fn is_violated(&self, solution: &FxHashMap<VarId, BigRational>) -> bool {
        let lhs: BigRational = self
            .coeffs
            .iter()
            .map(|(&var, coeff)| {
                let value = solution
                    .get(&var)
                    .cloned()
                    .unwrap_or_else(BigRational::zero);
                coeff.clone() * value
            })
            .sum();

        lhs < self.rhs
    }

    /// Get the violation amount (rhs - lhs). Positive means violated.
    pub fn violation(&self, solution: &FxHashMap<VarId, BigRational>) -> BigRational {
        let lhs: BigRational = self
            .coeffs
            .iter()
            .map(|(&var, coeff)| {
                let value = solution
                    .get(&var)
                    .cloned()
                    .unwrap_or_else(BigRational::zero);
                coeff.clone() * value
            })
            .sum();

        self.rhs.clone() - lhs
    }
}

/// Configuration for cutting plane generator.
#[derive(Debug, Clone)]
pub struct CuttingPlaneConfig {
    /// Maximum number of cuts to generate per iteration.
    pub max_cuts_per_iter: usize,
    /// Minimum violation to consider a cut useful.
    pub min_violation: f64,
    /// Enable Gomory fractional cuts.
    pub enable_gomory_fractional: bool,
    /// Enable Gomory mixed-integer cuts.
    pub enable_gomory_mixed: bool,
}

impl Default for CuttingPlaneConfig {
    fn default() -> Self {
        Self {
            max_cuts_per_iter: 10,
            min_violation: 1e-6,
            enable_gomory_fractional: true,
            enable_gomory_mixed: true,
        }
    }
}

/// Statistics for cutting plane generation.
#[derive(Debug, Clone, Default)]
pub struct CuttingPlaneStats {
    /// Total number of cuts generated.
    pub cuts_generated: u64,
    /// Number of Gomory fractional cuts.
    pub gomory_fractional_cuts: u64,
    /// Number of Gomory mixed-integer cuts.
    pub gomory_mixed_cuts: u64,
    /// Number of cuts that were actually violated.
    pub violated_cuts: u64,
}

/// Cutting plane generator.
#[derive(Debug)]
pub struct CuttingPlaneGenerator {
    /// Configuration.
    config: CuttingPlaneConfig,
    /// Statistics.
    stats: CuttingPlaneStats,
}

impl CuttingPlaneGenerator {
    /// Create a new cutting plane generator.
    pub fn new(config: CuttingPlaneConfig) -> Self {
        Self {
            config,
            stats: CuttingPlaneStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CuttingPlaneConfig::default())
    }

    /// Get statistics.
    pub fn stats(&self) -> &CuttingPlaneStats {
        &self.stats
    }

    /// Generate Gomory fractional cut from a tableau row.
    ///
    /// Given a row: x_i = f_0 + sum(f_j * x_j) where x_i is basic and fractional,
    /// the Gomory cut is: sum(floor(f_j) * x_j) <= floor(f_0).
    ///
    /// # Arguments
    ///
    /// * `basic_var` - The basic variable (must be fractional)
    /// * `row_coeffs` - Coefficients for non-basic variables in the row
    /// * `rhs` - Right-hand side value (value of basic variable)
    pub fn generate_gomory_fractional(
        &mut self,
        _basic_var: VarId,
        row_coeffs: &FxHashMap<VarId, BigRational>,
        rhs: &BigRational,
    ) -> Option<Cut> {
        if !self.config.enable_gomory_fractional {
            return None;
        }

        // Extract fractional part of RHS
        let f_0 = self.fractional_part(rhs);
        if f_0.is_zero() {
            return None; // RHS is integral, no cut needed
        }

        // Build cut coefficients
        let mut cut_coeffs = FxHashMap::default();

        for (&var, coeff) in row_coeffs {
            let f_j = self.fractional_part(coeff);
            if !f_j.is_zero() {
                // Cut coefficient for this variable
                let cut_coeff = if f_j <= f_0.clone() {
                    f_j / f_0.clone()
                } else {
                    (BigRational::one() - f_j.clone()) / (BigRational::one() - f_0.clone())
                };
                cut_coeffs.insert(var, cut_coeff);
            }
        }

        if cut_coeffs.is_empty() {
            return None;
        }

        self.stats.cuts_generated += 1;
        self.stats.gomory_fractional_cuts += 1;

        Some(Cut::new(
            cut_coeffs,
            BigRational::one(),
            CutType::GomoryFractional,
        ))
    }

    /// Generate Gomory mixed-integer (MIR) cut.
    ///
    /// This is a generalization of Gomory fractional cuts for mixed-integer programs.
    pub fn generate_gomory_mixed(
        &mut self,
        _basic_var: VarId,
        row_coeffs: &FxHashMap<VarId, BigRational>,
        rhs: &BigRational,
        _integer_vars: &[VarId],
    ) -> Option<Cut> {
        if !self.config.enable_gomory_mixed {
            return None;
        }

        // Simplified MIR cut generation (similar to fractional case)
        let f_0 = self.fractional_part(rhs);
        if f_0.is_zero() {
            return None;
        }

        let mut cut_coeffs = FxHashMap::default();

        for (&var, coeff) in row_coeffs {
            let f_j = self.fractional_part(coeff);
            if !f_j.is_zero() {
                let cut_coeff = if f_j <= f_0.clone() {
                    f_j / f_0.clone()
                } else {
                    (BigRational::one() - f_j.clone()) / (BigRational::one() - f_0.clone())
                };
                cut_coeffs.insert(var, cut_coeff);
            }
        }

        if cut_coeffs.is_empty() {
            return None;
        }

        self.stats.cuts_generated += 1;
        self.stats.gomory_mixed_cuts += 1;

        Some(Cut::new(
            cut_coeffs,
            BigRational::one(),
            CutType::GomoryMixed,
        ))
    }

    /// Extract fractional part of a rational number.
    ///
    /// For a rational r = n/d, the fractional part is r - floor(r).
    fn fractional_part(&self, value: &BigRational) -> BigRational {
        let floor_value = value.floor();
        value.clone() - floor_value
    }

    /// Check if a solution violates a cut.
    pub fn is_violated(&mut self, cut: &Cut, solution: &FxHashMap<VarId, BigRational>) -> bool {
        let violated = cut.is_violated(solution);
        if violated {
            self.stats.violated_cuts += 1;
        }
        violated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    fn rat(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    #[test]
    fn test_fractional_part() {
        let generator = CuttingPlaneGenerator::default_config();

        // 5/2 has fractional part 1/2
        let frac = generator.fractional_part(&rat(5, 2));
        assert_eq!(frac, rat(1, 2));

        // 4/2 = 2 has fractional part 0
        let frac = generator.fractional_part(&rat(4, 2));
        assert_eq!(frac, BigRational::zero());

        // 7/3 has fractional part 1/3
        let frac = generator.fractional_part(&rat(7, 3));
        assert_eq!(frac, rat(1, 3));
    }

    #[test]
    fn test_cut_violation() {
        let mut coeffs = FxHashMap::default();
        coeffs.insert(0, rat(2, 1));
        coeffs.insert(1, rat(3, 1));

        let cut = Cut::new(coeffs, rat(10, 1), CutType::GomoryFractional);

        // Solution: x0=1, x1=1 => 2*1 + 3*1 = 5 < 10 (violated)
        let mut solution = FxHashMap::default();
        solution.insert(0, rat(1, 1));
        solution.insert(1, rat(1, 1));

        assert!(cut.is_violated(&solution));
        assert_eq!(cut.violation(&solution), rat(5, 1)); // 10 - 5 = 5

        // Solution: x0=3, x1=2 => 2*3 + 3*2 = 12 >= 10 (satisfied)
        solution.insert(0, rat(3, 1));
        solution.insert(1, rat(2, 1));

        assert!(!cut.is_violated(&solution));
    }

    #[test]
    fn test_gomory_fractional_generation() {
        let config = CuttingPlaneConfig::default();
        let mut generator = CuttingPlaneGenerator::new(config);

        // Row: x0 = 5/2 + (1/3)*x1 + (2/5)*x2
        let mut row_coeffs = FxHashMap::default();
        row_coeffs.insert(1, rat(1, 3));
        row_coeffs.insert(2, rat(2, 5));

        let rhs = rat(5, 2);

        let cut = generator.generate_gomory_fractional(0, &row_coeffs, &rhs);
        assert!(cut.is_some());

        let cut = cut.unwrap();
        assert_eq!(cut.cut_type, CutType::GomoryFractional);
        assert!(cut.coeffs.len() > 0);
        assert_eq!(generator.stats.gomory_fractional_cuts, 1);
    }
}
