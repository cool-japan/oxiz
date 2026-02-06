//! Cutting Plane Methods for Mixed Integer Programming.
//!
//! Implements various cutting plane techniques including:
//! - Gomory mixed-integer cuts
//! - Lift-and-project cuts
//! - Cover inequalities

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashSet;

/// Cutting plane generator for MIP.
#[derive(Debug)]
pub struct CuttingPlaneGenerator {
    /// Integer variables
    integer_vars: FxHashSet<VarId>,
    /// Statistics
    stats: CuttingPlaneStats,
}

/// Variable identifier
pub type VarId = usize;

/// A cutting plane (linear inequality).
#[derive(Debug, Clone)]
pub struct CuttingPlane {
    /// Coefficients
    pub coeffs: Vec<(VarId, BigRational)>,
    /// Right-hand side
    pub rhs: BigRational,
    /// Type of cut
    pub cut_type: CutType,
}

/// Type of cutting plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutType {
    /// Gomory fractional cut
    Gomory,
    /// Gomory mixed-integer cut
    GomoryMI,
    /// Lift-and-project cut
    LiftProject,
    /// Cover inequality
    Cover,
    /// Clique inequality
    Clique,
}

/// Configuration for cutting plane generation.
#[derive(Debug, Clone)]
pub struct CuttingPlaneConfig {
    /// Maximum cuts to generate
    pub max_cuts: usize,
    /// Enable Gomory cuts
    pub enable_gomory: bool,
    /// Enable lift-and-project cuts
    pub enable_lift_project: bool,
    /// Enable cover cuts
    pub enable_cover: bool,
}

impl Default for CuttingPlaneConfig {
    fn default() -> Self {
        Self {
            max_cuts: 100,
            enable_gomory: true,
            enable_lift_project: true,
            enable_cover: true,
        }
    }
}

/// Cutting plane statistics.
#[derive(Debug, Clone, Default)]
pub struct CuttingPlaneStats {
    /// Total cuts generated
    pub cuts_generated: usize,
    /// Gomory cuts
    pub gomory_cuts: usize,
    /// Lift-and-project cuts
    pub lift_project_cuts: usize,
    /// Cover cuts
    pub cover_cuts: usize,
}

impl CuttingPlaneGenerator {
    /// Create a new cutting plane generator.
    pub fn new(integer_vars: FxHashSet<VarId>) -> Self {
        Self {
            integer_vars,
            stats: CuttingPlaneStats::default(),
        }
    }

    /// Generate Gomory fractional cut from a tableau row.
    pub fn generate_gomory_cut(
        &mut self,
        _basic_var: VarId,
        row: &[(VarId, BigRational)],
        rhs: &BigRational,
    ) -> Option<CuttingPlane> {
        // Extract fractional part of RHS
        let frac_rhs = self.fractional_part(rhs);
        if frac_rhs.is_zero() {
            return None; // No fractional part, no cut
        }

        let mut cut_coeffs = Vec::new();

        for (var_id, coeff) in row {
            let frac_coeff = self.fractional_part(coeff);

            if !frac_coeff.is_zero() {
                // Cut coefficient: -f_j if f_j <= f_0, else -(1-f_j)
                let cut_coeff = if frac_coeff <= frac_rhs {
                    -frac_coeff
                } else {
                    -(BigRational::one() - frac_coeff)
                };

                cut_coeffs.push((*var_id, cut_coeff));
            }
        }

        self.stats.gomory_cuts += 1;
        self.stats.cuts_generated += 1;

        Some(CuttingPlane {
            coeffs: cut_coeffs,
            rhs: -frac_rhs,
            cut_type: CutType::Gomory,
        })
    }

    /// Generate Gomory mixed-integer cut.
    pub fn generate_gomory_mi_cut(
        &mut self,
        basic_var: VarId,
        row: &[(VarId, BigRational)],
        rhs: &BigRational,
    ) -> Option<CuttingPlane> {
        if !self.integer_vars.contains(&basic_var) {
            return None; // Only for integer basic variables
        }

        let f0 = self.fractional_part(rhs);
        if f0.is_zero() {
            return None;
        }

        let mut cut_coeffs = Vec::new();

        for (var_id, coeff) in row {
            let fj = self.fractional_part(coeff);

            let cut_coeff = if self.integer_vars.contains(var_id) {
                // Integer variable
                if fj <= f0 {
                    -fj
                } else {
                    -(BigRational::one() - fj) * &f0 / (BigRational::one() - &f0)
                }
            } else {
                // Continuous variable
                if coeff >= &BigRational::zero() {
                    -coeff
                } else {
                    coeff * &f0 / (BigRational::one() - &f0)
                }
            };

            if !cut_coeff.is_zero() {
                cut_coeffs.push((*var_id, cut_coeff));
            }
        }

        self.stats.gomory_cuts += 1;
        self.stats.cuts_generated += 1;

        Some(CuttingPlane {
            coeffs: cut_coeffs,
            rhs: -f0,
            cut_type: CutType::GomoryMI,
        })
    }

    /// Generate lift-and-project cut.
    pub fn generate_lift_project_cut(
        &mut self,
        constraint: &[(VarId, BigRational)],
        rhs: &BigRational,
        lifting_var: VarId,
    ) -> Option<CuttingPlane> {
        // Simplified lift-and-project
        if !self.integer_vars.contains(&lifting_var) {
            return None;
        }

        // Generate disjunctive cut: x_j = 0 or x_j = 1
        let mut cut_coeffs = Vec::new();

        for (var_id, coeff) in constraint {
            if *var_id == lifting_var {
                continue;
            }

            // Lift coefficient
            let lifted_coeff = coeff.clone();
            cut_coeffs.push((*var_id, lifted_coeff));
        }

        self.stats.lift_project_cuts += 1;
        self.stats.cuts_generated += 1;

        Some(CuttingPlane {
            coeffs: cut_coeffs,
            rhs: rhs.clone(),
            cut_type: CutType::LiftProject,
        })
    }

    /// Generate cover inequality from a knapsack constraint.
    pub fn generate_cover_cut(
        &mut self,
        weights: &[(VarId, BigRational)],
        capacity: &BigRational,
    ) -> Option<CuttingPlane> {
        // Find minimal cover
        let cover = self.find_minimal_cover(weights, capacity)?;

        // Generate cover inequality: sum_{j in C} x_j <= |C| - 1
        let mut cut_coeffs = Vec::new();
        for var_id in &cover {
            cut_coeffs.push((*var_id, BigRational::one()));
        }

        let rhs = BigRational::from_integer(BigInt::from(cover.len() as i64 - 1));

        self.stats.cover_cuts += 1;
        self.stats.cuts_generated += 1;

        Some(CuttingPlane {
            coeffs: cut_coeffs,
            rhs,
            cut_type: CutType::Cover,
        })
    }

    /// Find minimal cover for knapsack.
    fn find_minimal_cover(
        &self,
        weights: &[(VarId, BigRational)],
        capacity: &BigRational,
    ) -> Option<Vec<VarId>> {
        // Greedy approach: sort by weight descending
        let mut sorted_weights = weights.to_vec();
        sorted_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cover = Vec::new();
        let mut total_weight = BigRational::zero();

        for (var_id, weight) in sorted_weights {
            cover.push(var_id);
            total_weight = &total_weight + &weight;

            if total_weight > *capacity {
                return Some(cover);
            }
        }

        None // No cover found
    }

    /// Extract fractional part of a rational number.
    fn fractional_part(&self, value: &BigRational) -> BigRational {
        // frac(x) = x - floor(x)
        let floor_val = self.floor(value);
        value - &floor_val
    }

    /// Floor function for rationals.
    fn floor(&self, value: &BigRational) -> BigRational {
        let numer = value.numer();
        let denom = value.denom();

        let quotient = numer / denom;
        BigRational::from_integer(quotient)
    }

    /// Get statistics.
    pub fn stats(&self) -> &CuttingPlaneStats {
        &self.stats
    }

    /// Check if a cut is violated by current solution.
    pub fn is_violated(&self, cut: &CuttingPlane, solution: &[(VarId, BigRational)]) -> bool {
        let mut lhs = BigRational::zero();

        for (var_id, coeff) in &cut.coeffs {
            if let Some((_, value)) = solution.iter().find(|(id, _)| id == var_id) {
                lhs += coeff * value;
            }
        }

        lhs > cut.rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cutting_plane_generator() {
        let mut integer_vars = FxHashSet::default();
        integer_vars.insert(0);
        integer_vars.insert(1);

        let generator = CuttingPlaneGenerator::new(integer_vars);
        assert_eq!(generator.stats.cuts_generated, 0);
    }

    #[test]
    fn test_fractional_part() {
        let integer_vars = FxHashSet::default();
        let generator = CuttingPlaneGenerator::new(integer_vars);

        let val = BigRational::new(BigInt::from(7), BigInt::from(3)); // 7/3 = 2.333...
        let frac = generator.fractional_part(&val);

        // Should be 1/3
        assert_eq!(frac, BigRational::new(BigInt::from(1), BigInt::from(3)));
    }
}
