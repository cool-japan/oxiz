//! Cutting planes for integer programming.
//!
//! This module implements cutting plane techniques to strengthen
//! integer constraints and eliminate fractional solutions.
//!
//! ## Key Techniques
//!
//! - **Gomory Cuts**: Cutting planes derived from simplex tableau
//! - **Split Cuts**: Disjunctive cuts based on variable splits
//! - **Cover Cuts**: Cuts based on minimal covers
//!
//! ## Reference
//!
//! - Gomory (1958): "Outline of an algorithm for integer solutions to linear programs"
//! - Cutting planes in mixed-integer programming

use crate::solver::Model;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive};
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;

/// Type of cutting plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutType {
    /// Gomory fractional cut.
    GomoryFractional,
    /// Gomory mixed-integer cut.
    GomoryMixed,
    /// Split cut (disjunctive).
    Split,
    /// Cover cut.
    Cover,
}

/// A cutting plane constraint.
#[derive(Debug, Clone)]
pub struct CuttingPlane {
    /// The polynomial constraint (p >= 0).
    pub poly: Polynomial,
    /// Type of cut.
    pub cut_type: CutType,
    /// Variables involved.
    pub vars: Vec<Var>,
}

impl CuttingPlane {
    /// Create a new cutting plane.
    pub fn new(poly: Polynomial, cut_type: CutType) -> Self {
        let vars = poly.vars().into_iter().collect();
        Self {
            poly,
            cut_type,
            vars,
        }
    }

    /// Check if a model satisfies this cutting plane.
    /// The cutting plane constraint is poly >= 0.
    pub fn is_satisfied(&self, model: &Model) -> bool {
        // Check if all variables in the polynomial are assigned in the model
        for &var in &self.vars {
            if model.arith_value(var).is_none() {
                return false; // Cannot evaluate if variable is unassigned
            }
        }

        // Convert HashMap to FxHashMap for evaluation
        let assignment: FxHashMap<Var, BigRational> = model
            .arith_values
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        // Evaluate the polynomial
        let value = self.poly.eval(&assignment);

        // Check if the constraint poly >= 0 is satisfied
        !value.is_negative()
    }
}

/// Cutting plane generator.
pub struct CuttingPlaneGenerator {
    /// Tolerance for fractional values.
    tolerance: f64,
    /// Maximum number of cuts to generate per call.
    max_cuts: usize,
}

impl Default for CuttingPlaneGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl CuttingPlaneGenerator {
    /// Create a new cutting plane generator with default settings.
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            max_cuts: 10,
        }
    }

    /// Create with custom settings.
    pub fn with_config(tolerance: f64, max_cuts: usize) -> Self {
        Self {
            tolerance,
            max_cuts,
        }
    }

    /// Generate Gomory fractional cuts from a fractional solution.
    ///
    /// For a basic variable x_i with fractional value, we generate a cut:
    /// sum_j f_j * x_j >= f_i
    /// where f_i is the fractional part of x_i.
    pub fn generate_gomory_cuts(&self, model: &Model, integer_vars: &[Var]) -> Vec<CuttingPlane> {
        let mut cuts = Vec::new();

        for &var in integer_vars {
            if cuts.len() >= self.max_cuts {
                break;
            }

            if let Some(value) = model.arith_value(var) {
                let frac = self.fractional_part(value);

                // Only generate cut if value is fractional
                if frac > self.tolerance
                    && frac < (1.0 - self.tolerance)
                    && let Some(cut) = self.generate_gomory_cut_for_var(var, value, frac)
                {
                    cuts.push(cut);
                }
            }
        }

        cuts
    }

    /// Generate a Gomory cut for a specific variable.
    fn generate_gomory_cut_for_var(
        &self,
        var: Var,
        value: &BigRational,
        _frac: f64,
    ) -> Option<CuttingPlane> {
        // Simplified Gomory cut: x - floor(value) >= frac
        // Rearranged: x >= ceil(value)
        // Or equivalently: ceil(value) - x <= 0
        // Which means: NOT(x - ceil(value) < 0)

        let floor_int = value.numer() / value.denom();
        let ceil_int = if &floor_int * value.denom() == *value.numer() {
            floor_int.clone()
        } else {
            floor_int + num_bigint::BigInt::one()
        };

        let ceil_val = BigRational::from_integer(ceil_int);

        // Create polynomial: x - ceil(value)
        let x = Polynomial::from_var(var);
        let poly = Polynomial::sub(&x, &Polynomial::constant(ceil_val));

        // The cut is: x - ceil(value) >= 0, i.e., poly >= 0
        // We'll return this as a constraint
        Some(CuttingPlane::new(poly, CutType::GomoryFractional))
    }

    /// Generate split cuts based on variable bounds.
    ///
    /// For an integer variable x with fractional value v:
    /// - Either x <= floor(v), or
    /// - x >= ceil(v)
    ///
    /// This creates a disjunction that can be turned into cuts.
    pub fn generate_split_cuts(&self, model: &Model, integer_vars: &[Var]) -> Vec<CuttingPlane> {
        let mut cuts = Vec::new();

        for &var in integer_vars {
            if cuts.len() >= self.max_cuts {
                break;
            }

            if let Some(value) = model.arith_value(var) {
                let frac = self.fractional_part(value);

                if frac > self.tolerance && frac < (1.0 - self.tolerance) {
                    // Generate both split cuts
                    let (floor_val, ceil_val) = self.floor_ceil(value);

                    // Cut 1: x <= floor  =>  floor - x >= 0
                    let x = Polynomial::from_var(var);
                    let poly1 = Polynomial::sub(&Polynomial::constant(floor_val), &x);
                    cuts.push(CuttingPlane::new(poly1, CutType::Split));

                    // Cut 2: x >= ceil  =>  x - ceil >= 0
                    let poly2 = Polynomial::sub(&x, &Polynomial::constant(ceil_val));
                    cuts.push(CuttingPlane::new(poly2, CutType::Split));

                    if cuts.len() >= self.max_cuts {
                        break;
                    }
                }
            }
        }

        cuts
    }

    /// Get the fractional part of a rational number.
    fn fractional_part(&self, value: &BigRational) -> f64 {
        let val_f64 = value.numer().to_f64().unwrap_or(0.0) / value.denom().to_f64().unwrap_or(1.0);
        (val_f64 - val_f64.floor()).abs()
    }

    /// Compute floor and ceiling of a rational number.
    fn floor_ceil(&self, value: &BigRational) -> (BigRational, BigRational) {
        let floor_int = value.numer() / value.denom();
        let floor_val = BigRational::from_integer(floor_int.clone());

        let ceil_val = if &floor_val == value {
            floor_val.clone()
        } else {
            floor_val.clone() + BigRational::one()
        };

        (floor_val, ceil_val)
    }

    /// Generate all applicable cuts.
    pub fn generate_all_cuts(&self, model: &Model, integer_vars: &[Var]) -> Vec<CuttingPlane> {
        let mut all_cuts = Vec::new();

        // Gomory cuts
        let gomory_cuts = self.generate_gomory_cuts(model, integer_vars);
        all_cuts.extend(gomory_cuts);

        // Split cuts (if we still have room)
        if all_cuts.len() < self.max_cuts {
            let remaining = self.max_cuts - all_cuts.len();
            let generator = Self::with_config(self.tolerance, remaining);
            let split_cuts = generator.generate_split_cuts(model, integer_vars);
            all_cuts.extend(split_cuts);
        }

        all_cuts
    }
}

/// Statistics for cutting plane generation.
#[derive(Debug, Clone, Default)]
pub struct CutStats {
    /// Number of Gomory cuts generated.
    pub gomory_cuts: usize,
    /// Number of split cuts generated.
    pub split_cuts: usize,
    /// Number of cuts applied.
    pub cuts_applied: usize,
    /// Number of cuts that were violated.
    pub violated_cuts: usize,
}

impl CutStats {
    /// Create new statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cut generation.
    pub fn record_cut(&mut self, cut_type: CutType) {
        match cut_type {
            CutType::GomoryFractional | CutType::GomoryMixed => self.gomory_cuts += 1,
            CutType::Split => self.split_cuts += 1,
            CutType::Cover => {}
        }
    }

    /// Record a cut application.
    pub fn record_applied(&mut self) {
        self.cuts_applied += 1;
    }

    /// Record a violated cut.
    pub fn record_violated(&mut self) {
        self.violated_cuts += 1;
    }

    /// Total cuts generated.
    pub fn total_cuts(&self) -> usize {
        self.gomory_cuts + self.split_cuts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    fn rat_frac(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    fn create_model_with_value(var: Var, value: BigRational) -> Model {
        let mut arith_values = HashMap::new();
        arith_values.insert(var, value);
        Model {
            bool_values: HashMap::new(),
            arith_values,
        }
    }

    #[test]
    fn test_cutting_plane_new() {
        let x = Polynomial::from_var(0);
        let poly = Polynomial::sub(&x, &Polynomial::constant(rat(1)));
        let cut = CuttingPlane::new(poly.clone(), CutType::GomoryFractional);

        assert_eq!(cut.cut_type, CutType::GomoryFractional);
        assert_eq!(cut.poly, poly);
        assert!(cut.vars.contains(&0));
    }

    #[test]
    fn test_generator_fractional_part() {
        let generator = CuttingPlaneGenerator::new();

        let val1 = rat_frac(5, 2); // 2.5
        let frac1 = generator.fractional_part(&val1);
        assert!((frac1 - 0.5).abs() < 0.01);

        let val2 = rat(3); // 3.0
        let frac2 = generator.fractional_part(&val2);
        assert!(frac2 < 0.01);
    }

    #[test]
    fn test_generator_floor_ceil() {
        let generator = CuttingPlaneGenerator::new();

        let val = rat_frac(7, 2); // 3.5
        let (floor, ceil) = generator.floor_ceil(&val);

        assert_eq!(floor, rat(3));
        assert_eq!(ceil, rat(4));
    }

    #[test]
    fn test_generate_gomory_cuts() {
        let generator = CuttingPlaneGenerator::new();

        // Model with x = 2.5 (fractional)
        let model = create_model_with_value(0, rat_frac(5, 2));
        let integer_vars = vec![0];

        let cuts = generator.generate_gomory_cuts(&model, &integer_vars);
        assert!(!cuts.is_empty());
        assert_eq!(cuts[0].cut_type, CutType::GomoryFractional);
    }

    #[test]
    fn test_generate_gomory_cuts_integer() {
        let generator = CuttingPlaneGenerator::new();

        // Model with x = 3 (integer, no cut needed)
        let model = create_model_with_value(0, rat(3));
        let integer_vars = vec![0];

        let cuts = generator.generate_gomory_cuts(&model, &integer_vars);
        assert!(cuts.is_empty()); // No cuts for integer values
    }

    #[test]
    fn test_generate_split_cuts() {
        let generator = CuttingPlaneGenerator::new();

        // Model with x = 2.5 (fractional)
        let model = create_model_with_value(0, rat_frac(5, 2));
        let integer_vars = vec![0];

        let cuts = generator.generate_split_cuts(&model, &integer_vars);
        assert_eq!(cuts.len(), 2); // Two split cuts per fractional variable
        assert_eq!(cuts[0].cut_type, CutType::Split);
        assert_eq!(cuts[1].cut_type, CutType::Split);
    }

    #[test]
    fn test_generate_all_cuts() {
        let generator = CuttingPlaneGenerator::with_config(1e-6, 10);

        // Model with x = 2.5 (fractional)
        let model = create_model_with_value(0, rat_frac(5, 2));
        let integer_vars = vec![0];

        let cuts = generator.generate_all_cuts(&model, &integer_vars);
        assert!(!cuts.is_empty());
        // Should have Gomory + split cuts
    }

    #[test]
    fn test_cut_stats() {
        let mut stats = CutStats::new();

        assert_eq!(stats.total_cuts(), 0);

        stats.record_cut(CutType::GomoryFractional);
        assert_eq!(stats.gomory_cuts, 1);
        assert_eq!(stats.total_cuts(), 1);

        stats.record_cut(CutType::Split);
        assert_eq!(stats.split_cuts, 1);
        assert_eq!(stats.total_cuts(), 2);

        stats.record_applied();
        assert_eq!(stats.cuts_applied, 1);

        stats.record_violated();
        assert_eq!(stats.violated_cuts, 1);
    }

    #[test]
    fn test_max_cuts_limit() {
        let generator = CuttingPlaneGenerator::with_config(1e-6, 2);

        // Model with multiple fractional variables
        let mut arith_values = HashMap::new();
        arith_values.insert(0, rat_frac(5, 2)); // 2.5
        arith_values.insert(1, rat_frac(7, 2)); // 3.5
        arith_values.insert(2, rat_frac(9, 2)); // 4.5

        let model = Model {
            bool_values: HashMap::new(),
            arith_values,
        };

        let integer_vars = vec![0, 1, 2];
        let cuts = generator.generate_gomory_cuts(&model, &integer_vars);

        // Should be limited by max_cuts
        assert!(cuts.len() <= 2);
    }
}
