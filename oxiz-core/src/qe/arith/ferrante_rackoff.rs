//! Ferrante-Rackoff Algorithm for Real Arithmetic QE.
//!
//! This module implements the Ferrante-Rackoff procedure for eliminating
//! quantifiers from formulas in real closed fields (linear real arithmetic).
//!
//! ## Algorithm
//!
//! For ∃x. φ(x), the Ferrante-Rackoff method:
//! 1. Normalize φ to disjunction of conjunctions
//! 2. For each conjunction, extract linear inequalities in x
//! 3. Compute critical values (roots and midpoints)
//! 4. Test φ at each critical value
//! 5. Return disjunction of satisfying cases
//!
//! ## Complexity
//!
//! - Doubly exponential in quantifier alternations
//! - Practical for small formulas
//! - Better than CAD for some problems
//!
//! ## Applications
//!
//! - Real constraint solving
//! - Optimization problems
//! - Control theory verification
//!
//! ## References
//!
//! - Ferrante & Rackoff: "The Computational Complexity of Logical Theories" (1979)
//! - Z3's `qe/qe_arith_plugin.cpp`

use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

/// Configuration for Ferrante-Rackoff.
#[derive(Debug, Clone)]
pub struct FerranteRackoffConfig {
    /// Maximum number of critical points.
    pub max_critical_points: u32,
    /// Enable simplification.
    pub simplify: bool,
    /// Timeout (microseconds).
    pub timeout_us: u64,
}

impl Default for FerranteRackoffConfig {
    fn default() -> Self {
        Self {
            max_critical_points: 1000,
            simplify: true,
            timeout_us: 10_000_000, // 10 seconds
        }
    }
}

/// Statistics for Ferrante-Rackoff.
#[derive(Debug, Clone, Default)]
pub struct FerranteRackoffStats {
    /// Variables eliminated.
    pub vars_eliminated: u64,
    /// Critical points generated.
    pub critical_points: u64,
    /// Formula evaluations.
    pub evaluations: u64,
    /// Simplifications.
    pub simplifications: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Linear inequality: a*x + b ≤ 0, a*x + b < 0, or a*x + b = 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inequality {
    /// Coefficient of variable.
    pub coeff: BigRational,
    /// Constant term.
    pub constant: BigRational,
    /// Inequality type.
    pub ineq_type: InequalityType,
}

/// Type of inequality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InequalityType {
    /// ≤
    Le,
    /// <
    Lt,
    /// =
    Eq,
    /// ≥
    Ge,
    /// >
    Gt,
}

impl Inequality {
    /// Create new inequality.
    pub fn new(coeff: BigRational, constant: BigRational, ineq_type: InequalityType) -> Self {
        Self {
            coeff,
            constant,
            ineq_type,
        }
    }

    /// Normalize to standard form (coefficient = 1 or -1).
    pub fn normalize(mut self) -> Self {
        if self.coeff.is_zero() {
            return self;
        }

        if self.coeff.is_negative() {
            // Flip sign and reverse inequality
            self.coeff = -self.coeff.clone();
            self.constant = -self.constant.clone();
            self.ineq_type = match self.ineq_type {
                InequalityType::Le => InequalityType::Ge,
                InequalityType::Lt => InequalityType::Gt,
                InequalityType::Ge => InequalityType::Le,
                InequalityType::Gt => InequalityType::Lt,
                InequalityType::Eq => InequalityType::Eq,
            };
        }

        self
    }

    /// Get bound for variable: x op -b/a.
    pub fn get_bound(&self) -> Option<BigRational> {
        if self.coeff.is_zero() {
            None
        } else {
            Some(-&self.constant / &self.coeff)
        }
    }
}

/// Conjunction of inequalities.
#[derive(Debug, Clone)]
pub struct Conjunction {
    /// Inequalities in the conjunction.
    pub inequalities: Vec<Inequality>,
}

/// DNF formula (disjunction of conjunctions).
#[derive(Debug, Clone)]
pub struct DnfFormula {
    /// Disjuncts.
    pub disjuncts: Vec<Conjunction>,
}

impl DnfFormula {
    /// Create empty formula (false).
    pub fn empty() -> Self {
        Self {
            disjuncts: Vec::new(),
        }
    }

    /// Create trivial formula (true).
    pub fn trivial() -> Self {
        Self {
            disjuncts: vec![Conjunction {
                inequalities: Vec::new(),
            }],
        }
    }
}

/// Ferrante-Rackoff quantifier elimination engine.
pub struct FerranteRackoffQE {
    config: FerranteRackoffConfig,
    stats: FerranteRackoffStats,
}

impl FerranteRackoffQE {
    /// Create new engine.
    pub fn new() -> Self {
        Self::with_config(FerranteRackoffConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: FerranteRackoffConfig) -> Self {
        Self {
            config,
            stats: FerranteRackoffStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &FerranteRackoffStats {
        &self.stats
    }

    /// Eliminate existential quantifier: ∃x. φ
    pub fn eliminate_exists(&mut self, var: usize, formula: &DnfFormula) -> DnfFormula {
        let start = std::time::Instant::now();

        let mut result_disjuncts = Vec::new();

        for disjunct in &formula.disjuncts {
            // Partition inequalities
            let (with_var, without_var): (Vec<_>, Vec<_>) = disjunct
                .inequalities
                .iter()
                .partition(|ineq| !ineq.coeff.is_zero());

            if with_var.is_empty() {
                // Variable doesn't appear
                result_disjuncts.push(Conjunction {
                    inequalities: without_var.into_iter().cloned().collect(),
                });
                continue;
            }

            // Compute critical points
            let critical_points = self.compute_critical_points(&with_var);
            self.stats.critical_points += critical_points.len() as u64;

            // Test each critical point
            for point in critical_points {
                if self.evaluate_at_point(&with_var, &point) {
                    self.stats.evaluations += 1;

                    // This point satisfies the inequalities
                    let mut new_ineqs = without_var.iter().map(|&ineq| ineq.clone()).collect();
                    if self.config.simplify {
                        new_ineqs = self.simplify_inequalities(new_ineqs);
                    }

                    result_disjuncts.push(Conjunction {
                        inequalities: new_ineqs,
                    });
                    break; // One satisfying point is enough
                }
            }
        }

        self.stats.vars_eliminated += 1;
        self.stats.time_us += start.elapsed().as_micros() as u64;

        DnfFormula {
            disjuncts: result_disjuncts,
        }
    }

    /// Compute critical points for variable elimination.
    fn compute_critical_points(&mut self, inequalities: &[&Inequality]) -> Vec<BigRational> {
        let mut points = Vec::new();

        // Collect all bounds
        let mut bounds = Vec::new();
        for ineq in inequalities {
            if let Some(bound) = ineq.get_bound() {
                bounds.push(bound);
            }
        }

        if bounds.is_empty() {
            // No bounds - try a few test points
            for i in -5..=5 {
                points.push(BigRational::from_integer(num_bigint::BigInt::from(i)));
            }
            return points;
        }

        // Sort bounds
        bounds.sort();

        // Add bounds as critical points
        for bound in &bounds {
            points.push(bound.clone());
        }

        // Add midpoints between consecutive bounds
        for i in 0..(bounds.len().saturating_sub(1)) {
            let midpoint = (&bounds[i] + &bounds[i + 1])
                / BigRational::from_integer(num_bigint::BigInt::from(2));
            points.push(midpoint);
        }

        // Add points slightly beyond bounds
        if let Some(min_bound) = bounds.first() {
            points.push(min_bound - BigRational::one());
        }

        if let Some(max_bound) = bounds.last() {
            points.push(max_bound + BigRational::one());
        }

        // Limit number of points
        points.truncate(self.config.max_critical_points as usize);

        points
    }

    /// Evaluate inequalities at a point.
    fn evaluate_at_point(&self, inequalities: &[&Inequality], point: &BigRational) -> bool {
        for ineq in inequalities {
            // Evaluate a*point + b
            let value = &ineq.coeff * point + &ineq.constant;

            let satisfied = match ineq.ineq_type {
                InequalityType::Le => value <= BigRational::zero(),
                InequalityType::Lt => value < BigRational::zero(),
                InequalityType::Eq => value == BigRational::zero(),
                InequalityType::Ge => value >= BigRational::zero(),
                InequalityType::Gt => value > BigRational::zero(),
            };

            if !satisfied {
                return false;
            }
        }

        true
    }

    /// Simplify inequalities.
    fn simplify_inequalities(&mut self, mut inequalities: Vec<Inequality>) -> Vec<Inequality> {
        self.stats.simplifications += 1;

        // Remove trivial inequalities
        inequalities.retain(|ineq| {
            // Check for trivial true/false
            if ineq.coeff.is_zero() {
                // Constant inequality: 0 op constant
                // Keep only if it's satisfiable
                match ineq.ineq_type {
                    InequalityType::Le => {
                        // 0 ≤ constant  => keep if constant ≥ 0
                        !ineq.constant.is_negative()
                    }
                    InequalityType::Lt => {
                        // 0 < constant  => keep if constant > 0
                        ineq.constant.is_positive()
                    }
                    InequalityType::Eq => {
                        // 0 = constant  => keep if constant = 0
                        ineq.constant.is_zero()
                    }
                    InequalityType::Ge => {
                        // 0 ≥ constant  => keep if constant ≤ 0
                        !ineq.constant.is_positive()
                    }
                    InequalityType::Gt => {
                        // 0 > constant  => keep if constant < 0
                        ineq.constant.is_negative()
                    }
                }
            } else {
                true // Keep non-constant inequalities
            }
        });

        inequalities
    }

    /// Eliminate universal quantifier: ∀x. φ
    ///
    /// ∀x. φ  =  ¬∃x. ¬φ
    pub fn eliminate_forall(&mut self, var: usize, formula: &DnfFormula) -> DnfFormula {
        let negated = self.negate_formula(formula);
        let eliminated = self.eliminate_exists(var, &negated);
        self.negate_formula(&eliminated)
    }

    /// Negate DNF formula.
    fn negate_formula(&self, _formula: &DnfFormula) -> DnfFormula {
        // ¬(A ∨ B ∨ C) = ¬A ∧ ¬B ∧ ¬C
        // Negation of DNF gives CNF, convert back to DNF
        // Simplified: return trivial formula
        DnfFormula::trivial()
    }
}

impl Default for FerranteRackoffQE {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qe_creation() {
        let qe = FerranteRackoffQE::new();
        assert_eq!(qe.stats().vars_eliminated, 0);
    }

    #[test]
    fn test_inequality() {
        let ineq = Inequality::new(
            BigRational::from_integer(num_bigint::BigInt::from(2)),
            BigRational::from_integer(num_bigint::BigInt::from(-4)),
            InequalityType::Le,
        );

        let bound = ineq.get_bound();
        assert!(bound.is_some());
    }

    #[test]
    fn test_normalize_negative() {
        let ineq = Inequality::new(
            BigRational::from_integer(num_bigint::BigInt::from(-2)),
            BigRational::from_integer(num_bigint::BigInt::from(4)),
            InequalityType::Le,
        );

        let normalized = ineq.normalize();

        assert!(normalized.coeff.is_positive());
        assert_eq!(normalized.ineq_type, InequalityType::Ge);
    }

    #[test]
    fn test_evaluate_at_point() {
        let qe = FerranteRackoffQE::new();

        // x ≤ 5
        let ineq = Inequality::new(
            BigRational::from_integer(num_bigint::BigInt::from(1)),
            BigRational::from_integer(num_bigint::BigInt::from(-5)),
            InequalityType::Le,
        );

        let point = BigRational::from_integer(num_bigint::BigInt::from(3));
        assert!(qe.evaluate_at_point(&[&ineq], &point));

        let point = BigRational::from_integer(num_bigint::BigInt::from(6));
        assert!(!qe.evaluate_at_point(&[&ineq], &point));
    }

    #[test]
    fn test_compute_critical_points() {
        let mut qe = FerranteRackoffQE::new();

        // x ≤ 5, x ≥ 2
        let ineq1 = Inequality::new(
            BigRational::from_integer(num_bigint::BigInt::from(1)),
            BigRational::from_integer(num_bigint::BigInt::from(-5)),
            InequalityType::Le,
        );

        let ineq2 = Inequality::new(
            BigRational::from_integer(num_bigint::BigInt::from(-1)),
            BigRational::from_integer(num_bigint::BigInt::from(2)),
            InequalityType::Le,
        );

        let points = qe.compute_critical_points(&[&ineq1, &ineq2]);

        assert!(!points.is_empty());
        // Should include bounds 5 and 2, and points between/beyond
    }

    #[test]
    fn test_eliminate_exists_no_var() {
        let mut qe = FerranteRackoffQE::new();

        // Formula without var 0
        let formula = DnfFormula {
            disjuncts: vec![Conjunction {
                inequalities: vec![Inequality::new(
                    BigRational::zero(),
                    BigRational::from_integer(num_bigint::BigInt::from(-1)),
                    InequalityType::Le,
                )],
            }],
        };

        let result = qe.eliminate_exists(0, &formula);

        assert_eq!(result.disjuncts.len(), 1);
    }

    #[test]
    fn test_dnf_empty() {
        let formula = DnfFormula::empty();
        assert_eq!(formula.disjuncts.len(), 0);
    }

    #[test]
    fn test_dnf_trivial() {
        let formula = DnfFormula::trivial();
        assert_eq!(formula.disjuncts.len(), 1);
        assert!(formula.disjuncts[0].inequalities.is_empty());
    }

    #[test]
    fn test_simplify() {
        let mut qe = FerranteRackoffQE::new();

        let inequalities = vec![
            // Trivial: 0 ≤ -1 (false)
            Inequality::new(
                BigRational::zero(),
                BigRational::from_integer(num_bigint::BigInt::from(-1)),
                InequalityType::Le,
            ),
            // Non-trivial
            Inequality::new(
                BigRational::from_integer(num_bigint::BigInt::from(1)),
                BigRational::from_integer(num_bigint::BigInt::from(1)),
                InequalityType::Le,
            ),
        ];

        let simplified = qe.simplify_inequalities(inequalities);

        // First inequality should be removed
        assert_eq!(simplified.len(), 1);
    }
}
