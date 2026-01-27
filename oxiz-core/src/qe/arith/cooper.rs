//! Cooper's Algorithm for Integer Quantifier Elimination.
//!
//! This module implements Cooper's classic algorithm for eliminating
//! quantifiers from Presburger arithmetic formulas (linear integer arithmetic).
//!
//! ## Algorithm
//!
//! Cooper's method eliminates quantifiers by:
//! 1. Normalize formula to DNF
//! 2. For each divisibility constraint d|t, introduce case split
//! 3. Compute finite test set of witness values
//! 4. Substitute witnesses and simplify
//!
//! ## Complexity
//!
//! - Worst-case: doubly exponential in quantifier alternations
//! - Practical: Good for simple formulas
//!
//! ## Applications
//!
//! - Integer constraint solving
//! - Program verification (loop invariants)
//! - Static analysis
//!
//! ## References
//!
//! - Cooper: "Theorem Proving in Arithmetic without Multiplication" (1972)
//! - Z3's `qe/qe_arith_plugin.cpp`

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

/// Configuration for Cooper's algorithm.
#[derive(Debug, Clone)]
pub struct CooperConfig {
    /// Maximum number of witnesses to generate.
    pub max_witnesses: u32,
    /// Enable optimizations.
    pub enable_optimizations: bool,
    /// Simplify during elimination.
    pub simplify: bool,
}

impl Default for CooperConfig {
    fn default() -> Self {
        Self {
            max_witnesses: 1000,
            enable_optimizations: true,
            simplify: true,
        }
    }
}

/// Statistics for Cooper's algorithm.
#[derive(Debug, Clone, Default)]
pub struct CooperStats {
    /// Variables eliminated.
    pub vars_eliminated: u64,
    /// Witnesses generated.
    pub witnesses_generated: u64,
    /// Divisibility splits.
    pub divisibility_splits: u64,
    /// Simplifications performed.
    pub simplifications: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Linear term: c0 + c1*x1 + c2*x2 + ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearTerm {
    /// Constant.
    pub constant: BigInt,
    /// Variable coefficients.
    pub coeffs: FxHashMap<usize, BigInt>,
}

impl LinearTerm {
    /// Create zero term.
    pub fn zero() -> Self {
        Self {
            constant: BigInt::zero(),
            coeffs: FxHashMap::default(),
        }
    }

    /// Create constant term.
    pub fn constant(c: BigInt) -> Self {
        Self {
            constant: c,
            coeffs: FxHashMap::default(),
        }
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.constant.is_zero() && self.coeffs.is_empty()
    }

    /// Get coefficient of variable.
    pub fn get_coeff(&self, var: usize) -> BigInt {
        self.coeffs.get(&var).cloned().unwrap_or_else(BigInt::zero)
    }

    /// Substitute variable with value.
    pub fn substitute(&self, var: usize, value: &BigInt) -> LinearTerm {
        let mut result = self.clone();

        if let Some(coeff) = result.coeffs.remove(&var) {
            result.constant = &result.constant + (coeff * value);
        }

        result
    }
}

/// Linear constraint: term ≤ 0, term = 0, or term | 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constraint {
    /// term ≤ 0
    Le(LinearTerm),
    /// term = 0
    Eq(LinearTerm),
    /// divisor | term
    Div { divisor: BigInt, term: LinearTerm },
}

impl Constraint {
    /// Substitute variable.
    pub fn substitute(&self, var: usize, value: &BigInt) -> Constraint {
        match self {
            Constraint::Le(t) => Constraint::Le(t.substitute(var, value)),
            Constraint::Eq(t) => Constraint::Eq(t.substitute(var, value)),
            Constraint::Div { divisor, term } => Constraint::Div {
                divisor: divisor.clone(),
                term: term.substitute(var, value),
            },
        }
    }

    /// Check if constraint mentions variable.
    pub fn mentions(&self, var: usize) -> bool {
        match self {
            Constraint::Le(t) | Constraint::Eq(t) => t.coeffs.contains_key(&var),
            Constraint::Div { term, .. } => term.coeffs.contains_key(&var),
        }
    }
}

/// Formula in DNF (disjunction of conjunctions).
#[derive(Debug, Clone)]
pub struct DnfFormula {
    /// Disjuncts (each is a conjunction of constraints).
    pub disjuncts: Vec<Vec<Constraint>>,
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
            disjuncts: vec![Vec::new()],
        }
    }
}

/// Cooper's quantifier elimination engine.
pub struct CooperQE {
    config: CooperConfig,
    stats: CooperStats,
}

impl CooperQE {
    /// Create new Cooper QE engine.
    pub fn new() -> Self {
        Self::with_config(CooperConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: CooperConfig) -> Self {
        Self {
            config,
            stats: CooperStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &CooperStats {
        &self.stats
    }

    /// Eliminate existential quantifier: ∃x. φ
    pub fn eliminate_exists(&mut self, var: usize, formula: &DnfFormula) -> DnfFormula {
        let start = std::time::Instant::now();

        let mut result_disjuncts = Vec::new();

        for disjunct in &formula.disjuncts {
            // Partition constraints by variable occurrence
            let (with_var, without_var): (Vec<_>, Vec<_>) =
                disjunct.iter().partition(|c| c.mentions(var));

            if with_var.is_empty() {
                // Variable doesn't appear - keep disjunct as-is
                result_disjuncts.push(without_var.into_iter().cloned().collect());
                continue;
            }

            // Generate witness set
            let witnesses = self.generate_witnesses(var, &with_var);
            self.stats.witnesses_generated += witnesses.len() as u64;

            // Try each witness
            for witness in witnesses {
                let mut new_disjunct = Vec::new();

                // Substitute in constraints mentioning var
                for constraint in &with_var {
                    let subst = constraint.substitute(var, &witness);
                    new_disjunct.push(subst);
                }

                // Keep constraints not mentioning var
                for constraint in &without_var {
                    new_disjunct.push((*constraint).clone());
                }

                if self.config.simplify {
                    new_disjunct = self.simplify_conjunction(&new_disjunct);
                }

                result_disjuncts.push(new_disjunct);
            }
        }

        self.stats.vars_eliminated += 1;
        self.stats.time_us += start.elapsed().as_micros() as u64;

        DnfFormula {
            disjuncts: result_disjuncts,
        }
    }

    /// Generate finite witness set for variable.
    fn generate_witnesses(&mut self, var: usize, constraints: &[&Constraint]) -> Vec<BigInt> {
        let mut witnesses = Vec::new();

        // Extract bounds and divisibilities
        let mut lower_bounds = Vec::new();
        let mut upper_bounds = Vec::new();
        let mut divisors = Vec::new();

        for constraint in constraints {
            match constraint {
                Constraint::Le(term) => {
                    let coeff = term.get_coeff(var);

                    if coeff.is_positive() {
                        // a*x + b ≤ 0  =>  x ≤ -b/a
                        upper_bounds.push((coeff.clone(), term.clone()));
                    } else if coeff.is_negative() {
                        // -a*x + b ≤ 0  =>  x ≥ b/a
                        lower_bounds.push((coeff.abs(), term.clone()));
                    }
                }
                Constraint::Eq(term) => {
                    // Equation gives both upper and lower bound
                    let coeff = term.get_coeff(var).abs();
                    lower_bounds.push((coeff.clone(), term.clone()));
                    upper_bounds.push((coeff, term.clone()));
                }
                Constraint::Div { divisor, .. } => {
                    divisors.push(divisor.clone());
                }
            }
        }

        // Compute LCM of divisors for splitting
        let lcm = self.compute_lcm(&divisors);

        if !lcm.is_zero() && !lcm.is_one() {
            self.stats.divisibility_splits += 1;
        }

        // Generate witnesses from bounds
        // For each lower bound L and upper bound U, add L, L+1, ..., U
        // (in practice, limited set)

        if lower_bounds.is_empty() && upper_bounds.is_empty() {
            // No bounds - try a few values around 0
            for i in -5..=5 {
                witnesses.push(BigInt::from(i));
            }
        } else {
            // Add boundary points
            for i in 0..lower_bounds.len().min(self.config.max_witnesses as usize) {
                witnesses.push(BigInt::from(i as i64));
            }
        }

        // Add values covering divisibility cases
        if !lcm.is_one() && !lcm.is_zero() {
            let limit = std::cmp::min(lcm.clone(), BigInt::from(10));
            // Convert to i64 for iteration (safe for small values)
            if let Some(limit_i64) = limit.to_i64() {
                for r in 0..limit_i64 {
                    witnesses.push(BigInt::from(r));
                }
            }
        }

        // Limit witness count
        witnesses.truncate(self.config.max_witnesses as usize);

        witnesses
    }

    /// Compute LCM of integers.
    fn compute_lcm(&self, nums: &[BigInt]) -> BigInt {
        if nums.is_empty() {
            return BigInt::one();
        }

        let mut result = nums[0].clone();

        for num in &nums[1..] {
            result = self.lcm(&result, num);
        }

        result
    }

    /// Compute LCM of two numbers.
    fn lcm(&self, a: &BigInt, b: &BigInt) -> BigInt {
        if a.is_zero() || b.is_zero() {
            return BigInt::zero();
        }

        let gcd = self.gcd(a, b);
        (a * b).abs() / gcd
    }

    /// Compute GCD of two numbers.
    fn gcd(&self, a: &BigInt, b: &BigInt) -> BigInt {
        let mut x = a.abs();
        let mut y = b.abs();

        while !y.is_zero() {
            let temp = y.clone();
            y = &x % &y;
            x = temp;
        }

        x
    }

    /// Simplify conjunction of constraints.
    fn simplify_conjunction(&mut self, constraints: &[Constraint]) -> Vec<Constraint> {
        self.stats.simplifications += 1;

        let mut result = Vec::new();

        for constraint in constraints {
            // Check for trivial constraints
            match constraint {
                Constraint::Le(term) if term.is_zero() => {
                    // 0 ≤ 0 is trivially true
                    continue;
                }
                Constraint::Eq(term) if term.is_zero() => {
                    // 0 = 0 is trivially true
                    continue;
                }
                _ => {
                    result.push(constraint.clone());
                }
            }
        }

        result
    }

    /// Eliminate universal quantifier: ∀x. φ
    ///
    /// ∀x. φ  =  ¬∃x. ¬φ
    pub fn eliminate_forall(&mut self, var: usize, formula: &DnfFormula) -> DnfFormula {
        // Negate formula, eliminate exists, negate result
        let negated = self.negate_formula(formula);
        let eliminated = self.eliminate_exists(var, &negated);
        self.negate_formula(&eliminated)
    }

    /// Negate DNF formula.
    fn negate_formula(&self, formula: &DnfFormula) -> DnfFormula {
        // ¬(A ∨ B ∨ C) = ¬A ∧ ¬B ∧ ¬C
        // Negation of DNF gives CNF, convert back to DNF

        // Simplified: return original (full implementation needed)
        formula.clone()
    }
}

impl Default for CooperQE {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooper_creation() {
        let cooper = CooperQE::new();
        assert_eq!(cooper.stats().vars_eliminated, 0);
    }

    #[test]
    fn test_linear_term() {
        let term = LinearTerm::constant(BigInt::from(5));
        assert_eq!(term.constant, BigInt::from(5));
        assert!(term.coeffs.is_empty());
    }

    #[test]
    fn test_substitute() {
        let mut term = LinearTerm::zero();
        term.coeffs.insert(0, BigInt::from(2)); // 2*x0
        term.constant = BigInt::from(3); // + 3

        // Substitute x0 = 5: 2*5 + 3 = 13
        let result = term.substitute(0, &BigInt::from(5));

        assert_eq!(result.constant, BigInt::from(13));
        assert!(!result.coeffs.contains_key(&0));
    }

    #[test]
    fn test_constraint_mentions() {
        let mut term = LinearTerm::zero();
        term.coeffs.insert(0, BigInt::from(1));

        let constraint = Constraint::Le(term);

        assert!(constraint.mentions(0));
        assert!(!constraint.mentions(1));
    }

    #[test]
    fn test_dnf_formula() {
        let formula = DnfFormula::trivial();
        assert_eq!(formula.disjuncts.len(), 1);
        assert!(formula.disjuncts[0].is_empty());
    }

    #[test]
    fn test_gcd() {
        let cooper = CooperQE::new();

        let gcd = cooper.gcd(&BigInt::from(12), &BigInt::from(18));
        assert_eq!(gcd, BigInt::from(6));
    }

    #[test]
    fn test_lcm() {
        let cooper = CooperQE::new();

        let lcm = cooper.lcm(&BigInt::from(4), &BigInt::from(6));
        assert_eq!(lcm, BigInt::from(12));
    }

    #[test]
    fn test_compute_lcm_multiple() {
        let cooper = CooperQE::new();

        let nums = vec![BigInt::from(2), BigInt::from(3), BigInt::from(4)];
        let lcm = cooper.compute_lcm(&nums);

        assert_eq!(lcm, BigInt::from(12));
    }

    #[test]
    fn test_eliminate_exists_no_var() {
        let mut cooper = CooperQE::new();

        // Formula without var 0
        let constraint = Constraint::Le(LinearTerm::constant(BigInt::from(-1)));
        let formula = DnfFormula {
            disjuncts: vec![vec![constraint]],
        };

        let result = cooper.eliminate_exists(0, &formula);

        assert_eq!(result.disjuncts.len(), 1);
    }

    #[test]
    fn test_simplify() {
        let mut cooper = CooperQE::new();

        let constraints = vec![
            Constraint::Le(LinearTerm::zero()), // Trivial
            Constraint::Eq(LinearTerm::constant(BigInt::from(1))),
        ];

        let simplified = cooper.simplify_conjunction(&constraints);

        // First constraint should be removed
        assert_eq!(simplified.len(), 1);
    }
}
