//! Cooper's Algorithm for Presburger Arithmetic QE.
//!
//! Implements Cooper's method for quantifier elimination in linear
//! integer arithmetic (Presburger arithmetic).

use crate::ast::{TermId, TermManager};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use rustc_hash::FxHashSet;

/// Cooper's algorithm QE engine.
pub struct CooperEliminator {
    /// Statistics
    stats: CooperStats,
}

/// Cooper elimination statistics.
#[derive(Debug, Clone, Default)]
pub struct CooperStats {
    /// Number of quantifiers eliminated
    pub quantifiers_eliminated: usize,
    /// Number of test cases generated
    pub test_cases: usize,
    /// Number of infinity tests
    pub infinity_tests: usize,
}

impl CooperEliminator {
    /// Create a new Cooper eliminator.
    pub fn new() -> Self {
        Self {
            stats: CooperStats::default(),
        }
    }

    /// Eliminate existential quantifier: exists x. φ(x).
    pub fn eliminate_exists(
        &mut self,
        var: String,
        formula: TermId,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        self.stats.quantifiers_eliminated += 1;

        // Step 1: Put formula in DNF (disjunctive normal form)
        let dnf = self.to_dnf(formula, tm)?;

        // Step 2: For each conjunct, eliminate quantifier
        let mut disjuncts = Vec::new();

        for conjunct in self.extract_conjuncts(dnf, tm)? {
            let eliminated = self.eliminate_from_conjunct(&var, conjunct, tm)?;
            disjuncts.push(eliminated);
        }

        // Step 3: Return disjunction of results
        if disjuncts.is_empty() {
            Ok(tm.mk_false())
        } else if disjuncts.len() == 1 {
            Ok(disjuncts[0])
        } else {
            Ok(tm.mk_or(disjuncts))
        }
    }

    /// Eliminate quantifier from a single conjunct.
    fn eliminate_from_conjunct(
        &mut self,
        var: &str,
        conjunct: TermId,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        // Extract atomic formulas
        let atoms = self.extract_atoms(conjunct, var, tm)?;

        // Compute LCM of all coefficients of x
        let lcm = self.compute_coefficient_lcm(&atoms, var, tm)?;

        // Generate test set based on Cooper's construction
        let test_set = self.generate_cooper_test_set(&atoms, var, lcm.clone(), tm)?;

        // Build disjunction over test set
        let mut test_disjuncts = Vec::new();

        for test_value in test_set {
            self.stats.test_cases += 1;

            // Substitute test_value for var
            let substituted = self.substitute(conjunct, var, test_value, tm)?;
            test_disjuncts.push(substituted);
        }

        // Add infinity tests: φ[x/+∞] and φ[x/-∞]
        self.stats.infinity_tests += 2;
        let plus_inf_test = self.test_at_infinity(conjunct, var, true, tm)?;
        let minus_inf_test = self.test_at_infinity(conjunct, var, false, tm)?;

        test_disjuncts.push(plus_inf_test);
        test_disjuncts.push(minus_inf_test);

        // Return disjunction
        if test_disjuncts.is_empty() {
            Ok(tm.mk_false())
        } else {
            Ok(tm.mk_or(test_disjuncts))
        }
    }

    /// Generate Cooper test set.
    ///
    /// For each lower bound b_i with coefficient a_i:
    /// Test x = b_i + j for j = 0, 1, ..., lcm(a_1,...,a_n) - 1
    fn generate_cooper_test_set(
        &mut self,
        atoms: &[TermId],
        var: &str,
        lcm: BigInt,
        tm: &mut TermManager,
    ) -> Result<Vec<TermId>, String> {
        let mut test_set = Vec::new();

        // Extract lower bounds
        let lower_bounds = self.extract_lower_bounds(atoms, var, tm)?;

        for lower_bound in lower_bounds {
            // Generate j = 0, ..., lcm - 1
            for j in 0..lcm.to_i64().unwrap_or(10) {
                // test_value = lower_bound + j
                let j_term = tm.mk_int(BigInt::from(j));
                let test_value = tm.mk_add(vec![lower_bound, j_term]);
                test_set.push(test_value);
            }
        }

        Ok(test_set)
    }

    /// Extract lower bounds on variable from atoms.
    fn extract_lower_bounds(
        &self,
        _atoms: &[TermId],
        _var: &str,
        _tm: &TermManager,
    ) -> Result<Vec<TermId>, String> {
        // Placeholder
        Ok(vec![])
    }

    /// Compute LCM of coefficients of variable.
    fn compute_coefficient_lcm(
        &self,
        atoms: &[TermId],
        _var: &str,
        _tm: &TermManager,
    ) -> Result<BigInt, String> {
        if atoms.is_empty() {
            Ok(BigInt::one())
        } else {
            // Simplified: return small default
            Ok(BigInt::from(2))
        }
    }

    /// Test formula at infinity (eliminate x by replacing with +∞ or -∞).
    fn test_at_infinity(
        &self,
        formula: TermId,
        _var: &str,
        _plus_infinity: bool,
        _tm: &mut TermManager,
    ) -> Result<TermId, String> {
        // Placeholder: would implement infinity substitution rules
        Ok(formula)
    }

    /// Convert to DNF.
    fn to_dnf(&self, formula: TermId, _tm: &mut TermManager) -> Result<TermId, String> {
        // Placeholder: would convert to DNF
        Ok(formula)
    }

    /// Extract conjuncts from DNF.
    fn extract_conjuncts(&self, dnf: TermId, tm: &TermManager) -> Result<Vec<TermId>, String> {
        let term = tm.get(dnf).ok_or("term not found")?;

        match &term.kind {
            crate::ast::TermKind::Or(args) => Ok(args.to_vec()),
            _ => Ok(vec![dnf]),
        }
    }

    /// Extract atomic formulas mentioning variable.
    fn extract_atoms(
        &self,
        conjunct: TermId,
        var: &str,
        tm: &TermManager,
    ) -> Result<Vec<TermId>, String> {
        let mut atoms = Vec::new();
        let mut visited = FxHashSet::default();
        let mut queue = vec![conjunct];

        while let Some(tid) = queue.pop() {
            if !visited.insert(tid) {
                continue;
            }

            let term = tm.get(tid).ok_or("term not found")?;

            match &term.kind {
                crate::ast::TermKind::And(args) => {
                    queue.extend(args);
                }
                _ => {
                    if self.mentions_var(tid, var, tm) {
                        atoms.push(tid);
                    }
                }
            }
        }

        Ok(atoms)
    }

    /// Check if term mentions variable.
    fn mentions_var(&self, _term_id: TermId, _var: &str, _tm: &TermManager) -> bool {
        // Placeholder
        false
    }

    /// Substitute value for variable.
    fn substitute(
        &self,
        formula: TermId,
        _var: &str,
        _value: TermId,
        _tm: &mut TermManager,
    ) -> Result<TermId, String> {
        // Placeholder
        Ok(formula)
    }

    /// Get statistics.
    pub fn stats(&self) -> &CooperStats {
        &self.stats
    }
}

impl Default for CooperEliminator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooper_eliminator() {
        let eliminator = CooperEliminator::new();
        assert_eq!(eliminator.stats.quantifiers_eliminated, 0);
    }
}
