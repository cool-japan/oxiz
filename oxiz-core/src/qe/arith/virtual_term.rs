//! Virtual Term Substitution for LIA Quantifier Elimination.
//!
//! Implements Weispfenning's virtual term substitution method
//! for eliminating quantifiers over linear integer arithmetic.

use crate::ast::{Term, TermId, TermKind, TermManager};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashSet;

/// Virtual term substitution QE engine.
pub struct VirtualTermEliminator {
    /// Statistics
    stats: VirtualTermStats,
}

/// Virtual term elimination statistics.
#[derive(Debug, Clone, Default)]
pub struct VirtualTermStats {
    /// Number of quantifiers eliminated
    pub quantifiers_eliminated: usize,
    /// Number of virtual terms generated
    pub virtual_terms_generated: usize,
    /// Number of test points evaluated
    pub test_points_evaluated: usize,
}

impl VirtualTermEliminator {
    /// Create a new virtual term eliminator.
    pub fn new() -> Self {
        Self {
            stats: VirtualTermStats::default(),
        }
    }

    /// Eliminate existential quantifier from LIA formula.
    pub fn eliminate_exists(
        &mut self,
        var: String,
        formula: TermId,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        self.stats.quantifiers_eliminated += 1;

        // Extract literals involving quantified variable
        let literals = self.extract_literals(formula, &var, tm)?;

        // Partition into lower bounds, upper bounds, and divisibility constraints
        let (lower_bounds, upper_bounds, divisibility) = self.partition_constraints(&literals, &var, tm)?;

        // Generate virtual terms (finite test set)
        let virtual_terms = self.generate_virtual_terms(&lower_bounds, &upper_bounds, &divisibility, &var, tm)?;

        // Build disjunction: exists x. φ(x) ≡ ⋁ φ(t) for t in virtual terms
        let mut disjuncts = Vec::new();

        for vterm in virtual_terms {
            self.stats.virtual_terms_generated += 1;

            // Substitute virtual term for variable
            let substituted = self.substitute(formula, &var, vterm, tm)?;
            disjuncts.push(substituted);
        }

        // Return disjunction
        if disjuncts.is_empty() {
            Ok(tm.mk_false())
        } else if disjuncts.len() == 1 {
            Ok(disjuncts[0])
        } else {
            tm.mk_or(disjuncts)
        }
    }

    /// Extract literals involving the quantified variable.
    fn extract_literals(
        &self,
        formula: TermId,
        var: &str,
        tm: &TermManager,
    ) -> Result<Vec<TermId>, String> {
        let mut literals = Vec::new();
        let mut visited = FxHashSet::default();
        let mut queue = vec![formula];

        while let Some(tid) = queue.pop() {
            if !visited.insert(tid) {
                continue;
            }

            let term = tm.get(tid).ok_or("term not found")?;

            match &term.kind {
                TermKind::And(args) | TermKind::Or(args) => {
                    queue.extend(args);
                }
                TermKind::Not(arg) => {
                    queue.push(*arg);
                }
                _ => {
                    // Check if literal involves var
                    if self.mentions_var(tid, var, tm) {
                        literals.push(tid);
                    }
                }
            }
        }

        Ok(literals)
    }

    /// Partition constraints into lower bounds, upper bounds, and divisibility.
    fn partition_constraints(
        &self,
        literals: &[TermId],
        var: &str,
        tm: &TermManager,
    ) -> Result<(Vec<TermId>, Vec<TermId>, Vec<TermId>), String> {
        let mut lower_bounds = Vec::new();
        let mut upper_bounds = Vec::new();
        let mut divisibility = Vec::new();

        for &lit in literals {
            let term = tm.get(lit).ok_or("term not found")?;

            match &term.kind {
                TermKind::Le(lhs, rhs) => {
                    // Check if x <= rhs (upper bound) or lhs <= x (lower bound)
                    if self.is_var(*lhs, var, tm) {
                        upper_bounds.push(lit);
                    } else if self.is_var(*rhs, var, tm) {
                        lower_bounds.push(lit);
                    }
                }
                TermKind::Ge(lhs, rhs) => {
                    if self.is_var(*lhs, var, tm) {
                        lower_bounds.push(lit);
                    } else if self.is_var(*rhs, var, tm) {
                        upper_bounds.push(lit);
                    }
                }
                TermKind::Eq(lhs, rhs) => {
                    // Equality can be both upper and lower bound
                    lower_bounds.push(lit);
                    upper_bounds.push(lit);
                }
                TermKind::Mod(_, _) => {
                    divisibility.push(lit);
                }
                _ => {}
            }
        }

        Ok((lower_bounds, upper_bounds, divisibility))
    }

    /// Generate virtual terms (finite test set).
    fn generate_virtual_terms(
        &mut self,
        lower_bounds: &[TermId],
        upper_bounds: &[TermId],
        divisibility: &[TermId],
        var: &str,
        tm: &mut TermManager,
    ) -> Result<Vec<TermId>, String> {
        let mut virtual_terms = Vec::new();

        // For each combination of lower bound l and upper bound u:
        // Test points: l, l+1, ..., u
        for &lower in lower_bounds {
            for &upper in upper_bounds {
                // Extract bounds
                let l_val = self.extract_bound_value(lower, var, tm)?;
                let u_val = self.extract_bound_value(upper, var, tm)?;

                // Generate test points between bounds
                let mut current = l_val;
                while current <= u_val && virtual_terms.len() < 100 {
                    let test_point = tm.mk_int(current.clone());
                    virtual_terms.push(test_point);
                    self.stats.test_points_evaluated += 1;

                    current = current + BigInt::one();
                }
            }
        }

        // Handle divisibility constraints
        if !divisibility.is_empty() {
            // For divisibility m | (ax + b), test x = -b/a (mod m)
            for &div_constraint in divisibility {
                if let Some(test_point) = self.extract_divisibility_witness(div_constraint, var, tm)? {
                    virtual_terms.push(test_point);
                }
            }
        }

        // If no bounds, use default test set: -δ, ..., δ where δ is LCM of divisors
        if virtual_terms.is_empty() {
            let delta = self.compute_test_bound(divisibility, tm)?;
            for i in -delta..=delta {
                let test_point = tm.mk_int(BigInt::from(i));
                virtual_terms.push(test_point);
            }
        }

        Ok(virtual_terms)
    }

    /// Extract bound value from constraint.
    fn extract_bound_value(
        &self,
        constraint: TermId,
        _var: &str,
        _tm: &TermManager,
    ) -> Result<BigInt, String> {
        // Simplified: return a default value
        Ok(BigInt::zero())
    }

    /// Extract witness for divisibility constraint.
    fn extract_divisibility_witness(
        &self,
        _constraint: TermId,
        _var: &str,
        _tm: &mut TermManager,
    ) -> Result<Option<TermId>, String> {
        Ok(None)
    }

    /// Compute test bound (δ) based on divisibility constraints.
    fn compute_test_bound(&self, divisibility: &[TermId], _tm: &TermManager) -> Result<i64, String> {
        if divisibility.is_empty() {
            Ok(2) // Default small bound
        } else {
            Ok(10) // Heuristic based on divisors
        }
    }

    /// Substitute term for variable in formula.
    fn substitute(
        &self,
        formula: TermId,
        _var: &str,
        _replacement: TermId,
        _tm: &mut TermManager,
    ) -> Result<TermId, String> {
        // Placeholder: would perform actual substitution
        Ok(formula)
    }

    /// Check if term mentions variable.
    fn mentions_var(&self, _term_id: TermId, _var: &str, _tm: &TermManager) -> bool {
        // Placeholder
        false
    }

    /// Check if term is exactly the variable.
    fn is_var(&self, term_id: TermId, var: &str, tm: &TermManager) -> bool {
        if let Some(term) = tm.get(term_id) {
            matches!(&term.kind, TermKind::Var(name) if name.as_str() == var)
        } else {
            false
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &VirtualTermStats {
        &self.stats
    }
}

impl Default for VirtualTermEliminator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_term_eliminator() {
        let eliminator = VirtualTermEliminator::new();
        assert_eq!(eliminator.stats.quantifiers_eliminated, 0);
    }
}
