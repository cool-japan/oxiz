//! Virtual Term Substitution for LIA Quantifier Elimination.
//!
//! Implements Weispfenning's virtual term substitution method
//! for eliminating quantifiers over linear integer arithmetic.
#![allow(clippy::type_complexity)]

use crate::ast::{TermId, TermKind, TermManager};
use num_bigint::BigInt;
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
        let (lower_bounds, upper_bounds, divisibility) =
            self.partition_constraints(&literals, &var, tm)?;

        // Generate virtual terms (finite test set)
        let virtual_terms =
            self.generate_virtual_terms(&lower_bounds, &upper_bounds, &divisibility, &var, tm)?;

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
            Ok(tm.mk_or(disjuncts))
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
                    if self.mentions_var(tid, var, tm) {
                        literals.push(tid);
                    }
                }
            }
        }

        Ok(literals)
    }

    /// Check if a term mentions a variable.
    fn mentions_var(&self, term_id: TermId, var: &str, tm: &TermManager) -> bool {
        let mut visited = FxHashSet::default();
        let mut queue = vec![term_id];

        while let Some(tid) = queue.pop() {
            if !visited.insert(tid) {
                continue;
            }

            let term = match tm.get(tid) {
                Some(t) => t,
                None => continue,
            };

            match &term.kind {
                TermKind::Var(name) => {
                    let name_str = tm.resolve_str(*name);
                    if name_str == var {
                        return true;
                    }
                }
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args) => {
                    queue.extend(args);
                }
                TermKind::Not(arg) | TermKind::Neg(arg) => {
                    queue.push(*arg);
                }
                TermKind::Sub(lhs, rhs)
                | TermKind::Lt(lhs, rhs)
                | TermKind::Le(lhs, rhs)
                | TermKind::Gt(lhs, rhs)
                | TermKind::Ge(lhs, rhs)
                | TermKind::Eq(lhs, rhs)
                | TermKind::Div(lhs, rhs)
                | TermKind::Mod(lhs, rhs) => {
                    queue.push(*lhs);
                    queue.push(*rhs);
                }
                TermKind::Ite(cond, then_br, else_br) => {
                    queue.push(*cond);
                    queue.push(*then_br);
                    queue.push(*else_br);
                }
                _ => {}
            }
        }

        false
    }

    /// Partition constraints into lower bounds, upper bounds, and divisibility.
    fn partition_constraints(
        &self,
        _literals: &[TermId],
        _var: &str,
        _tm: &TermManager,
    ) -> Result<(Vec<TermId>, Vec<TermId>, Vec<TermId>), String> {
        // Placeholder
        Ok((vec![], vec![], vec![]))
    }

    /// Generate virtual terms for substitution.
    fn generate_virtual_terms(
        &mut self,
        lower_bounds: &[TermId],
        _upper_bounds: &[TermId],
        _divisibility: &[TermId],
        _var: &str,
        tm: &mut TermManager,
    ) -> Result<Vec<TermId>, String> {
        let mut virtual_terms = Vec::new();

        // For each lower bound b, generate: b, b+1, b+2, ..., b+D-1
        // where D is the LCM of all coefficients
        let delta = BigInt::from(10); // Simplified

        for &lower in lower_bounds {
            for i in 0..10 {
                let offset = tm.mk_int(BigInt::from(i));
                let vterm = tm.mk_add(vec![lower, offset]);
                virtual_terms.push(vterm);
            }
        }

        // Also test at -∞ (represented by large negative)
        let neg_inf = tm.mk_int(-&delta * BigInt::from(1000));
        virtual_terms.push(neg_inf);

        Ok(virtual_terms)
    }

    /// Substitute a term for a variable.
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

    #[test]
    fn test_stats() {
        let eliminator = VirtualTermEliminator::new();
        let stats = eliminator.stats();
        assert_eq!(stats.virtual_terms_generated, 0);
        assert_eq!(stats.test_points_evaluated, 0);
    }
}
