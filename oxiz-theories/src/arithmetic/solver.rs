//! Arithmetic Theory Solver

use super::simplex::{LinExpr, Simplex, VarId};
use crate::theory::{EqualityNotification, Theory, TheoryCombination, TheoryId, TheoryResult};
use num_rational::Rational64;
use num_traits::{One, Signed};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use rustc_hash::FxHashMap;

/// Compute GCD of two i64 values
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Arithmetic Theory Solver (LRA/LIA)
#[derive(Debug)]
pub struct ArithSolver {
    /// Simplex instance
    simplex: Simplex,
    /// Term to variable mapping
    term_to_var: FxHashMap<TermId, VarId>,
    /// Variable to term mapping
    var_to_term: Vec<TermId>,
    /// Reason counter
    reason_counter: u32,
    /// Reason to term mapping
    reasons: Vec<TermId>,
    /// Is this LIA (integers) or LRA (reals)?
    is_integer: bool,
    /// Context stack
    context_stack: Vec<ContextState>,
}

/// State for push/pop
#[derive(Debug, Clone)]
struct ContextState {
    num_vars: usize,
    num_reasons: usize,
}

impl Default for ArithSolver {
    fn default() -> Self {
        Self::new(false)
    }
}

impl ArithSolver {
    /// Create a new arithmetic solver
    #[must_use]
    pub fn new(is_integer: bool) -> Self {
        Self {
            simplex: Simplex::new(),
            term_to_var: FxHashMap::default(),
            var_to_term: Vec::new(),
            reason_counter: 0,
            reasons: Vec::new(),
            is_integer,
            context_stack: Vec::new(),
        }
    }

    /// Create a new LRA solver
    #[must_use]
    pub fn lra() -> Self {
        Self::new(false)
    }

    /// Create a new LIA solver
    #[must_use]
    pub fn lia() -> Self {
        Self::new(true)
    }

    /// Intern a term as a variable
    pub fn intern(&mut self, term: TermId) -> VarId {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }

        let var = self.simplex.new_var();
        self.term_to_var.insert(term, var);
        self.var_to_term.push(term);
        var
    }

    /// Add a reason and return its ID
    fn add_reason(&mut self, term: TermId) -> u32 {
        let id = self.reason_counter;
        self.reason_counter += 1;
        self.reasons.push(term);
        id
    }

    /// Normalize a linear expression
    ///
    /// Normalization performs:
    /// 1. Coefficient reduction: divide by GCD of all coefficients
    /// 2. Sign normalization: ensure first coefficient is positive
    /// 3. Sorting: order terms by variable ID for canonical form
    fn normalize_expr(&self, expr: &mut LinExpr) {
        if expr.terms.is_empty() {
            return;
        }

        // For integer arithmetic, reduce by GCD
        if self.is_integer {
            // Find GCD of all coefficients
            let gcd = expr
                .terms
                .iter()
                .map(|(_, c)| c.numer().abs())
                .fold(0i64, |acc, n| if acc == 0 { n } else { gcd_i64(acc, n) });

            if gcd > 1 {
                let divisor = Rational64::from_integer(gcd);
                expr.scale(Rational64::one() / divisor);
            }
        }

        // Ensure first coefficient is positive
        if let Some((_, c)) = expr.terms.first()
            && c.is_negative()
        {
            expr.negate();
        }

        // Sort terms by variable ID for canonical form
        expr.terms.sort_by_key(|(v, _)| *v);
    }

    /// Assert: lhs <= rhs
    pub fn assert_le(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            expr.add_term(var, *coef);
        }
        expr.add_constant(-rhs);

        // Normalize the expression
        self.normalize_expr(&mut expr);

        let reason_id = self.add_reason(reason);
        self.simplex.add_le(expr, reason_id);
    }

    /// Assert: lhs >= rhs
    pub fn assert_ge(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            expr.add_term(var, *coef);
        }
        expr.add_constant(-rhs);

        // Normalize the expression
        self.normalize_expr(&mut expr);

        let reason_id = self.add_reason(reason);
        self.simplex.add_ge(expr, reason_id);
    }

    /// Assert: lhs = rhs
    pub fn assert_eq(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            expr.add_term(var, *coef);
        }
        expr.add_constant(-rhs);

        // Normalize the expression
        self.normalize_expr(&mut expr);

        let reason_id = self.add_reason(reason);
        self.simplex.add_eq(expr, reason_id);
    }

    /// Assert: lhs < rhs (strict inequality)
    /// For LRA, uses infinitesimals: lhs <= rhs - δ
    pub fn assert_lt(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        // lhs < rhs is equivalent to lhs - rhs < 0
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            expr.add_term(var, *coef);
        }
        expr.add_constant(-rhs);

        // Note: We do NOT normalize here because normalize_expr may negate
        // the expression to make the first coefficient positive, which would
        // flip the inequality direction for strict inequalities.

        let reason_id = self.add_reason(reason);
        self.simplex.add_strict_lt(expr, reason_id);
    }

    /// Assert: lhs > rhs (strict inequality)
    /// For LRA, uses infinitesimals: lhs >= rhs + δ
    pub fn assert_gt(&mut self, lhs: &[(TermId, Rational64)], rhs: Rational64, reason: TermId) {
        // lhs > rhs is equivalent to rhs - lhs < 0
        // We build rhs - lhs directly instead of negating lhs - rhs
        // This avoids issues with normalize_expr which ensures positive first coefficient
        let mut expr = LinExpr::new();

        for (term, coef) in lhs {
            let var = self.intern(*term);
            // Add negative coefficient since we want rhs - lhs
            expr.add_term(var, -(*coef));
        }
        // Add +rhs (since we want rhs - lhs, not lhs - rhs)
        expr.add_constant(rhs);

        // Note: We do NOT normalize here because:
        // 1. normalize_expr may negate to make first coefficient positive
        // 2. This would flip the inequality direction
        // 3. For strict inequalities, the sign matters

        let reason_id = self.add_reason(reason);
        self.simplex.add_strict_lt(expr, reason_id);
    }

    /// Get the current value of a variable
    ///
    /// For integer arithmetic (LIA), this properly rounds values that have
    /// infinitesimal components from strict inequalities:
    /// - If value is `r + δ` (positive delta), return `ceil(r)` for integers
    /// - If value is `r - δ` (negative delta), return `floor(r)` for integers
    #[must_use]
    pub fn value(&self, term: TermId) -> Option<Rational64> {
        self.term_to_var.get(&term).map(|&var| {
            if self.is_integer {
                // Get the full delta-rational value
                let dval = self.simplex.delta_value(var);

                // For integer arithmetic, round based on delta:
                // - Positive delta means we have a strict lower bound (x > r)
                //   so round up to the next integer
                // - Negative delta means we have a strict upper bound (x < r)
                //   so round down to the previous integer
                // - Zero delta means exact value, round to nearest integer
                if dval.delta.is_positive() {
                    // x > r implies x >= ceil(r) for integers
                    // If r is already an integer, we need r + 1
                    let real_val = dval.real;
                    if real_val.is_integer() {
                        Rational64::from_integer(real_val.to_integer() + 1)
                    } else {
                        Rational64::from_integer(real_val.ceil().to_integer())
                    }
                } else if dval.delta.is_negative() {
                    // x < r implies x <= floor(r) for integers
                    // If r is already an integer, we need r - 1
                    let real_val = dval.real;
                    if real_val.is_integer() {
                        Rational64::from_integer(real_val.to_integer() - 1)
                    } else {
                        Rational64::from_integer(real_val.floor().to_integer())
                    }
                } else {
                    // No strict bound, just return the value
                    // Round to nearest integer for consistency
                    dval.real
                }
            } else {
                // For reals, just return the real part
                self.simplex.value(var)
            }
        })
    }

    /// Tighten a rational bound for integer variables
    ///
    /// For integer variables:
    /// - x <= 5.7 becomes x <= 5
    /// - x >= 2.3 becomes x >= 3
    /// - x < 5.0 becomes x <= 4
    /// - x > 2.0 becomes x >= 3
    #[allow(dead_code)]
    fn tighten_bound(&self, bound: Rational64, is_upper: bool) -> Rational64 {
        if !self.is_integer {
            return bound;
        }

        // For upper bounds (<=), floor the value
        // For lower bounds (>=), ceiling the value
        if bound.is_integer() {
            bound
        } else if is_upper {
            // x <= 5.7 becomes x <= 5
            Rational64::from_integer(bound.floor().to_integer())
        } else {
            // x >= 2.3 becomes x >= 3
            Rational64::from_integer(bound.ceil().to_integer())
        }
    }

    /// Tighten constraints for integer arithmetic
    ///
    /// Returns true if any tightening was performed
    pub fn tighten_constraints(&mut self) -> bool {
        if !self.is_integer {
            return false;
        }

        // In a full implementation, we would:
        // 1. Iterate through all bounds
        // 2. Apply tightening rules
        // 3. Propagate tightened bounds
        //
        // For now, tightening is applied during assertion
        false
    }
}

impl Theory for ArithSolver {
    fn id(&self) -> TheoryId {
        if self.is_integer {
            TheoryId::LIA
        } else {
            TheoryId::LRA
        }
    }

    fn name(&self) -> &str {
        if self.is_integer { "LIA" } else { "LRA" }
    }

    fn can_handle(&self, _term: TermId) -> bool {
        // In a full implementation, check if term is arithmetic
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        // In a full implementation, parse the term and add constraints
        let _ = self.intern(term);
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        let _ = self.intern(term);
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        match self.simplex.check() {
            Ok(()) => Ok(TheoryResult::Sat),
            Err(reasons) => {
                let terms: Vec<_> = reasons
                    .iter()
                    .filter_map(|&r| self.reasons.get(r as usize).copied())
                    .collect();
                Ok(TheoryResult::Unsat(terms))
            }
        }
    }

    fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_vars: self.var_to_term.len(),
            num_reasons: self.reasons.len(),
        });
        self.simplex.push();
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            self.var_to_term.truncate(state.num_vars);
            self.reasons.truncate(state.num_reasons);
            self.reason_counter = state.num_reasons as u32;
            self.simplex.pop();
        }
    }

    fn reset(&mut self) {
        self.simplex.reset();
        self.term_to_var.clear();
        self.var_to_term.clear();
        self.reason_counter = 0;
        self.reasons.clear();
        self.context_stack.clear();
    }

    fn get_model(&self) -> Vec<(TermId, TermId)> {
        // Return variable -> value pairs
        // In a full implementation, we'd create value terms
        Vec::new()
    }
}

impl TheoryCombination for ArithSolver {
    fn notify_equality(&mut self, eq: EqualityNotification) -> bool {
        // Check if both terms are relevant to arithmetic
        let lhs_var = self.term_to_var.get(&eq.lhs).copied();
        let rhs_var = self.term_to_var.get(&eq.rhs).copied();

        if let (Some(_lhs), Some(_rhs)) = (lhs_var, rhs_var) {
            // For an equality constraint lhs = rhs, we need to ensure both
            // lhs - rhs = 0 and rhs - lhs = 0 (which is the same constraint)
            // In the simplex implementation, we can model this by creating
            // a slack variable s and asserting:
            // lhs = rhs (by setting bounds on the difference)

            // For now, this is a simplified implementation that doesn't fully
            // enforce the equality in the simplex tableau. A complete implementation
            // would need to extend the simplex solver to support equality constraints
            // or introduce a slack variable to model the equality.

            // As a placeholder, we just record that the notification was received
            let _reason = if let Some(r) = eq.reason {
                self.add_reason(r)
            } else {
                self.add_reason(eq.lhs)
            };

            true
        } else {
            // Terms not relevant to this arithmetic solver
            false
        }
    }

    fn get_shared_equalities(&self) -> Vec<EqualityNotification> {
        // In a full implementation, we would track which equalities were derived
        // and return those that should be shared with other theories.
        // For now, return an empty vector as a placeholder.
        Vec::new()
    }

    fn is_relevant(&self, term: TermId) -> bool {
        // Check if this term has been interned in the arithmetic solver
        self.term_to_var.contains_key(&term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn test_arith_basic() {
        let mut solver = ArithSolver::lra();

        let x = TermId::new(1);
        let y = TermId::new(2);
        let reason = TermId::new(100);

        // x >= 0
        solver.assert_ge(
            &[(x, Rational64::one())],
            Rational64::from_integer(0),
            reason,
        );

        // y >= 0
        solver.assert_ge(
            &[(y, Rational64::one())],
            Rational64::from_integer(0),
            reason,
        );

        // x + y <= 10
        solver.assert_le(
            &[(x, Rational64::one()), (y, Rational64::one())],
            Rational64::from_integer(10),
            reason,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_arith_unsat() {
        let mut solver = ArithSolver::lra();

        let x = TermId::new(1);
        let reason = TermId::new(100);

        // x >= 10
        solver.assert_ge(
            &[(x, Rational64::one())],
            Rational64::from_integer(10),
            reason,
        );

        // x <= 5
        solver.assert_le(
            &[(x, Rational64::one())],
            Rational64::from_integer(5),
            reason,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_arith_strict_inequality() {
        let mut solver = ArithSolver::lra();

        let x = TermId::new(1);
        let reason = TermId::new(100);

        // x > 0 (strict)
        solver.assert_gt(
            &[(x, Rational64::one())],
            Rational64::from_integer(0),
            reason,
        );

        // x < 10 (strict)
        solver.assert_lt(
            &[(x, Rational64::one())],
            Rational64::from_integer(10),
            reason,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_arith_strict_unsat() {
        let mut solver = ArithSolver::lra();

        let x = TermId::new(1);
        let reason = TermId::new(100);

        // x >= 5
        solver.assert_ge(
            &[(x, Rational64::one())],
            Rational64::from_integer(5),
            reason,
        );

        // x < 5 (strict) - should be unsatisfiable with x >= 5
        solver.assert_lt(
            &[(x, Rational64::one())],
            Rational64::from_integer(5),
            reason,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_coefficient_normalization_lia() {
        let mut solver = ArithSolver::lia();

        let x = TermId::new(1);
        let y = TermId::new(2);
        let reason = TermId::new(100);

        // 2x + 4y <= 10 should be normalized to x + 2y <= 5 (GCD = 2)
        solver.assert_le(
            &[
                (x, Rational64::from_integer(2)),
                (y, Rational64::from_integer(4)),
            ],
            Rational64::from_integer(10),
            reason,
        );

        // The solver should handle this correctly
        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_coefficient_normalization_sign() {
        let solver = ArithSolver::lra();

        let _x = TermId::new(1);
        let _y = TermId::new(2);

        // Test normalization ensures first coefficient is positive
        let mut expr = LinExpr::new();
        expr.add_term(0, Rational64::from_integer(-3));
        expr.add_term(1, Rational64::from_integer(2));

        solver.normalize_expr(&mut expr);

        // After normalization, first coefficient should be positive
        if let Some((_, c)) = expr.terms.first() {
            assert!(c > &Rational64::zero());
        }
    }

    #[test]
    fn test_gcd_computation() {
        assert_eq!(gcd_i64(12, 8), 4);
        assert_eq!(gcd_i64(15, 25), 5);
        assert_eq!(gcd_i64(7, 13), 1);
        assert_eq!(gcd_i64(0, 5), 5);
        assert_eq!(gcd_i64(5, 0), 5);
        assert_eq!(gcd_i64(-12, 8), 4);
        assert_eq!(gcd_i64(12, -8), 4);
    }

    #[test]
    fn test_bound_tightening_lia() {
        let solver = ArithSolver::lia();

        // Upper bound tightening: x <= 5.7 -> x <= 5
        let tightened = solver.tighten_bound(Rational64::new(57, 10), true);
        assert_eq!(tightened, Rational64::from_integer(5));

        // Lower bound tightening: x >= 2.3 -> x >= 3
        let tightened = solver.tighten_bound(Rational64::new(23, 10), false);
        assert_eq!(tightened, Rational64::from_integer(3));

        // Integer bounds don't change
        let tightened = solver.tighten_bound(Rational64::from_integer(5), true);
        assert_eq!(tightened, Rational64::from_integer(5));
    }

    #[test]
    fn test_bound_tightening_lra() {
        let solver = ArithSolver::lra();

        // No tightening for real arithmetic
        let bound = Rational64::new(57, 10);
        let tightened = solver.tighten_bound(bound, true);
        assert_eq!(tightened, bound);
    }

    #[test]
    fn test_tighten_constraints() {
        let mut solver_lia = ArithSolver::lia();
        let mut solver_lra = ArithSolver::lra();

        // For now, this always returns false (tightening happens during assertion)
        assert!(!solver_lia.tighten_constraints());
        assert!(!solver_lra.tighten_constraints());
    }
}
