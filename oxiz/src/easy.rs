//! Simplified high-level API for common SMT solving use cases.

use core::time::Duration;
use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::error::OxizError;
use oxiz_core::sort::SortId;
use oxiz_solver::resource_limits::{ResourceExhausted, ResourceLimits};
use oxiz_solver::{Solver, SolverResult};

/// Result of an easy solver check.
#[derive(Debug, Clone)]
pub enum EasyResult {
    /// The formula is satisfiable.
    Sat,
    /// The formula is unsatisfiable.
    Unsat,
    /// The solver could not determine satisfiability.
    Unknown,
    /// A resource limit was hit.
    ResourceExhausted(ResourceExhausted),
    /// An error occurred.
    Error(String),
}

impl EasyResult {
    /// Returns `true` if the result is satisfiable.
    #[must_use]
    pub fn is_sat(&self) -> bool {
        matches!(self, EasyResult::Sat)
    }

    /// Returns `true` if the result is unsatisfiable.
    #[must_use]
    pub fn is_unsat(&self) -> bool {
        matches!(self, EasyResult::Unsat)
    }

    /// Returns `true` if the result is unknown.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        matches!(self, EasyResult::Unknown)
    }

    /// Returns `true` if a resource limit was hit.
    #[must_use]
    pub fn is_resource_exhausted(&self) -> bool {
        matches!(self, EasyResult::ResourceExhausted(_))
    }

    /// Returns `true` if an error occurred.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, EasyResult::Error(_))
    }
}

/// A simplified SMT solver with a builder-pattern API.
#[derive(Debug)]
pub struct EasySolver {
    tm: TermManager,
    solver: Solver,
    vars: std::collections::HashMap<String, (TermId, SortId)>,
    limits: Option<ResourceLimits>,
    last_result: Option<SolverResult>,
}

impl Default for EasySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl EasySolver {
    /// Create a new easy solver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tm: TermManager::new(),
            solver: Solver::new(),
            vars: std::collections::HashMap::new(),
            limits: None,
            last_result: None,
        }
    }

    /// One-liner: check satisfiability of an SMT-LIB2 script string.
    pub fn check_sat_str(script: &str) -> Result<String, OxizError> {
        let mut ctx = oxiz_solver::Context::new();
        let output = ctx.execute_script(script)?;
        output
            .last()
            .cloned()
            .ok_or_else(|| OxizError::Internal("no output from script".into()))
    }

    /// Set the logic.
    pub fn set_logic(&mut self, logic: &str) -> &mut Self {
        self.solver.set_logic(logic);
        self
    }

    /// Declare an integer variable.
    pub fn int_var(&mut self, name: &str) -> &mut Self {
        let sort = self.tm.sorts.int_sort;
        let term = self.tm.mk_var(name, sort);
        self.vars.insert(name.to_string(), (term, sort));
        self
    }

    /// Declare a real variable.
    pub fn real_var(&mut self, name: &str) -> &mut Self {
        let sort = self.tm.sorts.real_sort;
        let term = self.tm.mk_var(name, sort);
        self.vars.insert(name.to_string(), (term, sort));
        self
    }

    /// Declare a boolean variable.
    pub fn bool_var(&mut self, name: &str) -> &mut Self {
        let sort = self.tm.sorts.bool_sort;
        let term = self.tm.mk_var(name, sort);
        self.vars.insert(name.to_string(), (term, sort));
        self
    }

    /// Assert that variable `name` > `value`.
    pub fn assert_gt(&mut self, name: &str, value: i64) -> &mut Self {
        if let Some(&(term, _)) = self.vars.get(name) {
            let val = self.tm.mk_int(BigInt::from(value));
            let constraint = self.tm.mk_gt(term, val);
            self.solver.assert(constraint, &mut self.tm);
        }
        self
    }

    /// Assert that variable `name` < `value`.
    pub fn assert_lt(&mut self, name: &str, value: i64) -> &mut Self {
        if let Some(&(term, _)) = self.vars.get(name) {
            let val = self.tm.mk_int(BigInt::from(value));
            let constraint = self.tm.mk_lt(term, val);
            self.solver.assert(constraint, &mut self.tm);
        }
        self
    }

    /// Assert that variable `name` >= `value`.
    pub fn assert_ge(&mut self, name: &str, value: i64) -> &mut Self {
        if let Some(&(term, _)) = self.vars.get(name) {
            let val = self.tm.mk_int(BigInt::from(value));
            let constraint = self.tm.mk_ge(term, val);
            self.solver.assert(constraint, &mut self.tm);
        }
        self
    }

    /// Assert that variable `name` <= `value`.
    pub fn assert_le(&mut self, name: &str, value: i64) -> &mut Self {
        if let Some(&(term, _)) = self.vars.get(name) {
            let val = self.tm.mk_int(BigInt::from(value));
            let constraint = self.tm.mk_le(term, val);
            self.solver.assert(constraint, &mut self.tm);
        }
        self
    }

    /// Assert that variable equals an integer.
    pub fn assert_eq_int(&mut self, name: &str, value: i64) -> &mut Self {
        if let Some(&(term, _)) = self.vars.get(name) {
            let val = self.tm.mk_int(BigInt::from(value));
            let constraint = self.tm.mk_eq(term, val);
            self.solver.assert(constraint, &mut self.tm);
        }
        self
    }

    /// Assert that two variables are equal.
    pub fn assert_eq_vars(&mut self, name1: &str, name2: &str) -> &mut Self {
        if let (Some(&(t1, _)), Some(&(t2, _))) = (self.vars.get(name1), self.vars.get(name2)) {
            let constraint = self.tm.mk_eq(t1, t2);
            self.solver.assert(constraint, &mut self.tm);
        }
        self
    }

    /// Assert that two variables are not equal.
    pub fn assert_neq_vars(&mut self, name1: &str, name2: &str) -> &mut Self {
        if let (Some(&(t1, _)), Some(&(t2, _))) = (self.vars.get(name1), self.vars.get(name2)) {
            let eq = self.tm.mk_eq(t1, t2);
            let neq = self.tm.mk_not(eq);
            self.solver.assert(neq, &mut self.tm);
        }
        self
    }

    /// Assert a boolean variable is true.
    pub fn assert_true(&mut self, name: &str) -> &mut Self {
        if let Some(&(term, _)) = self.vars.get(name) {
            self.solver.assert(term, &mut self.tm);
        }
        self
    }

    /// Assert a boolean variable is false.
    pub fn assert_false(&mut self, name: &str) -> &mut Self {
        if let Some(&(term, _)) = self.vars.get(name) {
            let neg = self.tm.mk_not(term);
            self.solver.assert(neg, &mut self.tm);
        }
        self
    }

    /// Assert `var1 + var2 = value`.
    pub fn assert_sum_eq(&mut self, var1: &str, var2: &str, value: i64) -> &mut Self {
        if let (Some(&(t1, _)), Some(&(t2, _))) = (self.vars.get(var1), self.vars.get(var2)) {
            let sum = self.tm.mk_add([t1, t2]);
            let val = self.tm.mk_int(BigInt::from(value));
            let constraint = self.tm.mk_eq(sum, val);
            self.solver.assert(constraint, &mut self.tm);
        }
        self
    }

    /// Set a wall-clock timeout.
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        let limits = self.limits.get_or_insert_with(ResourceLimits::new);
        limits.timeout = Some(timeout);
        self
    }

    /// Set a conflict limit.
    pub fn conflict_limit(&mut self, max_conflicts: u64) -> &mut Self {
        let limits = self.limits.get_or_insert_with(ResourceLimits::new);
        limits.max_conflicts = Some(max_conflicts);
        self
    }

    /// Set a decision limit.
    pub fn decision_limit(&mut self, max_decisions: u64) -> &mut Self {
        let limits = self.limits.get_or_insert_with(ResourceLimits::new);
        limits.max_decisions = Some(max_decisions);
        self
    }

    /// Check satisfiability.
    pub fn check_sat(&mut self) -> EasyResult {
        let result = if let Some(ref limits) = self.limits {
            match self.solver.check_with_limits(&mut self.tm, limits) {
                Ok(r) => r,
                Err(exhausted) => return EasyResult::ResourceExhausted(exhausted),
            }
        } else {
            self.solver.check(&mut self.tm)
        };
        self.last_result = Some(result);
        match result {
            SolverResult::Sat => EasyResult::Sat,
            SolverResult::Unsat => EasyResult::Unsat,
            SolverResult::Unknown => EasyResult::Unknown,
        }
    }

    /// Convenience: returns `true` if satisfiable.
    pub fn is_sat(&mut self) -> bool {
        self.check_sat().is_sat()
    }

    /// Convenience: returns `true` if unsatisfiable.
    pub fn is_unsat(&mut self) -> bool {
        self.check_sat().is_unsat()
    }

    /// Get the model value for a variable as a string.
    #[must_use]
    pub fn get_model_value(&self, name: &str) -> Option<String> {
        if self.last_result != Some(SolverResult::Sat) {
            return None;
        }
        let &(term, sort) = self.vars.get(name)?;
        let model = self.solver.model()?;
        let val_term = model.get(term)?;
        let val = self.tm.get(val_term)?;
        Some(self.format_term_value(&val.kind, sort))
    }

    /// Get the integer value for a variable.
    #[must_use]
    pub fn get_int_value(&self, name: &str) -> Option<i64> {
        if self.last_result != Some(SolverResult::Sat) {
            return None;
        }
        let &(term, _) = self.vars.get(name)?;
        let model = self.solver.model()?;
        let val_term = model.get(term)?;
        let val = self.tm.get(val_term)?;
        match &val.kind {
            TermKind::IntConst(n) => {
                use num_traits::ToPrimitive;
                n.to_i64()
            }
            _ => None,
        }
    }

    /// Get the boolean value for a variable.
    #[must_use]
    pub fn get_bool_value(&self, name: &str) -> Option<bool> {
        if self.last_result != Some(SolverResult::Sat) {
            return None;
        }
        let &(term, _) = self.vars.get(name)?;
        let model = self.solver.model()?;
        let val_term = model.get(term)?;
        let val = self.tm.get(val_term)?;
        match &val.kind {
            TermKind::True => Some(true),
            TermKind::False => Some(false),
            _ => None,
        }
    }

    /// Push a context level.
    pub fn push(&mut self) -> &mut Self {
        self.solver.push();
        self
    }

    /// Pop a context level.
    pub fn pop(&mut self) -> &mut Self {
        self.solver.pop();
        self
    }

    /// Reset the solver.
    pub fn reset(&mut self) {
        self.solver = Solver::new();
        self.tm = TermManager::new();
        self.vars.clear();
        self.last_result = None;
    }

    fn format_term_value(&self, kind: &TermKind, _sort: SortId) -> String {
        match kind {
            TermKind::True => "true".to_string(),
            TermKind::False => "false".to_string(),
            TermKind::IntConst(n) => n.to_string(),
            TermKind::RealConst(r) => {
                if *r.denom() == 1 {
                    format!("{}.0", r.numer())
                } else {
                    format!("{}/{}", r.numer(), r.denom())
                }
            }
            TermKind::BitVecConst { value, width } => {
                format!(
                    "#b{:0>width$}",
                    format!("{:b}", value),
                    width = *width as usize
                )
            }
            _ => "?".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_sat_str_sat() {
        let result =
            EasySolver::check_sat_str("(declare-const x Int) (assert (> x 5)) (check-sat)");
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed"), "sat");
    }

    #[test]
    fn test_check_sat_str_unsat() {
        let result = EasySolver::check_sat_str(
            "(declare-const x Int) (assert (> x 5)) (assert (< x 3)) (check-sat)",
        );
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed"), "unsat");
    }

    #[test]
    fn test_builder_sat() {
        let mut solver = EasySolver::new();
        solver.int_var("x").assert_gt("x", 0).assert_lt("x", 10);
        assert!(solver.check_sat().is_sat());
    }

    #[test]
    fn test_builder_unsat() {
        let mut solver = EasySolver::new();
        solver.int_var("x").assert_gt("x", 10).assert_lt("x", 5);
        assert!(solver.check_sat().is_unsat());
    }

    #[test]
    fn test_is_sat_is_unsat() {
        let mut solver = EasySolver::new();
        solver.bool_var("p").assert_true("p");
        assert!(solver.is_sat());

        let mut solver2 = EasySolver::new();
        solver2.bool_var("p").assert_true("p").assert_false("p");
        assert!(solver2.is_unsat());
    }

    #[test]
    fn test_get_model_value_int() {
        let mut solver = EasySolver::new();
        solver.int_var("x").assert_eq_int("x", 42);
        assert!(solver.check_sat().is_sat());
        assert_eq!(solver.get_int_value("x"), Some(42));
        assert_eq!(solver.get_model_value("x").as_deref(), Some("42"));
    }

    #[test]
    fn test_get_model_value_bool() {
        let mut solver = EasySolver::new();
        solver.bool_var("p").assert_true("p");
        assert!(solver.check_sat().is_sat());
        assert_eq!(solver.get_bool_value("p"), Some(true));
    }

    #[test]
    fn test_multiple_vars() {
        let mut solver = EasySolver::new();
        solver
            .int_var("x")
            .int_var("y")
            .assert_gt("x", 0)
            .assert_lt("y", 10)
            .assert_sum_eq("x", "y", 7);
        assert!(solver.check_sat().is_sat());
        // The solver should find values that satisfy x > 0, y < 10, x + y = 7
        // We just verify the sum constraint holds if model values are available
        if let (Some(x_val), Some(y_val)) = (solver.get_int_value("x"), solver.get_int_value("y")) {
            assert_eq!(x_val + y_val, 7);
        }
    }

    #[test]
    fn test_push_pop() {
        let mut solver = EasySolver::new();
        solver.int_var("x").assert_gt("x", 0);
        solver.push();
        solver.assert_lt("x", -1);
        assert!(solver.is_unsat());
        solver.pop();
        assert!(solver.is_sat());
    }

    #[test]
    fn test_ge_le_constraints() {
        let mut solver = EasySolver::new();
        solver.int_var("x").assert_ge("x", 5).assert_le("x", 5);
        assert!(solver.check_sat().is_sat());
        assert_eq!(solver.get_int_value("x"), Some(5));
    }

    #[test]
    fn test_eq_vars_constraint() {
        let mut solver = EasySolver::new();
        solver
            .int_var("x")
            .int_var("y")
            .assert_eq_int("x", 10)
            .assert_eq_vars("x", "y");
        assert!(solver.check_sat().is_sat());
        assert_eq!(solver.get_int_value("y"), Some(10));
    }

    #[test]
    fn test_neq_vars_constraint() {
        let mut solver = EasySolver::new();
        solver
            .int_var("x")
            .int_var("y")
            .assert_eq_int("x", 5)
            .assert_eq_int("y", 5)
            .assert_neq_vars("x", "y");
        assert!(solver.check_sat().is_unsat());
    }

    #[test]
    fn test_easy_result_methods() {
        assert!(EasyResult::Sat.is_sat());
        assert!(!EasyResult::Sat.is_unsat());
        assert!(EasyResult::Unsat.is_unsat());
        assert!(EasyResult::Unknown.is_unknown());
        assert!(EasyResult::Error("test".into()).is_error());
        assert!(EasyResult::ResourceExhausted(ResourceExhausted::Timeout).is_resource_exhausted());
    }

    #[test]
    fn test_timeout_integration() {
        let mut solver = EasySolver::new();
        solver
            .int_var("x")
            .assert_gt("x", 0)
            .timeout(Duration::from_secs(60));
        assert!(solver.check_sat().is_sat());
    }

    #[test]
    fn test_reset() {
        let mut solver = EasySolver::new();
        solver.int_var("x").assert_gt("x", 0);
        assert!(solver.is_sat());
        solver.reset();
        assert!(solver.check_sat().is_sat());
    }

    #[test]
    fn test_get_model_value_before_check() {
        let solver = EasySolver::new();
        assert!(solver.get_model_value("x").is_none());
        assert!(solver.get_int_value("x").is_none());
        assert!(solver.get_bool_value("x").is_none());
    }
}
