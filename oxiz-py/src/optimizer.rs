//! Python wrapper for the MaxSMT Optimizer.

use ::oxiz::core::ast::TermKind;
use ::oxiz::solver::{Model, OptimizationResult, Optimizer};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;

use crate::results::PyOptimizationResult;
use crate::term::{PyTerm, PyTermManager};

/// MaxSMT / Optimization solver.
///
/// Supports minimization and maximization of objective terms over
/// a set of hard constraints.
///
/// Example::
///
///     tm = oxiz.TermManager()
///     opt = oxiz.Optimizer()
///
///     x = tm.mk_var("x", "Int")
///     opt.assert_term(tm.mk_ge(x, tm.mk_int(0)))
///     opt.minimize(x)
///     result = opt.optimize(tm)
///     # result == OptimizationResult.Optimal
#[pyclass(name = "Optimizer", unsendable)]
pub struct PyOptimizer {
    inner: RefCell<Optimizer>,
    last_model: RefCell<Option<Model>>,
}

#[pymethods]
impl PyOptimizer {
    /// Create a new Optimizer.
    #[new]
    fn new() -> Self {
        Self {
            inner: RefCell::new(Optimizer::new()),
            last_model: RefCell::new(None),
        }
    }

    /// Assert a hard constraint.
    fn assert_term(&self, term: &PyTerm) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.assert(term.id);
    }

    /// Add a minimization objective.
    fn minimize(&self, term: &PyTerm) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.minimize(term.id);
    }

    /// Add a maximization objective.
    fn maximize(&self, term: &PyTerm) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.maximize(term.id);
    }

    /// Set the SMT-LIB2 logic.
    fn set_logic(&self, logic: &str) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.set_logic(logic);
    }

    /// Push a new scope.
    fn push(&self) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.push();
    }

    /// Pop a scope.
    fn pop(&self) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.pop();
    }

    /// Run optimization and return the result.
    fn optimize(&self, tm: &PyTermManager) -> PyOptimizationResult {
        let mut optimizer = self.inner.borrow_mut();
        let mut manager = tm.inner.borrow_mut();
        let result = optimizer.optimize(&mut manager);

        let py_result = PyOptimizationResult::from(&result);

        if let OptimizationResult::Optimal { model, .. } = result {
            *self.last_model.borrow_mut() = Some(model.clone());
        }

        py_result
    }

    /// Get the optimal model as a string-keyed dictionary.
    fn get_model<'py>(&self, py: Python<'py>, tm: &PyTermManager) -> PyResult<Bound<'py, PyDict>> {
        let manager = tm.inner.borrow();
        let dict = PyDict::new(py);

        if let Some(model) = self.last_model.borrow().as_ref() {
            for (&var_id, &value_id) in model.assignments() {
                if let Some(var_term) = manager.get(var_id) {
                    if let TermKind::Var(spur) = &var_term.kind {
                        let var_name = manager.resolve_str(*spur);

                        let value_str = if let Some(value_term) = manager.get(value_id) {
                            match &value_term.kind {
                                TermKind::True => "true".to_string(),
                                TermKind::False => "false".to_string(),
                                TermKind::IntConst(n) => n.to_string(),
                                TermKind::RealConst(r) => {
                                    if *r.denom() == 1 {
                                        r.numer().to_string()
                                    } else {
                                        format!("{}/{}", r.numer(), r.denom())
                                    }
                                }
                                _ => format!("{:?}", value_term.kind),
                            }
                        } else {
                            format!("Term({})", value_id.raw())
                        };

                        dict.set_item(var_name, value_str)?;
                    }
                }
            }
        }

        Ok(dict)
    }
}
