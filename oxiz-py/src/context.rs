//! Context class — ergonomic high-level API mirroring z3-python's Context.
//!
//! A `Context` bundles a `TermManager` and provides typed named-constant
//! factories that return `Term` objects with a back-reference to the owning
//! TermManager, enabling Python operator overloads (`+`, `<`, etc.) on the
//! returned `Term` objects.

use num_bigint::BigInt;
use num_rational::Rational64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;

use crate::term::{PyTerm, PyTermManager, parse_sort_name};
use ::oxiz::core::ast::TermManager;

/// High-level context that bundles a TermManager and provides named-constant
/// factories with operator-overload support.
///
/// Example::
///
///     ctx = oxiz.Context()
///     x = ctx.int_const("x")
///     y = ctx.real_const("y")
///     b = ctx.bool_const("b")
///     v = ctx.bv_const("v", 32)
///
///     solver = oxiz.Solver()
///     solver.add(x + y > ctx.real(0, 1))
///     result = solver.check(ctx.tm)
#[pyclass(name = "Context", unsendable)]
pub struct PyContext {
    /// The underlying TermManager (also accessible as `ctx.tm`).
    pub tm: Py<PyTermManager>,
}

#[pymethods]
impl PyContext {
    /// Create a new Context (and its internal TermManager).
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        let tm = Py::new(
            py,
            PyTermManager {
                inner: RefCell::new(TermManager::new()),
            },
        )?;
        Ok(Self { tm })
    }

    /// Access the underlying TermManager.
    #[getter]
    fn tm(&self, py: Python<'_>) -> Py<PyTermManager> {
        self.tm.clone_ref(py)
    }

    // ------------------------------------------------------------------ //
    // Named-constant factories                                             //
    // ------------------------------------------------------------------ //

    /// Declare an integer constant named `name`.
    fn int_const(&self, py: Python<'_>, name: &str) -> PyResult<PyTerm> {
        let tm_guard = self.tm.borrow(py);
        let mut inner = tm_guard.inner.borrow_mut();
        let sort = inner.sorts.int_sort;
        let id = inner.mk_var(name, sort);
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }

    /// Declare a real-valued constant named `name`.
    fn real_const(&self, py: Python<'_>, name: &str) -> PyResult<PyTerm> {
        let tm_guard = self.tm.borrow(py);
        let mut inner = tm_guard.inner.borrow_mut();
        let sort = inner.sorts.real_sort;
        let id = inner.mk_var(name, sort);
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }

    /// Declare a boolean constant named `name`.
    fn bool_const(&self, py: Python<'_>, name: &str) -> PyResult<PyTerm> {
        let tm_guard = self.tm.borrow(py);
        let mut inner = tm_guard.inner.borrow_mut();
        let sort = inner.sorts.bool_sort;
        let id = inner.mk_var(name, sort);
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }

    /// Declare a bitvector constant named `name` with bit-width `width`.
    fn bv_const(&self, py: Python<'_>, name: &str, width: u32) -> PyResult<PyTerm> {
        let tm_guard = self.tm.borrow(py);
        let mut inner = tm_guard.inner.borrow_mut();
        let sort = inner.sorts.bitvec(width);
        let id = inner.mk_var(name, sort);
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }

    /// Declare a constant with an explicit sort name.
    fn const_of_sort(&self, py: Python<'_>, name: &str, sort_name: &str) -> PyResult<PyTerm> {
        let tm_guard = self.tm.borrow(py);
        let mut inner = tm_guard.inner.borrow_mut();
        let sort = parse_sort_name(&mut inner, sort_name)?;
        let id = inner.mk_var(name, sort);
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }

    // ------------------------------------------------------------------ //
    // Literal factories                                                    //
    // ------------------------------------------------------------------ //

    /// Create an integer literal.
    fn int_val(&self, py: Python<'_>, value: i64) -> PyResult<PyTerm> {
        let tm_guard = self.tm.borrow(py);
        let mut inner = tm_guard.inner.borrow_mut();
        let id = inner.mk_int(BigInt::from(value));
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }

    /// Create a real literal (numerator / denominator).
    fn real_val(&self, py: Python<'_>, numerator: i64, denominator: i64) -> PyResult<PyTerm> {
        if denominator == 0 {
            return Err(PyValueError::new_err("Denominator cannot be zero"));
        }
        let tm_guard = self.tm.borrow(py);
        let mut inner = tm_guard.inner.borrow_mut();
        let rational = Rational64::new(numerator, denominator);
        let id = inner.mk_real(rational);
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }

    /// Create a boolean literal.
    fn bool_val(&self, py: Python<'_>, value: bool) -> PyResult<PyTerm> {
        let tm_guard = self.tm.borrow(py);
        let inner = tm_guard.inner.borrow();
        let id = inner.mk_bool(value);
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }

    /// Create a bitvector literal.
    fn bv_val(&self, py: Python<'_>, value: i64, width: u32) -> PyResult<PyTerm> {
        let tm_guard = self.tm.borrow(py);
        let mut inner = tm_guard.inner.borrow_mut();
        let id = inner.mk_bitvec(BigInt::from(value), width);
        Ok(PyTerm::with_owner(id, self.tm.clone_ref(py)))
    }
}
