//! Module-level boolean and arithmetic combinator functions.
//!
//! These mirror z3-python's top-level helpers:
//!   `And(a, b, c)`, `Or(a, b, c)`, `Not(a)`, `Implies(a, b)`, `If(cond, t, e)`

// Allow non-snake-case names to mirror z3-python's capitalized API conventions.
#![allow(non_snake_case)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::term::{PyTerm, PyTermManager};

/// Construct the conjunction of zero or more boolean Terms.
///
/// All terms must share the same TermManager. The manager is obtained from the
/// first term that carries an owner reference. If no terms have an owner, the
/// caller must pass `tm` explicitly via TermManager.mk_and().
///
/// Example::
///
///     result = oxiz.And(a, b, c)
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn And(py: Python<'_>, args: Vec<PyTerm>) -> PyResult<PyTerm> {
    if args.is_empty() {
        return Err(PyValueError::new_err(
            "And() requires at least one Term argument",
        ));
    }
    let tm_ref = first_owner(py, &args)?;
    let tm_guard = tm_ref.borrow(py);
    let mut inner = tm_guard.inner.borrow_mut();
    let ids: Vec<_> = args.iter().map(|t| t.id).collect();
    let result = inner.mk_and(ids);
    drop(inner);
    drop(tm_guard);
    Ok(PyTerm::with_owner(result, tm_ref))
}

/// Construct the disjunction of zero or more boolean Terms.
///
/// Example::
///
///     result = oxiz.Or(a, b, c)
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn Or(py: Python<'_>, args: Vec<PyTerm>) -> PyResult<PyTerm> {
    if args.is_empty() {
        return Err(PyValueError::new_err(
            "Or() requires at least one Term argument",
        ));
    }
    let tm_ref = first_owner(py, &args)?;
    let tm_guard = tm_ref.borrow(py);
    let mut inner = tm_guard.inner.borrow_mut();
    let ids: Vec<_> = args.iter().map(|t| t.id).collect();
    let result = inner.mk_or(ids);
    drop(inner);
    drop(tm_guard);
    Ok(PyTerm::with_owner(result, tm_ref))
}

/// Construct the logical negation of a boolean Term.
///
/// Example::
///
///     result = oxiz.Not(a)
#[pyfunction]
pub fn Not(py: Python<'_>, term: &PyTerm) -> PyResult<PyTerm> {
    let tm_ref = term_owner(py, term)?;
    let tm_guard = tm_ref.borrow(py);
    let mut inner = tm_guard.inner.borrow_mut();
    let result = inner.mk_not(term.id);
    drop(inner);
    drop(tm_guard);
    Ok(PyTerm::with_owner(result, tm_ref))
}

/// Construct the implication `lhs => rhs`.
///
/// Example::
///
///     result = oxiz.Implies(a, b)
#[pyfunction]
pub fn Implies(py: Python<'_>, lhs: &PyTerm, rhs: &PyTerm) -> PyResult<PyTerm> {
    let tm_ref = term_owner(py, lhs).or_else(|_| term_owner(py, rhs))?;
    let tm_guard = tm_ref.borrow(py);
    let mut inner = tm_guard.inner.borrow_mut();
    let result = inner.mk_implies(lhs.id, rhs.id);
    drop(inner);
    drop(tm_guard);
    Ok(PyTerm::with_owner(result, tm_ref))
}

/// Construct an if-then-else expression.
///
/// Example::
///
///     result = oxiz.If(cond, then_val, else_val)
#[pyfunction]
pub fn If(py: Python<'_>, cond: &PyTerm, then_: &PyTerm, else_: &PyTerm) -> PyResult<PyTerm> {
    let tm_ref = term_owner(py, cond)
        .or_else(|_| term_owner(py, then_))
        .or_else(|_| term_owner(py, else_))?;
    let tm_guard = tm_ref.borrow(py);
    let mut inner = tm_guard.inner.borrow_mut();
    let result = inner.mk_ite(cond.id, then_.id, else_.id);
    drop(inner);
    drop(tm_guard);
    Ok(PyTerm::with_owner(result, tm_ref))
}

// ------------------------------------------------------------------ //
// Versions that accept an explicit TermManager for bare Terms         //
// ------------------------------------------------------------------ //

/// And(...) variant accepting an explicit TermManager.
///
/// Use when your Terms were created without a Context (e.g., via
/// `TermManager.mk_var()`).
#[pyfunction]
#[pyo3(name = "And_tm")]
#[pyo3(signature = (tm, *args))]
pub fn and_tm(py: Python<'_>, tm: &PyTermManager, args: Vec<PyTerm>) -> PyResult<PyTerm> {
    if args.is_empty() {
        return Err(PyValueError::new_err(
            "And_tm() requires at least one Term argument",
        ));
    }
    let mut inner = tm.inner.borrow_mut();
    let ids: Vec<_> = args.iter().map(|t| t.id).collect();
    let result = inner.mk_and(ids);
    drop(inner);
    // Return a bare term (no owner) as the caller used an explicit TM.
    Ok(PyTerm::bare(result))
}

/// Or(...) variant accepting an explicit TermManager.
#[pyfunction]
#[pyo3(name = "Or_tm")]
#[pyo3(signature = (tm, *args))]
pub fn or_tm(py: Python<'_>, tm: &PyTermManager, args: Vec<PyTerm>) -> PyResult<PyTerm> {
    if args.is_empty() {
        return Err(PyValueError::new_err(
            "Or_tm() requires at least one Term argument",
        ));
    }
    let mut inner = tm.inner.borrow_mut();
    let ids: Vec<_> = args.iter().map(|t| t.id).collect();
    let result = inner.mk_or(ids);
    drop(inner);
    Ok(PyTerm::bare(result))
}

/// Not(...) variant accepting an explicit TermManager.
#[pyfunction]
#[pyo3(name = "Not_tm")]
pub fn not_tm(_py: Python<'_>, tm: &PyTermManager, term: &PyTerm) -> PyResult<PyTerm> {
    let mut inner = tm.inner.borrow_mut();
    let result = inner.mk_not(term.id);
    Ok(PyTerm::bare(result))
}

// ------------------------------------------------------------------ //
// Internal helpers                                                     //
// ------------------------------------------------------------------ //

/// Extract a cloned Py<PyTermManager> from the first term with an owner.
fn first_owner(py: Python<'_>, terms: &[PyTerm]) -> PyResult<Py<PyTermManager>> {
    terms
        .iter()
        .find_map(|t| t.owner.as_ref().map(|o| o.clone_ref(py)))
        .ok_or_else(|| {
            PyValueError::new_err(
                "None of the provided Terms carry a TermManager reference. \
                 Create terms via ctx.int_const() / ctx.bool_const(), or use \
                 the explicit _tm variants (And_tm, Or_tm, Not_tm).",
            )
        })
}

fn term_owner(py: Python<'_>, term: &PyTerm) -> PyResult<Py<PyTermManager>> {
    term.owner.as_ref().map(|o| o.clone_ref(py)).ok_or_else(|| {
        PyValueError::new_err(
            "This Term has no associated TermManager. \
                 Create terms via ctx.int_const() / ctx.bool_const().",
        )
    })
}
