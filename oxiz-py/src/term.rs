//! Python wrapper for Term, TermManager, and expression builder operators.

use ::oxiz::core::ast::{TermId, TermManager};
use num_bigint::BigInt;
use num_rational::Rational64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;

/// Python wrapper for TermId.
///
/// Terms are immutable handles into a TermManager's storage.
/// They support Python arithmetic and comparison operators for ergonomic
/// formula construction without requiring explicit mk_* calls.
#[pyclass(name = "Term", from_py_object)]
pub struct PyTerm {
    pub(crate) id: TermId,
    /// Optional back-reference to the owning TermManager, needed for operator
    /// overloads.  May be None for terms created without a context reference
    /// (e.g., when extracted from a model).
    pub(crate) owner: Option<Py<PyTermManager>>,
}

// PyO3's `from_py_object` requires Clone; we implement it manually so that
// cloned Terms keep their owner reference (requires the GIL, which we
// acquire temporarily via Python::attach in PyO3 0.28+).
impl Clone for PyTerm {
    fn clone(&self) -> Self {
        let owner = self
            .owner
            .as_ref()
            .map(|py_obj| Python::attach(|py| py_obj.clone_ref(py)));
        Self { id: self.id, owner }
    }
}

#[pymethods]
impl PyTerm {
    /// Raw numeric term ID.
    #[getter]
    pub fn id(&self) -> u32 {
        self.id.raw()
    }

    fn __repr__(&self) -> String {
        format!("Term({})", self.id.raw())
    }

    fn __eq__(&self, other: &PyTerm) -> bool {
        self.id == other.id
    }

    fn __hash__(&self) -> u64 {
        self.id.raw() as u64
    }

    // ------------------------------------------------------------------ //
    // Arithmetic operators                                                  //
    // ------------------------------------------------------------------ //

    fn __add__(&self, py: Python<'_>, other: &PyTerm) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_add(vec![self.id, other.id]);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }

    fn __sub__(&self, py: Python<'_>, other: &PyTerm) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_sub(self.id, other.id);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }

    fn __mul__(&self, py: Python<'_>, other: &PyTerm) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_mul(vec![self.id, other.id]);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_neg(self.id);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }

    // ------------------------------------------------------------------ //
    // Comparison operators (return boolean Term, not Python bool)          //
    // ------------------------------------------------------------------ //

    fn __lt__(&self, py: Python<'_>, other: &PyTerm) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_lt(self.id, other.id);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }

    fn __le__(&self, py: Python<'_>, other: &PyTerm) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_le(self.id, other.id);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }

    fn __gt__(&self, py: Python<'_>, other: &PyTerm) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_gt(self.id, other.id);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }

    fn __ge__(&self, py: Python<'_>, other: &PyTerm) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_ge(self.id, other.id);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }

    /// Structural equality as a SMT term (`==` returns a bool Term, not a Python bool).
    fn eq_term(&self, py: Python<'_>, other: &PyTerm) -> PyResult<PyTerm> {
        let tm_ref = self.require_owner()?;
        let tm = tm_ref.borrow(py);
        let mut inner = tm.inner.borrow_mut();
        let result = inner.mk_eq(self.id, other.id);
        Ok(PyTerm::with_owner(result, tm_ref.clone_ref(py)))
    }
}

impl PyTerm {
    /// Create a Term without an owner reference (e.g., from a raw TermId).
    pub fn bare(id: TermId) -> Self {
        Self { id, owner: None }
    }

    /// Create a Term with an owner reference.
    pub fn with_owner(id: TermId, owner: Py<PyTermManager>) -> Self {
        Self {
            id,
            owner: Some(owner),
        }
    }

    fn require_owner(&self) -> PyResult<&Py<PyTermManager>> {
        self.owner.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "This Term has no associated TermManager. \
                 Create terms via ctx.int_const(), ctx.bool_const(), or tm.mk_var().",
            )
        })
    }
}

impl From<TermId> for PyTerm {
    fn from(id: TermId) -> Self {
        Self::bare(id)
    }
}

impl From<PyTerm> for TermId {
    fn from(term: PyTerm) -> Self {
        term.id
    }
}

// ====================================================================== //
// PyTermManager                                                            //
// ====================================================================== //

/// Term manager: factory and storage for SMT terms.
///
/// All terms created by a TermManager must be used with the same manager.
/// The TermManager is not thread-safe; use separate instances per thread.
///
/// For a more ergonomic API that supports operator overloads, use `Context`
/// instead.
#[pyclass(name = "TermManager", unsendable)]
pub struct PyTermManager {
    pub(crate) inner: RefCell<TermManager>,
}

impl Default for PyTermManager {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyTermManager {
    /// Create a new TermManager.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(TermManager::new()),
        }
    }

    /// Create a variable with a given name and sort.
    ///
    /// Args:
    ///     name: Variable name.
    ///     sort_name: Sort name ("Bool", "Int", "Real", or "BitVec[N]").
    fn mk_var(&self, name: &str, sort_name: &str) -> PyResult<PyTerm> {
        let mut tm = self.inner.borrow_mut();
        let sort = parse_sort_name(&mut tm, sort_name)?;
        let term_id = tm.mk_var(name, sort);
        Ok(PyTerm::bare(term_id))
    }

    /// Create a boolean constant.
    fn mk_bool(&self, value: bool) -> PyTerm {
        let tm = self.inner.borrow();
        PyTerm::bare(tm.mk_bool(value))
    }

    /// Create an integer constant.
    fn mk_int(&self, value: i64) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_int(BigInt::from(value)))
    }

    /// Create a rational real constant (numerator/denominator).
    fn mk_real(&self, numerator: i64, denominator: i64) -> PyResult<PyTerm> {
        if denominator == 0 {
            return Err(PyValueError::new_err("Denominator cannot be zero"));
        }
        let mut tm = self.inner.borrow_mut();
        let rational = Rational64::new(numerator, denominator);
        Ok(PyTerm::bare(tm.mk_real(rational)))
    }

    fn mk_not(&self, term: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_not(term.id))
    }

    fn mk_and(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::bare(tm.mk_and(ids))
    }

    fn mk_or(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::bare(tm.mk_or(ids))
    }

    fn mk_implies(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_implies(lhs.id, rhs.id))
    }

    fn mk_eq(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_eq(lhs.id, rhs.id))
    }

    fn mk_add(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::bare(tm.mk_add(ids))
    }

    fn mk_sub(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_sub(lhs.id, rhs.id))
    }

    fn mk_mul(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::bare(tm.mk_mul(ids))
    }

    fn mk_lt(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_lt(lhs.id, rhs.id))
    }

    fn mk_le(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_le(lhs.id, rhs.id))
    }

    fn mk_gt(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_gt(lhs.id, rhs.id))
    }

    fn mk_ge(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_ge(lhs.id, rhs.id))
    }

    fn mk_ite(&self, cond: &PyTerm, then_branch: &PyTerm, else_branch: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_ite(cond.id, then_branch.id, else_branch.id))
    }

    fn mk_distinct(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::bare(tm.mk_distinct(ids))
    }

    fn mk_neg(&self, term: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_neg(term.id))
    }

    fn mk_div(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_div(lhs.id, rhs.id))
    }

    fn mk_mod(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_mod(lhs.id, rhs.id))
    }

    fn mk_xor(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_xor(lhs.id, rhs.id))
    }

    // ------------------------------------------------------------------ //
    // BitVec operations                                                    //
    // ------------------------------------------------------------------ //

    fn mk_bv(&self, value: i64, width: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bitvec(BigInt::from(value), width))
    }

    fn mk_bv_concat(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_concat(lhs.id, rhs.id))
    }

    fn mk_bv_extract(&self, high: u32, low: u32, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_extract(high, low, arg.id))
    }

    fn mk_bv_not(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_not(arg.id))
    }

    fn mk_bv_and(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_and(lhs.id, rhs.id))
    }

    fn mk_bv_or(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_or(lhs.id, rhs.id))
    }

    fn mk_bv_add(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_add(lhs.id, rhs.id))
    }

    fn mk_bv_sub(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_sub(lhs.id, rhs.id))
    }

    fn mk_bv_mul(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_mul(lhs.id, rhs.id))
    }

    fn mk_bv_neg(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_neg(arg.id))
    }

    fn mk_bv_ult(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_ult(lhs.id, rhs.id))
    }

    fn mk_bv_slt(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_slt(lhs.id, rhs.id))
    }

    fn mk_bv_ule(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_ule(lhs.id, rhs.id))
    }

    fn mk_bv_sle(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_sle(lhs.id, rhs.id))
    }

    fn mk_bv_udiv(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_udiv(lhs.id, rhs.id))
    }

    fn mk_bv_sdiv(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_sdiv(lhs.id, rhs.id))
    }

    fn mk_bv_urem(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_urem(lhs.id, rhs.id))
    }

    fn mk_bv_srem(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bv_srem(lhs.id, rhs.id))
    }

    // ------------------------------------------------------------------ //
    // Array operations                                                     //
    // ------------------------------------------------------------------ //

    fn mk_select(&self, array: &PyTerm, index: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_select(array.id, index.id))
    }

    fn mk_store(&self, array: &PyTerm, index: &PyTerm, value: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_store(array.id, index.id, value.id))
    }

    // ------------------------------------------------------------------ //
    // Utilities                                                            //
    // ------------------------------------------------------------------ //

    /// Return a human-readable string representation of a term.
    fn term_to_string(&self, term: &PyTerm) -> String {
        let tm = self.inner.borrow();
        if let Some(t) = tm.get(term.id) {
            format!("{:?}", t.kind)
        } else {
            format!("Term({})", term.id.raw())
        }
    }
}

/// Parse a sort name string into a SortId.
pub(crate) fn parse_sort_name(
    tm: &mut TermManager,
    sort_name: &str,
) -> PyResult<::oxiz::core::sort::SortId> {
    match sort_name {
        "Bool" => Ok(tm.sorts.bool_sort),
        "Int" => Ok(tm.sorts.int_sort),
        "Real" => Ok(tm.sorts.real_sort),
        s if s.starts_with("BitVec[") && s.ends_with(']') => {
            let width_str = &s[7..s.len() - 1];
            let width: u32 = width_str.parse().map_err(|_| {
                PyValueError::new_err(format!("Invalid BitVec width: {}", width_str))
            })?;
            Ok(tm.sorts.bitvec(width))
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown sort: '{}'. Use 'Bool', 'Int', 'Real', or 'BitVec[N]'",
            sort_name
        ))),
    }
}
