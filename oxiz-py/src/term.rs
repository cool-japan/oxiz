//! Python wrapper for Term, TermManager, and expression builder operators.

use ::oxiz::core::ast::{RoundingMode, TermId, TermKind, TermManager};
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
// PySort                                                                   //
// ====================================================================== //

/// An SMT sort (type) object, returned by sort constructors such as
/// :func:`FPSort`, :func:`ArraySort`, :func:`StringSort`, :func:`IntSort`,
/// and :func:`BoolSort`.
///
/// Sort objects are lightweight wrappers around the ``SortId`` handle stored
/// inside the ``TermManager``.  They carry extra metadata (``eb``, ``sb`` for
/// FP sorts; ``is_array``, ``is_string`` flags) so that higher-level
/// combinators can dispatch correctly.
#[pyclass(name = "Sort", from_py_object)]
#[derive(Clone)]
pub struct PySort {
    /// The underlying sort ID inside the TermManager.
    pub(crate) id: ::oxiz::core::SortId,
    /// Exponent bit-width for FP sorts, ``None`` otherwise.
    pub(crate) eb: Option<u32>,
    /// Significand bit-width for FP sorts, ``None`` otherwise.
    pub(crate) sb: Option<u32>,
    /// True when this is an array sort.
    pub(crate) is_array: bool,
    /// True when this is the string sort.
    pub(crate) is_string: bool,
}

#[pymethods]
impl PySort {
    fn __repr__(&self) -> String {
        if let (Some(eb), Some(sb)) = (self.eb, self.sb) {
            format!("FPSort({}, {})", eb, sb)
        } else if self.is_array {
            "ArraySort(...)".to_string()
        } else if self.is_string {
            "StringSort".to_string()
        } else {
            format!("Sort({})", self.id.raw())
        }
    }

    fn __eq__(&self, other: &PySort) -> bool {
        self.id == other.id
    }

    fn __hash__(&self) -> u64 {
        self.id.raw() as u64
    }

    /// True if this is a floating-point sort.
    #[getter]
    fn is_fp(&self) -> bool {
        self.eb.is_some()
    }

    /// Exponent bit-width (FP sorts only, ``None`` otherwise).
    #[getter]
    fn eb(&self) -> Option<u32> {
        self.eb
    }

    /// Significand bit-width (FP sorts only, ``None`` otherwise).
    #[getter]
    fn sb(&self) -> Option<u32> {
        self.sb
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
    ///     sort_name: Sort descriptor — see :func:`parse_sort_name` for the full
    ///         grammar.  Examples: ``"Bool"``, ``"Int"``, ``"Real"``,
    ///         ``"String"``, ``"BitVec[32]"``, ``"Float[8,24]"``,
    ///         ``"Array[Int,Bool]"``.
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

    /// Create a bitvector literal of `width` bits from an arbitrary-size
    /// Python `int`.
    ///
    /// `value` is accepted as a native `num-bigint::BigInt` (via PyO3's
    /// `num-bigint` conversion feature), so values that do not fit in an
    /// `i64` - e.g. literals for bitvectors wider than 64 bits, which are
    /// common in QF_BV crypto-style queries - round-trip exactly instead of
    /// overflowing/truncating.
    fn mk_bv(&self, value: BigInt, width: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_bitvec(value, width))
    }

    /// Concatenate two bitvector terms.
    ///
    /// Raises `ValueError` if either operand is not bitvector-sorted: a
    /// binding hands the term manager terms it did not sort-check itself, so
    /// the sort error is surfaced to Python rather than absorbed into a
    /// fabricated result width.
    fn mk_bv_concat(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyResult<PyTerm> {
        let mut tm = self.inner.borrow_mut();
        let id = tm
            .try_mk_bv_concat(lhs.id, rhs.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTerm::bare(id))
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
    // Quantifiers                                                          //
    // ------------------------------------------------------------------ //

    /// Create a universal quantifier.
    ///
    /// Args:
    ///     vars: List of ``(name, sort_name)`` pairs for bound variables.
    ///     body: The body term.
    ///
    /// Returns:
    ///     A Term representing ``forall vars. body``.
    fn mk_forall(&self, vars: Vec<(String, String)>, body: &PyTerm) -> PyResult<PyTerm> {
        let mut tm = self.inner.borrow_mut();
        let parsed: Vec<(String, ::oxiz::core::SortId)> = vars
            .iter()
            .map(|(name, sort_name)| {
                parse_sort_name(&mut tm, sort_name).map(|sid| (name.clone(), sid))
            })
            .collect::<PyResult<_>>()?;
        let refs: Vec<(&str, ::oxiz::core::SortId)> =
            parsed.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        Ok(PyTerm::bare(tm.mk_forall(refs, body.id)))
    }

    /// Create an existential quantifier.
    ///
    /// Args:
    ///     vars: List of ``(name, sort_name)`` pairs for bound variables.
    ///     body: The body term.
    ///
    /// Returns:
    ///     A Term representing ``exists vars. body``.
    fn mk_exists(&self, vars: Vec<(String, String)>, body: &PyTerm) -> PyResult<PyTerm> {
        let mut tm = self.inner.borrow_mut();
        let parsed: Vec<(String, ::oxiz::core::SortId)> = vars
            .iter()
            .map(|(name, sort_name)| {
                parse_sort_name(&mut tm, sort_name).map(|sid| (name.clone(), sid))
            })
            .collect::<PyResult<_>>()?;
        let refs: Vec<(&str, ::oxiz::core::SortId)> =
            parsed.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        Ok(PyTerm::bare(tm.mk_exists(refs, body.id)))
    }

    // ------------------------------------------------------------------ //
    // String operations                                                    //
    // ------------------------------------------------------------------ //

    /// Create a string literal term (alias: ``mk_string_val``).
    fn mk_string_lit(&self, value: &str) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_string_lit(value))
    }

    /// Create a string literal term.
    ///
    /// This is the preferred name; ``mk_string_lit`` is an alias.
    fn mk_string_val(&self, value: &str) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_string_lit(value))
    }

    /// Compute the length of a string term (alias for ``mk_str_len``).
    fn mk_str_length(&self, s: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_len(s.id))
    }

    /// Test whether ``s`` contains ``sub`` (alias for ``mk_str_contains``).
    fn mk_str_contains_term(&self, s: &PyTerm, sub: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_contains(s.id, sub.id))
    }

    /// Concatenate two string terms.
    fn mk_str_concat(&self, s1: &PyTerm, s2: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_concat(s1.id, s2.id))
    }

    /// Compute the length of a string term.
    fn mk_str_len(&self, s: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_len(s.id))
    }

    /// Extract a substring: ``substr(s, start, len)``.
    fn mk_str_substr(&self, s: &PyTerm, start: &PyTerm, len: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_substr(s.id, start.id, len.id))
    }

    /// Return the character of ``s`` at position ``i``.
    fn mk_str_at(&self, s: &PyTerm, i: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_at(s.id, i.id))
    }

    /// Test whether ``s`` contains ``sub``.
    fn mk_str_contains(&self, s: &PyTerm, sub: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_contains(s.id, sub.id))
    }

    /// Test whether ``prefix`` is a prefix of ``s``.
    fn mk_str_prefixof(&self, prefix: &PyTerm, s: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_prefixof(prefix.id, s.id))
    }

    /// Test whether ``suffix`` is a suffix of ``s``.
    fn mk_str_suffixof(&self, suffix: &PyTerm, s: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_suffixof(suffix.id, s.id))
    }

    /// Return the first occurrence of ``sub`` in ``s`` starting at ``offset``.
    fn mk_str_indexof(&self, s: &PyTerm, sub: &PyTerm, offset: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_indexof(s.id, sub.id, offset.id))
    }

    /// Replace the first occurrence of ``pattern`` in ``s`` with ``replacement``.
    fn mk_str_replace(&self, s: &PyTerm, pattern: &PyTerm, replacement: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_replace(s.id, pattern.id, replacement.id))
    }

    /// Replace all occurrences of ``pattern`` in ``s`` with ``replacement``.
    fn mk_str_replace_all(&self, s: &PyTerm, pattern: &PyTerm, replacement: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_replace_all(s.id, pattern.id, replacement.id))
    }

    /// Convert a string term to an integer term.
    fn mk_str_to_int(&self, s: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_str_to_int(s.id))
    }

    /// Convert an integer term to a string term.
    fn mk_int_to_str(&self, i: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_int_to_str(i.id))
    }

    // ------------------------------------------------------------------ //
    // Floating-point literals                                              //
    // ------------------------------------------------------------------ //

    /// Create an FP literal from sign/exponent/significand components.
    ///
    /// Args:
    ///     sign: Sign bit (``True`` = negative).
    ///     exp: Bitvector exponent as a signed integer.
    ///     sig: Bitvector significand as an unsigned integer.
    ///     eb: Exponent bit-width.
    ///     sb: Significand bit-width (including implicit leading bit).
    fn mk_fp_lit(&self, sign: bool, exp: i64, sig: u64, eb: u32, sb: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_lit(sign, BigInt::from(exp), BigInt::from(sig), eb, sb))
    }

    /// Create floating-point positive infinity for the given format.
    fn mk_fp_plus_infinity(&self, eb: u32, sb: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_plus_infinity(eb, sb))
    }

    /// Create floating-point negative infinity for the given format.
    fn mk_fp_minus_infinity(&self, eb: u32, sb: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_minus_infinity(eb, sb))
    }

    /// Create floating-point positive zero for the given format.
    fn mk_fp_plus_zero(&self, eb: u32, sb: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_plus_zero(eb, sb))
    }

    /// Create floating-point negative zero for the given format.
    fn mk_fp_minus_zero(&self, eb: u32, sb: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_minus_zero(eb, sb))
    }

    /// Create a floating-point NaN value for the given format.
    fn mk_fp_nan(&self, eb: u32, sb: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_nan(eb, sb))
    }

    // ------------------------------------------------------------------ //
    // Floating-point unary operations                                     //
    // ------------------------------------------------------------------ //

    /// Absolute value of an FP term.
    fn mk_fp_abs(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_abs(arg.id))
    }

    /// Negation of an FP term.
    fn mk_fp_neg(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_neg(arg.id))
    }

    /// Square root with rounding mode ``rm`` (``"RNE"``, ``"RNA"``, ``"RTP"``, ``"RTN"``, ``"RTZ"``).
    fn mk_fp_sqrt(&self, rm: &str, arg: &PyTerm) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_sqrt(rounding, arg.id)))
    }

    /// Round an FP term to integral with rounding mode ``rm``.
    fn mk_fp_round_to_integral(&self, rm: &str, arg: &PyTerm) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_round_to_integral(rounding, arg.id)))
    }

    // ------------------------------------------------------------------ //
    // Floating-point binary operations                                    //
    // ------------------------------------------------------------------ //

    /// FP addition with rounding mode.
    fn mk_fp_add(&self, rm: &str, lhs: &PyTerm, rhs: &PyTerm) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_add(rounding, lhs.id, rhs.id)))
    }

    /// FP subtraction with rounding mode.
    fn mk_fp_sub(&self, rm: &str, lhs: &PyTerm, rhs: &PyTerm) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_sub(rounding, lhs.id, rhs.id)))
    }

    /// FP multiplication with rounding mode.
    fn mk_fp_mul(&self, rm: &str, lhs: &PyTerm, rhs: &PyTerm) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_mul(rounding, lhs.id, rhs.id)))
    }

    /// FP division with rounding mode.
    fn mk_fp_div(&self, rm: &str, lhs: &PyTerm, rhs: &PyTerm) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_div(rounding, lhs.id, rhs.id)))
    }

    /// IEEE remainder (no rounding mode argument).
    fn mk_fp_rem(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_rem(lhs.id, rhs.id))
    }

    /// FP minimum.
    fn mk_fp_min(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_min(lhs.id, rhs.id))
    }

    /// FP maximum.
    fn mk_fp_max(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_max(lhs.id, rhs.id))
    }

    // ------------------------------------------------------------------ //
    // Floating-point comparisons                                          //
    // ------------------------------------------------------------------ //

    /// FP less-than-or-equal.
    fn mk_fp_leq(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_leq(lhs.id, rhs.id))
    }

    /// FP less-than.
    fn mk_fp_lt(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_lt(lhs.id, rhs.id))
    }

    /// FP greater-than-or-equal.
    fn mk_fp_geq(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_geq(lhs.id, rhs.id))
    }

    /// FP greater-than.
    fn mk_fp_gt(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_gt(lhs.id, rhs.id))
    }

    /// FP IEEE equality (not SMT ``=``).
    fn mk_fp_eq(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_eq(lhs.id, rhs.id))
    }

    // ------------------------------------------------------------------ //
    // Floating-point ternary                                              //
    // ------------------------------------------------------------------ //

    /// Fused multiply-add: ``(rm * x * y) + z``.
    fn mk_fp_fma(&self, rm: &str, x: &PyTerm, y: &PyTerm, z: &PyTerm) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_fma(rounding, x.id, y.id, z.id)))
    }

    // ------------------------------------------------------------------ //
    // Floating-point predicates                                           //
    // ------------------------------------------------------------------ //

    /// Test whether an FP term is a normal number.
    fn mk_fp_is_normal(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_is_normal(arg.id))
    }

    /// Test whether an FP term is subnormal.
    fn mk_fp_is_subnormal(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_is_subnormal(arg.id))
    }

    /// Test whether an FP term is zero.
    fn mk_fp_is_zero(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_is_zero(arg.id))
    }

    /// Test whether an FP term is infinite.
    fn mk_fp_is_infinite(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_is_infinite(arg.id))
    }

    /// Test whether an FP term is NaN.
    fn mk_fp_is_nan(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_is_nan(arg.id))
    }

    /// Test whether an FP term is negative.
    fn mk_fp_is_negative(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_is_negative(arg.id))
    }

    /// Test whether an FP term is positive.
    fn mk_fp_is_positive(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_is_positive(arg.id))
    }

    // ------------------------------------------------------------------ //
    // Floating-point conversion                                           //
    // ------------------------------------------------------------------ //

    /// Convert an FP term to a different FP format with rounding.
    fn mk_fp_to_fp(&self, rm: &str, arg: &PyTerm, eb: u32, sb: u32) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_to_fp(rounding, arg.id, eb, sb)))
    }

    /// Convert an FP term to a signed bitvector with rounding.
    fn mk_fp_to_sbv(&self, rm: &str, arg: &PyTerm, width: u32) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_to_sbv(rounding, arg.id, width)))
    }

    /// Convert an FP term to an unsigned bitvector with rounding.
    fn mk_fp_to_ubv(&self, rm: &str, arg: &PyTerm, width: u32) -> PyResult<PyTerm> {
        let rounding = parse_rounding_mode(rm)?;
        let mut tm = self.inner.borrow_mut();
        Ok(PyTerm::bare(tm.mk_fp_to_ubv(rounding, arg.id, width)))
    }

    /// Convert an FP term to a real term.
    fn mk_fp_to_real(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_fp_to_real(arg.id))
    }

    // ------------------------------------------------------------------ //
    // Utilities                                                            //
    // ------------------------------------------------------------------ //

    /// Return a human-readable string representation of a term.
    fn term_to_string(&self, term: &PyTerm) -> String {
        let tm = self.inner.borrow();
        if let Some(t) = tm.get(term.id) {
            match &t.kind {
                // The actual string content, not Rust's `{:?}` Debug
                // formatter (which would render e.g. `StringLit("a\"b")`,
                // using *Rust's* escape syntax -- not Python's, and not
                // SMT-LIB's). A Python caller of a "human-readable string
                // representation" of a String-sorted term wants the string
                // back, not a debug-formatted Rust literal.
                TermKind::StringLit(s) => s.clone(),
                other => format!("{other:?}"),
            }
        } else {
            format!("Term({})", term.id.raw())
        }
    }
}

/// Parse a sort name string into a SortId.
///
/// Supported formats:
/// - ``"Bool"``
/// - ``"Int"``
/// - ``"Real"``
/// - ``"String"``
/// - ``"BitVec[N]"`` — N-bit bitvector
/// - ``"Float[eb,sb]"`` or ``"FP[eb,sb]"`` — floating-point with *eb* exponent bits
///   and *sb* significand bits (including the implicit leading bit)
/// - ``"Array[D,R]"`` — array from sort D to sort R (nested bracketed sorts)
///
/// This parser is iterative (explicit heap stack). `Array[...]` nesting is
/// bounded only by the length of the Python string, so a one-liner such as
/// `oxiz.Context().const_of_sort("x", "Array[" * 200000 + "Int" + "]" * 200000)`
/// overflowed the stack against a recursive parser. Errors are still reported
/// through the existing `PyResult` channel, unchanged.
pub(crate) fn parse_sort_name(
    tm: &mut TermManager,
    sort_name: &str,
) -> PyResult<::oxiz::core::SortId> {
    /// Work item for the iterative sort parser.
    enum SortTask<'a> {
        /// Parse this sort name.
        Parse(&'a str),
        /// Build an array sort from the two most recently parsed sorts.
        BuildArray,
    }

    let mut tasks = vec![SortTask::Parse(sort_name)];
    let mut parsed: Vec<::oxiz::core::SortId> = Vec::new();

    while let Some(task) = tasks.pop() {
        let s = match task {
            SortTask::BuildArray => {
                let range = parsed.pop();
                let domain = parsed.pop();
                let (Some(domain), Some(range)) = (domain, range) else {
                    return Err(PyValueError::new_err(format!(
                        "Invalid Array sort '{}': expected 'Array[D,R]'",
                        sort_name
                    )));
                };
                parsed.push(tm.sorts.array(domain, range));
                continue;
            }
            SortTask::Parse(s) => s,
        };

        let sort = match s {
            "Bool" => tm.sorts.bool_sort,
            "Int" => tm.sorts.int_sort,
            "Real" => tm.sorts.real_sort,
            "String" => tm.sorts.string_sort(),
            s if s.starts_with("BitVec[") && s.ends_with(']') => {
                let width_str = &s[7..s.len() - 1];
                let width: u32 = width_str.parse().map_err(|_| {
                    PyValueError::new_err(format!("Invalid BitVec width: {}", width_str))
                })?;
                tm.sorts.bitvec(width)
            }
            s if (s.starts_with("Float[") || s.starts_with("FP[")) && s.ends_with(']') => {
                let inner = if s.starts_with("Float[") {
                    &s[6..s.len() - 1]
                } else {
                    &s[3..s.len() - 1]
                };
                let comma = inner.find(',').ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Invalid Float/FP sort '{}': expected 'Float[eb,sb]'",
                        s
                    ))
                })?;
                let eb: u32 = inner[..comma].trim().parse().map_err(|_| {
                    PyValueError::new_err(format!("Invalid exponent width in sort '{}'", s))
                })?;
                let sb: u32 = inner[comma + 1..].trim().parse().map_err(|_| {
                    PyValueError::new_err(format!("Invalid significand width in sort '{}'", s))
                })?;
                tm.sorts.float_sort(eb, sb)
            }
            s if s.starts_with("Array[") && s.ends_with(']') => {
                // Find the comma separating domain and range, respecting nested brackets.
                let inner = &s[6..s.len() - 1];
                let split = find_top_level_comma(inner).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Invalid Array sort '{}': expected 'Array[D,R]'",
                        s
                    ))
                })?;
                let domain_str = inner[..split].trim();
                let range_str = inner[split + 1..].trim();
                // Domain completes first, so it is pushed last.
                tasks.push(SortTask::BuildArray);
                tasks.push(SortTask::Parse(range_str));
                tasks.push(SortTask::Parse(domain_str));
                continue;
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown sort: '{}'. \
                     Supported: 'Bool', 'Int', 'Real', 'String', 'BitVec[N]', \
                     'Float[eb,sb]', 'FP[eb,sb]', 'Array[D,R]'",
                    other
                )));
            }
        };
        parsed.push(sort);
    }

    parsed
        .pop()
        .ok_or_else(|| PyValueError::new_err(format!("Invalid sort: '{}'", sort_name)))
}

/// Find the index of the first top-level comma in ``s`` (not inside brackets).
fn find_top_level_comma(s: &str) -> Option<usize> {
    let mut depth: usize = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth = depth.saturating_sub(1);
            }
            ',' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Parse a rounding-mode string into a [`RoundingMode`].
///
/// Valid values: ``"RNE"``, ``"RNA"``, ``"RTP"``, ``"RTN"``, ``"RTZ"``.
pub(crate) fn parse_rounding_mode(rm: &str) -> PyResult<RoundingMode> {
    match rm {
        "RNE" => Ok(RoundingMode::RNE),
        "RNA" => Ok(RoundingMode::RNA),
        "RTP" => Ok(RoundingMode::RTP),
        "RTN" => Ok(RoundingMode::RTN),
        "RTZ" => Ok(RoundingMode::RTZ),
        other => Err(PyValueError::new_err(format!(
            "Unknown rounding mode '{}'. Valid modes: RNE, RNA, RTP, RTN, RTZ",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `term_to_string` used to render *every* term via Rust's `{:?}` Debug
    /// formatter, so a String-sorted term came back as e.g.
    /// `StringLit("a\"b")` -- Rust's *own* escape syntax, not Python's and
    /// not SMT-LIB's -- instead of the actual string content. It must now
    /// return the raw string for a `"`, a `\`, a `\u`-prefixed literal
    /// substring, a non-ASCII code point, and a control character: the
    /// natural rendering for a Python `str` is the value itself, not an
    /// escaped literal.
    #[test]
    fn term_to_string_string_lit_returns_raw_content_not_debug_format() {
        let tm = PyTermManager::new();
        for raw in ["a\"b", "a\\b", "\\u0041", "caf\u{e9}", "line\u{0}break"] {
            let term_id = tm.inner.borrow_mut().mk_string_lit(raw);
            let term = PyTerm::bare(term_id);
            assert_eq!(
                tm.term_to_string(&term),
                raw,
                "expected the raw string back unchanged"
            );
        }
    }

    /// Control: plain ASCII (no special characters at all) must also come
    /// back unchanged, matching every case above.
    #[test]
    fn term_to_string_string_lit_plain_ascii_control() {
        let tm = PyTermManager::new();
        let term_id = tm.inner.borrow_mut().mk_string_lit("hello world");
        let term = PyTerm::bare(term_id);
        assert_eq!(tm.term_to_string(&term), "hello world");
    }

    /// Control: a non-string term's representation is unaffected by this
    /// fix -- it still goes through the pre-existing `{:?}` Debug fallback,
    /// confirming the new `StringLit` arm didn't accidentally swallow any
    /// other `TermKind`.
    #[test]
    fn term_to_string_non_string_term_still_uses_debug_format() {
        let tm = PyTermManager::new();
        let term_id = tm.inner.borrow_mut().mk_int(42);
        let term = PyTerm::bare(term_id);
        assert!(tm.term_to_string(&term).contains("42"));
    }

    /// `parse_sort_name` used to recurse once per `Array[` level, so a Python
    /// one-liner (`const_of_sort("x", "Array[" * 200000 + "Int" + "]" * 200000)`)
    /// overflowed the stack. A stack overflow aborts the process rather than
    /// failing a test, so *returning at all* is the assertion; the small stack
    /// makes a surviving recursion detectable.
    #[test]
    fn parse_sort_name_survives_deeply_nested_array_sorts() {
        std::thread::Builder::new()
            .name("py_deep_sort".to_string())
            .stack_size(1 << 20)
            .spawn(|| {
                let depth = 100_000;
                let mut sort_name = String::from("Int");
                for _ in 0..depth {
                    sort_name = format!("Array[Int,{sort_name}]");
                }

                let manager = PyTermManager::new();
                let mut inner = manager.inner.borrow_mut();
                let sort = parse_sort_name(&mut inner, &sort_name);
                assert!(sort.is_ok(), "deeply nested array sort should parse");
            })
            .expect("test thread should spawn")
            .join()
            .expect("test thread should not panic");
    }

    /// Semantic pins: the iterative parser accepts and rejects exactly what the
    /// recursive one did.
    #[test]
    fn parse_sort_name_results_are_unchanged() {
        let manager = PyTermManager::new();
        let mut inner = manager.inner.borrow_mut();

        let int_sort = parse_sort_name(&mut inner, "Int").expect("Int should parse");
        let nested =
            parse_sort_name(&mut inner, "Array[Int,Array[Int,Int]]").expect("nested should parse");
        let nested_again =
            parse_sort_name(&mut inner, "Array[Int, Array[Int, Int]]").expect("should parse");
        // Sorts are interned, so equal descriptions give the same id.
        assert_eq!(nested, nested_again);
        assert_ne!(int_sort, nested);

        assert!(parse_sort_name(&mut inner, "BitVec[8]").is_ok());
        assert!(parse_sort_name(&mut inner, "Float[8,24]").is_ok());
        assert!(parse_sort_name(&mut inner, "FP[8,24]").is_ok());

        // Errors still come back through PyResult, naming the offending sort.
        assert!(parse_sort_name(&mut inner, "Nope").is_err());
        assert!(parse_sort_name(&mut inner, "Array[Int]").is_err());
        assert!(parse_sort_name(&mut inner, "Array[Int,Nope]").is_err());
        assert!(parse_sort_name(&mut inner, "BitVec[x]").is_err());
    }
}
