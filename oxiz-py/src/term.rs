//! Python wrapper for Term, TermManager, and expression builder operators.

use ::oxiz::core::ast::{RoundingMode, TermId, TermManager};
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

    /// Create a string literal term.
    fn mk_string_lit(&self, value: &str) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::bare(tm.mk_string_lit(value))
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
            format!("{:?}", t.kind)
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
pub(crate) fn parse_sort_name(
    tm: &mut TermManager,
    sort_name: &str,
) -> PyResult<::oxiz::core::SortId> {
    match sort_name {
        "Bool" => Ok(tm.sorts.bool_sort),
        "Int" => Ok(tm.sorts.int_sort),
        "Real" => Ok(tm.sorts.real_sort),
        "String" => Ok(tm.sorts.string_sort()),
        s if s.starts_with("BitVec[") && s.ends_with(']') => {
            let width_str = &s[7..s.len() - 1];
            let width: u32 = width_str.parse().map_err(|_| {
                PyValueError::new_err(format!("Invalid BitVec width: {}", width_str))
            })?;
            Ok(tm.sorts.bitvec(width))
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
                    sort_name
                ))
            })?;
            let eb: u32 = inner[..comma].trim().parse().map_err(|_| {
                PyValueError::new_err(format!("Invalid exponent width in sort '{}'", sort_name))
            })?;
            let sb: u32 = inner[comma + 1..].trim().parse().map_err(|_| {
                PyValueError::new_err(format!("Invalid significand width in sort '{}'", sort_name))
            })?;
            Ok(tm.sorts.float_sort(eb, sb))
        }
        s if s.starts_with("Array[") && s.ends_with(']') => {
            // Find the comma separating domain and range, respecting nested brackets.
            let inner = &s[6..s.len() - 1];
            let split = find_top_level_comma(inner).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Invalid Array sort '{}': expected 'Array[D,R]'",
                    sort_name
                ))
            })?;
            let domain_str = inner[..split].trim();
            let range_str = inner[split + 1..].trim();
            let domain = parse_sort_name(tm, domain_str)?;
            let range = parse_sort_name(tm, range_str)?;
            Ok(tm.sorts.array(domain, range))
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown sort: '{}'. \
             Supported: 'Bool', 'Int', 'Real', 'String', 'BitVec[N]', \
             'Float[eb,sb]', 'FP[eb,sb]', 'Array[D,R]'",
            sort_name
        ))),
    }
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
fn parse_rounding_mode(rm: &str) -> PyResult<RoundingMode> {
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
