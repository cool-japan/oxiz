//! OxiZ Python Bindings
//!
//! Provides Python bindings for the OxiZ SMT solver via PyO3.

use ::oxiz::core::ast::{TermId, TermKind, TermManager};
use ::oxiz::solver::{Model, OptimizationResult, Optimizer, Solver, SolverResult};
use num_bigint::BigInt;
use num_rational::Rational64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;

/// Python wrapper for TermId
#[pyclass(name = "Term")]
#[derive(Clone)]
pub struct PyTerm {
    id: TermId,
}

#[pymethods]
impl PyTerm {
    /// Get the raw term ID
    #[getter]
    fn id(&self) -> u32 {
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
}

impl From<TermId> for PyTerm {
    fn from(id: TermId) -> Self {
        Self { id }
    }
}

impl From<PyTerm> for TermId {
    fn from(term: PyTerm) -> Self {
        term.id
    }
}

/// Python wrapper for SolverResult
#[pyclass(name = "SolverResult", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySolverResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout, incomplete, etc.)
    Unknown,
}

#[pymethods]
impl PySolverResult {
    fn __repr__(&self) -> &'static str {
        match self {
            PySolverResult::Sat => "SolverResult.Sat",
            PySolverResult::Unsat => "SolverResult.Unsat",
            PySolverResult::Unknown => "SolverResult.Unknown",
        }
    }

    fn __str__(&self) -> &'static str {
        match self {
            PySolverResult::Sat => "sat",
            PySolverResult::Unsat => "unsat",
            PySolverResult::Unknown => "unknown",
        }
    }

    /// Check if the result is satisfiable
    #[getter]
    fn is_sat(&self) -> bool {
        matches!(self, PySolverResult::Sat)
    }

    /// Check if the result is unsatisfiable
    #[getter]
    fn is_unsat(&self) -> bool {
        matches!(self, PySolverResult::Unsat)
    }

    /// Check if the result is unknown
    #[getter]
    fn is_unknown(&self) -> bool {
        matches!(self, PySolverResult::Unknown)
    }
}

impl From<SolverResult> for PySolverResult {
    fn from(result: SolverResult) -> Self {
        match result {
            SolverResult::Sat => PySolverResult::Sat,
            SolverResult::Unsat => PySolverResult::Unsat,
            SolverResult::Unknown => PySolverResult::Unknown,
        }
    }
}

/// Python wrapper for TermManager
///
/// Note: This class is not thread-safe (unsendable) because it uses RefCell
/// internally for interior mutability.
#[pyclass(name = "TermManager", unsendable)]
pub struct PyTermManager {
    inner: RefCell<TermManager>,
}

#[pymethods]
impl PyTermManager {
    /// Create a new TermManager
    #[new]
    fn new() -> Self {
        Self {
            inner: RefCell::new(TermManager::new()),
        }
    }

    /// Create a variable with a given name and sort
    ///
    /// Args:
    ///     name: Variable name
    ///     sort_name: Sort name ("Bool", "Int", "Real", or "BitVec[width]")
    ///
    /// Returns:
    ///     A new Term representing the variable
    fn mk_var(&self, name: &str, sort_name: &str) -> PyResult<PyTerm> {
        let mut tm = self.inner.borrow_mut();
        let sort = self.parse_sort_name(&mut tm, sort_name)?;
        let term_id = tm.mk_var(name, sort);
        Ok(PyTerm::from(term_id))
    }

    /// Create a boolean constant
    ///
    /// Args:
    ///     value: Boolean value
    ///
    /// Returns:
    ///     A Term representing the boolean constant
    fn mk_bool(&self, value: bool) -> PyTerm {
        let tm = self.inner.borrow();
        PyTerm::from(tm.mk_bool(value))
    }

    /// Create an integer constant
    ///
    /// Args:
    ///     value: Integer value
    ///
    /// Returns:
    ///     A Term representing the integer constant
    fn mk_int(&self, value: i64) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_int(BigInt::from(value)))
    }

    /// Create a real (rational) constant
    ///
    /// Args:
    ///     numerator: Numerator
    ///     denominator: Denominator (must be non-zero)
    ///
    /// Returns:
    ///     A Term representing the rational constant
    fn mk_real(&self, numerator: i64, denominator: i64) -> PyResult<PyTerm> {
        if denominator == 0 {
            return Err(PyValueError::new_err("Denominator cannot be zero"));
        }
        let mut tm = self.inner.borrow_mut();
        let rational = Rational64::new(numerator, denominator);
        Ok(PyTerm::from(tm.mk_real(rational)))
    }

    /// Create a logical NOT
    ///
    /// Args:
    ///     term: Term to negate
    ///
    /// Returns:
    ///     A Term representing NOT term
    fn mk_not(&self, term: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_not(term.id))
    }

    /// Create a logical AND of multiple terms
    ///
    /// Args:
    ///     terms: List of terms to AND together
    ///
    /// Returns:
    ///     A Term representing the conjunction
    fn mk_and(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let term_ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::from(tm.mk_and(term_ids))
    }

    /// Create a logical OR of multiple terms
    ///
    /// Args:
    ///     terms: List of terms to OR together
    ///
    /// Returns:
    ///     A Term representing the disjunction
    fn mk_or(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let term_ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::from(tm.mk_or(term_ids))
    }

    /// Create a logical implication (lhs => rhs)
    ///
    /// Args:
    ///     lhs: Left-hand side (antecedent)
    ///     rhs: Right-hand side (consequent)
    ///
    /// Returns:
    ///     A Term representing lhs => rhs
    fn mk_implies(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_implies(lhs.id, rhs.id))
    }

    /// Create an equality (lhs = rhs)
    ///
    /// Args:
    ///     lhs: Left-hand side
    ///     rhs: Right-hand side
    ///
    /// Returns:
    ///     A Term representing lhs = rhs
    fn mk_eq(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_eq(lhs.id, rhs.id))
    }

    /// Create an addition of multiple terms
    ///
    /// Args:
    ///     terms: List of terms to add together
    ///
    /// Returns:
    ///     A Term representing the sum
    fn mk_add(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let term_ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::from(tm.mk_add(term_ids))
    }

    /// Create a subtraction (lhs - rhs)
    ///
    /// Args:
    ///     lhs: Left-hand side
    ///     rhs: Right-hand side
    ///
    /// Returns:
    ///     A Term representing lhs - rhs
    fn mk_sub(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_sub(lhs.id, rhs.id))
    }

    /// Create a multiplication of multiple terms
    ///
    /// Args:
    ///     terms: List of terms to multiply together
    ///
    /// Returns:
    ///     A Term representing the product
    fn mk_mul(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let term_ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::from(tm.mk_mul(term_ids))
    }

    /// Create a less-than comparison (lhs < rhs)
    ///
    /// Args:
    ///     lhs: Left-hand side
    ///     rhs: Right-hand side
    ///
    /// Returns:
    ///     A Term representing lhs < rhs
    fn mk_lt(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_lt(lhs.id, rhs.id))
    }

    /// Create a less-than-or-equal comparison (lhs <= rhs)
    ///
    /// Args:
    ///     lhs: Left-hand side
    ///     rhs: Right-hand side
    ///
    /// Returns:
    ///     A Term representing lhs <= rhs
    fn mk_le(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_le(lhs.id, rhs.id))
    }

    /// Create a greater-than comparison (lhs > rhs)
    ///
    /// Args:
    ///     lhs: Left-hand side
    ///     rhs: Right-hand side
    ///
    /// Returns:
    ///     A Term representing lhs > rhs
    fn mk_gt(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_gt(lhs.id, rhs.id))
    }

    /// Create a greater-than-or-equal comparison (lhs >= rhs)
    ///
    /// Args:
    ///     lhs: Left-hand side
    ///     rhs: Right-hand side
    ///
    /// Returns:
    ///     A Term representing lhs >= rhs
    fn mk_ge(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_ge(lhs.id, rhs.id))
    }

    /// Create an if-then-else expression
    ///
    /// Args:
    ///     cond: Condition (boolean)
    ///     then_branch: Value if condition is true
    ///     else_branch: Value if condition is false
    ///
    /// Returns:
    ///     A Term representing if cond then then_branch else else_branch
    fn mk_ite(&self, cond: &PyTerm, then_branch: &PyTerm, else_branch: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_ite(cond.id, then_branch.id, else_branch.id))
    }

    /// Create a distinct constraint (all arguments are pairwise distinct)
    ///
    /// Args:
    ///     terms: List of terms that must all be different
    ///
    /// Returns:
    ///     A Term representing the distinct constraint
    fn mk_distinct(&self, terms: Vec<PyTerm>) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        let term_ids: Vec<TermId> = terms.into_iter().map(|t| t.id).collect();
        PyTerm::from(tm.mk_distinct(term_ids))
    }

    /// Create an arithmetic negation (-term)
    ///
    /// Args:
    ///     term: Term to negate
    ///
    /// Returns:
    ///     A Term representing -term
    fn mk_neg(&self, term: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_neg(term.id))
    }

    /// Create an integer division (lhs / rhs)
    ///
    /// Args:
    ///     lhs: Dividend
    ///     rhs: Divisor
    ///
    /// Returns:
    ///     A Term representing integer division
    fn mk_div(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_div(lhs.id, rhs.id))
    }

    /// Create a modulo operation (lhs % rhs)
    ///
    /// Args:
    ///     lhs: Dividend
    ///     rhs: Divisor
    ///
    /// Returns:
    ///     A Term representing lhs mod rhs
    fn mk_mod(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_mod(lhs.id, rhs.id))
    }

    /// Create a logical XOR (exclusive or)
    ///
    /// Args:
    ///     lhs: Left-hand side
    ///     rhs: Right-hand side
    ///
    /// Returns:
    ///     A Term representing lhs XOR rhs
    fn mk_xor(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_xor(lhs.id, rhs.id))
    }

    // ============ BitVec Operations ============

    /// Create a bitvector constant
    ///
    /// Args:
    ///     value: Integer value
    ///     width: Bit width
    ///
    /// Returns:
    ///     A Term representing the bitvector constant
    fn mk_bv(&self, value: i64, width: u32) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bitvec(BigInt::from(value), width))
    }

    /// Concatenate two bitvectors
    ///
    /// Args:
    ///     lhs: High-order bits
    ///     rhs: Low-order bits
    ///
    /// Returns:
    ///     A Term representing the concatenation
    fn mk_bv_concat(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_concat(lhs.id, rhs.id))
    }

    /// Extract bits from a bitvector
    ///
    /// Args:
    ///     high: High bit index (inclusive)
    ///     low: Low bit index (inclusive)
    ///     arg: Bitvector term
    ///
    /// Returns:
    ///     A Term representing bits[high:low]
    fn mk_bv_extract(&self, high: u32, low: u32, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_extract(high, low, arg.id))
    }

    /// Bitwise NOT
    ///
    /// Args:
    ///     arg: Bitvector term
    ///
    /// Returns:
    ///     A Term representing ~arg
    fn mk_bv_not(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_not(arg.id))
    }

    /// Bitwise AND
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A Term representing lhs & rhs
    fn mk_bv_and(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_and(lhs.id, rhs.id))
    }

    /// Bitwise OR
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A Term representing lhs | rhs
    fn mk_bv_or(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_or(lhs.id, rhs.id))
    }

    /// Bitvector addition
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A Term representing lhs + rhs (bitvector arithmetic)
    fn mk_bv_add(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_add(lhs.id, rhs.id))
    }

    /// Bitvector subtraction
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A Term representing lhs - rhs (bitvector arithmetic)
    fn mk_bv_sub(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_sub(lhs.id, rhs.id))
    }

    /// Bitvector multiplication
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A Term representing lhs * rhs (bitvector arithmetic)
    fn mk_bv_mul(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_mul(lhs.id, rhs.id))
    }

    /// Bitvector negation
    ///
    /// Args:
    ///     arg: Bitvector term
    ///
    /// Returns:
    ///     A Term representing -arg (two's complement)
    fn mk_bv_neg(&self, arg: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_neg(arg.id))
    }

    /// Unsigned less than
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A boolean Term representing lhs <u rhs
    fn mk_bv_ult(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_ult(lhs.id, rhs.id))
    }

    /// Signed less than
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A boolean Term representing lhs <s rhs
    fn mk_bv_slt(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_slt(lhs.id, rhs.id))
    }

    /// Unsigned less than or equal
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A boolean Term representing lhs <=u rhs
    fn mk_bv_ule(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_ule(lhs.id, rhs.id))
    }

    /// Signed less than or equal
    ///
    /// Args:
    ///     lhs: Left operand
    ///     rhs: Right operand
    ///
    /// Returns:
    ///     A boolean Term representing lhs <=s rhs
    fn mk_bv_sle(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_sle(lhs.id, rhs.id))
    }

    /// Unsigned division
    ///
    /// Args:
    ///     lhs: Dividend
    ///     rhs: Divisor
    ///
    /// Returns:
    ///     A Term representing lhs /u rhs
    fn mk_bv_udiv(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_udiv(lhs.id, rhs.id))
    }

    /// Signed division
    ///
    /// Args:
    ///     lhs: Dividend
    ///     rhs: Divisor
    ///
    /// Returns:
    ///     A Term representing lhs /s rhs
    fn mk_bv_sdiv(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_sdiv(lhs.id, rhs.id))
    }

    /// Unsigned remainder
    ///
    /// Args:
    ///     lhs: Dividend
    ///     rhs: Divisor
    ///
    /// Returns:
    ///     A Term representing lhs %u rhs
    fn mk_bv_urem(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_urem(lhs.id, rhs.id))
    }

    /// Signed remainder
    ///
    /// Args:
    ///     lhs: Dividend
    ///     rhs: Divisor
    ///
    /// Returns:
    ///     A Term representing lhs %s rhs
    fn mk_bv_srem(&self, lhs: &PyTerm, rhs: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_bv_srem(lhs.id, rhs.id))
    }

    // ============ Array Operations ============

    /// Array select operation
    ///
    /// Args:
    ///     array: Array term
    ///     index: Index term
    ///
    /// Returns:
    ///     A Term representing array[index]
    fn mk_select(&self, array: &PyTerm, index: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_select(array.id, index.id))
    }

    /// Array store operation
    ///
    /// Args:
    ///     array: Array term
    ///     index: Index term
    ///     value: Value to store
    ///
    /// Returns:
    ///     A Term representing store(array, index, value)
    fn mk_store(&self, array: &PyTerm, index: &PyTerm, value: &PyTerm) -> PyTerm {
        let mut tm = self.inner.borrow_mut();
        PyTerm::from(tm.mk_store(array.id, index.id, value.id))
    }

    /// Get the string representation of a term
    ///
    /// Args:
    ///     term: Term to convert to string
    ///
    /// Returns:
    ///     String representation of the term
    fn term_to_string(&self, term: &PyTerm) -> String {
        let tm = self.inner.borrow();
        if let Some(t) = tm.get(term.id) {
            format!("{:?}", t.kind)
        } else {
            format!("Term({})", term.id.raw())
        }
    }
}

impl PyTermManager {
    fn parse_sort_name(
        &self,
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
                "Unknown sort: {}. Use 'Bool', 'Int', 'Real', or 'BitVec[N]'",
                sort_name
            ))),
        }
    }
}

/// Python wrapper for Solver
///
/// Note: This class is not thread-safe (unsendable) because it uses RefCell
/// internally for interior mutability.
#[pyclass(name = "Solver", unsendable)]
pub struct PySolver {
    inner: RefCell<Solver>,
}

#[pymethods]
impl PySolver {
    /// Create a new Solver
    #[new]
    fn new() -> Self {
        Self {
            inner: RefCell::new(Solver::new()),
        }
    }

    /// Assert a term (add it as a constraint)
    ///
    /// Args:
    ///     term: Term to assert (must be boolean)
    ///     tm: TermManager that owns the term
    fn assert_term(&self, term: &PyTerm, tm: &PyTermManager) {
        let mut solver = self.inner.borrow_mut();
        let mut manager = tm.inner.borrow_mut();
        solver.assert(term.id, &mut manager);
    }

    /// Check satisfiability
    ///
    /// Args:
    ///     tm: TermManager
    ///
    /// Returns:
    ///     SolverResult indicating sat, unsat, or unknown
    fn check_sat(&self, tm: &PyTermManager) -> PySolverResult {
        let mut solver = self.inner.borrow_mut();
        let mut manager = tm.inner.borrow_mut();
        solver.check(&mut manager).into()
    }

    /// Push a new assertion scope
    fn push(&self) {
        let mut solver = self.inner.borrow_mut();
        solver.push();
    }

    /// Pop an assertion scope
    fn pop(&self) {
        let mut solver = self.inner.borrow_mut();
        solver.pop();
    }

    /// Reset the solver (remove all assertions)
    fn reset(&self) {
        let mut solver = self.inner.borrow_mut();
        solver.reset();
    }

    /// Get the model as a dictionary
    ///
    /// Args:
    ///     tm: TermManager
    ///
    /// Returns:
    ///     Dictionary mapping variable names to their values (as strings)
    ///     Returns an empty dict if no model is available
    fn get_model<'py>(&self, py: Python<'py>, tm: &PyTermManager) -> PyResult<Bound<'py, PyDict>> {
        let solver = self.inner.borrow();
        let manager = tm.inner.borrow();

        let dict = PyDict::new(py);

        if let Some(model) = solver.model() {
            for (&var_id, &value_id) in model.assignments() {
                // Get the variable name
                if let Some(var_term) = manager.get(var_id)
                    && let TermKind::Var(spur) = &var_term.kind
                {
                    let var_name = manager.resolve_str(*spur);

                    // Get the value as a string
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
                            TermKind::BitVecConst { value, width } => {
                                format!("#x{:0width$x}", value, width = (*width as usize) / 4)
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

        Ok(dict)
    }

    /// Get the number of assertions
    #[getter]
    fn num_assertions(&self) -> usize {
        let solver = self.inner.borrow();
        solver.num_assertions()
    }

    /// Get the current context level (push/pop depth)
    #[getter]
    fn context_level(&self) -> usize {
        let solver = self.inner.borrow();
        solver.context_level()
    }

    /// Set the logic
    ///
    /// Args:
    ///     logic: SMT-LIB2 logic name (e.g., "QF_LIA", "QF_LRA", "QF_UF")
    fn set_logic(&self, logic: &str) {
        let mut solver = self.inner.borrow_mut();
        solver.set_logic(logic);
    }
}

/// Python wrapper for OptimizationResult
#[pyclass(name = "OptimizationResult", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyOptimizationResult {
    /// Optimal solution found
    Optimal,
    /// Unbounded (no finite optimum)
    Unbounded,
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout, incomplete, etc.)
    Unknown,
}

#[pymethods]
impl PyOptimizationResult {
    fn __repr__(&self) -> &'static str {
        match self {
            PyOptimizationResult::Optimal => "OptimizationResult.Optimal",
            PyOptimizationResult::Unbounded => "OptimizationResult.Unbounded",
            PyOptimizationResult::Unsat => "OptimizationResult.Unsat",
            PyOptimizationResult::Unknown => "OptimizationResult.Unknown",
        }
    }

    fn __str__(&self) -> &'static str {
        match self {
            PyOptimizationResult::Optimal => "optimal",
            PyOptimizationResult::Unbounded => "unbounded",
            PyOptimizationResult::Unsat => "unsat",
            PyOptimizationResult::Unknown => "unknown",
        }
    }

    /// Check if the result is optimal
    #[getter]
    fn is_optimal(&self) -> bool {
        matches!(self, PyOptimizationResult::Optimal)
    }

    /// Check if the result is unbounded
    #[getter]
    fn is_unbounded(&self) -> bool {
        matches!(self, PyOptimizationResult::Unbounded)
    }

    /// Check if the result is unsatisfiable
    #[getter]
    fn is_unsat(&self) -> bool {
        matches!(self, PyOptimizationResult::Unsat)
    }
}

/// Python wrapper for Optimizer
///
/// Note: This class is not thread-safe (unsendable) because it uses RefCell
/// internally for interior mutability.
#[pyclass(name = "Optimizer", unsendable)]
pub struct PyOptimizer {
    inner: RefCell<Optimizer>,
    last_model: RefCell<Option<Model>>,
}

#[pymethods]
impl PyOptimizer {
    /// Create a new Optimizer
    #[new]
    fn new() -> Self {
        Self {
            inner: RefCell::new(Optimizer::new()),
            last_model: RefCell::new(None),
        }
    }

    /// Assert a term (add it as a constraint)
    ///
    /// Args:
    ///     term: Term to assert (must be boolean)
    fn assert_term(&self, term: &PyTerm) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.assert(term.id);
    }

    /// Add a term to minimize
    ///
    /// Args:
    ///     term: Term to minimize (must be Int or Real)
    fn minimize(&self, term: &PyTerm) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.minimize(term.id);
    }

    /// Add a term to maximize
    ///
    /// Args:
    ///     term: Term to maximize (must be Int or Real)
    fn maximize(&self, term: &PyTerm) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.maximize(term.id);
    }

    /// Set the logic
    ///
    /// Args:
    ///     logic: SMT-LIB2 logic name (e.g., "QF_LIA", "QF_LRA")
    fn set_logic(&self, logic: &str) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.set_logic(logic);
    }

    /// Push a new assertion scope
    fn push(&self) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.push();
    }

    /// Pop an assertion scope
    fn pop(&self) {
        let mut optimizer = self.inner.borrow_mut();
        optimizer.pop();
    }

    /// Optimize and find optimal solution
    ///
    /// Args:
    ///     tm: TermManager
    ///
    /// Returns:
    ///     OptimizationResult indicating optimal, unbounded, unsat, or unknown
    fn optimize(&self, tm: &PyTermManager) -> PyOptimizationResult {
        let mut optimizer = self.inner.borrow_mut();
        let mut manager = tm.inner.borrow_mut();
        let result = optimizer.optimize(&mut manager);

        // Store the model if optimal
        match &result {
            OptimizationResult::Optimal { model, .. } => {
                *self.last_model.borrow_mut() = Some(model.clone());
                PyOptimizationResult::Optimal
            }
            OptimizationResult::Unbounded => {
                PyOptimizationResult::Unbounded
            }
            OptimizationResult::Unsat => {
                PyOptimizationResult::Unsat
            }
            OptimizationResult::Unknown => {
                PyOptimizationResult::Unknown
            }
        }
    }

    /// Get the model as a dictionary (from last optimize call)
    ///
    /// Args:
    ///     tm: TermManager
    ///
    /// Returns:
    ///     Dictionary mapping variable names to their values (as strings)
    fn get_model<'py>(&self, py: Python<'py>, tm: &PyTermManager) -> PyResult<Bound<'py, PyDict>> {
        let manager = tm.inner.borrow();
        let dict = PyDict::new(py);

        if let Some(model) = self.last_model.borrow().as_ref() {
            for (&var_id, &value_id) in model.assignments() {
                if let Some(var_term) = manager.get(var_id)
                    && let TermKind::Var(spur) = &var_term.kind
                {
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

        Ok(dict)
    }
}

/// OxiZ SMT Solver Python bindings
#[pymodule]
fn oxiz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTerm>()?;
    m.add_class::<PySolverResult>()?;
    m.add_class::<PyOptimizationResult>()?;
    m.add_class::<PyTermManager>()?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyOptimizer>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
