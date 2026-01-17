//! OxiZ Python Bindings
//!
//! Provides Python bindings for the OxiZ SMT solver via PyO3.

use ::oxiz::core::ast::{TermId, TermKind, TermManager};
use ::oxiz::solver::{Solver, SolverResult};
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

/// OxiZ SMT Solver Python bindings
#[pymodule]
fn oxiz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTerm>()?;
    m.add_class::<PySolverResult>()?;
    m.add_class::<PyTermManager>()?;
    m.add_class::<PySolver>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
