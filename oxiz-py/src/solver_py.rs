//! Python wrapper for the CDCL(T) SMT Solver.

use ::oxiz::core::ast::{TermId, TermKind, TermManager};
use ::oxiz::core::smtlib::parse_term as smtlib_parse_term;
use ::oxiz::solver::{SolverConfig, SolverResult};
use num_traits::ToPrimitive;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;

use crate::results::PySolverResult;
use crate::term::{PyTerm, PyTermManager};

/// Typed value stored in the cached model.
///
/// Used by `model()` (the no-TM convenience method) so we can return
/// properly typed Python objects rather than raw strings.
#[derive(Clone, Debug)]
pub(crate) enum PyModelValue {
    Bool(bool),
    Int(i64),
    BigInt(String),
    Real(f64),
    RealRational(String),
    BitVec(u64, u32),
    Str(String),
}

/// CDCL(T) SMT Solver.
///
/// Supports incremental solving via push/pop, model extraction, and
/// unsatisfiable core computation.
///
/// The Solver is not thread-safe; use separate instances per thread.
///
/// Example::
///
///     ctx = oxiz.Context()
///     solver = oxiz.Solver()
///     x = ctx.int_const("x")
///     solver.add(x > ctx.int_val(0))
///     result = solver.check(ctx.tm)
///     if result.is_sat:
///         m = solver.model()
#[pyclass(name = "Solver", unsendable)]
pub struct PySolver {
    pub(crate) inner: RefCell<::oxiz::solver::Solver>,
    /// Cached mapping from variable name to typed Python value,
    /// populated at the end of check_sat() when the result is Sat.
    cached_model: RefCell<std::collections::HashMap<String, PyModelValue>>,
    /// Cached variable name map (TermId.raw → name), populated during check_sat().
    cached_term_names: RefCell<std::collections::HashMap<u32, String>>,
}

impl Default for PySolver {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PySolver {
    /// Create a new Solver.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(::oxiz::solver::Solver::new()),
            cached_model: RefCell::new(std::collections::HashMap::new()),
            cached_term_names: RefCell::new(std::collections::HashMap::new()),
        }
    }

    // ------------------------------------------------------------------ //
    // Assertion API                                                        //
    // ------------------------------------------------------------------ //

    /// Assert a term (add it as a hard constraint).
    ///
    /// Args:
    ///     term: A boolean Term.
    ///     tm: The TermManager that owns the term.
    fn assert_term(&self, term: &PyTerm, tm: &PyTermManager) {
        let mut solver = self.inner.borrow_mut();
        let mut manager = tm.inner.borrow_mut();
        solver.assert(term.id, &mut manager);
    }

    /// Convenience alias for assert_term that matches z3-python's `add()` API.
    fn add(&self, term: &PyTerm, tm: &PyTermManager) {
        self.assert_term(term, tm);
    }

    /// Assert an SMT-LIB2 formula string.
    fn assert_formula(&self, formula: &str, tm: &PyTermManager) -> PyResult<()> {
        let mut manager = tm.inner.borrow_mut();
        let term_id = smtlib_parse_term(formula, &mut manager)
            .map_err(|e| PyValueError::new_err(format!("Formula parse error: {e}")))?;
        let mut solver = self.inner.borrow_mut();
        solver.assert(term_id, &mut manager);
        Ok(())
    }

    /// Assert an SMT-LIB2 expression string with an optional name.
    ///
    /// Named assertions participate in unsat-core reporting when
    /// `produce-unsat-cores` is enabled via `set_option()`.
    #[pyo3(signature = (expr, tm, name = None))]
    fn assert_expr(&self, expr: &str, tm: &PyTermManager, name: Option<&str>) -> PyResult<()> {
        let mut manager = tm.inner.borrow_mut();
        let term_id = smtlib_parse_term(expr, &mut manager)
            .map_err(|e| PyValueError::new_err(format!("Formula parse error: {e}")))?;
        let mut solver = self.inner.borrow_mut();
        match name {
            Some(n) => solver.assert_named(term_id, n, &mut manager),
            None => solver.assert(term_id, &mut manager),
        }
        Ok(())
    }

    /// Assert a term and associate it with a tracking label for unsat cores.
    ///
    /// This is the direct equivalent of z3-python's `solver.assert_and_track(expr, label)`.
    /// The `produce-unsat-cores` option must be enabled via `set_option()` before
    /// calling `check_sat()`.
    ///
    /// Args:
    ///     term: A boolean Term.
    ///     label: A string label used in the unsat core.
    ///     tm: The TermManager that owns the term.
    fn assert_and_track(&self, term: &PyTerm, label: &str, tm: &PyTermManager) {
        let mut solver = self.inner.borrow_mut();
        let mut manager = tm.inner.borrow_mut();
        solver.assert_named(term.id, label, &mut manager);
    }

    // ------------------------------------------------------------------ //
    // Check                                                                //
    // ------------------------------------------------------------------ //

    /// Check satisfiability of the current assertion set.
    ///
    /// Returns SolverResult.Sat, SolverResult.Unsat, or SolverResult.Unknown.
    fn check_sat(&self, tm: &PyTermManager) -> PySolverResult {
        self.run_check(tm)
    }

    /// Alias for check_sat matching z3-python's `check()` name.
    fn check(&self, tm: &PyTermManager) -> PySolverResult {
        self.run_check(tm)
    }

    // ------------------------------------------------------------------ //
    // Push / pop                                                           //
    // ------------------------------------------------------------------ //

    /// Push a new assertion scope.
    fn push(&self) {
        let mut solver = self.inner.borrow_mut();
        solver.push();
    }

    /// Pop one or more assertion scopes.
    ///
    /// Args:
    ///     n: Number of scopes to pop (default 1).
    #[pyo3(signature = (n = 1))]
    fn pop(&self, n: usize) {
        let mut solver = self.inner.borrow_mut();
        for _ in 0..n {
            solver.pop();
        }
    }

    /// Reset the solver, removing all assertions and learned clauses.
    fn reset(&self) {
        let mut solver = self.inner.borrow_mut();
        solver.reset();
        self.cached_model.borrow_mut().clear();
        self.cached_term_names.borrow_mut().clear();
    }

    // ------------------------------------------------------------------ //
    // Model                                                                //
    // ------------------------------------------------------------------ //

    /// Get the satisfying model as a string-valued dictionary.
    ///
    /// Only meaningful after check_sat() returns SolverResult.Sat.
    fn get_model<'py>(&self, py: Python<'py>, tm: &PyTermManager) -> PyResult<Bound<'py, PyDict>> {
        let solver = self.inner.borrow();
        let manager = tm.inner.borrow();

        let dict = PyDict::new(py);

        if let Some(model) = solver.model() {
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
        }

        Ok(dict)
    }

    /// Get the satisfying model as a typed Python dictionary (no TermManager needed).
    ///
    /// Uses a snapshot cached during the last `check_sat()` call.
    /// Only meaningful after check_sat() returns SolverResult.Sat.
    ///
    /// Returns a dict mapping variable names to typed Python values:
    ///   - Booleans → `bool`
    ///   - Integers that fit in i64 → `int`
    ///   - Large integers → `str` (decimal)
    ///   - Whole-number rationals → `int`
    ///   - Non-whole rationals → `str` ("numer/denom")
    ///   - Bitvectors → `int` (unsigned, ≤ 64 bits)
    ///   - Other terms → `str`
    fn model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let cache = self.cached_model.borrow();
        let dict = PyDict::new(py);

        for (name, value) in cache.iter() {
            match value {
                PyModelValue::Bool(b) => dict.set_item(name.as_str(), *b)?,
                PyModelValue::Int(i) => dict.set_item(name.as_str(), *i)?,
                PyModelValue::BigInt(s) => dict.set_item(name.as_str(), s.as_str())?,
                PyModelValue::Real(f) => dict.set_item(name.as_str(), *f)?,
                PyModelValue::RealRational(s) => dict.set_item(name.as_str(), s.as_str())?,
                PyModelValue::BitVec(v, _width) => dict.set_item(name.as_str(), *v)?,
                PyModelValue::Str(s) => dict.set_item(name.as_str(), s.as_str())?,
            }
        }

        Ok(dict)
    }

    // ------------------------------------------------------------------ //
    // Unsat core                                                           //
    // ------------------------------------------------------------------ //

    /// Get the unsatisfiable core as a list of assertion label strings.
    ///
    /// Only populated after check_sat() returns Unsat AND produce-unsat-cores
    /// was enabled via set_option() before the check.
    fn get_unsat_core(&self) -> Vec<String> {
        let solver = self.inner.borrow();
        match solver.get_unsat_core() {
            Some(core) => core.names.clone(),
            None => Vec::new(),
        }
    }

    /// z3-python-style alias for get_unsat_core().
    fn unsat_core(&self) -> Vec<String> {
        self.get_unsat_core()
    }

    // ------------------------------------------------------------------ //
    // Configuration                                                        //
    // ------------------------------------------------------------------ //

    /// Set the SMT-LIB2 logic (e.g., "QF_LIA", "QF_LRA", "QF_UF").
    fn set_logic(&self, logic: &str) {
        let mut solver = self.inner.borrow_mut();
        solver.set_logic(logic);
    }

    /// Set a solver option by key and value string.
    ///
    /// Supported keys:
    ///   - `produce-unsat-cores` / `produce_unsat_cores`: `"true"` / `"false"`
    ///   - `logic`: SMT-LIB2 logic name (same as set_logic)
    ///   - `timeout`: timeout in milliseconds (integer string)
    fn set_option(&self, key: &str, value: &str) -> PyResult<()> {
        let mut solver = self.inner.borrow_mut();
        match key {
            "produce-unsat-cores" | "produce_unsat_cores" => {
                let flag = parse_bool_value(key, value)?;
                solver.set_produce_unsat_cores(flag);
                Ok(())
            }
            "logic" => {
                solver.set_logic(value);
                Ok(())
            }
            "timeout" => {
                let ms: u64 = value.parse().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Invalid timeout value '{}': expected an integer number of milliseconds",
                        value
                    ))
                })?;
                let new_config = SolverConfig::default().with_timeout(ms);
                solver.set_config(new_config);
                Ok(())
            }
            other => Err(PyValueError::new_err(format!(
                "Unknown solver option: '{}'. Supported: produce-unsat-cores, logic, timeout.",
                other
            ))),
        }
    }

    /// Set a timeout in milliseconds.
    ///
    /// When the timeout expires the solver returns SolverResult.Unknown.
    /// Pass 0 to disable the timeout.
    fn set_timeout(&self, milliseconds: u64) {
        let mut solver = self.inner.borrow_mut();
        let new_config = SolverConfig::default().with_timeout(milliseconds);
        solver.set_config(new_config);
    }

    // ------------------------------------------------------------------ //
    // Properties                                                           //
    // ------------------------------------------------------------------ //

    /// Number of assertions currently in the solver.
    #[getter]
    fn num_assertions(&self) -> usize {
        let solver = self.inner.borrow();
        solver.num_assertions()
    }

    /// Current push/pop context depth.
    #[getter]
    fn context_level(&self) -> usize {
        let solver = self.inner.borrow();
        solver.context_level()
    }
}

impl PySolver {
    /// Internal helper: run check(), rebuild caches, return result.
    fn run_check(&self, tm: &PyTermManager) -> PySolverResult {
        let mut solver = self.inner.borrow_mut();
        let mut manager = tm.inner.borrow_mut();
        let result = solver.check(&mut manager);

        // Rebuild caches from TM state so model() can be called without TM.
        {
            let mut name_cache = self.cached_term_names.borrow_mut();
            name_cache.clear();
            let mut model_cache = self.cached_model.borrow_mut();
            model_cache.clear();

            // Build term-name map (TermId.raw → variable name)
            let num_terms = manager.len();
            for raw in 0..(num_terms as u32) {
                let tid = TermId::new(raw);
                if let Some(term) = manager.get(tid) {
                    if let TermKind::Var(spur) = &term.kind {
                        name_cache.insert(raw, manager.resolve_str(*spur).to_string());
                    }
                }
            }

            // Build typed model cache if Sat
            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    for (&var_id, &value_id) in model.assignments() {
                        if let Some(var_name) = name_cache.get(&var_id.raw()) {
                            let typed_value = build_model_value(value_id, &manager);
                            model_cache.insert(var_name.clone(), typed_value);
                        }
                    }
                }
            }
        }

        result.into()
    }
}

/// Convert a TermId in the manager to a typed PyModelValue.
fn build_model_value(value_id: TermId, manager: &TermManager) -> PyModelValue {
    if let Some(value_term) = manager.get(value_id) {
        match &value_term.kind {
            TermKind::True => PyModelValue::Bool(true),
            TermKind::False => PyModelValue::Bool(false),
            TermKind::IntConst(n) => {
                if let Some(small) = n.to_i64() {
                    PyModelValue::Int(small)
                } else {
                    PyModelValue::BigInt(n.to_string())
                }
            }
            TermKind::RealConst(r) => {
                if *r.denom() == 1 {
                    // numer() returns &i64 for Rational64
                    PyModelValue::Int(*r.numer())
                } else {
                    PyModelValue::RealRational(format!("{}/{}", r.numer(), r.denom()))
                }
            }
            TermKind::BitVecConst { value, .. } => {
                let lo = value.iter_u64_digits().next().unwrap_or(0);
                PyModelValue::BitVec(lo, 0)
            }
            other => PyModelValue::Str(format!("{:?}", other)),
        }
    } else {
        PyModelValue::Str(format!("Term({})", value_id.raw()))
    }
}

fn parse_bool_value(key: &str, value: &str) -> PyResult<bool> {
    match value {
        "true" | "True" | "1" => Ok(true),
        "false" | "False" | "0" => Ok(false),
        other => Err(PyValueError::new_err(format!(
            "Invalid boolean value for option '{}': '{}'. Use 'true' or 'false'.",
            key, other
        ))),
    }
}
