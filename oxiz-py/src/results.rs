//! Python wrapper types for solver results.

use ::oxiz::solver::{OptimizationResult, SolverResult};

/// Python wrapper for SolverResult
#[pyo3::pyclass(name = "SolverResult", eq, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySolverResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout, incomplete, etc.)
    Unknown,
}

#[pyo3::pymethods]
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

/// Python wrapper for OptimizationResult
#[pyo3::pyclass(name = "OptimizationResult", eq, from_py_object)]
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

#[pyo3::pymethods]
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

impl From<&OptimizationResult> for PyOptimizationResult {
    fn from(result: &OptimizationResult) -> Self {
        match result {
            OptimizationResult::Optimal { .. } => PyOptimizationResult::Optimal,
            OptimizationResult::Unbounded => PyOptimizationResult::Unbounded,
            OptimizationResult::Unsat => PyOptimizationResult::Unsat,
            OptimizationResult::Unknown => PyOptimizationResult::Unknown,
        }
    }
}
