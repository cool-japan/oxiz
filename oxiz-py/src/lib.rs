//! OxiZ Python Bindings
//!
//! Provides Python bindings for the OxiZ SMT solver via PyO3/maturin.
//!
//! # Quick start
//!
//! ```python
//! import oxiz
//!
//! ctx = oxiz.Context()
//! solver = oxiz.Solver()
//!
//! x = ctx.int_const("x")
//! y = ctx.int_const("y")
//!
//! solver.add(x + y > ctx.int_val(0), ctx.tm)
//! solver.add(x < ctx.int_val(10), ctx.tm)
//!
//! result = solver.check(ctx.tm)
//! if result.is_sat:
//!     print(solver.model())
//! ```

pub mod builtins;
pub mod context;
pub mod optimizer;
pub mod results;
pub mod solver_py;
pub mod term;

use pyo3::prelude::*;

/// OxiZ SMT Solver Python module
#[pymodule]
fn oxiz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core term types
    m.add_class::<term::PyTerm>()?;
    m.add_class::<term::PyTermManager>()?;

    // Solver result enumerations
    m.add_class::<results::PySolverResult>()?;
    m.add_class::<results::PyOptimizationResult>()?;

    // High-level context (z3-python parity API)
    m.add_class::<context::PyContext>()?;

    // Solver and optimizer
    m.add_class::<solver_py::PySolver>()?;
    m.add_class::<optimizer::PyOptimizer>()?;

    // Module-level boolean / arithmetic combinators
    m.add_function(wrap_pyfunction!(builtins::And, m)?)?;
    m.add_function(wrap_pyfunction!(builtins::Or, m)?)?;
    m.add_function(wrap_pyfunction!(builtins::Not, m)?)?;
    m.add_function(wrap_pyfunction!(builtins::Implies, m)?)?;
    m.add_function(wrap_pyfunction!(builtins::If, m)?)?;

    // Explicit-TM variants (for users who work with bare TermManager)
    m.add_function(wrap_pyfunction!(builtins::and_tm, m)?)?;
    m.add_function(wrap_pyfunction!(builtins::or_tm, m)?)?;
    m.add_function(wrap_pyfunction!(builtins::not_tm, m)?)?;

    // Version metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
