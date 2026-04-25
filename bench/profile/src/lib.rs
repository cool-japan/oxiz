//! Small profiling workloads for hot-path benchmarks.

use oxiz_core::OxizError;
use oxiz_solver::Context;

/// Run a small SMT-LIB script through the high-level solver context.
pub fn run_script(script: &str) -> Result<Vec<String>, OxizError> {
    let mut ctx = Context::new();
    ctx.execute_script(script)
}

/// Small SAT-heavy formula that drives parser + Boolean propagation.
#[must_use]
pub fn sat_propagation_script() -> &'static str {
    r#"
        (set-logic QF_UF)
        (declare-const p Bool)
        (declare-const q Bool)
        (declare-const r Bool)
        (assert (or p q))
        (assert (or (not p) r))
        (assert (or (not q) r))
        (assert (not r))
        (check-sat)
    "#
}

/// Small arithmetic script for context-driven theory checking.
#[must_use]
pub fn theory_check_script() -> &'static str {
    r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (declare-const y Int)
        (assert (>= x 0))
        (assert (>= y 0))
        (assert (<= (+ x y) 10))
        (check-sat)
    "#
}

/// Small parser-only script with mixed declarations.
#[must_use]
pub fn parser_script() -> &'static str {
    r#"
        (set-logic QF_AUFBV)
        (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
        (declare-const i (_ BitVec 8))
        (declare-const v (_ BitVec 8))
        (assert (= (select (store a i v) i) v))
        (check-sat)
    "#
}
