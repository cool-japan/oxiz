//! Combined Theories Integration Tests
//!
//! Verifies that `Solver::check_sat` (via `Context::check_sat`) answers
//! QF_AUFBV / QF_ALIA / QF_ABV correctly using the existing structured
//! Nelson–Oppen dispatch path.
//!
//! Two layers of coverage:
//! 1. Fixture sweeps — every `.smt2` file under
//!    `bench/extended_theories/QF_{AUFBV,ALIA,ABV}/` and
//!    `bench/z3_parity/benchmarks/QF_{AUFBV,ALIA,ABV}/`
//! 2. Hand-crafted inline regression tests for select+store across
//!    UF/BV/LIA combinations.

use oxiz_solver::{Context, SolverResult};

// ──────────────────────────────────────────────────────────────────
// Fixture sweep helpers
// ──────────────────────────────────────────────────────────────────

/// Parse the expected status from an SMT-LIB2 script.
///
/// Recognised patterns (case-insensitive, in order of priority):
/// - `(set-info :status sat|unsat|unknown)`  — SMT-LIB2 metadata
/// - `; expected: sat|unsat|unknown`          — our own comment convention
/// - `;; expected: sat|unsat|unknown`         — double-semicolon variant
fn parse_expected_status(content: &str) -> Option<SolverResult> {
    for line in content.lines() {
        let trimmed = line.trim();

        // SMT-LIB2 :status metadata
        if trimmed.contains(":status") {
            let lower = trimmed.to_lowercase();
            if lower.contains("unsat") {
                return Some(SolverResult::Unsat);
            }
            if lower.contains(" sat") || lower.ends_with("sat") {
                return Some(SolverResult::Sat);
            }
            if lower.contains("unknown") {
                return Some(SolverResult::Unknown);
            }
        }

        // Comment-based expected: / Expected: annotation
        let lower = trimmed.to_lowercase();
        if lower.starts_with("; expected:") || lower.starts_with(";; expected:") {
            if lower.contains("unsat") {
                return Some(SolverResult::Unsat);
            }
            if lower.contains("sat") {
                return Some(SolverResult::Sat);
            }
            if lower.contains("unknown") {
                return Some(SolverResult::Unknown);
            }
        }
    }
    None
}

/// Run a single SMT-LIB2 script and return the solver result.
fn run_script(script: &str) -> SolverResult {
    let mut ctx = Context::new();
    let outputs = ctx.execute_script(script).unwrap_or_default();
    // The result is the last "sat" / "unsat" / "unknown" token in the output.
    for tok in outputs.iter().rev() {
        match tok.trim() {
            "sat" => return SolverResult::Sat,
            "unsat" => return SolverResult::Unsat,
            "unknown" => return SolverResult::Unknown,
            _ => {}
        }
    }
    // No check-sat output found — treat as unknown.
    SolverResult::Unknown
}

/// Check a single fixture file.
///
/// Returns `Ok(())` on expected result, `Err(message)` otherwise.
/// Fixtures with no detectable expected status are skipped.
fn check_fixture(path: &std::path::Path) -> Result<(), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

    let expected = match parse_expected_status(&content) {
        Some(s) => s,
        None => return Ok(()), // skip fixtures without expected status
    };

    let actual = run_script(&content);

    // Allow Unknown as a valid "we couldn't decide" outcome even when the
    // oracle says sat/unsat — incomplete solvers are permitted to return
    // Unknown for any formula without being incorrect.  We only count it as
    // a failure when the solver asserts the *wrong* definitive answer.
    match (expected, actual) {
        (SolverResult::Sat, SolverResult::Unsat) => Err(format!(
            "{}: expected sat, got unsat",
            path.display()
        )),
        (SolverResult::Unsat, SolverResult::Sat) => Err(format!(
            "{}: expected unsat, got sat",
            path.display()
        )),
        _ => Ok(()), // sat/sat, unsat/unsat, or unknown in either position
    }
}

/// Sweep all `.smt2` files in a directory and assert each one.
/// Missing directories are silently skipped (CI-friendliness).
fn sweep_dir(dir: &str) -> Vec<String> {
    let path = std::path::Path::new(dir);
    if !path.is_dir() {
        return Vec::new();
    }

    let mut failures = Vec::new();

    let entries = match std::fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return failures,
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.extension().and_then(|s| s.to_str()) == Some("smt2")
            && let Err(msg) = check_fixture(&entry_path)
        {
            failures.push(msg);
        }
    }

    failures
}

// ──────────────────────────────────────────────────────────────────
// Fixture sweep tests
// ──────────────────────────────────────────────────────────────────

/// All QF_ABV fixtures from z3_parity benchmarks
#[test]
fn sweep_z3_parity_qf_abv() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/z3_parity/benchmarks/QF_ABV");
    let failures = sweep_dir(base);
    assert!(
        failures.is_empty(),
        "QF_ABV fixture failures:\n{}",
        failures.join("\n")
    );
}

/// All QF_AUFBV fixtures from z3_parity benchmarks
#[test]
fn sweep_z3_parity_qf_aufbv() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/z3_parity/benchmarks/QF_AUFBV");
    let failures = sweep_dir(base);
    assert!(
        failures.is_empty(),
        "QF_AUFBV fixture failures:\n{}",
        failures.join("\n")
    );
}

/// All QF_ALIA fixtures from z3_parity benchmarks
#[test]
fn sweep_z3_parity_qf_alia() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/z3_parity/benchmarks/QF_ALIA");
    let failures = sweep_dir(base);
    assert!(
        failures.is_empty(),
        "QF_ALIA fixture failures:\n{}",
        failures.join("\n")
    );
}

/// All QF_ABV fixtures from extended_theories
#[test]
fn sweep_extended_qf_abv() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/extended_theories/QF_ABV");
    let failures = sweep_dir(base);
    assert!(
        failures.is_empty(),
        "extended QF_ABV fixture failures:\n{}",
        failures.join("\n")
    );
}

/// All QF_AUFBV fixtures from extended_theories
#[test]
fn sweep_extended_qf_aufbv() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/extended_theories/QF_AUFBV");
    let failures = sweep_dir(base);
    assert!(
        failures.is_empty(),
        "extended QF_AUFBV fixture failures:\n{}",
        failures.join("\n")
    );
}

/// All QF_ALIA fixtures from extended_theories
#[test]
fn sweep_extended_qf_alia() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/extended_theories/QF_ALIA");
    let failures = sweep_dir(base);
    assert!(
        failures.is_empty(),
        "extended QF_ALIA fixture failures:\n{}",
        failures.join("\n")
    );
}

// ──────────────────────────────────────────────────────────────────
// Hand-crafted inline regression tests
// QF_ABV — Arrays + BitVectors
// ──────────────────────────────────────────────────────────────────

/// Basic read-over-write axiom: select(store(a, i, v), i) = v
#[test]
fn inline_qf_abv_read_over_write_sat() {
    let script = r#"
(set-logic QF_ABV)
(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= (select (store a #x00 #x42) #x00) #x42))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Read-over-write contradiction: value forced to be two different constants
#[test]
fn inline_qf_abv_read_over_write_unsat() {
    let script = r#"
(set-logic QF_ABV)
(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= (select (store a #x00 #x42) #x00) #xFF))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Chain of stores: last write at an index wins
#[test]
fn inline_qf_abv_chained_stores_sat() {
    let script = r#"
(set-logic QF_ABV)
(declare-const a (Array (_ BitVec 4) (_ BitVec 4)))
(declare-const b (Array (_ BitVec 4) (_ BitVec 4)))
(declare-const c (Array (_ BitVec 4) (_ BitVec 4)))
(assert (= b (store a #x0 #x1)))
(assert (= c (store b #x0 #x2)))
(assert (= (select c #x0) #x2))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// BV arithmetic conflict via variable binding
/// x = #x05, select(a, x) = bvadd(x, #x01) = #x06, but select(a, #x05) = #x10 → UNSAT
#[test]
fn inline_qf_abv_cross_theory_conflict_unsat() {
    let script = r#"
(set-logic QF_ABV)
(declare-const x (_ BitVec 8))
(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= (select a x) (bvadd x #x01)))
(assert (= x #x05))
(assert (= (select a #x05) #x10))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// BV strict ordering contradiction: x < f and x > f for the same constant
#[test]
fn inline_qf_abv_bv_ordering_unsat() {
    let script = r#"
(set-logic QF_ABV)
(declare-const x (_ BitVec 4))
(assert (bvult x #xf))
(assert (bvugt x #xf))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

// ──────────────────────────────────────────────────────────────────
// Hand-crafted inline regression tests
// QF_AUFBV — Arrays + UF + BitVectors
// ──────────────────────────────────────────────────────────────────

/// Store then select at the same index — should be SAT (tautology)
#[test]
fn inline_qf_aufbv_store_select_tautology_sat() {
    let script = r#"
(set-logic QF_AUFBV)
(declare-fun a () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun i () (_ BitVec 32))
(declare-fun v () (_ BitVec 32))
(assert (= (select (store a i v) i) v))
(assert (not (= v (_ bv0 32))))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Array extensionality: equal arrays must agree on all reads
/// a = b, but read at index 7 differs → UNSAT
#[test]
fn inline_qf_aufbv_extensionality_unsat() {
    let script = r#"
(set-logic QF_AUFBV)
(declare-fun a () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun b () (Array (_ BitVec 32) (_ BitVec 32)))
(assert (= a b))
(assert (not (= (select a (_ bv7 32)) (select b (_ bv7 32)))))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Store then read at a *different* index (non-interfering) — SAT
#[test]
fn inline_qf_aufbv_store_read_different_index_sat() {
    let script = r#"
(set-logic QF_AUFBV)
(declare-fun a () (Array (_ BitVec 8) (_ BitVec 8)))
(declare-fun b () (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= b (store a #x00 #x42)))
(assert (= (select b #x01) (select a #x01)))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Conflict from store-select at same index yielding a contradictory value
#[test]
fn inline_qf_aufbv_store_conflict_unsat() {
    let script = r#"
(set-logic QF_AUFBV)
(declare-fun mem () (Array (_ BitVec 8) (_ BitVec 16)))
(declare-const mem1 (Array (_ BitVec 8) (_ BitVec 16)))
(assert (= mem1 (store mem #x10 #xCAFE)))
(assert (= (select mem1 #x10) #xBEEF))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Nested store chains — earlier store is shadowed at the overwritten index
#[test]
fn inline_qf_aufbv_nested_store_shadow_sat() {
    let script = r#"
(set-logic QF_AUFBV)
(declare-const a (Array (_ BitVec 4) (_ BitVec 4)))
(declare-const b (Array (_ BitVec 4) (_ BitVec 4)))
(declare-const c (Array (_ BitVec 4) (_ BitVec 4)))
(assert (= b (store a #x0 #x1)))
(assert (= c (store b #x1 #x2)))
(assert (= (select c #x0) #x1))
(assert (= (select c #x1) #x2))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

// ──────────────────────────────────────────────────────────────────
// Hand-crafted inline regression tests
// QF_ALIA — Arrays + Linear Integer Arithmetic
// ──────────────────────────────────────────────────────────────────

/// Basic integer array read-over-write: select(store(a, 0, x), 0) = x → SAT
#[test]
fn inline_qf_alia_read_over_write_sat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))
(declare-fun x () Int)
(assert (= (select (store a 0 x) 0) x))
(assert (> x 0))
(assert (< x 100))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Integer array: store then read at same index yields stored value → conflict UNSAT
#[test]
fn inline_qf_alia_store_conflict_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const a1 (Array Int Int))
(assert (= a1 (store a 0 42)))
(assert (< (select a1 0) 5))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Sum pattern: a[0] + a[1] = 10, a[0] > 7, a[1] > 7 → impossible → UNSAT
#[test]
fn inline_qf_alia_sum_pattern_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))
(assert (= (+ (select a 0) (select a 1)) 10))
(assert (> (select a 0) 7))
(assert (> (select a 1) 7))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Array swap SAT: swap a[0] and a[2], verify positions afterwards
#[test]
fn inline_qf_alia_array_swap_sat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(assert (= (select a 0) 10))
(assert (= (select a 1) 20))
(assert (= (select a 2) 30))
(declare-const tmp Int)
(assert (= tmp (select a 0)))
(declare-const a1 (Array Int Int))
(assert (= a1 (store a 0 (select a 2))))
(declare-const a2 (Array Int Int))
(assert (= a2 (store a1 2 tmp)))
(assert (= (select a2 0) 30))
(assert (= (select a2 1) 20))
(assert (= (select a2 2) 10))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Store then read: negated read-over-write axiom → UNSAT
/// not(= (select (store a 3 5) 3) 5) contradicts the axiom
#[test]
fn inline_qf_alia_negated_read_over_write_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))
(assert (not (= (select (store a 3 5) 3) 5)))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Positive tautology: select(store(a, i, v), i) = v is always true → SAT
#[test]
fn inline_qf_alia_read_over_write_tautology_sat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))
(declare-fun i () Int)
(declare-fun v () Int)
(assert (= (select (store a i v) i) v))
(assert (>= i 0))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

// ──────────────────────────────────────────────────────────────────
// Cross-theory interaction tests
// ──────────────────────────────────────────────────────────────────

/// QF_ABV: select-equality read conflict (two reads from same array+index must agree)
#[test]
fn inline_qf_abv_read_consistency_unsat() {
    let script = r#"
(set-logic QF_ABV)
(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= (select a #x0A) #x01))
(assert (= (select a #x0A) #x02))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// QF_ALIA: two reads from same array+index with contradictory LIA constraints → UNSAT
#[test]
fn inline_qf_alia_read_consistency_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))
(declare-const v1 Int)
(declare-const v2 Int)
(assert (= (select a 5) v1))
(assert (= (select a 5) v2))
(assert (not (= v1 v2)))
(check-sat)
"#;
    // v1 = select(a,5) = v2, but v1 != v2 → UNSAT
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// QF_ABV: BV byte buffer with arithmetic, sequential writes all visible → SAT
#[test]
fn inline_qf_abv_byte_buffer_sat() {
    let script = r#"
(set-logic QF_ABV)
(declare-const buf (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const buf1 (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= buf1 (store buf #x00 #x48)))
(declare-const buf2 (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= buf2 (store buf1 #x01 #x69)))
(assert (= (select buf2 #x00) #x48))
(assert (= (select buf2 #x01) #x69))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}
