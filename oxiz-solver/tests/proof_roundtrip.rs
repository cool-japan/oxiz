//! Round-trip tests: solve → write proof log → replay → verify.
//!
//! These tests exercise the wiring between `Context::check_sat` (with
//! `set_proof_log_path` enabled) and the offline `verify_proof_log` facility.

use std::env;
use std::path::PathBuf;

use oxiz_solver::{Context, SolverResult, VerificationResult};

/// Return a fresh temporary file path for a proof log.
fn tmp_log(name: &str) -> PathBuf {
    let mut p = env::temp_dir();
    p.push(format!("oxiz_proof_roundtrip_{}.oxizlog", name));
    p
}

#[test]
fn test_sat_proof_roundtrip() {
    let log_path = tmp_log("sat");

    // Clean up any leftover from a previous run.
    if log_path.exists() {
        std::fs::remove_file(&log_path).expect("failed to remove stale proof log");
    }

    let mut ctx = Context::new();
    ctx.set_proof_log_path(Some(log_path.clone()));

    let t = ctx.terms.mk_true();
    ctx.assert(t);

    let result = ctx.check_sat();
    assert_eq!(result, SolverResult::Sat, "expected SAT");

    // The log file must have been created by check_sat.
    assert!(log_path.exists(), "proof log file was not created");

    // Replay and verify.
    let vr = Context::verify_proof_log(&log_path).expect("proof log replay failed with I/O error");

    // A SAT log emits a single axiom; the replayer reports Valid because all
    // structural checks pass.
    assert!(
        matches!(
            vr,
            VerificationResult::Valid | VerificationResult::Incomplete
        ),
        "unexpected verification result: {:?}",
        vr
    );

    std::fs::remove_file(&log_path).ok();
}

#[test]
fn test_unsat_proof_roundtrip() {
    let log_path = tmp_log("unsat");

    if log_path.exists() {
        std::fs::remove_file(&log_path).expect("failed to remove stale proof log");
    }

    let mut ctx = Context::new();
    ctx.set_proof_log_path(Some(log_path.clone()));

    // Assert both true and false — immediately UNSAT.
    let t = ctx.terms.mk_true();
    let f = ctx.terms.mk_false();
    ctx.assert(t);
    ctx.assert(f);

    let result = ctx.check_sat();
    assert_eq!(result, SolverResult::Unsat, "expected UNSAT");

    assert!(log_path.exists(), "proof log file was not created");

    let vr = Context::verify_proof_log(&log_path).expect("proof log replay failed with I/O error");

    assert!(
        matches!(
            vr,
            VerificationResult::Valid | VerificationResult::Incomplete
        ),
        "unexpected verification result: {:?}",
        vr
    );

    std::fs::remove_file(&log_path).ok();
}

#[test]
fn test_verify_proof_log_missing_file() {
    let absent_path = tmp_log("this_file_does_not_exist_xyz");
    if absent_path.exists() {
        std::fs::remove_file(&absent_path).ok();
    }

    let result = Context::verify_proof_log(&absent_path);
    assert!(result.is_err(), "expected an I/O error for a missing file");
}

#[test]
fn test_proof_log_path_getter() {
    let mut ctx = Context::new();
    assert!(ctx.proof_log_path().is_none());

    let path = tmp_log("getter_test");
    ctx.set_proof_log_path(Some(path.clone()));
    assert_eq!(ctx.proof_log_path(), Some(path.as_path()));

    ctx.set_proof_log_path(None);
    assert!(ctx.proof_log_path().is_none());
}
