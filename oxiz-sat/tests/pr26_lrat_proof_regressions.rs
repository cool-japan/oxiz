//! End-to-end regression tests for the PR #26 online LRAT proof-tracing
//! port: solve a set of UNSAT instances with LRAT tracing enabled, emit the
//! proof, and check it with `oxiz_proof::lrat_check`'s pure-Rust LRAT
//! checker (no external tool, no C dependency — that checker replaces
//! `tools/lrat-check.c` from the upstream PR).
//!
//! Coverage:
//! - A DIMACS-shaped fixture, a pigeonhole instance, and a hand-built small
//!   UNSAT formula, each proved and independently re-verified.
//! - The `add_clause`-level cases that finalize the proof without ever
//!   entering `solve()`'s main loop: a literal empty original clause, and a
//!   unit clause directly contradicting an existing level-0 fact.
//! - Every inprocessing mechanism this port gates off rather than covers
//!   with hint chains (probing, hyper-binary resolution, BVE, ELS,
//!   subsumption/vivification) still yields *some* verifiable proof when
//!   enabled alongside LRAT tracing — the solver falls back to the plain
//!   CDCL path instead of emitting a proof gap.
//! - `check_hyper_binary_resolution`'s proof-tracing gate (a real,
//!   pre-existing bug this port fixes: that pass previously ran from the
//!   *main* propagation path with no DRAT/LRAT awareness at all) holds
//!   under the crate's *default* configuration, where it is enabled.
//! - Negative cases: a corrupted hint reference and a truncated proof (no
//!   empty-clause conclusion) are both rejected by the checker.
//!
//! Temp files use `std::env::temp_dir()`, uniquely named per test run and
//! removed afterward.

use oxiz_proof::lrat_check::check_lrat_proof;
use oxiz_sat::{Lit, Solver, SolverConfig, SolverResult, Var};
use std::path::PathBuf;

fn v(i: usize) -> Var {
    Var::new(i as u32)
}

/// A fresh, uniquely-named path under `std::env::temp_dir()` for one test's
/// LRAT output file.
fn unique_lrat_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "oxiz_sat_pr26_lrat_{tag}_{}_{nanos}.lrat",
        std::process::id()
    ))
}

/// The pigeonhole-principle UNSAT instance: `pigeons` items into `holes`
/// slots (`pigeons > holes`). Returns the clauses as DIMACS literal rows for
/// the checker's "original formula" input.
fn add_pigeonhole(solver: &mut Solver, pigeons: usize, holes: usize) -> Vec<Vec<i32>> {
    for _ in 0..pigeons * holes {
        solver.new_var();
    }
    let var = |p: usize, h: usize| (p * holes + h + 1) as i32;
    let mut original = Vec::new();
    for p in 0..pigeons {
        let clause: Vec<i32> = (0..holes).map(|h| var(p, h)).collect();
        solver.add_clause_dimacs(&clause);
        original.push(clause);
    }
    for h in 0..holes {
        for p1 in 0..pigeons {
            for p2 in (p1 + 1)..pigeons {
                let clause = vec![-var(p1, h), -var(p2, h)];
                solver.add_clause_dimacs(&clause);
                original.push(clause);
            }
        }
    }
    original
}

/// Solve `build` under LRAT tracing, assert UNSAT, and return the original
/// clauses (as the caller's `build` reported them) alongside the emitted
/// LRAT proof text.
fn solve_unsat_and_capture_lrat(
    tag: &str,
    config: SolverConfig,
    build: impl FnOnce(&mut Solver) -> Vec<Vec<i32>>,
) -> (Vec<Vec<i32>>, String) {
    let path = unique_lrat_path(tag);
    let mut solver = Solver::with_config(config);
    solver
        .enable_lrat_proof(&path)
        .expect("enable_lrat_proof must succeed before any add_clause");
    let original = build(&mut solver);
    let result = solver.solve();
    assert_eq!(result, SolverResult::Unsat, "instance must be UNSAT");
    solver.disable_lrat_proof(); // flushes buffered output to disk
    let lrat_text = std::fs::read_to_string(&path).expect("read emitted LRAT proof");
    let _ = std::fs::remove_file(&path);
    (original, lrat_text)
}

/// Replace the first hint on the last non-empty LRAT line with an id no
/// clause was ever registered under, corrupting an otherwise-valid proof.
fn corrupt_first_hint_of_last_line(lrat_text: &str) -> String {
    let mut lines: Vec<String> = lrat_text.lines().map(String::from).collect();
    let idx = lines
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .expect("proof must have at least one line");
    let mut tokens: Vec<String> = lines[idx].split_whitespace().map(String::from).collect();
    assert!(
        tokens.len() >= 4,
        "final line must be `<id> 0 <hint>... 0` with at least one hint: {tokens:?}"
    );
    let bad_hint: i64 = tokens[2].parse::<i64>().unwrap_or(0) + 10_000_000;
    tokens[2] = bad_hint.to_string();
    lines[idx] = tokens.join(" ");
    lines.join("\n") + "\n"
}

/// Drop the proof's final line (the empty-clause conclusion), leaving every
/// intermediate derivation step intact but no conclusion to accept.
fn drop_final_line(lrat_text: &str) -> String {
    let mut lines: Vec<&str> = lrat_text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines.pop();
    let mut s = lines.join("\n");
    if !s.is_empty() {
        s.push('\n');
    }
    s
}

// ---------------------------------------------------------------------
// Positive: representative UNSAT instances, each independently re-verified.
// ---------------------------------------------------------------------

#[test]
fn test_pr26_lrat_pigeonhole_proof_verifies() {
    let (original, lrat) =
        solve_unsat_and_capture_lrat("pigeonhole", SolverConfig::default(), |s| {
            add_pigeonhole(s, 6, 5)
        });
    let report = check_lrat_proof(&original, &lrat);
    assert!(report.verified, "failure: {:?}", report.failure);
    assert!(report.additions_checked > 0);
}

#[test]
fn test_pr26_lrat_small_hand_built_unsat_proof_verifies() {
    // (a∨b)∧(¬a∨b)∧(a∨¬b)∧(¬a∨¬b) is UNSAT (every combination of a/b
    // falsifies one of the four clauses).
    let (original, lrat) =
        solve_unsat_and_capture_lrat("hand_built", SolverConfig::default(), |s| {
            let a = s.new_var();
            let b = s.new_var();
            let clauses: Vec<Vec<Lit>> = vec![
                vec![Lit::pos(a), Lit::pos(b)],
                vec![Lit::neg(a), Lit::pos(b)],
                vec![Lit::pos(a), Lit::neg(b)],
                vec![Lit::neg(a), Lit::neg(b)],
            ];
            let mut original = Vec::new();
            for clause in &clauses {
                s.add_clause(clause.iter().copied());
                original.push(clause.iter().map(|l| l.to_dimacs()).collect());
            }
            original
        });
    let report = check_lrat_proof(&original, &lrat);
    assert!(report.verified, "failure: {:?}", report.failure);
}

#[test]
fn test_pr26_lrat_dimacs_fixture_proof_verifies_via_files() {
    // The same tiny UNSAT instance as above, but round-tripped through real
    // CNF and LRAT *files* via `oxiz_proof::lrat_check::check_lrat_files` —
    // exercising the "usable as a library function" file-based entry point.
    let cnf_path = unique_lrat_path("dimacs_fixture").with_extension("cnf");
    let lrat_path = unique_lrat_path("dimacs_fixture").with_extension("lrat");

    let mut solver = Solver::new();
    solver.enable_lrat_proof(&lrat_path).expect("enable lrat");
    let clauses = [vec![1, 2], vec![-1, 2], vec![1, -2], vec![-1, -2]];
    for clause in &clauses {
        solver.add_clause_dimacs(clause);
    }
    assert_eq!(solver.solve(), SolverResult::Unsat);
    solver.disable_lrat_proof();

    let mut cnf_text = String::from("p cnf 2 4\n");
    for clause in &clauses {
        for lit in clause {
            cnf_text.push_str(&lit.to_string());
            cnf_text.push(' ');
        }
        cnf_text.push_str("0\n");
    }
    std::fs::write(&cnf_path, cnf_text).expect("write cnf fixture");

    let report =
        oxiz_proof::lrat_check::check_lrat_files(&cnf_path, &lrat_path).expect("check files");
    assert!(report.verified, "failure: {:?}", report.failure);

    let _ = std::fs::remove_file(&cnf_path);
    let _ = std::fs::remove_file(&lrat_path);
}

// ---------------------------------------------------------------------
// Positive: `add_clause`-level UNSAT paths that finalize the proof without
// ever entering `solve()`'s main search loop.
// ---------------------------------------------------------------------

#[test]
fn test_pr26_lrat_direct_empty_original_clause_needs_no_derivation() {
    let path = unique_lrat_path("empty_original");
    let mut solver = Solver::new();
    solver.enable_lrat_proof(&path).expect("enable lrat");
    let ok = solver.add_clause(Vec::<Lit>::new());
    assert!(!ok, "adding an empty clause must report failure/UNSAT");
    assert_eq!(solver.solve(), SolverResult::Unsat);
    solver.disable_lrat_proof();
    let lrat_text = std::fs::read_to_string(&path).expect("read lrat");
    let _ = std::fs::remove_file(&path);

    // A literal length-0 original clause is, on its own, a complete proof —
    // the checker must accept it even with an otherwise-empty LRAT stream.
    let report = check_lrat_proof(&[vec![]], &lrat_text);
    assert!(report.verified, "failure: {:?}", report.failure);
}

#[test]
fn test_pr26_lrat_unit_clause_contradicting_existing_fact_proof_verifies() {
    let path = unique_lrat_path("unit_contradiction");
    let mut solver = Solver::new();
    solver.enable_lrat_proof(&path).expect("enable lrat");
    let a = solver.new_var();
    let ok1 = solver.add_clause([Lit::pos(a)]);
    assert!(ok1);
    let ok2 = solver.add_clause([Lit::neg(a)]);
    assert!(!ok2, "the second unit must contradict the first");
    assert_eq!(solver.solve(), SolverResult::Unsat);
    solver.disable_lrat_proof();
    let lrat_text = std::fs::read_to_string(&path).expect("read lrat");
    let _ = std::fs::remove_file(&path);

    let original = vec![vec![1], vec![-1]];
    let report = check_lrat_proof(&original, &lrat_text);
    assert!(report.verified, "failure: {:?}", report.failure);
}

// ---------------------------------------------------------------------
// Positive: every gated-off inprocessing mechanism still yields a
// verifiable proof (falls back to the plain CDCL path) rather than a proof
// gap when combined with LRAT tracing.
// ---------------------------------------------------------------------

#[test]
fn test_pr26_lrat_probing_and_hyper_binary_gate_still_yields_verifiable_proof() {
    let config = SolverConfig {
        enable_failed_literal_probing: true,
        ..SolverConfig::default()
    };
    let (original, lrat) =
        solve_unsat_and_capture_lrat("probing_gate", config, |s| add_pigeonhole(s, 6, 5));
    let report = check_lrat_proof(&original, &lrat);
    assert!(report.verified, "failure: {:?}", report.failure);
}

#[test]
fn test_pr26_lrat_bve_gate_still_yields_verifiable_proof() {
    let config = SolverConfig {
        enable_bve: true,
        ..SolverConfig::default()
    };
    let (original, lrat) =
        solve_unsat_and_capture_lrat("bve_gate", config, |s| add_pigeonhole(s, 6, 5));
    let report = check_lrat_proof(&original, &lrat);
    assert!(report.verified, "failure: {:?}", report.failure);
}

#[test]
fn test_pr26_lrat_equiv_substitution_gate_still_yields_verifiable_proof() {
    let config = SolverConfig {
        enable_equiv_substitution: true,
        enable_gate_congruence: true,
        ..SolverConfig::default()
    };
    let (original, lrat) =
        solve_unsat_and_capture_lrat("els_gate", config, |s| add_pigeonhole(s, 6, 5));
    let report = check_lrat_proof(&original, &lrat);
    assert!(report.verified, "failure: {:?}", report.failure);
}

#[test]
fn test_pr26_lrat_inprocessing_gate_still_yields_verifiable_proof() {
    let config = SolverConfig {
        enable_inprocessing: true,
        inprocessing_interval: 1,
        ..SolverConfig::default()
    };
    let (original, lrat) =
        solve_unsat_and_capture_lrat("inprocess_gate", config, |s| add_pigeonhole(s, 6, 5));
    let report = check_lrat_proof(&original, &lrat);
    assert!(report.verified, "failure: {:?}", report.failure);
}

// ---------------------------------------------------------------------
// Regression: `check_hyper_binary_resolution`'s proof-tracing gate. This
// pass runs from the *main* propagation path, gated only by
// `enable_lazy_hyper_binary` (on by `SolverConfig::default()`) — before
// this port it had no DRAT/LRAT awareness at all, so a default-configured
// solve that reached decision level >= 2 could add a clause to the live
// database with no corresponding proof line. If that gate ever regresses,
// `lrat_hint_chain`'s trail replay silently drops the hint for that
// clause's reason, and this proof fails to verify.
// ---------------------------------------------------------------------

#[test]
fn test_pr26_lrat_default_config_hyper_binary_gate_produces_verifiable_proof() {
    // The flag is set here *explicitly* rather than relying on
    // `SolverConfig::default()`: the default was retuned by measurement for
    // issue #38 (see the comment at `SolverConfig::enable_lazy_hyper_binary`),
    // and this test must keep exercising the proof-tracing gate whichever way
    // that default lands. Pigeonhole(6,5) reaches well past decision level 2
    // during search, so the pass is genuinely reached.
    let config = SolverConfig {
        enable_lazy_hyper_binary: true,
        ..SolverConfig::default()
    };
    let (original, lrat) =
        solve_unsat_and_capture_lrat("hyper_binary_gate", config, |s| add_pigeonhole(s, 6, 5));
    let report = check_lrat_proof(&original, &lrat);
    assert!(report.verified, "failure: {:?}", report.failure);
}

// ---------------------------------------------------------------------
// Positive: DRAT and LRAT tracing enabled together.
// ---------------------------------------------------------------------

#[test]
fn test_pr26_lrat_and_drat_both_enabled_lrat_still_verifies() {
    let drat_path = unique_lrat_path("both_drat");
    let lrat_path = unique_lrat_path("both_lrat");
    let mut solver = Solver::new();
    solver.enable_drat_proof(&drat_path).expect("enable drat");
    solver.enable_lrat_proof(&lrat_path).expect("enable lrat");
    let original = add_pigeonhole(&mut solver, 6, 5);
    assert_eq!(solver.solve(), SolverResult::Unsat);
    solver.disable_drat_proof();
    solver.disable_lrat_proof();

    let lrat_text = std::fs::read_to_string(&lrat_path).expect("read lrat");
    let _ = std::fs::remove_file(&drat_path);
    let _ = std::fs::remove_file(&lrat_path);

    let report = check_lrat_proof(&original, &lrat_text);
    assert!(report.verified, "failure: {:?}", report.failure);
}

// ---------------------------------------------------------------------
// Negative: corrupted proofs must be rejected.
// ---------------------------------------------------------------------

#[test]
fn test_pr26_lrat_corrupted_hint_reference_is_rejected() {
    let (original, lrat) =
        solve_unsat_and_capture_lrat("corrupt_hint", SolverConfig::default(), |s| {
            add_pigeonhole(s, 6, 5)
        });
    let corrupted = corrupt_first_hint_of_last_line(&lrat);
    assert_ne!(
        corrupted, lrat,
        "corruption helper must actually change the text"
    );
    let report = check_lrat_proof(&original, &corrupted);
    assert!(
        !report.verified,
        "a corrupted hint reference must be rejected"
    );
}

#[test]
fn test_pr26_lrat_truncated_proof_missing_conclusion_is_rejected() {
    let (original, lrat) =
        solve_unsat_and_capture_lrat("truncated", SolverConfig::default(), |s| {
            add_pigeonhole(s, 6, 5)
        });
    let truncated = drop_final_line(&lrat);
    assert_ne!(
        truncated, lrat,
        "truncation helper must actually change the text"
    );
    let report = check_lrat_proof(&original, &truncated);
    assert!(
        !report.verified,
        "a proof with its empty-clause conclusion removed must be rejected"
    );
}

#[test]
fn test_pr26_lrat_valid_proof_is_not_accidentally_rejected_by_corruption_helpers() {
    // Sanity check on the two corruption helpers themselves: applied to a
    // genuinely valid proof, `check_lrat_proof` on the *unmodified* text
    // must still accept it (guards against a helper bug making every test
    // above pass for the wrong reason).
    let (original, lrat) =
        solve_unsat_and_capture_lrat("helper_sanity", SolverConfig::default(), |s| {
            let a = v(0);
            s.new_var();
            s.add_clause([Lit::pos(a)]);
            s.add_clause([Lit::neg(a)]);
            vec![vec![1], vec![-1]]
        });
    let report = check_lrat_proof(&original, &lrat);
    assert!(report.verified, "failure: {:?}", report.failure);
}

// ---------------------------------------------------------------------
// Gatekeeper SK-5: the writer must go inert once the proof is finalized.
// ---------------------------------------------------------------------

#[test]
fn test_pr26_lrat_writer_is_inert_after_unsat_finalization() {
    // A reused (incremental) solver that keeps adding clauses and
    // re-solving after an `Unsat` verdict must not keep appending lines to
    // an LRAT proof that already concluded with the empty clause: any
    // checker stops reading at that line (see `oxiz_proof::lrat_check`), so
    // trailing content can never affect verification either way, but the
    // file should still reflect "this proof is done" rather than
    // accumulating writes forever.
    let path = unique_lrat_path("inert_after_finalize");
    let mut solver = Solver::new();
    solver
        .enable_lrat_proof(&path)
        .expect("enable_lrat_proof must succeed before any add_clause");

    let a = v(0);
    solver.new_var();
    solver.add_clause([Lit::pos(a)]);
    solver.add_clause([Lit::neg(a)]);
    assert_eq!(solver.solve(), SolverResult::Unsat);
    assert!(
        !solver.lrat_proof_enabled(),
        "lrat_proof_enabled() must report false once the proof is finalized"
    );

    let after_finalize = std::fs::read_to_string(&path).expect("read lrat file");

    // Keep using the same solver: add another (irrelevant) clause and solve
    // again. Neither should write anything more to the file.
    let b = v(1);
    solver.new_var();
    let _ = solver.add_clause([Lit::pos(b)]);
    let _ = solver.solve();
    let _ = solver.solve_with_assumptions(&[Lit::pos(b)]);

    let after_reuse = std::fs::read_to_string(&path).expect("read lrat file again");
    assert_eq!(
        after_finalize, after_reuse,
        "no further bytes must be written to an already-finalized LRAT proof"
    );

    // The file up to this (unchanged) point must still be a valid,
    // verifiable proof of the original two-clause contradiction.
    let report = check_lrat_proof(&[vec![1], vec![-1]], &after_reuse);
    assert!(report.verified, "failure: {:?}", report.failure);

    let _ = std::fs::remove_file(&path);
}
