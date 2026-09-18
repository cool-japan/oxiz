//! Regression guards for `n`-ary `distinct` over array store chains
//! (`#P2b-38`).
//!
//! The shape is the one the extensionality work of `#P2b-37` introduced: every
//! unordered pair of an `n`-ary array `distinct` is an equality pair, and each
//! pair used to draw a witness index, a congruence family *and* an off-chain
//! Skolem index eagerly.  Each of those indices is a fresh bit-vector argument
//! term of the reads taken at it, so it joins the BV↔EUF partition exchange's
//! candidate set, whose cost grows with the partitions of that set — and four
//! generated scripts that this tree answered in milliseconds before the pairs
//! existed ran for minutes with no answer at all.
//!
//! Every family that mints a fresh index is now built one pair per refinement
//! round and only for a pair the candidate model has not already separated
//! (`array_axioms`, phases 3a-3c), and the refinement runs under a
//! *deterministic* budget rather than a wall clock (`check_core`).  These four
//! scripts are the recheck's own; they are kept because the benchmark corpus
//! contains no `n`-ary array `distinct` at all and so cannot catch a return of
//! the blow-up.
//!
//! # No clock decides a verdict here (decision (16))
//!
//! Every script used to carry `(set-option :timeout 120000)` and the batch
//! used to assert a 45-second wall-clock `BUDGET` beside the verdicts.  Both
//! are gone.  A `:timeout` is a wall clock, so a test that asserts `sat`
//! behind one is asserting "this machine, this load, this profile" — the same
//! machine-dependence decision (9) removed from the solver's own refinement
//! loop, re-entering through the test suite.  What decides these four scripts
//! now is the array refinement's *deterministic* budget, so the verdict
//! assertions below stand on their own and the timing claim lives in an
//! `#[ignore]`d cost pin at the bottom of the file.

use oxiz_solver::{Context, SolverResult};

/// `(distinct <store chain> <store chain> brr)` at index width 1, element
/// width 2 — sat, and 0.004 s on the 0.3.4 base.
const N69_5: &str = "\
(set-logic QF_AUFBV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 2)))
(declare-const brr (Array (_ BitVec 1) (_ BitVec 2)))
(declare-const j (_ BitVec 1))
(declare-const v (_ BitVec 2))
(assert (distinct (store (store arr #b1 #b01) #b1 (select brr j)) \
(store ((as const (Array (_ BitVec 1) (_ BitVec 2))) #b11) #b1 v) brr))
(assert (= (select arr #b0) (select brr #b0)))
(check-sat)
(get-model)
";

/// The same shape at index width 2 with every read of the two arrays pinned
/// equal — sat, and 0.018 s on the base.
const N38_5: &str = "\
(set-logic QF_AUFBV)
(declare-const arr (Array (_ BitVec 2) (_ BitVec 1)))
(declare-const brr (Array (_ BitVec 2) (_ BitVec 1)))
(declare-const crr (Array (_ BitVec 2) (_ BitVec 1)))
(declare-const j (_ BitVec 2))
(declare-const w (_ BitVec 1))
(assert (distinct (store arr #b00 (select arr j)) crr (store (store brr j #b0) j w)))
(assert (= (select arr #b00) (select brr #b00)))
(assert (= (select arr #b01) (select brr #b01)))
(assert (= (select arr #b10) (select brr #b10)))
(assert (= (select arr #b11) (select brr #b11)))
(check-sat)
(get-model)
";

/// A three-operand `distinct` whose operands share a base array — sat, 0.005 s
/// on the base.
const N69_7: &str = "\
(set-logic QF_AUFBV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 2)))
(declare-const brr (Array (_ BitVec 1) (_ BitVec 2)))
(declare-const j (_ BitVec 1))
(declare-const v (_ BitVec 2))
(assert (distinct (store (store arr #b1 #b01) #b0 (select brr j)) \
(store ((as const (Array (_ BitVec 1) (_ BitVec 2))) #b11) #b1 v) arr))
(assert (= (select arr #b0) (select brr #b0)))
(check-sat)
(get-model)
";

/// Four operands at index width 2 — sat, 0.023 s on the base, and the one of
/// the four that still answered (in 33 s) before the fix.
const N78_51: &str = "\
(set-logic QF_AUFBV)
(declare-const arr (Array (_ BitVec 2) (_ BitVec 1)))
(declare-const brr (Array (_ BitVec 2) (_ BitVec 1)))
(declare-const crr (Array (_ BitVec 2) (_ BitVec 1)))
(declare-const i (_ BitVec 2))
(declare-const j (_ BitVec 2))
(declare-const v (_ BitVec 1))
(assert (distinct (store arr i v) (store brr j v) crr \
((as const (Array (_ BitVec 2) (_ BitVec 1))) #b1)))
(assert (= (select arr #b00) (select brr #b00)))
(check-sat)
(get-model)
";

/// Run a script and return its verdict.
fn verdict(script: &str) -> SolverResult {
    let mut ctx = Context::new();
    let lines = ctx.execute_script(script).unwrap_or_default();
    lines
        .iter()
        .rev()
        .find_map(|line| match line.as_str() {
            "sat" => Some(SolverResult::Sat),
            "unsat" => Some(SolverResult::Unsat),
            "unknown" => Some(SolverResult::Unknown),
            _ => None,
        })
        .unwrap_or(SolverResult::Unknown)
}

/// All four scripts, verdict only and with no clock anywhere: each one is
/// satisfiable and the deterministic array-refinement budget decides it.
///
/// Before the fix of `#P2b-38`, three of these did not answer in 60 s and the
/// fourth took 33 s, so a returning blow-up shows up here as a test that does
/// not come back — which is what `.config/nextest.toml`'s kill ceiling is for
/// — rather than as a verdict that depends on how fast the machine is.
#[test]
fn n_ary_array_distinct_over_store_chains_is_decided() {
    for (name, script) in [
        ("n69_5", N69_5),
        ("n38_5", N38_5),
        ("n69_7", N69_7),
        ("n78_51", N78_51),
    ] {
        assert_eq!(
            verdict(script),
            SolverResult::Sat,
            "{name} is satisfiable, and the array refinement's deterministic \
             budget decides it without a clock"
        );
    }
}

/// The cost pin, carrying this file's timing claim and asserting no verdict
/// behind a clock (decision (16)).
///
/// `#[ignore]`d because it is a *measurement*: what it prints is the wall
/// clock of the four scripts above, which depends on the machine, the profile
/// and the load. The verdicts are guarded, clock-free, by
/// [`n_ary_array_distinct_over_store_chains_is_decided`]; this only records
/// how long they take, and fails only if one of them stops being `sat`.
#[test]
#[ignore = "cost pin: prints a wall-clock measurement, asserts no verdict behind a clock"]
fn the_n_ary_array_distinct_cost_table() {
    use std::time::Instant;

    for (name, script) in [
        ("n69_5", N69_5),
        ("n38_5", N38_5),
        ("n69_7", N69_7),
        ("n78_51", N78_51),
    ] {
        let start = Instant::now();
        let answer = verdict(script);
        let elapsed = start.elapsed();
        println!("{name}: {answer:?} in {elapsed:?}");
        assert_eq!(answer, SolverResult::Sat, "{name} is satisfiable");
    }
}
