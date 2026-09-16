//! Timing regressions for `n`-ary `distinct` over array store chains
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
//! The off-chain family is now built in a phase of its own, for one pair per
//! refinement round and only when the three syntactic families add nothing
//! (`array_axioms::build_off_chain_family`), and the refinement's re-solves
//! run under a wall-clock budget (`check_core`).  These four scripts are the
//! recheck's own; they are kept as *timing* regressions because the benchmark
//! corpus contains no `n`-ary array `distinct` at all and so cannot catch a
//! return of the blow-up.
//!
//! Each script carries `(set-option :timeout 120000)`, so the regression this
//! test guards against shows up as a *verdict* — a blown-up search answers
//! `unknown` at the timeout instead of `sat` — rather than as a wall-clock
//! measurement that depends on the machine and on how loaded it is.  The
//! elapsed-time assertion below is the second line of defence, and is
//! deliberately loose for the same reason.

use oxiz_solver::{Context, SolverResult};
use std::time::{Duration, Instant};

/// Wall clock one batch of these scripts may take.  They answer in about 0.5 s
/// in release on the development machine (a few seconds in a debug build);
/// before the fix, three of them did not answer in 60 s and the fourth took
/// 33 s.  This bound is what keeps a returning blow-up from sitting on the
/// 120 s timeout each script carries, once per script in the batch.
const BUDGET: Duration = Duration::from_secs(45);

/// `(distinct <store chain> <store chain> brr)` at index width 1, element
/// width 2 — sat, and 0.004 s on the 0.3.4 base.
const N69_5: &str = "\
(set-logic QF_AUFBV)
(set-option :timeout 120000)
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
(set-option :timeout 120000)
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
(set-option :timeout 120000)
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
(set-option :timeout 120000)
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

/// The two scripts of the four that are cheap in a debug build, so the guard
/// runs on every `cargo nextest run`.  Both answered in milliseconds on the
/// 0.3.4 base and did not answer in 60 s with the blow-up.
#[test]
fn n_ary_array_distinct_over_store_chains_terminates_quickly() {
    check(&[("n69_7", N69_7), ("n78_51", N78_51)]);
}

/// The other two.  They are correct and fast in a release build (0.18 s and
/// 0.26 s) but cost about a minute each in the unoptimised build the test
/// suite uses by default, which is why they are opt-in — the same reason
/// `bv_wide_soundness`'s wide cases are.  Run with
/// `cargo nextest run -p oxiz-solver --run-ignored all`.
///
/// If this fails with `unknown` rather than with [`BUDGET`]: these two were
/// measured at 61.6 s and 64.8 s in a debug build on the development machine,
/// against the 120 s each script's `:timeout` allows, so a machine about twice
/// as slow answers `unknown` here without anything having regressed.  Raise
/// the `:timeout` in the two scripts before looking for a defect — the
/// blow-up this guards against did not answer in 60 s at *release* speed.
#[test]
#[ignore = "release-calibrated: about a minute each in a debug build"]
fn n_ary_array_distinct_release_calibrated_cases() {
    check(&[("n69_5", N69_5), ("n38_5", N38_5)]);
}

/// Run each script, require `sat`, and require the batch to finish inside
/// [`BUDGET`].
fn check(scripts: &[(&str, &str)]) {
    let start = Instant::now();
    let mut answers = Vec::new();
    for &(name, script) in scripts {
        let script_start = Instant::now();
        let answer = verdict(script);
        answers.push((name, answer, script_start.elapsed()));
    }
    let elapsed = start.elapsed();
    for &(name, answer, script_elapsed) in &answers {
        assert_eq!(
            answer,
            SolverResult::Sat,
            "{name} is satisfiable (answered in {script_elapsed:?}); \
             every one of these scripts has a model"
        );
    }
    assert!(
        elapsed < BUDGET,
        "these n-ary array `distinct` scripts took {elapsed:?}, over the {BUDGET:?} \
         budget: {answers:?}"
    );
}
